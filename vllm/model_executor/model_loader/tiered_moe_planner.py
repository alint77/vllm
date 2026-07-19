# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic hot/cold expert placement for tiered MoE."""

from collections.abc import Mapping
from dataclasses import dataclass

from vllm import envs
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.tiered_moe_manifest import (
    TieredMoECheckpointManifest,
)

MINIMUM_HBM_RESERVE_BYTES = 7_000_000_000
MINIMUM_HOST_RESERVE_BYTES = 8_000_000_000


@dataclass(frozen=True)
class LayerExpertPlacement:
    """Tier assignment for the experts owned in one routed layer."""

    layer_id: int
    hot_expert_ids: tuple[int, ...]
    cold_expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class RankTierPlan:
    """Exact routed-expert storage plan for one EP rank."""

    ep_rank: int
    ep_size: int
    expert_bytes: int
    owned_expert_ids: tuple[int, ...]
    layer_placements: tuple[LayerExpertPlacement, ...]
    fixed_hbm_allocations: tuple[tuple[str, int], ...]
    fixed_host_allocations: tuple[tuple[str, int], ...]
    fixed_hbm_bytes: int
    fixed_host_bytes: int
    hbm_capacity_bytes: int
    hbm_reserve_bytes: int
    hot_expert_bytes: int
    cold_expert_bytes: int
    host_capacity_bytes: int
    host_reserve_bytes: int
    transient_hbm_bytes: int = 0
    cold_allocation_kind: str = "pinned_uva"

    @property
    def hot_expert_slots(self) -> int:
        """Number of layer-expert instances assigned to HBM."""
        return sum(len(layer.hot_expert_ids) for layer in self.layer_placements)

    @property
    def cold_expert_slots(self) -> int:
        """Number of layer-expert instances assigned to Grace memory."""
        return sum(len(layer.cold_expert_ids) for layer in self.layer_placements)

    @property
    def owned_expert_ids_by_layer(self) -> dict[int, tuple[int, ...]]:
        """Return the strict loader ownership map for routed layers."""
        return {
            layer.layer_id: layer.hot_expert_ids + layer.cold_expert_ids
            for layer in self.layer_placements
        }

    @property
    def planned_hbm_bytes(self) -> int:
        """HBM bytes committed by fixed allocations, experts, and reserve."""
        return self.fixed_hbm_bytes + self.hot_expert_bytes + self.hbm_reserve_bytes

    @property
    def planned_host_bytes(self) -> int:
        """Grace bytes committed by fixed allocations, experts, and reserve."""
        return self.fixed_host_bytes + self.cold_expert_bytes + self.host_reserve_bytes

    @property
    def load_peak_hbm_bytes(self) -> int:
        """Peak HBM while one expert is converted into its final tier."""
        return self.fixed_hbm_bytes + self.hot_expert_bytes + self.transient_hbm_bytes

    def summary(self) -> dict:
        """Return a compact JSON-compatible plan summary."""
        return {
            "ep_rank": self.ep_rank,
            "ep_size": self.ep_size,
            "owned_expert_ids": list(self.owned_expert_ids),
            "routed_layer_count": len(self.layer_placements),
            "expert_bytes": self.expert_bytes,
            "hot_expert_slots": self.hot_expert_slots,
            "cold_expert_slots": self.cold_expert_slots,
            "fixed_hbm_allocations": dict(self.fixed_hbm_allocations),
            "fixed_host_allocations": dict(self.fixed_host_allocations),
            "fixed_hbm_bytes": self.fixed_hbm_bytes,
            "fixed_host_bytes": self.fixed_host_bytes,
            "hot_expert_bytes": self.hot_expert_bytes,
            "cold_expert_bytes": self.cold_expert_bytes,
            "hbm_reserve_bytes": self.hbm_reserve_bytes,
            "host_reserve_bytes": self.host_reserve_bytes,
            "planned_hbm_bytes": self.planned_hbm_bytes,
            "hbm_capacity_bytes": self.hbm_capacity_bytes,
            "planned_host_bytes": self.planned_host_bytes,
            "host_capacity_bytes": self.host_capacity_bytes,
            "transient_hbm_bytes": self.transient_hbm_bytes,
            "load_peak_hbm_bytes": self.load_peak_hbm_bytes,
            "cold_allocation_kind": self.cold_allocation_kind,
            "layers": [
                {
                    "layer_id": layer.layer_id,
                    "hot_expert_ids": list(layer.hot_expert_ids),
                    "cold_expert_ids": list(layer.cold_expert_ids),
                }
                for layer in self.layer_placements
            ],
        }


def _validate_capacity(name: str, capacity: int, reserve: int) -> None:
    if capacity <= 0:
        raise ValueError(f"{name} capacity must be positive")
    if reserve < 0:
        raise ValueError(f"{name} reserve must be non-negative")
    if reserve > capacity:
        raise ValueError(f"{name} reserve exceeds capacity")


def _even_layer_placement(
    routed_layers: tuple[int, ...],
    owned_expert_ids: tuple[int, ...],
    hot_slots: int,
) -> tuple[LayerExpertPlacement, ...]:
    layer_count = len(routed_layers)
    experts_per_layer = len(owned_expert_ids)
    base_hot, remainder = divmod(hot_slots, layer_count)
    placements = []
    for layer_offset, layer_id in enumerate(routed_layers):
        layer_hot_count = base_hot + (layer_offset < remainder)
        start = layer_offset % experts_per_layer
        rotated = owned_expert_ids[start:] + owned_expert_ids[:start]
        hot_ids = tuple(sorted(rotated[:layer_hot_count]))
        hot_id_set = set(hot_ids)
        cold_ids = tuple(
            expert_id for expert_id in owned_expert_ids if expert_id not in hot_id_set
        )
        placements.append(
            LayerExpertPlacement(
                layer_id=layer_id,
                hot_expert_ids=hot_ids,
                cold_expert_ids=cold_ids,
            )
        )
    return tuple(placements)


def _promote_underfilled_residency(
    hot_map: dict[int, tuple[int, ...]],
    ownership_map: dict[int, tuple[int, ...]],
    routed_layers: tuple[int, ...],
    extra_slots: int,
) -> dict[int, tuple[int, ...]]:
    """Deterministically promote cold experts when HBM outgrows a profile.

    A residency profile pins the trace-hot experts; when the physical budget
    grows (for example DCP shrinks the KV cache), the remaining slots are
    filled round-robin across layers in owned order rather than leaving HBM
    idle. Ordering within the promoted set carries no frequency information.
    """
    promoted = {layer_id: list(hot_map[layer_id]) for layer_id in routed_layers}
    remaining = extra_slots
    while remaining > 0:
        progressed = False
        for layer_id in routed_layers:
            if remaining == 0:
                break
            hot_set = set(promoted[layer_id])
            for expert_id in ownership_map[layer_id]:
                if expert_id not in hot_set:
                    promoted[layer_id].append(expert_id)
                    remaining -= 1
                    progressed = True
                    break
        if not progressed:
            raise ValueError("Residency promotion ran out of cold experts")
    return {layer_id: tuple(ids) for layer_id, ids in promoted.items()}


def _demote_overfilled_residency(
    hot_map: dict[int, tuple[int, ...]],
    routed_layers: tuple[int, ...],
    excess_slots: int,
) -> dict[int, tuple[int, ...]]:
    """Deterministically demote hot experts when HBM shrinks below a profile.

    The inverse of promotion (e.g. concurrent-sequence KV growth under DCP):
    trim one trailing hot expert per layer round-robin. Trailing order carries
    no frequency information, matching promotion's neutrality.
    """
    demoted = {layer_id: list(hot_map[layer_id]) for layer_id in routed_layers}
    remaining = excess_slots
    while remaining > 0:
        progressed = False
        for layer_id in routed_layers:
            if remaining == 0:
                break
            if demoted[layer_id]:
                demoted[layer_id].pop()
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("Residency demotion ran out of hot experts")
    return {layer_id: tuple(ids) for layer_id, ids in demoted.items()}


def build_layer_expert_ownership_map(
    manifest: TieredMoECheckpointManifest,
    ep_size: int,
    ep_rank: int,
    placement: str = "linear",
) -> dict[int, tuple[int, ...]]:
    """Build the strict per-layer checkpoint ownership map for one EP rank."""
    if not 0 <= ep_rank < ep_size:
        raise ValueError("EP rank must be within the EP world size")
    local_ids = compute_local_expert_ids(
        manifest.num_experts, ep_size, ep_rank, placement
    )
    if local_ids is None:
        local_ids = set(range(manifest.num_experts))
    owned_ids = tuple(sorted(local_ids))
    if not owned_ids:
        raise ValueError("EP rank does not own any routed experts")
    return {layer_id: owned_ids for layer_id in manifest.routed_layers}


def plan_rank_expert_tiers(
    manifest: TieredMoECheckpointManifest,
    ep_size: int,
    ep_rank: int,
    hbm_capacity_bytes: int,
    hbm_reserve_bytes: int,
    fixed_hbm_allocations: Mapping[str, int],
    host_capacity_bytes: int,
    host_reserve_bytes: int,
    placement: str = "linear",
    fixed_host_allocations: Mapping[str, int] | None = None,
    minimum_hbm_reserve_bytes: int = MINIMUM_HBM_RESERVE_BYTES,
    minimum_host_reserve_bytes: int = MINIMUM_HOST_RESERVE_BYTES,
    transient_hbm_bytes: int = 0,
    owned_expert_ids_by_layer: Mapping[int, tuple[int, ...]] | None = None,
    hot_expert_ids_by_layer: Mapping[int, tuple[int, ...]] | None = None,
) -> RankTierPlan:
    """Place as many owned expert instances in HBM as exact capacity permits.

    ``fixed_hbm_allocations`` must contain the separately derived physical
    allocations for non-routed weights, caches, workspaces, graphs, and runtime
    scratch. The function fails closed if either memory tier cannot satisfy the
    resulting plan.

    Args:
        manifest: Validated checkpoint header inventory.
        ep_size: Expert-parallel world size.
        ep_rank: Expert-parallel rank being planned.
        hbm_capacity_bytes: Physical HBM capacity for the rank.
        hbm_reserve_bytes: HBM bytes unavailable to planned allocations.
        fixed_hbm_allocations: Named non-routed physical HBM allocations.
        host_capacity_bytes: Physical paired-Grace capacity for the rank.
        host_reserve_bytes: Grace bytes unavailable to planned allocations.
        placement: Native EP ownership strategy.
        fixed_host_allocations: Named non-expert paired-Grace allocations.
        minimum_hbm_reserve_bytes: Smallest accepted HBM reserve.
        minimum_host_reserve_bytes: Smallest accepted paired-Grace reserve.
        transient_hbm_bytes: Bounded load-time HBM scratch released after conversion.

    Returns:
        An exact per-layer hot/cold expert assignment.

    Raises:
        ValueError: If ownership or either memory capacity is invalid.
    """
    _validate_capacity("HBM", hbm_capacity_bytes, hbm_reserve_bytes)
    _validate_capacity("Host", host_capacity_bytes, host_reserve_bytes)
    if minimum_hbm_reserve_bytes < 0 or minimum_host_reserve_bytes < 0:
        raise ValueError("Minimum reserves must be non-negative")
    if transient_hbm_bytes < 0:
        raise ValueError("Transient HBM scratch must be non-negative")
    if hbm_reserve_bytes < minimum_hbm_reserve_bytes:
        raise ValueError("HBM reserve is below the tiered-MoE minimum")
    if host_reserve_bytes < minimum_host_reserve_bytes:
        raise ValueError("Host reserve is below the tiered-MoE minimum")
    if not 0 <= ep_rank < ep_size:
        raise ValueError("EP rank must be within the EP world size")
    if fixed_host_allocations is None:
        fixed_host_allocations = {}
    if any(num_bytes < 0 for num_bytes in fixed_hbm_allocations.values()):
        raise ValueError("Fixed HBM allocation sizes must be non-negative")
    if any(num_bytes < 0 for num_bytes in fixed_host_allocations.values()):
        raise ValueError("Fixed host allocation sizes must be non-negative")

    fixed_hbm_bytes = sum(fixed_hbm_allocations.values())
    fixed_host_bytes = sum(fixed_host_allocations.values())
    available_hbm = hbm_capacity_bytes - hbm_reserve_bytes - fixed_hbm_bytes
    if available_hbm < 0:
        raise ValueError("Fixed allocations and reserve exceed HBM capacity")

    if owned_expert_ids_by_layer is None:
        ownership_map = build_layer_expert_ownership_map(
            manifest, ep_size, ep_rank, placement
        )
    else:
        ownership_map = dict(owned_expert_ids_by_layer)
        if set(ownership_map) != set(manifest.routed_layers):
            raise ValueError("Static ownership map does not cover routed layers")
        expected_count = manifest.num_experts // ep_size
        for expert_ids in ownership_map.values():
            if len(expert_ids) != expected_count or len(set(expert_ids)) != len(
                expert_ids
            ):
                raise ValueError("Static ownership map is not EP balanced")
            if any(
                not 0 <= expert_id < manifest.num_experts for expert_id in expert_ids
            ):
                raise ValueError("Static ownership map has an invalid expert ID")

    owned_expert_ids = next(iter(ownership_map.values()))
    total_slots = sum(len(expert_ids) for expert_ids in ownership_map.values())
    hot_slots = min(total_slots, available_hbm // manifest.runtime_expert_bytes)
    if hot_expert_ids_by_layer is not None and envs.VLLM_TIERED_MOE_PROFILE_CAP:
        profile_slots = sum(len(ids) for ids in hot_expert_ids_by_layer.values())
        hot_slots = min(hot_slots, profile_slots)
    cold_slots = total_slots - hot_slots
    hot_expert_bytes = hot_slots * manifest.runtime_expert_bytes
    cold_expert_bytes = cold_slots * manifest.runtime_expert_bytes
    if fixed_hbm_bytes + hot_expert_bytes + transient_hbm_bytes > hbm_capacity_bytes:
        raise ValueError("Load-time conversion scratch exceeds HBM capacity")
    if fixed_host_bytes + cold_expert_bytes + host_reserve_bytes > host_capacity_bytes:
        raise ValueError(
            "Fixed allocations, cold experts, and reserve exceed paired Grace capacity"
        )

    if hot_expert_ids_by_layer is None:
        if any(expert_ids != owned_expert_ids for expert_ids in ownership_map.values()):
            raise ValueError("Arbitrary ownership requires explicit hot expert IDs")
        layer_placements = _even_layer_placement(
            manifest.routed_layers, owned_expert_ids, hot_slots
        )
    else:
        hot_map = dict(hot_expert_ids_by_layer)
        if set(hot_map) != set(manifest.routed_layers):
            raise ValueError("Static residency map does not cover routed layers")
        map_slots = sum(len(expert_ids) for expert_ids in hot_map.values())
        if map_slots > hot_slots:
            hot_map = _demote_overfilled_residency(
                hot_map, manifest.routed_layers, map_slots - hot_slots
            )
        elif map_slots < hot_slots:
            hot_map = _promote_underfilled_residency(
                hot_map, ownership_map, manifest.routed_layers, hot_slots - map_slots
            )
        placements = []
        for layer_id in manifest.routed_layers:
            owned = ownership_map[layer_id]
            hot = hot_map[layer_id]
            if len(set(hot)) != len(hot) or not set(hot).issubset(owned):
                raise ValueError("Static hot experts must be unique and locally owned")
            hot_set = set(hot)
            cold = tuple(expert_id for expert_id in owned if expert_id not in hot_set)
            placements.append(LayerExpertPlacement(layer_id, tuple(sorted(hot)), cold))
        layer_placements = tuple(placements)
    return RankTierPlan(
        ep_rank=ep_rank,
        ep_size=ep_size,
        expert_bytes=manifest.runtime_expert_bytes,
        owned_expert_ids=owned_expert_ids,
        layer_placements=layer_placements,
        fixed_hbm_allocations=tuple(sorted(fixed_hbm_allocations.items())),
        fixed_host_allocations=tuple(sorted(fixed_host_allocations.items())),
        fixed_hbm_bytes=fixed_hbm_bytes,
        fixed_host_bytes=fixed_host_bytes,
        hbm_capacity_bytes=hbm_capacity_bytes,
        hbm_reserve_bytes=hbm_reserve_bytes,
        hot_expert_bytes=hot_expert_bytes,
        cold_expert_bytes=cold_expert_bytes,
        host_capacity_bytes=host_capacity_bytes,
        host_reserve_bytes=host_reserve_bytes,
        transient_hbm_bytes=transient_hbm_bytes,
    )
