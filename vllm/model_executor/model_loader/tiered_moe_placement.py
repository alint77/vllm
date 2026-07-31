# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static trace-derived owner and residency profiles for tiered MoE."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.model_executor.model_loader.tiered_moe_manifest import (
    TieredMoECheckpointManifest,
)


@dataclass(frozen=True)
class TieredMoEPlacementProfile:
    """Validated per-layer owner and HBM-residency assignment."""

    path: Path
    sha256: str
    config_sha256: str
    index_sha256: str
    ep_size: int
    num_experts: int
    routed_layers: tuple[int, ...]
    owners: tuple[tuple[int, ...], ...]
    hot_experts: tuple[tuple[int, ...], ...]
    secondary_ranks: tuple[tuple[int, ...], ...]
    optimizer: str
    training_request_hashes: tuple[str, ...]
    heldout_request_hashes: tuple[str, ...]

    def ownership_for_rank(self, ep_rank: int) -> dict[int, tuple[int, ...]]:
        """Return global expert IDs owned by one rank in each layer."""
        if not 0 <= ep_rank < self.ep_size:
            raise ValueError("EP rank is outside the placement profile")
        return {
            layer_id: tuple(
                expert_id
                for expert_id, owner in enumerate(layer_owners)
                if owner == ep_rank
            )
            for layer_id, layer_owners in zip(self.routed_layers, self.owners)
        }

    def hot_for_rank(self, ep_rank: int) -> dict[int, tuple[int, ...]]:
        """Return HBM expert IDs owned by one rank in each layer."""
        ownership = self.ownership_for_rank(ep_rank)
        result = {}
        for layer_id, layer_hot in zip(self.routed_layers, self.hot_experts):
            owned = set(ownership[layer_id])
            result[layer_id] = tuple(
                expert_id for expert_id in layer_hot if expert_id in owned
            )
        return result

    def replicas_for_rank(self, ep_rank: int) -> dict[int, tuple[int, ...]]:
        """Return secondary Grace copies assigned to one rank."""
        if not 0 <= ep_rank < self.ep_size:
            raise ValueError("EP rank is outside the placement profile")
        return {
            layer_id: tuple(
                expert_id
                for expert_id, rank in enumerate(layer_secondary)
                if rank == ep_rank
            )
            for layer_id, layer_secondary in zip(
                self.routed_layers, self.secondary_ranks
            )
        }

    def summary(self) -> dict[str, Any]:
        """Return profile identity and aggregate slot counts."""
        hot_by_rank = [
            sum(len(ids) for ids in self.hot_for_rank(rank).values())
            for rank in range(self.ep_size)
        ]
        replicas_by_rank = [
            sum(len(ids) for ids in self.replicas_for_rank(rank).values())
            for rank in range(self.ep_size)
        ]
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "ep_size": self.ep_size,
            "num_experts": self.num_experts,
            "routed_layer_count": len(self.routed_layers),
            "hot_expert_slots_by_rank": hot_by_rank,
            "replica_expert_slots_by_rank": replicas_by_rank,
            "optimizer": self.optimizer,
            "training_request_count": len(self.training_request_hashes),
            "heldout_request_count": len(self.heldout_request_hashes),
        }


def _string_list(data: dict[str, Any], name: str) -> tuple[str, ...]:
    value = data.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"Placement profile {name} must be a string array")
    return tuple(value)


def load_tiered_moe_placement_profile(
    path: str | Path,
    manifest: TieredMoECheckpointManifest,
    ep_size: int,
) -> TieredMoEPlacementProfile:
    """Load a fingerprinted static profile and validate every assignment."""
    profile_path = Path(path).resolve()
    raw = profile_path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Placement profile must contain a JSON object")
    expected_fields = {
        "profile_version",
        "config_sha256",
        "index_sha256",
        "ep_size",
        "num_experts",
        "routed_layers",
        "owners",
        "hot_experts",
        "optimizer",
        "training_request_hashes",
        "heldout_request_hashes",
    }
    if data.get("profile_version") == 2:
        expected_fields.add("secondary_ranks")
    if set(data) != expected_fields:
        raise ValueError("Placement profile fields do not match its schema version")
    if data["profile_version"] not in (1, 2):
        raise ValueError("Only placement profile versions 1 and 2 are supported")
    if data["config_sha256"] != manifest.config_sha256:
        raise ValueError("Placement profile config fingerprint does not match")
    if data["index_sha256"] != manifest.index_sha256:
        raise ValueError("Placement profile index fingerprint does not match")
    if data["ep_size"] != ep_size:
        raise ValueError("Placement profile EP size does not match")
    if data["num_experts"] != manifest.num_experts:
        raise ValueError("Placement profile expert count does not match")
    if data["routed_layers"] != list(manifest.routed_layers):
        raise ValueError("Placement profile routed layers do not match")

    owners_value = data["owners"]
    hot_value = data["hot_experts"]
    secondary_value = data.get(
        "secondary_ranks",
        [[-1] * manifest.num_experts for _ in manifest.routed_layers],
    )
    layer_count = len(manifest.routed_layers)
    if not isinstance(owners_value, list) or len(owners_value) != layer_count:
        raise ValueError("Placement profile owners must have one row per layer")
    if not isinstance(hot_value, list) or len(hot_value) != layer_count:
        raise ValueError("Placement profile hot experts must have one row per layer")
    if not isinstance(secondary_value, list) or len(secondary_value) != layer_count:
        raise ValueError(
            "Placement profile secondary ranks must have one row per layer"
        )

    experts_per_rank, remainder = divmod(manifest.num_experts, ep_size)
    if remainder:
        raise ValueError("Placement profiles require evenly divisible experts")
    owners = []
    hot_experts = []
    secondary_ranks = []
    for layer_offset, (layer_owners, layer_hot, layer_secondary) in enumerate(
        zip(owners_value, hot_value, secondary_value)
    ):
        if (
            not isinstance(layer_owners, list)
            or len(layer_owners) != manifest.num_experts
        ):
            raise ValueError(
                f"Placement profile owners row {layer_offset} has invalid length"
            )
        if any(
            not isinstance(owner, int)
            or isinstance(owner, bool)
            or not 0 <= owner < ep_size
            for owner in layer_owners
        ):
            raise ValueError("Placement profile contains an invalid owner rank")
        if any(layer_owners.count(rank) != experts_per_rank for rank in range(ep_size)):
            raise ValueError("Placement profile owner rows must be EP balanced")
        if not isinstance(layer_hot, list) or any(
            not isinstance(expert_id, int)
            or isinstance(expert_id, bool)
            or not 0 <= expert_id < manifest.num_experts
            for expert_id in layer_hot
        ):
            raise ValueError("Placement profile contains an invalid hot expert")
        if len(set(layer_hot)) != len(layer_hot):
            raise ValueError("Placement profile hot experts must be unique per layer")
        if (
            not isinstance(layer_secondary, list)
            or len(layer_secondary) != manifest.num_experts
            or any(
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or not -1 <= rank < ep_size
                for rank in layer_secondary
            )
        ):
            raise ValueError("Placement profile contains an invalid secondary rank")
        if any(
            rank == layer_owners[expert_id]
            for expert_id, rank in enumerate(layer_secondary)
            if rank >= 0
        ):
            raise ValueError("Placement profile secondary rank matches its owner")
        owners.append(tuple(layer_owners))
        hot_experts.append(tuple(sorted(layer_hot)))
        secondary_ranks.append(tuple(layer_secondary))

    optimizer = data["optimizer"]
    if not isinstance(optimizer, str) or not optimizer:
        raise ValueError("Placement profile optimizer must be a non-empty string")
    training_hashes = _string_list(data, "training_request_hashes")
    heldout_hashes = _string_list(data, "heldout_request_hashes")
    if not training_hashes or not heldout_hashes:
        raise ValueError("Placement profile requires train and held-out requests")
    if set(training_hashes) & set(heldout_hashes):
        raise ValueError("Placement profile train and held-out requests overlap")

    return TieredMoEPlacementProfile(
        path=profile_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        config_sha256=data["config_sha256"],
        index_sha256=data["index_sha256"],
        ep_size=ep_size,
        num_experts=manifest.num_experts,
        routed_layers=manifest.routed_layers,
        owners=tuple(owners),
        hot_experts=tuple(hot_experts),
        secondary_ranks=tuple(secondary_ranks),
        optimizer=optimizer,
        training_request_hashes=training_hashes,
        heldout_request_hashes=heldout_hashes,
    )
