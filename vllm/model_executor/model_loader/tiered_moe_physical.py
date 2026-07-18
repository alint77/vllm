# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared physical-plan construction for plan-only and worker loading."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import regex as re

from vllm.model_executor.model_loader.tiered_moe_kv import (
    TieredKVCachePlan,
    plan_glm_kv_cache_from_path,
)
from vllm.model_executor.model_loader.tiered_moe_machine import (
    GraceMachineProfile,
    load_grace_machine_profile,
)
from vllm.model_executor.model_loader.tiered_moe_manifest import (
    TieredMoECheckpointManifest,
    build_glm_w4a16_manifest,
)
from vllm.model_executor.model_loader.tiered_moe_non_routed import (
    NonRoutedRuntimeInventory,
    build_glm_non_routed_runtime_inventory,
)
from vllm.model_executor.model_loader.tiered_moe_placement import (
    TieredMoEPlacementProfile,
    load_tiered_moe_placement_profile,
)
from vllm.model_executor.model_loader.tiered_moe_planner import (
    MINIMUM_HBM_RESERVE_BYTES,
    LayerExpertPlacement,
    RankTierPlan,
    plan_rank_expert_tiers,
)
from vllm.model_executor.model_loader.tiered_moe_runtime import (
    TieredMoERuntimeBuffers,
    plan_tiered_glm_runtime_buffers,
)

_DERIVED_ALLOCATION_NAMES = {
    "non_routed_weights",
    "main_mla_cache",
    "indexer_cache",
    "tiered_moe_runtime_buffers",
}

_OBSERVED_HBM_RESERVE_TOLERANCE_BYTES = 1_000_000_000
MINIMUM_OBSERVED_HBM_RESERVE_BYTES = 4_000_000_000


@dataclass(frozen=True)
class TieredMoEScenarioPlan:
    """One selected cache-tier scenario across every EP rank."""

    kv_cache: TieredKVCachePlan
    rank_plans: tuple[RankTierPlan, ...]


@dataclass(frozen=True)
class TieredMoERankLoadPlan:
    """Validated physical load inputs and placement for one worker rank."""

    machine_profile: GraceMachineProfile
    manifest: TieredMoECheckpointManifest
    non_routed_runtime: NonRoutedRuntimeInventory
    runtime_buffers: TieredMoERuntimeBuffers
    kv_cache: TieredKVCachePlan
    rank_plan: RankTierPlan


_ACTIVE_RANK_LOAD_PLAN: ContextVar[TieredMoERankLoadPlan | None] = ContextVar(
    "tiered_moe_rank_load_plan", default=None
)


@contextmanager
def use_tiered_moe_rank_load_plan(plan: TieredMoERankLoadPlan):
    """Expose one worker plan during model construction and loading."""
    token = _ACTIVE_RANK_LOAD_PLAN.set(plan)
    try:
        yield
    finally:
        _ACTIVE_RANK_LOAD_PLAN.reset(token)


def get_tiered_moe_rank_load_plan() -> TieredMoERankLoadPlan | None:
    """Return the worker plan active in the current model-loading context."""
    return _ACTIVE_RANK_LOAD_PLAN.get()


def validate_tiered_moe_observed_hbm_reserve(
    vllm_config: Any, free_memory_bytes: int
) -> int:
    """Validate the physical HBM margin after cache allocation and warmup.

    Args:
        vllm_config: Active engine configuration.
        free_memory_bytes: Physical free HBM reported by the accelerator.

    Returns:
        The minimum accepted physical free-memory margin.

    Raises:
        RuntimeError: If the measured margin violates the tiered plan contract.
    """
    tiered = vllm_config.tiered_moe_config
    if not tiered.enabled:
        return 0
    if free_memory_bytes < 0:
        raise ValueError("Observed free HBM must be non-negative")

    planned_reserve = int(tiered.hbm_reserve_gb * 1_000_000_000)
    if planned_reserve < MINIMUM_HBM_RESERVE_BYTES:
        raise RuntimeError("Tiered MoE planned HBM reserve is below 7 GB")
    required_free = max(
        MINIMUM_OBSERVED_HBM_RESERVE_BYTES,
        planned_reserve - _OBSERVED_HBM_RESERVE_TOLERANCE_BYTES,
    )
    if free_memory_bytes < required_free:
        raise RuntimeError(
            "Tiered MoE observed free HBM is below the runtime reserve: "
            f"{free_memory_bytes} bytes available, {required_free} required. "
            "Replan more experts into Grace memory."
        )
    return required_free


def resolve_layer_expert_placement(
    plan: TieredMoERankLoadPlan, layer_name: str | None
) -> LayerExpertPlacement:
    """Resolve one routed layer's placement from its module prefix."""
    if layer_name is None:
        raise ValueError("Tiered MoE routed layers require a module name")
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        raise ValueError(f"Cannot parse routed layer ID from {layer_name}")
    layer_id = int(match.group(1))
    for placement in plan.rank_plan.layer_placements:
        if placement.layer_id == layer_id:
            return placement
    raise ValueError(f"Tiered MoE plan has no placement for layer {layer_id}")


def plan_tiered_moe_scenario(
    model: str,
    *,
    manifest: TieredMoECheckpointManifest,
    non_routed: NonRoutedRuntimeInventory,
    runtime_buffers: TieredMoERuntimeBuffers,
    ep_size: int,
    hbm_capacity_bytes: int,
    hbm_reserve_bytes: int,
    host_capacity_bytes: int,
    host_reserve_bytes: int,
    max_model_len: int,
    kv_block_size: int,
    kv_cache_dtype: str,
    cache_tier: str,
    base_hbm_allocations: Mapping[str, int],
    base_host_allocations: Mapping[str, int],
    expert_placement: str,
    num_mtp_layers: int = 0,
    placement_profile: TieredMoEPlacementProfile | None = None,
    dcp_world_size: int = 1,
) -> TieredMoEScenarioPlan:
    """Build one complete cache/expert physical scenario."""
    supplied_names = set(base_hbm_allocations) | set(base_host_allocations)
    overlap = _DERIVED_ALLOCATION_NAMES & supplied_names
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Planner derives reserved allocation names: {names}")
    if set(base_hbm_allocations) & set(base_host_allocations):
        raise ValueError("Fixed allocation names must be unique across tiers")
    if cache_tier not in ("hbm", "host_uva"):
        raise ValueError("Physical planning requires an explicit MLA cache tier")

    kv_plan = plan_glm_kv_cache_from_path(
        model,
        max_model_len=max_model_len,
        block_size=kv_block_size,
        kv_cache_dtype=kv_cache_dtype,
        main_cache_tier=cache_tier,
        num_mtp_layers=num_mtp_layers,
        dcp_world_size=dcp_world_size,
    )
    fixed_hbm_allocations = dict(base_hbm_allocations)
    fixed_host_allocations = dict(base_host_allocations)
    fixed_hbm_allocations["non_routed_weights"] = non_routed.runtime_bytes_per_rank
    fixed_hbm_allocations["indexer_cache"] = kv_plan.indexer_cache_bytes
    fixed_hbm_allocations["tiered_moe_runtime_buffers"] = (
        runtime_buffers.steady_hbm_bytes
    )
    if cache_tier == "hbm":
        fixed_hbm_allocations["main_mla_cache"] = kv_plan.main_cache_bytes
    else:
        fixed_host_allocations["main_mla_cache"] = kv_plan.main_cache_bytes

    if placement_profile is not None and placement_profile.ep_size != ep_size:
        raise ValueError("Placement profile EP size does not match the scenario")

    rank_plans = tuple(
        plan_rank_expert_tiers(
            manifest,
            ep_size=ep_size,
            ep_rank=rank,
            hbm_capacity_bytes=hbm_capacity_bytes,
            hbm_reserve_bytes=hbm_reserve_bytes,
            fixed_hbm_allocations=fixed_hbm_allocations,
            host_capacity_bytes=host_capacity_bytes,
            host_reserve_bytes=host_reserve_bytes,
            placement=expert_placement,
            fixed_host_allocations=fixed_host_allocations,
            transient_hbm_bytes=runtime_buffers.conversion_scratch_bytes,
            owned_expert_ids_by_layer=(
                placement_profile.ownership_for_rank(rank)
                if placement_profile is not None
                else None
            ),
            hot_expert_ids_by_layer=(
                placement_profile.hot_for_rank(rank)
                if placement_profile is not None
                else None
            ),
        )
        for rank in range(ep_size)
    )
    return TieredMoEScenarioPlan(kv_plan, rank_plans)


def build_tiered_moe_rank_load_plan(
    vllm_config: Any, ep_rank: int
) -> TieredMoERankLoadPlan:
    """Resolve the selected physical plan inside one initialized worker."""
    tiered = vllm_config.tiered_moe_config
    if not tiered.enabled or tiered.plan_only:
        raise ValueError("A non-plan-only tiered MoE configuration is required")
    if tiered.grace_machine_profile is None:
        raise ValueError("Tiered MoE loading requires a machine profile")
    if tiered.mla_cache_tier == "auto":
        raise ValueError("Tiered MoE loading requires an explicit MLA cache tier")
    profile = load_grace_machine_profile(tiered.grace_machine_profile)
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    if model is None:
        raise ValueError("Tiered MoE loading requires a model configuration")
    ep_size = parallel.tensor_parallel_size
    manifest = build_glm_w4a16_manifest(model.model)
    non_routed = build_glm_non_routed_runtime_inventory(manifest, ep_size)
    runtime_buffers = plan_tiered_glm_runtime_buffers(
        manifest,
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        ep_size=ep_size,
        sm_count=profile.gpu_sm_count,
    )
    placement_profile = (
        load_tiered_moe_placement_profile(tiered.placement_profile, manifest, ep_size)
        if tiered.placement_profile is not None
        else None
    )
    speculative = getattr(vllm_config, "speculative_config", None)
    num_mtp_layers = 0
    if speculative is not None and speculative.method == "mtp":
        num_mtp_layers = (
            speculative.draft_model_config.hf_config.num_nextn_predict_layers
        )
    scenario = plan_tiered_moe_scenario(
        model.model,
        manifest=manifest,
        non_routed=non_routed,
        runtime_buffers=runtime_buffers,
        ep_size=ep_size,
        hbm_capacity_bytes=profile.hbm_capacity_bytes,
        hbm_reserve_bytes=int(tiered.hbm_reserve_gb * 1_000_000_000),
        host_capacity_bytes=profile.host_capacity_bytes,
        host_reserve_bytes=int(tiered.host_reserve_gb * 1_000_000_000),
        max_model_len=model.max_model_len,
        kv_block_size=vllm_config.cache_config.block_size,
        kv_cache_dtype=vllm_config.cache_config.cache_dtype,
        cache_tier=tiered.mla_cache_tier,
        base_hbm_allocations=dict(profile.fixed_hbm_allocations),
        base_host_allocations=dict(profile.fixed_host_allocations),
        expert_placement=parallel.expert_placement_strategy,
        num_mtp_layers=num_mtp_layers,
        placement_profile=placement_profile,
        dcp_world_size=parallel.decode_context_parallel_size,
    )
    if not 0 <= ep_rank < len(scenario.rank_plans):
        raise ValueError("EP rank is outside the physical plan")
    return TieredMoERankLoadPlan(
        profile,
        manifest,
        non_routed,
        runtime_buffers,
        scenario.kv_cache,
        scenario.rank_plans[ep_rank],
    )
