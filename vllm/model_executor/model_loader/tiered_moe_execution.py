# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared hot/cold Marlin setup and execution for tiered MoE methods."""

from typing import Any

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    make_wna16_moe_kernel,
    make_wna16_moe_quant_config,
)
from vllm.model_executor.model_loader.tiered_moe_planner import (
    LayerExpertPlacement,
)
from vllm.model_executor.model_loader.tiered_moe_scheduler import (
    assign_replicated_experts,
    validate_replicated_routes,
)


def _active_tier_expert_ids(
    placement: LayerExpertPlacement,
    *,
    ep_rank: int,
    ep_size: int,
    primary_ranks: tuple[int, ...] | None,
    secondary_ranks: tuple[int, ...] | None,
    assignment: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]:
    """Resolve the experts executed by this rank in each physical tier."""
    if assignment == "off":
        return placement.hot_expert_ids, placement.cold_expert_ids, None
    if assignment not in ("secondary", "greedy"):
        raise ValueError(f"Unsupported tiered replica assignment: {assignment}")
    if primary_ranks is None or secondary_ranks is None:
        raise ValueError("Tiered replica assignment requires cross-rank maps")
    if len(primary_ranks) != len(secondary_ranks):
        raise ValueError("Tiered replica assignment maps have different lengths")
    if any(not 0 <= rank < ep_size for rank in primary_ranks):
        raise ValueError("Tiered replica assignment has an invalid primary rank")
    if any(not -1 <= rank < ep_size for rank in secondary_ranks):
        raise ValueError("Tiered replica assignment has an invalid secondary rank")
    if any(
        secondary == primary
        for primary, secondary in zip(primary_ranks, secondary_ranks)
        if secondary >= 0
    ):
        raise ValueError("Tiered replica assignment duplicates a rank")

    expected_primary = {
        expert_id for expert_id, rank in enumerate(primary_ranks) if rank == ep_rank
    }
    if set(placement.primary_expert_ids) != expected_primary:
        raise ValueError("Tiered primary assignment does not match local storage")
    expected_replicas = {
        expert_id for expert_id, rank in enumerate(secondary_ranks) if rank == ep_rank
    }
    if set(placement.replica_expert_ids) != expected_replicas:
        raise ValueError("Tiered secondary assignment does not match local storage")

    if assignment == "greedy":
        return placement.hot_expert_ids, placement.cold_expert_ids, None

    selected_ranks = tuple(
        secondary if secondary >= 0 else primary
        for primary, secondary in zip(primary_ranks, secondary_ranks)
    )
    hot = tuple(
        expert_id
        for expert_id in placement.hot_expert_ids
        if selected_ranks[expert_id] == ep_rank
    )
    cold = tuple(
        expert_id
        for expert_id in placement.cold_expert_ids + placement.replica_expert_ids
        if selected_ranks[expert_id] == ep_rank
    )
    return hot, cold, selected_ranks


def setup_tiered_moe_kernels(
    method: Any,
    layer: torch.nn.Module,
    group_size: int,
    num_bits: int,
) -> None:
    """Build the independent hot and cold kernels for one routed layer."""
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    speculative = vllm_config.speculative_config
    verify_tokens = (
        speculative.num_speculative_tokens + 1 if speculative is not None else 1
    )
    method.tiered_overlap_max_tokens = (
        verify_tokens * vllm_config.scheduler_config.max_num_seqs
    )
    method.tiered_moe_kernels = []
    if method.experts_cls is None:
        raise ValueError("Tiered GLM requires a modular Marlin expert backend")

    placement = layer.tiered_moe_placement
    storage = layer.tiered_moe_storage
    assignment = vllm_config.tiered_moe_config.replica_assignment
    method.tiered_replica_assignment = assignment
    ep_size = vllm_config.parallel_config.tensor_parallel_size
    method.tiered_moe_ep_size = ep_size
    active_hot_ids, active_cold_ids, selected_ranks = _active_tier_expert_ids(
        placement,
        ep_rank=layer.tiered_moe_ep_rank,
        ep_size=ep_size,
        primary_ranks=getattr(layer, "tiered_moe_primary_ranks", None),
        secondary_ranks=getattr(layer, "tiered_moe_secondary_ranks", None),
        assignment=assignment,
    )
    active_by_tier = {"hot": active_hot_ids, "cold": active_cold_ids}
    first_tier = storage.hot if storage.hot is not None else storage.cold
    if assignment == "greedy":
        if storage.hot is None or storage.cold is None:
            raise ValueError("Greedy replica assignment requires both expert tiers")
        primary_ranks = layer.tiered_moe_primary_ranks
        secondary_ranks = layer.tiered_moe_secondary_ranks
        primary_hot = getattr(layer, "tiered_moe_primary_hot", None)
        if primary_hot is None:
            raise ValueError("Greedy replica assignment requires global HBM residency")
        if len(primary_hot) != layer.global_num_experts:
            raise ValueError("Greedy replica HBM residency has an invalid length")
        device = first_tier.buffer.device
        layer.register_buffer(
            "tiered_replica_primary_rank_map",
            torch.tensor(primary_ranks, dtype=torch.int32, device=device),
        )
        layer.register_buffer(
            "tiered_replica_secondary_rank_map",
            torch.tensor(secondary_ranks, dtype=torch.int32, device=device),
        )
        layer.register_buffer(
            "tiered_replica_primary_hot",
            torch.tensor(primary_hot, dtype=torch.int32, device=device),
        )
        layer.register_buffer(
            "tiered_replica_selected_ranks",
            torch.full(
                (layer.global_num_experts,),
                -1,
                dtype=torch.int32,
                device=device,
            ),
        )
    if selected_ranks is not None:
        layer.register_buffer(
            "tiered_replica_selected_ranks",
            torch.tensor(
                selected_ranks,
                dtype=torch.int32,
                device=first_tier.buffer.device,
            ),
        )
    owned_expert_ids = placement.primary_expert_ids
    expert_map = torch.full(
        (layer.global_num_experts,),
        -1,
        dtype=torch.int32,
        device=first_tier.buffer.device,
    )
    expert_map[
        torch.tensor(owned_expert_ids, dtype=torch.long, device=expert_map.device)
    ] = torch.arange(len(owned_expert_ids), dtype=torch.int32, device=expert_map.device)
    layer._buffers["_expert_map"] = expert_map

    for tier_name in ("hot", "cold"):
        tier = getattr(storage, tier_name)
        if tier is None:
            continue
        components = tier.components
        quant_config = make_wna16_moe_quant_config(
            w1_scale=components["w13_weight_scale"],
            w2_scale=components["w2_weight_scale"],
            group_size=group_size,
            num_bits=num_bits,
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
        )
        primary_map = torch.full(
            (layer.global_num_experts,),
            -1,
            dtype=torch.int32,
            device=tier.buffer.device,
        )
        active_expert_ids = active_by_tier[tier_name]
        primary_map[
            torch.tensor(
                active_expert_ids,
                dtype=torch.long,
                device=tier.buffer.device,
            )
        ] = torch.tensor(
            [tier.expert_ids.index(expert_id) for expert_id in active_expert_ids],
            dtype=torch.int32,
            device=tier.buffer.device,
        )
        if assignment == "greedy":
            layer.register_buffer(
                f"tiered_{tier_name}_primary_expert_map",
                primary_map,
            )
            tier_map = torch.full_like(primary_map, -1)
        else:
            tier_map = primary_map
        layer.register_buffer(f"tiered_{tier_name}_expert_map", tier_map)
        if tier_name == "cold" and (
            placement.replica_expert_ids or assignment == "greedy"
        ):
            physical_map = torch.full_like(tier_map, -1)
            physical_map[
                torch.tensor(
                    tier.expert_ids,
                    dtype=torch.long,
                    device=tier.buffer.device,
                )
            ] = torch.arange(
                len(tier.expert_ids),
                dtype=torch.int32,
                device=tier.buffer.device,
            )
            layer.register_buffer("tiered_cold_physical_expert_map", physical_map)
        kernel = make_wna16_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=method.moe,
            experts_cls=method.experts_cls,
            routing_tables=None,
            is_k_full=method.is_k_full,
        )
        _apply_tier_launch_policy(
            kernel,
            tier_name,
            tier.buffer.device,
            method.tiered_overlap_max_tokens,
        )
        method.tiered_moe_kernels.append((kernel, components, tier_map))
        if len(method.tiered_moe_kernels) == 1:
            method.moe_quant_config = quant_config
            method.moe_kernel = kernel
    if assignment == "greedy" and not hasattr(layer, "tiered_cold_physical_expert_map"):
        raise ValueError("Greedy replica assignment requires cold replica storage")
    layer.tiered_moe_load_complete = True


# CTAs per SM to request for each tier. Marlin's default launch asks for the
# whole SM's shared memory, so the two tiers can never be co-resident and their
# union is serial. Splitting the shared memory instead - hot at 2 CTAs/SM, cold
# at 1 - lets both tiers spread over every SM and overlap. Measured on GH200 at
# the q4 and q16 decode shapes, this cuts the two-tier union by 41-48%.
_TIER_BLOCKS_PER_SM = {"hot": 2, "cold": 1}


def _apply_tier_launch_policy(
    kernel: Any, tier_name: str, device: torch.device, max_tokens: int
) -> None:
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        MarlinExpertsBase,
        MarlinLaunchPolicy,
    )

    if not envs.VLLM_TIERED_MOE_TIGHT_SMEM:
        return
    blocks_per_sm = _TIER_BLOCKS_PER_SM.get(tier_name)
    if blocks_per_sm is None:
        return
    # kernel.impl is the modular impl; the experts object holding the policy is
    # one level deeper. Fail closed rather than silently leaving the tiers
    # serialized.
    experts = getattr(getattr(kernel, "impl", None), "fused_experts", None)
    if not isinstance(experts, MarlinExpertsBase):
        raise TypeError(
            "Tiered MoE expected a Marlin expert backend to apply the "
            f"{tier_name} tier's launch policy, got {type(experts).__name__}"
        )
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    experts.launch_policy = MarlinLaunchPolicy(
        smem_mode=ops.MARLIN_SMEM_TIGHT,
        grid_blocks=sms * blocks_per_sm,
        max_tokens=max_tokens,
    )


def apply_tiered_moe(
    method: Any,
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts: Any,
    shared_experts_input: torch.Tensor | None,
) -> torch.Tensor:
    """Execute the hot and cold expert tiers through their shared runtime."""
    if getattr(method, "tiered_replica_assignment", "off") == "greedy":
        check_interval = envs.VLLM_TIERED_MOE_ROUTE_CHECK_INTERVAL
        if check_interval < 0:
            raise ValueError("VLLM_TIERED_MOE_ROUTE_CHECK_INTERVAL cannot be negative")
        if check_interval and getattr(layer, "tiered_replica_route_check", False):
            layer.tiered_replica_route_check_count += 1
            if layer.tiered_replica_route_check_count % check_interval == 0:
                validate_replicated_routes(topk_ids, method.tiered_moe_ep_size)
        assign_replicated_experts(
            topk_ids,
            layer.tiered_replica_primary_rank_map,
            layer.tiered_replica_secondary_rank_map,
            layer.tiered_replica_primary_hot,
            layer.tiered_hot_primary_expert_map,
            layer.tiered_cold_primary_expert_map,
            layer.tiered_cold_physical_expert_map,
            layer.tiered_hot_expert_map,
            layer.tiered_cold_expert_map,
            layer.tiered_replica_selected_ranks,
            ep_size=method.tiered_moe_ep_size,
            ep_rank=layer.tiered_moe_ep_rank,
            schedule=topk_ids.shape[0] <= method.tiered_overlap_max_tokens,
        )
    primary_kernel = method.tiered_moe_kernels[0][0]
    tiers = [
        (
            kernel,
            components["w13_weight_packed"],
            components["w2_weight_packed"],
            expert_map,
        )
        for kernel, components, expert_map in method.tiered_moe_kernels
    ]
    return primary_kernel.apply_tiered(
        x,
        tiers,
        topk_weights,
        topk_ids,
        activation=layer.activation,
        global_num_experts=layer.global_num_experts,
        prepare_expert_map=layer.expert_map,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        shared_experts=shared_experts,
        shared_experts_input=shared_experts_input,
        overlap_max_tokens=getattr(method, "tiered_overlap_max_tokens", 4),
    )
