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
    owned_expert_ids = placement.hot_expert_ids + placement.cold_expert_ids
    storage = layer.tiered_moe_storage
    first_tier = storage.hot if storage.hot is not None else storage.cold
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
        tier_map = torch.full(
            (layer.global_num_experts,),
            -1,
            dtype=torch.int32,
            device=tier.buffer.device,
        )
        tier_map[
            torch.tensor(tier.expert_ids, dtype=torch.long, device=tier.buffer.device)
        ] = torch.arange(
            len(tier.expert_ids), dtype=torch.int32, device=tier.buffer.device
        )
        layer.register_buffer(f"tiered_{tier_name}_expert_map", tier_map)
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
    _attach_replica_tensors(method, layer, first_tier.buffer.device)
    layer.tiered_moe_load_complete = True


def _attach_replica_tensors(
    method: Any, layer: torch.nn.Module, device: torch.device
) -> None:
    """Materialise the profile's per-layer replica tables on the device."""
    from vllm.config import get_current_vllm_config

    assignment = get_current_vllm_config().tiered_moe_config.replica_assignment
    method.tiered_replica_assignment = assignment
    if assignment == "off":
        return
    if not hasattr(layer, "tiered_moe_primary_ranks"):
        raise ValueError(
            "Tiered MoE replica assignment requires a placement profile "
            "carrying secondary ranks"
        )

    def table(values: Any) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    layer.tiered_replica_primary_rank_map = table(layer.tiered_moe_primary_ranks)
    layer.tiered_replica_secondary_rank_map = table(layer.tiered_moe_secondary_ranks)
    layer.tiered_replica_primary_hot = table(
        [int(flag) for flag in layer.tiered_moe_primary_hot]
    )


# CTAs per SM to request for each tier. Marlin's default launch asks for the
# whole SM's shared memory, so the two tiers can never be co-resident and their
# union is serial. Splitting the shared memory instead - hot at 2 CTAs/SM, cold
# at 1 - lets both tiers spread over every SM and overlap. Measured on GH200 at
# the q4 and q16 decode shapes, this cuts the two-tier union by 41-48%.
_TIER_BLOCKS_PER_SM = {"hot": 2, "cold": 1}

# Marlin's own heuristic picks this for every decode shape the tiered path
# runs, and the fused alignment must agree with what the GEMM will use.
_TIER_BLOCK_M = 16


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


def _tier_experts(kernel: Any) -> Any:
    """Return the Marlin experts object holding a tier's launch attributes."""
    return getattr(getattr(kernel, "impl", None), "fused_experts", None)


def prepare_replica_routing(
    method: Any,
    layer: torch.nn.Module,
    topk_ids: torch.Tensor,
) -> bool:
    """Assign replicas and pre-align both tiers for one decode step.

    Returns True when the fused metadata was produced and attached, so the
    Marlin tiers should consume it instead of aligning themselves.
    """
    from vllm.model_executor.model_loader.tiered_moe_scheduler import (
        allocate_fused_routing,
    )

    if getattr(method, "tiered_replica_assignment", "off") == "off":
        return False
    num_tokens = topk_ids.shape[0]
    if num_tokens > method.tiered_overlap_max_tokens:
        return False
    if len(method.tiered_moe_kernels) != 2:
        return False

    if envs.VLLM_TIERED_MOE_ROUTE_CHECK and not hasattr(
        layer, "tiered_replica_route_fingerprint"
    ):
        layer.tiered_replica_route_fingerprint = torch.zeros(
            2, dtype=torch.int64, device=topk_ids.device
        )

    routing = getattr(layer, "tiered_replica_routing", None)
    if routing is None:
        routing = allocate_fused_routing(
            topk_ids.numel(),
            layer.global_num_experts,
            _TIER_BLOCK_M,
            _TIER_BLOCK_M,
            topk_ids.device,
        )
        layer.tiered_replica_routing = routing

    if envs.VLLM_TIERED_MOE_ROUTE_CHECK:
        torch.ops.vllm.tiered_moe_route_fingerprint(
            topk_ids, layer.tiered_replica_route_fingerprint
        )

    torch.ops.vllm.tiered_moe_assign_align(
        topk_ids,
        layer.tiered_replica_primary_rank_map,
        layer.tiered_replica_secondary_rank_map,
        layer.tiered_replica_primary_hot,
        layer.tiered_hot_expert_map,
        layer.tiered_cold_expert_map,
        routing.scratch,
        routing.selected_rank,
        routing.hot_out_map,
        routing.cold_out_map,
        routing.hot_sorted,
        routing.hot_expert_ids,
        routing.hot_num_post,
        routing.cold_sorted,
        routing.cold_expert_ids,
        routing.cold_num_post,
        layer.tiered_moe_ep_rank,
        _TIER_BLOCK_M,
        _TIER_BLOCK_M,
        method.tiered_replica_assignment == "exact",
    )

    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        MarlinPrealignedRouting,
    )

    tiers = (
        ("hot", routing.hot_sorted, routing.hot_expert_ids, routing.hot_num_post),
        ("cold", routing.cold_sorted, routing.cold_expert_ids, routing.cold_num_post),
    )
    for (kernel, _, _), (_, sorted_ids, expert_ids, num_post) in zip(
        method.tiered_moe_kernels, tiers
    ):
        _tier_experts(kernel).prealigned_routing = MarlinPrealignedRouting(
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_post,
            block_size_m=_TIER_BLOCK_M,
            max_tokens=method.tiered_overlap_max_tokens,
        )
    return True


def clear_replica_routing(method: Any) -> None:
    """Detach pre-aligned metadata so later shapes align normally."""
    for kernel, _, _ in method.tiered_moe_kernels:
        experts = _tier_experts(kernel)
        if experts is not None:
            experts.prealigned_routing = None


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
    assigned = prepare_replica_routing(method, layer, topk_ids)
    if not assigned:
        clear_replica_routing(method)
    primary_kernel = method.tiered_moe_kernels[0][0]
    routing = getattr(layer, "tiered_replica_routing", None) if assigned else None
    dynamic_maps = (
        (routing.hot_out_map, routing.cold_out_map) if routing is not None else None
    )
    tiers = [
        (
            kernel,
            components["w13_weight_packed"],
            components["w2_weight_packed"],
            expert_map if dynamic_maps is None else dynamic_maps[index],
        )
        for index, (kernel, components, expert_map) in enumerate(
            method.tiered_moe_kernels
        )
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
