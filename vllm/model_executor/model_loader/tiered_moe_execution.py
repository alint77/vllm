# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared hot/cold Marlin setup and execution for tiered MoE methods."""

from typing import Any

import torch

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
        method.tiered_moe_kernels.append((kernel, components, tier_map))
        if len(method.tiered_moe_kernels) == 1:
            method.moe_quant_config = quant_config
            method.moe_kernel = kernel
    layer.tiered_moe_load_complete = True


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
