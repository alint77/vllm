# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Header-only tiered MoE planning entry point."""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from vllm.model_executor.model_loader.tiered_moe_machine import (
    load_grace_machine_profile,
)
from vllm.model_executor.model_loader.tiered_moe_manifest import (
    build_glm_w4a16_manifest,
)
from vllm.model_executor.model_loader.tiered_moe_non_routed import (
    build_glm_non_routed_runtime_inventory,
)
from vllm.model_executor.model_loader.tiered_moe_physical import (
    plan_tiered_moe_scenario,
)
from vllm.model_executor.model_loader.tiered_moe_placement import (
    load_tiered_moe_placement_profile,
)
from vllm.model_executor.model_loader.tiered_moe_planner import (
    MINIMUM_HBM_RESERVE_BYTES,
    MINIMUM_HOST_RESERVE_BYTES,
)
from vllm.model_executor.model_loader.tiered_moe_runtime import (
    plan_tiered_glm_runtime_buffers,
)


def _fixed_allocation(value: str) -> tuple[str, int]:
    try:
        name, raw_bytes = value.split("=", 1)
        num_bytes = int(raw_bytes)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected NAME=BYTES") from error
    if not name or num_bytes < 0:
        raise argparse.ArgumentTypeError("Expected non-negative NAME=BYTES")
    return name, num_bytes


def build_parser() -> argparse.ArgumentParser:
    """Build the plan-only command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plan GLM W4A16 expert tiers from safetensors headers"
    )
    parser.add_argument("model", help="Local immutable checkpoint directory")
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--hbm-capacity-bytes", type=int, required=True)
    parser.add_argument(
        "--hbm-reserve-bytes", type=int, default=MINIMUM_HBM_RESERVE_BYTES
    )
    parser.add_argument("--host-capacity-bytes", type=int, required=True)
    parser.add_argument(
        "--host-reserve-bytes", type=int, default=MINIMUM_HOST_RESERVE_BYTES
    )
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-sm-count", type=int, default=132)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--kv-cache-dtype", default="fp8_ds_mla")
    parser.add_argument(
        "--mla-cache-tier",
        choices=("auto", "hbm", "host_uva"),
        default="auto",
        help="Price both cache tiers with auto, or emit one selected tier",
    )
    parser.add_argument(
        "--fixed-hbm-allocation",
        action="append",
        type=_fixed_allocation,
        required=True,
        metavar="NAME=BYTES",
        help="Exact non-cache runtime HBM allocation; repeat per category",
    )
    parser.add_argument(
        "--fixed-host-allocation",
        action="append",
        type=_fixed_allocation,
        default=[],
        metavar="NAME=BYTES",
        help="Exact non-cache Grace allocation; repeat per category",
    )
    parser.add_argument(
        "--expert-placement", choices=("linear", "round_robin"), default="linear"
    )
    parser.add_argument("--placement-profile")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-layer expert ID maps",
    )
    return parser


def build_tiered_moe_plan(
    model: str,
    *,
    ep_size: int,
    tp_size: int,
    hbm_capacity_bytes: int,
    hbm_reserve_bytes: int,
    host_capacity_bytes: int,
    host_reserve_bytes: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_sm_count: int,
    kv_block_size: int,
    kv_cache_dtype: str,
    mla_cache_tier: str,
    base_hbm_allocations: Mapping[str, int],
    base_host_allocations: Mapping[str, int],
    expert_placement: str,
    summary_only: bool,
    placement_profile_path: str | None = None,
    num_mtp_layers: int = 0,
    dcp_world_size: int = 1,
) -> dict[str, Any]:
    """Build a fail-closed physical plan without reading model payloads."""
    start = time.perf_counter()
    manifest = build_glm_w4a16_manifest(model)
    non_routed = build_glm_non_routed_runtime_inventory(manifest, tp_size)
    runtime_buffers = plan_tiered_glm_runtime_buffers(
        manifest,
        max_num_batched_tokens=max_num_batched_tokens,
        ep_size=ep_size,
        sm_count=gpu_sm_count,
    )
    placement_profile = (
        load_tiered_moe_placement_profile(placement_profile_path, manifest, ep_size)
        if placement_profile_path is not None
        else None
    )
    reserved_names = {
        "non_routed_weights",
        "main_mla_cache",
        "indexer_cache",
        "tiered_moe_runtime_buffers",
    }
    supplied_names = set(base_hbm_allocations) | set(base_host_allocations)
    overlap = reserved_names & supplied_names
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Planner derives reserved allocation names: {names}")
    if set(base_hbm_allocations) & set(base_host_allocations):
        raise ValueError("Fixed allocation names must be unique across tiers")
    if "non_routed_weights" in base_hbm_allocations:
        raise ValueError("non_routed_weights is derived from the checkpoint")

    cache_tiers = ("host_uva", "hbm") if mla_cache_tier == "auto" else (mla_cache_tier,)
    scenarios = {}
    for cache_tier in cache_tiers:
        scenario = plan_tiered_moe_scenario(
            model,
            manifest=manifest,
            non_routed=non_routed,
            runtime_buffers=runtime_buffers,
            ep_size=ep_size,
            hbm_capacity_bytes=hbm_capacity_bytes,
            hbm_reserve_bytes=hbm_reserve_bytes,
            host_capacity_bytes=host_capacity_bytes,
            host_reserve_bytes=host_reserve_bytes,
            max_model_len=max_model_len,
            kv_block_size=kv_block_size,
            kv_cache_dtype=kv_cache_dtype,
            cache_tier=cache_tier,
            base_hbm_allocations=base_hbm_allocations,
            base_host_allocations=base_host_allocations,
            expert_placement=expert_placement,
            num_mtp_layers=num_mtp_layers,
            placement_profile=placement_profile,
            dcp_world_size=dcp_world_size,
        )
        plan_summaries = [plan.summary() for plan in scenario.rank_plans]
        if summary_only:
            for summary in plan_summaries:
                summary.pop("layers")
        scenarios[cache_tier] = {
            "kv_cache": scenario.kv_cache.summary(),
            "plans": plan_summaries,
        }

    result = {
        "manifest": manifest.summary(ep_size=ep_size),
        "non_routed_runtime": non_routed.summary(),
        "runtime_buffers": runtime_buffers.summary(),
        "scenarios": scenarios,
        "planning_seconds": time.perf_counter() - start,
    }
    if placement_profile is not None:
        result["placement_profile"] = placement_profile.summary()
    return result


def build_tiered_moe_plan_from_vllm_config(vllm_config: Any) -> dict[str, Any]:
    """Build the plan requested by a fully validated vLLM configuration."""
    tiered = vllm_config.tiered_moe_config
    if not tiered.enabled or not tiered.plan_only:
        raise ValueError("A tiered MoE plan-only configuration is required")
    if tiered.grace_machine_profile is None:
        raise ValueError("Tiered MoE plan-only requires a machine profile")
    profile = load_grace_machine_profile(tiered.grace_machine_profile)
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    scheduler = vllm_config.scheduler_config
    model = vllm_config.model_config
    if model is None:
        raise ValueError("Tiered MoE planning requires a model configuration")
    speculative = getattr(vllm_config, "speculative_config", None)
    num_mtp_layers = 0
    if speculative is not None and speculative.method == "mtp":
        num_mtp_layers = (
            speculative.draft_model_config.hf_config.num_nextn_predict_layers
        )
    result = build_tiered_moe_plan(
        model=model.model,
        ep_size=parallel.tensor_parallel_size,
        tp_size=parallel.tensor_parallel_size,
        hbm_capacity_bytes=profile.hbm_capacity_bytes,
        hbm_reserve_bytes=int(tiered.hbm_reserve_gb * 1_000_000_000),
        host_capacity_bytes=profile.host_capacity_bytes,
        host_reserve_bytes=int(tiered.host_reserve_gb * 1_000_000_000),
        max_model_len=model.max_model_len,
        max_num_batched_tokens=scheduler.max_num_batched_tokens,
        gpu_sm_count=profile.gpu_sm_count,
        kv_block_size=cache.block_size,
        kv_cache_dtype=cache.cache_dtype,
        mla_cache_tier=tiered.mla_cache_tier,
        base_hbm_allocations=dict(profile.fixed_hbm_allocations),
        base_host_allocations=dict(profile.fixed_host_allocations),
        expert_placement=parallel.expert_placement_strategy,
        summary_only=False,
        placement_profile_path=tiered.placement_profile,
        num_mtp_layers=num_mtp_layers,
        dcp_world_size=parallel.decode_context_parallel_size,
    )
    result["machine_profile"] = profile.summary()
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Build and print a fail-closed tiered expert plan."""
    args = build_parser().parse_args(argv)
    base_hbm_allocations = dict(args.fixed_hbm_allocation)
    if len(base_hbm_allocations) != len(args.fixed_hbm_allocation):
        raise ValueError("Fixed HBM allocation names must be unique")
    base_host_allocations = dict(args.fixed_host_allocation)
    if len(base_host_allocations) != len(args.fixed_host_allocation):
        raise ValueError("Fixed host allocation names must be unique")
    result = build_tiered_moe_plan(
        args.model,
        ep_size=args.ep_size,
        tp_size=args.tp_size,
        hbm_capacity_bytes=args.hbm_capacity_bytes,
        hbm_reserve_bytes=args.hbm_reserve_bytes,
        host_capacity_bytes=args.host_capacity_bytes,
        host_reserve_bytes=args.host_reserve_bytes,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_sm_count=args.gpu_sm_count,
        kv_block_size=args.kv_block_size,
        kv_cache_dtype=args.kv_cache_dtype,
        mla_cache_tier=args.mla_cache_tier,
        base_hbm_allocations=base_hbm_allocations,
        base_host_allocations=base_host_allocations,
        expert_placement=args.expert_placement,
        summary_only=args.summary_only,
        placement_profile_path=args.placement_profile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
