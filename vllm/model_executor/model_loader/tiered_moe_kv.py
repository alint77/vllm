# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact native KV-cache geometry for the pinned GLM tier plan."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.model_loader.tiered_moe_non_routed import (
    glm_full_indexer_layers,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_MAIN_CACHE_DTYPE = "fp8_ds_mla"
_NATIVE_BLOCK_SIZE = 64
_SCHEDULER_NULL_BLOCKS = 1


def _get_tiered_kv_spec_kind(spec: KVCacheSpec) -> str:
    if not isinstance(spec, MLAAttentionSpec):
        raise ValueError("Tiered GLM KV allocation only supports MLA cache specs")
    if spec.cache_dtype_str == _MAIN_CACHE_DTYPE and spec.model_version is None:
        if spec.page_size_bytes != spec.block_size * 656:
            raise ValueError("Tiered main MLA cache has an unexpected layout")
        return "main"
    if (
        spec.cache_dtype_str is None
        and spec.dtype == torch.uint8
        and spec.page_size_bytes == spec.block_size * 132
    ):
        return "indexer"
    raise ValueError("Tiered GLM encountered an unknown MLA cache spec")


def get_tiered_kv_memory_tier(
    vllm_config: "VllmConfig", spec: KVCacheSpec
) -> tuple[str, str]:
    """Resolve the physical tier and capacity domain for one cache spec."""
    tiered = vllm_config.tiered_moe_config
    if not tiered.enabled:
        return "hbm", "gpu_rank"
    spec_kind = _get_tiered_kv_spec_kind(spec)
    if tiered.mla_cache_tier == "hbm":
        return "hbm", "gpu_rank"
    if tiered.mla_cache_tier != "host_uva":
        raise ValueError("Tiered KV allocation requires an explicit cache tier")
    if spec_kind == "main":
        return "host_uva", "grace_numa_node"
    return "hbm", "gpu_rank"


def get_tiered_kv_available_memory(
    vllm_config: "VllmConfig", kv_cache_groups: list[KVCacheGroupSpec]
) -> int | None:
    """Return the exact cross-tier cache bytes used to derive block count."""
    tiered = vllm_config.tiered_moe_config
    if not tiered.enabled:
        return None
    if len(kv_cache_groups) != 1 or not isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        raise ValueError("Tiered GLM requires one uniform-type MLA cache group")
    specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs.values()
    main_specs = []
    indexer_specs = []
    for spec in specs:
        if _get_tiered_kv_spec_kind(spec) == "main":
            main_specs.append(spec)
        else:
            indexer_specs.append(spec)
    speculative = getattr(vllm_config, "speculative_config", None)
    mtp_layers = 0
    if speculative is not None and speculative.method == "mtp":
        mtp_layers = speculative.draft_model_config.hf_config.num_nextn_predict_layers
    if len(main_specs) != 78 + mtp_layers or len(indexer_specs) != 21 + mtp_layers:
        raise ValueError("Tiered GLM cache spec counts do not match target plus MTP")
    block_sizes = {spec.block_size for spec in (*main_specs, *indexer_specs)}
    if block_sizes != {_NATIVE_BLOCK_SIZE}:
        raise ValueError("Tiered GLM cache specs must use 64-token blocks")
    max_model_len = vllm_config.model_config.max_model_len
    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    logical_block_tokens = _NATIVE_BLOCK_SIZE * dcp_world_size
    blocks_per_seq = (max_model_len + logical_block_tokens - 1) // logical_block_tokens
    num_blocks = max_num_seqs * blocks_per_seq + _SCHEDULER_NULL_BLOCKS
    bytes_per_block = sum(
        spec.page_size_bytes for spec in (*main_specs, *indexer_specs)
    )
    return num_blocks * bytes_per_block


@dataclass(frozen=True)
class TieredKVCachePlan:
    """Exact main-MLA and sparse-indexer cache allocation for one rank."""

    max_model_len: int
    block_size: int
    num_blocks: int
    allocated_tokens: int
    dcp_world_size: int
    main_layer_count: int
    indexer_layer_ids: tuple[int, ...]
    main_page_bytes_per_layer: int
    indexer_page_bytes_per_layer: int
    main_cache_bytes: int
    indexer_cache_bytes: int
    main_cache_tier: str
    main_spec_kind: str = "MLAAttentionSpec"
    main_cache_dtype: str = _MAIN_CACHE_DTYPE
    main_model_version: str | None = None
    indexer_cache_tier: str = "hbm"

    @property
    def hbm_bytes(self) -> int:
        """Physical HBM bytes used by KV caches."""
        main_bytes = self.main_cache_bytes if self.main_cache_tier == "hbm" else 0
        return main_bytes + self.indexer_cache_bytes

    @property
    def host_bytes(self) -> int:
        """Physical paired-Grace bytes used by KV caches."""
        return self.main_cache_bytes if self.main_cache_tier == "host_uva" else 0

    def summary(self) -> dict:
        """Return a compact JSON-compatible cache plan."""
        return {
            "max_model_len": self.max_model_len,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "allocated_tokens": self.allocated_tokens,
            "dcp_world_size": self.dcp_world_size,
            "main_spec_kind": self.main_spec_kind,
            "main_cache_dtype": self.main_cache_dtype,
            "main_model_version": self.main_model_version,
            "main_layer_count": self.main_layer_count,
            "indexer_layer_ids": list(self.indexer_layer_ids),
            "indexer_layer_count": len(self.indexer_layer_ids),
            "main_page_bytes_per_layer": self.main_page_bytes_per_layer,
            "indexer_page_bytes_per_layer": self.indexer_page_bytes_per_layer,
            "main_cache_bytes": self.main_cache_bytes,
            "indexer_cache_bytes": self.indexer_cache_bytes,
            "main_cache_tier": self.main_cache_tier,
            "indexer_cache_tier": self.indexer_cache_tier,
            "hbm_bytes": self.hbm_bytes,
            "host_bytes": self.host_bytes,
        }


def _require_int(config: Mapping[str, Any], name: str, expected: int) -> int:
    value = config.get(name)
    if value != expected:
        raise ValueError(f"Unsupported GLM {name}: expected {expected}, got {value}")
    return expected


def plan_glm_kv_cache(
    config: Mapping[str, Any],
    max_model_len: int,
    block_size: int,
    kv_cache_dtype: str,
    main_cache_tier: str,
    num_mtp_layers: int = 0,
    dcp_world_size: int = 1,
    max_num_seqs: int = 1,
) -> TieredKVCachePlan:
    """Build the exact native cache allocation without loading model tensors.

    Args:
        config: Pinned GLM checkpoint configuration.
        max_model_len: Scheduler-visible token capacity to allocate.
        block_size: Native cache block size selected by the platform.
        kv_cache_dtype: Requested main MLA cache dtype.
        main_cache_tier: Physical tier for the main MLA cache.
        num_mtp_layers: Grafted MTP layers that also hold KV cache.
        dcp_world_size: Decode-context-parallel size; tokens shard across the
            DCP group so each rank stores 1/dcp of every sequence.
        max_num_seqs: Concurrent full-length sequences to provision for.

    Returns:
        Exact per-rank cache geometry and physical byte totals.

    Raises:
        ValueError: If the configuration could select a different cache layout.
    """
    if max_model_len <= 0:
        raise ValueError("Maximum model length must be positive")
    if dcp_world_size < 1:
        raise ValueError("DCP world size must be positive")
    if max_num_seqs < 1:
        raise ValueError("Maximum sequence count must be positive")
    max_position_embeddings = config.get("max_position_embeddings")
    if not isinstance(max_position_embeddings, int):
        raise ValueError("GLM max_position_embeddings must be an integer")
    if max_model_len > max_position_embeddings:
        raise ValueError("Maximum model length exceeds the checkpoint limit")
    if block_size != _NATIVE_BLOCK_SIZE:
        raise ValueError(
            f"GLM sparse MLA requires native block size {_NATIVE_BLOCK_SIZE}"
        )
    if kv_cache_dtype != _MAIN_CACHE_DTYPE:
        raise ValueError(f"GLM main MLA cache requires {_MAIN_CACHE_DTYPE}")
    if main_cache_tier not in {"hbm", "host_uva"}:
        raise ValueError("Main MLA cache tier must be hbm or host_uva")
    if config.get("model_version") is not None:
        raise ValueError("DeepSeek-v4 MLA cache layout is not supported")
    if config.get("sliding_window") is not None or config.get(
        "use_sliding_window", False
    ):
        raise ValueError("Sliding-window MLA cache is not supported")

    num_hidden_layers = _require_int(config, "num_hidden_layers", 78)
    if num_mtp_layers not in (0, 1) or num_mtp_layers > config.get(
        "num_nextn_predict_layers", 0
    ):
        raise ValueError("Unsupported GLM MTP cache layer count")
    cache_layer_count = num_hidden_layers + num_mtp_layers
    kv_lora_rank = _require_int(config, "kv_lora_rank", 512)
    rope_head_dim = _require_int(config, "qk_rope_head_dim", 64)
    index_head_dim = _require_int(config, "index_head_dim", 128)
    _require_int(config, "index_topk_freq", 4)
    _require_int(config, "index_skip_topk_offset", 3)
    if config.get("index_topk_pattern") is not None:
        raise ValueError("Explicit GLM indexer patterns are not supported")

    main_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=kv_lora_rank + rope_head_dim,
        dtype=torch.uint8,
        cache_dtype_str=kv_cache_dtype,
    )
    indexer_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=index_head_dim + 4,
        dtype=torch.uint8,
    )
    if main_spec.model_version is not None or main_spec.sliding_window is not None:
        raise ValueError("Main cache must use ordinary non-v4 MLAAttentionSpec")
    if main_spec.page_size_bytes != block_size * 656:
        raise ValueError("Unexpected GLM main MLA cache layout")
    if indexer_spec.page_size_bytes != block_size * 132:
        raise ValueError("Unexpected GLM sparse-indexer cache layout")

    indexer_layer_ids = glm_full_indexer_layers(
        num_hidden_layers=cache_layer_count,
        frequency=4,
        offset=3,
    )
    if len(indexer_layer_ids) != 21 + num_mtp_layers:
        raise ValueError("Pinned GLM indexer cache count is inconsistent")

    # The KV manager's logical block spans block_size * dcp tokens, one
    # physical block per DCP rank (single_type_kv_cache_manager). Each
    # concurrent sequence needs its own full-length block run.
    logical_block_tokens = block_size * dcp_world_size
    blocks_per_seq = (max_model_len + logical_block_tokens - 1) // logical_block_tokens
    num_blocks = max_num_seqs * blocks_per_seq + _SCHEDULER_NULL_BLOCKS
    return TieredKVCachePlan(
        max_model_len=max_model_len,
        block_size=block_size,
        num_blocks=num_blocks,
        allocated_tokens=num_blocks * block_size,
        dcp_world_size=dcp_world_size,
        main_layer_count=cache_layer_count,
        indexer_layer_ids=indexer_layer_ids,
        main_page_bytes_per_layer=main_spec.page_size_bytes,
        indexer_page_bytes_per_layer=indexer_spec.page_size_bytes,
        main_cache_bytes=num_blocks * main_spec.page_size_bytes * cache_layer_count,
        indexer_cache_bytes=(
            num_blocks * indexer_spec.page_size_bytes * len(indexer_layer_ids)
        ),
        main_cache_tier=main_cache_tier,
    )


def plan_glm_kv_cache_from_path(
    model_path: str | Path,
    max_model_len: int,
    block_size: int,
    kv_cache_dtype: str,
    main_cache_tier: str,
    num_mtp_layers: int = 0,
    dcp_world_size: int = 1,
    max_num_seqs: int = 1,
) -> TieredKVCachePlan:
    """Read only config metadata and build the exact cache plan."""
    config_path = Path(model_path) / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must contain a JSON object")
    return plan_glm_kv_cache(
        config,
        max_model_len=max_model_len,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        main_cache_tier=main_cache_tier,
        num_mtp_layers=num_mtp_layers,
        dcp_world_size=dcp_world_size,
        max_num_seqs=max_num_seqs,
    )
