# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact non-routed GLM runtime weight accounting."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import regex as re

from vllm.model_executor.model_loader.tiered_moe_manifest import (
    TensorManifestEntry,
    TieredMoECheckpointManifest,
)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


class RuntimePlacement(str, Enum):
    """Physical placement behavior for a checkpoint tensor."""

    REPLICATED = "replicated"
    TP_SHARDED = "tp_sharded"
    EP_SHARDED = "ep_sharded"
    DROPPED = "dropped"


@dataclass(frozen=True)
class NonRoutedRuntimeInventory:
    """Per-rank runtime bytes derived from non-routed checkpoint tensors."""

    tp_size: int
    checkpoint_bytes: int
    replicated_bytes: int
    tp_sharded_checkpoint_bytes: int
    tp_sharded_runtime_bytes: int
    ep_sharded_checkpoint_bytes: int
    ep_sharded_runtime_bytes: int
    dropped_checkpoint_bytes: int
    fusion_savings_bytes: int
    format_conversion_bytes: int
    runtime_bytes_per_rank: int
    full_indexer_layers: tuple[int, ...]
    placement_counts: tuple[tuple[str, int], ...]

    def summary(self) -> dict:
        """Return a compact JSON-compatible inventory."""
        return {
            "tp_size": self.tp_size,
            "checkpoint_bytes": self.checkpoint_bytes,
            "replicated_bytes": self.replicated_bytes,
            "tp_sharded_checkpoint_bytes": self.tp_sharded_checkpoint_bytes,
            "tp_sharded_runtime_bytes": self.tp_sharded_runtime_bytes,
            "ep_sharded_checkpoint_bytes": self.ep_sharded_checkpoint_bytes,
            "ep_sharded_runtime_bytes": self.ep_sharded_runtime_bytes,
            "dropped_checkpoint_bytes": self.dropped_checkpoint_bytes,
            "fusion_savings_bytes": self.fusion_savings_bytes,
            "format_conversion_bytes": self.format_conversion_bytes,
            "runtime_bytes_per_rank": self.runtime_bytes_per_rank,
            "full_indexer_layers": list(self.full_indexer_layers),
            "placement_counts": dict(self.placement_counts),
        }


def glm_full_indexer_layers(
    num_hidden_layers: int = 78,
    frequency: int = 4,
    offset: int = 3,
) -> tuple[int, ...]:
    """Return layers that instantiate the native GLM sparse indexer."""
    return tuple(
        layer
        for layer in range(num_hidden_layers)
        if max(layer - offset + 1, 0) % frequency == 0
    )


def _linear_component_placement(
    suffix: str, parallel: RuntimePlacement
) -> RuntimePlacement:
    if suffix.endswith(".weight_shape"):
        return RuntimePlacement.REPLICATED
    if suffix.endswith(
        (".weight", ".weight_packed", ".weight_scale", ".weight_scale_inv")
    ):
        return parallel
    raise ValueError(f"Unsupported quantized linear component: {suffix}")


def _classify_layer_tensor(
    layer_id: int,
    suffix: str,
    full_indexer_layers: set[int],
) -> RuntimePlacement:
    if layer_id == 78:
        if re.match(r"^mlp\.experts\.\d+\.", suffix):
            return RuntimePlacement.EP_SHARDED
        if suffix.startswith("self_attn.indexer."):
            return RuntimePlacement.REPLICATED
        if suffix in {
            "eh_proj.weight",
            "enorm.weight",
            "hnorm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "shared_head.norm.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias",
        }:
            return RuntimePlacement.REPLICATED
        if suffix.startswith(("self_attn.q_a_proj.", "self_attn.kv_a_proj_with_mqa.")):
            return _linear_component_placement(suffix, RuntimePlacement.REPLICATED)
        if suffix.startswith(
            (
                "self_attn.q_b_proj.",
                "self_attn.kv_b_proj.",
                "self_attn.o_proj.",
                "mlp.shared_experts.gate_proj.",
                "mlp.shared_experts.up_proj.",
                "mlp.shared_experts.down_proj.",
            )
        ):
            return _linear_component_placement(suffix, RuntimePlacement.TP_SHARDED)
        raise ValueError(f"Unclassified GLM MTP tensor: model.layers.78.{suffix}")
    if suffix.startswith("self_attn.indexer."):
        return (
            RuntimePlacement.REPLICATED
            if layer_id in full_indexer_layers
            else RuntimePlacement.DROPPED
        )
    if suffix in {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
    }:
        return RuntimePlacement.REPLICATED

    replicated_linears = (
        "self_attn.q_a_proj.",
        "self_attn.kv_a_proj_with_mqa.",
    )
    if suffix.startswith(replicated_linears):
        return _linear_component_placement(suffix, RuntimePlacement.REPLICATED)

    sharded_linears = (
        "self_attn.q_b_proj.",
        "self_attn.kv_b_proj.",
        "self_attn.o_proj.",
        "mlp.gate_proj.",
        "mlp.up_proj.",
        "mlp.down_proj.",
        "mlp.shared_experts.gate_proj.",
        "mlp.shared_experts.up_proj.",
        "mlp.shared_experts.down_proj.",
    )
    if suffix.startswith(sharded_linears):
        return _linear_component_placement(suffix, RuntimePlacement.TP_SHARDED)
    raise ValueError(
        f"Unclassified GLM runtime tensor: model.layers.{layer_id}.{suffix}"
    )


def _classify_entry(
    entry: TensorManifestEntry,
    full_indexer_layers: set[int],
) -> RuntimePlacement:
    if entry.name in {"lm_head.weight", "model.embed_tokens.weight"}:
        return RuntimePlacement.TP_SHARDED
    if entry.name == "model.norm.weight":
        return RuntimePlacement.REPLICATED
    match = _LAYER_RE.match(entry.name)
    if match is None:
        raise ValueError(f"Unclassified non-routed checkpoint tensor: {entry.name}")
    layer_id, suffix = match.groups()
    return _classify_layer_tensor(int(layer_id), suffix, full_indexer_layers)


def _shape_fusion_group(entry: TensorManifestEntry) -> tuple[int, str] | None:
    if not entry.name.endswith(".weight_shape"):
        return None
    match = _LAYER_RE.match(entry.name)
    if match is None:
        return None
    layer_id = int(match.group(1))
    suffix = match.group(2)
    groups = {
        "self_attn.q_a_proj.weight_shape": "qkv_a",
        "self_attn.kv_a_proj_with_mqa.weight_shape": "qkv_a",
        "mlp.gate_proj.weight_shape": "dense_gate_up",
        "mlp.up_proj.weight_shape": "dense_gate_up",
        "mlp.shared_experts.gate_proj.weight_shape": "shared_gate_up",
        "mlp.shared_experts.up_proj.weight_shape": "shared_gate_up",
    }
    group = groups.get(suffix)
    return None if group is None else (layer_id, group)


def build_glm_non_routed_runtime_inventory(
    manifest: TieredMoECheckpointManifest,
    tp_size: int,
) -> NonRoutedRuntimeInventory:
    """Classify every non-routed tensor into its physical TP4 runtime shape.

    Args:
        manifest: Validated pinned GLM checkpoint manifest.
        tp_size: Tensor-parallel world size.

    Returns:
        Exact per-rank non-routed runtime weight bytes.

    Raises:
        ValueError: If a tensor is unknown or cannot be evenly TP-sharded.
    """
    if tp_size <= 0:
        raise ValueError("TP size must be positive")
    full_indexer_layers = glm_full_indexer_layers()
    full_indexer_layer_set = set(full_indexer_layers)
    totals: defaultdict[RuntimePlacement, int] = defaultdict(int)
    counts: defaultdict[RuntimePlacement, int] = defaultdict(int)
    fusion_shapes: dict[tuple[int, str], list[int]] = defaultdict(list)

    for entry in manifest.entries:
        if entry.is_routed_expert:
            continue
        placement = _classify_entry(entry, full_indexer_layer_set)
        totals[placement] += entry.num_bytes
        counts[placement] += 1
        if placement is not RuntimePlacement.DROPPED:
            fusion_group = _shape_fusion_group(entry)
            if fusion_group is not None:
                fusion_shapes[fusion_group].append(entry.num_bytes)

    classified_checkpoint_bytes = sum(totals.values())
    if classified_checkpoint_bytes != manifest.non_routed_bytes:
        raise ValueError(
            "Non-routed classifier did not reconcile with the checkpoint manifest"
        )
    sharded_checkpoint_bytes = totals[RuntimePlacement.TP_SHARDED]
    if sharded_checkpoint_bytes % tp_size:
        raise ValueError("Non-routed tensor bytes are not evenly TP-shardable")

    fusion_savings_bytes = 0
    for group, shape_sizes in fusion_shapes.items():
        if len(shape_sizes) != 2 or len(set(shape_sizes)) != 1:
            raise ValueError(f"Incomplete fused runtime shape metadata for {group}")
        fusion_savings_bytes += shape_sizes[0]

    replicated_bytes = totals[RuntimePlacement.REPLICATED]
    sharded_runtime_bytes = sharded_checkpoint_bytes // tp_size
    ep_sharded_checkpoint_bytes = totals[RuntimePlacement.EP_SHARDED]
    if ep_sharded_checkpoint_bytes % tp_size:
        raise ValueError("MTP expert tensor bytes are not evenly EP-shardable")
    ep_sharded_runtime_bytes = ep_sharded_checkpoint_bytes // tp_size
    mtp_indexer_wk_bytes = sum(
        entry.num_bytes
        for entry in manifest.entries
        if entry.name == "model.layers.78.self_attn.indexer.wk.weight"
    )
    mtp_indexer_scale_bytes = sum(
        entry.num_bytes
        for entry in manifest.entries
        if entry.name == "model.layers.78.self_attn.indexer.wk.weight_scale_inv"
    )
    format_conversion_bytes = mtp_indexer_wk_bytes - mtp_indexer_scale_bytes
    runtime_bytes_per_rank = (
        replicated_bytes
        + sharded_runtime_bytes
        + ep_sharded_runtime_bytes
        + format_conversion_bytes
        - fusion_savings_bytes
    )
    return NonRoutedRuntimeInventory(
        tp_size=tp_size,
        checkpoint_bytes=manifest.non_routed_bytes,
        replicated_bytes=replicated_bytes,
        tp_sharded_checkpoint_bytes=sharded_checkpoint_bytes,
        tp_sharded_runtime_bytes=sharded_runtime_bytes,
        ep_sharded_checkpoint_bytes=ep_sharded_checkpoint_bytes,
        ep_sharded_runtime_bytes=ep_sharded_runtime_bytes,
        dropped_checkpoint_bytes=totals[RuntimePlacement.DROPPED],
        fusion_savings_bytes=fusion_savings_bytes,
        format_conversion_bytes=format_conversion_bytes,
        runtime_bytes_per_rank=runtime_bytes_per_rank,
        full_indexer_layers=full_indexer_layers,
        placement_counts=tuple(
            sorted(
                (placement.value, counts[placement]) for placement in RuntimePlacement
            )
        ),
    )
