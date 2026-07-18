# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Header-only checkpoint manifests for tiered MoE planning."""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
from safetensors import safe_open

from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)

_INDEX_FILE = "model.safetensors.index.json"
_CONFIG_FILE = "config.json"
_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\."
    r"(weight_packed|weight_scale|weight_shape)$"
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorManifestEntry:
    """Safetensors metadata without materialized tensor data."""

    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    num_bytes: int
    layer_id: int | None = None
    expert_id: int | None = None
    projection: str | None = None
    component: str | None = None

    @property
    def is_routed_expert(self) -> bool:
        """Whether this tensor is part of one routed expert."""
        return self.expert_id is not None


@dataclass(frozen=True)
class TieredMoECheckpointManifest:
    """Exact header-derived inventory for one checkpoint revision."""

    model_path: Path
    config_sha256: str
    index_sha256: str
    entries: tuple[TensorManifestEntry, ...]
    checkpoint_bytes: int
    routed_expert_bytes: int
    non_routed_bytes: int
    routed_layers: tuple[int, ...]
    num_experts: int
    checkpoint_expert_bytes: int
    runtime_expert_bytes: int
    runtime_expert_format: str = "vllm_marlin_static_w4a16"

    def rank_checkpoint_expert_bytes(
        self,
        ep_size: int,
        ep_rank: int,
        placement: str = "linear",
    ) -> int:
        """Return exact checkpoint expert bytes owned by one EP rank."""
        local_ids = compute_local_expert_ids(
            self.num_experts, ep_size, ep_rank, placement
        )
        local_expert_count = self.num_experts if local_ids is None else len(local_ids)
        return (
            len(self.routed_layers) * local_expert_count * self.checkpoint_expert_bytes
        )

    def rank_runtime_expert_bytes(
        self,
        ep_size: int,
        ep_rank: int,
        placement: str = "linear",
    ) -> int:
        """Return final fused-Marlin expert bytes owned by one EP rank."""
        local_ids = compute_local_expert_ids(
            self.num_experts, ep_size, ep_rank, placement
        )
        local_expert_count = self.num_experts if local_ids is None else len(local_ids)
        return len(self.routed_layers) * local_expert_count * self.runtime_expert_bytes

    def summary(self, ep_size: int | None = None) -> dict[str, Any]:
        """Return a compact JSON-compatible manifest summary."""
        result: dict[str, Any] = {
            "model_path": str(self.model_path),
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "tensor_count": len(self.entries),
            "checkpoint_bytes": self.checkpoint_bytes,
            "routed_expert_bytes": self.routed_expert_bytes,
            "non_routed_bytes": self.non_routed_bytes,
            "routed_layers": list(self.routed_layers),
            "num_experts": self.num_experts,
            "checkpoint_expert_bytes": self.checkpoint_expert_bytes,
            "runtime_expert_bytes": self.runtime_expert_bytes,
            "runtime_expert_format": self.runtime_expert_format,
        }
        if ep_size is not None:
            result["ep_size"] = ep_size
            result["rank_checkpoint_expert_bytes"] = [
                self.rank_checkpoint_expert_bytes(ep_size, rank)
                for rank in range(ep_size)
            ]
            result["rank_runtime_expert_bytes"] = [
                self.rank_runtime_expert_bytes(ep_size, rank) for rank in range(ep_size)
            ]
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as file:
            value = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Required checkpoint file is missing: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"Unsupported GLM W4A16 {field}: expected {expected!r}, got {actual!r}"
        )


def _validate_glm_w4a16_config(config: dict[str, Any]) -> tuple[int, ...]:
    _require_equal(
        config.get("architectures"), ["GlmMoeDsaForCausalLM"], "architecture"
    )
    _require_equal(config.get("model_type"), "glm_moe_dsa", "model type")
    _require_equal(config.get("num_hidden_layers"), 78, "layer count")
    _require_equal(config.get("first_k_dense_replace"), 3, "first routed layer")
    _require_equal(config.get("n_routed_experts"), 256, "routed expert count")
    _require_equal(config.get("num_experts_per_tok"), 8, "top-k")
    _require_equal(config.get("hidden_size"), 6144, "hidden size")
    _require_equal(config.get("moe_intermediate_size"), 2048, "MoE size")
    model_dtype = config.get("dtype") or config.get("torch_dtype")
    _require_equal(model_dtype, "bfloat16", "model dtype")

    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("GLM W4A16 quantization_config is missing")
    _require_equal(
        quantization.get("quant_method"), "compressed-tensors", "quant method"
    )
    _require_equal(quantization.get("format"), "pack-quantized", "quant format")
    group = quantization.get("config_groups", {}).get("group_0", {})
    weights = group.get("weights", {})
    required_weights = {
        "actorder": "static",
        "group_size": 128,
        "num_bits": 4,
        "strategy": "group",
        "symmetric": True,
        "type": "int",
    }
    for field, expected in required_weights.items():
        _require_equal(weights.get(field), expected, f"weight {field}")
    return tuple(range(3, 78))


def _tensor_num_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    try:
        element_size = _DTYPE_BYTES[dtype]
    except KeyError as error:
        raise ValueError(
            f"Unsupported safetensors dtype in manifest: {dtype}"
        ) from error
    return math.prod(shape) * element_size


def _make_entry(
    name: str,
    shard: str,
    dtype: str,
    shape: tuple[int, ...],
) -> TensorManifestEntry:
    match = _EXPERT_RE.match(name)
    layer_id = None
    expert_id = None
    projection = None
    component = None
    if match is not None:
        layer, expert, projection, component = match.groups()
        layer_id = int(layer)
        expert_id = int(expert)
    return TensorManifestEntry(
        name=name,
        shard=shard,
        dtype=dtype,
        shape=shape,
        num_bytes=_tensor_num_bytes(dtype, shape),
        layer_id=layer_id,
        expert_id=expert_id,
        projection=projection,
        component=component,
    )


def _validate_expert_entries(
    entries: tuple[TensorManifestEntry, ...],
    routed_layers: tuple[int, ...],
    num_experts: int,
) -> tuple[int, int]:
    expert_entries = [entry for entry in entries if entry.is_routed_expert]
    expected_components = {
        (projection, component)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for component in ("weight_packed", "weight_scale", "weight_shape")
    }
    grouped: dict[tuple[int, int], list[TensorManifestEntry]] = defaultdict(list)
    for entry in expert_entries:
        assert entry.layer_id is not None and entry.expert_id is not None
        grouped[(entry.layer_id, entry.expert_id)].append(entry)

    expected_keys = {
        (layer, expert) for layer in routed_layers for expert in range(num_experts)
    }
    actual_keys = set(grouped)
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        unexpected = len(actual_keys - expected_keys)
        raise ValueError(
            "Incomplete routed expert inventory: "
            f"{missing} missing experts and {unexpected} unexpected experts"
        )

    expert_sizes = set()
    expert_shape_sizes = set()
    for key, components in grouped.items():
        actual_components = {
            (entry.projection, entry.component) for entry in components
        }
        if actual_components != expected_components:
            raise ValueError(f"Incomplete routed expert components for {key}")
        expert_sizes.add(sum(entry.num_bytes for entry in components))
        expert_shape_sizes.add(
            sum(
                entry.num_bytes
                for entry in components
                if entry.component == "weight_shape"
            )
        )
    if len(expert_sizes) != 1:
        raise ValueError("Routed experts do not have a uniform stored size")
    if expert_shape_sizes != {48}:
        raise ValueError("Unexpected routed expert shape metadata size")

    checkpoint_expert_bytes = expert_sizes.pop()
    runtime_shape_bytes = 2 * 2 * 2
    runtime_expert_bytes = checkpoint_expert_bytes - 48 + runtime_shape_bytes
    return checkpoint_expert_bytes, runtime_expert_bytes


def build_glm_w4a16_manifest(model_path: str | Path) -> TieredMoECheckpointManifest:
    """Build an exact manifest using safetensors headers only.

    Args:
        model_path: Local immutable checkpoint directory.

    Returns:
        A validated checkpoint inventory with exact stored byte counts.

    Raises:
        ValueError: If the checkpoint schema, index, or expert coverage is invalid.
    """
    model_path = Path(model_path).resolve()
    config_path = model_path / _CONFIG_FILE
    index_path = model_path / _INDEX_FILE
    routed_layers = _validate_glm_w4a16_config(_read_json(config_path))
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Checkpoint weight_map is missing or empty: {index_path}")

    names_by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("Checkpoint weight_map entries must be strings")
        if Path(shard).name != shard:
            raise ValueError(f"Checkpoint shard must be a local filename: {shard}")
        names_by_shard[shard].add(name)

    entries = []
    for shard, expected_names in sorted(names_by_shard.items()):
        shard_path = model_path / shard
        if not shard_path.is_file():
            raise ValueError(f"Checkpoint shard is missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            actual_names = set(file.keys())
            if actual_names != expected_names:
                missing = len(expected_names - actual_names)
                unexpected = len(actual_names - expected_names)
                raise ValueError(
                    f"Safetensors index mismatch in {shard}: "
                    f"{missing} missing and {unexpected} unexpected tensors"
                )
            for name in sorted(actual_names):
                tensor_slice = file.get_slice(name)
                entries.append(
                    _make_entry(
                        name,
                        shard,
                        tensor_slice.get_dtype(),
                        tuple(tensor_slice.get_shape()),
                    )
                )

    manifest_entries = tuple(sorted(entries, key=lambda entry: entry.name))
    checkpoint_bytes = sum(entry.num_bytes for entry in manifest_entries)
    metadata = index.get("metadata", {})
    declared_bytes = metadata.get("total_size") if isinstance(metadata, dict) else None
    if declared_bytes != checkpoint_bytes:
        raise ValueError(
            "Safetensors index total_size mismatch: "
            f"declared {declared_bytes}, headers contain {checkpoint_bytes}"
        )

    checkpoint_expert_bytes, runtime_expert_bytes = _validate_expert_entries(
        manifest_entries, routed_layers, 256
    )
    routed_expert_bytes = sum(
        entry.num_bytes for entry in manifest_entries if entry.is_routed_expert
    )
    return TieredMoECheckpointManifest(
        model_path=model_path,
        config_sha256=_sha256(config_path),
        index_sha256=_sha256(index_path),
        entries=manifest_entries,
        checkpoint_bytes=checkpoint_bytes,
        routed_expert_bytes=routed_expert_bytes,
        non_routed_bytes=checkpoint_bytes - routed_expert_bytes,
        routed_layers=routed_layers,
        num_experts=256,
        checkpoint_expert_bytes=checkpoint_expert_bytes,
        runtime_expert_bytes=runtime_expert_bytes,
    )
