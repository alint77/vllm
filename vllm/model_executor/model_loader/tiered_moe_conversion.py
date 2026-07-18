# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded one-expert checkpoint staging and Marlin conversion."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
)

_EXPERT_COMPONENT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\."
    r"(weight_packed|weight_scale|weight_shape)$"
)
_CHECKPOINT_COMPONENTS = {
    "gate_proj.weight_packed": ((2048, 768), torch.int32),
    "gate_proj.weight_scale": ((2048, 48), torch.bfloat16),
    "gate_proj.weight_shape": ((2,), torch.int64),
    "up_proj.weight_packed": ((2048, 768), torch.int32),
    "up_proj.weight_scale": ((2048, 48), torch.bfloat16),
    "up_proj.weight_shape": ((2,), torch.int64),
    "down_proj.weight_packed": ((6144, 256), torch.int32),
    "down_proj.weight_scale": ((6144, 16), torch.bfloat16),
    "down_proj.weight_shape": ((2,), torch.int64),
}


def is_glm_expert_checkpoint_tensor(name: str) -> bool:
    """Return whether ``name`` is a routed GLM expert component."""
    return _EXPERT_COMPONENT_RE.fullmatch(name) is not None


@dataclass(frozen=True)
class GlmExpertCheckpointBundle:
    """Every stored component for one routed GLM expert."""

    layer_id: int
    expert_id: int
    components: Mapping[str, torch.Tensor]

    @property
    def num_bytes(self) -> int:
        """Return exact staged CPU payload bytes."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.components.values()
        )


class OneExpertCheckpointStager:
    """Accumulate one complete expert and reject unbounded interleaving."""

    def __init__(self) -> None:
        self._key: tuple[int, int] | None = None
        self._components: dict[str, torch.Tensor] = {}

    def add(self, name: str, tensor: torch.Tensor) -> GlmExpertCheckpointBundle | None:
        """Stage one component and return a bundle when all nine are present."""
        match = _EXPERT_COMPONENT_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"Cannot stage non-GLM expert tensor: {name}")
        layer_id, expert_id = (int(value) for value in match.groups()[:2])
        component_name = ".".join(match.groups()[2:])
        key = (layer_id, expert_id)
        if self._key is None:
            self._key = key
        elif self._key != key:
            raise ValueError(
                "Checkpoint expert components are interleaved or incomplete"
            )
        if component_name in self._components:
            raise ValueError(f"Duplicate expert component: {component_name}")
        expected_shape, expected_dtype = _CHECKPOINT_COMPONENTS[component_name]
        if tensor.shape != expected_shape or tensor.dtype != expected_dtype:
            raise ValueError(f"Checkpoint {component_name} does not match pinned GLM")
        self._components[component_name] = tensor
        if set(self._components) != set(_CHECKPOINT_COMPONENTS):
            return None
        bundle = GlmExpertCheckpointBundle(layer_id, expert_id, dict(self._components))
        if bundle.num_bytes != 19_464_240:
            raise AssertionError("Staged expert checkpoint bytes do not match manifest")
        self._key = None
        self._components.clear()
        return bundle

    def finish(self) -> None:
        """Fail if iteration ended with a partial expert bundle."""
        if self._key is not None or self._components:
            raise ValueError("Checkpoint ended with an incomplete expert bundle")


def convert_glm_w4a16_expert(
    bundle: GlmExpertCheckpointBundle,
    layer: torch.nn.Module,
    quant_method: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Fuse and repack one staged expert into the final Marlin format."""
    if quant_method.wna16_backend != WNA16MoEBackend.MARLIN:
        raise ValueError("Tiered GLM conversion initially requires Marlin")
    if quant_method.actorder != "static" or quant_method.group_size != 128:
        raise ValueError("Tiered GLM conversion requires static group-128 weights")
    checkpoint = bundle.components

    def transposed(name: str) -> torch.Tensor:
        return checkpoint[name].t().contiguous().to(device)

    staging = {
        "w13": torch.cat(
            [
                transposed("gate_proj.weight_packed"),
                transposed("up_proj.weight_packed"),
            ],
            dim=1,
        ).unsqueeze(0),
        "w2": transposed("down_proj.weight_packed").unsqueeze(0),
        "w13_scale": torch.cat(
            [
                transposed("gate_proj.weight_scale"),
                transposed("up_proj.weight_scale"),
            ],
            dim=1,
        ).unsqueeze(0),
        "w2_scale": transposed("down_proj.weight_scale").unsqueeze(0),
        "w13_g_idx": torch.empty((1, 6144), dtype=torch.int32, device=device),
        "w2_g_idx": torch.empty((1, 2048), dtype=torch.int32, device=device),
    }
    converted = convert_to_wna16_moe_kernel_format(
        backend=quant_method.wna16_backend,
        layer=layer,
        quant_config=quant_method.weight_quant,
        input_dtype=quant_method.marlin_input_dtype,
        **staging,
    )
    if converted is None:
        raise AssertionError("Marlin conversion must return final tensors")
    return {
        "w13_weight_packed": converted[0][0],
        "w2_weight_packed": converted[1][0],
        "w13_weight_scale": converted[2][0],
        "w2_weight_scale": converted[3][0],
        "w13_weight_shape": checkpoint["gate_proj.weight_shape"].to(
            device=device, dtype=torch.bfloat16
        ),
        "w2_weight_shape": checkpoint["down_proj.weight_shape"].to(
            device=device, dtype=torch.bfloat16
        ),
    }
