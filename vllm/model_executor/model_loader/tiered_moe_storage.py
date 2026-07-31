# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact final-destination storage for tiered GLM Marlin experts."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm.model_executor.model_loader.tiered_moe_planner import (
    LayerExpertPlacement,
)
from vllm.model_executor.offloader.grace import GraceAllocation


@dataclass(frozen=True)
class ExpertComponentSpec:
    """One component of the pinned GLM fused-Marlin expert format."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def bytes_per_expert(self) -> int:
        """Return the exact storage occupied by one expert component."""
        return self.dtype.itemsize * self.numel_per_expert

    @property
    def numel_per_expert(self) -> int:
        """Return the number of elements in one expert component."""
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


def glm_marlin_components(group_size: int) -> tuple[ExpertComponentSpec, ...]:
    """Return the fused-Marlin expert layout for one supported group size."""
    if group_size not in (64, 128):
        raise ValueError(f"Unsupported tiered GLM group size: {group_size}")
    return (
        ExpertComponentSpec("w13_weight_packed", (384, 8192), torch.int32),
        ExpertComponentSpec("w2_weight_packed", (128, 12288), torch.int32),
        ExpertComponentSpec(
            "w13_weight_scale", (6144 // group_size, 4096), torch.bfloat16
        ),
        ExpertComponentSpec(
            "w2_weight_scale", (2048 // group_size, 6144), torch.bfloat16
        ),
        ExpertComponentSpec("w13_weight_shape", (2,), torch.bfloat16),
        ExpertComponentSpec("w2_weight_shape", (2,), torch.bfloat16),
    )


GLM_MARLIN_COMPONENTS = glm_marlin_components(128)
GLM_MARLIN_EXPERT_BYTES = sum(
    component.bytes_per_expert for component in GLM_MARLIN_COMPONENTS
)


def build_expert_component_views(
    buffer: torch.Tensor,
    expert_count: int,
    group_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Partition a byte buffer into contiguous component-major expert tensors."""
    if expert_count <= 0:
        raise ValueError("Expert count must be positive")
    invalid_buffer = (
        buffer.dtype != torch.uint8 or buffer.ndim != 1 or not buffer.is_contiguous()
    )
    if invalid_buffer:
        raise ValueError(
            "Expert backing must be a contiguous one-dimensional byte tensor"
        )
    component_specs = glm_marlin_components(group_size)
    expert_bytes = sum(spec.bytes_per_expert for spec in component_specs)
    required_bytes = expert_count * expert_bytes
    if buffer.numel() != required_bytes:
        raise ValueError(
            f"Expert backing has {buffer.numel()} bytes, expected {required_bytes}"
        )
    views = {}
    offset = 0
    for component in component_specs:
        component_bytes = expert_count * component.bytes_per_expert
        raw = buffer.narrow(0, offset, component_bytes)
        views[component.name] = raw.view(component.dtype).view(
            expert_count, *component.shape
        )
        offset += component_bytes
    if offset != required_bytes:
        raise AssertionError("Tiered expert component layout is incomplete")
    return views


@dataclass(frozen=True)
class ExpertTierStorage:
    """Compact tensors and owner backing for one physical expert tier."""

    expert_ids: tuple[int, ...]
    buffer: torch.Tensor
    components: Mapping[str, torch.Tensor]
    component_specs: tuple[ExpertComponentSpec, ...] = GLM_MARLIN_COMPONENTS
    grace_allocation: GraceAllocation | None = None

    @property
    def num_bytes(self) -> int:
        """Return the exact backing size."""
        return self.buffer.numel()

    def copy_expert_from(
        self,
        expert_id: int,
        components: Mapping[str, torch.Tensor],
    ) -> None:
        """Commit one converted expert into its final compact tier slot."""
        try:
            local_index = self.expert_ids.index(expert_id)
        except ValueError as error:
            raise ValueError(
                f"Expert {expert_id} is not assigned to this tier"
            ) from error
        expected_names = {component.name for component in self.component_specs}
        if set(components) != expected_names:
            raise ValueError("Converted expert components do not match GLM Marlin")
        for component in self.component_specs:
            source = components[component.name]
            if source.shape != component.shape or source.dtype != component.dtype:
                raise ValueError(
                    f"Converted {component.name} does not match its final layout"
                )
        for component in self.component_specs:
            self.components[component.name][local_index].copy_(
                components[component.name]
            )


@dataclass(frozen=True)
class LayerTieredExpertStorage:
    """Hot and cold compact fused-Marlin storage for one routed layer."""

    layer_id: int
    hot: ExpertTierStorage | None
    cold: ExpertTierStorage | None

    @property
    def num_bytes(self) -> int:
        """Return total expert storage across both tiers."""
        return sum(tier.num_bytes for tier in (self.hot, self.cold) if tier is not None)


def allocate_layer_expert_storage(
    placement: LayerExpertPlacement,
    device_index: int,
    numa_node: int,
    group_size: int = 128,
) -> LayerTieredExpertStorage:
    """Allocate compact final HBM and pinned-Grace buffers for one layer."""
    component_specs = glm_marlin_components(group_size)
    expert_bytes = sum(spec.bytes_per_expert for spec in component_specs)
    hot = None
    if placement.hot_expert_ids:
        hot_bytes = len(placement.hot_expert_ids) * expert_bytes
        hot_buffer = torch.empty(
            hot_bytes, dtype=torch.uint8, device=torch.device("cuda", device_index)
        )
        hot = ExpertTierStorage(
            expert_ids=placement.hot_expert_ids,
            buffer=hot_buffer,
            components=build_expert_component_views(
                hot_buffer, len(placement.hot_expert_ids), group_size
            ),
            component_specs=component_specs,
        )

    cold = None
    if placement.cold_expert_ids:
        cold_bytes = len(placement.cold_expert_ids) * expert_bytes
        allocation = GraceAllocation.allocate_pinned(
            (cold_bytes,), torch.uint8, device_index, numa_node
        )
        cold = ExpertTierStorage(
            expert_ids=placement.cold_expert_ids,
            buffer=allocation.cuda_alias,
            components=build_expert_component_views(
                allocation.cuda_alias, len(placement.cold_expert_ids), group_size
            ),
            component_specs=component_specs,
            grace_allocation=allocation,
        )

    storage = LayerTieredExpertStorage(placement.layer_id, hot, cold)
    expected_bytes = (
        len(placement.hot_expert_ids) + len(placement.cold_expert_ids)
    ) * expert_bytes
    if storage.num_bytes != expected_bytes:
        raise AssertionError("Tiered layer allocation does not match its placement")
    return storage
