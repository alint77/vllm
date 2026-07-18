# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact static runtime and bounded conversion buffers for tiered GLM MoE."""

from dataclasses import dataclass

from vllm.model_executor.model_loader.tiered_moe_manifest import (
    TieredMoECheckpointManifest,
)

GLM_ONE_EXPERT_CONVERSION_SCRATCH_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TieredMoERuntimeBuffers:
    """Per-rank HBM buffers required by independent hot/cold Marlin calls."""

    max_num_batched_tokens: int
    topk: int
    hidden_size: int
    intermediate_size: int
    global_num_experts: int
    local_num_experts: int
    routed_layer_count: int
    sm_count: int
    marlin_block_size: int
    intermediate_bytes: int
    alignment_bytes: int
    marlin_lock_bytes: int
    placement_map_bytes: int
    remap_bytes: int
    steady_hbm_bytes: int
    conversion_scratch_bytes: int

    def summary(self) -> dict:
        """Return a JSON-compatible runtime buffer inventory."""
        return {
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "topk": self.topk,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "global_num_experts": self.global_num_experts,
            "local_num_experts": self.local_num_experts,
            "routed_layer_count": self.routed_layer_count,
            "sm_count": self.sm_count,
            "marlin_block_size": self.marlin_block_size,
            "intermediate_bytes": self.intermediate_bytes,
            "alignment_bytes": self.alignment_bytes,
            "marlin_lock_bytes": self.marlin_lock_bytes,
            "placement_map_bytes": self.placement_map_bytes,
            "remap_bytes": self.remap_bytes,
            "steady_hbm_bytes": self.steady_hbm_bytes,
            "conversion_scratch_bytes": self.conversion_scratch_bytes,
        }


def plan_tiered_glm_runtime_buffers(
    manifest: TieredMoECheckpointManifest,
    max_num_batched_tokens: int,
    ep_size: int,
    sm_count: int,
) -> TieredMoERuntimeBuffers:
    """Size the static two-tier Marlin buffers and one-expert load scratch.

    The hot and cold calls have independent buffers so they may execute on
    separate streams. Each call follows ``MarlinExperts.workspace_shapes``:
    two simultaneous BF16 arenas sized for ``M * topk * max(2N, K)``. The
    pinned GLM dimensions make both arenas the same size.

    Args:
        manifest: Validated pinned GLM checkpoint inventory.
        max_num_batched_tokens: Largest scheduler batch handled by one rank.
        ep_size: Expert-parallel world size.
        sm_count: Physical streaming multiprocessors on the Hopper GPU.

    Returns:
        Exact steady runtime buffers and bounded transient conversion scratch.

    Raises:
        ValueError: If the initial pinned TP4/EP4 buffer contract is violated.
    """
    if max_num_batched_tokens <= 0:
        raise ValueError("Maximum batched tokens must be positive")
    if ep_size != 4:
        raise ValueError("Tiered GLM runtime buffers initially require EP4")
    if sm_count <= 0:
        raise ValueError("SM count must be positive")
    if manifest.num_experts % ep_size:
        raise ValueError("Experts must divide evenly across EP ranks")

    topk = 8
    hidden_size = 6144
    intermediate_size = 2048
    local_num_experts = manifest.num_experts // ep_size
    routed_layer_count = len(manifest.routed_layers)
    tier_count = 2
    element_bytes = 2

    arena_elements = (
        max_num_batched_tokens * topk * max(2 * intermediate_size, hidden_size)
    )
    intermediate_bytes = tier_count * 2 * arena_elements * element_bytes

    marlin_block_size = 64
    assignments = max_num_batched_tokens * topk
    aligned_ids_per_tier = assignments + manifest.num_experts * (marlin_block_size - 1)
    blocks_per_tier = (
        aligned_ids_per_tier + marlin_block_size - 1
    ) // marlin_block_size
    alignment_bytes = tier_count * (aligned_ids_per_tier * 4 + blocks_per_tier * 4 + 4)

    marlin_lock_bytes = tier_count * sm_count * 4 * 4
    map_entries_per_layer = 3 * manifest.num_experts + local_num_experts
    placement_map_bytes = routed_layer_count * map_entries_per_layer * 4
    remap_bytes = tier_count * assignments * (4 + 4)
    steady_hbm_bytes = (
        intermediate_bytes
        + alignment_bytes
        + marlin_lock_bytes
        + placement_map_bytes
        + remap_bytes
    )
    conversion_scratch_bytes = GLM_ONE_EXPERT_CONVERSION_SCRATCH_BYTES
    return TieredMoERuntimeBuffers(
        max_num_batched_tokens=max_num_batched_tokens,
        topk=topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        global_num_experts=manifest.num_experts,
        local_num_experts=local_num_experts,
        routed_layer_count=routed_layer_count,
        sm_count=sm_count,
        marlin_block_size=marlin_block_size,
        intermediate_bytes=intermediate_bytes,
        alignment_bytes=alignment_bytes,
        marlin_lock_bytes=marlin_lock_bytes,
        placement_map_bytes=placement_map_bytes,
        remap_bytes=remap_bytes,
        steady_hbm_bytes=steady_hbm_bytes,
        conversion_scratch_bytes=conversion_scratch_bytes,
    )
