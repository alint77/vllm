# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-side replica assignment for tiered MoE decode."""

from collections.abc import Sequence

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_HBM_COST = 1280
_GRACE_COST = 3467


def _replica_route_hash(topk_ids: torch.Tensor) -> torch.Tensor:
    routes = topk_ids.reshape(-1).to(torch.int64) + 1
    positions = torch.arange(
        1,
        routes.numel() + 1,
        dtype=torch.int64,
        device=routes.device,
    )
    modulus = 1_000_003
    return torch.stack(
        (
            routes.new_tensor(routes.numel() % modulus),
            routes.sum() % modulus,
            (routes * (positions % 251)).sum() % modulus,
            (routes * (positions % 509)).sum() % modulus,
        )
    ).to(torch.float32)


def validate_replicated_routes(topk_ids: torch.Tensor, ep_size: int) -> None:
    """Raise when replica scheduling inputs differ across EP ranks."""
    from vllm.distributed.parallel_state import get_ep_group

    local_hash = _replica_route_hash(topk_ids)
    reduced_hash = get_ep_group().all_reduce(local_hash)
    if not torch.equal(reduced_hash, local_hash * ep_size):
        raise RuntimeError(
            "Tiered MoE replica assignment detected divergent cross-rank routes"
        )


def greedy_replica_assignment(
    route_counts: Sequence[int],
    primary_ranks: Sequence[int],
    secondary_ranks: Sequence[int],
    primary_hot: Sequence[bool],
    ep_size: int,
) -> tuple[int, ...]:
    """Return the deterministic reference assignment used by the GPU kernel."""
    if not (
        len(route_counts)
        == len(primary_ranks)
        == len(secondary_ranks)
        == len(primary_hot)
    ):
        raise ValueError("Replica assignment inputs have different lengths")

    hbm = [0] * ep_size
    grace = [0] * ep_size
    selected = [-1] * len(route_counts)
    fixed = [
        expert
        for expert, count in enumerate(route_counts)
        if count and secondary_ranks[expert] < 0
    ]
    for expert in fixed:
        rank = primary_ranks[expert]
        selected[expert] = rank
        if primary_hot[expert]:
            hbm[rank] += _HBM_COST
        else:
            grace[rank] += _GRACE_COST

    flexible = [
        expert
        for expert, count in enumerate(route_counts)
        if count and secondary_ranks[expert] >= 0
    ]
    flexible.sort(
        key=lambda expert: (
            not primary_hot[expert],
            route_counts[expert],
            -expert,
        ),
        reverse=True,
    )

    for expert in flexible:
        primary = primary_ranks[expert]
        secondary = secondary_ranks[expert]
        choices = []
        for rank in (primary, secondary):
            use_hbm = rank == primary and primary_hot[expert]
            candidate_hbm = hbm.copy()
            candidate_grace = grace.copy()
            if use_hbm:
                candidate_hbm[rank] += _HBM_COST
            else:
                candidate_grace[rank] += _GRACE_COST
            times = [
                max(hbm_time, grace_time)
                for hbm_time, grace_time in zip(
                    candidate_hbm,
                    candidate_grace,
                )
            ]
            choices.append((max(times), sum(times), rank, use_hbm))
        _, _, rank, use_hbm = min(choices)
        selected[expert] = rank
        if use_hbm:
            hbm[rank] += _HBM_COST
        else:
            grace[rank] += _GRACE_COST
    return tuple(selected)


@triton.jit
def _assign_replicated_experts_kernel(
    topk_ids_ptr,
    primary_ranks_ptr,
    secondary_ranks_ptr,
    primary_hot_ptr,
    hot_primary_map_ptr,
    cold_primary_map_ptr,
    cold_physical_map_ptr,
    hot_output_map_ptr,
    cold_output_map_ptr,
    selected_ranks_ptr,
    NUM_ROUTES,
    NUM_EXPERTS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EP_RANK: tl.constexpr,
    SCHEDULE: tl.constexpr,
    HBM_COST: tl.constexpr,
    GRACE_COST: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_RANKS: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_EXPERTS)
    expert_mask = experts < NUM_EXPERTS
    primary_ranks = tl.load(
        primary_ranks_ptr + experts,
        mask=expert_mask,
        other=0,
    ).to(tl.int32)
    hot_primary_map = tl.load(
        hot_primary_map_ptr + experts,
        mask=expert_mask,
        other=-1,
    ).to(tl.int32)
    cold_primary_map = tl.load(
        cold_primary_map_ptr + experts,
        mask=expert_mask,
        other=-1,
    ).to(tl.int32)
    if not SCHEDULE:
        tl.store(
            hot_output_map_ptr + experts,
            hot_primary_map,
            mask=expert_mask,
        )
        tl.store(
            cold_output_map_ptr + experts,
            cold_primary_map,
            mask=expert_mask,
        )
        tl.store(
            selected_ranks_ptr + experts,
            primary_ranks,
            mask=expert_mask,
        )
        return

    secondary_ranks = tl.load(
        secondary_ranks_ptr + experts,
        mask=expert_mask,
        other=-1,
    ).to(tl.int32)
    primary_hot = tl.load(
        primary_hot_ptr + experts,
        mask=expert_mask,
        other=0,
    ).to(tl.int32)
    counts = tl.zeros((BLOCK_EXPERTS,), dtype=tl.int32)
    for route_index in tl.range(0, NUM_ROUTES):
        routed_expert = tl.load(topk_ids_ptr + route_index).to(tl.int32)
        counts += (experts == routed_expert).to(tl.int32)

    ranks = tl.arange(0, BLOCK_RANKS)
    rank_mask = ranks < EP_SIZE
    hbm = tl.zeros((BLOCK_RANKS,), dtype=tl.int32)
    grace = tl.zeros((BLOCK_RANKS,), dtype=tl.int32)

    active = (counts > 0) & expert_mask
    fixed = active & (secondary_ranks < 0)
    hot_output_map = tl.where(
        fixed & (primary_hot > 0),
        hot_primary_map,
        -1,
    )
    cold_output_map = tl.where(
        fixed & (primary_hot == 0),
        cold_primary_map,
        -1,
    )
    selected_ranks_output = tl.where(
        fixed,
        primary_ranks,
        -1,
    )
    for rank in tl.static_range(0, EP_SIZE):
        fixed_hbm = tl.sum(
            (fixed & (primary_ranks == rank) & (primary_hot > 0)).to(tl.int32),
            axis=0,
        )
        fixed_grace = tl.sum(
            (fixed & (primary_ranks == rank) & (primary_hot == 0)).to(tl.int32),
            axis=0,
        )
        hbm += tl.where(ranks == rank, fixed_hbm * HBM_COST, 0)
        grace += tl.where(ranks == rank, fixed_grace * GRACE_COST, 0)

    flexible = active & (secondary_ranks >= 0)
    flexible_count = tl.sum(flexible.to(tl.int32), axis=0)
    cold_priority = (1 - primary_hot) * (NUM_ROUTES + 1)
    score = (cold_priority + counts) * NUM_EXPERTS
    score += NUM_EXPERTS - 1 - experts
    score = tl.where(flexible, score, -1)
    ordered_scores = tl.sort(score, descending=True)
    ordered_experts = NUM_EXPERTS - 1 - (ordered_scores % NUM_EXPERTS)
    # Triton cannot dynamically index register vectors, so reuse the declared
    # output as CTA-local ordering scratch until the final assignment is ready.
    tl.store(
        selected_ranks_ptr + experts,
        ordered_experts,
        mask=expert_mask,
    )
    tl.debug_barrier()

    index = 0
    while index < flexible_count:
        expert = tl.load(selected_ranks_ptr + index).to(tl.int32)
        chosen_mask = experts == expert

        primary = tl.load(primary_ranks_ptr + expert).to(tl.int32)
        secondary = tl.load(secondary_ranks_ptr + expert).to(tl.int32)
        is_hot = tl.load(primary_hot_ptr + expert).to(tl.int32) > 0

        primary_hbm = hbm + tl.where(
            rank_mask & (ranks == primary) & is_hot,
            HBM_COST,
            0,
        )
        primary_grace = grace + tl.where(
            rank_mask & (ranks == primary) & ~is_hot,
            GRACE_COST,
            0,
        )
        primary_times = tl.where(
            rank_mask,
            tl.maximum(primary_hbm, primary_grace),
            0,
        )
        primary_max = tl.max(primary_times, axis=0)
        primary_sum = tl.sum(primary_times, axis=0)

        secondary_grace = grace + tl.where(
            rank_mask & (ranks == secondary),
            GRACE_COST,
            0,
        )
        secondary_times = tl.where(
            rank_mask,
            tl.maximum(hbm, secondary_grace),
            0,
        )
        secondary_max = tl.max(secondary_times, axis=0)
        secondary_sum = tl.sum(secondary_times, axis=0)
        use_secondary = (secondary >= 0) & (
            (secondary_max < primary_max)
            | (
                (secondary_max == primary_max)
                & (
                    (secondary_sum < primary_sum)
                    | ((secondary_sum == primary_sum) & (secondary < primary))
                )
            )
        )

        selected_rank = tl.where(use_secondary, secondary, primary)
        selected_hot = is_hot & ~use_secondary
        update_mask = rank_mask & (ranks == selected_rank)
        hbm += tl.where(update_mask & selected_hot, HBM_COST, 0)
        grace += tl.where(update_mask & ~selected_hot, GRACE_COST, 0)

        local_choice = chosen_mask & (selected_rank == EP_RANK)
        hot_index = tl.load(hot_primary_map_ptr + expert).to(tl.int32)
        cold_index = tl.load(cold_physical_map_ptr + expert).to(tl.int32)
        hot_output_map = tl.where(
            local_choice & selected_hot,
            hot_index,
            hot_output_map,
        )
        cold_output_map = tl.where(
            local_choice & ~selected_hot,
            cold_index,
            cold_output_map,
        )
        selected_ranks_output = tl.where(
            chosen_mask,
            selected_rank,
            selected_ranks_output,
        )
        index += 1

    tl.store(
        hot_output_map_ptr + experts,
        hot_output_map,
        mask=expert_mask,
    )
    tl.store(
        cold_output_map_ptr + experts,
        cold_output_map,
        mask=expert_mask,
    )
    tl.store(
        selected_ranks_ptr + experts,
        selected_ranks_output,
        mask=expert_mask,
    )


def _assign_replicated_experts(
    topk_ids: torch.Tensor,
    primary_ranks: torch.Tensor,
    secondary_ranks: torch.Tensor,
    primary_hot: torch.Tensor,
    hot_primary_map: torch.Tensor,
    cold_primary_map: torch.Tensor,
    cold_physical_map: torch.Tensor,
    hot_output_map: torch.Tensor,
    cold_output_map: torch.Tensor,
    selected_ranks: torch.Tensor,
    ep_size: int,
    ep_rank: int,
    schedule: bool,
) -> None:
    num_experts = primary_ranks.numel()
    _assign_replicated_experts_kernel[(1,)](
        topk_ids,
        primary_ranks,
        secondary_ranks,
        primary_hot,
        hot_primary_map,
        cold_primary_map,
        cold_physical_map,
        hot_output_map,
        cold_output_map,
        selected_ranks,
        NUM_ROUTES=topk_ids.numel(),
        NUM_EXPERTS=num_experts,
        EP_SIZE=ep_size,
        EP_RANK=ep_rank,
        SCHEDULE=schedule,
        HBM_COST=_HBM_COST,
        GRACE_COST=_GRACE_COST,
        BLOCK_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_RANKS=triton.next_power_of_2(ep_size),
        num_warps=4 if topk_ids.numel() <= 32 else 8,
    )


def _assign_replicated_experts_fake(
    topk_ids: torch.Tensor,
    primary_ranks: torch.Tensor,
    secondary_ranks: torch.Tensor,
    primary_hot: torch.Tensor,
    hot_primary_map: torch.Tensor,
    cold_primary_map: torch.Tensor,
    cold_physical_map: torch.Tensor,
    hot_output_map: torch.Tensor,
    cold_output_map: torch.Tensor,
    selected_ranks: torch.Tensor,
    ep_size: int,
    ep_rank: int,
    schedule: bool,
) -> None:
    return


direct_register_custom_op(
    op_name="tiered_moe_assign_replicas",
    op_func=_assign_replicated_experts,
    mutates_args=[
        "hot_output_map",
        "cold_output_map",
        "selected_ranks",
    ],
    fake_impl=_assign_replicated_experts_fake,
)


def assign_replicated_experts(
    topk_ids: torch.Tensor,
    primary_ranks: torch.Tensor,
    secondary_ranks: torch.Tensor,
    primary_hot: torch.Tensor,
    hot_primary_map: torch.Tensor,
    cold_primary_map: torch.Tensor,
    cold_physical_map: torch.Tensor,
    hot_output_map: torch.Tensor,
    cold_output_map: torch.Tensor,
    selected_ranks: torch.Tensor,
    *,
    ep_size: int,
    ep_rank: int,
    schedule: bool,
) -> None:
    """Assign active experts and update this rank's tier maps in-place."""
    torch.ops.vllm.tiered_moe_assign_replicas(
        topk_ids,
        primary_ranks,
        secondary_ranks,
        primary_hot,
        hot_primary_map,
        cold_primary_map,
        cold_physical_map,
        hot_output_map,
        cold_output_map,
        selected_ranks,
        ep_size,
        ep_rank,
        schedule,
    )
