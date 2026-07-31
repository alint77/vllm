# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replica assignment and fused dual-tier alignment for tiered MoE decode.

Module contract: choose exactly one physical copy per active logical route and
build both Marlin tiers' routing metadata, under fixed memory and availability
constraints.

Guarded against, in order of severity: a route executed twice or not at all;
an assignment that is not load-optimal; metadata that disagrees with the
`moe_align_block_size` the tiers would otherwise have run; and behaviour
changing when the feature is off.
"""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.model_loader.tiered_moe_scheduler import (
    EP,
    PAIR_INDEX,
    PAIRS,
    allocate_fused_routing,
)
from vllm.platforms import current_platform

NUM_EXPERTS = 256
BLOCK_M = 16


def _placement(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Owners, HBM residency and replica holders for one synthetic layer."""
    rng = np.random.default_rng(seed)
    primary = np.repeat(np.arange(EP, dtype=np.int32), NUM_EXPERTS // EP)
    secondary = np.full(NUM_EXPERTS, -1, dtype=np.int32)
    replicated = rng.random(NUM_EXPERTS) < 0.4
    secondary[replicated] = (
        primary[replicated] + 1 + rng.integers(0, EP - 1, replicated.sum())
    ) % EP
    secondary[secondary == primary] = -1
    hot = (rng.random(NUM_EXPERTS) < 0.5).astype(np.int32)
    return primary, secondary, hot


def _tier_maps(
    primary: np.ndarray, hot: np.ndarray, secondary: np.ndarray, ep_rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """Global-to-local maps, matching `allocate_layer_expert_storage` order."""
    hot_ids = [e for e in range(NUM_EXPERTS) if primary[e] == ep_rank and hot[e]]
    cold_ids = [e for e in range(NUM_EXPERTS) if primary[e] == ep_rank and not hot[e]]
    replica_ids = [e for e in range(NUM_EXPERTS) if secondary[e] == ep_rank]
    hot_map = np.full(NUM_EXPERTS, -1, dtype=np.int32)
    cold_map = np.full(NUM_EXPERTS, -1, dtype=np.int32)
    hot_map[hot_ids] = np.arange(len(hot_ids), dtype=np.int32)
    cold_map[cold_ids + replica_ids] = np.arange(
        len(cold_ids) + len(replica_ids), dtype=np.int32
    )
    return hot_map, cold_map


def _run(
    routes: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    hot: np.ndarray,
    ep_rank: int,
    schedule: bool = True,
):
    device = torch.device("cuda")
    topk = torch.from_numpy(routes).to(device)
    hot_map, cold_map = _tier_maps(primary, hot, secondary, ep_rank)
    routing = allocate_fused_routing(routes.size, NUM_EXPERTS, BLOCK_M, BLOCK_M, device)
    torch.ops.vllm.tiered_moe_assign_align(
        topk,
        torch.from_numpy(primary).to(device),
        torch.from_numpy(secondary).to(device),
        torch.from_numpy(hot).to(device),
        torch.from_numpy(hot_map).to(device),
        torch.from_numpy(cold_map).to(device),
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
        ep_rank,
        BLOCK_M,
        BLOCK_M,
        schedule,
    )
    return routing, hot_map, cold_map


def _optimal_max_load(
    counts: np.ndarray, primary: np.ndarray, secondary: np.ndarray, hot: np.ndarray
) -> int:
    """Brute-force the minimum achievable maximum cold load for one layer."""
    active = counts > 0
    cold = active & (hot == 0)
    offsets = np.zeros(EP, dtype=np.int64)
    pair_counts = np.zeros(len(PAIRS), dtype=np.int64)
    for expert in np.flatnonzero(cold):
        rank, replica = int(primary[expert]), int(secondary[expert])
        if replica < 0:
            offsets[rank] += 1
        else:
            pair_counts[PAIR_INDEX[(min(rank, replica), max(rank, replica))]] += 1

    def feasible(limit: int) -> bool:
        caps = limit - offsets
        if np.any(caps < 0):
            return False
        for mask in range(1, 1 << EP):
            subset = [r for r in range(EP) if mask >> r & 1]
            inside = sum(
                pair_counts[PAIR_INDEX[pair]]
                for pair in PAIRS
                if pair[0] in subset and pair[1] in subset
            )
            if inside > sum(caps[r] for r in subset):
                return False
        return True

    limit = int(offsets.max())
    while not feasible(limit):
        limit += 1
    return limit


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [4, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_active_route_executes_exactly_once(num_tokens: int, seed: int):
    """The invariant a replicated placement can silently break."""
    primary, secondary, hot = _placement(seed)
    rng = np.random.default_rng(seed + 100)
    routes = rng.integers(0, NUM_EXPERTS, size=(num_tokens, 8)).astype(np.int32)
    counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)

    executed = np.zeros(NUM_EXPERTS, dtype=np.int64)
    for ep_rank in range(EP):
        routing, _, _ = _run(routes, primary, secondary, hot, ep_rank)
        executed += (routing.hot_out_map.cpu().numpy() >= 0).astype(np.int64)
        executed += (routing.cold_out_map.cpu().numpy() >= 0).astype(np.int64)

    np.testing.assert_array_equal(executed, (counts > 0).astype(np.int64))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_assignment_minimises_the_maximum_cold_load(seed: int):
    """A load-balancing kernel that is not optimal is not worth its cost."""
    primary, secondary, hot = _placement(seed)
    rng = np.random.default_rng(seed + 200)
    routes = rng.integers(0, NUM_EXPERTS, size=(16, 8)).astype(np.int32)
    counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)

    load = np.zeros(EP, dtype=np.int64)
    for ep_rank in range(EP):
        routing, _, _ = _run(routes, primary, secondary, hot, ep_rank)
        load[ep_rank] = int((routing.cold_out_map.cpu().numpy() >= 0).sum())

    assert int(load.max()) == _optimal_max_load(counts, primary, secondary, hot)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [4, 16])
def test_fused_alignment_agrees_with_moe_align_block_size(num_tokens: int):
    """Both tiers' metadata must describe the same work Marlin would have done.

    Block order differs by design - the kernel lays blocks out in ascending
    global expert id, the reference in tier-local order - and within-expert
    route order is not stable in either. What must agree is each expert's route
    set and the padding, which is all the GEMM observes.
    """
    primary, secondary, hot = _placement(7)
    rng = np.random.default_rng(7)
    routes = rng.integers(0, NUM_EXPERTS, size=(num_tokens, 8)).astype(np.int32)
    counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
    device = torch.device("cuda")
    topk = torch.from_numpy(routes).to(device)

    for ep_rank in range(EP):
        routing, _, _ = _run(routes, primary, secondary, hot, ep_rank)
        for tier_map, sorted_out, ids_out, post_out in (
            (
                routing.hot_out_map,
                routing.hot_sorted,
                routing.hot_expert_ids,
                routing.hot_num_post,
            ),
            (
                routing.cold_out_map,
                routing.cold_sorted,
                routing.cold_expert_ids,
                routing.cold_num_post,
            ),
        ):
            local = tier_map.cpu().numpy()
            ref_sorted, _, ref_post = moe_align_block_size(
                topk, BLOCK_M, NUM_EXPERTS, tier_map, ignore_invalid_experts=True
            )
            assert int(post_out.item()) == int(ref_post.item())

            mine = local >= 0
            blocks = np.where(mine, -(-counts // BLOCK_M), 0)
            starts = np.cumsum(blocks) - blocks
            ids = ids_out.cpu().numpy()
            got = sorted_out[: ref_sorted.numel()].cpu().numpy()
            seen: list[int] = []
            for expert in np.flatnonzero(blocks):
                lo = int(starts[expert]) * BLOCK_M
                span = int(blocks[expert]) * BLOCK_M
                block = got[lo : lo + span]
                assert ids[int(starts[expert])] == local[expert]
                seen.extend(int(r) for r in block if r < routes.size)
                # Padding uses the sentinel the GEMM skips.
                assert (block[block >= routes.size] == routes.size).all()
            expected = [
                index for index, expert in enumerate(routes.reshape(-1)) if mine[expert]
            ]
            assert sorted(seen) == sorted(expected)
            assert (got[int(blocks.sum()) * BLOCK_M :] == routes.size).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_disabled_scheduling_keeps_every_route_on_its_primary():
    """With scheduling off the kernel must reproduce primary-only ownership."""
    primary, secondary, hot = _placement(5)
    rng = np.random.default_rng(5)
    routes = rng.integers(0, NUM_EXPERTS, size=(16, 8)).astype(np.int32)
    counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
    active = counts > 0

    for ep_rank in range(EP):
        routing, hot_map, cold_map = _run(
            routes, primary, secondary, hot, ep_rank, schedule=False
        )
        expected_hot = np.where(active & (hot != 0) & (primary == ep_rank), hot_map, -1)
        expected_cold = np.where(
            active & (hot == 0) & (primary == ep_rank), cold_map, -1
        )
        np.testing.assert_array_equal(routing.hot_out_map.cpu().numpy(), expected_hot)
        np.testing.assert_array_equal(routing.cold_out_map.cpu().numpy(), expected_cold)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_capture_and_replay_tracks_changing_routes():
    """The served path captures once and replays with new routes every step."""
    primary, secondary, hot = _placement(9)
    device = torch.device("cuda")
    rng = np.random.default_rng(9)
    topk = torch.zeros((16, 8), dtype=torch.int32, device=device)
    hot_map, cold_map = _tier_maps(primary, hot, secondary, 0)
    routing = allocate_fused_routing(
        topk.numel(), NUM_EXPERTS, BLOCK_M, BLOCK_M, device
    )
    args = (
        torch.from_numpy(primary).to(device),
        torch.from_numpy(secondary).to(device),
        torch.from_numpy(hot).to(device),
        torch.from_numpy(hot_map).to(device),
        torch.from_numpy(cold_map).to(device),
    )

    def launch() -> None:
        torch.ops.vllm.tiered_moe_assign_align(
            topk,
            *args,
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
            0,
            BLOCK_M,
            BLOCK_M,
            True,
        )

    launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for _ in range(8):
        routes = rng.integers(0, NUM_EXPERTS, size=(16, 8)).astype(np.int32)
        topk.copy_(torch.from_numpy(routes).to(device))
        graph.replay()
        torch.accelerator.synchronize()
        replayed = routing.selected_rank.cpu().numpy().copy()

        launch()
        torch.accelerator.synchronize()
        np.testing.assert_array_equal(replayed, routing.selected_rank.cpu().numpy())
