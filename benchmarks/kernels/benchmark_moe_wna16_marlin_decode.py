# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune `moe_wna16_marlin_gemm` for small-M spec-decode verify batches.

The Marlin MoE launch heuristic picks its threadblock count from problem shape.
At GLM-5.2 W4A16 decode shapes it selects 3 blocks/SM on every call regardless
of token count, which is exactly one full occupancy wave (shared-memory limited)
and leaves the kernel at roughly a third of HBM peak. `blocks_per_sm`,
`thread_n` and `thread_k` are all exposed on the op and default to -1 (auto),
so they can be swept without touching the kernel.

Two modes:

* `single` sweeps the launch configuration for one tier resident in HBM.
* `overlap` runs a hot (HBM) and a cold (pinned Grace, reached over C2C) tier on
  two streams, as the tiered MoE does, and reports the union span. Because both
  tiers request a full wave, they cannot co-reside; splitting the SM budget is
  the hypothesis under test.

Run on a Booster node — login-node C2C and clock behaviour do not transfer.
"""

import argparse
import json

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.scalar_type import scalar_types

# GLM-5.2 geometry
HIDDEN = 6144
INTERMEDIATE = 2048
GROUP = 128
TOP_K = 8
BLOCK_M = 16
QUANT = scalar_types.uint4b8
# W4 packed weights + bf16 group scales, per expert
W13_BYTES = (2 * INTERMEDIATE * HIDDEN) // 2 + (2 * INTERMEDIATE) * (
    HIDDEN // GROUP
) * 2
W2_BYTES = (HIDDEN * INTERMEDIATE) // 2 + HIDDEN * (INTERMEDIATE // GROUP) * 2


def make_tier(num_experts, k, n, device, pinned, numa_node=0):
    """Marlin-format weights for one tier, in HBM or in pinned Grace memory."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )

    qs, ss = [], []
    for _ in range(num_experts):
        w = torch.randn((k, n), dtype=torch.bfloat16, device=device) / (k**0.5)
        _, q, s, _, _, _ = marlin_quantize(w, QUANT, GROUP, act_order=False)
        qs.append(q)
        ss.append(s)
    q = torch.stack(qs)
    s = marlin_moe_permute_scales(torch.stack(ss), k, n, GROUP)
    if pinned:
        # Final storage on the local Grace node, reached by the GPU over C2C
        # through the same pinned-UVA alias the tiered loader uses.
        from vllm.model_executor.offloader.grace import GraceAllocation

        allocs = []
        for t in (q, s):
            a = GraceAllocation.allocate_pinned(
                tuple(t.shape), t.dtype, device.index or 0, numa_node
            )
            a.copy_from(t.cpu())
            allocs.append(a)
        del q, s
        torch.accelerator.empty_cache()
        for a in allocs:
            placement = a.audit_numa(samples=64)
            print(
                f"    Grace alias {tuple(a.cuda_alias.shape)}: "
                f"{placement.local_fraction:.0%} of sampled pages on NUMA "
                f"node {a.numa_node}"
            )
        return allocs[0].cuda_alias, allocs[1].cuda_alias
    return q, s


def tier_map(all_ids, tier_ids, num_experts, device):
    """expert_map: global expert id -> local index within this tier, else -1."""
    m = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    m[torch.as_tensor(tier_ids, device=device)] = torch.arange(
        len(tier_ids), dtype=torch.int32, device=device
    )
    return m


def global_routing(m, hot_ids, cold_ids, num_experts, device, seed, cold_share):
    """One shared topk_ids over both tiers, as the production dispatch builds it.

    Assignments are split so the cold tier receives `cold_share` of the routing
    mass. Passing all mass to a handful of cold experts is what makes a tier
    re-stream its weights, so this split matters more than the expert counts.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = m * TOP_K
    n_cold = int(round(n * cold_share))
    pick = torch.cat(
        [
            torch.as_tensor(cold_ids)[
                torch.randint(0, len(cold_ids), (n_cold,), generator=g)
            ],
            torch.as_tensor(hot_ids)[
                torch.randint(0, len(hot_ids), (n - n_cold,), generator=g)
            ],
        ]
    )
    pick = pick[torch.randperm(n, generator=g)]
    return pick.view(m, TOP_K).to(device).to(torch.int32)


def align(topk_ids, bsm, num_experts, emap):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    tok, eids, npost = moe_align_block_size(
        topk_ids, bsm, num_experts, emap, ignore_invalid_experts=True
    )
    blocks = int((eids >= 0).sum())
    return tok, eids, npost, blocks


def time_call(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.accelerator.synchronize()
    return s.elapsed_time(e) * 1000 / iters  # us


def build_gemm(a, out, q, s, ws, tok, eids, npost, weights, m, n, k, cfg):
    bps, tn, tk = cfg

    def run():
        ops.moe_wna16_marlin_gemm(
            a,
            out,
            q,
            None,
            s,
            None,
            None,
            None,
            None,
            None,
            ws,
            tok,
            eids,
            npost,
            weights,
            moe_block_size=BLOCK_M,
            top_k=TOP_K,
            mul_topk_weights=False,
            b_q_type=QUANT,
            size_m=m,
            size_n=n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
            thread_k=tk,
            thread_n=tn,
            blocks_per_sm=bps,
        )

    return run


def run_tier(
    label,
    q,
    sc,
    ws,
    a,
    out,
    weights,
    topk_ids,
    emap,
    m,
    n,
    k,
    num_experts,
    wbytes,
    roof,
    bsms,
    cfgs,
):
    """Time one tier over block sizes and launch configs; report physical BW."""
    rows = []
    for bsm in bsms:
        tok, eids, npost, blocks = align(topk_ids, bsm, num_experts, emap)
        phys = blocks * wbytes
        for cfg in cfgs:
            try:
                us = time_call(
                    build_gemm(
                        a, out, q, sc, ws, tok, eids, npost, weights, m, n, k, cfg
                    )
                )
            except Exception as exc:
                print(
                    f"    bsm={bsm:2d} {str(cfg):14s} unsupported "
                    f"({type(exc).__name__})"
                )
                continue
            bw = phys / (us * 1e-6) / 1e9
            print(
                f"    bsm={bsm:2d} {str(cfg):14s} {us:8.2f} us  "
                f"{blocks:3d} blocks  {phys / 1e6:7.1f} MB physical  "
                f"{bw:7.1f} GB/s  {100 * bw * 1e9 / roof:5.1f}% roof"
            )
            rows.append(
                dict(
                    tier=label,
                    bsm=bsm,
                    cfg=list(cfg),
                    us=us,
                    blocks=blocks,
                    phys_bytes=phys,
                    gbs=bw,
                )
            )
    return rows


def single(args, dev):
    """Both tiers resident in HBM: isolates the effect of routing share."""
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    print(f"SMs={sms}\n")
    hot_ids = list(range(args.hot_act))
    cold_ids = list(range(args.hot_act, args.hot_act + args.cold_act))
    rows = []
    for shard, k, n in (
        ("w13", HIDDEN, 2 * INTERMEDIATE),
        ("w2", INTERMEDIATE, HIDDEN),
    ):
        q, sc = make_tier(args.experts, k, n, dev, pinned=False)
        ws = marlin_make_workspace_new(dev, 4)
        wbytes = W13_BYTES if shard == "w13" else W2_BYTES
        for m in args.tokens:
            topk = global_routing(
                m, hot_ids, cold_ids, args.experts, dev, args.seed, args.cold_share
            )
            a = torch.randn((m, k), dtype=torch.bfloat16, device=dev)
            out = torch.empty((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
            w = torch.ones((m, TOP_K), dtype=torch.float32, device=dev)
            for label, ids in (("hot", hot_ids), ("cold", cold_ids)):
                emap = tier_map(None, ids, args.experts, dev)
                print(
                    f"--- {shard} {label}: M={m}, {len(ids)} experts, "
                    f"cold_share={args.cold_share}"
                )
                rows += run_tier(
                    f"{shard}-{label}",
                    q,
                    sc,
                    ws,
                    a,
                    out,
                    w,
                    topk,
                    emap,
                    m,
                    n,
                    k,
                    args.experts,
                    wbytes,
                    3.5e12,
                    args.block_sizes,
                    args.configs,
                )
        del q, sc
        torch.accelerator.empty_cache()
    return rows


def overlap(args, dev):
    """Hot tier in HBM + cold tier in pinned Grace, sharing one routing."""
    k, n = HIDDEN, 2 * INTERMEDIATE
    hot_ids = list(range(args.hot_act))
    cold_ids = list(range(args.hot_act, args.hot_act + args.cold_act))
    hq, hs = make_tier(args.experts, k, n, dev, pinned=False)
    cq, cs = make_tier(args.experts, k, n, dev, pinned=True, numa_node=args.numa_node)
    ws_h = marlin_make_workspace_new(dev, 4)
    ws_c = marlin_make_workspace_new(dev, 4)
    aux = torch.cuda.Stream()
    rows = []
    for m in args.tokens:
        topk = global_routing(
            m, hot_ids, cold_ids, args.experts, dev, args.seed, args.cold_share
        )
        a = torch.randn((m, k), dtype=torch.bfloat16, device=dev)
        oh = torch.empty((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
        oc = torch.empty((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
        w = torch.ones((m, TOP_K), dtype=torch.float32, device=dev)
        mh = tier_map(None, hot_ids, args.experts, dev)
        mc = tier_map(None, cold_ids, args.experts, dev)
        print(
            f"--- M={m}: hot {len(hot_ids)} (HBM) / cold {len(cold_ids)} "
            f"(Grace), cold_share={args.cold_share}"
        )
        print("  HOT tier alone (HBM roof 3.5 TB/s)")
        rows += run_tier(
            "hot",
            hq,
            hs,
            ws_h,
            a,
            oh,
            w,
            topk,
            mh,
            m,
            n,
            k,
            args.experts,
            W13_BYTES,
            3.5e12,
            args.block_sizes,
            [(-1, -1, -1)],
        )
        print("  COLD tier alone (C2C roof 421 GB/s)")
        rows += run_tier(
            "cold",
            cq,
            cs,
            ws_c,
            a,
            oc,
            w,
            topk,
            mc,
            m,
            n,
            k,
            args.experts,
            W13_BYTES,
            421e9,
            args.block_sizes,
            [(-1, -1, -1)],
        )
        for bsm in args.block_sizes:
            th, eh, ph, bh = align(topk, bsm, args.experts, mh)
            tc, ec, pc, bc = align(topk, bsm, args.experts, mc)
            fh = build_gemm(a, oh, hq, hs, ws_h, th, eh, ph, w, m, n, k, (-1, -1, -1))
            fc = build_gemm(a, oc, cq, cs, ws_c, tc, ec, pc, w, m, n, k, (-1, -1, -1))

            def both(fh=fh, fc=fc):
                aux.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(aux):
                    fc()
                fh()
                torch.cuda.current_stream().wait_stream(aux)

            u_h, u_c, u_b = time_call(fh), time_call(fc), time_call(both)
            ideal = max(u_h, u_c)
            print(
                f"  bsm={bsm:2d} overlap: hot {u_h:7.1f} ({bh:3d} blk)  "
                f"cold {u_c:7.1f} ({bc:3d} blk)  union {u_b:7.1f}  "
                f"serial {u_h + u_c:7.1f}  ideal {ideal:7.1f}  "
                f"over ideal {100 * (u_b - ideal) / ideal:+5.1f}%"
            )
            rows.append(
                dict(
                    tier="overlap",
                    m=m,
                    bsm=bsm,
                    hot_us=u_h,
                    cold_us=u_c,
                    union_us=u_b,
                    hot_blocks=bh,
                    cold_blocks=bc,
                )
            )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("single", "overlap", "both"), default="both")
    p.add_argument("--tokens", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--hot-act", type=int, default=19, help="activated hot experts")
    p.add_argument("--cold-act", type=int, default=3, help="activated cold experts")
    p.add_argument(
        "--cold-share",
        type=float,
        default=0.13,
        help="fraction of routing mass landing on the cold tier",
    )
    p.add_argument("--block-sizes", type=int, nargs="+", default=[16])
    p.add_argument("--numa-node", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str)
    args = p.parse_args()
    args.configs = [(b, -1, -1) for b in (-1, 2, 3)]

    dev = torch.device("cuda:0")
    torch.accelerator.set_device_index(dev.index or 0)
    print(f"device: {torch.cuda.get_device_name(dev)}")
    print(f"per-expert bytes: w13={W13_BYTES:,} w2={W2_BYTES:,}\n")

    out = {}
    if args.mode in ("single", "both"):
        out["single"] = single(args, dev)
    if args.mode in ("overlap", "both"):
        print()
        out["overlap"] = overlap(args, dev)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
