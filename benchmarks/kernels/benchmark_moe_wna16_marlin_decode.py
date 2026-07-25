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
import itertools
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


def routing(m, num_experts, activated, device, seed):
    """topk_ids hitting exactly `activated` distinct experts, block-aligned."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    g = torch.Generator(device="cpu").manual_seed(seed)
    live = torch.randperm(num_experts, generator=g)[:activated]
    ids = live[torch.randint(0, activated, (m, TOP_K), generator=g)]
    topk_ids = ids.to(device).to(torch.int32)
    tok, expert_ids, n_post = moe_align_block_size(topk_ids, BLOCK_M, num_experts)
    return topk_ids, tok, expert_ids, n_post


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


def single(args, dev):
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    print(f"SMs={sms}  configs are (blocks_per_sm, thread_n, thread_k); -1 = auto\n")
    rows = []
    for shard, k, n in (
        ("w13", HIDDEN, 2 * INTERMEDIATE),
        ("w2", INTERMEDIATE, HIDDEN),
    ):
        q, s = make_tier(args.experts, k, n, dev, pinned=False)
        ws = marlin_make_workspace_new(dev, 4)
        wbytes = W13_BYTES if shard == "w13" else W2_BYTES
        for m, act in itertools.product(args.tokens, args.activated):
            if act > args.experts:
                continue
            _, tok, eids, npost = routing(m, args.experts, act, dev, args.seed)
            a = torch.randn((m, k), dtype=torch.bfloat16, device=dev)
            out = torch.empty((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
            weights = torch.ones((m, TOP_K), dtype=torch.float32, device=dev)
            base = None
            print(
                f"--- {shard}: M={m} activated={act} "
                f"({act * wbytes / 1e6:.0f} MB of weights)"
            )
            for cfg in args.configs:
                try:
                    us = time_call(
                        build_gemm(
                            a, out, q, s, ws, tok, eids, npost, weights, m, n, k, cfg
                        )
                    )
                except Exception as exc:  # unsupported launch config
                    print(f"    {str(cfg):18s}  unsupported ({type(exc).__name__})")
                    continue
                bw = act * wbytes / (us * 1e-6) / 1e12
                base = base if base is not None else us
                print(
                    f"    {str(cfg):18s} {us:8.2f} us  {bw:6.2f} TB/s  "
                    f"{100 * bw / 3.5:5.1f}% HBM  {us / base:5.3f}x"
                )
                rows.append(
                    dict(shard=shard, m=m, activated=act, cfg=list(cfg), us=us, tbs=bw)
                )
        del q, s
        torch.accelerator.empty_cache()
    return rows


def overlap(args, dev):
    """Hot tier in HBM + cold tier in pinned Grace, on two streams."""
    k, n = HIDDEN, 2 * INTERMEDIATE
    hq, hs = make_tier(args.hot_experts, k, n, dev, pinned=False)
    cq, cs = make_tier(
        args.cold_experts, k, n, dev, pinned=True, numa_node=args.numa_node
    )
    ws_h = marlin_make_workspace_new(dev, 4)
    ws_c = marlin_make_workspace_new(dev, 4)
    aux = torch.cuda.Stream()
    rows = []
    for m in args.tokens:
        a = torch.randn((m, k), dtype=torch.bfloat16, device=dev)
        oh = torch.empty((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
        oc = torch.empty((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
        w = torch.ones((m, TOP_K), dtype=torch.float32, device=dev)
        _, th, eh, ph = routing(m, args.hot_experts, args.hot_act, dev, args.seed)
        _, tc, ec, pc = routing(m, args.cold_experts, args.cold_act, dev, args.seed + 1)
        print(
            f"--- overlap M={m}: hot {args.hot_act} experts (HBM), "
            f"cold {args.cold_act} experts (Grace)"
        )
        for hb, cb in itertools.product(args.hot_bps, args.cold_bps):
            fh = build_gemm(a, oh, hq, hs, ws_h, th, eh, ph, w, m, n, k, (hb, -1, -1))
            fc = build_gemm(a, oc, cq, cs, ws_c, tc, ec, pc, w, m, n, k, (cb, -1, -1))

            def both(fh=fh, fc=fc):
                aux.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(aux):
                    fc()
                fh()
                torch.cuda.current_stream().wait_stream(aux)

            try:
                u_h, u_c, u_b = time_call(fh), time_call(fc), time_call(both)
            except Exception as exc:
                print(f"    hot={hb} cold={cb}  unsupported ({type(exc).__name__})")
                continue
            ideal = max(u_h, u_c)
            print(
                f"    hot_bps={hb:2d} cold_bps={cb:2d}  hot {u_h:7.1f}  "
                f"cold {u_c:7.1f}  union {u_b:7.1f} us  "
                f"(serial {u_h + u_c:7.1f}, ideal {ideal:7.1f}, "
                f"over ideal {100 * (u_b - ideal) / ideal:+5.1f}%)"
            )
            rows.append(
                dict(
                    m=m, hot_bps=hb, cold_bps=cb, hot_us=u_h, cold_us=u_c, union_us=u_b
                )
            )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("single", "overlap", "both"), default="both")
    p.add_argument("--tokens", type=int, nargs="+", default=[8, 12, 16, 32])
    p.add_argument("--activated", type=int, nargs="+", default=[5, 16, 22])
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--hot-experts", type=int, default=40)
    p.add_argument("--cold-experts", type=int, default=24)
    p.add_argument("--hot-act", type=int, default=19)
    p.add_argument("--cold-act", type=int, default=3)
    p.add_argument("--hot-bps", type=int, nargs="+", default=[-1, 1, 2, 3])
    p.add_argument("--cold-bps", type=int, nargs="+", default=[-1, 1, 2, 3])
    p.add_argument("--numa-node", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str)
    args = p.parse_args()
    args.configs = [(b, -1, -1) for b in (-1, 1, 2, 3, 4)] + [
        (-1, 128, 128),
        (-1, 256, 64),
        (2, 128, 128),
    ]

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
