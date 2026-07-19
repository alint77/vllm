# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashMLA sparse FP8 decode under DCP sharding matches the full-index run.

Exercises the real kernel with per-rank filtered indices and the base-e LSE
combine, guarding the two silent-corruption risks of the DCP port: a wrong
LSE base and mishandled filtered (-1) index rows.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashmla_sparse import mask_empty_dcp_lse
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.ops.common import correct_attn_out
from vllm.v1.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
    is_flashmla_sparse_supported,
)

BLOCK_SIZE = 64
HEAD_DIM = 576
HEAD_DIM_V = 512
NUM_HEADS = 64
TOPK = 2048

if not current_platform.is_cuda():
    pytest.skip("FlashMLA sparse requires CUDA", allow_module_level=True)
if not is_flashmla_sparse_supported()[0]:
    pytest.skip("FlashMLA sparse kernels unavailable", allow_module_level=True)


def test_dcp_variable_decode_expands_local_block_table():
    """Variable MTP lengths copy only the DCP-local block-table prefix."""
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.decode_seq_lens_buffer = torch.zeros(16, dtype=torch.int32)
    builder.expanded_block_table_buffer = torch.zeros((16, 3), dtype=torch.int32)
    builder.decode_lens_buffer = torch.zeros(16, dtype=torch.int32)
    builder.arange_buffer = torch.arange(16, dtype=torch.int32)

    block_table = torch.arange(16, dtype=torch.int32).view(2, 8)
    decode_lens = torch.tensor([3, 1], dtype=torch.int32)
    seq_lens, expanded, lens, batch_size, requires_padding = (
        builder._prepare_decode_tensors(
            seq_lens=torch.tensor([12, 9], dtype=torch.int32),
            block_table=block_table,
            decode_lens=decode_lens,
            decode_lens_cpu=decode_lens,
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            num_decodes=2,
            num_decode_tokens=4,
            use_native=False,
            next_n=4,
            max_decode_len=3,
        )
    )

    assert batch_size == 4
    assert not requires_padding
    torch.testing.assert_close(
        seq_lens, torch.tensor([10, 11, 12, 9], dtype=torch.int32)
    )
    torch.testing.assert_close(
        expanded,
        torch.tensor(
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [8, 9, 10]],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(lens, torch.ones(4, dtype=torch.int32))


def _pack_fp8_ds_mla_cache(kv_c: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """Pack latents into the 656-byte fp8_ds_mla token layout."""
    num_tokens = kv_c.shape[0]
    padded = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    cache = torch.zeros(padded, 656, dtype=torch.uint8, device=kv_c.device)

    groups = kv_c.float().view(num_tokens, 4, 128)
    scales = groups.abs().amax(dim=-1) / 448.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quant = (groups / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    cache[:num_tokens, :512] = quant.reshape(num_tokens, 512).view(torch.uint8)
    cache[:num_tokens, 512:528] = scales.contiguous().view(torch.uint8)
    cache[:num_tokens, 528:] = k_pe.to(torch.bfloat16).contiguous().view(torch.uint8)
    return cache.view(-1, BLOCK_SIZE, 656)


def _run_sparse_decode(
    q: torch.Tensor, cache: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One mixed-batch-style sparse decode call; returns (out, lse)."""
    scheduler_metadata, _ = get_mla_metadata()
    cache_lens = torch.full(
        (1,), cache.shape[0] * BLOCK_SIZE, dtype=torch.int32, device=q.device
    )
    dummy_block_table = torch.zeros((1, 1), dtype=torch.int32, device=q.device)
    out, lse = flash_mla_with_kvcache(
        q=q.unsqueeze(0),
        k_cache=cache.unsqueeze(-2),
        block_table=dummy_block_table,
        head_dim_v=HEAD_DIM_V,
        cache_seqlens=cache_lens,
        tile_scheduler_metadata=scheduler_metadata,
        is_fp8_kvcache=True,
        indices=indices.unsqueeze(0),
        softmax_scale=HEAD_DIM**-0.5,
    )
    return out.squeeze(0), lse.squeeze(0).transpose(0, 1)


@pytest.mark.parametrize("num_query_tokens", [1, 4])
def test_flashmla_sparse_dcp_shards_match_full_index(num_query_tokens: int):
    torch.manual_seed(7)
    device = torch.device("cuda")
    world = 4
    num_ctx_tokens = 4096

    kv_c = torch.randn(num_ctx_tokens, 512, dtype=torch.bfloat16, device=device)
    k_pe = 0.5 * torch.randn(num_ctx_tokens, 64, dtype=torch.bfloat16, device=device)
    q = torch.randn(
        num_query_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    topk_logical = torch.stack(
        [
            torch.randperm(num_ctx_tokens, device=device)[:TOPK]
            for _ in range(num_query_tokens)
        ]
    ).to(torch.int32)
    # Adversarial row: every selected token lives on DCP rank 0, so ranks
    # 1-3 see an all-invalid index row and must contribute exactly zero.
    topk_logical[0] = -1
    rank0_positions = torch.arange(
        0, num_ctx_tokens, 4, device=device, dtype=torch.int32
    )
    topk_logical[0, : rank0_positions.shape[0]] = rank0_positions

    full_cache = _pack_fp8_ds_mla_cache(kv_c, k_pe)
    ref_out, ref_lse = _run_sparse_decode(q, full_cache, topk_logical)

    req_id = torch.zeros(num_query_tokens, dtype=torch.int32, device=device)
    merged = torch.zeros_like(ref_out.float())
    lses = torch.empty(
        world, num_query_tokens, NUM_HEADS, dtype=torch.float32, device=device
    )
    rank_outs = []
    for rank in range(world):
        owned = torch.arange(rank, num_ctx_tokens, world, device=device)
        local_cache = _pack_fp8_ds_mla_cache(kv_c[owned], k_pe[owned])
        num_local_blocks = local_cache.shape[0]
        block_table = (
            torch.arange(num_local_blocks, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .contiguous()
        )
        local_indices, valid_counts = triton_filter_and_convert_dcp_index(
            req_id,
            block_table,
            topk_logical,
            dcp_size=world,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=1,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_TOPK_TOKENS=TOPK,
            return_valid_counts=True,
        )
        rank_out, rank_lse = _run_sparse_decode(q, local_cache, local_indices)
        rank_outs.append(rank_out)
        lses[rank] = mask_empty_dcp_lse(rank_lse, valid_counts)

    for rank in range(world):
        corrected, merged_lse = correct_attn_out(
            rank_outs[rank].float(), lses, rank, ctx=None, is_lse_base_on_e=True
        )
        merged += corrected

    torch.testing.assert_close(merged.to(ref_out.dtype), ref_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(merged_lse, ref_lse, atol=1e-2, rtol=1e-3)
