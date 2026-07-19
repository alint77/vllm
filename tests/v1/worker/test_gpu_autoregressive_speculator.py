# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_build_draft_attn_metadata_updates_dcp_lengths(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.arange = torch.arange(5, dtype=torch.int32)
    speculator.input_buffers = SimpleNamespace(
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        seq_lens=torch.tensor([17, 18, 19, 0], dtype=torch.int32),
        dcp_local_seq_lens=torch.zeros(4, dtype=torch.int32),
    )
    speculator.block_tables = SimpleNamespace(
        cp_size=4,
        cp_rank=2,
        cp_interleave=1,
        input_block_tables=[torch.zeros((4, 2), dtype=torch.int32)],
        slot_mappings=torch.zeros((1, 4), dtype=torch.int64),
    )
    speculator.attn_groups = []
    speculator.kv_cache_config = SimpleNamespace()
    speculator.draft_max_seq_len = 19
    prepare_dcp_lengths = Mock()
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(
        base_spec_module, "prepare_dcp_local_seq_lens", prepare_dcp_lengths
    )
    monkeypatch.setattr(base_spec_module, "build_attn_metadata", build_attn_metadata)

    speculator._build_draft_attn_metadata(3, 4, 4)

    prepare_dcp_lengths.assert_called_once_with(
        speculator.input_buffers.dcp_local_seq_lens,
        speculator.input_buffers.seq_lens,
        3,
        4,
        2,
        1,
    )
    dcp_lengths = build_attn_metadata.call_args.kwargs["dcp_local_seq_lens"]
    assert (
        dcp_lengths.data_ptr() == speculator.input_buffers.dcp_local_seq_lens.data_ptr()
    )
    assert dcp_lengths.shape == (4,)
