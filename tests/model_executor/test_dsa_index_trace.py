# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.dsa_index_trace import DSAIndexTraceCapturer


def test_dsa_index_trace_captures_interval_aligned_rows(tmp_path):
    capturer = DSAIndexTraceCapturer(
        output_dir=tmp_path,
        layer_ids=(0, 2),
        topk=3,
        max_num_reqs=2,
        interval=4,
        device="cpu",
        writer=True,
    )
    capturer.prepare_step(
        request_ids=("request-a", "request-b"),
        num_scheduled_tokens=(5, 3),
        num_computed_tokens=(0, 5),
    )
    indices = torch.arange(24, dtype=torch.int32).reshape(8, 3)
    capturer.capture(0, indices)
    capturer.capture(2, indices + 100)

    paths = capturer.drain()

    assert len(paths) == 1
    with np.load(paths[0]) as sample:
        assert str(sample["request_id"]) == "request-a"
        assert int(sample["context_position"]) == 4
        np.testing.assert_array_equal(sample["indices"][0], indices[4])
        np.testing.assert_array_equal(sample["indices"][1], indices[4] + 100)


def test_dsa_index_trace_rejects_mismatched_step_metadata(tmp_path):
    capturer = DSAIndexTraceCapturer(
        output_dir=tmp_path,
        layer_ids=(0,),
        topk=3,
        max_num_reqs=1,
        interval=4,
        device="cpu",
        writer=False,
    )

    with pytest.raises(ValueError, match="metadata lengths"):
        capturer.prepare_step(("request-a",), (1,), ())


def test_dsa_index_trace_rejects_nonempty_output_directory(tmp_path):
    (tmp_path / "existing.npz").touch()

    with pytest.raises(ValueError, match="not empty"):
        DSAIndexTraceCapturer(
            output_dir=tmp_path,
            layer_ids=(0,),
            topk=3,
            max_num_reqs=1,
            interval=4,
            device="cpu",
            writer=True,
        )
