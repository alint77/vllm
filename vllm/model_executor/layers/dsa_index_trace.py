# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded sidecar capture for sparse-attention index traces."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


class DSAIndexTraceCapturer:
    """Capture one sampled DSA position set per request and model step."""

    def __init__(
        self,
        output_dir: str | Path,
        layer_ids: Sequence[int],
        topk: int,
        max_num_reqs: int,
        interval: int,
        device: torch.device | str,
        *,
        writer: bool,
    ) -> None:
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("DSA trace layer IDs must be non-empty and unique")
        if topk <= 0 or max_num_reqs <= 0 or interval <= 0:
            raise ValueError("DSA trace dimensions and interval must be positive")

        self.output_dir = Path(output_dir)
        self.layer_ids = tuple(layer_ids)
        self.layer_offsets = {
            layer_id: offset for offset, layer_id in enumerate(self.layer_ids)
        }
        self.interval = interval
        self.writer = writer
        self.row_indices = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.device_buffer = torch.empty(
            (max_num_reqs, len(self.layer_ids), topk),
            dtype=torch.int32,
            device=device,
        )
        self._samples: list[tuple[str, int]] = []
        self._sequence = 0

        if self.writer:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if any(self.output_dir.iterdir()):
                raise ValueError("DSA index trace output directory is not empty")
            header = {
                "version": 1,
                "endian": "little",
                "layer_ids": list(self.layer_ids),
                "topk": topk,
                "sample_interval": interval,
                "dtype": "int32",
            }
            (self.output_dir / "header.json").write_text(
                json.dumps(header, indent=2, sort_keys=True) + "\n"
            )

    def prepare_step(
        self,
        request_ids: Sequence[str],
        num_scheduled_tokens: Sequence[int],
        num_computed_tokens: Sequence[int],
    ) -> None:
        """Select the last interval-aligned context row for each request."""
        if not (
            len(request_ids) == len(num_scheduled_tokens) == len(num_computed_tokens)
        ):
            raise ValueError("DSA trace step metadata lengths do not match")
        if len(request_ids) > self.row_indices.shape[0]:
            raise ValueError("DSA trace request count exceeds its bounded buffer")

        rows = []
        samples = []
        row_offset = 0
        for request_id, scheduled, computed in zip(
            request_ids, num_scheduled_tokens, num_computed_tokens
        ):
            scheduled = int(scheduled)
            computed = int(computed)
            if scheduled < 0 or computed < 0:
                raise ValueError("DSA trace token counts must be non-negative")
            last_position = computed + scheduled - 1
            sampled_position = last_position - last_position % self.interval
            if scheduled and sampled_position >= computed:
                rows.append(row_offset + sampled_position - computed)
                samples.append((request_id, sampled_position))
            row_offset += scheduled

        padded_rows = rows + [0] * (self.row_indices.shape[0] - len(rows))
        self.row_indices.copy_(
            torch.tensor(
                padded_rows,
                dtype=self.row_indices.dtype,
                device=self.row_indices.device,
            )
        )
        self._samples = samples

    def capture(self, layer_id: int, topk_indices: torch.Tensor) -> None:
        """Capture selected rows for one full-indexer layer."""
        layer_offset = self.layer_offsets.get(layer_id)
        if layer_offset is None:
            return
        torch.index_select(
            topk_indices,
            0,
            self.row_indices,
            out=self.device_buffer[:, layer_offset, :],
        )

    def drain(self) -> list[Path]:
        """Synchronously write this step's rank-zero samples as NPZ files."""
        if not self._samples:
            return []
        if not self.writer:
            self._samples = []
            return []

        values = self.device_buffer[: len(self._samples)].cpu().numpy()
        paths = []
        for (request_id, position), indices in zip(self._samples, values):
            path = self.output_dir / f"sample-{self._sequence:08d}.npz"
            np.savez_compressed(
                path,
                request_id=np.array(request_id),
                context_position=np.array(position, dtype=np.int64),
                indices=indices,
            )
            paths.append(path)
            self._sequence += 1
        self._samples = []
        return paths
