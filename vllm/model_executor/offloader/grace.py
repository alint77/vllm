# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final-destination allocations in Grace memory."""

import ctypes
import os
from collections import Counter
from dataclasses import dataclass
from enum import Enum

import torch

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


class GraceAllocationKind(str, Enum):
    """Supported Grace allocation backing types."""

    PINNED_UVA = "pinned_uva"


@dataclass(frozen=True)
class GracePagePlacement:
    """Sampled physical placement of a Grace allocation."""

    expected_node: int
    page_counts: tuple[tuple[int, int], ...]
    local_fraction: float


def _sample_numa_page_counts(
    tensor: torch.Tensor, samples: int
) -> tuple[tuple[int, int], ...]:
    if samples <= 0:
        raise ValueError("NUMA audit sample count must be positive")
    num_bytes = tensor.numel() * tensor.element_size()
    if num_bytes == 0:
        raise ValueError("Cannot audit an empty Grace allocation")

    page_size = os.sysconf("SC_PAGE_SIZE")
    step = max(page_size, num_bytes // samples)
    step -= step % page_size
    offsets = list(range(0, num_bytes, step))[:samples]
    pages = (ctypes.c_void_p * len(offsets))(
        *(tensor.data_ptr() + offset for offset in offsets)
    )
    status = (ctypes.c_int * len(offsets))()
    try:
        libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError as error:
        raise RuntimeError("NUMA page auditing requires libnuma") from error

    result = libnuma.move_pages(0, len(offsets), pages, None, status, 0)
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    failed = [node for node in status if node < 0]
    if failed:
        raise RuntimeError(f"NUMA page audit failed for {len(failed)} sampled pages")
    return tuple(sorted(Counter(status).items()))


@dataclass
class GraceAllocation:
    """CPU-owned Grace storage and its CUDA alias."""

    cpu_tensor: torch.Tensor
    cuda_alias: torch.Tensor
    num_bytes: int
    device_index: int
    numa_node: int
    kind: GraceAllocationKind
    page_placement: GracePagePlacement | None = None

    @classmethod
    def allocate_pinned(
        cls,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device_index: int,
        numa_node: int,
    ) -> "GraceAllocation":
        """Allocate final pinned-UVA storage on the bound worker.

        The calling worker must already be CPU- and memory-bound to
        ``numa_node``. The CPU tensor is allocated first, so checkpoint or
        converted data can be copied directly into its final backing.

        Args:
            shape: Tensor shape.
            dtype: Tensor element type.
            device_index: CUDA device that will access the allocation.
            numa_node: Expected Grace NUMA node for subsequent page auditing.

        Returns:
            An allocation containing the CPU owner and CUDA alias.
        """
        with torch.accelerator.device_index(device_index):
            cpu_tensor = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            cuda_alias = get_accelerator_view_from_cpu_tensor(cpu_tensor)
        return cls(
            cpu_tensor=cpu_tensor,
            cuda_alias=cuda_alias,
            num_bytes=cpu_tensor.numel() * cpu_tensor.element_size(),
            device_index=device_index,
            numa_node=numa_node,
            kind=GraceAllocationKind.PINNED_UVA,
        )

    def copy_from(self, source: torch.Tensor) -> None:
        """Copy one source tensor directly into the final Grace backing."""
        shape_mismatch = source.shape != self.cpu_tensor.shape
        dtype_mismatch = source.dtype != self.cpu_tensor.dtype
        if shape_mismatch or dtype_mismatch:
            raise ValueError("Source shape and dtype must match the Grace allocation")
        self.cpu_tensor.copy_(source)

    def audit_numa(
        self,
        samples: int = 256,
        min_local_fraction: float = 0.95,
        strict: bool = True,
    ) -> GracePagePlacement:
        """Sample physical pages and enforce local Grace residency.

        Args:
            samples: Maximum number of pages to inspect.
            min_local_fraction: Minimum fraction required on ``numa_node``.
            strict: Raise when the local fraction is below the minimum.

        Returns:
            The recorded NUMA placement snapshot.

        Raises:
            RuntimeError: If strict locality validation fails.
            ValueError: If the requested fraction is outside [0, 1].
        """
        if not 0 <= min_local_fraction <= 1:
            raise ValueError("Minimum local fraction must be between zero and one")
        page_counts = _sample_numa_page_counts(self.cpu_tensor, samples)
        sample_count = sum(count for _, count in page_counts)
        local_count = dict(page_counts).get(self.numa_node, 0)
        placement = GracePagePlacement(
            expected_node=self.numa_node,
            page_counts=page_counts,
            local_fraction=local_count / sample_count,
        )
        self.page_placement = placement
        if strict and placement.local_fraction < min_local_fraction:
            raise RuntimeError(
                "Grace allocation locality check failed: "
                f"{placement.local_fraction:.1%} is on NUMA node {self.numa_node}, "
                f"below the required {min_local_fraction:.1%}"
            )
        return placement
