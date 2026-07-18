# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import pytest
import torch

from vllm.model_executor.offloader import grace as grace_module
from vllm.model_executor.offloader.grace import (
    GraceAllocation,
    GraceAllocationKind,
)
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    get_accelerator_view_from_cpu_tensor,
    get_pageable_accelerator_view_from_cpu_tensor,
    is_pageable_accelerator_view_supported,
)

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cpu_tensor[0, 0] = 1
    cpu_tensor[2, 3] = 2
    cpu_tensor[4, 5] = -1

    cuda_view.mul_(2)
    assert cuda_view[0, 0] == 2
    assert cuda_view[2, 3] == 4
    assert cuda_view[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_gpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cuda_view[0, 0] = 1
    cuda_view[2, 3] = 2
    cuda_view[4, 5] = -1
    cuda_view.mul_(2)

    assert cpu_tensor[0, 0] == 2
    assert cpu_tensor[2, 3] == 4
    assert cpu_tensor[4, 5] == -2


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_pageable_cpu_and_gpu_write(device):
    device_index = torch.device(device).index
    assert device_index is not None
    if not is_pageable_accelerator_view_supported(device_index):
        pytest.skip("CUDA pageable access through host page tables is unavailable")

    cpu_tensor = torch.arange(256, device="cpu", dtype=torch.int32)
    cuda_view = get_pageable_accelerator_view_from_cpu_tensor(cpu_tensor, device_index)

    assert cuda_view.device == torch.device(device)
    assert cuda_view.data_ptr() == cpu_tensor.data_ptr()
    cpu_tensor.add_(1)
    assert cuda_view[7].item() == 8

    cuda_view.mul_(2)
    torch.accelerator.synchronize(device_index)
    assert cpu_tensor[7].item() == 16

    del cpu_tensor
    gc.collect()
    cuda_view.add_(3)
    torch.accelerator.synchronize(device_index)
    assert cuda_view[7].item() == 19


@pytest.mark.parametrize("device", CUDA_DEVICES[:1])
def test_pageable_view_rejects_incompatible_storage(device):
    device_index = torch.device(device).index
    assert device_index is not None
    if not is_pageable_accelerator_view_supported(device_index):
        pytest.skip("CUDA pageable access through host page tables is unavailable")

    pinned = torch.empty(256, device="cpu", pin_memory=True)
    with pytest.raises(RuntimeError, match="pageable CPU memory"):
        get_pageable_accelerator_view_from_cpu_tensor(pinned, device_index)

    noncontiguous = torch.empty((16, 16), device="cpu").T
    with pytest.raises(RuntimeError, match="contiguous"):
        get_pageable_accelerator_view_from_cpu_tensor(noncontiguous, device_index)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES[:1])
def test_grace_allocation_owns_final_pinned_uva_storage(device, monkeypatch):
    device_index = torch.device(device).index
    assert device_index is not None
    allocation = GraceAllocation.allocate_pinned(
        (256,), torch.int32, device_index=device_index, numa_node=0
    )

    assert allocation.kind == GraceAllocationKind.PINNED_UVA
    assert allocation.cpu_tensor.is_pinned()
    assert allocation.cuda_alias.device == torch.device(device)
    assert allocation.num_bytes == 256 * 4
    assert allocation.device_index == device_index
    assert allocation.numa_node == 0

    source = torch.arange(256, dtype=torch.int32)
    allocation.copy_from(source)
    allocation.cuda_alias.add_(1)
    torch.accelerator.synchronize(device_index)
    assert allocation.cpu_tensor[7].item() == 8

    with pytest.raises(ValueError, match="shape and dtype"):
        allocation.copy_from(torch.empty(1))

    monkeypatch.setattr(
        grace_module,
        "_sample_numa_page_counts",
        lambda tensor, samples: ((0, 15), (1, 1)),
    )
    placement = allocation.audit_numa(samples=16, strict=False)
    assert placement.expected_node == 0
    assert placement.page_counts == ((0, 15), (1, 1))
    assert placement.local_fraction == 15 / 16
    assert allocation.page_placement is placement

    with pytest.raises(RuntimeError, match="locality check failed"):
        allocation.audit_numa(samples=16, min_local_fraction=0.95)
