#include <ATen/DLConvertor.h>
#include <Python.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>
#include <vector>

namespace {

struct PageableTensorContext {
  at::Tensor owner;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  int device_index;
  DLManagedTensor managed{};
};

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation,
              " failed: ", cudaGetErrorString(status));
}

void delete_pageable_tensor(DLManagedTensor* managed) {
  auto* context = static_cast<PageableTensorContext*>(managed->manager_ctx);
  int original_device = -1;
  if (cudaGetDevice(&original_device) == cudaSuccess &&
      cudaSetDevice(context->device_index) == cudaSuccess) {
    cudaDeviceSynchronize();
    cudaSetDevice(original_device);
  }
  delete context;
}

at::Tensor get_cuda_pageable_view_from_cpu_tensor(const at::Tensor& cpu_tensor,
                                                  int64_t device_index) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");
  TORCH_CHECK(cpu_tensor.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(!cpu_tensor.is_pinned(),
              "Input tensor must use pageable CPU memory");

  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  TORCH_CHECK(device_index >= 0 && device_index < device_count,
              "CUDA device index out of range: ", device_index);

  int pageable_memory_access = 0;
  int uses_host_page_tables = 0;
  check_cuda(cudaDeviceGetAttribute(&pageable_memory_access,
                                    cudaDevAttrPageableMemoryAccess,
                                    static_cast<int>(device_index)),
             "cudaDeviceGetAttribute(pageable memory access)");
  check_cuda(
      cudaDeviceGetAttribute(&uses_host_page_tables,
                             cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                             static_cast<int>(device_index)),
      "cudaDeviceGetAttribute(host page tables)");
  TORCH_CHECK(
      pageable_memory_access && uses_host_page_tables, "CUDA device ",
      device_index,
      " does not support pageable memory access through host page tables");

  if (cpu_tensor.numel() == 0) {
    return at::empty(cpu_tensor.sizes(),
                     cpu_tensor.options().device(
                         at::Device(at::DeviceType::CUDA, device_index)));
  }

  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(cpu_tensor.data_ptr()) % 64 == 0,
              "Input tensor data must be 64-byte aligned");

#if CUDART_VERSION >= 13000
  const cudaMemLocation preferred_location{
      cudaMemLocationTypeHostNumaCurrent,
      0,
  };
#else
  const int preferred_location = cudaCpuDeviceId;
#endif
  check_cuda(
      cudaMemAdvise(cpu_tensor.data_ptr(),
                    cpu_tensor.numel() * cpu_tensor.element_size(),
                    cudaMemAdviseSetPreferredLocation, preferred_location),
      "cudaMemAdvise(preferred host location)");
  auto* context = new PageableTensorContext{
      cpu_tensor,
      cpu_tensor.sizes().vec(),
      cpu_tensor.strides().vec(),
      static_cast<int>(device_index),
  };
  context->managed.manager_ctx = context;
  context->managed.deleter = delete_pageable_tensor;
  context->managed.dl_tensor.data = cpu_tensor.data_ptr();
  context->managed.dl_tensor.device = {
      kDLCUDA,
      static_cast<int32_t>(device_index),
  };
  context->managed.dl_tensor.ndim = cpu_tensor.dim();
  context->managed.dl_tensor.dtype = at::getDLDataType(cpu_tensor);
  context->managed.dl_tensor.shape = context->shape.data();
  context->managed.dl_tensor.strides = context->strides.data();
  context->managed.dl_tensor.byte_offset = 0;
  return at::fromDLPack(&context->managed);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "get_cuda_pageable_view_from_cpu_tensor(Tensor cpu_tensor, int "
      "device_index) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C, CPU, ops) {
  ops.impl("get_cuda_pageable_view_from_cpu_tensor",
           &get_cuda_pageable_view_from_cpu_tensor);
}

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_pageable_grace_C", nullptr, -1, nullptr,
};

PyMODINIT_FUNC PyInit__pageable_grace_C() { return PyModule_Create(&module); }
