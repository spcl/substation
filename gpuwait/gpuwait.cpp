#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Forward declaration.
void gpu_wait(double length, cudaStream_t stream);

void do_gpu_wait(double length) {
  gpu_wait(length, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wait", &do_gpu_wait, "GPU wait");
}
