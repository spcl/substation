#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define REPS 1000

template<typename AccumT>
__global__ void kernel(half *a, int count) {
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccumT> acc_frag;
  //wmma::fill_fragment(acc_frag, 0.0f);

  for (int i = 0; i < count; ++i) {
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  a[blockIdx.x * blockDim.x + threadIdx.x] = acc_frag.x[5];
}

template <typename AccumT>
void invoke(half *A, int mults, int gridsz, int blocksz, int warps) {
  // Warmup runs
  for (int i = 0; i < 100; ++i)
    kernel<AccumT><<<gridsz, blocksz>>>(A, mults);

  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < REPS; ++i)
    kernel<AccumT><<<gridsz, blocksz>>>(A, mults);
  cudaDeviceSynchronize();
  auto t2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
  std::chrono::duration<double> fp_s = t2 - t1;
  double s = fp_s.count() / double(REPS);

  double flop = double(mults) * 2.0 * double(WMMA_M) * double(WMMA_N) * double(WMMA_K) * double(gridsz) * double(warps);

  printf("  Time: %lf ms\n", fp_ms.count() / double(REPS));
  printf("  Tflop/s: %lf (%llu ops)\n", (flop / s) * 1e-12, (unsigned long long)(flop)); 
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("USAGE: tcbench <gridsize> <warps per block> <#mults>\n");
    return 1;
  }

  int gridsz = atoi(argv[1]);
  int warps = atoi(argv[2]);
  int blocksz = warps * 32;
  int mults = atoi(argv[3]);
  printf("Test %d x %d\n", gridsz, blocksz);

  half *A;
  cudaMalloc(&A, sizeof(half) * gridsz * blocksz);

  printf("FP32 accumulator:\n");
  invoke<float>(A, mults, gridsz, blocksz, warps);

  printf("FP16 accumulator:\n");
  invoke<half>(A, mults, gridsz, blocksz, warps);

  cudaFree(A);
  return 0;
}
