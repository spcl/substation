#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

#ifndef CUBLAS_ALGO
#define CUBLAS_ALGO CUBLAS_GEMM_DEFAULT
#endif

#ifndef C_TYPE
#define C_TYPE half
#define CUBLAS_C_TYPE CUDA_R_16F
#endif

//#if C_TYPE == float
//#define CUBLAS_C_TYPE CUDA_R_32F
//#endif

#define CUDA_CHECK(expr)                                                                   \
  do                                                                                       \
  {                                                                                        \
    auto err = (expr);                                                                     \
    if (err != 0)                                                                          \
    {                                                                                      \
      std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << "\n"; \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

int main(int argc, char **argv)
{
  cublasHandle_t handle;
  CUDA_CHECK(cublasCreate(&handle));
  //CUDA_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_DEVICE));
  int M, N, K, lda, ldb, ldc;
  char ta, tb;
  int stride_a, stride_b, stride_c, batch;

  if (argc != 5 && argc != 14)
  {
    printf("cublastest <TITLE> <M> <N> <K> [lda] [ldb] [ldc] [ta=N] [tb=N] [stride_a] [stride_b] [stride_c] [batch]\n");
    return 2;
  }

  char *title = argv[1];
  ++argv;
  M = atoi(argv[1]);
  N = atoi(argv[2]);
  K = atoi(argv[3]);
  if (argc == 4)
  {
    lda = M;
    ldb = K;
    ldc = M;
    ta = 'N';
    tb = 'N';
    stride_a = 0;
    stride_b = 0;
    stride_c = 0;
    batch = 1;
  }
  else
  {
    ta = argv[7][0];
    tb = argv[8][0];
    lda = atoi(argv[4]);
    ldb = atoi(argv[5]);
    ldc = atoi(argv[6]);
    // lda = ta == 'N' ? M : K;
    // ldb = tb == 'N' ? K : N;
    // ldc = M;
    stride_a = atoi(argv[9]);
    stride_b = atoi(argv[10]);
    stride_c = atoi(argv[11]);
    batch = atoi(argv[12]);
  }
  auto trans_a = ta == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto trans_b = tb == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;

  cudaEvent_t begin, end;

  // events
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  printf("\n%s,%s,%d,%d,%d,%c,%c,%d,%d,%d,%d,%d,%d,%d,", title, (batch <= 1 ? "GEMM" : "BMM"), M, N, K,
         ta, tb, lda, ldb, ldc, stride_a, stride_b, stride_c, batch);

  size_t asize = (ta == 'N') ? (lda * K) : (M * lda);
  size_t bsize = (tb == 'N') ? (ldb * N) : (K * ldb);
  size_t csize = M * N;

  // malloc
  C_TYPE *gA, *gB;
  C_TYPE *gC;
  CUDA_CHECK(cudaMalloc(&gA, (asize + stride_a * (batch - 1)) * sizeof(C_TYPE)));
  CUDA_CHECK(cudaMalloc(&gB, (bsize + stride_b * (batch - 1)) * sizeof(C_TYPE)));
  CUDA_CHECK(cudaMalloc(&gC, (csize + stride_c * (batch - 1)) * sizeof(C_TYPE)));

  // Type of alpha and beta must match compute type
  float one[1];
  float zero[1];
  *one = 1.0f;  //__float2half(1.0f);
  *zero = 0.0f; //__float2half(0.0f);
  std::vector<float> times;
  times.reserve(100);

  // warmup
  for (int rep = 1; rep < 1000; rep++)
  {
    if (rep > 3)
      CUDA_CHECK(cudaEventRecord(begin, nullptr));
    if (batch <= 1)
    {
      CUDA_CHECK(cublasGemmEx(handle, trans_a, trans_b,
                              M, N, K,
                              one,
                              gA, CUBLAS_C_TYPE, lda,
                              gB, CUBLAS_C_TYPE, ldb,
                              zero,
                              gC, CUBLAS_C_TYPE, ldc,
                              CUDA_R_32F, CUBLAS_ALGO));
    }
    else
    {
      CUDA_CHECK(cublasGemmStridedBatchedEx(handle, trans_a, trans_b,
                                            M, N, K,
                                            one,
                                            gA, CUBLAS_C_TYPE, lda, stride_a,
                                            gB, CUBLAS_C_TYPE, ldb, stride_b,
                                            zero,
                                            gC, CUBLAS_C_TYPE, ldc, stride_c,
                                            batch,
                                            CUDA_R_32F, CUBLAS_ALGO));
    }
    if (rep > 3)
    {
      CUDA_CHECK(cudaEventRecord(end, nullptr));
      CUDA_CHECK(cudaEventSynchronize(end));
      float msTime = -1.0f;
      CUDA_CHECK(cudaEventElapsedTime(&msTime, begin, end));
      times.push_back(msTime);
    }
  }

  cudaFree(gA);
  cudaFree(gB);
  cudaFree(gC);

  for (auto ms : times)
    printf("%f,", ms);
  printf("\n");

  CUDA_CHECK(cublasDestroy(handle));
}
