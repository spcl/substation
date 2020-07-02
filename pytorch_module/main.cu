#include <iostream>


#include "all.cuh"
#include "half8.cuh"
#include "warpreduce.cuh"
#include "stridemapper.cuh"
#include "gemm.cuh"

#include "metal.hpp"
#include "blocks.cuh"

#define CUDA_CHECK(expr) do {\
    auto err = (expr);\
    if (err != 0) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << "\n"; \
        abort(); \
    }\
} while(0)

#define B 8
#define S 512
#define H 16
#define P 64
#define N (H * P)
#define U (4 * N)

struct dimB { enum { value = B }; };
struct dimK { enum { value = S }; };
struct dimJ { enum { value = S }; };
struct dimH { enum { value = H }; };
struct dimP { enum { value = P }; };
struct dimN { enum { value = N }; };
struct dimU { enum { value = U }; };
struct dimQKV { enum { value = 3 }; };

cublasHandle_t handle;

cudaStream_t stream1;

cudaEvent_t event1;

using lX = metal::list<dimN, dimB, dimJ>;
half* gX = nullptr;
using lWKQV = metal::list<dimQKV, dimP, dimH, dimN>;
half* gWKQV = nullptr;
using lKKQQVV = metal::list<dimQKV, dimP, dimH, dimB, dimJ>;
half* gKKQQVV = nullptr;
using lKK = metal::list<dimP, dimH, dimB, dimK>;
half* gKK = nullptr;
using lQQ = metal::list<dimP, dimH, dimB, dimJ>;
half* gQQ = nullptr;
using lVV = metal::list<dimP, dimH, dimB, dimK>;
half* gVV = nullptr;
using lBETA = metal::list<dimH, dimB, dimJ, dimK>;
half* gBETA = nullptr;
using lALPHA = metal::list<dimH, dimB, dimJ, dimK>;
half* gALPHA = nullptr;
using lGAMMA = metal::list<dimP, dimH, dimB, dimJ>;
half* gGAMMA = nullptr;
using lWO = metal::list<dimP, dimH, dimN>;
half* gWO = nullptr;
using lATT = metal::list<dimN, dimB, dimJ>;
half* gATT = nullptr;
using lATT1 = metal::list<dimB, dimJ, dimN>;
half* gATT1 = nullptr;
using lNORM1_SCALE = metal::list<dimN>;
half* gNORM1_SCALE = nullptr;
using lNORM1_BIAS = metal::list<dimN>;
half* gNORM1_BIAS = nullptr;
using lLINEAR1_B = metal::list<dimU>;
half* gLINEAR1_B = nullptr;
using lLINEAR1_W = metal::list<dimU, dimN>;
half* gLINEAR1_W = nullptr;
using lFF1 = metal::list<dimB, dimJ, dimU>;
half* gFF1 = nullptr;
using lLINEAR2_B = metal::list<dimN>;
half* gLINEAR2_B = nullptr;
using lLINEAR2_W = metal::list<dimN, dimU>;
half* gLINEAR2_W = nullptr;
using lFF = metal::list<dimB, dimJ, dimN>;
half* gFF = nullptr;
using lNORM2_SCALE = metal::list<dimN>;
half* gNORM2_SCALE = nullptr;
using lNORM2_BIAS = metal::list<dimN>;
half* gNORM2_BIAS = nullptr;
using lENC = metal::list<dimB, dimJ, dimN>;
half* gENC = nullptr;

half one[1];
half zero[1];

void attention_forward() {
    
    Einsum<half, lX, lWKQV, lKKQQVV>::run(gX, gWKQV, gKKQQVV, handle, stream1);

    Einsum<half, lKK, lQQ, lBETA>::run(gKK, gQQ, gBETA, handle, stream1);
    
    Softmax<lBETA, lALPHA, dimK>::run(gBETA, gALPHA, stream1);
    
    Einsum<half, lVV, lALPHA, lGAMMA>::run(gVV, gALPHA, gGAMMA, handle, stream1);
    
    Einsum<half, lWO, lGAMMA, lATT>::run(gWO, gGAMMA, gATT, handle, stream1);
}

int main() {
    CUDA_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    // streams
    CUDA_CHECK(cudaStreamCreate(&stream1));
    
    // events
    CUDA_CHECK(cudaEventCreate(&event1));
    
    // malloc
    
    CUDA_CHECK(cudaMalloc(&gX, N * B * S * sizeof(half)));
    CUDA_CHECK(cudaMemset(gX, 0, N * B * S * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gWKQV, 3 * P * H * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gWKQV, 0, 3 * P * H * N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gKKQQVV, 3 * P * H * B * S * sizeof(half)));
    CUDA_CHECK(cudaMemset(gKKQQVV, 0, 3 * P * H * B * S * sizeof(half)));
    
    gKK = gKKQQVV + 0 * P * H * B * S;
    gQQ = gKKQQVV + 1 * P * H * B * S;
    gVV = gKKQQVV + 2 * P * H * B * S;
    
    CUDA_CHECK(cudaMalloc(&gBETA, H * B * S * S * sizeof(half)));
    CUDA_CHECK(cudaMemset(gBETA, 0, H * B * S * S * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gALPHA, H * B * S * S * sizeof(half)));
    CUDA_CHECK(cudaMemset(gALPHA, 0, H * B * S * S * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gGAMMA, P * H * B * S * sizeof(half)));
    CUDA_CHECK(cudaMemset(gGAMMA, 0, P * H * B * S * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gWO, P * H * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gWO, 0, P * H * N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gATT, B * N * S * sizeof(half)));
    CUDA_CHECK(cudaMemset(gATT, 0, B * N * S * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gATT1, B * N * S * sizeof(half)));
    CUDA_CHECK(cudaMemset(gATT1, 0, B * N * S * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gNORM1_SCALE, N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gNORM1_SCALE, 0, N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gNORM1_BIAS, N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gNORM1_BIAS, 0, N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gLINEAR1_W, U * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gLINEAR1_W, 0, U * N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gLINEAR1_B, U * sizeof(half)));
    CUDA_CHECK(cudaMemset(gLINEAR1_B, 0, U * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gFF1, B * S * U * sizeof(half)));
    CUDA_CHECK(cudaMemset(gFF1, 0, B * S * U * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gFF, B * S * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gFF, 0, B * S * N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gLINEAR2_W, U * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gLINEAR2_W, 0, U * N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gLINEAR2_B, N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gLINEAR2_B, 0, N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gNORM2_SCALE, N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gNORM2_SCALE, 0, N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gNORM2_BIAS, N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gNORM2_BIAS, 0, N * sizeof(half)));
    
    CUDA_CHECK(cudaMalloc(&gENC, B * S * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(gENC, 0, B * S * N * sizeof(half)));
    
    *one = __float2half(1);
    *zero = __float2half(0);
    
    // computation
    
    for (int rep = 1; rep < 100; rep++) {
        attention_forward();
        
        BiasDropoutResidualLinearNorm<false, lATT1, lATT, lX, dimN>::run(
            gATT1, gATT, gX, gNORM1_SCALE, gNORM1_BIAS, nullptr, 0.5f, stream1);
        
        Einsum<half, lATT1, lLINEAR1_W, lFF1>::run(gATT1, gLINEAR1_W, gFF1, handle, stream1);
        
        BiasActivationDropout<lFF1, lFF1, dimU>::run(
            gFF1, gFF1, gLINEAR1_B, 0.5f, stream1);
        
        Einsum<half, lFF1, lLINEAR2_W, lFF>::run(gFF1, gLINEAR2_W, gFF, handle, stream1);
        
        BiasDropoutResidualLinearNorm<true, lENC, lFF, lATT1, dimN>::run(
            gENC, gFF, gATT1, gNORM2_SCALE, gNORM2_BIAS, gLINEAR2_B, 0.5f, stream1);
        
        CUDA_CHECK(cudaStreamSynchronize(stream1));
    }
    
    CUDA_CHECK(cublasDestroy(handle));
}
