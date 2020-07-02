#pragma once

#include "all.cuh"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdlib>

#include "einsumparser.h"

template <typename Real>
struct CublasType;

template <>
struct CublasType<half> {
    static const cudaDataType_t value = CUDA_R_16F;
};

template <>
struct CublasType<float> {
    static const cudaDataType_t value = CUDA_R_32F;
};

template <>
struct CublasType<double> {
    static const cudaDataType_t value = CUDA_R_64F;
};

// possible order (C, row based) of dimensions in input array
// and computed result based on 
// 1. N/T - transpose flag in cublas
// 2. LR/RL - order in which A and B are passed into cublas 
//     k m, n k -> n m (LR, N, N)
//     m k, n k -> n m (LR, T, N)
//     k m, k n -> n m (LR, N, T)
//     m k, k n -> n m (LR, T, T)
//     m k, k n -> m n (RL, N, N)
//     m k, n k -> m n (RL, N, T)
//     k m, k n -> m n (RL, T, N)
//     k m, n k -> m n (RL, T, T)
//       |    |      |
//     use these 3 to detect correct option

// template <int M, int K>
// __global__ void testkernel(half* A) {
//     printf("array:\n");
//     for (int i = 0; i < M; i++)
//         for (int j = 0; j < K; j++)
//             printf("%f ", __half2float(A[i * K + j]));
//     printf("\n");
// }

template <typename Real,
    int B, int M, int N, int K,
    int SAM, int SAK, int SAB,
    int SBN, int SBK, int SBB,
    int SCM, int SCN, int SCB>
struct Gemm {
    static void run(Real* a, Real* b, Real* c, cublasHandle_t handle, cudaStream_t stream, float multiplier, cublasGemmAlgo_t algo) {
        static_assert(SAM == 1 || SAK == 1);
        static_assert(SBK == 1 || SBN == 1);
        static_assert(SCM == 1 || SCN == 1);
        cublasOperation_t opA = CUBLAS_OP_N;
        cublasOperation_t opB = CUBLAS_OP_N;
        int ldA = 0;
        int ldB = 0;
        int ldC = 0;
        int stA = SAB;
        int stB = SBB;
        int stC = SCB;
        bool swap = false;
        if (SAM == 1 && SBK == 1 && SCM == 1) {
            ldA = SAK; ldB = SBN; ldC = SCN; opA = CUBLAS_OP_N; opB = CUBLAS_OP_N; swap = false;
        } else if (SAM != 1 && SBK == 1 && SCM == 1) {
            ldA = SAM; ldB = SBN; ldC = SCN; opA = CUBLAS_OP_T; opB = CUBLAS_OP_N; swap = false;
        } else if (SAM == 1 && SBK != 1 && SCM == 1) {
            ldA = SAK; ldB = SBK; ldC = SCN; opA = CUBLAS_OP_N; opB = CUBLAS_OP_T; swap = false;
        } else if (SAM != 1 && SBK != 1 && SCM == 1) {
            ldA = SAM; ldB = SBK; ldC = SCN; opA = CUBLAS_OP_T; opB = CUBLAS_OP_T; swap = false;
        } else if (SAM != 1 && SBK != 1 && SCM != 1) {
            ldA = SAM; ldB = SBK; ldC = SCM; opA = CUBLAS_OP_N; opB = CUBLAS_OP_N; swap = true;
        } else if (SAM != 1 && SBK == 1 && SCM != 1) {
            ldA = SAM; ldB = SBN; ldC = SCM; opA = CUBLAS_OP_N; opB = CUBLAS_OP_T; swap = true;
        } else if (SAM == 1 && SBK != 1 && SCM != 1) {
            ldA = SAK; ldB = SBK; ldC = SCM; opA = CUBLAS_OP_T; opB = CUBLAS_OP_N; swap = true;
        } else if (SAM == 1 && SBK == 1 && SCM != 1) {
            ldA = SAK; ldB = SBN; ldC = SCM; opA = CUBLAS_OP_T; opB = CUBLAS_OP_T; swap = true;
        } else {
            abort();
        }
        
        int m = M;
        int n = N;
        if (swap) {
            std::swap(ldA, ldB);
            std::swap(stA, stB);
            std::swap(a, b);
            std::swap(opA, opB);
            std::swap(m, n);
        }
        
        // WARNING: these two can't have half precision;
        AccType<Real> alpha = multiplier;
        AccType<Real> beta = 0.0;
        CHECK(cublasSetStream(handle, stream));
        
        const cudaDataType_t TYPE = CublasType<Real>::value;
        const cudaDataType_t ACC_TYPE = CublasType<AccType<Real>>::value;
        
        if (B == 1) {
            CHECK(cublasGemmEx(
                handle, opA, opB,
                m, n, K,
                &alpha,
                a, TYPE, ldA,
                b, TYPE, ldB,
                &beta,
                c, TYPE, ldC,
                ACC_TYPE, algo));
        } else {
            CHECK(cublasGemmStridedBatchedEx(
                handle, opA, opB,
                m, n, K,
                &alpha,
                a, TYPE, ldA, stA,
                b, TYPE, ldB, stB,
                &beta,
                c, TYPE, ldC, stC,
                B,
                ACC_TYPE, algo));
        }
    }
};

template <typename Real, typename A, typename B, typename C>
struct Einsum {
    
    using parser = EinsumParser::EinsumParser<A, B, C>;
    
    static void run(Real* a, Real* b, Real* c, cublasHandle_t handle, cudaStream_t stream, float multiplier = 1.f, cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT) {
        Gemm<Real,
            parser::batch::value, parser::m::value, parser::n::value, parser::k::value,
            parser::sAM::value, parser::sAK::value, parser::sAB::value,
            parser::sBN::value, parser::sBK::value, parser::sBB::value,
            parser::sCM::value, parser::sCN::value, parser::sCB::value
        >::run(a, b, c, handle, stream, multiplier, algo);
    } 
};

