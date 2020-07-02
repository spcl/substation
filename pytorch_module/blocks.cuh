#pragma once

#include <cuda.h>

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#include "metal.hpp"
#include "gemm.cuh"
#include "einsumparser.h"
#include "half8.cuh"
#include "warpreduce.cuh"

#include "all.cuh"

#include <cstdlib>

#include "vector_types.cuh"

#include <cfloat>


template <typename Real>
__device__ __forceinline__ 
void dropout(const VectorType<Real>& IN, curandStatePhilox4_32_10_t& state, float probability, VectorType<Real>& out, VectorType<Real>& mask) {
    if (VectorType<Real>::ELEMS % 4 == 0) {
        // efficient implementation
        #pragma unroll
        for (int k = 0; k < VectorType<Real>::ELEMS; k += 4) {
            float4 r4 = curand_uniform4(&state);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                mask.h[k + i] = ((float*)&r4)[i] < probability ? 0. : 1.;
            }
        }
    } else {
        // general implementation
        #pragma unroll
        for (int k = 0; k < VectorType<Real>::ELEMS; k++) {
            mask.h[k] = curand_uniform(&state) < probability ? 0. : 1.;
        }
    } 
    out = mask * IN;
}


struct GlobalRandomState {
    int seed = 0;
    int offset = 0;
};
