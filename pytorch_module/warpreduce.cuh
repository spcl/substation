#pragma once

#include <type_traits>

#include "mydouble2.cuh"
#include "myfloat4.cuh"
#include "half8.cuh"

#include <cuda.h>

template <typename T, typename BinaryOp, int size = sizeof(T)>
struct WarpReduceImpl {
    static __forceinline__ __device__ T run(T v) {
        #pragma unroll
        for (int i = 1; i < 32; i = i * 2) {            
            v = BinaryOp::compute(v, __shfl_xor_sync(0xffffffff, v, i));
        }
        return v;
    }
};

template <typename T, template <typename> typename BinaryOp>
struct WarpReduceImpl<T, BinaryOp<T>, 16> {
    static __forceinline__ __device__ T run(T v) {
        double& x = ((double*) &v)[0];
        double& y = ((double*) &v)[1];
        x = WarpReduceImpl<double, BinaryOp<double>>::run(x);
        y = WarpReduceImpl<double, BinaryOp<double>>::run(y);
        return v;
    }
};

struct WarpReduce {

    template <typename T, typename BinaryOp>
    static __forceinline__ __device__ T reduce(const T& v) {
        return WarpReduceImpl<T, BinaryOp>::run(v);
    }

    template <typename T>
    struct Sum {
        __forceinline__ __device__
        static T compute(const T& a, const T& b) {
            return a + b;
        }
    };

    template <typename T>
    struct Max {
        __forceinline__ __device__
        static T compute(const T& a, const T& b) {
            return ::max(a, b);
        }
    };

    template <typename T>
    static __forceinline__ __device__ T sum(const T& v) {
        return reduce<T, Sum<T>>(v);
    }

    template <typename T>
    static __forceinline__ __device__ T max(const T& v) {
        return reduce<T, Max<T>>(v);
    }
};
