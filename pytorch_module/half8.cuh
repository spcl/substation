#pragma once

#include <cuda_fp16.h>

__forceinline__ __device__ half max(half a, half b) {
    return __hgt(a, b) ? a : b;
}

/* __forceinline__ __device__ half2 hmax(half2 a, half2 b) {
    half2 gt = __hgt2(a, b);
    half low = __low2half(gt) ? __low2half(a) : __low2half(b);
    half high = __high2half(gt) ? __high2half(a) : __high2half(b);
    return __halves2half2(low, high);
} */

struct __align__(16) half8 {
    enum { ELEMS = 8 };

    half h[ELEMS];
    
    __forceinline__ __host__ __device__ 
    static half8 fillall(half value) {
        half8 res;
        #if defined(__CUDA_ARCH__)
            half2 in = __half2half2(value);
            res.h2<0>() = in;
            res.h2<1>() = in;
            res.h2<2>() = in;
            res.h2<3>() = in;
        #else
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                res.h[i] = value;
            }
        #endif
        return res;
    }

    template <int stride>
    __forceinline__ __host__ __device__ 
    static half8 load(half* ptr) {
        half8 res;
        if (stride == 1) {
            res = *(half8*)ptr;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                res.h[i] = ptr[i * stride];
            }
        }
        return res;
    }
    
    template <int stride>
    __forceinline__ __host__ __device__ 
    void store(half* ptr) {
        if (stride == 1) {
            *(half8*)ptr = *this;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                ptr[i * stride] = h[i];
            }
        }
    }
    
/*     __device__ __forceinline__
    half2 max2() {
        half2 a = hmax(h2<0>(), h2<1>());
        half2 b = hmax(h2<2>(), h2<3>());
        return hmax(a, b); 
    } */
    
    __device__ __forceinline__
    void sum(float& res) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            res += __half2float(h[k]);
        }
    }
    
    template <int i>
    __device__ __forceinline__ half2& h2() {
        return ((half2*)h)[i];
    }
    
    template <int i>
    __device__ __forceinline__ const half2& h2() const {
        return ((const half2*)h)[i];
    }
    
    __device__ __forceinline__
    half8 exp() {
        half8 res;
        res.h2<0>() = h2exp(h2<0>());
        res.h2<1>() = h2exp(h2<1>());
        res.h2<2>() = h2exp(h2<2>());
        res.h2<3>() = h2exp(h2<3>());
        return res;
    }
    
    __forceinline__ __device__ void operator*=(const half8& v) {
        h2<0>() = __hmul2(h2<0>(), v.h2<0>());
        h2<1>() = __hmul2(h2<1>(), v.h2<1>());
        h2<2>() = __hmul2(h2<2>(), v.h2<2>());
        h2<3>() = __hmul2(h2<3>(), v.h2<3>());
    }
    
    __forceinline__ __device__ void operator+=(const half8& v) {
        h2<0>() = __hadd2(h2<0>(), v.h2<0>());
        h2<1>() = __hadd2(h2<1>(), v.h2<1>());
        h2<2>() = __hadd2(h2<2>(), v.h2<2>());
        h2<3>() = __hadd2(h2<3>(), v.h2<3>());
    }
    
    __forceinline__ __device__ void operator-=(const half8& v) {
        h2<0>() = __hsub2(h2<0>(), v.h2<0>());
        h2<1>() = __hsub2(h2<1>(), v.h2<1>());
        h2<2>() = __hsub2(h2<2>(), v.h2<2>());
        h2<3>() = __hsub2(h2<3>(), v.h2<3>());
    }
    
    __forceinline__ __device__ void operator*=(const half2& v) {
        h2<0>() = __hmul2(h2<0>(), v);
        h2<1>() = __hmul2(h2<1>(), v);
        h2<2>() = __hmul2(h2<2>(), v);
        h2<3>() = __hmul2(h2<3>(), v);
    }
    
    __forceinline__ __device__ void operator+=(const half2& v) {
        h2<0>() = __hadd2(h2<0>(), v);
        h2<1>() = __hadd2(h2<1>(), v);
        h2<2>() = __hadd2(h2<2>(), v);
        h2<3>() = __hadd2(h2<3>(), v);
    }
    
    __forceinline__ __device__ void operator-=(const half2& v) {
        h2<0>() = __hsub2(h2<0>(), v);
        h2<1>() = __hsub2(h2<1>(), v);
        h2<2>() = __hsub2(h2<2>(), v);
        h2<3>() = __hsub2(h2<3>(), v);
    }
    
    __forceinline__ __device__ void operator*=(const half& v) {
        *this *= __half2half2(v);
    }
    
    __forceinline__ __device__ void operator+=(const half& v) {
        *this += __half2half2(v);
    }
    
    __forceinline__ __device__ void operator-=(const half& v) {
        *this -= __half2half2(v);
    }
};

static_assert(sizeof(half8) == 8 * sizeof(half));


__device__ __forceinline__
half8 operator+(half8 a, const half8& b) {
    a += b;
    return a;
}

__device__ __forceinline__
half8 operator-(half8 a, const half8& b) {
    a -= b;
    return a;
}

__device__ __forceinline__
half8 operator*(half8 a, const half8& b) {
    a *= b;
    return a;
}


__device__ __forceinline__
half8 operator-(half8 a, half2 b) {
    a -= b;
    return a;
}

__device__ __forceinline__
half8 operator+(half8 a, half2 b) {
    a -= b;
    return a;
}

__device__ __forceinline__
half8 operator*(half8 a, half2 b) {
    a *= b;
    return a;
}


__device__ __forceinline__
half8 operator-(half8 a, half b) {
    a -= b;
    return a;
}

__device__ __forceinline__
half8 operator+(half8 a, half b) {
    a -= b;
    return a;
}

__device__ __forceinline__
half8 operator*(half8 a, half b) {
    a *= b;
    return a;
}

__device__ __forceinline__
half8 exp(half8 val) {
    val.h2<0>() = h2exp(val.h2<0>());
    val.h2<1>() = h2exp(val.h2<1>());
    val.h2<2>() = h2exp(val.h2<2>());
    val.h2<3>() = h2exp(val.h2<3>());
    return val;
}

__device__ __forceinline__
half8 reciprocal(half8 val) {
    val.h2<0>() = h2rcp(val.h2<0>());
    val.h2<1>() = h2rcp(val.h2<1>());
    val.h2<2>() = h2rcp(val.h2<2>());
    val.h2<3>() = h2rcp(val.h2<3>());
    return val;
}

__device__ __forceinline__
half reciprocal(half val) {
    return hrcp(val);
}

__device__ __forceinline__
half8 max(half8 a, half8 b) {
    half8 res;
    #pragma unroll
    for (int i = 0; i < half8::ELEMS; i++) {
        res.h[i] = __hgt(a.h[i], b.h[i]) ? a.h[i] : b.h[i];
    }
    return res;
}

__device__ __forceinline__
half exp(half val) {
    return hexp(val);
}