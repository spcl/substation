#pragma once

struct __align__(16) myfloat4 {
    enum { ELEMS = 4 };

    float h[ELEMS];

    __forceinline__ __host__ __device__ 
    static myfloat4 fillall(float val) {
        myfloat4 res;
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            res.h[i] = val;
        }
        return res;
    }

    template <int stride>
    __forceinline__ __host__ __device__ 
    static myfloat4 load(float* ptr) {
        myfloat4 res;
        if (stride == 1) {
            res = *(myfloat4*)ptr;
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
    void store(float* ptr) {
        if (stride == 1) {
            *(myfloat4*)ptr = *this;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                ptr[i * stride] = h[i];
            }
        }
    }
    
    __device__ __forceinline__
    myfloat4 exp() {
        myfloat4 res;
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            res.h[i] = std::exp(h[i]);
        }
        return res;
    }
    
    __forceinline__ __device__ void operator*=(const myfloat4& v) {
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            h[i] *= v.h[i];
        }
    }
    
    __forceinline__ __device__ void operator+=(const myfloat4& v) {
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            h[i] += v.h[i];
        }
    }
    
    __forceinline__ __device__ void operator-=(const myfloat4& v) {
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            h[i] -= v.h[i];
        }
    }
    
    __forceinline__ __device__ void operator*=(const float& v) {
        *this *= fillall(v);
    }
    
    __forceinline__ __device__ void operator+=(const float& v) {
        *this += fillall(v);
    }
    
    __forceinline__ __device__ void operator-=(const float& v) {
        *this -= fillall(v);
    }
};

static_assert(sizeof(myfloat4) == myfloat4::ELEMS * sizeof(float));


__device__ __forceinline__
myfloat4 operator+(myfloat4 a, const myfloat4& b) {
    a += b;
    return a;
}

__device__ __forceinline__
myfloat4 operator-(myfloat4 a, const myfloat4& b) {
    a -= b;
    return a;
}

__device__ __forceinline__
myfloat4 operator*(myfloat4 a, const myfloat4& b) {
    a *= b;
    return a;
}


__device__ __forceinline__
myfloat4 operator-(myfloat4 a, float b) {
    a -= b;
    return a;
}

__device__ __forceinline__
myfloat4 operator+(myfloat4 a, float b) {
    a -= b;
    return a;
}

__device__ __forceinline__
myfloat4 operator*(myfloat4 a, float b) {
    a *= b;
    return a;
}

__device__ __forceinline__
float reciprocal(float val) {
    return 1.f/val;
}

__device__ __forceinline__
myfloat4 max(myfloat4 a, myfloat4 b) {
    myfloat4 res;
    #pragma unroll
    for (int i = 0; i < myfloat4::ELEMS; i++) {
        res.h[i] = a.h[i] > b.h[i] ? a.h[i] : b.h[i];
    }
    return res;
}


