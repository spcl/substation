#pragma once

struct __align__(16) mydouble2 {
    enum { ELEMS = 2 };

    double h[ELEMS];

    __forceinline__ __host__ __device__ 
    static mydouble2 fillall(double val) {
        mydouble2 res;
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            res.h[i] = val;
        }
        return res;
    }

    template <int stride>
    __forceinline__ __host__ __device__ 
    static mydouble2 load(double* ptr) {
        mydouble2 res;
        if (stride == 1) {
            res = *(mydouble2*)ptr;
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
    void store(double* ptr) {
        if (stride == 1) {
            *(mydouble2*)ptr = *this;
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {
                ptr[i * stride] = h[i];
            }
        }
    }
    
    __forceinline__ __device__ void operator*=(const mydouble2& v) {
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            h[i] *= v.h[i];
        }
    }
    
    __forceinline__ __device__ void operator+=(const mydouble2& v) {
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            h[i] += v.h[i];
        }
    }
    
    __forceinline__ __device__ void operator-=(const mydouble2& v) {
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            h[i] -= v.h[i];
        }
    }
    
    __forceinline__ __device__ void operator*=(const double& v) {
        *this *= fillall(v);
    }
    
    __forceinline__ __device__ void operator+=(const double& v) {
        *this += fillall(v);
    }
    
    __forceinline__ __device__ void operator-=(const double& v) {
        *this -= fillall(v);
    }
};

static_assert(sizeof(mydouble2) == mydouble2::ELEMS * sizeof(double));


__device__ __forceinline__
mydouble2 operator+(mydouble2 a, const mydouble2& b) {
    a += b;
    return a;
}

__device__ __forceinline__
mydouble2 operator-(mydouble2 a, const mydouble2& b) {
    a -= b;
    return a;
}

__device__ __forceinline__
mydouble2 operator*(mydouble2 a, const mydouble2& b) {
    a *= b;
    return a;
}


__device__ __forceinline__
mydouble2 operator-(mydouble2 a, double b) {
    a -= b;
    return a;
}

__device__ __forceinline__
mydouble2 operator+(mydouble2 a, double b) {
    a -= b;
    return a;
}

__device__ __forceinline__
mydouble2 operator*(mydouble2 a, double b) {
    a *= b;
    return a;
}

__device__ __forceinline__
mydouble2 max(mydouble2 a, const mydouble2& b) {
    #pragma unroll
    for (int i = 0; i < mydouble2::ELEMS; i++) {
       a.h[i] = a.h[i] > b.h[i] ? a.h[i] : b.h[i];
    }
    return a;
}

__device__ __forceinline__
mydouble2 exp(mydouble2 val) {
    #pragma unroll
    for (int i = 0; i < mydouble2::ELEMS; i++) {
        val.h[i] = exp(val.h[i]);
    }
    return val;
}

__device__ __forceinline__
mydouble2 reciprocal(mydouble2 val) {
    #pragma unroll
    for (int i = 0; i < mydouble2::ELEMS; i++) {
        val.h[i] = 1 / val.h[i];
    }
    return val;
}

__device__ __forceinline__
double reciprocal(double val) {
    return 1 / val;
}