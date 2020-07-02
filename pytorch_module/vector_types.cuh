#pragma once

#include "half8.cuh"
#include "myfloat4.cuh"
#include "mydouble2.cuh"

template <typename Real>
struct VectorTypeTrait;

template <> struct VectorTypeTrait<half> {
    using vector = half8;
};

template <> struct VectorTypeTrait<float> { 
    using vector = myfloat4;
};

template <> struct VectorTypeTrait<double> {
    using vector = mydouble2;
};

template <typename Real>
using VectorType = typename VectorTypeTrait<Real>::vector;
