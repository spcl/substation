#pragma once

#include "blocks.cuh"

template <typename Real,
    typename D1, typename D2, typename D3,
    typename DV, typename DT,
    int DATT_IN_S1, int DATT_IN_S2, int DATT_IN_S3,
    int DRESID_S1, int DRESID_S2, int DRESID_S3,
    int DIN_S1, int DIN_S2, int DIN_S3
>
__global__ void backwardEncoderInput(
    Real* DATT_IN, Real* DRESID,
    Real* DIN)
{
    using Vec = VectorType<Real>;

    static_assert(DV::value % Vec::ELEMS == 0, "");
    
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;

    if (metal::same<D1, DT>::value) i1 = i1 * blockDim.x + threadIdx.x;
    if (metal::same<D2, DT>::value) i2 = i2 * blockDim.x + threadIdx.x;
    if (metal::same<D3, DT>::value) i3 = i3 * blockDim.x + threadIdx.x;

    if (metal::same<D1, DV>::value) i1 *= Vec::ELEMS;
    if (metal::same<D2, DV>::value) i2 *= Vec::ELEMS;
    if (metal::same<D3, DV>::value) i3 *= Vec::ELEMS;

    constexpr int DATT_IN_SV = 
        metal::same<D1, DV>::value ? DATT_IN_S1 :
        metal::same<D2, DV>::value ? DATT_IN_S2 : DATT_IN_S3;

    constexpr int DRESID_SV = 
        metal::same<D1, DV>::value ? DRESID_S1 :
        metal::same<D2, DV>::value ? DRESID_S2 : DRESID_S3;

    constexpr int DIN_SV = 
        metal::same<D1, DV>::value ? DIN_S1 :
        metal::same<D2, DV>::value ? DIN_S2 : DIN_S3;
    
    Vec datt_in = Vec::load<DATT_IN_SV>(DATT_IN + DATT_IN_S1 * i1 + DATT_IN_S2 * i2 + DATT_IN_S3 * i3);
    Vec dresid = Vec::load<DRESID_SV>(DRESID + DRESID_S1 * i1 + DRESID_S2 * i2 + DRESID_S3 * i3);
    
    Vec din = datt_in + dresid;
    
    din.store<DIN_SV>(DIN + DIN_S1 * i1 + DIN_S2 * i2 + DIN_S3 * i3);
}

template <typename Real,
    typename d1, typename d2, typename d3,
    typename dv, typename dt,
    typename datt_in_layout,
    typename dresid_layout,
    typename din_layout>
struct BackwardEncoderInput {
    using datt_in_s1 = MetaHelpers::elemStride<datt_in_layout, d1>;
    using datt_in_s2 = MetaHelpers::elemStride<datt_in_layout, d2>;
    using datt_in_s3 = MetaHelpers::elemStride<datt_in_layout, d3>;
    
    using dresid_s1 = MetaHelpers::elemStride<dresid_layout, d1>;
    using dresid_s2 = MetaHelpers::elemStride<dresid_layout, d2>;
    using dresid_s3 = MetaHelpers::elemStride<dresid_layout, d3>;
    
    using din_s1 = MetaHelpers::elemStride<din_layout, d1>;
    using din_s2 = MetaHelpers::elemStride<din_layout, d2>;
    using din_s3 = MetaHelpers::elemStride<din_layout, d3>;
        
    static_assert(dv::value % VectorType<Real>::ELEMS == 0);
    
    static constexpr int threadable = metal::same<dt, dv>::value ? (dt::value / VectorType<Real>::ELEMS) : dt::value;

    static constexpr int threads = std::min(int(threadable), 128);

    static_assert(threadable % threads == 0);

    static void run(
        Real* DATT_IN, Real* DRESID,
        Real* DIN,
        cudaStream_t stream) 
    {
        dim3 blocks(d1::value, d2::value, d3::value);

        if (metal::same<d1, dt>::value) blocks.x /= threads;
        if (metal::same<d2, dt>::value) blocks.y /= threads;
        if (metal::same<d3, dt>::value) blocks.z /= threads;

        if (metal::same<d1, dv>::value) blocks.x /= VectorType<Real>::ELEMS;
        if (metal::same<d2, dv>::value) blocks.y /= VectorType<Real>::ELEMS;
        if (metal::same<d3, dv>::value) blocks.z /= VectorType<Real>::ELEMS;

        backwardEncoderInput<Real,
            d1, d2, d3,
            dv, dt,
            datt_in_s1::value, datt_in_s2::value, datt_in_s3::value,
            dresid_s1::value, dresid_s2::value, dresid_s3::value,
            din_s1::value, din_s2::value, din_s3::value
        >
            <<<blocks, threads, 0, stream>>>(
                DATT_IN, DRESID,
                DIN);
        CHECK(cudaPeekAtLastError());
    }
};