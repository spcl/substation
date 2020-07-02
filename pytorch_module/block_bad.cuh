
#pragma once

#include "blocks.cuh"

__forceinline__
__device__ half relu(half x) {
    return max(half(0.), x);
}

template <typename Real>
__forceinline__
__device__ VectorType<Real> relu(VectorType<Real> x) {
    return max(VectorType<Real>::fillall(0), x);
}





template <typename Real,
    typename D1, typename D2, typename DN,
    typename DV, typename DT,
    int IN_S1, int IN_S2, int IN_SN,
    int OUT_S1, int OUT_S2, int OUT_SN,
    int LIN_S1, int LIN_S2, int LIN_SN,
    int MASK_S1, int MASK_S2, int MASK_SN>
__global__ void biasActivationDropoutKernel(Real* IN, Real* OUT, Real* BIAS, Real* LIN, Real* MASK, float probability, GlobalRandomState grs) {
    using Vec = VectorType<Real>;
    
    constexpr int IN_SV = 
        metal::same<D1, DV>::value ? IN_S1 :
        metal::same<D2, DV>::value ? IN_S2 : IN_SN;
    constexpr int OUT_SV = 
        metal::same<D1, DV>::value ? OUT_S1 :
        metal::same<D2, DV>::value ? OUT_S2 : OUT_SN;
    constexpr int LIN_SV = 
        metal::same<D1, DV>::value ? LIN_S1 :
        metal::same<D2, DV>::value ? LIN_S2 : LIN_SN;
    constexpr int MASK_SV = 
        metal::same<D1, DV>::value ? MASK_S1 :
        metal::same<D2, DV>::value ? MASK_S2 : MASK_SN;

    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int iN = blockIdx.z;

    if (metal::same<D1, DT>::value) { i1 = i1 * blockDim.x + threadIdx.x; }
    if (metal::same<D2, DT>::value) { i2 = i2 * blockDim.x + threadIdx.x; }
    if (metal::same<DN, DT>::value) { iN = iN * blockDim.x + threadIdx.x; }

    if (metal::same<D1, DV>::value) { i1 *= Vec::ELEMS; }
    if (metal::same<D2, DV>::value) { i2 *= Vec::ELEMS; }
    if (metal::same<DN, DV>::value) { iN *= Vec::ELEMS; }
    
    Vec vec_in = Vec::load<IN_SV>(IN + i1 * IN_S1 + i2 * IN_S2 + IN_SN * iN);
    Vec vec_bias;
    if (metal::same<DN, DV>::value) {
        vec_bias = Vec::load<1>(BIAS + iN);
    } else {
        vec_bias = Vec::fillall(BIAS[iN]);
    }
    
    Vec biased = vec_in + vec_bias;
    biased.store<LIN_SV>(LIN + i1 * LIN_S1 + i2 * LIN_S2 + LIN_SN * iN);
    
    int rank = threadIdx.x
        + blockDim.x * blockIdx.x
        + blockDim.x * gridDim.x * blockIdx.y
        + blockDim.x * gridDim.x * gridDim.y * blockIdx.z;
    curandStatePhilox4_32_10_t state;
    curand_init(grs.seed, rank, grs.offset, &state);
    
    Vec activation = relu<Real>(biased);
    
    Vec res;
    Vec drop_mask;
    dropout<Real>(activation, state, probability, res, drop_mask);
    res.store<OUT_SV>(OUT + i1 * OUT_S1 + i2 * OUT_S2 + OUT_SN * iN);
    drop_mask.store<MASK_SV>(MASK + i1 * MASK_S1 + i2 * MASK_S2 + MASK_SN * iN);
}

template <typename Real,
    typename d1, typename d2, typename dn,
    typename dv, typename dt,
    typename in_layout, typename out_layout, typename lin_layout, typename mask_layout>
struct BiasActivationDropout {
    
    static_assert(metal::size<in_layout>::value == 3, "Only 3D arrays are supported");
    
    using in_s1 = MetaHelpers::elemStride<in_layout, d1>;
    using in_s2 = MetaHelpers::elemStride<in_layout, d2>;
    using in_sn = MetaHelpers::elemStride<in_layout, dn>;
    
    using out_s1 = MetaHelpers::elemStride<out_layout, d1>;
    using out_s2 = MetaHelpers::elemStride<out_layout, d2>;
    using out_sn = MetaHelpers::elemStride<out_layout, dn>;

    using lin_s1 = MetaHelpers::elemStride<lin_layout, d1>;
    using lin_s2 = MetaHelpers::elemStride<lin_layout, d2>;
    using lin_sn = MetaHelpers::elemStride<lin_layout, dn>;

    using mask_s1 = MetaHelpers::elemStride<mask_layout, d1>;
    using mask_s2 = MetaHelpers::elemStride<mask_layout, d2>;
    using mask_sn = MetaHelpers::elemStride<mask_layout, dn>;
    
    static constexpr int elems = VectorType<Real>::ELEMS;

    static_assert(dv::value % elems == 0);
    static_assert(metal::distinct<dv, dt>::value || dv::value % elems == 0);

    static constexpr int threads = std::min(int(metal::same<dv, dt>::value ? dt::value / elems : dt::value), 128);
    
    static_assert(dt::value % threads == 0);
    static_assert(metal::distinct<dv, dt>::value || dv::value % (elems * threads) == 0);
    

    static void run(Real* IN, Real* OUT, Real* BIAS, Real* LIN, Real* MASK, float probability, GlobalRandomState& grs, cudaStream_t stream) {

        int block1 = d1::value;
        int block2 = d2::value;
        int blockN = dn::value;

        if (metal::same<d1, dv>::value) { block1 /= elems; }
        if (metal::same<d2, dv>::value) { block2 /= elems; }
        if (metal::same<dn, dv>::value) { blockN /= elems; }

        if (metal::same<d1, dt>::value) { block1 /= threads; }
        if (metal::same<d2, dt>::value) { block2 /= threads; }
        if (metal::same<dn, dt>::value) { blockN /= threads; }

        dim3 blocks = dim3(block1, block2, blockN);
        biasActivationDropoutKernel<Real,
            d1, d2, dn,
            dv, dt,
            in_s1::value, in_s2::value, in_sn::value,
            out_s1::value, out_s2::value, out_sn::value,
            lin_s1::value, lin_s2::value, lin_sn::value,
            mask_s1::value, mask_s2::value, mask_sn::value>
            <<<blocks, threads, 0, stream>>>(IN, OUT, BIAS, LIN, MASK, probability, grs);
        CHECK(cudaPeekAtLastError());

        grs.offset += elems;
    }
};
