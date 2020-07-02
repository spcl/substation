#pragma once

#include "blocks.cuh"

template <typename Real,
    typename DN1, typename DN2, typename DN3, typename DR,
    typename DV,
    int IN_S1, int IN_S2, int IN_S3, int IN_SR,
    int OUT_S1, int OUT_S2, int OUT_S3, int OUT_SR,
    int DROP_MASK_S1, int DROP_MASK_S2, int DROP_MASK_S3, int DROP_MASK_SR,
    int DROP_S1, int DROP_S2, int DROP_S3, int DROP_SR>
__global__ void softmaxKernel(Real* IN, Real* OUT, Real* DROP_MASK, Real* DROP, float probability, GlobalRandomState grs) {
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;

    static_assert(DV::value % Vec::ELEMS == 0, "");
    
    int i1 = blockIdx.x;
    if (metal::same<DN1, DV>::value) i1 *= Vec::ELEMS;
    int i2 = blockIdx.y;
    if (metal::same<DN2, DV>::value) i2 *= Vec::ELEMS;
    int i3 = blockIdx.z;
    if (metal::same<DN3, DV>::value) i3 *= Vec::ELEMS;

    int thread = threadIdx.x;

    constexpr int AccSize = metal::same<DR, DV>::value ? 1 : Vec::ELEMS;
    
    int r_start = thread;
    int r_step = 32;
    if (metal::same<DR, DV>::value) {
        r_start *= Vec::ELEMS;
        r_step *= Vec::ELEMS;
    }

    constexpr int OUT_SV = 
        metal::same<DN1, DV>::value ? OUT_S1 :
        metal::same<DN2, DV>::value ? OUT_S2 : 
        metal::same<DN3, DV>::value ? OUT_S3 : OUT_SR;
    constexpr int IN_SV = 
        metal::same<DN1, DV>::value ? IN_S1 :
        metal::same<DN2, DV>::value ? IN_S2 : 
        metal::same<DN3, DV>::value ? IN_S3 : IN_SR;
    constexpr int DROP_MASK_SV = 
        metal::same<DN1, DV>::value ? DROP_MASK_S1 :
        metal::same<DN2, DV>::value ? DROP_MASK_S2 : 
        metal::same<DN3, DV>::value ? DROP_MASK_S3 : DROP_MASK_SR;
    constexpr int DROP_SV = 
        metal::same<DN1, DV>::value ? DROP_S1 :
        metal::same<DN2, DV>::value ? DROP_S2 : 
        metal::same<DN3, DV>::value ? DROP_S3 : DROP_SR;
    
    Real local_max[AccSize];
    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        local_max[i] = Real(-DBL_MAX);
    }
    
    #pragma unroll
    for (int iR = r_start; iR < DR::value; iR += r_step) {
        Vec b = Vec::load<IN_SV>(IN + i1 * IN_S1 + i2 * IN_S2 + i3 * IN_S3 + iR * IN_SR);

        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            int idx = AccSize == 1 ? 0 : i;
            local_max[idx] = max(local_max[idx], b.h[i]);
        }
    }
    if (AccSize == 1) {
        local_max[0] = WarpReduce::max(local_max[0]);
    } else {
        Vec& local_max_vec = *(Vec*)local_max;
        local_max_vec = WarpReduce::max(local_max_vec);
    }
        
    Acc local_sum[AccSize] = {0.};
    #pragma unroll
    for (int iR = r_start; iR < DR::value; iR += r_step) {
        Vec b = Vec::load<IN_SV>(IN + i1 * IN_S1 + i2 * IN_S2 + i3 * IN_S3 + iR * IN_SR);

        Vec tmp;
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            int idx = AccSize == 1 ? 0: i;
            tmp.h[i] = exp(b.h[i] - local_max[idx]);
        }
        tmp.store<OUT_SV>(OUT + i1 * OUT_S1 + i2 * OUT_S2 + i3 * OUT_S3 + iR * OUT_SR);
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            int idx = AccSize == 1 ? 0 : i;
            local_sum[idx] += Acc(tmp.h[i]);
        }
    }

    Acc inv_sum[AccSize];
    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        inv_sum[i] = reciprocal(WarpReduce::sum(local_sum[i]));
    }

    int rank = threadIdx.x
        + blockDim.x * blockIdx.x
        + blockDim.x * gridDim.x * blockIdx.y
        + blockDim.x * gridDim.x * gridDim.y * blockIdx.z;
    curandStatePhilox4_32_10_t state;
    curand_init(grs.seed, rank, grs.offset, &state);
    
    #pragma unroll
    for (int iR = r_start; iR < DR::value; iR += r_step) {
        Vec tmp = Vec::load<OUT_SV>(OUT + i1 * OUT_S1 + i2 * OUT_S2 + i3 * OUT_S3 + iR * OUT_SR);
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            int idx = AccSize == 1 ? 0 : i;
            tmp.h[i] *= Real(inv_sum[idx]);
        }
        tmp.store<OUT_SV>(OUT + i1 * OUT_S1 + i2 * OUT_S2 + i3 * OUT_S3 + iR * OUT_SR);
        Vec drp;
        Vec drp_mask;
        dropout<Real>(tmp, state, probability, drp, drp_mask);
        drp.store<DROP_SV>(DROP + i1 * DROP_S1 + i2 * DROP_S2 + i3 * DROP_S3 + iR * DROP_SR);
        drp_mask.store<DROP_MASK_SV>(DROP_MASK + i1 * DROP_MASK_S1 + i2 * DROP_MASK_S2 + i3 * DROP_MASK_S3 + iR * DROP_MASK_SR);
    }
    
}


template <typename Real,
    typename dr,
    typename dv,
    typename in_layout,
    typename out_layout,
    typename drop_mask_layout,
    typename drop_layout>
struct Softmax {
    static_assert(metal::size<in_layout>::value == 4, "Only 4D arrays are supported");

    using dr_idx = metal::find<in_layout, dr>;
    using in_layout_123 = metal::erase<in_layout, dr_idx>;

    using d1 = metal::at<in_layout_123, metal::number<0>>;
    using d2 = metal::at<in_layout_123, metal::number<1>>;
    using d3 = metal::at<in_layout_123, metal::number<2>>;

    using in_s1 = MetaHelpers::elemStride<in_layout, d1>;
    using in_s2 = MetaHelpers::elemStride<in_layout, d2>;
    using in_s3 = MetaHelpers::elemStride<in_layout, d3>;
    using in_sr = MetaHelpers::elemStride<in_layout, dr>;
    
    using out_s1 = MetaHelpers::elemStride<out_layout, d1>;
    using out_s2 = MetaHelpers::elemStride<out_layout, d2>;
    using out_s3 = MetaHelpers::elemStride<out_layout, d3>;
    using out_sr = MetaHelpers::elemStride<out_layout, dr>;

    using drop_mask_s1 = MetaHelpers::elemStride<drop_mask_layout, d1>;
    using drop_mask_s2 = MetaHelpers::elemStride<drop_mask_layout, d2>;
    using drop_mask_s3 = MetaHelpers::elemStride<drop_mask_layout, d3>;
    using drop_mask_sr = MetaHelpers::elemStride<drop_mask_layout, dr>;

    using drop_s1 = MetaHelpers::elemStride<drop_layout, d1>;
    using drop_s2 = MetaHelpers::elemStride<drop_layout, d2>;
    using drop_s3 = MetaHelpers::elemStride<drop_layout, d3>;
    using drop_sr = MetaHelpers::elemStride<drop_layout, dr>;
    
    enum { VEC_SIZE = VectorType<Real>::ELEMS };

    static void run(Real* IN, Real* OUT, Real* DROP_MASK, Real* DROP, float probability, GlobalRandomState& grs, cudaStream_t stream) {
        dim3 blocks(d1::value, d2::value, d3::value);
        
        if (metal::same<d1, dv>::value) blocks.x /= VEC_SIZE;
        if (metal::same<d2, dv>::value) blocks.y /= VEC_SIZE;
        if (metal::same<d3, dv>::value) blocks.z /= VEC_SIZE;

        softmaxKernel<
            Real,
            d1, d2, d3, dr,
            dv,
            in_s1::value, in_s2::value, in_s3::value, in_sr::value,
            out_s1::value, out_s2::value, out_s3::value, out_sr::value,
            drop_mask_s1::value, drop_mask_s2::value, drop_mask_s3::value, drop_mask_sr::value,
            drop_s1::value, drop_s2::value, drop_s3::value, drop_sr::value
        >
            <<<blocks, 32, 0, stream>>>(IN, OUT, DROP_MASK, DROP, probability, grs);
        CHECK(cudaPeekAtLastError());

        int rand_per_thread_factor = 32 * (metal::same<dr, dv>::value ? VEC_SIZE : 1);
        grs.offset += dr::value * VEC_SIZE / rand_per_thread_factor + 1;
    }
};
