#pragma once

#include "blocks.cuh"

template <typename Real,
    bool ENABLE_BIAS,
    int DN, int DV, int DR,
    int OUT_SN, int OUT_SV, int OUT_SR,
    int IN_SN, int IN_SV, int IN_SR,
    int RESID_SN, int RESID_SV, int RESID_SR,
    int NORMED_SN, int NORMED_SV, int NORMED_SR,
    int DIFF_SN, int DIFF_SV, int DIFF_SR,
    int ISTD_SN, int ISTD_SV,
    int LIN_SN, int LIN_SV, int LIN_SR,
    int MASK_SN, int MASK_SV, int MASK_SR
>
__global__ void biasDropoutResidualLinearNormKernelSeparateVecReduce(
    Real* OUT, Real* IN, Real* RESID, Real* NORMED,
    Real* SCALE, Real* BIAS, Real* LINEAR_B, Real* LIN, Real* DIFF, Real* ISTD, Real* MASK, float probability, GlobalRandomState grs)
{
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;
    constexpr const int VEC_SIZE = Vec::ELEMS;
    static_assert(DV % VEC_SIZE == 0);
    
    int iN = blockIdx.x;
    int iV = blockIdx.y * VEC_SIZE;
    int thread = threadIdx.x;

    Real* in = IN + iN * IN_SN + iV * IN_SV;
    Real* resid = RESID + iN * RESID_SN + iV * RESID_SV;
    Real* out = OUT + iN * OUT_SN + iV * OUT_SV;
    Real* scale = SCALE;
    Real* bias = BIAS;
    Real* linear = LINEAR_B;
    Real* linear_out = LIN + iN * LIN_SN + iV * LIN_SV;
    Real* out_normed = NORMED + iN * NORMED_SN + iV * NORMED_SV;
    Real* diff = DIFF + iN * DIFF_SN + iV * DIFF_SV;
    Real* istd_out = ISTD + iN * ISTD_SN + iV * ISTD_SV;
    Real* mask = MASK + iN * MASK_SN + iV * MASK_SV;
    
    int rank = blockIdx.y * DN * 32 + blockIdx.x * 32 + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(grs.seed, rank, grs.offset, &state);
    
    Acc sum[VEC_SIZE] = {0};
    Acc squaredSum[VEC_SIZE] = {0};
    
    #pragma unroll
    for (int iR = thread; iR < DR; iR += 32) {
        Vec biased = Vec::load<IN_SV>(in + iR * IN_SR);
        if (ENABLE_BIAS) {
            Vec lin = Vec::fillall(linear[iR]);
            biased = biased + lin;
            biased.store<LIN_SV>(linear_out + iR * LIN_SR);
        }
        Vec drp;
        Vec drp_mask;
        dropout<Real>(biased, state, probability, drp, drp_mask);
        drp_mask.store<MASK_SV>(mask + iR * MASK_SR);
        Vec res = Vec::load<RESID_SV>(resid + iR * RESID_SR);
        Vec residual = drp + res;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            sum[i] += Acc(residual.h[i]);
            squaredSum[i] += Acc(residual.h[i] * residual.h[i]);
        }

        residual.store<OUT_SV>(out + iR * OUT_SR);
    }

    Vec mean;
    Vec istd;

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        Acc finalSum = WarpReduce::sum(sum[i]);
        Acc finalSquaredSum = WarpReduce::sum(squaredSum[i]);

        Acc invSize = 1. / DR;
        Acc tmpMean = finalSum * invSize;
        Acc squaredStd = (finalSquaredSum - finalSum * finalSum * invSize) * invSize;
        squaredStd += 1e-5;
        Acc tmpIstd = rsqrtf(squaredStd);

        mean.h[i] = tmpMean;
        istd.h[i] = tmpIstd;
    }

    istd.store<ISTD_SV>(istd_out);
    
    #pragma unroll
    for (int iR = thread; iR < DR; iR += 32) {
        Vec diff_vec = Vec::load<OUT_SV>(out + iR * OUT_SR) - mean;
        diff_vec.store<DIFF_SV>(diff + iR * DIFF_SR);
        Vec normed = diff_vec * istd;
        normed.store<NORMED_SV>(out_normed + iR * NORMED_SR);
        Vec scaled = normed * Vec::fillall(scale[iR]);
        Vec biased = scaled + Vec::fillall(bias[iR]);
        biased.store<OUT_SV>(out + iR * OUT_SR);
    }
}

template <
    typename Real,
    bool ENABLE_BIAS,
    typename dr,
    typename dv,
    typename out_layout,
    typename in_layout,
    typename resid_layout,
    typename normed_layout,
    typename diff_layout,
    typename istd_layout,
    typename lin_layout,
    typename mask_layout>
struct BiasDropoutResidualLinearNorm {
    static_assert(metal::size<in_layout>::value == 3, "only 3D arrays are supported");
    
    using dr_idx = metal::find<in_layout, dr>;
    using in_layout_nv = metal::erase<in_layout, dr_idx>;
    using dv_idx = metal::find<in_layout_nv, dv>;
    using in_layout_n = metal::erase<in_layout_nv, dv_idx>;
    using dn = metal::at<in_layout_n, metal::number<0>>;

    using out_sn = MetaHelpers::elemStride<out_layout, dn>;
    using out_sv = MetaHelpers::elemStride<out_layout, dv>;
    using out_sr = MetaHelpers::elemStride<out_layout, dr>;

    using in_sn = MetaHelpers::elemStride<in_layout, dn>;
    using in_sv = MetaHelpers::elemStride<in_layout, dv>;
    using in_sr = MetaHelpers::elemStride<in_layout, dr>;
    
    using resid_sn = MetaHelpers::elemStride<resid_layout, dn>;
    using resid_sv = MetaHelpers::elemStride<resid_layout, dv>;
    using resid_sr = MetaHelpers::elemStride<resid_layout, dr>;
    
    using normed_sn = MetaHelpers::elemStride<normed_layout, dn>;
    using normed_sv = MetaHelpers::elemStride<normed_layout, dv>;
    using normed_sr = MetaHelpers::elemStride<normed_layout, dr>;

    using diff_sn = MetaHelpers::elemStride<diff_layout, dn>;
    using diff_sv = MetaHelpers::elemStride<diff_layout, dv>;
    using diff_sr = MetaHelpers::elemStride<diff_layout, dr>;

    using istd_sn = MetaHelpers::elemStride<istd_layout, dn>;
    using istd_sv = MetaHelpers::elemStride<istd_layout, dv>;

    using lin_sn = MetaHelpers::elemStride<lin_layout, dn>;
    using lin_sv = MetaHelpers::elemStride<lin_layout, dv>;
    using lin_sr = MetaHelpers::elemStride<lin_layout, dr>;

    using mask_sn = MetaHelpers::elemStride<mask_layout, dn>;
    using mask_sv = MetaHelpers::elemStride<mask_layout, dv>;
    using mask_sr = MetaHelpers::elemStride<mask_layout, dr>;
    
    static void run(Real* OUT, Real* IN, Real* RESID, Real* NORMED, Real* SCALE, Real* BIAS, Real* LINEAR_B, Real* LIN, Real* DIFF, Real* ISTD, Real* MASK, float probability, GlobalRandomState& grs, cudaStream_t stream) {
        
        biasDropoutResidualLinearNormKernelSeparateVecReduce
        <
            Real,
            ENABLE_BIAS,
            dn::value, dv::value, dr::value,
            out_sn::value, out_sv::value, out_sr::value,
            in_sn::value, in_sv::value, in_sr::value,
            resid_sn::value, resid_sv::value, resid_sr::value,
            normed_sn::value, normed_sv::value, normed_sr::value,
            diff_sn::value, diff_sv::value, diff_sr::value,
            istd_sn::value, istd_sv::value,
            lin_sn::value, lin_sv::value, lin_sr::value,
            mask_sn::value, mask_sv::value, mask_sr::value
        >
            <<<dim3(dn::value, dv::value / VectorType<Real>::ELEMS), 32, 0, stream>>>(OUT, IN, RESID, NORMED, SCALE, BIAS, LINEAR_B, LIN, DIFF, ISTD, MASK, probability, grs);
        CHECK(cudaPeekAtLastError());

        int rand_per_thread_factor = 32;
        grs.offset += dr::value * VectorType<Real>::ELEMS / rand_per_thread_factor + 1;
    }
};


template <typename Real,
    bool ENABLE_BIAS,
    int D1, int D2, int DR,
    int OUT_S1, int OUT_S2, int OUT_SR,
    int IN_S1, int IN_S2, int IN_SR,
    int RESID_S1, int RESID_S2, int RESID_SR,
    int NORMED_S1, int NORMED_S2, int NORMED_SR,
    int DIFF_S1, int DIFF_S2, int DIFF_SR,
    int ISTD_S1, int ISTD_S2,
    int LIN_S1, int LIN_S2, int LIN_SR,
    int MASK_S1, int MASK_S2, int MASK_SR
>
__global__ void biasDropoutResidualLinearNormKernelSameVecReduce(
    Real* OUT, Real* IN, Real* RESID, Real* NORMED,
    Real* SCALE, Real* BIAS, Real* LINEAR_B, Real* LIN, Real* DIFF, Real* ISTD, Real* MASK, float probability, GlobalRandomState grs)
{
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;
    constexpr const int VEC_SIZE = Vec::ELEMS;
    static_assert(DR % VEC_SIZE == 0);
    
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int thread = threadIdx.x;

    Real* in = IN + i1 * IN_S1 + i2 * IN_S2;
    Real* resid = RESID + i1 * RESID_S1 + i2 * RESID_S2;
    Real* out = OUT + i1 * OUT_S1 + i2 * OUT_S2;
    Real* scale = SCALE;
    Real* bias = BIAS;
    Real* linear = LINEAR_B;
    Real* linear_out = LIN + i1 * LIN_S1 + i2 * LIN_S2;
    Real* out_normed = NORMED + i1 * NORMED_S1 + i2 * NORMED_S2;
    Real* diff = DIFF + i1 * DIFF_S1 + i2 * DIFF_S2;
    Real* istd_out = ISTD + i1 * ISTD_S1 + i2 * ISTD_S2;
    Real* mask = MASK + i1 * MASK_S1 + i2 * MASK_S2;
    
    int rank = blockIdx.y * D1 * 32 + blockIdx.x * 32 + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(grs.seed, rank, grs.offset, &state);
    
    Acc sum = Acc();
    Acc squaredSum = Acc();
    
    #pragma unroll
    for (int iR = thread * VEC_SIZE; iR < DR; iR += 32 * VEC_SIZE) {
        Vec biased = Vec::load<IN_SR>(in + iR * IN_SR);
        if (ENABLE_BIAS) {
            Vec lin = Vec::load<1>(linear + iR);
            biased = biased + lin;
            biased.store<LIN_SR>(linear_out + iR * LIN_SR);
        }
        Vec drp;
        Vec drp_mask;
        dropout<Real>(biased, state, probability, drp, drp_mask);
        drp_mask.store<MASK_SR>(mask + iR * MASK_SR);
        Vec res = Vec::load<RESID_SR>(resid + iR * RESID_SR);
        Vec residual = drp + res;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            sum += Acc(residual.h[i]);
            squaredSum += Acc(residual.h[i] * residual.h[i]);
        }

        residual.store<OUT_SR>(out + iR * OUT_SR);
    }
    
    Acc finalSum = WarpReduce::sum(sum);
    Acc finalSquaredSum = WarpReduce::sum(squaredSum);

    Acc invSize = 1. / DR;
    Acc mean = finalSum * invSize;
    Acc squaredStd = (finalSquaredSum - finalSum * finalSum * invSize) * invSize;
    squaredStd += 1e-5;
    Acc istd = rsqrtf(squaredStd);

    *istd_out = Real(istd);
    
    #pragma unroll
    for (int iR = thread * VEC_SIZE; iR < DR; iR += 32 * VEC_SIZE) {
        Vec diff_vec = Vec::load<OUT_SR>(out + iR * OUT_SR) - mean;
        diff_vec.store<DIFF_SR>(diff + iR * DIFF_SR);
        Vec normed = diff_vec * istd;
        normed.store<NORMED_SR>(out_normed + iR * NORMED_SR);
        Vec scaled = normed * Vec::load<1>(scale + iR);
        Vec biased = scaled + Vec::load<1>(bias + iR);
        biased.store<OUT_SR>(out + iR * OUT_SR);
    }
}


template <
    typename Real,
    bool ENABLE_BIAS,
    typename dr,
    typename out_layout,
    typename in_layout,
    typename resid_layout,
    typename normed_layout,
    typename diff_layout,
    typename istd_layout,
    typename lin_layout,
    typename mask_layout>
struct BiasDropoutResidualLinearNorm<
    Real, ENABLE_BIAS, dr, dr,
    out_layout, in_layout, resid_layout, normed_layout, diff_layout, istd_layout, lin_layout, mask_layout> 
{
    static_assert(metal::size<in_layout>::value == 3, "only 3D arrays are supported");
    
    using dr_idx = metal::find<in_layout, dr>;
    using in_layout_12 = metal::erase<in_layout, dr_idx>;

    using d1 = metal::at<in_layout_12, metal::number<0>>;
    using d2 = metal::at<in_layout_12, metal::number<1>>;

    using out_s1 = MetaHelpers::elemStride<out_layout, d1>;
    using out_s2 = MetaHelpers::elemStride<out_layout, d2>;
    using out_sr = MetaHelpers::elemStride<out_layout, dr>;

    using in_s1 = MetaHelpers::elemStride<in_layout, d1>;
    using in_s2 = MetaHelpers::elemStride<in_layout, d2>;
    using in_sr = MetaHelpers::elemStride<in_layout, dr>;
    
    using resid_s1 = MetaHelpers::elemStride<resid_layout, d1>;
    using resid_s2 = MetaHelpers::elemStride<resid_layout, d2>;
    using resid_sr = MetaHelpers::elemStride<resid_layout, dr>;
    
    using normed_s1 = MetaHelpers::elemStride<normed_layout, d1>;
    using normed_s2 = MetaHelpers::elemStride<normed_layout, d2>;
    using normed_sr = MetaHelpers::elemStride<normed_layout, dr>;

    using diff_s1 = MetaHelpers::elemStride<diff_layout, d1>;
    using diff_s2 = MetaHelpers::elemStride<diff_layout, d2>;
    using diff_sr = MetaHelpers::elemStride<diff_layout, dr>;

    using istd_s1 = MetaHelpers::elemStride<istd_layout, d1>;
    using istd_s2 = MetaHelpers::elemStride<istd_layout, d2>;

    using lin_s1 = MetaHelpers::elemStride<lin_layout, d1>;
    using lin_s2 = MetaHelpers::elemStride<lin_layout, d2>;
    using lin_sr = MetaHelpers::elemStride<lin_layout, dr>;

    using mask_s1 = MetaHelpers::elemStride<mask_layout, d1>;
    using mask_s2 = MetaHelpers::elemStride<mask_layout, d2>;
    using mask_sr = MetaHelpers::elemStride<mask_layout, dr>;
    
    static void run(Real* OUT, Real* IN, Real* RESID, Real* NORMED, Real* SCALE, Real* BIAS, Real* LINEAR_B, Real* LIN, Real* DIFF, Real* ISTD, Real* MASK, float probability, GlobalRandomState& grs, cudaStream_t stream) {
        
        biasDropoutResidualLinearNormKernelSameVecReduce
        <
            Real,
            ENABLE_BIAS,
            d1::value, d2::value, dr::value,
            out_s1::value, out_s2::value, out_sr::value,
            in_s1::value, in_s2::value, in_sr::value,
            resid_s1::value, resid_s2::value, resid_sr::value,
            normed_s1::value, normed_s2::value, normed_sr::value,
            diff_s1::value, diff_s2::value, diff_sr::value,
            istd_s1::value, istd_s2::value,
            lin_s1::value, lin_s2::value, lin_sr::value,
            mask_s1::value, mask_s2::value, mask_sr::value
        >
            <<<dim3(d1::value, d2::value), 32, 0, stream>>>(OUT, IN, RESID, NORMED, SCALE, BIAS, LINEAR_B, LIN, DIFF, ISTD, MASK, probability, grs);
        CHECK(cudaPeekAtLastError());
        
        int rand_per_thread_factor = 32 * VectorType<Real>::ELEMS;
        grs.offset += dr::value * VectorType<Real>::ELEMS / rand_per_thread_factor + 1;
    }
};
