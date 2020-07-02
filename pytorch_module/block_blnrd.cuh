#pragma once

#include "blocks.cuh"

template <typename Real,
    int DN, int DV, int DR,
    int DOUT_SN, int DOUT_SV, int DOUT_SR,
    int STD_SN, int STD_SV,
    int DIFF_SN, int DIFF_SV, int DIFF_SR,
    int DROP_MASK_SN, int DROP_MASK_SV, int DROP_MASK_SR,
    int D_LN_IN_SN, int D_LN_IN_SV, int D_LN_IN_SR,
    int D_DROP_IN_SN, int D_DROP_IN_SV, int D_DROP_IN_SR
>
__global__ void backwardLayerNormResidualDropoutKernelSeparateVecReduce(
    Real* DOUT, Real* STD, Real* DIFF, Real* DROP_MASK,
    Real* D_LN_IN, Real* D_DROP_IN)
{
    using Vec = VectorType<Real>;

    static_assert(DV % Vec::ELEMS == 0);
    
    int iN = blockIdx.x;
    int iV = blockIdx.y * Vec::ELEMS;
    int thread = threadIdx.x;
    
    AccType<Real> dout_sum[Vec::ELEMS] = {0};
    AccType<Real> dout_diff_sum[Vec::ELEMS] = {0};
    
    #pragma unroll
    for (int iR = thread; iR < DR; iR += 32) {
        Vec dout = Vec::load<DOUT_SV>(DOUT + iN * DOUT_SN + iV * DOUT_SV + iR * DOUT_SR);
        Vec diff = Vec::load<DIFF_SV>(DIFF + iN * DIFF_SN + iV * DIFF_SV + iR * DIFF_SR);
        Vec dout_diff = dout * diff;
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            dout_sum[i] += AccType<Real>(dout.h[i]);
            dout_diff_sum[i] += AccType<Real>(dout_diff.h[i]);
        }
    }
    
    #pragma unroll
    for (int i = 0; i < Vec::ELEMS; i++) {
        dout_sum[i] = WarpReduce::sum(dout_sum[i]);
        dout_diff_sum[i] = WarpReduce::sum(dout_diff_sum[i]);
    }
    
    Vec inv_std = Vec::load<STD_SV>(STD + iN * STD_SN + iV * STD_SV);
    Vec inv_n = Vec::fillall(1.f / DR);
    Vec dout_sum_inv_std_n = inv_n * inv_std;
    Vec dout_diff_sum_inv_std3_n = dout_sum_inv_std_n * inv_std * inv_std;
    
    #pragma unroll
    for (int i = 0; i < Vec::ELEMS; i++) {
        dout_sum_inv_std_n.h[i] *= dout_sum[i];
        dout_diff_sum_inv_std3_n.h[i] *= dout_diff_sum[i];
    }
    
    #pragma unroll
    for (int iR = thread; iR < DR; iR += 32) {
        Vec dout = Vec::load<DOUT_SV>(DOUT + iN * DOUT_SN + iV * DOUT_SV + iR * DOUT_SR);
        Vec diff = Vec::load<DIFF_SV>(DIFF + iN * DIFF_SN + iV * DIFF_SV + iR * DIFF_SR);
        Vec d_ln_in = dout * inv_std - dout_sum_inv_std_n - diff * dout_diff_sum_inv_std3_n;
        d_ln_in.store<D_LN_IN_SV>(D_LN_IN + iN * D_LN_IN_SN + iV * D_LN_IN_SV + iR * D_LN_IN_SR);
        
        Vec mask = Vec::load<DROP_MASK_SV>(DROP_MASK + iN * DROP_MASK_SN + iV * DROP_MASK_SV + iR * DROP_MASK_SR);
        Vec d_drop_in = mask * d_ln_in;
        d_drop_in.store<D_DROP_IN_SV>(D_DROP_IN + iN * D_DROP_IN_SN + iV * D_DROP_IN_SV + iR * D_DROP_IN_SR);
    }
}

template <typename Real,
    typename dr, typename dv,
    typename dout_layout,
    typename std_layout,
    typename diff_layout,
    typename drop_mask_layout,
    typename d_ln_in_layout,
    typename d_drop_in_layout>
struct BackwardLayerNormResidualDropout {
    using dr_idx = metal::find<dout_layout, dr>;
    using dout_layout_nv = metal::erase<dout_layout, dr_idx>;

    using dv_idx = metal::find<dout_layout_nv, dv>;
    using dout_layout_n = metal::erase<dout_layout_nv, dv_idx>;

    using dn = metal::at<dout_layout_n, metal::number<0>>;

    using dout_sn = MetaHelpers::elemStride<dout_layout, dn>;
    using dout_sv = MetaHelpers::elemStride<dout_layout, dv>;
    using dout_sr = MetaHelpers::elemStride<dout_layout, dr>;
    
    using std_sn = MetaHelpers::elemStride<std_layout, dn>;
    using std_sv = MetaHelpers::elemStride<std_layout, dv>;
    
    using diff_sn = MetaHelpers::elemStride<diff_layout, dn>;
    using diff_sv = MetaHelpers::elemStride<diff_layout, dv>;
    using diff_sr = MetaHelpers::elemStride<diff_layout, dr>;
    
    using drop_mask_sn = MetaHelpers::elemStride<drop_mask_layout, dn>;
    using drop_mask_sv = MetaHelpers::elemStride<drop_mask_layout, dv>;
    using drop_mask_sr = MetaHelpers::elemStride<drop_mask_layout, dr>;
    
    using d_ln_in_sn = MetaHelpers::elemStride<d_ln_in_layout, dn>;
    using d_ln_in_sv = MetaHelpers::elemStride<d_ln_in_layout, dv>;
    using d_ln_in_sr = MetaHelpers::elemStride<d_ln_in_layout, dr>;
    
    using d_drop_in_sn = MetaHelpers::elemStride<d_drop_in_layout, dn>;
    using d_drop_in_sv = MetaHelpers::elemStride<d_drop_in_layout, dv>;
    using d_drop_in_sr = MetaHelpers::elemStride<d_drop_in_layout, dr>;
    
    static_assert(dv::value % VectorType<Real>::ELEMS == 0, "");
    
    static void run(Real* DOUT, Real* STD, Real* DIFF, Real* DROP_MASK,
                    Real* D_LN_IN, Real* D_DROP_IN, cudaStream_t stream) {
        backwardLayerNormResidualDropoutKernelSeparateVecReduce<Real,
            dn::value, dv::value, dr::value,
            dout_sn::value, dout_sv::value, dout_sr::value,
            std_sn::value, std_sv::value,
            diff_sn::value, diff_sv::value, diff_sr::value,
            drop_mask_sn::value, drop_mask_sv::value, drop_mask_sr::value,
            d_ln_in_sn::value, d_ln_in_sv::value, d_ln_in_sr::value,
            d_drop_in_sn::value, d_drop_in_sv::value, d_drop_in_sr::value
        >
            <<<dim3(dn::value, dv::value / VectorType<Real>::ELEMS), 32, 0, stream>>>(
                DOUT, STD, DIFF, DROP_MASK,
                D_LN_IN, D_DROP_IN);
        CHECK(cudaPeekAtLastError());
    }
};


template <typename Real,
    int D1, int D2, int DR,
    int DOUT_S1, int DOUT_S2, int DOUT_SR,
    int STD_S1, int STD_S2,
    int DIFF_S1, int DIFF_S2, int DIFF_SR,
    int DROP_MASK_S1, int DROP_MASK_S2, int DROP_MASK_SR,
    int D_LN_IN_S1, int D_LN_IN_S2, int D_LN_IN_SR,
    int D_DROP_IN_S1, int D_DROP_IN_S2, int D_DROP_IN_SR
>
__global__ void backwardLayerNormResidualDropoutKernelSameVecReduce(
    Real* DOUT, Real* STD, Real* DIFF, Real* DROP_MASK,
    Real* D_LN_IN, Real* D_DROP_IN)
{
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;

    static_assert(DR % Vec::ELEMS == 0);
    
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int thread = threadIdx.x;
    
    Acc dout_sum = Acc();
    Acc dout_diff_sum = Acc();
    
    #pragma unroll
    for (int iR = thread * Vec::ELEMS; iR < DR; iR += 32 * Vec::ELEMS) {
        Vec dout = Vec::load<DOUT_SR>(DOUT + i1 * DOUT_S1 + i2 * DOUT_S2 + iR * DOUT_SR);
        Vec diff = Vec::load<DIFF_SR>(DIFF + i1 * DIFF_S1 + i2 * DIFF_S2 + iR * DIFF_SR);
        Vec dout_diff = dout * diff;
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            dout_sum += Acc(dout.h[i]);
            dout_diff_sum += Acc(dout_diff.h[i]);
        }
    }
    
    dout_sum = WarpReduce::sum(dout_sum);
    dout_diff_sum = WarpReduce::sum(dout_diff_sum);
    
    Acc inv_std = Acc(STD[i1 * STD_S1 + i2 * STD_S2]);
    Acc inv_n = Acc(1. / DR);
    Acc dout_sum_inv_std_n = inv_n * inv_std;
    Acc dout_diff_sum_inv_std3_n = dout_sum_inv_std_n * inv_std * inv_std;
    
    Vec f_inv_std = Vec::fillall(inv_std);
    Vec f_dout_sum_inv_std_n = Vec::fillall(dout_sum_inv_std_n * dout_sum);
    Vec f_dout_diff_sum_inv_std3_n = Vec::fillall(dout_diff_sum_inv_std3_n * dout_diff_sum);
    
    #pragma unroll
    for (int iR = thread * Vec::ELEMS; iR < DR; iR += 32 * Vec::ELEMS) {
        Vec dout = Vec::load<DOUT_SR>(DOUT + i1 * DOUT_S1 + i2 * DOUT_S2 + iR * DOUT_SR);
        Vec diff = Vec::load<DIFF_SR>(DIFF + i1 * DIFF_S1 + i2 * DIFF_S2 + iR * DIFF_SR);
        Vec d_ln_in = dout * f_inv_std - f_dout_sum_inv_std_n - diff * f_dout_diff_sum_inv_std3_n;
        d_ln_in.store<D_LN_IN_SR>(D_LN_IN + i1 * D_LN_IN_S1 + i2 * D_LN_IN_S2 + iR * D_LN_IN_SR);
        
        Vec mask = Vec::load<DROP_MASK_SR>(DROP_MASK + i1 * DROP_MASK_S1 + i2 * DROP_MASK_S2 + iR * DROP_MASK_SR);
        Vec d_drop_in = mask * d_ln_in;
        d_drop_in.store<D_DROP_IN_SR>(D_DROP_IN + i1 * D_DROP_IN_S1 + i2 * D_DROP_IN_S2 + iR * D_DROP_IN_SR);
    }
}


template <typename Real,
    typename dr,
    typename dout_layout,
    typename std_layout,
    typename diff_layout,
    typename drop_mask_layout,
    typename d_ln_in_layout,
    typename d_drop_in_layout>
struct BackwardLayerNormResidualDropout<
    Real, dr, dr,
    dout_layout,
    std_layout,
    diff_layout,
    drop_mask_layout,
    d_ln_in_layout,
    d_drop_in_layout
> {
    using dr_idx = metal::find<dout_layout, dr>;
    using dout_layout_12 = metal::erase<dout_layout, dr_idx>;

    using d1 = metal::at<dout_layout_12, metal::number<0>>;
    using d2 = metal::at<dout_layout_12, metal::number<1>>;

    using dout_s1 = MetaHelpers::elemStride<dout_layout, d1>;
    using dout_s2 = MetaHelpers::elemStride<dout_layout, d2>;
    using dout_sr = MetaHelpers::elemStride<dout_layout, dr>;
    
    using std_s1 = MetaHelpers::elemStride<std_layout, d1>;
    using std_s2 = MetaHelpers::elemStride<std_layout, d2>;
    
    using diff_s1 = MetaHelpers::elemStride<diff_layout, d1>;
    using diff_s2 = MetaHelpers::elemStride<diff_layout, d2>;
    using diff_sr = MetaHelpers::elemStride<diff_layout, dr>;
    
    using drop_mask_s1= MetaHelpers::elemStride<drop_mask_layout, d1>;
    using drop_mask_s2 = MetaHelpers::elemStride<drop_mask_layout, d2>;
    using drop_mask_sr = MetaHelpers::elemStride<drop_mask_layout, dr>;
    
    using d_ln_in_s1 = MetaHelpers::elemStride<d_ln_in_layout, d1>;
    using d_ln_in_s2 = MetaHelpers::elemStride<d_ln_in_layout, d2>;
    using d_ln_in_sr = MetaHelpers::elemStride<d_ln_in_layout, dr>;
    
    using d_drop_in_s1 = MetaHelpers::elemStride<d_drop_in_layout, d1>;
    using d_drop_in_s2 = MetaHelpers::elemStride<d_drop_in_layout, d2>;
    using d_drop_in_sr = MetaHelpers::elemStride<d_drop_in_layout, dr>;
    
    static_assert(dr::value % VectorType<Real>::ELEMS == 0, "");
    
    static void run(Real* DOUT, Real* STD, Real* DIFF, Real* DROP_MASK,
                    Real* D_LN_IN, Real* D_DROP_IN, cudaStream_t stream) {
        backwardLayerNormResidualDropoutKernelSameVecReduce<Real,
            d1::value, d2::value, dr::value,
            dout_s1::value, dout_s2::value, dout_sr::value,
            std_s1::value, std_s2::value,
            diff_s1::value, diff_s2::value, diff_sr::value,
            drop_mask_s1::value, drop_mask_s2::value, drop_mask_sr::value,
            d_ln_in_s1::value, d_ln_in_s2::value, d_ln_in_sr::value,
            d_drop_in_s1::value, d_drop_in_s2::value, d_drop_in_sr::value
        >
            <<<dim3(d1::value, d2::value), 32, 0, stream>>>(
                DOUT, STD, DIFF, DROP_MASK,
                D_LN_IN, D_DROP_IN);
        CHECK(cudaPeekAtLastError());
    }
};