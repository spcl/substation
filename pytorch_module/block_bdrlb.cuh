#pragma once

#include "blocks.cuh"

template <typename Real,
    typename DR1, typename DR2, typename DN,
    typename DV, typename DW,
    int DDROP_SR1, int DDROP_SR2, int DDROP_SN,
    int DROP_MASK_SR1, int DROP_MASK_SR2, int DROP_MASK_SN,
    int LIN_SR1, int LIN_SR2, int LIN_SN,
    int DLINEAR_SR1, int DLINEAR_SR2, int DLINEAR_SN
>
__global__ void backwardDroupoutReluLinearBias(
    Real* DDROP, Real* DROP_MASK, Real* LIN,
    Real* DLINEAR, Real* DBIAS)
{
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;

    static_assert(DV::value % Vec::ELEMS == 0);

    int iN = blockIdx.x;
    if (metal::same<DN, DV>::value) {
        iN *= Vec::ELEMS;
    }

    int thread = threadIdx.x;
    
    constexpr int AccSize = metal::same<DN, DV>::value ? Vec::ELEMS : 1;

    Acc dlinear_bias[AccSize] = {};
    
    int r1_start = metal::same<DR1, DW>::value ? thread : 0;
    int r1_step = metal::same<DR1, DW>::value ? 32 : 1;
    if (metal::same<DR1, DV>::value) {
        r1_start *= Vec::ELEMS;
        r1_step *= Vec::ELEMS;
    }

    int r2_start = metal::same<DR2, DW>::value ? thread : 0;
    int r2_step = metal::same<DR2, DW>::value ? 32 : 1;
    if (metal::same<DR2, DV>::value) {
        r2_start *= Vec::ELEMS;
        r2_step *= Vec::ELEMS;
    }

    constexpr int DDROP_SV =
        metal::same<DR1, DV>::value ? DDROP_SR1 :
        metal::same<DR2, DV>::value ? DDROP_SR2 : DDROP_SN;
    constexpr int DROP_MASK_SV =
        metal::same<DR1, DV>::value ? DROP_MASK_SR1 :
        metal::same<DR2, DV>::value ? DROP_MASK_SR2 : DROP_MASK_SN;
    constexpr int LIN_SV = 
        metal::same<DR1, DV>::value ? LIN_SR1 :
        metal::same<DR2, DV>::value ? LIN_SR2 : LIN_SN;
    constexpr int DLINEAR_SV = 
        metal::same<DR1, DV>::value ? DLINEAR_SR1 :
        metal::same<DR2, DV>::value ? DLINEAR_SR2 : DLINEAR_SN;

    #pragma unroll
    for (int iR1 = r1_start; iR1 < DR1::value; iR1 += r1_step) {
        #pragma unroll
        for (int iR2 = r2_start; iR2 < DR2::value; iR2 += r2_step) {
            Vec ddrop = Vec::load<DDROP_SV>(DDROP + iR1 * DDROP_SR1 + iR2 * DDROP_SR2 + iN * DDROP_SN);
            Vec drop_mask = Vec::load<DROP_MASK_SV>(DROP_MASK + iR1 * DROP_MASK_SR1 + iR2 * DROP_MASK_SR2 + iN * DROP_MASK_SN);
            Vec lin = Vec::load<LIN_SV>(LIN + iR1 * LIN_SR1 + iR2 * LIN_SR2 + iN * LIN_SN);
            
            Vec dlinear = ddrop * drop_mask;
            
            #pragma unroll
            for (int i = 0; i < Vec::ELEMS; i++) {
                if (lin.h[i] > Real(0)) {
                    int idx = AccSize == 1 ? 0 : i;
                    dlinear_bias[idx] += Acc(dlinear.h[i]);
                } else {
                    dlinear.h[i] = Real(0);
                }
            }
            
            dlinear.store<DLINEAR_SV>(DLINEAR + iR1 * DLINEAR_SR1 + iR2 * DLINEAR_SR2 + iN * DLINEAR_SN);
            
        }
    }

    Real dbias[AccSize];
    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        dbias[i] = WarpReduce::sum(dlinear_bias[i]);
    }

    if (AccSize == 1) {
        DBIAS[iN] = dbias[0];
    } else {
        (*(Vec*)&dbias).store<1>(DBIAS + iN);
    }
}

template <typename Real,
    typename dr1, typename dr2, typename dn,
    typename dv, typename dw,
    typename ddrop_layout,
    typename drop_mask_layout,
    typename lin_layout,
    typename dlinear_layout>
struct BackwardDropoutReluLinearBias {
    using ddrop_sr1 = MetaHelpers::elemStride<ddrop_layout, dr1>;
    using ddrop_sr2 = MetaHelpers::elemStride<ddrop_layout, dr2>;
    using ddrop_sn = MetaHelpers::elemStride<ddrop_layout, dn>;
    
    using drop_mask_sr1 = MetaHelpers::elemStride<drop_mask_layout, dr1>;
    using drop_mask_sr2 = MetaHelpers::elemStride<drop_mask_layout, dr2>;
    using drop_mask_sn = MetaHelpers::elemStride<drop_mask_layout, dn>;
    
    using lin_sr1 = MetaHelpers::elemStride<lin_layout, dr1>;
    using lin_sr2 = MetaHelpers::elemStride<lin_layout, dr2>;
    using lin_sn = MetaHelpers::elemStride<lin_layout, dn>;
    
    using dlinear_sr1 = MetaHelpers::elemStride<dlinear_layout, dr1>;
    using dlinear_sr2 = MetaHelpers::elemStride<dlinear_layout, dr2>;
    using dlinear_sn = MetaHelpers::elemStride<dlinear_layout, dn>;
    
    static_assert(dv::value % VectorType<Real>::ELEMS == 0, "");
    
    static void run(
        Real* DDROP, Real* DROP_MASK, Real* LIN,
        Real* DLINEAR, Real* DBIAS,
        cudaStream_t stream)
    {
        int blocks = dn::value;
        if (metal::same<dn, dv>::value) {
            blocks /= VectorType<Real>::ELEMS;
        }
        backwardDroupoutReluLinearBias<Real,
            dr1, dr2, dn,
            dv, dw,
            ddrop_sr1::value, ddrop_sr2::value, ddrop_sn::value,
            drop_mask_sr1::value, drop_mask_sr2::value, drop_mask_sn::value,
            lin_sr1::value, lin_sr2::value, lin_sn::value,
            dlinear_sr1::value, dlinear_sr2::value, dlinear_sn::value
        >
            <<<blocks, 32, 0, stream>>>(
                DDROP, DROP_MASK, LIN,
                DLINEAR, DBIAS);
        CHECK(cudaPeekAtLastError());
    }
};