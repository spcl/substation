#pragma once

#include "blocks.cuh"

template <typename Real,
    typename DN, typename DR1, typename DR2,
    typename DV, typename DW,
    int DOUT_SN, int DOUT_SR1, int DOUT_SR2
>
__global__ void backwardAttnOutBias(Real* DOUT, Real* DBO) {
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;

    static_assert(DV::value % Vec::ELEMS == 0, "");
    
    int iN = blockIdx.x;

    if (metal::same<DN, DV>::value) iN *= Vec::ELEMS;

    int thread = threadIdx.x;

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

    constexpr int DOUT_SV = 
        metal::same<DR1, DV>::value ? DOUT_SR1 :
        metal::same<DR2, DV>::value ? DOUT_SR2 : DOUT_SN;

    constexpr int AccSize = metal::same<DN, DV>::value ? Vec::ELEMS : 1;
    
    Acc dbo[AccSize] = {};

    #pragma unroll
    for (int iR1 = r1_start; iR1 < DR1::value; iR1 += r1_step) {
        #pragma unroll
        for (int iR2 = r2_start; iR2 < DR2::value; iR2 += r2_step) {
            Vec dout = Vec::load<DOUT_SV>(DOUT + DOUT_SN * iN + DOUT_SR1 * iR1 + DOUT_SR2 * iR2);
            for (int i = 0; i < Vec::ELEMS; i++) {
                int idx = AccSize == 1 ? 0 : i;
                dbo[idx] += Acc(dout.h[i]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        dbo[i] = WarpReduce::sum(dbo[i]);
    }

    Real* dbo_out = DBO + iN;

    if (AccSize == 1) {
        *dbo_out = Real(dbo[0]);
    } else {
        Vec dbo_write;
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            dbo_write.h[i] = Real(dbo[i]); 
        }
        dbo_write.store<1>(dbo_out);
    }
}

template <typename Real,
    typename dn, typename dr1, typename dr2,
    typename dv, typename dw,
    typename dout>
struct BackwardAttnOutBias {
    static constexpr int dout_sn = MetaHelpers::elemStride<dout, dn>::value;
    static constexpr int dout_sr1 = MetaHelpers::elemStride<dout, dr1>::value;
    static constexpr int dout_sr2 = MetaHelpers::elemStride<dout, dr2>::value;
    
    static void run(Real* DOUT, Real* DBO, cudaStream_t stream) 
    {
        dim3 blocks(dn::value);

        if (metal::same<dn, dv>::value) blocks.x /= VectorType<Real>::ELEMS;

        backwardAttnOutBias<Real,
            dn, dr1, dr2,
            dv, dw,
            dout_sn, dout_sr1, dout_sr2
        ><<<blocks, 32, 0, stream>>>(DOUT, DBO);
        CHECK(cudaPeekAtLastError());
    }
};