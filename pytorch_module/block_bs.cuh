#pragma once

#include "blocks.cuh"

template <typename Real,
    typename DN1, typename DN2, typename DN3, typename DR,
    typename DV,
    int OUT_SN1, int OUT_SN2, int OUT_SN3, int OUT_SR,
    int DIN_SN1, int DIN_SN2, int DIN_SN3, int DIN_SR,
    int ATTN_DROP_MASK_SN1, int ATTN_DROP_MASK_SN2, int ATTN_DROP_MASK_SN3, int ATTN_DROP_MASK_SR,
    int DATTN_DROP_SN1, int DATTN_DROP_SN2, int DATTN_DROP_SN3, int DATTN_DROP_SR
>
__global__ void backwardSoftmaxKernel(
    Real* OUT,
    Real* DIN,
    Real* ATTN_DROP_MASK, Real* DATTN_DROP)
{
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;

    static_assert(DV::value % Vec::ELEMS == 0, "");
    
    int iN1 = blockIdx.x;
    if (metal::same<DN1, DV>::value) iN1 *= Vec::ELEMS;
    int iN2 = blockIdx.y;
    if (metal::same<DN2, DV>::value) iN2 *= Vec::ELEMS;
    int iN3 = blockIdx.z;
    if (metal::same<DN3, DV>::value) iN3 *= Vec::ELEMS;

    int thread = threadIdx.x;

    constexpr int AccSize = metal::same<DR, DV>::value ? 1 : Vec::ELEMS;
    Acc dout_out_sum[AccSize] = {};
    
    int r_start = thread;
    int r_step = 32;
    if (metal::same<DR, DV>::value) {
        r_start *= Vec::ELEMS;
        r_step *= Vec::ELEMS;
    }

    constexpr int OUT_SV = 
        metal::same<DN1, DV>::value ? OUT_SN1 :
        metal::same<DN2, DV>::value ? OUT_SN2 : 
        metal::same<DN3, DV>::value ? OUT_SN3 : OUT_SR;
    constexpr int DIN_SV = 
        metal::same<DN1, DV>::value ? DIN_SN1 :
        metal::same<DN2, DV>::value ? DIN_SN2 : 
        metal::same<DN3, DV>::value ? DIN_SN3 : DIN_SR;
    constexpr int ATTN_DROP_MASK_SV = 
        metal::same<DN1, DV>::value ? ATTN_DROP_MASK_SN1 :
        metal::same<DN2, DV>::value ? ATTN_DROP_MASK_SN2 : 
        metal::same<DN3, DV>::value ? ATTN_DROP_MASK_SN3 : ATTN_DROP_MASK_SR;
    constexpr int DATTN_DROP_SV = 
        metal::same<DN1, DV>::value ? DATTN_DROP_SN1 :
        metal::same<DN2, DV>::value ? DATTN_DROP_SN2 : 
        metal::same<DN3, DV>::value ? DATTN_DROP_SN3 : DATTN_DROP_SR;

    #pragma unroll
    for (int iR = r_start; iR < DR::value; iR += r_step) {
        Vec attn_drop_mask = Vec::load<ATTN_DROP_MASK_SV>(ATTN_DROP_MASK + iN1 * ATTN_DROP_MASK_SN1 + iN2 * ATTN_DROP_MASK_SN2 + iN3 * ATTN_DROP_MASK_SN3 + iR * ATTN_DROP_MASK_SR);
        Vec dattn_drop = Vec::load<DATTN_DROP_SV>(DATTN_DROP + iN1 * DATTN_DROP_SN1 + iN2 * DATTN_DROP_SN2 + iN3 * DATTN_DROP_SN3 + iR * DATTN_DROP_SR);

        Vec dalpha = attn_drop_mask * dattn_drop;

        dalpha.store<DIN_SV>(DIN + iN1 * DIN_SN1 + iN2 * DIN_SN2 + iN3 * DIN_SN3 + iR * DIN_SR);

        Vec out = Vec::load<OUT_SV>(OUT + iN1 * OUT_SN1 + iN2 * OUT_SN2 + iN3 * OUT_SN3 + iR * OUT_SR);
        
        Vec dout_out = dalpha * out;
        
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            int idx = metal::same<DR, DV>::value ? 0 : i;
            dout_out_sum[idx] += Acc(dout_out.h[i]);
        }
    }
    
    Vec dout_out_final_sum;
    if (metal::same<DR, DV>::value) {
        dout_out_final_sum = Vec::fillall(WarpReduce::sum(dout_out_sum[0]));
    } else {
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            dout_out_final_sum.h[i] = WarpReduce::sum(dout_out_sum[i]);
        }
    }
    
    #pragma unroll
    for (int iR = r_start; iR < DR::value; iR += r_step) {
        Vec dalpha = Vec::load<DIN_SV>(DIN + iN1 * DIN_SN1 + iN2 * DIN_SN2 + iN3 * DIN_SN3 + iR * DIN_SR);
        Vec out = Vec::load<OUT_SV>(OUT + iN1 * OUT_SN1 + iN2 * OUT_SN2 + iN3 * OUT_SN3 + iR * OUT_SR);
        
        Vec din = out * (dalpha - dout_out_final_sum);
        
        din.store<DIN_SV>(DIN + iN1 * DIN_SN1 + iN2 * DIN_SN2 + iN3 * DIN_SN3 + iR * DIN_SR);
    }
}

template <typename Real,
    typename dn1, typename dn2, typename dn3, typename dr,
    typename dv,
    typename out_layout,
    typename din_layout,
    typename attn_drop_mask_layout,
    typename dattn_drop_layout>
struct BackwardSoftmax {
    using out_sn1 = MetaHelpers::elemStride<out_layout, dn1>;
    using out_sn2 = MetaHelpers::elemStride<out_layout, dn2>;
    using out_sn3 = MetaHelpers::elemStride<out_layout, dn3>;
    using out_sr = MetaHelpers::elemStride<out_layout, dr>;
    
    using din_sn1 = MetaHelpers::elemStride<din_layout, dn1>;
    using din_sn2 = MetaHelpers::elemStride<din_layout, dn2>;
    using din_sn3 = MetaHelpers::elemStride<din_layout, dn3>;
    using din_sr = MetaHelpers::elemStride<din_layout, dr>;
    
    using attn_drop_mask_sn1 = MetaHelpers::elemStride<attn_drop_mask_layout, dn1>;
    using attn_drop_mask_sn2 = MetaHelpers::elemStride<attn_drop_mask_layout, dn2>;
    using attn_drop_mask_sn3 = MetaHelpers::elemStride<attn_drop_mask_layout, dn3>;
    using attn_drop_mask_sr = MetaHelpers::elemStride<attn_drop_mask_layout, dr>;

    using dattn_drop_sn1 = MetaHelpers::elemStride<dattn_drop_layout, dn1>;
    using dattn_drop_sn2 = MetaHelpers::elemStride<dattn_drop_layout, dn2>;
    using dattn_drop_sn3 = MetaHelpers::elemStride<dattn_drop_layout, dn3>;
    using dattn_drop_sr = MetaHelpers::elemStride<dattn_drop_layout, dr>;
    
    static_assert(dv::value % VectorType<Real>::ELEMS == 0);

    static void run(
        Real* OUT,
        Real* DIN,
        Real* ATTN_DROP_MASK, Real* DATTN_DROP,
        cudaStream_t stream)
    {
        int blocksX = dn1::value;
        if (metal::same<dn1, dv>::value) blocksX /= VectorType<Real>::ELEMS;
        int blocksY = dn2::value;
        if (metal::same<dn2, dv>::value) blocksY /= VectorType<Real>::ELEMS;
        int blocksZ = dn3::value;
        if (metal::same<dn3, dv>::value) blocksZ /= VectorType<Real>::ELEMS;
        backwardSoftmaxKernel<Real,
            dn1, dn2, dn3, dr,
            dv,
            out_sn1::value, out_sn2::value, out_sn3::value, out_sr::value,
            din_sn1::value, din_sn2::value, din_sn3::value, din_sr::value,
            attn_drop_mask_sn1::value, attn_drop_mask_sn2::value, attn_drop_mask_sn3::value, attn_drop_mask_sr::value,
            dattn_drop_sn1::value, dattn_drop_sn2::value, dattn_drop_sn3::value, dattn_drop_sr::value
        >
            <<<dim3(blocksX, blocksY, blocksZ), 32, 0, stream>>>(
                OUT,
                DIN,
                ATTN_DROP_MASK, DATTN_DROP);
        CHECK(cudaPeekAtLastError());
    }
};
