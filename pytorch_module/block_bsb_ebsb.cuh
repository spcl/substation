#pragma once

#include "blocks.cuh"

template <typename Real,
    typename DR1, typename DR2, typename DN,
    typename DV, typename DW,
    int IN_SR1, int IN_SR2, int IN_SN,
    int DIN_SR1, int DIN_SR2, int DIN_SN,
    int DOUT_SR1, int DOUT_SR2, int DOUT_SN
>
__forceinline__ __device__ void backwardScaleBiasKernel(
    Real* IN, Real* SCALE, Real* DOUT,
    Real* DIN, Real* DSCALE, Real* DBIAS)
{
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;

    static_assert(DV::value % Vec::ELEMS == 0);

    // if N is simple:
    //  int iN = blockIdx.x;
    // else:
    //  int iN = blockIdx.x * Vec::ELEMS;
    int iN = blockIdx.x;
    if (metal::same<DN, DV>::value) {
        iN *= Vec::ELEMS;
    }

    int thread = threadIdx.x;


    constexpr int AccSize = metal::same<DN, DV>::value ? Vec::ELEMS : 1;
    // if N is vec
    //  AccType<Real> dscale[Vec::ELEMS];
    // else
    //  AccType<Real> dscale;

    Acc dscale[AccSize] = {};
    Acc dbias[AccSize] = {};
    
    // if N is simple
    //  Real scale = SCALE[iV];
    // else
    //  Vec scale = Vec::load<1>(SCALE + iV);
    Vec scale;
    if (AccSize == 1) {
        scale = Vec::fillall(SCALE[iN]);
    } else {
        scale = Vec::load<1>(SCALE + iN);
    }
    
    // general loop strategies
    // simple: 0, DWR, 1
    // warp: thread, DWR, 32
    // vec: 0 * ELEMS, DWR, 1 * ELEMS
    // warp + vec: thread * ELEMS, DWR, 32 * ELEMS

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

    constexpr int IN_SV = 
        metal::same<DR1, DV>::value ? IN_SR1 :
        metal::same<DR2, DV>::value ? IN_SR2 : IN_SN;
    constexpr int DIN_SV = 
        metal::same<DR1, DV>::value ? DIN_SR1 :
        metal::same<DR2, DV>::value ? DIN_SR2 : DIN_SN;
    constexpr int DOUT_SV = 
        metal::same<DR1, DV>::value ? DOUT_SR1 :
        metal::same<DR2, DV>::value ? DOUT_SR2 : DOUT_SN;

    #pragma unroll
    for (int iR1 = r1_start; iR1 < DR1::value; iR1 += r1_step) {
        #pragma unroll
        for (int iR2 = r2_start; iR2 < DR2::value; iR2 += r2_step) {
            // template param is always vector dim
            Vec dout = Vec::load<DOUT_SV>(DOUT + iR1 * DOUT_SR1 + iR2 * DOUT_SR2 + iN * DOUT_SN);
            Vec din = dout * scale;
            // template param is always vector dim
            din.store<DIN_SV>(DIN + iR1 * DIN_SR1 + iR2 * DIN_SR2 + iN * DIN_SN);
            // template param is always vector dim
            Vec in = Vec::load<IN_SV>(IN + iR1 * IN_SR1 + iR2 * IN_SR2 + iN * IN_SN);
            Vec dscale_item = dout * in;
            // if N is vec 
            // #pragma unroll
            // for (int i = 0; i < Vec::ELEMS; i++) {
            //     dscale[i] += dscale_item.h[i];
            //     dbias[i] += dout.h[i];
            // }
            // else
            // #pragma unroll
            // for (int i = 0; i < Vec::ELEMS; i++) {
            //     dscale += dscale_item.h[i];
            //     dbias += dout.h[i];
            // }

            #pragma unroll
            for (int i = 0; i < Vec::ELEMS; i++) {
                if (AccSize == 1) {
                    dscale[0] += Acc(dscale_item.h[i]);
                    dbias[0] += Acc(dout.h[i]);
                } else {
                    dscale[i] += Acc(dscale_item.h[i]);
                    dbias[i] += Acc(dout.h[i]);
                }
            }
        }
    }

    // if N is vec dscale[i] or dscale
    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        dscale[i] = WarpReduce::sum(dscale[i]);
        dbias[i] = WarpReduce::sum(dbias[i]);
    }
    
    // if N is vec: Vec or Real
    Real dscaleWrite[AccSize];
    Real dbiasWrite[AccSize];

    // if N is vec with loop, otherwise without loop
    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        dscaleWrite[i] = dscale[i];
        dbiasWrite[i] = dbias[i];
    }
    
    // write single elem or store vector depending on N
    if (AccSize == 1) {
        DSCALE[iN] = dscaleWrite[0];
        DBIAS[iN] = dbiasWrite[0];
    } else {
        (*(Vec*)&dscaleWrite).store<1>(DSCALE + iN);
        (*(Vec*)&dbiasWrite).store<1>(DBIAS + iN);
    }
}


template <
    typename Real,
    typename DR1, typename DR2, typename DN,
    typename DV, typename DW,
    int IN_SR1, int IN_SR2, int IN_SN,
    int DIN_SR1, int DIN_SR2, int DIN_SN,
    int DOUT_SR1, int DOUT_SR2, int DOUT_SN
>
__global__ void simpleBackwardScaleBiasKernel(
    Real* IN, Real* SCALE, Real* DOUT,
    Real* DIN, Real* DSCALE, Real* DBIAS)
{
    backwardScaleBiasKernel<Real, 
        DR1, DR2, DN,
        DV, DW,
        IN_SR1, IN_SR2, IN_SN,
        DIN_SR1, DIN_SR2, DIN_SN,
        DOUT_SR1, DOUT_SR2, DOUT_SN
    >(IN, SCALE, DOUT, DIN, DSCALE, DBIAS);
}

template <
    typename Real,
    typename dr1, typename dr2, typename dn,
    typename dv, typename dw,
    typename in_layout,
    typename din_layout,
    typename dout_layout>
struct BackwardScaleBias {
    static_assert(metal::size<in_layout>::value == 3, "only 3D arrays are supported");
    
    using in_sr1 = MetaHelpers::elemStride<in_layout, dr1>;
    using in_sr2 = MetaHelpers::elemStride<in_layout, dr2>;
    using in_sn = MetaHelpers::elemStride<in_layout, dn>;
    
    using din_sr1 = MetaHelpers::elemStride<din_layout, dr1>;
    using din_sr2 = MetaHelpers::elemStride<din_layout, dr2>;
    using din_sn = MetaHelpers::elemStride<din_layout, dn>;
    
    using dout_sr1 = MetaHelpers::elemStride<dout_layout, dr1>;
    using dout_sr2 = MetaHelpers::elemStride<dout_layout, dr2>;
    using dout_sn = MetaHelpers::elemStride<dout_layout, dn>;
    
    
    static void run(Real* IN, Real* SCALE, Real* DOUT,
                    Real* DIN, Real* DSCALE, Real* DBIAS, cudaStream_t stream) {
        static_assert((!metal::same<dn, dv>::value) || (dn::value % VectorType<Real>::ELEMS == 0));
        
        int blocks = dn::value;
        if (metal::same<dn, dv>::value) {
            blocks /= VectorType<Real>::ELEMS;
        }

        simpleBackwardScaleBiasKernel<
            Real,
            dr1, dr2, dn,
            dv, dw,
            in_sr1::value, in_sr2::value, in_sn::value,
            din_sr1::value, din_sr2::value, din_sn::value,
            dout_sr1::value, dout_sr2::value, dout_sn::value
        >
            <<<blocks, 32, 0, stream>>>(
                IN, SCALE, DOUT,
                DIN, DSCALE, DBIAS);
        CHECK(cudaPeekAtLastError());
    }
};


template <typename Real,
    typename DR1, typename DR2, typename DN,
    typename DV, typename DW,
    int IN_SR1, int IN_SR2, int IN_SN,
    int DIN_SR1, int DIN_SR2, int DIN_SN,
    int DOUT1_SR1, int DOUT1_SR2, int DOUT1_SN,
    int DOUT2_SR1, int DOUT2_SR2, int DOUT2_SN,
    int DLINEAR_SR1, int DLINEAR_SR2, int DLINEAR_SN
>
__global__ void extendedBackwardScaleBiasKernel(
    Real* IN, Real* SCALE, Real* DOUT1, Real* DOUT2, Real* DLINEAR,
    Real* DIN, Real* DSCALE, Real* DBIAS, Real* DLINEAR_BIAS)
{
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;

    int iN = blockIdx.x;
    if (metal::same<DN, DV>::value) iN *= Vec::ELEMS;
    int thread = threadIdx.x;
     
    constexpr int AccSize = (metal::same<DN, DV>::value) ? Vec::ELEMS : 1;

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

    constexpr int DLINEAR_SV = 
        metal::same<DR1, DV>::value ? DLINEAR_SR1 :
        metal::same<DR2, DV>::value ? DLINEAR_SR2 : DLINEAR_SN;
    constexpr int DOUT1_SV = 
        metal::same<DR1, DV>::value ? DOUT1_SR1 :
        metal::same<DR2, DV>::value ? DOUT1_SR2 : DOUT1_SN;
    constexpr int DOUT2_SV = 
        metal::same<DR1, DV>::value ? DOUT2_SR1 :
        metal::same<DR2, DV>::value ? DOUT2_SR2 : DOUT2_SN;
    constexpr int IN_SV = 
        metal::same<DR1, DV>::value ? IN_SR1 :
        metal::same<DR2, DV>::value ? IN_SR2 : IN_SN;
    constexpr int DIN_SV = 
        metal::same<DR1, DV>::value ? DIN_SR1 :
        metal::same<DR2, DV>::value ? DIN_SR2 : DIN_SN;
        

    #pragma unroll
    for (int iR1 = r1_start; iR1 < DR1::value; iR1 += r1_step) {
        #pragma unroll
        for (int iR2 = r2_start; iR2 < DR2::value; iR2 += r2_step) {
            Vec dlinear = Vec::load<DLINEAR_SV>(DLINEAR + iR1 * DLINEAR_SR1 + iR2 * DLINEAR_SR2 + iN * DLINEAR_SN);
            #pragma unroll
            for (int i = 0; i < Vec::ELEMS; i++) {
                int idx = AccSize == 1 ? 0 : i;
                dlinear_bias[idx] += Acc(dlinear.h[i]);
            }
        }
    }

    if (metal::same<DN, DV>::value) {
        Vec write_linear_bias;
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            write_linear_bias.h[i] = WarpReduce::sum(dlinear_bias[i]);
        }
        write_linear_bias.store<1>(DLINEAR_BIAS + iN);
    } else {
        DLINEAR_BIAS[iN] = WarpReduce::sum(dlinear_bias[0]);
    }
    
    Acc dscale[AccSize] = {};
    Acc dbias[AccSize] = {};
    
    Vec scale;
    if (AccSize == 1) {
        scale = Vec::fillall(SCALE[iN]);
    } else {
        scale = Vec::load<1>(SCALE + iN);
    }

    #pragma unroll
    for (int iR1 = r1_start; iR1 < DR1::value; iR1 += r1_step) {
        #pragma unroll
        for (int iR2 = r2_start; iR2 < DR2::value; iR2 += r2_step) {
            Vec dout1 = Vec::load<DOUT1_SV>(DOUT1 + iR1 * DOUT1_SR1 + iR2 * DOUT1_SR2 + iN * DOUT1_SN);
            Vec dout2 = Vec::load<DOUT2_SV>(DOUT2 + iR1 * DOUT2_SR1 + iR2 * DOUT2_SR2 + iN * DOUT2_SN);
            
            Vec dout = dout1 + dout2;
            //dout12.store<DOUT12_SV>(DOUT12 + iR1 * DOUT12_SR1 + iR2 * DOUT12_SR2 + iN * DOUT12_SN);

            //Vec dout = Vec::load<DOUT12_SV>(DOUT12 + iR1 * DOUT12_SR1 + iR2 * DOUT12_SR2 + iN * DOUT12_SN);
            Vec din = dout * scale;

            din.store<DIN_SV>(DIN + iR1 * DIN_SR1 + iR2 * DIN_SR2 + iN * DIN_SN);

            Vec in = Vec::load<IN_SV>(IN + iR1 * IN_SR1 + iR2 * IN_SR2 + iN * IN_SN);
            Vec dscale_item = dout * in;

            #pragma unroll
            for (int i = 0; i < Vec::ELEMS; i++) {
                if (AccSize == 1) {
                    dscale[0] += Acc(dscale_item.h[i]);
                    dbias[0] += Acc(dout.h[i]);
                } else {
                    dscale[i] += Acc(dscale_item.h[i]);
                    dbias[i] += Acc(dout.h[i]);
                }
            }
        }
    }

    // if N is vec dscale[i] or dscale
    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        dscale[i] = WarpReduce::sum(dscale[i]);
        dbias[i] = WarpReduce::sum(dbias[i]);
    }
    
    // if N is vec: Vec or Real
    Real dscaleWrite[AccSize];
    Real dbiasWrite[AccSize];

    // if N is vec with loop, otherwise without loop
    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        dscaleWrite[i] = dscale[i];
        dbiasWrite[i] = dbias[i];
    }
    
    // write single elem or store vector depending on N
    if (AccSize == 1) {
        DSCALE[iN] = dscaleWrite[0];
        DBIAS[iN] = dbiasWrite[0];
    } else {
        (*(Vec*)&dscaleWrite).store<1>(DSCALE + iN);
        (*(Vec*)&dbiasWrite).store<1>(DBIAS + iN);
    }

}

template <typename Real,
    typename dr1, typename dr2, typename dn,
    typename dv, typename dw,
    typename in_layout,
    typename din_layout,
    typename dout1_layout,
    typename dout2_layout,
    typename dlinear_layout>
struct ExtendedBackwardScaleBias {
    using in_sr1 = MetaHelpers::elemStride<in_layout, dr1>;
    using in_sr2 = MetaHelpers::elemStride<in_layout, dr2>;
    using in_sn = MetaHelpers::elemStride<in_layout, dn>;
    
    using din_sr1 = MetaHelpers::elemStride<din_layout, dr1>;
    using din_sr2 = MetaHelpers::elemStride<din_layout, dr2>;
    using din_sn = MetaHelpers::elemStride<din_layout, dn>;
    
    using dout1_sr1 = MetaHelpers::elemStride<dout1_layout, dr1>;
    using dout1_sr2 = MetaHelpers::elemStride<dout1_layout, dr2>;
    using dout1_sn = MetaHelpers::elemStride<dout1_layout, dn>;
    
    using dout2_sr1 = MetaHelpers::elemStride<dout2_layout, dr1>;
    using dout2_sr2 = MetaHelpers::elemStride<dout2_layout, dr2>;
    using dout2_sn = MetaHelpers::elemStride<dout2_layout, dn>;
    
    using dlinear_sr1 = MetaHelpers::elemStride<dlinear_layout, dr1>;
    using dlinear_sr2 = MetaHelpers::elemStride<dlinear_layout, dr2>;
    using dlinear_sn = MetaHelpers::elemStride<dlinear_layout, dn>;
    
    static_assert(dv::value % VectorType<Real>::ELEMS == 0, "");
    
    static void run(
        Real* IN, Real* SCALE, Real* DOUT1, Real* DOUT2, Real* DLINEAR,
        Real* DIN, Real* DSCALE, Real* DBIAS, Real* DLINEAR_BIAS,
        cudaStream_t stream) 
    {
        int blocks = dn::value;
        if (metal::same<dn, dv>::value) blocks /= VectorType<Real>::ELEMS;
        extendedBackwardScaleBiasKernel<Real,
            dr1, dr2, dn,
            dv, dw,
            in_sr1::value, in_sr2::value, in_sn::value,
            din_sr1::value, din_sr2::value, din_sn::value,
            dout1_sr1::value, dout1_sr2::value, dout1_sn::value,
            dout2_sr1::value, dout2_sr2::value, dout2_sn::value,
            dlinear_sr1::value, dlinear_sr2::value, dlinear_sn::value
        >
            <<<blocks, 32, 0, stream>>>(
                IN, SCALE, DOUT1, DOUT2, DLINEAR,
                DIN, DSCALE, DBIAS, DLINEAR_BIAS);
        CHECK(cudaPeekAtLastError());
    }
};
