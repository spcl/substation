#pragma once

#include "blocks.cuh"

template <typename Real,
    typename DN1, typename DN2, typename DR1, typename DR2,
    typename DV, typename DW,
    int DKK_SN1, int DKK_SN2, int DKK_SR1, int DKK_SR2,
    int DVV_SN1, int DVV_SN2, int DVV_SR1, int DVV_SR2,
    int DQQ_SN1, int DQQ_SN2, int DQQ_SR1, int DQQ_SR2,
    int DBK_SN1, int DBK_SN2,
    int DBV_SN1, int DBV_SN2,
    int DBQ_SN1, int DBQ_SN2
>
__global__ void backwardAttentionInputBiases(
    Real* DKK, Real* DVV, Real* DQQ,
    Real* DBK, Real* DBV, Real* DBQ)
{
    using Vec = VectorType<Real>;
    using Acc = AccType<Real>;

    static_assert(DV::value % Vec::ELEMS == 0, "");
    
    int iN1 = blockIdx.x;
    int iN2 = blockIdx.y;

    if (metal::same<DN1, DV>::value) iN1 *= Vec::ELEMS;
    if (metal::same<DN2, DV>::value) iN2 *= Vec::ELEMS;

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

    constexpr int DKK_SV = 
        metal::same<DR1, DV>::value ? DKK_SR1 :
        metal::same<DR2, DV>::value ? DKK_SR2 :
        metal::same<DN1, DV>::value ? DKK_SN1 : DKK_SN2;

    constexpr int DVV_SV = 
        metal::same<DR1, DV>::value ? DVV_SR1 :
        metal::same<DR2, DV>::value ? DVV_SR2 :
        metal::same<DN1, DV>::value ? DVV_SN1 : DVV_SN2;

    constexpr int DQQ_SV = 
        metal::same<DR1, DV>::value ? DQQ_SR1 :
        metal::same<DR2, DV>::value ? DQQ_SR2 :
        metal::same<DN1, DV>::value ? DQQ_SN1 : DQQ_SN2;

    constexpr int DBK_SV = 
        metal::same<DN1, DV>::value ? DBK_SN1 :
        metal::same<DN2, DV>::value ? DBK_SN2 : 1;

    constexpr int DBV_SV = 
        metal::same<DN1, DV>::value ? DBV_SN1 :
        metal::same<DN2, DV>::value ? DBV_SN2 : 1;
    
    constexpr int DBQ_SV = 
        metal::same<DN1, DV>::value ? DBQ_SN1 :
        metal::same<DN2, DV>::value ? DBQ_SN2 : 1;

    constexpr int AccSize = (metal::same<DN1, DV>::value || metal::same<DN2, DV>::value) ? Vec::ELEMS : 1;
    
    Acc dbk[AccSize] = {};
    Acc dbv[AccSize] = {};
    Acc dbq[AccSize] = {};

    #pragma unroll
    for (int iR1 = r1_start; iR1 < DR1::value; iR1 += r1_step) {
        #pragma unroll
        for (int iR2 = r2_start; iR2 < DR2::value; iR2 += r2_step) {
            Vec dkk = Vec::load<DKK_SV>(DKK + DKK_SN1 * iN1 + DKK_SN2 * iN2 + DKK_SR1 * iR1 + DKK_SR2 * iR2);
            for (int i = 0; i < Vec::ELEMS; i++) {
                int idx = AccSize == 1 ? 0 : i;
                dbk[idx] += Acc(dkk.h[i]);
            }
        }
    }

    #pragma unroll
    for (int iR1 = r1_start; iR1 < DR1::value; iR1 += r1_step) {
        #pragma unroll
        for (int iR2 = r2_start; iR2 < DR2::value; iR2 += r2_step) {
            Vec dvv = Vec::load<DVV_SV>(DVV + DVV_SN1 * iN1 + DVV_SN2 * iN2 + DVV_SR1 * iR1 + DVV_SR2 * iR2);
            for (int i = 0; i < Vec::ELEMS; i++) {
                int idx = AccSize == 1 ? 0 : i;
                dbv[idx] += Acc(dvv.h[i]);
            }
        }
    }

    #pragma unroll
    for (int iR1 = r1_start; iR1 < DR1::value; iR1 += r1_step) {
        #pragma unroll
        for (int iR2 = r2_start; iR2 < DR2::value; iR2 += r2_step) {
            Vec dqq = Vec::load<DQQ_SV>(DQQ + DQQ_SN1 * iN1 + DQQ_SN2 * iN2 + DQQ_SR1 * iR1 + DQQ_SR2 * iR2);
            for (int i = 0; i < Vec::ELEMS; i++) {
                int idx = AccSize == 1 ? 0 : i;
                dbq[idx] += Acc(dqq.h[i]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < AccSize; i++) {
        dbk[i] = WarpReduce::sum(dbk[i]);
        dbv[i] = WarpReduce::sum(dbv[i]);
        dbq[i] = WarpReduce::sum(dbq[i]);
    }

    Real* dbk_out = DBK + iN1 * DBK_SN1 + iN2 * DBK_SN2;
    Real* dbv_out = DBV + iN1 * DBV_SN1 + iN2 * DBV_SN2;
    Real* dbq_out = DBQ + iN1 * DBQ_SN1 + iN2 * DBQ_SN2;

    if (AccSize == 1) {
        *dbk_out = Real(dbk[0]);
        *dbq_out = Real(dbq[0]);
        *dbv_out = Real(dbv[0]);
    } else {
        Vec dbk_write;
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            dbk_write.h[i] = dbk[i]; 
        }
        dbk_write.store<DBK_SV>(dbk_out);

        Vec dbv_write;
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            dbv_write.h[i] = dbv[i]; 
        }
        dbv_write.store<DBV_SV>(dbv_out);

        Vec dbq_write;
        #pragma unroll
        for (int i = 0; i < Vec::ELEMS; i++) {
            dbq_write.h[i] = dbq[i]; 
        }
        dbq_write.store<DBQ_SV>(dbq_out);
    }

    
}

template <typename Real,
    typename dn1, typename dn2, typename dr1, typename dr2,
    typename dv, typename dw,
    typename dkk, typename dvv, typename dqq,
    typename dbk, typename dbv, typename dbq>
struct BackwardAttentionInputBiases {
    static constexpr int dkk_sn1 = MetaHelpers::elemStride<dkk, dn1>::value;
    static constexpr int dkk_sn2 = MetaHelpers::elemStride<dkk, dn2>::value;
    static constexpr int dkk_sr1 = MetaHelpers::elemStride<dkk, dr1>::value;
    static constexpr int dkk_sr2 = MetaHelpers::elemStride<dkk, dr2>::value;

    static constexpr int dvv_sn1 = MetaHelpers::elemStride<dvv, dn1>::value;
    static constexpr int dvv_sn2 = MetaHelpers::elemStride<dvv, dn2>::value;
    static constexpr int dvv_sr1 = MetaHelpers::elemStride<dvv, dr1>::value;
    static constexpr int dvv_sr2 = MetaHelpers::elemStride<dvv, dr2>::value;

    static constexpr int dqq_sn1 = MetaHelpers::elemStride<dqq, dn1>::value;
    static constexpr int dqq_sn2 = MetaHelpers::elemStride<dqq, dn2>::value;
    static constexpr int dqq_sr1 = MetaHelpers::elemStride<dqq, dr1>::value;
    static constexpr int dqq_sr2 = MetaHelpers::elemStride<dqq, dr2>::value;

    static constexpr int dbk_sn1 = MetaHelpers::elemStride<dbk, dn1>::value;
    static constexpr int dbk_sn2 = MetaHelpers::elemStride<dbk, dn2>::value;

    static constexpr int dbv_sn1 = MetaHelpers::elemStride<dbv, dn1>::value;
    static constexpr int dbv_sn2 = MetaHelpers::elemStride<dbv, dn2>::value;

    static constexpr int dbq_sn1 = MetaHelpers::elemStride<dbq, dn1>::value;
    static constexpr int dbq_sn2 = MetaHelpers::elemStride<dbq, dn2>::value;
        
    static void run(
        Real* DKK, Real* DVV, Real* DQQ,
        Real* DBK, Real* DBV, Real* DBQ,
        cudaStream_t stream) 
    {
        dim3 blocks(dn1::value, dn2::value);

        if (metal::same<dn1, dv>::value) blocks.x /= VectorType<Real>::ELEMS;
        if (metal::same<dn2, dv>::value) blocks.y /= VectorType<Real>::ELEMS;

        backwardAttentionInputBiases<Real,
            dn1, dn2, dr1, dr2,
            dv, dw,
            dkk_sn1, dkk_sn2, dkk_sr1, dkk_sr2,
            dvv_sn1, dvv_sn2, dvv_sr1, dvv_sr2,
            dqq_sn1, dqq_sn2, dqq_sr1, dqq_sr2,
            dbk_sn1, dbk_sn2,
            dbv_sn1, dbv_sn2,
            dbq_sn1, dbq_sn2
        >
            <<<blocks, 32, 0, stream>>>(
                DKK, DVV, DQQ,
                DBK, DBV, DBQ);
        CHECK(cudaPeekAtLastError());
    }
};