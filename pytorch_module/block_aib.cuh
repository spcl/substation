#pragma once
#include "blocks.cuh"

template <typename Real,
    typename DN1, typename DN2, typename DB1, typename DB2,
    typename DV, typename DT,
    int WKK_SN1, int WKK_SN2, int WKK_SB1, int WKK_SB2,
    int WVV_SN1, int WVV_SN2, int WVV_SB1, int WVV_SB2,
    int WQQ_SN1, int WQQ_SN2, int WQQ_SB1, int WQQ_SB2,
    int BK_SN1, int BK_SN2,
    int BV_SN1, int BV_SN2,
    int BQ_SN1, int BQ_SN2,
    int KK_SN1, int KK_SN2, int KK_SB1, int KK_SB2,
    int VV_SN1, int VV_SN2, int VV_SB1, int VV_SB2,
    int QQ_SN1, int QQ_SN2, int QQ_SB1, int QQ_SB2
>
__global__ void attentionInputBiases(
    Real* WKK, Real* WVV, Real* WQQ,
    Real* BK, Real* BV, Real* BQ,
    Real* KK, Real* VV, Real* QQ)
{
    using Vec = VectorType<Real>;

    int iB1 = blockIdx.x;
    int iB2 = blockIdx.y;
    int iN1 = blockIdx.z / DN2::value;
    int iN2 = blockIdx.z % DN2::value;

    if (metal::same<DB1, DT>::value) iB1 = iB1 * blockDim.x + threadIdx.x;
    if (metal::same<DB2, DT>::value) iB2 = iB2 * blockDim.x + threadIdx.x;

    if (metal::same<DB1, DV>::value) iB1 *= Vec::ELEMS;
    if (metal::same<DB2, DV>::value) iB2 *= Vec::ELEMS;

    constexpr int WKK_SV = metal::same<DB1, DV>::value ? WKK_SB1 : WKK_SB2;

    constexpr int WVV_SV = metal::same<DB1, DV>::value ? WVV_SB1 : WVV_SB2;

    constexpr int WQQ_SV = metal::same<DB1, DV>::value ? WQQ_SB1 : WQQ_SB2;

    constexpr int KK_SV = metal::same<DB1, DV>::value ? KK_SB1 : KK_SB2;

    constexpr int VV_SV = metal::same<DB1, DV>::value ? VV_SB1 : VV_SB2;

    constexpr int QQ_SV = metal::same<DB1, DV>::value ? QQ_SB1 : QQ_SB2;

    Vec bk = Vec::fillall(*(BK + BK_SN1 * iN1 + BK_SN2 * iN2));
    Vec bv = Vec::fillall(*(BV + BV_SN1 * iN1 + BV_SN2 * iN2));
    Vec bq = Vec::fillall(*(BQ + BQ_SN1 * iN1 + BQ_SN2 * iN2));

    Vec wkk = Vec::load<WKK_SV>(WKK + WKK_SB1 * iB1 + WKK_SB2 * iB2 + WKK_SN1 * iN1 + WKK_SN2 * iN2);
    Vec kk = wkk + bk;
    kk.store<KK_SV>(KK + KK_SB1 * iB1 + KK_SB2 * iB2 + KK_SN1 * iN1 + KK_SN2 * iN2);

    Vec wvv = Vec::load<WVV_SV>(WVV + WVV_SB1 * iB1 + WVV_SB2 * iB2 + WVV_SN1 * iN1 + WVV_SN2 * iN2);
    Vec vv = wvv + bv;
    vv.store<VV_SV>(VV + VV_SB1 * iB1 + VV_SB2 * iB2 + VV_SN1 * iN1 + VV_SN2 * iN2);

    Vec wqq = Vec::load<WQQ_SV>(WQQ + WQQ_SB1 * iB1 + WQQ_SB2 * iB2 + WQQ_SN1 * iN1 + WQQ_SN2 * iN2);
    Vec qq = wqq + bq;
    qq.store<QQ_SV>(QQ + QQ_SB1 * iB1 + QQ_SB2 * iB2 + QQ_SN1 * iN1 + QQ_SN2 * iN2);
}

template <typename Real,
    typename dn1, typename dn2, typename db1, typename db2,
    typename dv, typename dt,
    typename wkk, typename wvv, typename wqq,
    typename bk, typename bv, typename bq,
    typename kk, typename vv, typename qq>
struct AttentionInputBiases {
    static constexpr int wkk_sn1 = MetaHelpers::elemStride<wkk, dn1>::value;
    static constexpr int wkk_sn2 = MetaHelpers::elemStride<wkk, dn2>::value;
    static constexpr int wkk_sb1 = MetaHelpers::elemStride<wkk, db1>::value;
    static constexpr int wkk_sb2 = MetaHelpers::elemStride<wkk, db2>::value;

    static constexpr int wvv_sn1 = MetaHelpers::elemStride<wvv, dn1>::value;
    static constexpr int wvv_sn2 = MetaHelpers::elemStride<wvv, dn2>::value;
    static constexpr int wvv_sb1 = MetaHelpers::elemStride<wvv, db1>::value;
    static constexpr int wvv_sb2 = MetaHelpers::elemStride<wvv, db2>::value;

    static constexpr int wqq_sn1 = MetaHelpers::elemStride<wqq, dn1>::value;
    static constexpr int wqq_sn2 = MetaHelpers::elemStride<wqq, dn2>::value;
    static constexpr int wqq_sb1 = MetaHelpers::elemStride<wqq, db1>::value;
    static constexpr int wqq_sb2 = MetaHelpers::elemStride<wqq, db2>::value;

    static constexpr int bk_sn1 = MetaHelpers::elemStride<bk, dn1>::value;
    static constexpr int bk_sn2 = MetaHelpers::elemStride<bk, dn2>::value;

    static constexpr int bv_sn1 = MetaHelpers::elemStride<bv, dn1>::value;
    static constexpr int bv_sn2 = MetaHelpers::elemStride<bv, dn2>::value;

    static constexpr int bq_sn1 = MetaHelpers::elemStride<bq, dn1>::value;
    static constexpr int bq_sn2 = MetaHelpers::elemStride<bq, dn2>::value;

    static constexpr int kk_sn1 = MetaHelpers::elemStride<kk, dn1>::value;
    static constexpr int kk_sn2 = MetaHelpers::elemStride<kk, dn2>::value;
    static constexpr int kk_sb1 = MetaHelpers::elemStride<kk, db1>::value;
    static constexpr int kk_sb2 = MetaHelpers::elemStride<kk, db2>::value;

    static constexpr int vv_sn1 = MetaHelpers::elemStride<vv, dn1>::value;
    static constexpr int vv_sn2 = MetaHelpers::elemStride<vv, dn2>::value;
    static constexpr int vv_sb1 = MetaHelpers::elemStride<vv, db1>::value;
    static constexpr int vv_sb2 = MetaHelpers::elemStride<vv, db2>::value;

    static constexpr int qq_sn1 = MetaHelpers::elemStride<qq, dn1>::value;
    static constexpr int qq_sn2 = MetaHelpers::elemStride<qq, dn2>::value;
    static constexpr int qq_sb1 = MetaHelpers::elemStride<qq, db1>::value;
    static constexpr int qq_sb2 = MetaHelpers::elemStride<qq, db2>::value;

        
    static constexpr int threads = std::min(metal::same<dv, dt>::value ? int(dt::value / VectorType<Real>::ELEMS) : int(dt::value), 128);

    static_assert(metal::same<dv, db1>::value || metal::same<dv, db2>::value);
    static_assert(metal::same<dt, db1>::value || metal::same<dt, db2>::value);

    static_assert(dt::value % threads == 0);

    static_assert(dv::value % VectorType<Real>::ELEMS == 0);

    static_assert(metal::distinct<dv, dt>::value || (dv::value % (threads * VectorType<Real>::ELEMS) == 0));

    static void run(
        Real* WKK, Real* WVV, Real* WQQ,
        Real* BK, Real* BV, Real* BQ,
        Real* KK, Real* VV, Real* QQ,
        cudaStream_t stream) 
    {
        dim3 blocks(db1::value, db2::value, dn1::value * dn2::value);

        if (metal::same<db1, dv>::value) blocks.x /= VectorType<Real>::ELEMS;
        if (metal::same<db2, dv>::value) blocks.y /= VectorType<Real>::ELEMS;

        if (metal::same<db1, dt>::value) blocks.x /= threads;
        if (metal::same<db2, dt>::value) blocks.y /= threads;

        attentionInputBiases<Real,
            dn1, dn2, db1, db2,
            dv, dt,
            wkk_sn1, wkk_sn2, wkk_sb1, wkk_sb2,
            wvv_sn1, wvv_sn2, wvv_sb1, wvv_sb2,
            wqq_sn1, wqq_sn2, wqq_sb1, wqq_sb2,
            bk_sn1, bk_sn2,
            bv_sn1, bv_sn2,
            bq_sn1, bq_sn2,
            kk_sn1, kk_sn2, kk_sb1, kk_sb2,
            vv_sn1, vv_sn2, vv_sb1, vv_sb2,
            qq_sn1, qq_sn2, qq_sb1, qq_sb2
        >
            <<<blocks, threads, 0, stream>>>(
                WKK, WVV, WQQ,
                BK, BV, BQ,
                KK, VV, QQ);
        CHECK(cudaPeekAtLastError());
    }
};

