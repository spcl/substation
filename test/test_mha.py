import dace
from dace.codegen.compiler import CompiledSDFG
import numpy as np
import itertools
import sys

from substation.dtypes import *
from substation.attention import create_attn_forward_and_compile


def attn_forward_regression(q, k, v, wq, wk, wv, wo, scaler):
    from scipy.special import softmax as sm
    qq = np.einsum("phi,ibj->phbj", wq, q)
    kk = np.einsum("phi,ibk->phbk", wk, k)
    vv = np.einsum("phi,ibk->phbk", wv, v)

    beta = scaler * np.einsum("phbk,phbj->hbjk", kk, qq)
    alpha = sm(beta, axis=3)

    gamma = np.einsum("phbk,hbjk->phbj", vv, alpha)
    out = np.einsum("phi,phbj->bij", wo, gamma)

    return out


def mha(B: int, SN: int, SM: int, H: int, P: int, sdfg: dace.SDFG,
        compiled_sdfg: CompiledSDFG, iterations=100):
    N = H * P
    scaler = P ** -0.5

    Q = np.random.rand(N, B, SN).astype(np_dtype) + 1
    K = np.random.rand(N, B, SM).astype(np_dtype) + 1
    V = np.random.rand(N, B, SM).astype(np_dtype) + 1
    WQ = np.random.rand(P, H, N).astype(np_dtype) + 1
    WK = np.random.rand(P, H, N).astype(np_dtype) + 1
    WV = np.random.rand(P, H, N).astype(np_dtype) + 1
    WO = np.random.rand(P, H, N).astype(np_dtype) + 1

    OUT_SDFG = np.zeros((B, N, SN), dtype=np_dtype)
    OUT = attn_forward_regression(Q, K, V, WQ, WK, WV, WO, scaler)
    scaler_array = np.full(1, fill_value=scaler,
                           dtype=np_dtype)

    compiled_sdfg(
        Q=Q, K=K, V=V, WQ=WQ, WK=WK, WV=WV, WO=WO,
        OUT=OUT_SDFG, scaler=scaler_array,
        H=np.int32(H), P=np.int32(P), N=np.int32(N),
        SN=np.int32(SN),
        SM=np.int32(SM), B=np.int32(B),
        iters=np.int32(iterations))

    time_list = \
        sdfg.get_latest_report().entries[
            'cudaev_State main_state']
    time = np.median(time_list)

    print(
        "B = {B} SN = {SN} SM = {SM} H = {H} P = {P} N = {N} "
        "time = {time} ms".format(
            B=B, SN=SN, SM=SM, H=H, P=P, N=N, time=time))

    assert np.allclose(OUT, OUT_SDFG)


def test_mha(iterations=100):
    Bs = [2]
    SMs = [512]
    SNs = [512]
    Hs = [16]
    Ps = [64]

    compiled_sdfg = create_attn_forward_and_compile(iters=True)

    for B, SN, SM, H, P in itertools.product(Bs, SNs, SMs, Hs, Ps):
        mha(B, SN, SM, H, P, compiled_sdfg.sdfg, compiled_sdfg,
            iterations=iterations)


def test_mha_exhaustive():
    Bs = [1, 2, 4, 8]
    SNs = [512, 1024, 2048]
    SMs = [512, 1024, 2048]
    Hs = [16, 20, 24, 32]
    Ps = [64, 96, 128, 192, 384]

    compiled_sdfg = create_attn_forward_and_compile(iters=True)

    for SM in SMs:
        for B, SN, H, P in itertools.product(Bs, SNs, Hs, Ps):
            mha(B, SN, SM, H, P, compiled_sdfg.sdfg, compiled_sdfg)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        iterations = int(sys.argv[1])
        print('Running %d iterations' % iterations)
        test_mha(iterations)
    else:
        test_mha()
