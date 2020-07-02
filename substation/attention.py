#!/usr/bin/env python
# coding: utf-8
""" File containing multi-head attention implementations. """

# ### works with dace commit 19f7314c882466af0a8c259f06f46f8112ffd576

import os
import numpy as np
import dace

from substation.dtypes import *


def softmax(x, dim):
    """Perform softmax on x along dimension dim."""
    exps = np.exp(x - x.max(axis=dim, keepdims=True))
    return exps / exps.sum(axis=dim, keepdims=True)


def softmax_backward_data(y, dy, dim):
    """Backward-data of softmax.

    Note that this requires y as input, not x (for simplicity)!

    """
    elem_prod = y * dy
    sums = elem_prod.sum(axis=dim, keepdims=True)
    return elem_prod - y * sums


def attn_forward_numpy(q, k, v, wq, wk, wv, wo, in_b, out_b, scale, mask=None):
    """Multi-head attention on queries q, keys k, and values v.

    q, k, v have shape (batch, sequence length, embedding).
    wq, wk, wv have shape (heads, proj size, embedding).
    wo has shape (embedding, embedding).
    in_b is a bias for each linear projection for each head,
    shape (3, heads, proj size).
    out_b is a bias for wo, shape (embedding,).
    scale is a scalar.
    mask has shape (sequence length, sequence length).

    Returns the attention output and intermediates (for backprop).

    """
    # Note: Need to do either the batch or head dimension explicitly,
    # NumPy can't broadcast the outer dimensions together.
    out = np.empty((q.shape[0], q.shape[-2], q.shape[-1]))  # (B, L, E)
    # Need to save intermediates for backprop.
    proj_q_, proj_k_, proj_v_ = [], [], []
    concat_, scaled_scores_ = [], []
    for batch_idx in range(q.shape[0]):
        proj_q = np.matmul(q[batch_idx, :], wq.transpose(0, 2, 1))
        # Insert axis so bias broadcasts correctly.
        proj_q += np.expand_dims(in_b[0], axis=1)
        proj_q_.append(proj_q)
        proj_k = np.matmul(k[batch_idx, :], wk.transpose(0, 2, 1))
        proj_k += np.expand_dims(in_b[1, :], axis=1)
        proj_k_.append(proj_k)
        proj_v = np.matmul(v[batch_idx, :], wv.transpose(0, 2, 1))
        proj_v += np.expand_dims(in_b[2, :], axis=1)
        proj_v_.append(proj_v)
        scores = np.matmul(proj_q, proj_k.transpose(0, 2, 1))

        if mask is not None:
            scores += mask

        scaled_scores = softmax(scale * scores, dim=-1)
        scaled_scores_.append(scaled_scores)
        weighted_values = np.matmul(scaled_scores, proj_v)
        # Reshape out the head dimension to concatenate.
        # weighted_values is (heads, sequence length, proj size)
        # Convert to (sequence length, embedding).
        concat = weighted_values.transpose(1, 0, 2).reshape(
            (weighted_values.shape[1],
             weighted_values.shape[0] * weighted_values.shape[2]))
        concat_.append(concat)
        out[batch_idx, :] = np.matmul(concat, wo.T) + out_b
    return out, concat_, proj_q_, proj_k_, proj_v_, scaled_scores_


def attn_backward_numpy(q,
                        k,
                        v,
                        wq,
                        wk,
                        wv,
                        wo,
                        scale,
                        dy,
                        concat,
                        proj_q,
                        proj_k,
                        proj_v,
                        scaled_scores,
                        mask=None):
    """Backward data and weights for multi-head attention.

    Arguments are as in attn_forward_numpy.
    dy is the same shape as the output of attn_forward_numpy.

    This does both backward data and backward weights, since the same
    intermediate results are needed for both.

    Returns dq, dk, dv, dwq, dwk, dwv, dwo (in that order).

    """
    dq = np.zeros(q.shape)
    dk = np.zeros(k.shape)
    dv = np.zeros(v.shape)
    dwq = np.zeros(wq.shape)
    dwk = np.zeros(wk.shape)
    dwv = np.zeros(wv.shape)
    dwo = np.zeros(wo.shape)
    din_b = np.zeros((3, wq.shape[0], wq.shape[1]))
    dout_b = np.zeros(wo.shape[0])
    for batch_idx in range(q.shape[0]):
        # Backward data for output projection.
        dconcat = np.matmul(dy[batch_idx, :], wo)
        # Backward weights for wo.
        dwo += np.matmul(dy[batch_idx, :].T, concat[batch_idx])
        # Backward weights for out_b.
        dout_b += dy[batch_idx, :].sum(axis=0)
        # Reshape dconcat to be dweighted_values (for this batch).
        # It is currently (sequence length, embedding).
        # Convert to (heads, sequence length, proj size) by reshaping to
        # (sequence length, heads, proj size) and transposing.
        dweighted_values = dconcat.reshape(
            (q.shape[1], wq.shape[0], wq.shape[1])).transpose(1, 0, 2)
        # Backward data for scaled scores.
        dscaled_scores = np.matmul(dweighted_values,
                                   proj_v[batch_idx].transpose(0, 2, 1))
        # Backward data for proj_v.
        dproj_v = np.matmul(scaled_scores[batch_idx].transpose(0, 2, 1),
                            dweighted_values)
        # Backward data for scores.
        dscores = softmax_backward_data(scaled_scores[batch_idx],
                                        scale * dscaled_scores, -1)
        # Backward data for proj_k.
        dproj_k = np.matmul(dscores.transpose(0, 2, 1), proj_q[batch_idx])
        # Backward data for proj_q.
        dproj_q = np.matmul(dscores, proj_k[batch_idx])
        # Backward data for v.
        dv[batch_idx, :] = np.matmul(dproj_v,
                                     wv).sum(axis=0)  # Sum over heads.
        # Backward weights for wv.
        dwv += np.matmul(dproj_v.transpose(0, 2, 1), v[batch_idx, :])
        # Backward weights for in_b for values.
        din_b[2, :] += dproj_v.sum(axis=1)
        # Backward data for k.
        dk[batch_idx, :] = np.matmul(dproj_k,
                                     wk).sum(axis=0)  # Sum over heads.
        # Backward weights for wk.
        dwk += np.matmul(dproj_k.transpose(0, 2, 1), k[batch_idx, :])
        # Backward weights for in_b for keys.
        din_b[1, :] += dproj_k.sum(axis=1)
        # Backward data for q.
        dq[batch_idx, :] = np.matmul(dproj_q,
                                     wq).sum(axis=0)  # Sum over heads.
        # Backward weights for wq.
        dwq += np.matmul(dproj_q.transpose(0, 2, 1), q[batch_idx, :])
        # Backward weights for in_b for queries.
        din_b[0, :] += dproj_q.sum(axis=1)
    # Handle when inputs are the same (and we hence need to sum gradient
    # contributions from every input).
    # Could avoid by specializing appropriately.
    dq_all = np.copy(dq)
    dk_all = np.copy(dk)
    dv_all = np.copy(dv)
    if q is k:
        dq_all += dk
        dk_all += dq
    if q is v:
        dq_all += dv
        dv_all += dq
    if k is v:
        dk_all += dv
        dv_all += dk
    return dq_all, dk_all, dv_all, dwq, dwk, dwv, dwo, din_b, dout_b


attn_mask_forward_numpy = attn_forward_numpy


def attn_forward_sdfg(copy_to_device=True,
                      iters=False,
                      mask=False) -> dace.SDFG:
    H = dace.symbol('H')
    P = dace.symbol('P')
    N = dace.symbol('N')
    SN = dace.symbol('SN')
    SM = dace.symbol('SM')
    B = dace.symbol('B')

    sdfg = dace.SDFG('attn_forward')

    sdfg.add_array('scaler', shape=[1], dtype=dace_dtype)
    if copy_to_device:
        sdfg.add_array('Q', shape=[B, SN, N], dtype=dace_dtype)
        sdfg.add_array('K', shape=[B, SM, N], dtype=dace_dtype)
        sdfg.add_array('V', shape=[B, SM, N], dtype=dace_dtype)
        sdfg.add_array('OUT', shape=[B, SN, N], dtype=dace_dtype)
        sdfg.add_array('WQ', shape=[P, H, N], dtype=dace_dtype)
        sdfg.add_array('WK', shape=[P, H, N], dtype=dace_dtype)
        sdfg.add_array('WV', shape=[P, H, N], dtype=dace_dtype)
        sdfg.add_array('WO', shape=[P, H, N], dtype=dace_dtype)
        if mask:
            sdfg.add_array('MASK', shape=[SN, SM], dtype=dace_dtype)

    sdfg.add_array('gWQ', [P, H, N],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=copy_to_device)
    sdfg.add_array('gQ', [B, SN, N],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=copy_to_device)
    sdfg.add_array('gQ1', [N, B, SN],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=True)
    sdfg.add_array('gWK', [P, H, N],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=copy_to_device)
    sdfg.add_array('gK', [B, SM, N],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=copy_to_device)
    sdfg.add_array('gK1', [N, B, SM],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=True)
    sdfg.add_array('gWV', [P, H, N],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=copy_to_device)
    sdfg.add_array('gV', [B, SM, N],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=copy_to_device)
    sdfg.add_array('gV1', [N, B, SM],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=True)
    sdfg.add_array('gWO', [P, H, N],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=copy_to_device)
    sdfg.add_array('gOUT', [B, SN, N],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=copy_to_device)
    sdfg.add_array('gOUT1', [B, N, SN],
                   dace_dtype,
                   dace.StorageType.GPU_Global,
                   toplevel=True,
                   transient=True)
    if mask:
        sdfg.add_array('gMASK',
                       shape=[SN, SM],
                       dtype=dace_dtype,
                       storage=dace.StorageType.GPU_Global,
                       toplevel=True,
                       transient=copy_to_device)

    # Transients
    sdfg.add_transient('gBETA',
                       shape=[H, B, SN, SM],
                       dtype=dace_dtype,
                       storage=dace.StorageType.GPU_Global,
                       toplevel=True)
    if mask:
        sdfg.add_transient('gBETA_mask',
                           shape=[H, B, SN, SM],
                           dtype=dace_dtype,
                           storage=dace.StorageType.GPU_Global,
                           toplevel=True)
    sdfg.add_transient('gALPHA',
                       shape=[H, B, SN, SM],
                       dtype=dace_dtype,
                       storage=dace.StorageType.GPU_Global,
                       toplevel=True)
    sdfg.add_transient('gQQ', [P, H, B, SN],
                       dace_dtype,
                       dace.StorageType.GPU_Global,
                       toplevel=True)
    sdfg.add_transient('gKK', [P, H, B, SM],
                       dace_dtype,
                       dace.StorageType.GPU_Global,
                       toplevel=True)
    sdfg.add_transient('gVV', [P, H, B, SM],
                       dace_dtype,
                       dace.StorageType.GPU_Global,
                       toplevel=True)
    sdfg.add_transient('gGAMMA', [P, H, B, SN],
                       dace_dtype,
                       dace.StorageType.GPU_Global,
                       toplevel=True)

    # setup states
    init_state = sdfg.add_state('init_state')
    end_state = sdfg.add_state('end_state')
    state = sdfg.add_state('main_state')
    state.instrument = dace.InstrumentationType.CUDA_Events

    if iters:
        guard_state = sdfg.add_state('guard_state')
        sdfg.add_edge(init_state, guard_state,
                      dace.InterstateEdge(assignments=dict(it='0')))
        sdfg.add_edge(guard_state, state, dace.InterstateEdge('it < iters'))
        sdfg.add_edge(guard_state, end_state,
                      dace.InterstateEdge('it >= iters'))
        sdfg.add_edge(state, guard_state,
                      dace.InterstateEdge(assignments=dict(it='it+1')))
    else:
        sdfg.add_edge(init_state, state, dace.InterstateEdge())
        sdfg.add_edge(state, end_state, dace.InterstateEdge())

    # copy input on gpu in init_state
    if copy_to_device:
        init_Q = init_state.add_read('Q')
        init_gQ = init_state.add_write('gQ')
        init_state.add_edge(init_Q, None, init_gQ, None,
                            dace.Memlet.from_array('gQ', sdfg.arrays['gQ']))

        init_K = init_state.add_read('K')
        init_gK = init_state.add_write('gK')
        init_state.add_edge(init_K, None, init_gK, None,
                            dace.Memlet.from_array('gK', sdfg.arrays['gK']))

        init_V = init_state.add_read('V')
        init_gV = init_state.add_write('gV')
        init_state.add_edge(init_V, None, init_gV, None,
                            dace.Memlet.from_array('gV', sdfg.arrays['gV']))

        init_WQ = init_state.add_read('WQ')
        init_gWQ = init_state.add_write('gWQ')
        init_state.add_edge(init_WQ, None, init_gWQ, None,
                            dace.Memlet.simple('gWQ', '0:P, 0:H, 0:N'))

        init_WK = init_state.add_read('WK')
        init_gWK = init_state.add_write('gWK')
        init_state.add_edge(init_WK, None, init_gWK, None,
                            dace.Memlet.simple('gWK', '0:P, 0:H, 0:N'))

        init_WV = init_state.add_read('WV')
        init_gWV = init_state.add_write('gWV')
        init_state.add_edge(init_WV, None, init_gWV, None,
                            dace.Memlet.simple('gWV', '0:P, 0:H, 0:N'))

        init_WO = init_state.add_read('WO')
        init_gWO = init_state.add_write('gWO')
        init_state.add_edge(init_WO, None, init_gWO, None,
                            dace.Memlet.simple('gWO', '0:P, 0:H, 0:N'))

        if mask:
            init_mask = init_state.add_read('MASK')
            init_gmask = init_state.add_write('gMASK')
            init_state.add_nedge(
                init_mask, init_gmask,
                dace.Memlet.from_array(init_gmask.data,
                                       sdfg.arrays[init_gmask.data]))

        # copy output from gpu in last_state
        end_gOUT = end_state.add_read('gOUT')
        end_OUT = end_state.add_write('OUT')
        end_state.add_edge(end_gOUT, None, end_OUT, None,
                           dace.Memlet.from_array('OUT', sdfg.arrays['OUT']))

    # Data layout transformations
    gqin = state.add_read('gQ')
    gQ = state.add_access('gQ1')
    gkin = state.add_read('gK')
    gK = state.add_access('gK1')
    gvin = state.add_read('gV')
    gV = state.add_access('gV1')

    state.add_mapped_tasklet('transform_q',
                             dict(i='0:N', j='0:B', k='0:SN'),
                             dict(inp=dace.Memlet.simple('gQ', 'j, k, i')),
                             'out = inp',
                             dict(out=dace.Memlet.simple('gQ1', 'i, j, k')),
                             schedule=dace.ScheduleType.GPU_Device,
                             external_edges=True,
                             input_nodes=dict(gQ=gqin),
                             output_nodes=dict(gQ1=gQ))
    state.add_mapped_tasklet('transform_k',
                             dict(i='0:N', j='0:B', k='0:SM'),
                             dict(inp=dace.Memlet.simple('gK', 'j, k, i')),
                             'out = inp',
                             dict(out=dace.Memlet.simple('gK1', 'i, j, k')),
                             schedule=dace.ScheduleType.GPU_Device,
                             external_edges=True,
                             input_nodes=dict(gK=gkin),
                             output_nodes=dict(gK1=gK))
    state.add_mapped_tasklet('transform_v',
                             dict(i='0:N', j='0:B', k='0:SM'),
                             dict(inp=dace.Memlet.simple('gV', 'j, k, i')),
                             'out = inp',
                             dict(out=dace.Memlet.simple('gV1', 'i, j, k')),
                             schedule=dace.ScheduleType.GPU_Device,
                             external_edges=True,
                             input_nodes=dict(gV=gvin),
                             output_nodes=dict(gV1=gV))

    ##### qq #####
    gWQ = state.add_access('gWQ')
    gQQ = state.add_access('gQQ')

    qq_tasklet = state.add_tasklet(name="qq_tasklet",
                                   inputs={'wq', 'q'},
                                   outputs={'qq'},
                                   code='''
        cublasSetStream(handle, __dace_current_stream);
        {c_dtype} alpha = 1.0, beta = 0.0;
        {cublas_gemm}(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    B * SN, P * H, N,
                    &alpha,
                    q, B * SN,
                    wq, N, 
                    &beta,
                    qq, B * SN);
        '''.format(c_dtype=c_dtype, cublas_gemm=cublas_gemm),
                                   code_global='''
        #include <cublas_v2.h>
        cublasHandle_t handle;
        ''',
                                   code_init='cublasCreate(&handle);',
                                   code_exit='cublasDestroy(handle);',
                                   language=dace.Language.CPP)

    state.add_edge(gWQ, None, qq_tasklet, 'wq',
                   dace.Memlet.simple('gWQ', '0:P, 0:H, 0:N'))
    state.add_edge(gQ, None, qq_tasklet, 'q',
                   dace.Memlet.simple(gQ.data, '0:N, 0:B, 0:SN'))
    state.add_edge(qq_tasklet, 'qq', gQQ, None,
                   dace.Memlet.simple('gQQ', '0:P, 0:H, 0:B, 0:SN'))

    ##### kk #####

    gWK = state.add_access('gWK')
    gKK = state.add_access('gKK')

    kk_tasklet = state.add_tasklet(name="kk_tasklet",
                                   inputs={'wk', 'k'},
                                   outputs={'kk'},
                                   code='''
        cublasSetStream(handle, __dace_current_stream);
        {c_dtype} alpha = 1.0, beta = 0.0;
        {cublas_gemm}(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    B * SM, P * H, N,
                    &alpha,
                    k, B * SM,
                    wk, N, 
                    &beta,
                    kk, B * SM);
        '''.format(c_dtype=c_dtype, cublas_gemm=cublas_gemm),
                                   language=dace.Language.CPP)

    state.add_edge(gWK, None, kk_tasklet, 'wk',
                   dace.Memlet.simple('gWK', '0:P, 0:H, 0:N'))
    state.add_edge(gK, None, kk_tasklet, 'k',
                   dace.Memlet.simple(gK.data, '0:N, 0:B, 0:SM'))
    state.add_edge(kk_tasklet, 'kk', gKK, None,
                   dace.Memlet.simple('gKK', '0:P, 0:H, 0:B, 0:SM'))

    ##### vv #####

    gWV = state.add_access('gWV')
    gVV = state.add_access('gVV')

    vv_tasklet = state.add_tasklet(name="vv_tasklet",
                                   inputs={'wv', 'v'},
                                   outputs={'vv'},
                                   code='''
        cublasSetStream(handle, __dace_current_stream);
        {c_dtype} alpha = 1.0, beta = 0.0;
        {cublas_gemm}(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    B * SM, P * H, N,
                    &alpha,
                    v, B * SM,
                    wv, N, 
                    &beta,
                    vv, B * SM);
        '''.format(c_dtype=c_dtype, cublas_gemm=cublas_gemm),
                                   language=dace.Language.CPP)

    state.add_edge(gWV, None, vv_tasklet, 'wv',
                   dace.Memlet.simple('gWV', '0:P, 0:H, 0:N'))
    state.add_edge(gV, None, vv_tasklet, 'v',
                   dace.Memlet.simple(gV.data, '0:N, 0:B, 0:SM'))
    state.add_edge(vv_tasklet, 'vv', gVV, None,
                   dace.Memlet.simple('gVV', '0:P, 0:H, 0:B, 0:SM'))

    ##### beta #####

    scaler = state.add_read('scaler')
    gBETA = state.add_access('gBETA')
    gALPHA = state.add_access('gALPHA')

    beta_tasklet = state.add_tasklet(name="beta_tasklet",
                                     inputs={'sc', 'kk', 'qq'},
                                     outputs={'beta'},
                                     code='''
        cublasSetStream(handle, __dace_current_stream);
        {c_dtype} beta_unused = 0.0;
        {cublas_gemm}StridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    SM, SN, P,
                    &sc,
                    kk, H * B * SM, SM,
                    qq, H * B * SN, SN,
                    &beta_unused,
                    beta, SM, SN * SM,
                    H * B);
        '''.format(c_dtype=c_dtype, cublas_gemm=cublas_gemm),
                                     language=dace.Language.CPP)

    state.add_edge(gKK, None, beta_tasklet, 'kk',
                   dace.Memlet.simple('gKK', '0:P, 0:H, 0:B, 0:SM'))
    state.add_edge(gQQ, None, beta_tasklet, 'qq',
                   dace.Memlet.simple('gQQ', '0:P, 0:H, 0:B, 0:SN'))
    state.add_edge(scaler, None, beta_tasklet, 'sc',
                   dace.Memlet.simple('scaler', '0:1'))
    state.add_edge(beta_tasklet, 'beta', gBETA, None,
                   dace.Memlet.simple('gBETA', '0:H, 0:B, 0:SN, 0:SM'))

    # Mask
    if mask:
        gMASK = state.add_read('gMASK')
        gBETA_beforemask = gBETA
        gBETA = state.add_access('gBETA_mask')

        state.add_mapped_tasklet(
            'mask',
            dict(i='0:H', j='0:B', k='0:SN', l='0:SM'),
            dict(inp=dace.Memlet.simple('gBETA', 'i, j, k, l'),
                 msk=dace.Memlet.simple('gMASK', 'k, l')),
            'out = inp + msk',
            dict(out=dace.Memlet.simple('gBETA_mask', 'i, j, k, l')),
            schedule=dace.ScheduleType.GPU_Device,
            external_edges=True,
            input_nodes=dict(gBETA=gBETA_beforemask, gMASK=gMASK),
            output_nodes=dict(gBETA_mask=gBETA))

    ##### softmax entry #####

    softmax_entry, softmax_exit = state.add_map(
        "softmax_map",
        ndrange={
            "h": "0:H",
            "b": "0:B",
            "j": "0:SN"
        },
        schedule=dace.dtypes.ScheduleType.GPU_Device)

    softmax_block_entry, softmax_block_exit = state.add_map(
        "softmax_map_block",
        ndrange={"warp_thread": "0:32"},
        schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)

    softmax_entry.add_in_connector("IN_gBETA")
    softmax_entry.add_out_connector("OUT_gBETA")

    softmax_block_entry.add_in_connector("IN_gBETA")
    softmax_block_entry.add_out_connector("OUT_gBETA")

    state.add_edge(gBETA, None, softmax_entry, "IN_gBETA",
                   dace.memlet.Memlet.simple(gBETA.data, "0:H,0:B,0:SN,0:SM"))
    state.add_edge(softmax_entry, 'OUT_gBETA', softmax_block_entry, "IN_gBETA",
                   dace.memlet.Memlet.simple(gBETA.data, "h,b,j,0:SM"))

    ##### softmax tasklet #####

    softmax_tasklet = state.add_tasklet(name="softmax_tasklet",
                                        inputs={'beta'},
                                        outputs={'alpha'},
                                        code='''
            typedef cub::WarpReduce<{c_dtype}> WarpReduce;
            __shared__ typename WarpReduce::TempStorage temp_storage;
            
            int elems_per_thread = SM / 32 + (warp_thread < SM % 32);
            
            {c_dtype} local_max = -1.0e15;
            for (int k = warp_thread; k < elems_per_thread * 32 + warp_thread; k += 32) {{
                local_max = {max_func}(local_max, beta[k]);
            }}
            local_max = WarpReduce(temp_storage).Reduce(local_max, cub::Max());
            __shared__ {c_dtype} sm_max;
            if (warp_thread == 0) sm_max = local_max;
            __syncthreads();
            
            {c_dtype} local_sum = 0.;
            for (int k = warp_thread; k < elems_per_thread * 32 + warp_thread; k += 32) {{
                alpha[k] = {exp_func}(beta[k] - sm_max);
                local_sum += alpha[k];
            }}
            local_sum = WarpReduce(temp_storage).Sum(local_sum);
            __shared__ {c_dtype} sm_sum;
            if (warp_thread == 0) sm_sum = local_sum;
            __syncthreads();
            
            for (int k = warp_thread; k < elems_per_thread * 32 + warp_thread; k += 32) {{
                alpha[k] /= sm_sum;
            }}
        '''.format(c_dtype=c_dtype, exp_func=exp_func, max_func=max_func),
                                        language=dace.Language.CPP)

    state.add_edge(softmax_block_entry, 'OUT_gBETA', softmax_tasklet, 'beta',
                   dace.memlet.Memlet.simple(gBETA.data, "h,b,j,0:SM"))

    state.add_edge(softmax_tasklet, 'alpha', softmax_block_exit, 'IN_gALPHA',
                   dace.memlet.Memlet.simple(gALPHA.data, "h,b,j,0:SM"))

    ##### softmax exit #####

    softmax_block_exit.add_in_connector("IN_gALPHA")
    softmax_block_exit.add_out_connector("OUT_gALPHA")

    softmax_exit.add_in_connector("IN_gALPHA")
    softmax_exit.add_out_connector("OUT_gALPHA")

    state.add_edge(softmax_block_exit, "OUT_gALPHA", softmax_exit, 'IN_gALPHA',
                   dace.memlet.Memlet.simple(gALPHA.data, "h,b,j,0:SM"))
    state.add_edge(softmax_exit, "OUT_gALPHA", gALPHA, None,
                   dace.memlet.Memlet.simple(gALPHA.data, "0:H,0:B,0:SN,0:SM"))

    ##### gamma #####

    gGAMMA = state.add_access('gGAMMA')

    gamma_tasklet = state.add_tasklet(name="gamma_tasklet",
                                      inputs={'alpha', 'vv'},
                                      outputs={'gamma'},
                                      code='''
        cublasSetStream(handle, __dace_current_stream);
        {c_dtype} alpha_unused = 1.0, beta_unused = 0.0;
        {cublas_gemm}StridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    SN, P, SM,
                    &alpha_unused,
                    alpha, SM, SN * SM,
                    vv, H * B * SM, SM,
                    &beta_unused,
                    gamma, H * B * SN, SN,
                    H * B);
        '''.format(c_dtype=c_dtype, cublas_gemm=cublas_gemm),
                                      language=dace.Language.CPP)

    state.add_edge(gALPHA, None, gamma_tasklet, 'alpha',
                   dace.Memlet.simple('gALPHA', '0:H, 0:B, 0:SN, 0:SM'))
    state.add_edge(gVV, None, gamma_tasklet, 'vv',
                   dace.Memlet.simple('gVV', '0:P, 0:H, 0:B, 0:SM'))
    state.add_edge(gamma_tasklet, 'gamma', gGAMMA, None,
                   dace.Memlet.simple('gGAMMA', '0:P, 0:H, 0:B, 0:SN'))

    ##### omega and out #####

    gWO = state.add_access('gWO')

    gOUT = state.add_access('gOUT1')
    gout_out = state.add_write('gOUT')

    out_tasklet = state.add_tasklet(name="out_tasklet",
                                    inputs={'gamma', 'wo'},
                                    outputs={'out'},
                                    code='''
        cublasSetStream(handle, __dace_current_stream);
        {c_dtype} alpha_unused = 1.0, beta_unused = 0.0;
        {cublas_gemm}StridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    SN, N, P * H,
                    &alpha_unused,
                    gamma, B * SN, SN,
                    wo, N, 0,
                    &beta_unused,
                    out, SN, N * SN,
                    B);
        '''.format(c_dtype=c_dtype, cublas_gemm=cublas_gemm),
                                    language=dace.Language.CPP)

    state.add_edge(gGAMMA, None, out_tasklet, 'gamma',
                   dace.Memlet.simple('gGAMMA', '0:P, 0:H, 0:B, 0:SN'))
    state.add_edge(gWO, None, out_tasklet, 'wo',
                   dace.Memlet.simple('gWO', '0:P, 0:H, 0:N'))
    state.add_edge(out_tasklet, 'out', gOUT, None,
                   dace.Memlet.simple(gOUT.data, '0:B, 0:N, 0:SN'))

    state.add_mapped_tasklet('transform_out',
                             dict(i='0:B', j='0:SN', k='0:N'),
                             dict(inp=dace.Memlet.simple('gOUT1', 'i, k, j')),
                             'out = inp',
                             dict(out=dace.Memlet.simple('gOUT', 'i, j, k')),
                             schedule=dace.ScheduleType.GPU_Device,
                             external_edges=True,
                             input_nodes=dict(gOUT1=gOUT),
                             output_nodes=dict(gOUT=gout_out))

    return sdfg


def create_einsum(state, map_ranges, code, inputs, outputs=[], wcr_outputs=[]):
    inpdict = {access_node.data: access_node for access_node, _ in inputs}
    outdict = {
        access_node.data: access_node
        for access_node, _ in (outputs + wcr_outputs)
    }

    input_memlets = {(access_node.data + "_inp"):
                     dace.Memlet.simple(access_node.data, access_range)
                     for access_node, access_range in inputs}

    output_memlets = {(access_node.data + "_out"):
                      dace.Memlet.simple(access_node.data, access_range)
                      for access_node, access_range in outputs}

    wcr_output_memlets = {(access_node.data + "_out"):
                          dace.Memlet.simple(access_node.data,
                                             access_range,
                                             wcr_str='lambda x, y: x + y')
                          for access_node, access_range in wcr_outputs}

    state.add_mapped_tasklet(input_nodes=inpdict,
                             output_nodes=outdict,
                             name="einsum_tasklet",
                             map_ranges=map_ranges,
                             inputs=input_memlets,
                             code=code,
                             outputs={
                                 **output_memlets,
                                 **wcr_output_memlets
                             },
                             external_edges=True)


def create_zero_initialization(init_state, array_name):
    sdfg = init_state.parent
    array_shape = sdfg.arrays[array_name].shape

    array_access_node = init_state.add_write(array_name)

    indices = ["i" + str(k) for k, _ in enumerate(array_shape)]

    init_state.add_mapped_tasklet(
        input_nodes={},
        output_nodes={array_name: array_access_node},
        name=(array_name + "_init_tasklet"),
        map_ranges={k: "0:" + str(v)
                    for k, v in zip(indices, array_shape)},
        inputs={},
        code='val = 0',
        outputs=dict(
            val=dace.Memlet.simple(array_access_node.data, ",".join(indices))),
        external_edges=True)


def create_attn_backward_sdfg(create_init_state=True, need_dk_dv=True):
    B = dace.symbol('B')
    H = dace.symbol('H')
    P = dace.symbol('P')
    N = dace.symbol('N')
    SN = dace.symbol('SN')
    SM = dace.symbol('SM')

    sdfg = dace.SDFG('attn_backward')

    sdfg.add_scalar('scaler', dtype=dace_dtype)
    sdfg.add_array('Q', shape=[B, SN, N], dtype=dace_dtype)
    sdfg.add_array('K', shape=[B, SM, N], dtype=dace_dtype)
    sdfg.add_array('V', shape=[B, SM, N], dtype=dace_dtype)
    sdfg.add_array('WQ', shape=[P, H, N], dtype=dace_dtype)
    if need_dk_dv:
        sdfg.add_array('WK', shape=[P, H, N], dtype=dace_dtype)
        sdfg.add_array('WV', shape=[P, H, N], dtype=dace_dtype)
    sdfg.add_array('WO', shape=[P, H, N], dtype=dace_dtype)
    sdfg.add_array('DOUT', shape=[B, SN, N], dtype=dace_dtype)

    sdfg.add_array('DWO', shape=[P, H, N], dtype=dace_dtype)
    sdfg.add_array('DWV', shape=[P, H, N], dtype=dace_dtype)
    sdfg.add_array('DWQ', shape=[P, H, N], dtype=dace_dtype)
    sdfg.add_array('DWK', shape=[P, H, N], dtype=dace_dtype)
    sdfg.add_array('DQ', shape=[B, SN, N], dtype=dace_dtype)
    if need_dk_dv:
        sdfg.add_array('DK', shape=[B, SM, N], dtype=dace_dtype)
        sdfg.add_array('DV', shape=[B, SM, N], dtype=dace_dtype)

    sdfg.add_array('QQ', shape=[B, H, SN, P], dtype=dace_dtype)
    sdfg.add_array('KK', shape=[B, H, SM, P], dtype=dace_dtype)
    sdfg.add_array('VV', shape=[B, H, SM, P], dtype=dace_dtype)
    sdfg.add_array('ALPHA', shape=[B, H, SM, SN], dtype=dace_dtype)
    sdfg.add_array('GAMMA', shape=[B, SN, H, P], dtype=dace_dtype)

    sdfg.add_transient('DGAMMA', shape=[B, H, P, SN], dtype=dace_dtype)
    sdfg.add_transient('DALPHA', shape=[B, H, SM, SN], dtype=dace_dtype)
    sdfg.add_transient('DBETA', shape=[B, H, SM, SN], dtype=dace_dtype)
    sdfg.add_transient('DQQ', shape=[B, H, P, SN], dtype=dace_dtype)
    sdfg.add_transient('DKK', shape=[B, H, P, SM], dtype=dace_dtype)
    sdfg.add_transient('DVV', shape=[B, H, P, SM], dtype=dace_dtype)

    init_state = sdfg.add_state("attention_init_state")
    state = sdfg.add_state("attention_backward_state")

    sdfg.add_edge(init_state, state, dace.InterstateEdge())

    if create_init_state:
        arrays = [
            'DWO', 'DVV', 'DGAMMA', 'DWV', 'DALPHA', 'DBETA', 'DQQ', 'DKK',
            'DWK', 'DQ', 'DWQ'
        ]
        if need_dk_dv:
            arrays.extend(['DK', 'DV'])
        for arr in arrays:
            create_zero_initialization(init_state, arr)

    scaler = state.add_read('scaler')
    Q = state.add_read('Q')
    K = state.add_read('K')
    V = state.add_read('V')
    WQ = state.add_read('WQ')
    WO = state.add_read('WO')
    DOUT = state.add_read('DOUT')

    DWO = state.add_write('DWO')
    DWV = state.add_write('DWV')
    DWQ = state.add_write('DWQ')
    DWK = state.add_write('DWK')
    DQ = state.add_write('DQ')

    DGAMMA = state.add_access('DGAMMA')
    DVV = state.add_access('DVV')
    DALPHA = state.add_access('DALPHA')
    DBETA = state.add_access('DBETA')
    DQQ = state.add_access('DQQ')
    DKK = state.add_access('DKK')

    QQ = state.add_access('QQ')
    KK = state.add_access('KK')
    VV = state.add_access('VV')
    ALPHA = state.add_read('ALPHA')
    GAMMA = state.add_read('GAMMA')

    ##### dwo #####

    create_einsum(state,
                  map_ranges=dict(b='0:B', h='0:H', p='0:P', i='0:N',
                                  j='0:SN'),
                  code='DWO_out = GAMMA_inp * DOUT_inp',
                  inputs=[(GAMMA, 'b,j,h,p'), (DOUT, 'b,j,i')],
                  wcr_outputs=[(DWO, 'p,h,i')])

    ##### dgamma #####

    create_einsum(state,
                  map_ranges=dict(b='0:B', h='0:H', p='0:P', i='0:N',
                                  j='0:SN'),
                  code='DGAMMA_out = WO_inp * DOUT_inp',
                  inputs=[(WO, 'p,h,i'), (DOUT, 'b,j,i')],
                  wcr_outputs=[(DGAMMA, 'b,h,p,j')])

    ##### dvv #####
    create_einsum(state,
                  map_ranges=dict(b='0:B',
                                  h='0:H',
                                  p='0:P',
                                  k='0:SM',
                                  j='0:SN'),
                  code='DVV_out = ALPHA_inp * DGAMMA_inp',
                  inputs=[(ALPHA, 'b,h,k,j'), (DGAMMA, 'b,h,p,j')],
                  wcr_outputs=[(DVV, 'b,h,p,k')])

    ##### dwv #####

    create_einsum(state,
                  map_ranges=dict(b='0:B', h='0:H', p='0:P', k='0:SM',
                                  i='0:N'),
                  code='DWV_out = V_inp * DVV_inp',
                  inputs=[(V, 'b,k,i'), (DVV, 'b,h,p,k')],
                  wcr_outputs=[(DWV, 'p,h,i')])

    ##### dalpha #####

    create_einsum(state,
                  map_ranges=dict(b='0:B',
                                  h='0:H',
                                  p='0:P',
                                  k='0:SM',
                                  j='0:SN'),
                  code='DALPHA_out = VV_inp * DGAMMA_inp',
                  inputs=[(VV, 'b,h,k,p'), (DGAMMA, 'b,h,p,j')],
                  wcr_outputs=[(DALPHA, 'b,h,k,j')])

    ##### dbeta #####

    state.add_mapped_tasklet(
        input_nodes=dict(ALPHA=ALPHA, DALPHA=DALPHA),
        output_nodes=dict(DBETA=DBETA),
        name="einsum_tasklet",
        map_ranges=dict(b='0:B', h='0:H', k='0:SM', m='0:SM', j='0:SN'),
        inputs=dict(ALPHA1_inp=dace.Memlet.simple(ALPHA.data, 'b,h,k,j'),
                    ALPHA2_inp=dace.Memlet.simple(ALPHA.data, 'b,h,m,j'),
                    DALPHA_inp=dace.Memlet.simple(DALPHA.data, 'b,h,m,j')),
        code=
        'DBETA_out = ALPHA1_inp * ((1. if k == m else 0.) - ALPHA2_inp) * DALPHA_inp',
        outputs=dict(DBETA_out=dace.Memlet.simple(
            DBETA.data, 'b,h,k,j', wcr_str='lambda x, y: x + y')),
        external_edges=True)

    ##### dqq #####

    create_einsum(state,
                  map_ranges=dict(b='0:B',
                                  h='0:H',
                                  p='0:P',
                                  k='0:SM',
                                  j='0:SN'),
                  code='DQQ_out = scaler_inp * KK_inp * DBETA_inp',
                  inputs=[(scaler, '0'), (KK, 'b,h,k,p'), (DBETA, 'b,h,k,j')],
                  wcr_outputs=[(DQQ, 'b,h,p,j')])

    ##### dwq #####

    create_einsum(state,
                  map_ranges=dict(b='0:B', h='0:H', p='0:P', i='0:N',
                                  j='0:SN'),
                  code='DWQ_out = Q_inp * DQQ_inp',
                  inputs=[(Q, 'b,j,i'), (DQQ, 'b,h,p,j')],
                  wcr_outputs=[(DWQ, 'p,h,i')])

    ##### dkk #####

    create_einsum(state,
                  map_ranges=dict(b='0:B',
                                  h='0:H',
                                  p='0:P',
                                  k='0:SM',
                                  j='0:SN'),
                  code='DKK_out = scaler_inp * QQ_inp * DBETA_inp',
                  inputs=[(scaler, '0'), (QQ, 'b,h,j,p'), (DBETA, 'b,h,k,j')],
                  wcr_outputs=[(DKK, 'b,h,p,k')])

    ##### dwk #####

    create_einsum(state,
                  map_ranges=dict(b='0:B', h='0:H', p='0:P', k='0:SM',
                                  i='0:N'),
                  code='DWK_out = K_inp * DKK_inp',
                  inputs=[(K, 'b,k,i'), (DKK, 'b,h,p,k')],
                  wcr_outputs=[(DWK, 'p,h,i')])

    ##### dq #####

    create_einsum(state,
                  map_ranges=dict(b='0:B', h='0:H', p='0:P', i='0:N',
                                  j='0:SN'),
                  code='DQ_out = WQ_inp * DQQ_inp',
                  inputs=[(WQ, 'p,h,i'), (DQQ, 'b,h,p,j')],
                  wcr_outputs=[(DQ, 'b,j,i')])

    ##### dk #####
    if need_dk_dv:
        WK = state.add_read('WK')
        WV = state.add_read('WV')
        DK = state.add_write('DK')
        DV = state.add_write('DV')
        create_einsum(state,
                      map_ranges=dict(b='0:B',
                                      h='0:H',
                                      p='0:P',
                                      k='0:SM',
                                      i='0:N'),
                      code='DK_out = WK_inp * DKK_inp',
                      inputs=[(WK, 'p,h,i'), (DKK, 'b,h,p,k')],
                      wcr_outputs=[(DK, 'b,k,i')])

        ##### dv #####

        create_einsum(state,
                      map_ranges=dict(b='0:B',
                                      h='0:H',
                                      p='0:P',
                                      k='0:SM',
                                      i='0:N'),
                      code='DV_out = WV_inp * DVV_inp',
                      inputs=[(WV, 'p,h,i'), (DVV, 'b,h,p,k')],
                      wcr_outputs=[(DV, 'b,k,i')])

    return sdfg


def create_attn_forward_and_compile(**kwargs):
    sdfg = attn_forward_sdfg(**kwargs)

    if os.name == 'nt':
        dace.Config.append('compiler', 'cpu', 'libs', value='cublas.lib')
    else:
        dace.Config.append('compiler', 'cpu', 'libs', value='libcublas.so')

    compiled_sdfg = sdfg.compile(optimizer=False)

    return compiled_sdfg
