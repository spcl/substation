import dace
import numpy as np
import math
from substation.dtypes import *
from substation.transformer_sdfg import *
from substation.attention import create_attn_backward_sdfg


@dace.program
def relu_backward_data(x, dy):
    """Derivative of ReLU."""
    return (x > 0) * dy


@dace.program
def gelu_backward_data(x: dace_dtype[B, SM, N], dy: dace_dtype[B, SM, N]):
    """Derivative of GeLU.

    x is the original input.
    dy is the input error signal.

    """
    res = np.ndarray([B, SM, N], dace_dtype)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            inx << x[i, j, k]
            iny << dy[i, j, k]
            xcubed = inx**3
            xcubed03 = 0.0356774 * xcubed
            x03plus79 = xcubed03 + 0.797885 * inx
            sech = lambda a: 1 / math.cosh(a)
            out = (0.5 * math.tanh(x03plus79) +
                   (0.0535161 * xcubed + 0.398942 * inx) *
                   (sech(x03plus79)**2) + 0.5) * iny
            out >> res[i, j, k]

    return res


@dace.program
def layer_norm_backward_data(x: dace_dtype[B, SM, N], dy: dace_dtype[B, SM, N],
                             mean: dace_dtype[B, SM], std: dace_dtype[B, SM],
                             scale: dace_dtype[N]):
    """Derivative of input of layer norm."""
    dx = np.ndarray([B, SM, N], dace_dtype)

    # used to be "scaled_dy = dy*scale"  # Backprop through the scale.
    scaled_dy = np.ndarray([B, SM, N], dace_dtype)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            idy << dy[i, j, k]
            iscl << scale[k]
            out = idy * iscl
            out >> scaled_dy[i, j, k]

    # used to be "scaled_dy * (x - mean)"
    scaled_normdy = np.ndarray([B, SM, N], dace_dtype)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            ix << x[i, j, k]
            im << mean[i, j]
            isdy << scaled_dy[i, j, k]
            out = isdy * (ix - im)
            out >> scaled_normdy[i, j, k]

    dmeansum = dace.reduce(lambda a, b: a + b, scaled_dy, axis=2, identity=0)
    dstdsum = dace.reduce(lambda a, b: a + b,
                          scaled_normdy,
                          axis=2,
                          identity=0)
    dmean = -dmeansum / std
    dstd = -dstdsum / (std**2)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            sdy << scaled_dy[i, j, k]
            istd << std[i, j]
            idmean << dmean[i, j]
            idstd << dstd[i, j]
            ix << x[i, j, k]
            imean << mean[i, j]
            out = sdy / istd + idmean / N + idstd * (ix - imean) / N / istd
            out >> dx[i, j, k]

    return dx


@dace.program
def layer_norm_backward_weights(dy: dace_dtype[B, SM, N],
                                normed: dace_dtype[B, SM, N]):
    """Derivative of scale/bias of layernorm, if any.

    Returns dscale and dbias.

    """
    dscale = dace.reduce(lambda a, b: a + b, (normed * dy),
                         axis=(1, 0),
                         identity=0)
    dbias = dace.reduce(lambda a, b: a + b, dy, axis=(1, 0), identity=0)
    return dscale, dbias


@dace.program
def dropout_backward_data(dy, mask):
    """Derivative of dropout with drop probability p.

    mask is as calculated in forward prop.

    """
    return dy * mask


@dace.program
def linear_backward_data(w: dace_dtype[N, emb], dy: dace_dtype[B, SM, N]):
    """Derivative of input of fully-connected layer.

    Bias does not matter.

    dy is the input error signal of shape (batch, *, num_outputs).

    """
    res = np.ndarray([B, SM, emb], dace_dtype)
    for i in dace.map[0:B]:
        res[i] = dy[i] @ w[:]
    return res


@dace.program
def linear_backward_weights(x: dace_dtype[B, SM, emb], dy: dace_dtype[B, SM,
                                                                      N]):
    """Derivative of weights of fully-connected layer.

    Returns a tuple of (weight derivative, bias derivative).
    Bias derivative is None if bias is None.

    """
    # Sum out dimensions to obtain 1D bias.
    dbias = dace.reduce(lambda a, b: a + b, dy, axis=(1, 0), identity=0)

    # Transpose only the last two dimensions.
    # replaces np.transpose(dy, (0, 2, 1))
    dyt = np.ndarray([B, N, SM], dace_dtype)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            inp << dy[i, j, k]
            out >> dyt[i, k, j]
            out = inp

    dw = np.ndarray([B, N, emb], dace_dtype)
    for i in dace.map[0:B]:
        dw[i] = dyt[i] @ x[i]

    # Sum batch dimension
    dwsum = dace.reduce(lambda a, b: a + b, dw, axis=0, identity=0)

    return dwsum, dbias


act_backward_data = relu_backward_data
attn_backward_sdfg = create_attn_backward_sdfg(create_init_state=False,
                                               need_dk_dv=False)


@dace.program
def encoder_backward(
    x: dace_dtype[B, SM, N], dy: dace_dtype[B, SM,
                                            N], attn_concat: dace_dtype[B, SM,
                                                                        H, P],
    attn_proj_q: dace_dtype[B, H, SM, P], attn_proj_k: dace_dtype[B, H, SM, P],
    attn_proj_v: dace_dtype[B, H, SM,
                            P], attn_scaled_scores: dace_dtype[B, H, SM, SM],
    attn_dropout_mask: dace_dtype[B, SM, N], norm1_mean: dace_dtype[B, SM],
    norm1_std: dace_dtype[B, SM], norm1_normed: dace_dtype[B, SM, N],
    linear1_dropout_mask: dace_dtype[B, SM,
                                     emb], ff_dropout_mask: dace_dtype[B, SM,
                                                                       N],
    norm2_mean: dace_dtype[B, SM], norm2_std: dace_dtype[B, SM],
    norm2_normed: dace_dtype[B, SM, N], ff_resid: dace_dtype[B, SM, N],
    ff1: dace_dtype[B, SM, emb], ff1_linear: dace_dtype[B, SM, emb],
    normed1: dace_dtype[B, SM, N], attn_resid: dace_dtype[B, SM, N],
    attn_wq: dace_dtype[P, H, N], attn_wk: dace_dtype[P, H, N],
    attn_wv: dace_dtype[P, H,
                        N], attn_wo: dace_dtype[P, H,
                                                N], attn_scale: dace_dtype,
    norm1_scale: dace_dtype[N], norm1_bias: dace_dtype[N],
    norm2_scale: dace_dtype[N], norm2_bias: dace_dtype[N],
    linear1_w: dace_dtype[emb, N], linear1_b: dace_dtype[emb],
    linear2_w: dace_dtype[N, emb], linear2_b: dace_dtype[N]):
    """Backward data and weights for an encoder.

    Arguments as in forward version.

    This does both backward data and backward weights, since the same
    intermediate results are needed for both.

    Returns dx, dattn_wq, dattn_wk, dattn_wv, dattn_wo,
    dnorm1_scale, dnorm1_bias, dnorm2_scale, dnorm2_bias,
    dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b in this order.

    """

    # Backward through norm2.
    dff_resid = layer_norm_backward_data(ff_resid, dy, norm2_mean, norm2_std,
                                         norm2_scale)
    dnorm2_scale, dnorm2_bias = layer_norm_backward_weights(dy, norm2_normed)
    # Backward through residual connection.
    dff_dropout = dff_resid
    dnormed1_resid = dff_resid
    # Backward through FF dropout.
    dff = dropout_backward_data(dff_dropout, ff_dropout_mask)
    # Backward through linear2.
    dff1 = linear_backward_data(linear2_w, dff)
    dlinear2_w, dlinear2_b = linear_backward_weights(ff1, dff)
    # Backward through ff1 dropout.
    dff1_act = dropout_backward_data(dff1, linear1_dropout_mask)
    # Backward through ff1 activation.
    dff1_linear = act_backward_data(ff1_linear, dff1_act)
    # Backward through ff1 linear.
    dnormed1_linear = linear_backward_data(linear1_w, dff1_linear)
    dlinear1_w, dlinear1_b = linear_backward_weights(normed1, dff1_linear)
    # Combine residuals.
    dnormed1 = dnormed1_resid + dnormed1_linear
    # Backward through norm1.
    dattn_resid = layer_norm_backward_data(attn_resid, dnormed1, norm1_mean,
                                           norm1_std, norm1_scale)
    dnorm1_scale, dnorm1_bias = layer_norm_backward_weights(
        dnormed1, norm1_normed)
    # Backward through residual connection.
    dattn_dropout = dattn_resid
    dx_resid = dattn_resid
    # Backward through attention dropout.
    dattn = dropout_backward_data(dattn_dropout, attn_dropout_mask)
    # Backward through self-attention.
    # dx, dk, and dv are the same.
    dx_attn = np.ndarray([B, SM, N], dace_dtype)
    # dk_unused = np.ndarray([B, SM, N], dace_dtype)
    # dv_unused = np.ndarray([B, SM, N], dace_dtype)
    dattn_wq = np.ndarray([P, H, N], dace_dtype)
    dattn_wk = np.ndarray([P, H, N], dace_dtype)
    dattn_wv = np.ndarray([P, H, N], dace_dtype)
    dattn_wo = np.ndarray([P, H, N], dace_dtype)
    attn_backward_sdfg(
        Q=x,
        K=x,
        V=x,
        WQ=attn_wq,
        WO=attn_wo,
        scaler=attn_scale,
        DOUT=dattn,
        QQ=attn_proj_q,
        KK=attn_proj_k,
        VV=attn_proj_v,
        ALPHA=attn_scaled_scores,
        GAMMA=attn_concat,
        DQ=dx_attn,
        DWQ=dattn_wq,
        DWK=dattn_wk,
        DWV=dattn_wv,
        DWO=dattn_wo,
        # DK=dk_unused, DV=dv_unused, WK=attn_wk, WV=attn_wv,
        B=B,
        SM=SM,
        SN=SM,
        H=H,
        P=P,
        N=N)

    # Finally compute dx.
    dx = dx_resid + dx_attn
    return (dx, dattn_wq, dattn_wk, dattn_wv, dattn_wo, dnorm1_scale,
            dnorm1_bias, dnorm2_scale, dnorm2_bias, dlinear1_w, dlinear1_b,
            dlinear2_w, dlinear2_b)


if __name__ == '__main__':
    # B = 2
    # H = 16
    # P = 64
    # N = P * H
    # SM, SN = 512, 512
    # hidden = 4 * N
    from dace.transformation.dataflow import MapFusion
    from dace.transformation.interstate import StateFusion

    dace.Config.set('optimizer',
                    'automatic_strict_transformations',
                    value=False)
    dace.Config.set('optimizer',
                    'automatic_strict_transformation',
                    value=False)
    esdfg = encoder_backward.to_sdfg(strict=False)
    esdfg.apply_transformations_repeated([StateFusion, MergeSourceSinkArrays])
    #esdfg.apply_strict_transformations()
    #esdfg.apply_transformations_repeated(MapFusion)
    esdfg.save('encoder_bwd-nonfused.sdfg')
    exit()

    # dsdfg = decoder.to_sdfg()
    # dsdfg.apply_strict_transformations()
    # dsdfg.apply_transformations_repeated(MapFusion)
    # dsdfg.save('decoder_bwd.sdfg')
    #
    # desdfg = dec_with_enc_attn.to_sdfg()
    # desdfg.apply_strict_transformations()
    # desdfg.apply_transformations_repeated(MapFusion)
    # desdfg.save('decoder_encattn_bwd.sdfg')

    # Remove duplicate CUBLAS creation code. TODO: Use library nodes instead
    # cublas_found = False
    # for node, parent in desdfg.all_nodes_recursive():
    #     if isinstance(node, dace.nodes.Tasklet):
    #         if 'cublasHandle_t' in node.code_global:
    #             if cublas_found:
    #                 node.code_global = ''
    #                 node.code_init = ''
    #                 node.code_exit = ''
    #             cublas_found = True

    # For compilation, ensure we link with cublas
    if os.name == 'nt':
        dace.Config.append('compiler', 'cpu', 'libs', value='cublas.lib')
    else:
        dace.Config.append('compiler', 'cpu', 'libs', value='libcublas.so')

    esdfg.compile(optimizer=False)
    # dsdfg.compile(optimizer=False)
    # desdfg.compile(optimizer=False)
