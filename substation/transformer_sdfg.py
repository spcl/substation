import dace
import numpy as np
import math
import os

# Imported to register transformation
from substation.xforms.merge_source_sink import MergeSourceSinkArrays

from substation.attention import attn_forward_sdfg
from substation.dtypes import *

B, H, N, P, SM, SN = (dace.symbol(s) for s in ['B', 'H', 'N', 'P', 'SM', 'SN'])
emb = dace.symbol('emb')
eps = 1e-5


@dace.program
def gelu(x: dace_dtype[B, SM, N]):
    """Gaussian Error Linear Unit applied to x."""
    out = np.ndarray(x.shape, x.dtype)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            inp << x[i, j, k]
            outp >> out[i, j, k]
            outp = 0.5 * inp * (1 + math.tanh(
                math.sqrt(2.0 / math.pi) * (inp + 0.044715 * (inp**3))))
    return out


@dace.program
def linear(x: dace_dtype[B, SM, N], w: dace_dtype[emb, N]):
    """Fully-connected layer with weights w applied to x, and optional bias b.

    x is of shape (batch, *).
    w is of shape (num_inputs, num_outputs).

    """
    return x @ np.transpose(w)


@dace.program
def linear_with_bias(x: dace_dtype[B, SM, N], w: dace_dtype[emb, N],
                     bias: dace_dtype[emb]):
    """Fully-connected layer with weights w and bias."""
    out = np.ndarray([B, SM, emb], x.dtype)
    outb = np.ndarray([B, SM, emb], x.dtype)
    for i in dace.map[0:B]:
        out[i] = x[i] @ np.transpose(w[:])
    for i, j, k in dace.map[0:B, 0:SM, 0:emb]:
        with dace.tasklet:
            inp << out[i, j, k]
            b << bias[k]
            outp >> outb[i, j, k]
            outp = inp + b
    return outb


@dace.program
def meanstd(x: dace_dtype[B, SM, N]):
    mean = np.ndarray([B, SM], x.dtype)
    std = np.ndarray([B, SM], x.dtype)

    moment = dace.reduce(lambda a, b: a + b, x, axis=2, identity=0)
    second_moment = dace.reduce(lambda a, b: a + b * b, x, axis=2, identity=0)

    for i, j in dace.map[0:B, 0:SM]:
        with dace.tasklet:
            fmom << moment[i, j]
            mn >> mean[i, j]
            mn = fmom / (SM * B)
        with dace.tasklet:
            fmom << moment[i, j]
            smom << second_moment[i, j]
            st >> std[i, j]
            st = (smom - (fmom * fmom)) / (SM * B)

    return mean, std


@dace.program
def layer_norm(x: dace_dtype[B, SM, N]):
    """ Apply layer normalization to x. """
    out = np.ndarray(x.shape, x.dtype)
    mean, std = meanstd(x)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            in_x << x[i, j, k]
            in_m << mean[i, j]
            in_s << std[i, j]
            o >> out[i, j, k]
            o = (in_x - in_m) / (in_s + eps)
    return out


@dace.program
def layer_norm_scaled(x: dace_dtype[B, SM, N], scale: dace_dtype[N],
                      bias: dace_dtype[N]):
    """Apply layer normalization to x, with scale and bias.

    scale and bias are the same shape as the final axis.

    """
    out = np.ndarray(x.shape, x.dtype)
    mean, std = meanstd(x)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            in_scal << scale[k]
            in_bias << bias[k]
            in_x << x[i, j, k]
            in_m << mean[i, j]
            in_s << std[i, j]
            o >> out[i, j, k]
            o = in_scal * ((in_x - in_m) / (in_s + eps)) + in_bias
    return out


@dace.program
def dropout(x: dace_dtype[B, SM, N], mask: dace_dtype[B, SM, N]):
    """Apply dropout with pre-randomized dropout mask."""
    return x * mask


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, SM]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3)
    tmp_out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)
    out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)

    # No broadcasting rules
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << X_in[i, j, k, l]
            mx << tmp_max[i, j, k]
            o >> tmp_out[i, j, k, l]
            o = math.exp(inp - mx)
    #tmp_out = np.exp(X_in - tmp_max)

    tmp_sum = dace.reduce(lambda a, b: a + b, tmp_out, identity=0, axis=3)
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << tmp_out[i, j, k, l]
            sm << tmp_sum[i, j, k]
            o >> out[i, j, k, l]
            o = inp / sm

    return out


@dace.program
def mha_forward(q: dace_dtype[B, SN, N], k: dace_dtype[B, SM, N],
                v: dace_dtype[B, SM, N], wq: dace_dtype[P, H, N],
                wk: dace_dtype[P, H, N], wv: dace_dtype[P, H, N],
                wo: dace_dtype[P, H, N], scaler: dace_dtype):
    qq = np.einsum("phi,bji->phbj", wq, q)
    kk = np.einsum("phi,bki->phbk", wk, k)
    vv = np.einsum("phi,bki->phbk", wv, v)
    beta = scaler * np.einsum("phbk,phbj->hbjk", kk, qq)
    alpha = softmax(beta)
    gamma = np.einsum("phbk,hbjk->phbj", vv, alpha)
    out = np.einsum("phi,phbj->bji", wo, gamma)
    return out



@dace.program
def encoder(x: dace_dtype[B, SM,
                          N], attn_wq: dace_dtype[P, H,
                                                  N], attn_wk: dace_dtype[P, H,
                                                                          N],
            attn_wv: dace_dtype[P, H, N], attn_wo: dace_dtype[P, H, N],
            attn_scale: dace_dtype, norm1_scale: dace_dtype[N],
            norm1_bias: dace_dtype[N], norm2_scale: dace_dtype[N],
            norm2_bias: dace_dtype[N], linear1_w: dace_dtype[emb, N],
            linear1_b: dace_dtype[emb], linear2_w: dace_dtype[N, emb],
            linear2_b: dace_dtype[N], attn_dropout: dace_dtype[B, SM, N],
            linear1_dropout: dace_dtype[B, SM,
                                        emb], ff_dropout: dace_dtype[B, SM,
                                                                     N]):

    # Self-attention.
    # attn = np.ndarray(x.shape, x.dtype)
    # attn_forward(Q=x,
    #              K=x,
    #              V=x,
    #              WQ=attn_wq,
    #              WK=attn_wk,
    #              WV=attn_wv,
    #              WO=attn_wo,
    #              scaler=attn_scale,
    #              OUT=attn,
    #              B=B,
    #              H=H,
    #              N=N,
    #              P=P,
    #              SM=SM,
    #              SN=SM)
    attn = mha_forward(x, x, x, attn_wq, attn_wk, attn_wv, attn_wo, attn_scale)

    # Residual connection.
    attn_resid = dropout(attn, attn_dropout) + x  # B x SM x N

    normed1 = layer_norm_scaled(attn_resid, norm1_scale,
                                norm1_bias)  # B x SM x N

    # Feedforward network.
    ff = linear_with_bias(
        dropout(
            gelu(linear_with_bias(normed1, linear1_w,
                                  linear1_b)),  # B x SM x emb
            linear1_dropout),
        linear2_w,
        linear2_b)  # B x SM x N

    # Residual connection.
    ff_resid = dropout(ff, ff_dropout) + normed1  # B x SM x N
    normed2 = layer_norm_scaled(ff_resid, norm2_scale,
                                norm2_bias)  # B x SM x N
    return normed2


@dace.program
def decoder(x: dace_dtype[B, SM, N], attn_wq: dace_dtype[P, H, N],
            attn_wk: dace_dtype[P, H, N], attn_wv: dace_dtype[P, H, N],
            attn_wo: dace_dtype[P, H, N], attn_scale: dace_dtype,
            attn_mask: dace_dtype[SM, SM], norm1_scale: dace_dtype[N],
            norm1_bias: dace_dtype[N], norm2_scale: dace_dtype[N],
            norm2_bias: dace_dtype[N], linear1_w: dace_dtype[emb, N],
            linear1_b: dace_dtype[emb], linear2_w: dace_dtype[N, emb],
            linear2_b: dace_dtype[N], attn_dropout: dace_dtype[B, SM, N],
            linear1_dropout: dace_dtype[B, SM,
                                        emb], ff_dropout: dace_dtype[B, SM,
                                                                     N]):
    # Masked self-attention.
    attn = np.ndarray(x.shape, x.dtype)
    attn_forward_mask(Q=x,
                      K=x,
                      V=x,
                      WQ=attn_wq,
                      WK=attn_wk,
                      WV=attn_wv,
                      WO=attn_wo,
                      scaler=attn_scale,
                      OUT=attn,
                      MASK=attn_mask,
                      B=B,
                      H=H,
                      N=N,
                      P=P,
                      SM=SM,
                      SN=SM)

    # Residual connection.
    attn_resid = dropout(attn, attn_dropout) + x
    normed1 = layer_norm(attn_resid, norm1_scale, norm1_bias)
    # Feedforward network.
    ff = linear_with_bias(
        dropout(gelu(linear_with_bias(normed1, linear1_w, linear1_b)),
                linear1_dropout), linear2_w, linear2_b)
    # Residual connection.
    ff_resid = dropout(ff, ff_dropout) + normed1
    normed2 = layer_norm(ff_resid, norm2_scale, norm2_bias)
    return normed2


@dace.program
def dec_with_enc_attn(
    x: dace_dtype[B, SN, N], encoder_out: dace_dtype[B, SM, N],
    sattn_wq: dace_dtype[P, H, N], sattn_wk: dace_dtype[P, H, N],
    sattn_wv: dace_dtype[P, H,
                         N], sattn_wo: dace_dtype[P, H,
                                                  N], sattn_scale: dace_dtype,
    sattn_mask: dace_dtype[SN, SN], edattn_wq: dace_dtype[P, H, N],
    edattn_wk: dace_dtype[P, H, N], edattn_wv: dace_dtype[P, H, N],
    edattn_wo: dace_dtype[P, H, N], edattn_scale: dace_dtype,
    norm1_scale: dace_dtype[N], norm1_bias: dace_dtype[N],
    norm2_scale: dace_dtype[N], norm2_bias: dace_dtype[N],
    norm3_scale: dace_dtype[N], norm3_bias: dace_dtype[N],
    linear1_w: dace_dtype[emb, N], linear1_b: dace_dtype[emb],
    linear2_w: dace_dtype[N, emb], linear2_b: dace_dtype[N],
    sattn_dropout: dace_dtype[B, SN, N], edattn_dropout: dace_dtype[B, SN, N],
    linear1_dropout: dace_dtype[B, SN, emb], ff_dropout: dace_dtype[B, SN, N]):
    # Masked self-attention.
    sattn = np.ndarray(x.shape, x.dtype)
    attn_forward_mask(Q=x,
                      K=x,
                      V=x,
                      WQ=sattn_wq,
                      WK=sattn_wk,
                      WV=sattn_wv,
                      WO=sattn_wo,
                      scaler=sattn_scale,
                      OUT=sattn,
                      MASK=sattn_mask,
                      B=B,
                      H=H,
                      N=N,
                      P=P,
                      SM=SN,
                      SN=SN)
    # Residual connection.
    sattn_resid = dropout(sattn, sattn_dropout) + x
    normed1 = layer_norm(sattn_resid, norm1_scale, norm1_bias)

    # Encoder-decoder attention.
    edattn = np.ndarray(normed1.shape, dace.float32)
    attn_forward(Q=normed1,
                 K=encoder_out,
                 V=encoder_out,
                 WQ=edattn_wq,
                 WK=edattn_wk,
                 WV=edattn_wv,
                 WO=edattn_wo,
                 scaler=edattn_scale,
                 OUT=edattn,
                 B=B,
                 H=H,
                 N=N,
                 P=P,
                 SM=SM,
                 SN=SN)

    # Residual connection.
    edattn_resid = dropout(edattn, edattn_dropout) + normed1
    normed2 = layer_norm(edattn_resid, norm2_scale, norm2_bias)
    # Feedforward network.
    ff = linear_with_bias(
        dropout(gelu(linear_with_bias(normed2, linear1_w, linear1_b)),
                linear1_dropout), linear2_w, linear2_b)
    # Residual connection.
    ff_resid = dropout(ff, ff_dropout) + normed2
    normed3 = layer_norm(ff_resid, norm3_scale, norm3_bias)
    return normed3


if __name__ == '__main__':
    # B = 2
    # H = 16
    # P = 64
    # N = P * H
    # SM, SN = 512, 512
    # hidden = 4 * N
    from dace.transformation.dataflow import MapFusion
    from dace.transformation.interstate import StateFusion

    # dace.Config.set('optimizer',
    #                 'automatic_strict_transformations',
    #                 value=False)
    # dace.Config.set('optimizer',
    #                 'automatic_strict_transformation',
    #                 value=False)

    sdfg = mha_forward.to_sdfg()
    #sdfg.apply_transformations_repeated([StateFusion])
    sdfg.save('mha3.sdfg')

    esdfg = encoder.to_sdfg()  #strict=False)
    #esdfg.apply_transformations_repeated([StateFusion, MergeSourceSinkArrays])
    #esdfg.apply_strict_transformations()
    #esdfg.apply_transformations_repeated(MapFusion)
    esdfg.save('encoder.sdfg')

    dsdfg = decoder.to_sdfg()
    dsdfg.apply_strict_transformations()
    dsdfg.apply_transformations_repeated(MapFusion)
    dsdfg.save('decoder-nonstrict.sdfg')

    desdfg = dec_with_enc_attn.to_sdfg()
    desdfg.apply_strict_transformations()
    desdfg.apply_transformations_repeated(MapFusion)
    desdfg.save('decoder_encattn.sdfg')

    # Remove duplicate CUBLAS creation code. TODO: Use library nodes instead
    cublas_found = False
    for node, parent in desdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.Tasklet):
            if 'cublasHandle_t' in node.code_global:
                if cublas_found:
                    node.code_global = ''
                    node.code_init = ''
                    node.code_exit = ''
                cublas_found = True

    # For compilation, ensure we link with cublas
    if os.name == 'nt':
        dace.Config.append('compiler', 'cpu', 'libs', value='cublas.lib')
    else:
        dace.Config.append('compiler', 'cpu', 'libs', value='libcublas.so')

    esdfg.compile(optimizer=False)
    dsdfg.compile(optimizer=False)
    desdfg.compile(optimizer=False)
