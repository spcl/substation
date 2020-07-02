import numpy as np
from substation.attention import (attn_forward_numpy, attn_mask_forward_numpy,
                                  attn_backward_numpy,
                                  softmax, softmax_backward_data)


def embedding():
    pass


def gelu(x):
    """Gaussian Error Linear Unit applied to x."""
    return (0.5*x*(1 + np.tanh(
        np.sqrt(2.0 / np.pi)*(x + 0.044715*(x**3)))))


def gelu_backward_data(x, dy):
    """Derivative of GeLU.

    x is the original input.
    dy is the input error signal.

    """
    xcubed = x**3
    xcubed03 = 0.0356774*xcubed
    x03plus79 = xcubed03 + 0.797885*x
    def sech(a): return 1 / np.cosh(a)
    return (0.5*np.tanh(x03plus79) + (0.0535161*xcubed + 0.398942*x)*(sech(x03plus79)**2)+0.5)*dy


def relu(x):
    """ReLU applied to x."""
    return np.maximum(0, x)


def relu_backward_data(x, dy):
    """Derivative of ReLU."""
    return (x > 0) * dy


def linear(x, w, bias=None):
    """Fully-connected layer with weights w applied to x, and optional bias.

    x is of shape (batch, *, num_inputs).
    w is of shape (num_inputs, num_outputs).
    bias is of shape (num_outputs,)

    """
    if bias is not None:
        return np.matmul(x, w.T) + bias
    else:
        return np.matmul(x, w.T)


def linear_backward_data(x, w, dy):
    """Derivative of input of fully-connected layer.

    Bias does not matter.

    dy is the input error signal of shape (batch, *, num_outputs).

    """
    return np.matmul(dy, w)


def linear_backward_weights(x, w, dy, bias=None):
    """Derivative of weights of fully-connected layer.

    Returns a tuple of (weight derivative, bias derivative).
    Bias derivative is None if bias is None.

    """
    # Only works for 3d x-- other cases need to sum out dimensions as needed.
    if bias is not None:
        dbias = dy.sum(axis=(1, 0))
    else:
        dbias = None
    # Transpose only the last two dimensions.
    transposed_axes = list(range(dy.ndim))
    transposed_axes[-2], transposed_axes[-1] = transposed_axes[-1], transposed_axes[-2]
    # Sum out batch dimension.
    dw = np.matmul(np.transpose(dy, transposed_axes), x).sum(axis=0)
    return dw, dbias


def layer_norm(x, scale=None, bias=None, eps=1e-5):
    """Apply layer normalization to x, with optional scale and bias.

    scale and bias are the same shape as the final axis.

    """
    if ((scale is None and bias is not None)
        or (bias is not None and scale is None)):
        raise ValueError('Must provide both or neither scale and bias')
    # Note:
    # PyTorch uses 1/n when computing the variance/standard deviation in its
    # layer normalization.
    # np.std does so as well by default (ddof=0).
    # torch.std uses 1/(n-1) by default (unbiased=True).
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    normed = (x - mean) / (std + eps)
    if scale is not None and bias is not None:
        return normed*scale + bias, mean, std, normed
    return normed, mean, std, None


def layer_norm_backward_data(x, dy, mean, std, scale=None, bias=None, eps=1e-5):
    """Derivative of input of layer norm."""
    n = x.shape[-1]  # Amount of things we take the mean/std of.
    if scale is not None:
        scaled_dy = dy*scale  # Backprop through the scale.
    else:
        scaled_dy = dy
    dmean = -scaled_dy.sum(axis=-1, keepdims=True) / std
    dstd = -(scaled_dy*(x-mean)).sum(axis=-1, keepdims=True) / (std**2)
    dx = scaled_dy/std + dmean/n + dstd*(x-mean)/n/std
    return dx


def layer_norm_backward_weights(dy, normed, scale=None, bias=None, eps=1e-5):
    """Derivative of scale/bias of layernorm, if any.

    Returns dscale and dbias.

    """
    if scale is None and bias is None:
        return (None, None)
    dbias = dy.sum(axis=1).sum(axis=0)
    dscale = (normed*dy).sum(axis=1).sum(axis=0)
    return dscale, dbias


def dropout(x, p):
    """Apply dropout with drop probability p."""
    if p == 0: return x, None
    mask = np.random.choice([0, 1/p], size=x.shape, p=[p, 1-p])
    return x*mask, mask


def dropout_backward_data(dy, p, mask):
    """Derivative of dropout with drop probability p.

    mask is as calculated in forward prop.

    """
    if p == 0: return dy
    return dy*mask


def gen_attn_mask(q, k):
    """Return an attention mask for q and k.

    The mask does not have the batch dimension.

    """
    return np.triu(np.full((q.shape[1], k.shape[1]), float('-inf')), k=1)


def _get_activation_forward(act):
    if act == 'relu':
        return relu
    elif act == 'gelu':
        return gelu
    else:
        raise ValueError('Unknown activation ' + act)

def _get_activation_backward(act):
    if act == 'relu':
        return relu_backward_data
    elif act == 'gelu':
        return gelu_backward_data
    else:
        raise ValueError('Unknown activation ' + act)

def encoder(x, attn_wq, attn_wk, attn_wv, attn_wo,
            attn_in_b, attn_out_b, attn_scale,
            norm1_scale, norm1_bias, norm2_scale, norm2_bias,
            linear1_w, linear1_b, linear2_w, linear2_b,
            attn_dropout_p, linear1_dropout_p, ff_dropout_p,
            activation='gelu'):
    act = _get_activation_forward(activation)
    # Self-attention.
    (attn, attn_concat, attn_proj_q, attn_proj_k, attn_proj_v,
     attn_scaled_scores) = attn_forward_numpy(
        x, x, x, attn_wq, attn_wk, attn_wv, attn_wo,
         attn_in_b, attn_out_b, attn_scale)
    # Residual connection.
    attn_dropout, attn_dropout_mask = dropout(attn, attn_dropout_p)
    attn_resid  = attn_dropout + x
    normed1, norm1_mean, norm1_std, norm1_normed = layer_norm(
        attn_resid, norm1_scale, norm1_bias)
    # Feedforward network.
    ff1_linear = linear(normed1, linear1_w, bias=linear1_b)
    ff1_act = act(ff1_linear)
    ff1, linear1_dropout_mask = dropout(ff1_act, linear1_dropout_p)
    ff = linear(ff1, linear2_w, bias=linear2_b)
    # Residual connection.
    ff_dropout, ff_dropout_mask = dropout(ff, ff_dropout_p)
    ff_resid = ff_dropout + normed1
    normed2, norm2_mean, norm2_std, norm2_normed = layer_norm(
        ff_resid, norm2_scale, norm2_bias)
    return (normed2,
            attn_concat, attn_proj_q, attn_proj_k, attn_proj_v, attn_scaled_scores,
            attn_dropout_mask,
            norm1_mean, norm1_std, norm1_normed,
            linear1_dropout_mask, ff_dropout_mask,
            norm2_mean, norm2_std, norm2_normed,
            ff_resid, ff1, ff1_linear, normed1, attn_resid)


def encoder_backward(
        x, dy,
        attn_concat, attn_proj_q, attn_proj_k, attn_proj_v, attn_scaled_scores,
        attn_dropout_mask,
        norm1_mean, norm1_std, norm1_normed,
        linear1_dropout_mask, ff_dropout_mask,
        norm2_mean, norm2_std, norm2_normed,
        ff_resid, ff1, ff1_linear, normed1, attn_resid,
        attn_wq, attn_wk, attn_wv, attn_wo, attn_scale,
        norm1_scale, norm1_bias, norm2_scale, norm2_bias,
        linear1_w, linear1_b, linear2_w, linear2_b,
        attn_dropout_p, linear1_dropout_p, ff_dropout_p,
        activation='gelu'):
    """Backward data and weights for an encoder.

    Arguments as in forward version.

    This does both backward data and backward weights, since the same
    intermediate results are needed for both.

    Returns dx, dattn_wq, dattn_wk, dattn_wv, dattn_wo,
    dnorm1_scale, dnorm1_bias, dnorm2_scale, dnorm2_bias,
    dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b in this order.

    """
    act_backward_data = _get_activation_backward(activation)

    # Backward through norm2.
    dff_resid = layer_norm_backward_data(
        ff_resid, dy, norm2_mean, norm2_std, norm2_scale, norm2_bias)
    dnorm2_scale, dnorm2_bias = layer_norm_backward_weights(
        dy, norm2_normed, norm2_scale, norm2_bias)
    # Backward through residual connection.
    dff_dropout = dff_resid
    dnormed1_resid = dff_resid
    # Backward through FF dropout.
    dff = dropout_backward_data(dff_dropout, ff_dropout_p, ff_dropout_mask)
    # Backward through linear2.
    dff1 = linear_backward_data(ff1, linear2_w, dff)
    dlinear2_w, dlinear2_b = linear_backward_weights(
        ff1, linear2_w, dff, bias=linear2_b)
    # Backward through ff1 dropout.
    dff1_act = dropout_backward_data(
        dff1, linear1_dropout_p, linear1_dropout_mask)
    # Backward through ff1 activation.
    dff1_linear = act_backward_data(ff1_linear, dff1_act)
    # Backward through ff1 linear.
    dnormed1_linear = linear_backward_data(normed1, linear1_w, dff1_linear)
    dlinear1_w, dlinear1_b = linear_backward_weights(
        normed1, linear1_w, dff1_linear, bias=linear1_b)
    # Combine residuals.
    dnormed1 = dnormed1_resid + dnormed1_linear
    # Backward through norm1.
    dattn_resid = layer_norm_backward_data(
        attn_resid, dnormed1, norm1_mean, norm1_std, norm1_scale, norm1_bias)
    dnorm1_scale, dnorm1_bias = layer_norm_backward_weights(
        dnormed1, norm1_normed, norm1_scale, norm1_bias)
    # Backward through residual connection.
    dattn_dropout = dattn_resid
    dx_resid = dattn_resid
    # Backward through attention dropout.
    dattn = dropout_backward_data(
        dattn_dropout, attn_dropout_p, attn_dropout_mask)
    # Backward through self-attention.
    # dx, dk, and dv are the same.
    dx_attn, dk_unused, dv_unused, dattn_wq, dattn_wk, dattn_wv, dattn_wo, dattn_in_b, dattn_out_b,  = attn_backward_numpy(
        x, x, x, attn_wq, attn_wk, attn_wv, attn_wo, attn_scale, dattn,
        attn_concat, attn_proj_q, attn_proj_k, attn_proj_v, attn_scaled_scores)
    # Finally compute dx.
    dx = dx_resid + dx_attn
    return (dx,
            dattn_wq, dattn_wk, dattn_wv, dattn_wo, dattn_in_b, dattn_out_b,
            dnorm1_scale, dnorm1_bias, dnorm2_scale, dnorm2_bias,
            dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b)
    

def decoder_with_encoder_attention(
        x, encoder_out,
        sattn_wq, sattn_wk, sattn_wv, sattn_wo, sattn_in_b, sattn_out_b, sattn_scale, sattn_mask,
        edattn_wq, edattn_wk, edattn_wv, edattn_wo, edattn_in_b, edattn_out_b, edattn_scale,
        norm1_scale, norm1_bias,
        norm2_scale, norm2_bias,
        norm3_scale, norm3_bias,
        linear1_w, linear1_b, linear2_w, linear2_b,
        sattn_dropout_p, edattn_dropout_p, linear1_dropout_p, ff_dropout_p,
        activation='gelu'):
    act = _get_activation_forward(activation)
    # Masked self-attention.
    (sattn, sattn_concat, sattn_proj_q, sattn_proj_k, sattn_proj_v,
     sattn_scaled_scores) = attn_mask_forward_numpy(
        x, x, x, sattn_wq, sattn_wk, sattn_wv,
        sattn_wo, sattn_in_b, sattn_out_b, sattn_scale, sattn_mask)
    # Residual connection.
    sattn_dropout, sattn_dropout_mask = dropout(sattn, sattn_dropout_p)
    sattn_resid = sattn_dropout + x
    normed1, norm1_mean, norm1_std, norm1_normed = layer_norm(
        sattn_resid, norm1_scale, norm1_bias)
    # Encoder-decoder attention.
    (edattn, edattn_concat, edattn_proj_q, edattn_proj_k, edattn_proj_v,
     edattn_scaled_scores) = attn_forward_numpy(
        normed1, encoder_out, encoder_out,
        edattn_wq, edattn_wk, edattn_wv,
        edattn_wo, edattn_in_b, edattn_out_b, edattn_scale)
    # Residual connection.
    edattn_dropout, edattn_dropout_mask = dropout(edattn, edattn_dropout_p)
    edattn_resid = edattn_dropout + normed1
    normed2, norm2_mean, norm2_std, norm2_normed = layer_norm(
        edattn_resid, norm2_scale, norm2_bias)
    # Feedforward network.
    ff1_linear = linear(normed2, linear1_w, bias=linear1_b)
    ff1_act = act(ff1_linear)
    ff1, linear1_dropout_mask = dropout(ff1_act, linear1_dropout_p)
    ff = linear(ff1, linear2_w, bias=linear2_b)
    # Residual connection.
    ff_dropout, ff_dropout_mask = dropout(ff, ff_dropout_p)
    ff_resid = ff_dropout + normed2
    normed3, norm3_mean, norm3_std, norm3_normed = layer_norm(
        ff_resid, norm3_scale, norm3_bias)
    return (normed3,
            sattn_concat, sattn_proj_q, sattn_proj_k, sattn_proj_v,
            sattn_scaled_scores, sattn_dropout_mask,
            norm1_mean, norm1_std, norm1_normed,
            edattn_concat, edattn_proj_q, edattn_proj_k, edattn_proj_v,
            edattn_scaled_scores, edattn_dropout_mask,
            norm2_mean, norm2_std, norm2_normed,
            linear1_dropout_mask, ff_dropout_mask,
            norm3_mean, norm3_std, norm3_normed,
            ff_resid, ff1, ff1_linear, normed2,
            edattn_resid, normed1, sattn_resid)


def decoder_with_encoder_attention_backward(
        x, encoder_out, dy,
        sattn_concat, sattn_proj_q, sattn_proj_k, sattn_proj_v,
        sattn_scaled_scores, sattn_dropout_mask,
        norm1_mean, norm1_std, norm1_normed,
        edattn_concat, edattn_proj_q, edattn_proj_k, edattn_proj_v,
        edattn_scaled_scores, edattn_dropout_mask,
        norm2_mean, norm2_std, norm2_normed,
        linear1_dropout_mask, ff_dropout_mask,
        norm3_mean, norm3_std, norm3_normed,
        ff_resid, ff1, ff1_linear, normed2,
        edattn_resid, normed1, sattn_resid,
        sattn_wq, sattn_wk, sattn_wv, sattn_wo, sattn_scale, sattn_mask,
        edattn_wq, edattn_wk, edattn_wv, edattn_wo, edattn_scale,
        norm1_scale, norm1_bias,
        norm2_scale, norm2_bias,
        norm3_scale, norm3_bias,
        linear1_w, linear1_b, linear2_w, linear2_b,
        sattn_dropout_p, edattn_dropout_p, linear1_dropout_p, ff_dropout_p,
        activation='gelu'):
    """Backward data and weights for decoder with encoder attention.

    Arguments as in forward version.

    """
    act_backward_data = _get_activation_backward(activation)

    # Backward through norm3.
    dff_resid = layer_norm_backward_data(
        ff_resid, dy, norm3_mean, norm3_std, norm3_scale, norm3_bias)
    dnorm3_scale, dnorm3_bias = layer_norm_backward_weights(
        dy, norm3_normed, norm3_scale, norm3_bias)
    # Backward through residual connection.
    dff_dropout = dff_resid
    dnormed2_resid = dff_resid
    # Backward through FF dropout.
    dff = dropout_backward_data(dff_dropout, ff_dropout_p, ff_dropout_mask)
    # Backward through linear2.
    dff1 = linear_backward_data(ff1, linear2_w, dff)
    dlinear2_w, dlinear2_b = linear_backward_weights(
        ff1, linear2_w, dff, bias=linear2_b)
    # Backward through ff1 dropout.
    dff1_act = dropout_backward_data(
        dff1, linear1_dropout_p, linear1_dropout_mask)
    # Backward through ff1 activation.
    dff1_linear = act_backward_data(ff1_linear, dff1_act)
    # Backward through ff1 linear.
    dnormed2_linear = linear_backward_data(normed2, linear1_w, dff1_linear)
    dlinear1_w, dlinear1_b = linear_backward_weights(
        normed2, linear1_w, dff1_linear, bias=linear1_b)
    # Combine residuals.
    dnormed2 = dnormed2_resid + dnormed2_linear
    # Backward through norm2.
    dedattn_resid = layer_norm_backward_data(
        edattn_resid, dnormed2, norm2_mean, norm2_std, norm2_scale, norm2_bias)
    dnorm2_scale, dnorm2_bias = layer_norm_backward_weights(
        dnormed2, norm2_normed, norm2_scale, norm2_bias)
    # Backward through residual connection.
    dedattn_dropout = dedattn_resid
    dnormed1_resid = dedattn_resid
    # Backward through edattn dropout.
    dedattn = dropout_backward_data(
        dedattn_dropout, edattn_dropout_p, edattn_dropout_mask)
    # Backward through encoder/decoder attention.
    (dnormed1_attn, dencoder_out, dencoder_out_unused,
     dedattn_wq, dedattn_wk, dedattn_wv, dedattn_wo, dedattn_in_b, dedattn_out_b) = attn_backward_numpy(
         normed1, encoder_out, encoder_out, edattn_wq, edattn_wk, edattn_wv,
         edattn_wo, edattn_scale, dedattn,
         edattn_concat, edattn_proj_q, edattn_proj_k, edattn_proj_v, edattn_scaled_scores)
    # Combine from residual connection.
    dnormed1 = dnormed1_resid + dnormed1_attn
    # Backward through norm1.
    dsattn_resid = layer_norm_backward_data(
        sattn_resid, dnormed1, norm1_mean, norm1_std, norm1_scale, norm1_bias)
    dnorm1_scale, dnorm1_bias = layer_norm_backward_weights(
        dnormed1, norm1_normed, norm1_scale, norm1_bias)
    # Backward through residual connection.
    dsattn_dropout = dsattn_resid
    dx_resid = dsattn_resid
    # Backward through self-attention dropout.
    dsattn = dropout_backward_data(
        dsattn_dropout, sattn_dropout_p, sattn_dropout_mask)
    # Backward through self-attention.
    (dx_attn, dsk_unused, dsv_unused,
     dsattn_wq, dsattn_wk, dsattn_wv, dsattn_wo, dsattn_in_b, dsattn_out_b) = attn_backward_numpy(
         x, x, x, sattn_wq, sattn_wk, sattn_wv, sattn_wo, sattn_scale,
         dsattn,
         sattn_concat, sattn_proj_q, sattn_proj_k, sattn_proj_v, sattn_scaled_scores,
         mask=sattn_mask)
    # Finally compute dx.
    dx = dx_resid + dx_attn
    return (dx, dencoder_out,
            dsattn_wq, dsattn_wk, dsattn_wv, dsattn_wo, dsattn_in_b, dsattn_out_b,
            dedattn_wq, dedattn_wk, dedattn_wv, dedattn_wo, dedattn_in_b, dedattn_out_b,
            dnorm1_scale, dnorm1_bias,
            dnorm2_scale, dnorm2_bias,
            dnorm3_scale, dnorm3_bias,
            dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b)


def decoder(x, attn_wq, attn_wk, attn_wv, attn_wo, attn_in_b, attn_out_b, attn_scale, attn_mask,
            norm1_scale, norm1_bias, norm2_scale, norm2_bias,
            linear1_w, linear1_b, linear2_w, linear2_b,
            attn_dropout_p, linear1_dropout_p, ff_dropout_p,
            activation='gelu'):
    act = _get_activation_forward(activation)
    # Masked self-attention.
    (attn, attn_concat, attn_proj_q, attn_proj_k, attn_proj_v,
     attn_scaled_scores) = attn_mask_forward_numpy(
        x, x, x, attn_wq, attn_wk, attn_wv,
        attn_wo, attn_in_b, attn_out_b, attn_scale, attn_mask)
    # Residual connection.
    attn_dropout, attn_dropout_mask = dropout(attn, attn_dropout_p)
    attn_resid = attn_dropout + x
    normed1, norm1_mean, norm1_std, norm1_normed = layer_norm(
        attn_resid, norm1_scale, norm1_bias)
    # Feedforward network.
    ff1_linear = linear(normed1, linear1_w, bias=linear1_b)
    ff1_act = act(ff1_linear)
    ff1, linear1_dropout_mask = dropout(ff1_act, linear1_dropout_p)
    ff = linear(ff1, linear2_w, bias=linear2_b)
    # Residual connection.
    ff_dropout, ff_dropout_mask = dropout(ff, ff_dropout_p)
    ff_resid = ff_dropout + normed1
    normed2, norm2_mean, norm2_std, norm2_normed = layer_norm(
        ff_resid, norm2_scale, norm2_bias)
    return (normed2,
            attn_concat, attn_proj_q, attn_proj_k, attn_proj_v, attn_scaled_scores,
            attn_dropout_mask,
            norm1_mean, norm1_std, norm1_normed,
            linear1_dropout_mask, ff_dropout_mask,
            norm2_mean, norm2_std, norm2_normed,
            ff_resid, ff1, ff1_linear, normed1, attn_resid)


def decoder_backward(
        x, dy,
        attn_concat, attn_proj_q, attn_proj_k, attn_proj_v, attn_scaled_scores,
        attn_dropout_mask,
        norm1_mean, norm1_std, norm1_normed,
        linear1_dropout_mask, ff_dropout_mask,
        norm2_mean, norm2_std, norm2_normed,
        ff_resid, ff1, ff1_linear, normed1, attn_resid,
        attn_wq, attn_wk, attn_wv, attn_wo, attn_scale, attn_mask,
        norm1_scale, norm1_bias, norm2_scale, norm2_bias,
        linear1_w, linear1_b, linear2_w, linear2_b,
        attn_dropout_p, linear1_dropout_p, ff_dropout_p,
        activation='gelu'):
    """Backward data and weights for a decoder.

    Arguments as in forward version.

    """
    act_backward_data = _get_activation_backward(activation)

    # Backward through norm2.
    dff_resid = layer_norm_backward_data(
        ff_resid, dy, norm2_mean, norm2_std, norm2_scale, norm2_bias)
    dnorm2_scale, dnorm2_bias = layer_norm_backward_weights(
        dy, norm2_normed, norm2_scale, norm2_bias)
    # Backward through residual connection.
    # Note dnormed1 will be modified later.
    dff_dropout = dff_resid
    dnormed1_resid = dff_resid
    # Backward through FF dropout.
    dff = dropout_backward_data(dff_dropout, ff_dropout_p, ff_dropout_mask)
    # Backward through linear2.
    dff1 = linear_backward_data(ff1, linear2_w, dff)
    dlinear2_w, dlinear2_b = linear_backward_weights(
        ff1, linear2_w, dff, bias=linear2_b)
    # Backward through ff1 dropout.
    dff1_act = dropout_backward_data(
        dff1, linear1_dropout_p, linear1_dropout_mask)
    # Backward through ff1 activation.
    dff1_linear = act_backward_data(ff1_linear, dff1_act)
    # Backward through ff1 linear.
    dnormed1_linear = linear_backward_data(normed1, linear1_w, dff1_linear)
    dlinear1_w, dlinear1_b = linear_backward_weights(
        normed1, linear1_w, dff1_linear, bias=linear1_b)
    # Combine residuals.
    dnormed1 = dnormed1_resid + dnormed1_linear
    # Backward through norm1.
    dattn_resid = layer_norm_backward_data(
        attn_resid, dnormed1, norm1_mean, norm1_std, norm1_scale, norm1_bias)
    dnorm1_scale, dnorm1_bias = layer_norm_backward_weights(
        dnormed1, norm1_normed, norm1_scale, norm1_bias)
    # Backward through residual connection.
    dattn_dropout = dattn_resid
    dx_resid = dattn_resid
    # Backward through attention dropout.
    dattn = dropout_backward_data(
        dattn_dropout, attn_dropout_p, attn_dropout_mask)
    # Backward through self-attention.
    # dx, dk, and dv are the same.
    dx_attn, dk_unused, dv_unused, dattn_wq, dattn_wk, dattn_wv, dattn_wo, dattn_in_b, dattn_out_b = attn_backward_numpy(
        x, x, x, attn_wq, attn_wk, attn_wv, attn_wo, attn_scale, dattn,
        attn_concat, attn_proj_q, attn_proj_k, attn_proj_v, attn_scaled_scores,
        mask=attn_mask)
    # Finally compute dx.
    dx = dx_resid + dx_attn
    return (dx,
            dattn_wq, dattn_wk, dattn_wv, dattn_wo, dattn_in_b, dattn_out_b,
            dnorm1_scale, dnorm1_bias, dnorm2_scale, dnorm2_bias,
            dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b)
