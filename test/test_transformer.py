import argparse

import numpy as np

from substation import transformer

parser = argparse.ArgumentParser(
    description='Test NumPy transformer')
parser.add_argument(
    '--batch-size', default=3, type=int,
    help='Mini-batch size')
parser.add_argument(
    '--max-seq-len', default=7, type=int,
    help='Maximum sequence length')
parser.add_argument(
    '--max-enc-seq-len', default=5, type=int,
    help='Maximum encoder sequence length (for enc/dec attention)')
parser.add_argument(
    '--embed-size', default=8, type=int,
    help='Embedding size')
parser.add_argument(
    '--num-heads', default=4, type=int,
    help='Number of attention heads')
parser.add_argument(
    '--encdec-attn', default=False, action='store_true',
    help='Do encoder/decoder attention')
parser.add_argument(
    '--self-attn', default=False, action='store_true',
    help='Do self attention')
parser.add_argument(
    '--linear-neurons', default=None, type=int,
    help='Number of neurons in (internal) linear layer')
parser.add_argument(
    '--linear-bias', default=False, action='store_true',
    help='Use bias on linear layer')
parser.add_argument(
    '--softmax-dim', default=-1, type=int,
    help='Dimension to apply softmax over')
parser.add_argument(
    '--attn-mask', default=False, action='store_true',
    help='Do masked attention')
parser.add_argument(
    '--dropout-p', default=0.5, type=float,
    help='Dropout probability')
parser.add_argument(
    '--activation', default='relu',
    choices=['gelu', 'relu'],
    help='Activation in tranformer feedforward layer')
parser.add_argument(
    '--component', required=True,
    choices=['gelu', 'relu', 'linear', 'softmax', 'mha', 'layernorm', 'dropout',
            'encoder', 'decoder', 'encdec'],
    help='Which component to test')

def print_tensor(name, t):
    if t is None:
        print(f'{name}: None')
    else:
        print(f'{name} {t.shape}:\n{t}')

if __name__ == '__main__':
    args = parser.parse_args()
    x = np.random.randn(args.batch_size, args.max_seq_len, args.embed_size)
    print_tensor('x', x)
    if args.component == 'gelu':
        y = transformer.gelu(x)
        print_tensor('GeLU', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        print_tensor('dx', transformer.gelu_backward_data(x, dy))
    elif args.component == 'relu':
        y = transformer.relu(x)
        print_tensor('ReLU', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        print_tensor('dx', transformer.relu_backward_data(x, dy))
    elif args.component == 'linear':
        if not args.linear_neurons:
            raise ValueError('Must give --linear-neurons')
        w = np.random.randn(args.linear_neurons, args.embed_size)
        if args.linear_bias:
            bias = np.random.randn(args.linear_neurons)
        else:
            bias = None
        print_tensor('w', w)
        print_tensor('bias', bias)
        y = transformer.linear(x, w, bias=bias)
        print_tensor('Linear', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        dx = transformer.linear_backward_data(x, w, dy)
        print('dx', dx)
        dw, dbias = transformer.linear_backward_weights(x, w, dy, bias=bias)
        print('dw', dw)
        print('dbias', dbias)
    elif args.component == 'softmax':
        y = transformer.softmax(x, dim=args.softmax_dim)
        print_tensor('Softmax', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        print_tensor('dx', transformer.softmax_backward_data(
            y, dy, dim=args.softmax_dim))
    elif args.component == 'mha':
        if not args.num_heads:
            raise ValueError('Must give --num-heads')
        if args.embed_size % args.num_heads != 0:
            raise ValueError('Number of heads must evenly divide embedding size')
        proj_size = args.embed_size // args.num_heads
        # Construct weights with explicit head dimension.
        wq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        wk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        wv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        in_b = np.random.randn(3, args.num_heads, proj_size)
        # Output assumes concatenated results.
        wo = np.random.randn(args.embed_size, args.embed_size)
        out_b = np.random.randn(args.embed_size)
        # Generate inputs.
        q = x
        if args.self_attn:
            k = x
            v = x
        elif args.encdec_attn:
            if not args.max_enc_seq_len:
                raise ValueError('Must give --max-enc-seq-len')
            k = np.random.randn(args.batch_size, args.max_enc_seq_len, args.embed_size)
            v = k
        else:
            raise RuntimeError('Not supporting arbitrary attention right now')
        if args.attn_mask:
            mask = transformer.gen_attn_mask(q, k)
        else:
            mask = None
        scale = 1.0 / np.sqrt(proj_size)
        print_tensor('q', q)
        print_tensor('k', k)
        print_tensor('v', v)
        print_tensor('Wq', wq)
        print_tensor('Wk', wk)
        print_tensor('Wv', wv)
        print_tensor('Wo', wo)
        print_tensor('in_b', in_b)
        print_tensor('out_b', out_b)
        print_tensor('mask', mask)
        (y, i_concat, i_proj_q, i_proj_k, i_proj_v,
         i_scaled_scores) = transformer.attn_forward_numpy(
            q, k, v, wq, wk, wv, wo, in_b, out_b, scale, mask)
        print_tensor('MHA', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        dq, dk, dv, dwq, dwk, dwv, dwo, din_b, dout_b = transformer.attn_backward_numpy(
            q, k, v, wq, wk, wv, wo, scale, dy,
            i_concat, i_proj_q, i_proj_k, i_proj_v, i_scaled_scores, mask)
        print_tensor('dq', dq)
        print_tensor('dk', dk)
        print_tensor('dv', dv)
        print_tensor('dwq', dwq)
        print_tensor('dwk', dwk)
        print_tensor('dwv', dwv)
        print_tensor('dwo', dwo)
        print_tensor('din_b', din_b)
        print_tensor('dout_b', dout_b)
    elif args.component == 'layernorm':
        scale = np.random.randn(args.embed_size)
        bias = np.random.randn(args.embed_size)
        print_tensor('scale', scale)
        print_tensor('bias', bias)
        y, i_mean, i_std, i_normed = transformer.layer_norm(x, scale, bias)
        print_tensor('LayerNorm', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        dx = transformer.layer_norm_backward_data(x, dy, i_mean, i_std, scale, bias)
        print_tensor('dx', dx)
        dscale, dbias = transformer.layer_norm_backward_weights(dy, i_normed, scale, bias)
        print_tensor('dscale', dscale)
        print_tensor('dbias', dbias)
    elif args.component == 'dropout':
        y, mask = transformer.dropout(x, args.dropout_p)
        print_tensor('Dropout', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        dx = transformer.dropout_backward_data(dy, args.dropout_p, mask)
        print_tensor('dx', dx)
    elif args.component == 'encoder':
        proj_size = args.embed_size // args.num_heads
        wq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        wk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        wv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        in_b = np.random.randn(3, args.num_heads, proj_size)
        wo = np.random.randn(args.embed_size, args.embed_size)
        out_b = np.random.randn(args.embed_size)
        scale = 1.0 / np.sqrt(proj_size)
        norm1_scale = np.random.randn(args.embed_size)
        norm1_bias = np.random.randn(args.embed_size)
        norm2_scale = np.random.randn(args.embed_size)
        norm2_bias = np.random.randn(args.embed_size)
        linear1_w = np.random.randn(4*args.embed_size, args.embed_size)
        linear1_b = np.random.randn(4*args.embed_size)
        linear2_w = np.random.randn(args.embed_size, 4*args.embed_size)
        linear2_b = np.random.randn(args.embed_size)
        (y,
         i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
         i_attn_scaled_scores, i_attn_dropout_mask,
         i_norm1_mean, i_norm1_std, i_norm1_normed,
         i_linear1_dropout_mask, i_ff_dropout_mask,
         i_norm2_mean, i_norm2_std, i_norm2_normed,
         i_ff_resid, i_ff1, iff1_linear, i_normed1,
         i_attn_resid) = transformer.encoder(
             x, wq, wk, wv, wo, in_b, out_b, scale,
             norm1_scale, norm1_bias, norm2_scale, norm2_bias,
             linear1_w, linear1_b, linear2_w, linear2_b,
             0.5, 0.5, 0.5, activation=args.activation)
        print_tensor('Encoder', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        (dx,
         dattn_wq, dattn_wk, dattn_wv, dattn_wo, dattn_in_b, dattn_out_b,
         dnorm1_scale, dnorm1_bias, dnorm2_scale, dnorm2_bias,
         dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b) = transformer.encoder_backward(
             x, dy,
             i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
             i_attn_scaled_scores, i_attn_dropout_mask,
             i_norm1_mean, i_norm1_std, i_norm1_normed,
             i_linear1_dropout_mask, i_ff_dropout_mask,
             i_norm2_mean, i_norm2_std, i_norm2_normed,
             i_ff_resid, i_ff1, iff1_linear, i_normed1,
             i_attn_resid,
             wq, wk, wv, wo, scale,
             norm1_scale, norm1_bias, norm2_scale, norm2_bias,
             linear1_w, linear1_b, linear2_w, linear2_b,
             0.5, 0.5, 0.5, activation=args.activation)
        print_tensor('dx', dx)
        print_tensor('dattn_wq', dattn_wq)
        print_tensor('dattn_wk', dattn_wk)
        print_tensor('dattn_wv', dattn_wv)
        print_tensor('dattn_wo', dattn_wo)
        print_tensor('dattn_in_b', dattn_in_b)
        print_tensor('dattn_out_b', dattn_out_b)
        print_tensor('dnorm1_scale', dnorm1_scale)
        print_tensor('dnorm1_bias', dnorm1_bias)
        print_tensor('dnorm2_scale', dnorm2_scale)
        print_tensor('dnorm2_bias', dnorm2_bias)
        print_tensor('dlinear1_w', dlinear1_w)
        print_tensor('dlinear1_b', dlinear1_b)
        print_tensor('dlinear2_w', dlinear2_w)
        print_tensor('dlinear2_b', dlinear2_b)
    elif args.component == 'decoder':
        proj_size = args.embed_size // args.num_heads
        wq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        wk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        wv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        in_b = np.random.randn(3, args.num_heads, proj_size)
        wo = np.random.randn(args.embed_size, args.embed_size)
        out_b = np.random.randn(args.embed_size)
        scale = 1.0 / np.sqrt(proj_size)
        mask = transformer.gen_attn_mask(x, x)
        norm1_scale = np.random.randn(args.embed_size)
        norm1_bias = np.random.randn(args.embed_size)
        norm2_scale = np.random.randn(args.embed_size)
        norm2_bias = np.random.randn(args.embed_size)
        linear1_w = np.random.randn(4*args.embed_size, args.embed_size)
        linear1_b = np.random.randn(4*args.embed_size)
        linear2_w = np.random.randn(args.embed_size, 4*args.embed_size)
        linear2_b = np.random.randn(args.embed_size)
        (y, 
         i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
         i_attn_scaled_scores, i_attn_dropout_mask,
         i_norm1_mean, i_norm1_std, i_norm1_normed,
         i_linear1_dropout_mask, i_ff_dropout_mask,
         i_norm2_mean, i_norm2_std, i_norm2_normed,
         i_ff_resid, i_ff1, iff1_linear, i_normed1,
         i_attn_resid) = transformer.decoder(
            x, wq, wk, wv, wo, in_b, out_b, scale, mask,
            norm1_scale, norm1_bias, norm2_scale, norm2_bias,
            linear1_w, linear1_b, linear2_w, linear2_b,
            0.5, 0.5, 0.5, activation=args.activation)
        print_tensor('Decoder', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        (dx,
         dattn_wq, dattn_wk, dattn_wv, dattn_wo, dattn_in_b, dattn_out_b,
         dnorm1_scale, dnorm1_bias, dnorm2_scale, dnorm2_bias,
         dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b) = transformer.decoder_backward(
             x, dy,
             i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
             i_attn_scaled_scores, i_attn_dropout_mask,
             i_norm1_mean, i_norm1_std, i_norm1_normed,
             i_linear1_dropout_mask, i_ff_dropout_mask,
             i_norm2_mean, i_norm2_std, i_norm2_normed,
             i_ff_resid, i_ff1, iff1_linear, i_normed1,
             i_attn_resid,
             wq, wk, wv, wo, scale, mask,
             norm1_scale, norm1_bias, norm2_scale, norm2_bias,
             linear1_w, linear1_b, linear2_w, linear2_b,
             0.5, 0.5, 0.5, activation=args.activation)
        print_tensor('dx', dx)
        print_tensor('dattn_wq', dattn_wq)
        print_tensor('dattn_wk', dattn_wk)
        print_tensor('dattn_wv', dattn_wv)
        print_tensor('dattn_wo', dattn_wo)
        print_tensor('dattn_in_b', dattn_in_b)
        print_tensor('dattn_out_b', dattn_out_b)
        print_tensor('dnorm1_scale', dnorm1_scale)
        print_tensor('dnorm1_bias', dnorm1_bias)
        print_tensor('dnorm2_scale', dnorm2_scale)
        print_tensor('dnorm2_bias', dnorm2_bias)
        print_tensor('dlinear1_w', dlinear1_w)
        print_tensor('dlinear1_b', dlinear1_b)
        print_tensor('dlinear2_w', dlinear2_w)
        print_tensor('dlinear2_b', dlinear2_b)
    elif args.component == 'encdec':
        proj_size = args.embed_size // args.num_heads
        encoder_out = np.random.randn(args.batch_size, args.max_enc_seq_len, args.embed_size)
        swq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        swk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        swv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        sin_b = np.random.randn(3, args.num_heads, proj_size)
        swo = np.random.randn(args.embed_size, args.embed_size)
        sout_b = np.random.randn(args.embed_size)
        scale = 1.0 / np.sqrt(proj_size)
        mask = transformer.gen_attn_mask(x, x)
        edwq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        edwk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        edwv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        edin_b = np.random.randn(3, args.num_heads, proj_size)
        edwo = np.random.randn(args.embed_size, args.embed_size)
        edout_b = np.random.randn(args.embed_size)
        norm1_scale = np.random.randn(args.embed_size)
        norm1_bias = np.random.randn(args.embed_size)
        norm2_scale = np.random.randn(args.embed_size)
        norm2_bias = np.random.randn(args.embed_size)
        norm3_scale = np.random.randn(args.embed_size)
        norm3_bias = np.random.randn(args.embed_size)
        linear1_w = np.random.randn(4*args.embed_size, args.embed_size)
        linear1_b = np.random.randn(4*args.embed_size)
        linear2_w = np.random.randn(args.embed_size, 4*args.embed_size)
        linear2_b = np.random.randn(args.embed_size)
        (y,
         i_sattn_concat, i_sattn_proj_q, i_sattn_proj_k, i_sattn_proj_v,
         i_sattn_scaled_scores, i_sattn_dropout_mask,
         i_norm1_mean, i_norm1_std, i_norm1_normed,
         i_edattn_concat, i_edattn_proj_q, i_edattn_proj_k, i_edattn_proj_v,
         i_edattn_scaled_scores, i_edattn_dropout_mask,
         i_norm2_mean, i_norm2_std, i_norm2_normed,
         i_linear1_dropout_mask, i_ff_dropout_mask,
         i_norm3_mean, i_norm3_std, i_norm3_normed,
         i_ff_resid, i_ff1, iff1_linear, i_normed2,
         i_edattn_resid, i_normed1, i_sattn_resid) = transformer.decoder_with_encoder_attention(
            x, encoder_out,
            swq, swk, swv, swo, sin_b, sout_b, scale, mask,
            edwq, edwk, edwv, edwo, edin_b, edout_b, scale,
            norm1_scale, norm1_bias, norm2_scale, norm2_bias, norm3_scale, norm3_bias,
            linear1_w, linear1_b, linear2_w, linear2_b,
            0.5, 0.5, 0.5, 0.5, activation=args.activation)
        print_tensor('Decoder w/ Encoder', y)
        dy = np.random.randn(*y.shape)
        print_tensor('dy', dy)
        (dx, dencoder_out,
         dsattn_wq, dsattn_wk, dsattn_wv, dsattn_wo, dsattn_in_b, dsattn_out_b,
         dedattn_wq, dedattn_wk, dedattn_wv, dedattn_wo, dedattn_in_b, dedattn_out_b,
         dnorm1_scale, dnorm1_bias,
         dnorm2_scale, dnorm2_bias,
         dnorm3_scale, dnorm3_bias,
         dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b) = transformer.decoder_with_encoder_attention_backward(
             x, encoder_out, dy,
             i_sattn_concat, i_sattn_proj_q, i_sattn_proj_k, i_sattn_proj_v,
             i_sattn_scaled_scores, i_sattn_dropout_mask,
             i_norm1_mean, i_norm1_std, i_norm1_normed,
             i_edattn_concat, i_edattn_proj_q, i_edattn_proj_k, i_edattn_proj_v,
             i_edattn_scaled_scores, i_edattn_dropout_mask,
             i_norm2_mean, i_norm2_std, i_norm2_normed,
             i_linear1_dropout_mask, i_ff_dropout_mask,
             i_norm3_mean, i_norm3_std, i_norm3_normed,
             i_ff_resid, i_ff1, iff1_linear, i_normed2,
             i_edattn_resid, i_normed1, i_sattn_resid,
             swq, swk, swv, swo, scale, mask,
             edwq, edwk, edwv, edwo, scale,
             norm1_scale, norm1_bias, norm2_scale, norm2_bias, norm3_scale, norm3_bias,
             linear1_w, linear1_b, linear2_w, linear2_b,
             0.5, 0.5, 0.5, 0.5, activation=args.activation)
        print_tensor('dx', dx)
        print_tensor('dencoder_out', dencoder_out)
        print_tensor('dsattn_wq', dsattn_wq)
        print_tensor('dsattn_wk', dsattn_wk)
        print_tensor('dsattn_wv', dsattn_wv)
        print_tensor('dsattn_wo', dsattn_wo)
        print_tensor('dsattn_in_b', dsattn_in_b)
        print_tensor('dsattn_out_b', dsattn_out_b)
        print_tensor('dedattn_wq', dedattn_wq)
        print_tensor('dedattn_wk', dedattn_wk)
        print_tensor('dedattn_wv', dedattn_wv)
        print_tensor('dedattn_wo', dedattn_wo)
        print_tensor('dedattn_in_b', dedattn_in_b)
        print_tensor('dedattn_out_b', dedattn_out_b)
        print_tensor('dnorm1_scale', dnorm1_scale)
        print_tensor('dnorm1_bias', dnorm1_bias)
        print_tensor('dnorm2_scale', dnorm2_scale)
        print_tensor('dnorm2_bias', dnorm2_bias)
        print_tensor('dnorm3_scale', dnorm3_scale)
        print_tensor('dnorm3_bias', dnorm3_bias)
        print_tensor('dlinear1_w', dlinear1_w)
        print_tensor('dlinear1_b', dlinear1_b)
        print_tensor('dlinear2_w', dlinear2_w)
        print_tensor('dlinear2_b', dlinear2_b)
