import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from substation import attention, transformer

parser = argparse.ArgumentParser(
    description='Compare NumPy transformer with PyTorch')
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

if __name__ == '__main__':
    args = parser.parse_args()
    np_x = np.random.randn(args.batch_size, args.max_seq_len, args.embed_size)
    t_x = torch.from_numpy(np_x).requires_grad_()
    np_dw, t_dw = None, None
    if args.component == 'gelu':
        print('Warning: Torch GeLU uses a different approximation')
        # Forward
        np_r = transformer.gelu(np_x)
        t_r = F.gelu(t_x)
        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy)
        np_dx = transformer.gelu_backward_data(np_x, np_dy)
        t_r.backward(t_dy)
        t_dx = t_x.grad.detach().numpy()
        t_r = t_r.detach().numpy()
    elif args.component == 'relu':
        # Forward
        np_r = transformer.relu(np_x)
        t_r = F.relu(t_x)
        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy)
        np_dx = transformer.relu_backward_data(np_x, np_dy)
        t_r.backward(t_dy)
        t_dx = t_x.grad.detach().numpy()
        t_r = t_r.detach().numpy()
    elif args.component == 'linear':
        if not args.linear_neurons:
            raise ValueError('Must give --linear-neurons')
        # Forward
        np_w = np.random.randn(args.linear_neurons, args.embed_size)
        t_w = torch.from_numpy(np_w).requires_grad_()
        if args.linear_bias:
            np_bias = np.random.randn(args.linear_neurons)
            t_bias = torch.from_numpy(np_bias).requires_grad_()
        else:
            np_bias = None
            t_bias = None
        np_r = transformer.linear(np_x, np_w, bias=np_bias)
        t_r = F.linear(t_x, t_w, bias=t_bias)
        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy)
        np_dx = transformer.linear_backward_data(np_x, np_w, np_dy)
        np_dw = transformer.linear_backward_weights(np_x, np_w, np_dy, bias=np_bias)
        t_r.backward(t_dy)
        t_dx = t_x.grad.detach().numpy()
        t_dw = (t_w.grad.detach().numpy(), None if t_bias is None else t_bias.grad.detach().numpy())        
        t_r = t_r.detach().numpy()
    elif args.component == 'softmax':
        # Forward
        np_r = transformer.softmax(np_x, dim=args.softmax_dim)
        t_r = F.softmax(t_x, dim=args.softmax_dim)
        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy)
        np_dx = transformer.softmax_backward_data(np_r, np_dy, dim=args.softmax_dim)
        t_r.backward(t_dy)
        t_dx = t_x.grad.detach().numpy()
        t_r = t_r.detach().numpy()
    elif args.component == 'mha':
        # Torch attention expects (sequence, batch, embedding) order.
        if not args.num_heads:
            raise ValueError('Must give --num-heads')
        if args.embed_size % args.num_heads != 0:
            raise ValueError('Number of heads must evenly divide embedding size')
        # Forward
        proj_size = args.embed_size // args.num_heads
        np_wq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wq = torch.from_numpy(np_wq).reshape((args.embed_size, args.embed_size))
        np_wk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wk = torch.from_numpy(np_wk).reshape((args.embed_size, args.embed_size))
        np_wv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wv = torch.from_numpy(np_wv).reshape((args.embed_size, args.embed_size))
        t_input_weights = torch.cat((t_wq, t_wk, t_wv), dim=0).requires_grad_()
        np_in_b = np.random.randn(3, args.num_heads, proj_size)
        t_in_b = torch.from_numpy(np_in_b).reshape(3*args.embed_size).requires_grad_()
        np_wo = np.random.randn(args.embed_size, args.embed_size)
        t_wo = torch.from_numpy(np_wo).requires_grad_()
        np_out_b = np.random.randn(args.embed_size)
        t_out_b = torch.from_numpy(np_out_b).requires_grad_()
        scale = 1.0 / np.sqrt(proj_size)
        np_q = np_x
        t_q = torch.from_numpy(np_q).transpose(0, 1).requires_grad_()
        if args.self_attn:
            np_k = np_q
            np_v = np_q
            t_k = t_q
            t_v = t_q
        elif args.encdec_attn:
            if not args.max_enc_seq_len:
                raise ValueError('Must give --max-enc-seq-len')
            np_k = np.random.randn(args.batch_size, args.max_enc_seq_len, args.embed_size)
            np_v = np_k
            t_k = torch.from_numpy(np_k).transpose(0, 1).requires_grad_()
            t_v = t_k
        else:
            raise RuntimeError('Not supporting arbitrary attention right now')
        if args.attn_mask:
            np_mask = transformer.gen_attn_mask(np_q, np_k)
            t_mask = torch.from_numpy(np_mask)
            (np_r, i_concat, i_proj_q, i_proj_k, i_proj_v,
             i_scaled_scores) = attention.attn_mask_forward_numpy(
                np_q, np_k, np_v, np_wq,
                np_wk, np_wv, np_wo, np_in_b, np_out_b, scale,
                np_mask)
        else:
            np_mask = None
            t_mask = None
            (np_r, i_concat, i_proj_q, i_proj_k, i_proj_v,
             i_scaled_scores) = attention.attn_forward_numpy(
                np_q, np_k, np_v, np_wq,
                np_wk, np_wv, np_wo, np_in_b, np_out_b, scale)

        # Easier to do this than figure out the not-publicly-documented
        # functional API.
        attn = nn.MultiheadAttention(args.embed_size, args.num_heads)
        attn.in_proj_weight.data = t_input_weights
        attn.in_proj_bias.data = t_in_b
        attn.out_proj.weight.data = t_wo
        attn.out_proj.bias.data = t_out_b
        attn.train()
        t_r = attn.forward(t_q, t_k, t_v, need_weights=False, attn_mask=t_mask)[0]

        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy).transpose(0, 1)
        np_dq, np_dk, np_dv, np_dwq, np_dwk, np_dwv, np_dwo, np_din_b, np_dout_b = attention.attn_backward_numpy(
            np_q, np_k, np_v, np_wq, np_wk, np_wv, np_wo, scale, np_dy,
            i_concat, i_proj_q, i_proj_k, i_proj_v, i_scaled_scores, np_mask)
        np_dx = (np_dq, np_dk, np_dv)
        np_dw = (np_dwq, np_dwk, np_dwv, np_dwo, np_din_b, np_dout_b)
        t_r.backward(t_dy)
        t_dq = t_q.grad.detach().numpy().transpose(1, 0, 2)
        t_dk = t_k.grad.detach().numpy().transpose(1, 0, 2)
        t_dv = t_v.grad.detach().numpy().transpose(1, 0, 2)
        t_dx = (t_dq, t_dk, t_dv)
        t_dout_b = attn.out_proj.bias.grad.detach().numpy()
        t_dwo = attn.out_proj.weight.grad.detach().numpy()
        t_din_b = attn.in_proj_bias.grad.detach().reshape(3, args.num_heads, proj_size).numpy()
        t_dinput_weights = torch.chunk(attn.in_proj_weight.grad, 3, dim=0)
        t_dwq = t_dinput_weights[0].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dwk = t_dinput_weights[1].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dwv = t_dinput_weights[2].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dw = (t_dwq, t_dwk, t_dwv, t_dwo, t_din_b, t_dout_b)
        t_r = t_r.detach().transpose(0, 1).numpy()
    elif args.component == 'layernorm':
        # Forward
        np_scale = np.random.randn(args.embed_size)
        np_bias = np.random.randn(args.embed_size)
        t_scale = torch.from_numpy(np_scale).requires_grad_()
        t_bias = torch.from_numpy(np_bias).requires_grad_()
        np_r, i_mean, i_std, i_normed = transformer.layer_norm(
            np_x, np_scale, np_bias)
        t_r = F.layer_norm(t_x, (args.embed_size,), t_scale, t_bias)
        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy)
        np_dx = transformer.layer_norm_backward_data(
            np_x, np_dy, i_mean, i_std, np_scale, np_bias)
        np_dscale, np_dbias = transformer.layer_norm_backward_weights(
            np_dy, i_normed, np_scale, np_bias)
        np_dw = (np_dscale, np_dbias)
        t_r.backward(t_dy)
        t_dx = t_x.grad.detach().numpy()
        t_dw = (t_scale.grad.detach().numpy(), t_bias.grad.detach().numpy())
        t_r = t_r.detach().numpy()
    elif args.component == 'dropout':
        print('Cannot test dropout due to randomness')
        sys.exit(0)
    elif args.component == 'encoder':
        # Does not use dropout due to randomness.
        # Set up all the weights.
        proj_size = args.embed_size // args.num_heads
        np_wq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wq = torch.from_numpy(np_wq).reshape((args.embed_size, args.embed_size))
        np_wk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wk = torch.from_numpy(np_wk).reshape((args.embed_size, args.embed_size))
        np_wv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wv = torch.from_numpy(np_wv).reshape((args.embed_size, args.embed_size))
        t_input_weights = torch.cat((t_wq, t_wk, t_wv), dim=0).requires_grad_()
        np_in_b = np.random.randn(3, args.num_heads, proj_size)
        t_in_b = torch.from_numpy(np_in_b).reshape(3*args.embed_size).requires_grad_()
        np_wo = np.random.randn(args.embed_size, args.embed_size)
        t_wo = torch.from_numpy(np_wo).requires_grad_()
        np_out_b = np.random.randn(args.embed_size)
        t_out_b = torch.from_numpy(np_out_b).requires_grad_()
        scale = 1.0 / np.sqrt(proj_size)
        np_norm1_scale = np.random.randn(args.embed_size)
        t_norm1_scale = torch.from_numpy(np_norm1_scale).requires_grad_()
        np_norm1_bias = np.random.randn(args.embed_size)
        t_norm1_bias = torch.from_numpy(np_norm1_bias).requires_grad_()
        np_norm2_scale = np.random.randn(args.embed_size)
        t_norm2_scale = torch.from_numpy(np_norm2_scale).requires_grad_()
        np_norm2_bias = np.random.randn(args.embed_size)
        t_norm2_bias = torch.from_numpy(np_norm2_bias).requires_grad_()
        np_linear1_w = np.random.randn(4*args.embed_size, args.embed_size)
        t_linear1_w = torch.from_numpy(np_linear1_w).requires_grad_()
        np_linear1_b = np.random.randn(4*args.embed_size)
        t_linear1_b = torch.from_numpy(np_linear1_b).requires_grad_()
        np_linear2_w = np.random.randn(args.embed_size, 4*args.embed_size)
        t_linear2_w = torch.from_numpy(np_linear2_w).requires_grad_()
        np_linear2_b = np.random.randn(args.embed_size)
        t_linear2_b = torch.from_numpy(np_linear2_b).requires_grad_()
        np_seq = np_x
        t_seq = torch.from_numpy(np_seq).transpose(0, 1).requires_grad_()
        (np_r,
         i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
         i_attn_scaled_scores, i_attn_dropout_mask,
         i_norm1_mean, i_norm1_std, i_norm1_normed,
         i_linear1_dropout_mask, i_ff_dropout_mask,
         i_norm2_mean, i_norm2_std, i_norm2_normed,
         i_ff_resid, i_ff1, iff1_linear, i_normed1,
         i_attn_resid) = transformer.encoder(
            np_seq, np_wq, np_wk, np_wv, np_wo, np_in_b, np_out_b, scale,
            np_norm1_scale, np_norm1_bias, np_norm2_scale, np_norm2_bias,
            np_linear1_w, np_linear1_b, np_linear2_w, np_linear2_b,
            0, 0, 0, activation=args.activation)
        # Stuff everything into the PyTorch layer.
        t_encoder = nn.TransformerEncoderLayer(
            args.embed_size, args.num_heads, dim_feedforward=4*args.embed_size,
            dropout=0, activation=args.activation)
        t_encoder.self_attn.in_proj_weight.data = t_input_weights
        t_encoder.self_attn.in_proj_bias.data = t_in_b
        t_encoder.self_attn.out_proj.weight.data = t_wo
        t_encoder.self_attn.out_proj.bias.data = t_out_b
        t_encoder.linear1.weight.data = t_linear1_w
        t_encoder.linear1.bias.data = t_linear1_b
        t_encoder.linear2.weight.data = t_linear2_w
        t_encoder.linear2.bias.data = t_linear2_b
        t_encoder.norm1.weight.data = t_norm1_scale
        t_encoder.norm1.bias.data = t_norm1_bias
        t_encoder.norm2.weight.data = t_norm2_scale
        t_encoder.norm2.bias.data = t_norm2_bias
        t_encoder.train()
        t_r = t_encoder.forward(t_seq)
        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy).transpose(0, 1)
        (np_dx,
         np_dattn_wq, np_dattn_wk, np_dattn_wv, np_dattn_wo, np_dattn_in_b, np_dattn_out_b,
         np_dnorm1_scale, np_dnorm1_bias, np_dnorm2_scale, np_dnorm2_bias,
         np_dlinear1_w, np_dlinear1_b, np_dlinear2_w, np_dlinear2_b) = transformer.encoder_backward(
             np_seq, np_dy,
             i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
             i_attn_scaled_scores, i_attn_dropout_mask,
             i_norm1_mean, i_norm1_std, i_norm1_normed,
             i_linear1_dropout_mask, i_ff_dropout_mask,
             i_norm2_mean, i_norm2_std, i_norm2_normed,
             i_ff_resid, i_ff1, iff1_linear, i_normed1,
             i_attn_resid,
             np_wq, np_wk, np_wv, np_wo, scale,
             np_norm1_scale, np_norm1_bias, np_norm2_scale, np_norm2_bias,
             np_linear1_w, np_linear1_b, np_linear2_w, np_linear2_b,
             0, 0, 0, activation=args.activation)
        np_dw = (np_dattn_wq, np_dattn_wk, np_dattn_wv, np_dattn_wo, np_dattn_in_b, np_dattn_out_b,
                 np_dnorm1_scale, np_dnorm1_bias, np_dnorm2_scale, np_dnorm2_bias,
                 np_dlinear1_w, np_dlinear1_b, np_dlinear2_w, np_dlinear2_b)
        t_r.backward(t_dy)
        t_dx = t_seq.grad.detach().numpy().transpose(1, 0, 2)
        t_dout_b = t_encoder.self_attn.out_proj.bias.grad.detach().numpy()
        t_dwo = t_encoder.self_attn.out_proj.weight.grad.detach().numpy()
        t_din_b = t_encoder.self_attn.in_proj_bias.grad.detach().reshape(3, args.num_heads, proj_size).numpy()
        t_dinput_weights = torch.chunk(
            t_encoder.self_attn.in_proj_weight.grad, 3, dim=0)
        t_dwq = t_dinput_weights[0].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dwk = t_dinput_weights[1].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dwv = t_dinput_weights[2].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dnorm1_scale = t_encoder.norm1.weight.grad.detach().numpy()
        t_dnorm1_bias = t_encoder.norm1.bias.grad.detach().numpy()
        t_dnorm2_scale = t_encoder.norm2.weight.grad.detach().numpy()
        t_dnorm2_bias = t_encoder.norm2.bias.grad.detach().numpy()
        t_dlinear1_w = t_encoder.linear1.weight.grad.detach().numpy()
        t_dlinear1_b = t_encoder.linear1.bias.grad.detach().numpy()
        t_dlinear2_w = t_encoder.linear2.weight.grad.detach().numpy()
        t_dlinear2_b = t_encoder.linear2.bias.grad.detach().numpy()
        t_dw = (t_dwq, t_dwk, t_dwv, t_dwo, t_din_b, t_dout_b,
                t_dnorm1_scale, t_dnorm1_bias, t_dnorm2_scale, t_dnorm2_bias,
                t_dlinear1_w, t_dlinear1_b, t_dlinear2_w, t_dlinear2_b)
        t_r = t_r.detach().transpose(0, 1).numpy()
    elif args.component == 'decoder':
        # Does not use dropout due to randomness.
        # Set up all the weights.
        proj_size = args.embed_size // args.num_heads
        np_wq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wq = torch.from_numpy(np_wq).reshape((args.embed_size, args.embed_size))
        np_wk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wk = torch.from_numpy(np_wk).reshape((args.embed_size, args.embed_size))
        np_wv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_wv = torch.from_numpy(np_wv).reshape((args.embed_size, args.embed_size))
        t_input_weights = torch.cat((t_wq, t_wk, t_wv), dim=0).requires_grad_()
        np_in_b = np.random.randn(3, args.num_heads, proj_size)
        t_in_b = torch.from_numpy(np_in_b).reshape(3*args.embed_size).requires_grad_()
        np_wo = np.random.randn(args.embed_size, args.embed_size)
        t_wo = torch.from_numpy(np_wo).requires_grad_()
        np_out_b = np.random.randn(args.embed_size)
        t_out_b = torch.from_numpy(np_out_b).requires_grad_()
        scale = 1.0 / np.sqrt(proj_size)
        np_mask = transformer.gen_attn_mask(np_x, np_x)
        t_mask = torch.from_numpy(np_mask)
        np_norm1_scale = np.random.randn(args.embed_size)
        t_norm1_scale = torch.from_numpy(np_norm1_scale).requires_grad_()
        np_norm1_bias = np.random.randn(args.embed_size)
        t_norm1_bias = torch.from_numpy(np_norm1_bias).requires_grad_()
        np_norm2_scale = np.random.randn(args.embed_size)
        t_norm2_scale = torch.from_numpy(np_norm2_scale).requires_grad_()
        np_norm2_bias = np.random.randn(args.embed_size)
        t_norm2_bias = torch.from_numpy(np_norm2_bias).requires_grad_()
        np_linear1_w = np.random.randn(4*args.embed_size, args.embed_size)
        t_linear1_w = torch.from_numpy(np_linear1_w).requires_grad_()
        np_linear1_b = np.random.randn(4*args.embed_size)
        t_linear1_b = torch.from_numpy(np_linear1_b).requires_grad_()
        np_linear2_w = np.random.randn(args.embed_size, 4*args.embed_size)
        t_linear2_w = torch.from_numpy(np_linear2_w).requires_grad_()
        np_linear2_b = np.random.randn(args.embed_size)
        t_linear2_b = torch.from_numpy(np_linear2_b).requires_grad_()
        np_seq = np_x
        t_seq = torch.from_numpy(np_seq).transpose(0, 1).requires_grad_()
        (np_r,
         i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
         i_attn_scaled_scores, i_attn_dropout_mask,
         i_norm1_mean, i_norm1_std, i_norm1_normed,
         i_linear1_dropout_mask, i_ff_dropout_mask,
         i_norm2_mean, i_norm2_std, i_norm2_normed,
         i_ff_resid, i_ff1, iff1_linear, i_normed1,
         i_attn_resid) = transformer.decoder(
            np_seq, np_wq, np_wk, np_wv, np_wo, np_in_b, np_out_b, scale, np_mask,
            np_norm1_scale, np_norm1_bias, np_norm2_scale, np_norm2_bias,
            np_linear1_w, np_linear1_b, np_linear2_w, np_linear2_b,
            0, 0, 0, activation=args.activation)
        # Stuff everything into the PyTorch layer.
        # The PyTorch encoder is the same as our decoder, if we pass it
        # an attention mask.
        t_decoder = nn.TransformerEncoderLayer(
            args.embed_size, args.num_heads, dim_feedforward=4*args.embed_size,
            dropout=0, activation=args.activation)
        t_decoder.self_attn.in_proj_weight.data = t_input_weights
        t_decoder.self_attn.in_proj_bias.data = t_in_b
        t_decoder.self_attn.out_proj.weight.data = t_wo
        t_decoder.self_attn.out_proj.bias.data = t_out_b
        t_decoder.linear1.weight.data = t_linear1_w
        t_decoder.linear1.bias.data = t_linear1_b
        t_decoder.linear2.weight.data = t_linear2_w
        t_decoder.linear2.bias.data = t_linear2_b
        t_decoder.norm1.weight.data = t_norm1_scale
        t_decoder.norm1.bias.data = t_norm1_bias
        t_decoder.norm2.weight.data = t_norm2_scale
        t_decoder.norm2.bias.data = t_norm2_bias
        t_decoder.train()
        t_r = t_decoder.forward(t_seq, src_mask=t_mask)
        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy).transpose(0, 1)
        (np_dx,
         np_dattn_wq, np_dattn_wk, np_dattn_wv, np_dattn_wo, np_dattn_in_b, np_dattn_out_b,
         np_dnorm1_scale, np_dnorm1_bias, np_dnorm2_scale, np_dnorm2_bias,
         np_dlinear1_w, np_dlinear1_b, np_dlinear2_w, np_dlinear2_b) = transformer.decoder_backward(
             np_seq, np_dy,
             i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
             i_attn_scaled_scores, i_attn_dropout_mask,
             i_norm1_mean, i_norm1_std, i_norm1_normed,
             i_linear1_dropout_mask, i_ff_dropout_mask,
             i_norm2_mean, i_norm2_std, i_norm2_normed,
             i_ff_resid, i_ff1, iff1_linear, i_normed1,
             i_attn_resid,
             np_wq, np_wk, np_wv, np_wo, scale, np_mask,
             np_norm1_scale, np_norm1_bias, np_norm2_scale, np_norm2_bias,
             np_linear1_w, np_linear1_b, np_linear2_w, np_linear2_b,
             0, 0, 0, activation=args.activation)
        np_dw = (np_dattn_wq, np_dattn_wk, np_dattn_wv, np_dattn_wo, np_dattn_in_b, np_dattn_out_b,
                 np_dnorm1_scale, np_dnorm1_bias, np_dnorm2_scale, np_dnorm2_bias,
                 np_dlinear1_w, np_dlinear1_b, np_dlinear2_w, np_dlinear2_b)
        t_r.backward(t_dy)
        t_dx = t_seq.grad.detach().numpy().transpose(1, 0, 2)
        t_dout_b = t_decoder.self_attn.out_proj.bias.grad.detach().numpy()
        t_dwo = t_decoder.self_attn.out_proj.weight.grad.detach().numpy()
        t_din_b = t_decoder.self_attn.in_proj_bias.grad.detach().reshape(3, args.num_heads, proj_size).numpy()
        t_dinput_weights = torch.chunk(
            t_decoder.self_attn.in_proj_weight.grad, 3, dim=0)
        t_dwq = t_dinput_weights[0].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dwk = t_dinput_weights[1].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dwv = t_dinput_weights[2].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dnorm1_scale = t_decoder.norm1.weight.grad.detach().numpy()
        t_dnorm1_bias = t_decoder.norm1.bias.grad.detach().numpy()
        t_dnorm2_scale = t_decoder.norm2.weight.grad.detach().numpy()
        t_dnorm2_bias = t_decoder.norm2.bias.grad.detach().numpy()
        t_dlinear1_w = t_decoder.linear1.weight.grad.detach().numpy()
        t_dlinear1_b = t_decoder.linear1.bias.grad.detach().numpy()
        t_dlinear2_w = t_decoder.linear2.weight.grad.detach().numpy()
        t_dlinear2_b = t_decoder.linear2.bias.grad.detach().numpy()
        t_dw = (t_dwq, t_dwk, t_dwv, t_dwo, t_din_b, t_dout_b,
                t_dnorm1_scale, t_dnorm1_bias, t_dnorm2_scale, t_dnorm2_bias,
                t_dlinear1_w, t_dlinear1_b, t_dlinear2_w, t_dlinear2_b)
        t_r = t_r.detach().transpose(0, 1).numpy()
    elif args.component == 'encdec':
        # Does not use dropout due to randomness.
        # Set up all the weights.
        proj_size = args.embed_size // args.num_heads
        np_swq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_swq = torch.from_numpy(np_swq).reshape((args.embed_size, args.embed_size))
        np_swk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_swk = torch.from_numpy(np_swk).reshape((args.embed_size, args.embed_size))
        np_swv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_swv = torch.from_numpy(np_swv).reshape((args.embed_size, args.embed_size))
        t_sattn_input_weights = torch.cat((t_swq, t_swk, t_swv), dim=0).requires_grad_()
        np_sin_b = np.random.randn(3, args.num_heads, proj_size)
        t_sin_b = torch.from_numpy(np_sin_b).reshape(3*args.embed_size).requires_grad_()
        np_swo = np.random.randn(args.embed_size, args.embed_size)
        t_swo = torch.from_numpy(np_swo).requires_grad_()
        np_sout_b = np.random.randn(args.embed_size)
        t_sout_b = torch.from_numpy(np_sout_b).requires_grad_()
        scale = 1.0 / np.sqrt(proj_size)
        np_mask = transformer.gen_attn_mask(np_x, np_x)
        t_mask = torch.from_numpy(np_mask)
        np_edwq = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_edwq = torch.from_numpy(np_edwq).reshape((args.embed_size, args.embed_size))
        np_edwk = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_edwk = torch.from_numpy(np_edwk).reshape((args.embed_size, args.embed_size))
        np_edwv = np.random.randn(args.num_heads, proj_size, args.embed_size)
        t_edwv = torch.from_numpy(np_edwv).reshape((args.embed_size, args.embed_size))
        t_edattn_input_weights = torch.cat((t_edwq, t_edwk, t_edwv), dim=0).requires_grad_()
        np_edin_b = np.random.randn(3, args.num_heads, proj_size)
        t_edin_b = torch.from_numpy(np_edin_b).reshape(3*args.embed_size).requires_grad_()
        np_edwo = np.random.randn(args.embed_size, args.embed_size)
        t_edwo = torch.from_numpy(np_edwo).requires_grad_()
        np_edout_b = np.random.randn(args.embed_size)
        t_edout_b = torch.from_numpy(np_edout_b).requires_grad_()
        np_norm1_scale = np.random.randn(args.embed_size)
        t_norm1_scale = torch.from_numpy(np_norm1_scale).requires_grad_()
        np_norm1_bias = np.random.randn(args.embed_size)
        t_norm1_bias = torch.from_numpy(np_norm1_bias).requires_grad_()
        np_norm2_scale = np.random.randn(args.embed_size)
        t_norm2_scale = torch.from_numpy(np_norm2_scale).requires_grad_()
        np_norm2_bias = np.random.randn(args.embed_size)
        t_norm2_bias = torch.from_numpy(np_norm2_bias).requires_grad_()
        np_norm3_scale = np.random.randn(args.embed_size)
        t_norm3_scale = torch.from_numpy(np_norm3_scale).requires_grad_()
        np_norm3_bias = np.random.randn(args.embed_size)
        t_norm3_bias = torch.from_numpy(np_norm3_bias).requires_grad_()
        np_linear1_w = np.random.randn(4*args.embed_size, args.embed_size)
        t_linear1_w = torch.from_numpy(np_linear1_w).requires_grad_()
        np_linear1_b = np.random.randn(4*args.embed_size)
        t_linear1_b = torch.from_numpy(np_linear1_b).requires_grad_()
        np_linear2_w = np.random.randn(args.embed_size, 4*args.embed_size)
        t_linear2_w = torch.from_numpy(np_linear2_w).requires_grad_()
        np_linear2_b = np.random.randn(args.embed_size)
        t_linear2_b = torch.from_numpy(np_linear2_b).requires_grad_()
        np_seq = np_x
        t_seq = torch.from_numpy(np_seq).transpose(0, 1).requires_grad_()
        np_encoder_out = np.random.randn(args.batch_size, args.max_enc_seq_len, args.embed_size)
        t_encoder_out = torch.from_numpy(np_encoder_out).transpose(0, 1).requires_grad_()
        (np_r,
         i_sattn_concat, i_sattn_proj_q, i_sattn_proj_k, i_sattn_proj_v,
         i_sattn_scaled_scores, i_sattn_dropout_mask,
         i_norm1_mean, i_norm1_std, i_norm1_normed,
         i_edattn_concat, i_edattn_proj_q, i_edattn_proj_k, i_edattn_proj_v,
         i_edattn_scaled_scores, i_edattn_dropout_mask,
         i_norm2_mean, i_norm2_std, i_norm2_normed,
         i_linear1_dropout_mask, i_ff_dropout_mask,
         i_norm3_mean, i_norm3_std, i_norm3_normed,
         i_ff_resid, i_ff1, i_ff1_linear, i_normed2,
         i_edattn_resid, i_normed1, i_sattn_resid) = transformer.decoder_with_encoder_attention(
            np_seq, np_encoder_out,
            np_swq, np_swk, np_swv, np_swo, np_sin_b, np_sout_b, scale, np_mask,
            np_edwq, np_edwk, np_edwv, np_edwo, np_edin_b, np_edout_b, scale,
            np_norm1_scale, np_norm1_bias,
            np_norm2_scale, np_norm2_bias,
            np_norm3_scale, np_norm3_bias,
            np_linear1_w, np_linear1_b, np_linear2_w, np_linear2_b,
            0, 0, 0, 0, activation=args.activation)
        # Stuff everything into the PyTorch layer.
        t_decoder = nn.TransformerDecoderLayer(
            args.embed_size, args.num_heads, dim_feedforward=4*args.embed_size,
            dropout=0, activation=args.activation)
        t_decoder.self_attn.in_proj_weight.data = t_sattn_input_weights
        t_decoder.self_attn.in_proj_bias.data = t_sin_b
        t_decoder.self_attn.out_proj.weight.data = t_swo
        t_decoder.self_attn.out_proj.bias.data = t_sout_b
        t_decoder.multihead_attn.in_proj_weight.data = t_edattn_input_weights
        t_decoder.multihead_attn.in_proj_bias.data = t_edin_b
        t_decoder.multihead_attn.out_proj.weight.data = t_edwo
        t_decoder.multihead_attn.out_proj.bias.data = t_edout_b
        t_decoder.linear1.weight.data = t_linear1_w
        t_decoder.linear1.bias.data = t_linear1_b
        t_decoder.linear2.weight.data = t_linear2_w
        t_decoder.linear2.bias.data = t_linear2_b
        t_decoder.norm1.weight.data = t_norm1_scale
        t_decoder.norm1.bias.data = t_norm1_bias
        t_decoder.norm2.weight.data = t_norm2_scale
        t_decoder.norm2.bias.data = t_norm2_bias
        t_decoder.norm3.weight.data = t_norm3_scale
        t_decoder.norm3.bias.data = t_norm3_bias
        t_decoder.train()
        t_r = t_decoder.forward(t_seq, t_encoder_out, tgt_mask=t_mask)
        # Backward
        np_dy = np.random.randn(*np_r.shape)
        t_dy = torch.from_numpy(np_dy).transpose(0, 1)
        (np_dseq, np_dencoder_out,
         np_dswq, np_dswk, np_dswv, np_dswo, np_dsin_b, np_dsout_b,
         np_dedwq, np_dedwk, np_dedwv, np_dedwo, np_dedin_b, np_dedout_b,
         np_dnorm1_scale, np_dnorm1_bias,
         np_dnorm2_scale, np_dnorm2_bias,
         np_dnorm3_scale, np_dnorm3_bias,
         np_dlinear1_w, np_dlinear1_b, np_dlinear2_w, np_dlinear2_b) = transformer.decoder_with_encoder_attention_backward(
             np_seq, np_encoder_out, np_dy,
             i_sattn_concat, i_sattn_proj_q, i_sattn_proj_k, i_sattn_proj_v,
             i_sattn_scaled_scores, i_sattn_dropout_mask,
             i_norm1_mean, i_norm1_std, i_norm1_normed,
             i_edattn_concat, i_edattn_proj_q, i_edattn_proj_k, i_edattn_proj_v,
             i_edattn_scaled_scores, i_edattn_dropout_mask,
             i_norm2_mean, i_norm2_std, i_norm2_normed,
             i_linear1_dropout_mask, i_ff_dropout_mask,
             i_norm3_mean, i_norm3_std, i_norm3_normed,
             i_ff_resid, i_ff1, i_ff1_linear, i_normed2,
             i_edattn_resid, i_normed1, i_sattn_resid,
             np_swq, np_swk, np_swv, np_swo, scale, np_mask,
             np_edwq, np_edwk, np_edwv, np_edwo, scale,
             np_norm1_scale, np_norm1_bias,
             np_norm2_scale, np_norm2_bias,
             np_norm3_scale, np_norm3_bias,
             np_linear1_w, np_linear1_b, np_linear2_w, np_linear2_b,
             0, 0, 0, 0, activation=args.activation)
        np_dx = (np_dseq, np_dencoder_out)
        np_dw = (np_dswq, np_dswk, np_dswv, np_dswo, np_dsin_b, np_dsout_b,
                 np_dedwq, np_dedwk, np_dedwv, np_dedwo, np_dedin_b, np_dedout_b,
                 np_dnorm1_scale, np_dnorm1_bias,
                 np_dnorm2_scale, np_dnorm2_bias,
                 np_dnorm3_scale, np_dnorm3_bias,
                 np_dlinear1_w, np_dlinear1_b, np_dlinear2_w, np_dlinear2_b)
        t_r.backward(t_dy)
        t_dseq = t_seq.grad.detach().numpy().transpose(1, 0, 2)
        t_dencoder_out = t_encoder_out.grad.detach().numpy().transpose(1, 0, 2)
        t_dx = (t_dseq, t_dencoder_out)
        t_dsout_b = t_decoder.self_attn.out_proj.bias.grad.detach().numpy()
        t_dswo = t_decoder.self_attn.out_proj.weight.grad.detach().numpy()
        t_dsin_b = t_decoder.self_attn.in_proj_bias.grad.detach().reshape(3, args.num_heads, proj_size).numpy()
        t_dsinput_weights = torch.chunk(
            t_decoder.self_attn.in_proj_weight.grad, 3, dim=0)
        t_dswq = t_dsinput_weights[0].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dswk = t_dsinput_weights[1].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dswv = t_dsinput_weights[2].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dedout_b = t_decoder.multihead_attn.out_proj.bias.grad.detach().numpy()
        t_dedwo = t_decoder.multihead_attn.out_proj.weight.grad.detach().numpy()
        t_dedin_b = t_decoder.multihead_attn.in_proj_bias.grad.detach().reshape(3, args.num_heads, proj_size).numpy()
        t_dedinput_weights = torch.chunk(
            t_decoder.multihead_attn.in_proj_weight.grad, 3, dim=0)
        t_dedwq = t_dedinput_weights[0].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dedwk = t_dedinput_weights[1].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dedwv = t_dedinput_weights[2].detach().reshape((args.num_heads, proj_size, args.embed_size)).numpy()
        t_dnorm1_scale = t_decoder.norm1.weight.grad.detach().numpy()
        t_dnorm1_bias = t_decoder.norm1.bias.grad.detach().numpy()
        t_dnorm2_scale = t_decoder.norm2.weight.grad.detach().numpy()
        t_dnorm2_bias = t_decoder.norm2.bias.grad.detach().numpy()
        t_dnorm3_scale = t_decoder.norm3.weight.grad.detach().numpy()
        t_dnorm3_bias = t_decoder.norm3.bias.grad.detach().numpy()
        t_dlinear1_w = t_decoder.linear1.weight.grad.detach().numpy()
        t_dlinear1_b = t_decoder.linear1.bias.grad.detach().numpy()
        t_dlinear2_w = t_decoder.linear2.weight.grad.detach().numpy()
        t_dlinear2_b = t_decoder.linear2.bias.grad.detach().numpy()
        t_dw = (t_dswq, t_dswk, t_dswv, t_dswo, t_dsin_b, t_dsout_b,
                t_dedwq, t_dedwk, t_dedwv, t_dedwo, t_dedin_b, t_dedout_b,
                t_dnorm1_scale, t_dnorm1_bias,
                t_dnorm2_scale, t_dnorm2_bias,
                t_dnorm3_scale, t_dnorm3_bias,
                t_dlinear1_w, t_dlinear1_b, t_dlinear2_w, t_dlinear2_b)
        t_r = t_r.detach().transpose(0, 1).numpy()
    if np.allclose(np_r, t_r, atol=1e-3):
        print('Torch and NumPy forward match!')
    else:
        print('Forward: Torch != NumPy :(')
        print(f'NumPy:\n{np_r}')
        print(f'Torch:\n{t_r}')
    if not isinstance(np_dx, tuple): np_dx = (np_dx,)
    if not isinstance(t_dx, tuple): t_dx = (t_dx,)
    for i, (np_item, t_item) in enumerate(zip(np_dx, t_dx)):
        if np_item is None or t_item is None:
            print(f'Skipping backward data {i}')
            continue
        if np.allclose(np_item, t_item, atol=1e-3):
            print(f'Torch and NumPy backward data {i} match!')
        else:
            print(f'Backward data {i}: Torch != NumPy >:(')
            print(f'NumPy:\n{np_item}')
            print(f'Torch:\n{t_item}')
    if np_dw is not None and t_dw is not None:
        if not isinstance(np_dw, tuple): np_dw = (np_dw,)
        if not isinstance(t_dw, tuple): t_dw = (t_dw,)
        for i, (np_item, t_item) in enumerate(zip(np_dw, t_dw)):
            if np_item is None or t_item is None:
                print(f'Skipping backward weights {i}')
                continue
            if np.allclose(np_item, t_item, atol=1e-3):
                print(f'Torch and NumPy backward weights {i} match!')
            else:
                print(f'Backward weights {i}: Torch != NumPy ;_;')
                print(f'NumPy:\n{np_item}')
                print(f'Torch:\n{t_item}')
