"""Profile encoder/decoder transformer layers."""

import sys
import os.path

import argparse
import torch
import torch.nn

import profiling

parser = argparse.ArgumentParser(
    description='Profile multi-head attention')
parser.add_argument(
    '--batch-size', default=8, type=int,
    help='Mini-batch size (default: 8)')
parser.add_argument(
    '--max-seq-len', default=512, type=int,
    help='Maximum sequence length (default: 512)')
parser.add_argument(
    '--max-enc-seq-len', default=512, type=int,
    help='Maximum encoder sequence length (default: 512)')
parser.add_argument(
    '--embed-size', default=1024, type=int,
    help='Embedding size (default: 1024)')
parser.add_argument(
    '--num-heads', default=16, type=int,
    help='Number of attention heads (default: 16)')
parser.add_argument(
    '--activation', default='relu', type=str,
    choices=['relu', 'gelu'],
    help='Activation type to use (default: relu)')
parser.add_argument(
    '--no-attn-bias', default=False, action='store_true',
    help='Do not use bias on attention (default: use)')
parser.add_argument(
    '--no-attn-dropout', default=False, action='store_true',
    help='Do not use dropout in attention (default: use)')
parser.add_argument(
    '--layer', required=True, type=str,
    choices=['encoder', 'decoder', 'encdec'],
    help='Type of layer to profile')
parser.add_argument(
    '--no-backprop', default=False, action='store_true',
    help='Do not run backprop')
parser.add_argument(
    '--fp', default='fp32', type=str,
    choices=['fp16', 'fp32'],
    help='Precision to use for training (default: fp32)')
parser.add_argument(
    '--apex', default=False, action='store_true',
    help='Enable Nvidia Apex AMP')
parser.add_argument(
    '--num-iters', default=100, type=int,
    help='Number of benchmark iterations')
parser.add_argument(
    '--num-warmups', default=5, type=int,
    help='Number of warmup iterations')
parser.add_argument(
    '--plot-file', default=None, type=str,
    help='Save violin plots to file')


def time_encoder(
        x, num_heads, activation='relu', bias=True, dropout=True, do_backprop=True, fp='fp32',
        use_apex=False, num_iters=100, num_warmups=5):
    """Benchmark a transformer encoder layer.

    x is the input sequence in (sequence, batch, embedding) order.

    num_heads is the number of multi-head attention heads.

    activation is the activation function to use.

    bias is whether to use bias in attention.

    do_backprop is whether to benchmark backprop.

    fp is the precision to perform operations in.

    use_apex is whether to import and use Nvidia's Apex library.

    num_iters and num_warmups are the number of warmup and benchmarking
    iterations, respectively.

    Returns the runtimes for each iteration of each function.

    """
    if use_apex:
        from apex import amp
    embed_size = x.size(2)
    encoder = torch.nn.TransformerEncoderLayer(
        embed_size, num_heads, dim_feedforward=4*embed_size,
        activation=activation)
    if not bias or not dropout:
        new_bias = bias
        new_dropout = 0.1 if dropout else 0.0
        encoder.self_attn = torch.nn.MultiheadAttention(
            embed_size, num_heads, dropout=new_dropout, bias=new_bias)
    encoder = encoder.to(profiling.cuda_device)
    encoder.train()
    #x = x.requires_grad_().to(profiling.cuda_device)
    x = x.to(profiling.cuda_device).requires_grad_()
    dy = profiling.generate_batch(
        x.size(1), x.size(0), embed_size).to(profiling.cuda_device)
    if fp == 'fp16':
        if use_apex:
            encoder = amp.initialize(encoder)
        else:
            encoder = encoder.half()
            x = x.half()
            dy = dy.half()
    result, backward_result = None, None
    def forward():
        nonlocal result
        result = encoder.forward(x)
    def backward():
        nonlocal backward_result
        backward_result = result.backward(dy)
    def clear():
        encoder.zero_grad()
    return profiling.time_funcs(
        [forward, backward, clear], name='Encoder',
        func_names=['forward', 'backward', 'clear'],
        num_iters=num_iters, warmups=num_warmups)


def time_decoder(
        x, num_heads, activation='relu', bias=True, dropout=True, do_backprop=True, fp='fp32',
        use_apex=False, num_iters=100, num_warmups=5):
    """Benchmark a transformer decoder layer.

    x is the input sequence in (sequence, batch, embedding) order.

    num_heads is the number of multi-head attention heads.

    activation is the activation function to use.

    do_backprop is whether to benchmark backprop.

    fp is the precision to perform operations in.

    use_apex is whether to import and use Nvidia's Apex library.

    num_iters and num_warmups are the number of warmup and benchmarking
    iterations, respectively.

    Returns the runtimes for each iteration of each function.

    """
    if use_apex:
        from apex import amp
    embed_size = x.size(2)
    decoder = torch.nn.TransformerEncoderLayer(
        embed_size, num_heads, dim_feedforward=4*embed_size,
        activation=activation)
    if not bias or not dropout:
        new_bias = bias
        new_dropout = 0.1 if dropout else 0.0
        decoder.self_attn = torch.nn.MultiheadAttention(
            embed_size, num_heads, dropout=new_dropout, bias=new_bias)
    decoder = decoder.to(profiling.cuda_device)
    decoder.train()
    x = x.to(profiling.cuda_device).requires_grad_()
    dy = profiling.generate_batch(
        x.size(1), x.size(0), embed_size).to(profiling.cuda_device)
    mask = profiling.gen_attention_mask(
        x.size(0), x.size(0)).to(profiling.cuda_device)
    if fp == 'fp16':
        if use_apex:
            decoder = amp.initialize(decoder)
        else:
            decoder = decoder.half()
            x = x.half()
            dy = dy.half()
            mask = mask.half()
    result, backward_result = None, None
    def forward():
        nonlocal result
        result = decoder.forward(x, src_mask=mask)
    def backward():
        nonlocal backward_result
        backward_result = result.backward(dy)
    def clear():
        decoder.zero_grad()
    return profiling.time_funcs(
        [forward, backward, clear], name='Decoder',
        func_names=['forward', 'backward', 'clear'],
        num_iters=num_iters, warmups=num_warmups)


def time_encdec(
        x, encoder_out, num_heads, activation='relu', bias=True, dropout=True, do_backprop=True,
        use_apex=False, fp='fp32', num_iters=100, num_warmups=5):
    """Benchmark a transformer decoder layer with encoder/decoder attention.

    x is the input sequence in (sequence, batch, embedding) order.

    encoder_out is the output from an encoder in (sequence, batch, embedding)
    order.

    num_heads is the number of multi-head attention heads.

    activation is the activation function to use.

    do_backprop is whether to benchmark backprop.

    fp is the precision to perform operations in.

    use_apex is whether to import and use Nvidia's Apex library.

    num_iters and num_warmups are the number of warmup and benchmarking
    iterations, respectively.

    Returns the runtimes for each iteration of each function.

    """
    if use_apex:
        from apex import amp
    if not bias or not dropout:
        raise ValueError('Not supported')
    embed_size = x.size(2)
    decoder = torch.nn.TransformerDecoderLayer(
        embed_size, num_heads, dim_feedforward=4*embed_size,
        activation=activation).to(profiling.cuda_device)
    decoder.train()
    x = x.to(profiling.cuda_device).requires_grad_()
    encoder_out = encoder_out.to(profiling.cuda_device)
    dy = profiling.generate_batch(
        x.size(1), x.size(0), embed_size).to(profiling.cuda_device)
    mask = profiling.gen_attention_mask(
        x.size(0), x.size(0)).to(profiling.cuda_device)
    if fp == 'fp16':
        if use_apex:
            decoder = amp.initialize(decoder)
        else:
            decoder = decoder.half()
            x = x.half()
            encoder_out = encoder_out.half()
            dy = dy.half()
            mask = mask.half()
    # Must compute a gradient for this.
    encoder_out = encoder_out.requires_grad_()
    result, backward_result = None, None
    def forward():
        nonlocal result
        result = decoder.forward(x, encoder_out, tgt_mask=mask)
    def backward():
        nonlocal backward_result
        backward_result = result.backward(dy)
    def clear():
        decoder.zero_grad()
    return profiling.time_funcs(
        [forward, backward, clear], name='Encdec',
        func_names=['forward', 'backward', 'clear'],
        num_iters=num_iters, warmups=num_warmups)


if __name__ == '__main__':
    args = parser.parse_args()
    # Check this here first.
    if args.plot_file and os.path.exists(args.plot_file):
        print(f'{args.plot_file} exists, aborting.')
        sys.exit(1)
    x = profiling.generate_batch(
        args.batch_size, args.max_seq_len, args.embed_size)
    if args.layer == 'encoder':
        times = time_encoder(
            x, args.num_heads, activation=args.activation,
            bias=not args.no_attn_bias, dropout=not args.no_attn_dropout,
            do_backprop=not args.no_backprop, fp=args.fp, use_apex=args.apex,
            num_iters=args.num_iters, num_warmups=args.num_warmups)
    elif args.layer == 'decoder':
        times = time_decoder(
            x, args.num_heads, activation=args.activation,
            bias=not args.no_attn_bias, dropout=not args.no_attn_dropout,
            do_backprop=not args.no_backprop, fp=args.fp, use_apex=args.apex,
            num_iters=args.num_iters, num_warmups=args.num_warmups)
    elif args.layer == 'encdec':
        encoder_out = profiling.generate_batch(
            args.batch_size, args.max_enc_seq_len, args.embed_size)
        times = time_encdec(
            x, encoder_out, args.num_heads, activation=args.activation,
            bias=not args.no_attn_bias, dropout = not args.no_attn_dropout,
            do_backprop=not args.no_backprop, fp=args.fp, use_apex=args.apex,
            num_iters=args.num_iters, num_warmups=args.num_warmups)
    print(f'layer={args.layer} activation={args.activation} batch={args.batch_size} seqlen={args.max_seq_len} enc-seqlen={args.max_enc_seq_len} embed={args.embed_size} heads={args.num_heads} fp={args.fp}')
    profiling.print_time_statistics(times, ['forward', 'backward', 'clear'])
    if args.plot_file:
        profiling.plot_violins(times, ['forward', 'backward', 'clear'],
                               args.plot_file)
