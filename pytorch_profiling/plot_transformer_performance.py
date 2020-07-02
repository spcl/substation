"""Sweep and profile transformer layers."""

import sys
import os
import os.path
import argparse
import statistics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from transformer_layer import *
import profiling

parser = argparse.ArgumentParser(
    description='Profile different transformer configurations')
parser.add_argument(
    '--batch-sizes', default='1,2,4,8,16', type=str,
    help='Mini-batch sizes')
parser.add_argument(
    '--seq-lens', default='128,256,512,1024,2048', type=str,
    help='Sequence sizes')
parser.add_argument(
    '--embed-sizes', default='128,256,512,1024,2048', type=str,
    help='Embedding sizes')
parser.add_argument(
    '--num-heads', default='4,8,16,32,64', type=str,
    help='Number of heads')
parser.add_argument(
    '--layer-type', default='encoder', type=str,
    choices=['encoder', 'decoder', 'encdec'],
    help='Type of attention to use (default: encoder)')
parser.add_argument(
    '--fp', default='fp32', type=str,
    choices=['fp16', 'fp32'],
    help='Precision to use for training (default: fp32)')
parser.add_argument(
    '--base-batch-size', default=4, type=int,
    help='Fixed batch size for plotting')
parser.add_argument(
    '--base-seq-len', default=512, type=int,
    help='Fixed sequence length for plotting')
parser.add_argument(
    '--base-embed-size', default=1024, type=int,
    help='Fixed embedding size for plotting')
parser.add_argument(
    '--base-heads', default=16, type=int,
    help='Fixed number of heads for plotting')
parser.add_argument(
    'out_dir', type=str,
    help='Output directory')

def str2intlist(s):
    return [int(x) for x in s.split(',')]


args = parser.parse_args()

if os.path.exists(args.out_dir):
    raise ValueError(f'Output directory {args.out_dir} exists')
    sys.exit(1)
    
batch_sizes = str2intlist(args.batch_sizes)
seq_lens = str2intlist(args.seq_lens)
embed_sizes = str2intlist(args.embed_sizes)
heads = str2intlist(args.num_heads)

times = {
    'batch': [],
    'seq_len': [],
    'embed': [],
    'heads': [],
    'fwd_time': [],
    'fwd_stdev': [],
    'bwd_time': [],
    'bwd_stdev': []
}

# Print table header.
print('Batch'.rjust(12) + '\t'
      + 'Seq Len'.rjust(12) + '\t'
      + 'Embed'.rjust(12) + '\t'
      + 'Heads'.rjust(12) + '\t'
      + 'Fwd'.rjust(12) + '\t'
      + 'Stdev'.rjust(12) + '\t'
      + 'Bwd'.rjust(12) + '\t'
      + 'Stdev'.rjust(12) + '\t')

# Profile all configurations.
for batch_size in batch_sizes:
    for seq_len in seq_lens:
        for embed_size in embed_sizes:
            x = profiling.generate_batch(batch_size, seq_len, embed_size)
            if args.layer_type == 'encdec':
                encoder_out = profiling.generate_batch(batch_size, seq_len, embed_size)
            for num_heads in heads:
                times['batch'].append(batch_size)
                times['seq_len'].append(seq_len)
                times['embed'].append(embed_size)
                times['heads'].append(num_heads)
                try:
                    if args.layer_type == 'encoder':
                        t_times = time_encoder(x, num_heads, fp=args.fp)
                    elif args.layer_type == 'decoder':
                        t_times = time_decoder(x, num_heads, fp=args.fp)
                    elif args.layer_type == 'encdec':
                        t_times = time_encdec(x, encoder_out, num_heads, fp=args.fp)
                    fwd_t = statistics.mean(t_times[0])
                    fwd_stdev = statistics.stdev(t_times[0])
                    bwd_t = statistics.mean(t_times[1])
                    bwd_stdev = statistics.stdev(t_times[1])
                except RuntimeError as e:
                    fwd_t = float('nan')
                    fwd_stdev = 0.0
                    bwd_t = float('nan')
                    bwd_stdev = 0.0
                times['fwd_time'].append(fwd_t)
                times['fwd_stdev'].append(fwd_stdev)
                times['bwd_time'].append(bwd_t)
                times['bwd_stdev'].append(bwd_stdev)
                print(f'{batch_size:>12}\t{seq_len:>12}\t{embed_size:>12}\t{num_heads:>12}\t{fwd_t:12.4f}\t{fwd_stdev:12.4f}\t{bwd_t:12.4f}\t{bwd_stdev:12.4f}')

# Build and save data frame.
df = pd.DataFrame.from_dict(times)
os.mkdir(args.out_dir)
df.to_csv(os.path.join(args.out_dir, 'times.csv'))

def do_plot(x, df, x_name, title, filename):
    fig, (ax_fwd, ax_bwd) = plt.subplots(1, 2)
    ax_fwd.errorbar(x, df.fwd_time, yerr=df.fwd_stdev, fmt='o-')
    ax_fwd.set_xscale('log', basex=2)
    ax_fwd.set_xlabel(x_name)
    ax_fwd.set_ylabel('Time (ms)')
    ax_bwd.errorbar(x, df.bwd_time, yerr=df.bwd_stdev, fmt='o-')
    ax_bwd.set_xscale('log', basex=2)
    ax_bwd.set_xlabel(x_name)
    ax_bwd.set_ylabel('Time (ms)')
    fig.suptitle(title, fontsize='small')
    fig.tight_layout()
    fig.savefig(filename)
    

# Plot batch.
batch_df = df[(df.seq_len == args.base_seq_len) &
              (df.embed == args.base_embed_size) &
              (df.heads == args.base_heads)]
do_plot(batch_df.batch, batch_df, 'Mini-batch size',
        f'Batch time, seq len={args.base_seq_len} embed size={args.base_embed_size} heads={args.base_heads} layer={args.layer_type} fp={args.fp}',
        os.path.join(args.out_dir, 'batch.pdf'))

# Plot sequence length.
seq_df = df[(df.batch == args.base_batch_size) &
            (df.embed == args.base_embed_size) &
            (df.heads == args.base_heads)]
do_plot(seq_df.seq_len, seq_df, 'Sequence length',
        f'Sequence length time, batch={args.base_batch_size} embed size={args.base_embed_size} heads={args.base_heads} attn={args.layer_type} fp={args.fp}',
        os.path.join(args.out_dir, 'seq.pdf'))

# Plot embedding size.
embed_df = df[(df.batch == args.base_batch_size) &
              (df.seq_len == args.base_seq_len) &
              (df.heads == args.base_heads)]
do_plot(embed_df.embed, embed_df, 'Embedding size',
        f'Embedding size time, batch={args.base_batch_size} seq len={args.base_seq_len} heads={args.base_heads} attn={args.layer_type} fp={args.fp}',
        os.path.join(args.out_dir, 'embed.pdf'))

# Plot heads.
heads_df = df[(df.batch == args.base_batch_size) &
              (df.seq_len == args.base_seq_len) &
              (df.embed == args.base_embed_size)]
do_plot(heads_df.heads, heads_df, 'Heads',
        f'Heads time, batch={args.base_batch_size} seq len={args.base_seq_len} embed size={args.base_embed_size} attn={args.layer_type} fp={args.fp}',
        os.path.join(args.out_dir, 'heads.pdf'))
