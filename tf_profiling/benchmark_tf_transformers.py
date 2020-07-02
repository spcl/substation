import time
import argparse
import math

import tensorflow as tf

from transformers import (
    BertConfig,
    GPT2Config,
    TFBertLayer,
    TFBlock,
)

parser = argparse.ArgumentParser(
    description='Benchmark TensorFlow transformer layers')
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
    '--fp', default='fp32', type=str,
    choices=['fp16', 'fp32'],
    help='Precision to use for training (default: fp32)')
parser.add_argument(
    '--no-xla', default=False, action='store_true',
    help='Disable XLA')
parser.add_argument(
    '--num-iters', default=100, type=int,
    help='Number of benchmark iterations')


def setup_tf(use_xla, use_amp):
    tf.config.optimizer.set_jit(use_xla)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": use_amp})


def generate_xy(batch_size, max_seq_len, embed_size):
    x = tf.random.normal((batch_size, max_seq_len, embed_size))
    y = tf.random.normal((batch_size, max_seq_len, embed_size))
    return x, y


def run_encoder(batch_size, max_seq_len, embed_size, num_heads,
                act='relu', num_iters=100):
    config = BertConfig(hidden_size=embed_size,
                        num_hidden_layers=24,
                        num_attention_heads=num_heads,
                        intermediate_size=4*embed_size,
                        hidden_act=act)
    class EncoderModel(tf.keras.models.Model):
        def __init__(self):
            super().__init__()
            self.encoder = TFBertLayer(config)
        def call(self, inputs):
            return self.encoder([inputs, None, None])
    
    encoder_model = EncoderModel()
    encoder_model.compile(
        optimizer=tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), "dynamic"),
        loss=tf.keras.losses.MeanSquaredError())
    x, y = generate_xy(batch_size, max_seq_len, embed_size)
    encoder_model.fit(x, y, batch_size=batch_size, epochs=num_iters)


def run_decoder(batch_size, max_seq_len, embed_size, num_heads,
                act='relu', num_iters=100):
    config = GPT2Config(
        n_positions=max_seq_len,
        n_ctx=max_seq_len,
        n_embd=embed_size,
        n_head=num_heads)
    class DecoderModel(tf.keras.models.Model):
        def __init__(self):
            super().__init__()
            scale = 1.0/math.sqrt(embed_size // num_heads)
            self.decoder = TFBlock(max_seq_len, config, scale=scale)
        def call(self, inputs):
            # TFAttention handles the causal attention automatically.
            return self.decoder([inputs, None, None, None], training=True)[0]

    decoder_model = DecoderModel()
    decoder_model.compile(
        optimizer=tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), "dynamic"),
        loss=tf.keras.losses.MeanSquaredError())
    x, y = generate_xy(batch_size, max_seq_len, embed_size)
    decoder_model.fit(x, y, batch_size=batch_size, epochs=num_iters)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.no_attn_bias or args.no_attn_dropout:
        print('WARNING: Running without attention bias/dropout is a hack and you must modeify modeling_tf_bert manually')
    print(f'layer={args.layer} activation={args.activation} batch={args.batch_size} seqlen={args.max_seq_len} enc-seqlen={args.max_enc_seq_len} embed={args.embed_size} heads={args.num_heads} fp={args.fp}')
    setup_tf(not args.no_xla, args.fp == 'fp16')
    if args.layer == 'encoder':
        run_encoder(args.batch_size, args.max_seq_len, args.embed_size,
                    args.num_heads, act=args.activation,
                    num_iters=args.num_iters)
    elif args.layer == 'decoder':
        run_decoder(args.batch_size, args.max_seq_len, args.embed_size,
                    args.num_heads, act=args.activation,
                    num_iters=args.num_iters)
    elif args.layer == 'encdec':
        raise RuntimeError('Not supported')
