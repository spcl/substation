"""Interface for benchmarking and testing Substation encoders."""

import argparse
import pickle
import statistics

import torch
import numpy as np

from encoder import Encoder as SubstationEncoder

parser = argparse.ArgumentParser(
    description='Test and benchmark Substation encoders')
parser.add_argument(
    '--batch-size', type=int, required=True,
    help='Mini-batch size')
parser.add_argument(
    '--max-seq-len', type=int, required=True,
    help='Maximum sequence length')
parser.add_argument(
    '--num-heads', type=int, required=True,
    help='Number of attention heads')
parser.add_argument(
    '--emb-size', type=int, required=True,
    help='Embedding size')
parser.add_argument(
    '--layout', type=str, required=True,
    help='Optimized layout file')
parser.add_argument(
    '--softmax-dropout-prob', type=float, default=0.1,
    help='Softmax dropout probability')
parser.add_argument(
    '--residual1-dropout-prob', type=float, default=0.1,
    help='First residual dropout probability')
parser.add_argument(
    '--activation-dropout-prob', type=float, default=0.1,
    help='Activation dropout probability')
parser.add_argument(
    '--residual2-dropout-prob', type=float, default=0.1,
    help='Second residual dropout probability')
parser.add_argument(
    '--dtype', type=str, default='fp16',
    help='Datatype to use')
parser.add_argument(
    '--iters', type=int, default=1000,
    help='Number of iterations to benchmark')
parser.add_argument(
    '--warmup', type=int, default=100,
    help='Number of warmup iterations (in addition to --iters)')
parser.add_argument(
    '--profile', default=False, action='store_true',
    help='Enable NVTX annotations for profiling')
parser.add_argument(
    '--test', default=False, action='store_true',
    help='Validate encoder against a reference implementation')
parser.add_argument(
    '--test-dtype', type=str, default='fp64',
    help='Datatype to use for testing')
parser.add_argument(
    '--compilation-dir', type=str, default='./compilation_dir/',
    help='Path to directory used for compilation and caching')

_str_to_dtype = {
    'fp16': torch.float16,
    'fp32': torch.float32,
    'fp64': torch.float64
}


class CudaTimer:
    def __init__(self, n):
        """Support timing n regions."""
        self.events = []
        for _ in range(n + 1):
            self.events.append(torch.cuda.Event(enable_timing=True))
        self.cur = 0

    def start(self):
        """Start timing the first region."""
        self.events[0].record()
        self.cur = 1

    def next(self):
        """Start timing for the next region."""
        if self.cur >= len(self.events):
            raise RuntimeError('Trying to time too many regions')
        self.events[self.cur].record()
        self.cur += 1

    def end(self):
        """Finish timing."""
        if self.cur != len(self.events):
            raise RuntimeError('Called end too early')
        self.events[-1].synchronize()

    def get_times(self):
        """Return measured time for each region."""
        times = []
        for i in range(1, len(self.events)):
            times.append(self.events[i-1].elapsed_time(self.events[i]))
        return times



def create_encoder(compilation_dir, batch_size, max_seq_len, num_heads, embedding_size,
                   layouts_file, datatype,
                   softmax_dropout_probability, residual1_dropout_probability, activation_dropout_probability, residual2_dropout_probability,
                   enable_debug=False, enable_profiling=False):
    """Return an encoder instance from arguments."""
    with open(layouts_file, 'rb') as f:
        layouts = pickle.load(f)
    encoder = SubstationEncoder(
        compilation_dir,
        batch_size, max_seq_len, num_heads, embedding_size,
        layouts, _str_to_dtype[datatype],
        softmax_dropout_probability, residual1_dropout_probability, activation_dropout_probability, residual2_dropout_probability,
        enable_debug=enable_debug,
        profiling=enable_profiling).cuda()
    encoder.train()
    return encoder


def isclose(a, b, atol=1e-4, rtol=1e-4):
    """Return True if a is close to b."""
    if a.shape != b.shape:
        print(f'Shape mismatch: {a.shape} != {b.shape}')
        return False
    closeness = np.absolute(a - b) < (atol + rtol*np.absolute(b))
    if closeness.all():
        return True
    else:
        a_bad = np.extract(1 - closeness, a)
        b_bad = np.extract(1 - closeness, b)
        bad_ratio = a_bad.size * 1.0/closeness.size
        print(f'Bad ratio: {bad_ratio}')
        print(a_bad)
        print(b_bad)
        return False


def test_encoder(encoder):
    from substation.transformer import encoder as ref_encoder, encoder_backward as ref_encoder_backward

    sizes_map = {'B': encoder.batch_size,
                 'J': encoder.max_seq_len,
                 'H': encoder.num_heads,
                 'P': encoder.proj_size,
                 'I': encoder.embedding_size}
    X = torch.randn(
        tuple(sizes_map[x] for x in encoder.layouts['input'][1]),
        dtype=encoder.torch_dtype, device='cuda', requires_grad=True)
    X.retain_grad()

    # Set up appropriate NumPy layout.
    numpy_layout = dict(
        WKQV='QHPI',
        BKQV='QHP',
        WO='IHP',
        BO='I',
        S1='I',
        B1='I',
        LINB1='U',
        LINW1='UI',
        S2='I',
        B2='I',
        LINB2='I',
        LINW2='IU',
    )

    numpy_params = {}
    for (name, layout), pt_array in zip(encoder.layouts['params'], encoder.params):
        numpy_params[name] = np.einsum(layout + '->' + numpy_layout[name],
                                       pt_array.detach().cpu().numpy())

    numpy_params['X'] = np.einsum(f'{encoder.layouts["input"][1]}->BJI',
                                  X.clone().detach().cpu().numpy())
    numpy_params['WK'] = numpy_params['WKQV'][0]
    numpy_params['WQ'] = numpy_params['WKQV'][1]
    numpy_params['WV'] = numpy_params['WKQV'][2]
    numpy_params['BQKV'] = np.stack((numpy_params['BKQV'][1],
                                     numpy_params['BKQV'][0],
                                     numpy_params['BKQV'][2]))
    numpy_params['WO'] = numpy_params['WO'].reshape((encoder.embedding_size,
                                                     encoder.embedding_size))

    # Run forward pass.
    Y = encoder.forward(X)
    (y,
     i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
     i_attn_scaled_scores, i_attn_dropout_mask,
     i_norm1_mean, i_norm1_std, i_norm1_normed,
     i_linear1_dropout_mask, i_ff_dropout_mask,
     i_norm2_mean, i_norm2_std, i_norm2_normed,
     i_ff_resid, i_ff1, iff1_linear, i_normed1,
     i_attn_resid) = ref_encoder(
         numpy_params['X'],
         numpy_params['WQ'], numpy_params['WK'],
         numpy_params['WV'], numpy_params['WO'],
         numpy_params['BQKV'], numpy_params['BO'], 1/np.sqrt(encoder.proj_size),
         numpy_params['S1'], numpy_params['B1'],
         numpy_params['S2'], numpy_params['B2'],
         numpy_params['LINW1'], numpy_params['LINB1'],
         numpy_params['LINW2'], numpy_params['LINB2'],
         0, 0, 0, activation='relu')

    # Check forward pass.
    KK = encoder.debug['KKQQVV'][0]
    QQ = encoder.debug['KKQQVV'][1]
    VV = encoder.debug['KKQQVV'][2]

    WK = encoder.debug['WKQV'][0]
    WQ = encoder.debug['WKQV'][1]
    WV = encoder.debug['WKQV'][2]

    refWQ = numpy_params['WQ']
    print('WQ: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["WKQV"][1:]}->HPI', WQ), refWQ))

    refWK = numpy_params['WK']
    print('WK: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["WKQV"][1:]}->HPI', WK), refWK))

    refWV = numpy_params['WV']
    print('WV: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["WKQV"][1:]}->HPI', WV), refWV))
    
    layoutQQ = dict(encoder.layouts['forward_interm'])['KKQQVV'][1:]
    refQQ = np.einsum(f'BHJP->{layoutQQ}', i_attn_proj_q)
    print('QQ: ', isclose(QQ, refQQ))

    layoutKK = layoutQQ.translate({ord('J'): 'K'})
    refKK = np.einsum(f'BHKP->{layoutKK}', i_attn_proj_k)
    print('KK: ', isclose(KK, refKK))

    refVV = np.einsum(f'BHKP->{layoutKK}', i_attn_proj_v)
    print('VV: ', isclose(VV, refVV))

    refALPHA = np.einsum(
        f'BHJK->{dict(encoder.layouts["forward_interm"])["ALPHA"]}',
        i_attn_scaled_scores)
    ALPHA = encoder.debug['ALPHA']
    print('ALPHA: ', isclose(ALPHA, refALPHA))

    myY = np.einsum(f'{encoder.layouts["output"][1]}->BJI',
                    Y.detach().cpu().numpy())
    print('Y: ', isclose(myY, y))

    # Run backward pass.
    DY = torch.randn(
        tuple(sizes_map[x] for x in encoder.layouts['output'][1]),
              dtype=encoder.torch_dtype, device='cuda')
    dy = np.einsum(f'{encoder.layouts["output"][1]}->BJI',
                   DY.detach().cpu().numpy())

    Y.backward(DY)
    DX = X.grad.detach().cpu().numpy()

    (dx,
    dattn_wq, dattn_wk, dattn_wv, dattn_wo, dattn_in_b, dattn_out_b,
    dnorm1_scale, dnorm1_bias, dnorm2_scale, dnorm2_bias,
    dlinear1_w, dlinear1_b, dlinear2_w, dlinear2_b) = ref_encoder_backward(
        numpy_params['X'], dy,
        i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
        i_attn_scaled_scores, i_attn_dropout_mask,
        i_norm1_mean, i_norm1_std, i_norm1_normed,
        i_linear1_dropout_mask, i_ff_dropout_mask,
        i_norm2_mean, i_norm2_std, i_norm2_normed,
        i_ff_resid, i_ff1, iff1_linear, i_normed1,
        i_attn_resid,
        numpy_params['WQ'], numpy_params['WK'], numpy_params['WV'],
        numpy_params['WO'], 1/np.sqrt(encoder.proj_size),
        numpy_params['S1'], numpy_params['B1'],
        numpy_params['S2'], numpy_params['B2'],
        numpy_params['LINW1'], numpy_params['LINB1'],
        numpy_params['LINW2'], numpy_params['LINB2'],
        0, 0, 0, activation='relu')

    # Check backward pass.
    print('DX: ', isclose(np.einsum(f'{encoder.layouts["input"][1]}->BJI', DX), dx))

    DWK = encoder.debug['DWKQV'][0]
    DWQ = encoder.debug['DWKQV'][1]
    DWV = encoder.debug['DWKQV'][2]
    DWK = DWK.reshape(encoder.proj_size, encoder.num_heads,
                      encoder.embedding_size)
    print('DWK: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["WKQV"][1:]}->HPI', DWK), dattn_wk))
    print('DWQ: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["WKQV"][1:]}->HPI', DWQ), dattn_wq))
    print('DWV: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["WKQV"][1:]}->HPI', DWV), dattn_wv))

    DBKQV = encoder.debug['DBKQV']
    DBQKV = np.stack((DBKQV[1], DBKQV[0], DBKQV[2]))
    print('DBQKV: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["BKQV"]}->QHP', DBQKV), dattn_in_b))

    DWO = encoder.debug['DWO']
    refDWO = np.einsum(
        f'IHP->{dict(encoder.layouts["params"])["WO"]}',
        dattn_wo.reshape(encoder.embedding_size, encoder.num_heads,
                         encoder.proj_size))
    print('DWO: ', isclose(DWO, refDWO))

    print('DBO: ', isclose(encoder.debug['DBO'], dattn_out_b))

    print('DS1: ', isclose(encoder.debug['DS1'], dnorm1_scale))

    print('DS2: ', isclose(encoder.debug['DS2'], dnorm2_scale))

    print('DB1: ', isclose(encoder.debug['DB1'], dnorm1_bias))

    print('DB2: ', isclose(encoder.debug['DB2'], dnorm2_bias))

    print('DLINB1: ', isclose(encoder.debug['DLINB1'], dlinear1_b))

    print('DLINW1: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["LINW1"]}->UI',
        encoder.debug['DLINW1']), dlinear1_w))

    print('DLINB2: ', isclose(encoder.debug['DLINB2'], dlinear2_b))

    print('DLINW2: ', isclose(np.einsum(
        f'{dict(encoder.layouts["params"])["LINW2"]}->IU',
        encoder.debug['DLINW2']), dlinear2_w))


if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        encoder = create_encoder(
            args.compilation_dir,
            args.batch_size, args.max_seq_len, args.num_heads, args.emb_size,
            args.layout, args.test_dtype, 
            0.0, 0.0, 0.0, 0.0, enable_debug=True)
        test_encoder(encoder)

    encoder = create_encoder(
        args.compilation_dir,
        args.batch_size, args.max_seq_len, args.num_heads, args.emb_size,
        args.layout, args.dtype, 
        args.softmax_dropout_prob, args.residual1_dropout_prob, args.activation_dropout_prob, args.residual2_dropout_prob,
        enable_profiling=args.profile)
    total_iters = args.iters + args.warmup
    forward_times = [0.0] * args.iters
    backward_times = [0.0] * args.iters
    timer = CudaTimer(2)

    input_layout = encoder.layouts['input'][1]
    sizes_map = {'B': encoder.batch_size,
                 'J': encoder.max_seq_len,
                 'I': encoder.embedding_size}
    input_shape = tuple(sizes_map[x] for x in input_layout)

    X = torch.randn(input_shape,
                    dtype=encoder.torch_dtype, device='cuda')
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(total_iters):
        timer.start()
        Y = encoder(X)
        timer.next()
        (Y.sum()).backward()
        timer.next()
        timer.end()
        times = timer.get_times()
        if i >= args.warmup:
            forward_times[i-args.warmup] = times[0]
            backward_times[i-args.warmup] = times[1]
    torch.cuda.cudart().cudaProfilerStop()

    print('Forward: median: {:.6f} ms (min: {:.6f}, max: {:.6f}, stddev: {:.6f})'.format(
        statistics.median(forward_times),
        min(forward_times),
        max(forward_times),
        statistics.stdev(forward_times)))
    print('Backward: median: {:.6f} ms (min: {:.6f}, max: {:.6f}, stddev: {:.6f})'.format(
        statistics.median(backward_times),
        min(backward_times),
        max(backward_times),
        statistics.stdev(backward_times)))
    print('Total (median): {:.6f} ms'.format(
        statistics.median(forward_times) + statistics.median(backward_times)))
