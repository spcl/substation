import math
import torch
import subprocess
import ctypes
import torch.cuda
import textwrap
import pickle

from substation.transformer import encoder as ref_encoder, encoder_backward as ref_encoder_backward

with open('layouts.pickle', 'rb') as handle:
    layouts = pickle.load(handle)

libcudart = ctypes.CDLL('libcudart.so')

layout_input = layouts['layout_input']
layout_output = layouts['layout_output']
special_dims = layouts['special_dims']
algorithms = layouts['algorithms']
layouts_params = layouts['layouts_params']
layout_forward_interm = layouts['layout_forward_interm']
layout_backward_interm = layouts['layout_backward_interm']

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

layouts = layouts_params + layout_forward_interm + layout_backward_interm

#special_arrays = ['K', 'Q', 'V', 'WK', 'WQ', 'WV', 'KK', 'QQ', 'VV', 'DKK', 'DQQ', 'DVV']

def compile_module(defines, datatype, dropout_probability, instance_id):
    with open("encoder_parameters.cuh", "w") as f:
        sizes = "\n".join([f'#define size{k} {v}' for k, v in defines.items()])

        f.write("#pragma once\n\n")

        f.write(sizes)
        f.write("\n\n")

        f.write(f'using Real = {datatype};\n\n')

        f.write(f'#define ENCODER_DROPOUT_PROBABILITY {dropout_probability}\n\n')

        f.write(f'#define ARRAY_X_DEF Real* g{layout_input[0]}\n\n')
        f.write(f'#define ARRAY_X g{layout_input[0]}\n\n')

        f.write("#define ARRAY_WEIGHTS_DEF " + ", ".join([f"Real* g{k}" for k, v in layouts_params]))
        f.write('\n')
        f.write("#define ARRAY_WEIGHTS " + ", ".join([f"g{k}" for k, v in layouts_params]))
        f.write("\n\n")

        f.write("#define ARRAY_FWD_INTERM_DEF " + ", ".join([f"Real* g{k}" for k, v in layout_forward_interm]))
        f.write('\n')
        f.write("#define ARRAY_FWD_INTERM " + ", ".join([f"g{k}" for k, v in layout_forward_interm]))
        f.write("\n\n")

        f.write(f'#define ARRAY_Y_DEF Real* g{layout_output[0]}\n')
        f.write(f'#define ARRAY_Y g{layout_output[0]}\n\n')

        f.write(f'#define ARRAY_D_Y_DEF Real* gD{layout_output[0]}\n')
        f.write(f'#define ARRAY_D_Y gD{layout_output[0]}\n\n')

        f.write("#define ARRAY_D_WEIGHTS_DEF " + ", ".join([f"Real* gD{k}" for k, v in layouts_params]))
        f.write('\n')
        f.write("#define ARRAY_D_WEIGHTS " + ", ".join([f"gD{k}" for k, v in layouts_params]))
        f.write("\n\n")

        f.write("#define ARRAY_BWD_INTERM_DEF " + ", ".join([f"Real* g{k}" for k, v in layout_backward_interm]))
        f.write('\n')
        f.write("#define ARRAY_BWD_INTERM " + ", ".join([f"g{k}" for k, v in layout_backward_interm]))
        f.write("\n\n")

        f.write(f'#define ARRAY_D_X_DEF Real* gD{layout_input[0]}\n')
        f.write(f'#define ARRAY_D_X gD{layout_input[0]}\n\n')

        f.write(textwrap.dedent("""\

            #define ENCODER_FORWARD_DEF \
                ARRAY_X_DEF, \
                ARRAY_WEIGHTS_DEF, \
                ARRAY_FWD_INTERM_DEF, \
                ARRAY_Y_DEF

            #define ENCODER_BACKWARD_DEF \
                ARRAY_D_Y_DEF, \
                ARRAY_D_WEIGHTS_DEF, \
                ARRAY_BWD_INTERM_DEF, \
                ARRAY_WEIGHTS_DEF, \
                ARRAY_FWD_INTERM_DEF, \
                ARRAY_X_DEF, \
                ARRAY_D_X_DEF

            #define ENCODER_FORWARD \
                ARRAY_X, \
                ARRAY_WEIGHTS, \
                ARRAY_FWD_INTERM, \
                ARRAY_Y

            #define ENCODER_BACKWARD \
                ARRAY_D_Y, \
                ARRAY_D_WEIGHTS, \
                ARRAY_BWD_INTERM, \
                ARRAY_WEIGHTS, \
                ARRAY_FWD_INTERM, \
                ARRAY_X, \
                ARRAY_D_X

            struct dimB { enum { value = sizeB }; };
            struct dimK { enum { value = sizeS }; };
            struct dimJ { enum { value = sizeS }; };
            struct dimH { enum { value = sizeH }; };
            struct dimP { enum { value = sizeP }; };
            struct dimI { enum { value = sizeI }; };
            struct dimU { enum { value = sizeU }; };
            struct dimQ { enum { value = 3 }; };
        
        """))

        all_layouts = layout_forward_interm + layout_backward_interm + layouts_params
        all_layouts += [layout_input, layout_output]
        all_layouts += [('D' + k, v) for k, v in [*layouts_params, layout_input, layout_output]]


        dlp = dict(all_layouts)

        assert(dlp['KKQQVV'][0] == 'Q')
        KK_layout = dlp['KKQQVV'][1:]

        assert(dlp['DKKQQVV'][0] == 'Q')
        DKK_layout = dlp['DKKQQVV'][1:]

        assert(dlp['WKKWQQWVV'][0] == 'Q')
        WK_K_layout = dlp['WKKWQQWVV'][1:]

        assert(dlp['BKQV'][0] == 'Q')
        BK_layout = dlp['BKQV'][1:]

        assert(dlp['WKQV'][0] == 'Q')
        WK_layout = dlp['WKQV'][1:]

        # additional definitions
        all_layouts += [
            ('WKKself', WK_K_layout),
            ('WQQself', WK_K_layout),
            ('WVVself', WK_K_layout),
            ('BK', BK_layout),
            ('BQ', BK_layout),
            ('BV', BK_layout),
            ('DBK', BK_layout),
            ('DBQ', BK_layout),
            ('DBV', BK_layout),
            ('Q', dlp['X']),
            ('K', dlp['X'].translate({ord('J'): 'K'})),
            ('V', dlp['X'].translate({ord('J'): 'K'})),
            ('KK', KK_layout.translate({ord('J'): 'K'})),
            ('QQ', KK_layout),
            ('VV', KK_layout.translate({ord('J'): 'K'})),
            ('DKK', DKK_layout.translate({ord('J'): 'K'})),
            ('DQQ', DKK_layout),
            ('DVV', DKK_layout.translate({ord('J'): 'K'})),
            ('KKself', KK_layout),
            ('QQself', KK_layout),
            ('VVself', KK_layout),
            ('DKKself', DKK_layout),
            ('DQQself', DKK_layout),
            ('DVVself', DKK_layout),
            ('WKK', WK_K_layout),
            ('WQQ', WK_K_layout),
            ('WVV', WK_K_layout),
            ('WK', WK_layout),
            ('WQ', WK_layout),
            ('WV', WK_layout),
            ('DWK', WK_layout),
            ('DWQ', WK_layout),
            ('DWV', WK_layout),
        ]


        for k, v in all_layouts:
            layout = ", ".join([f'dim{x}' for x in v])
            layout_def = f'using l{k} = metal::list<{layout}>;\n'
            f.write(layout_def)

        f.write('\n\n')
        
        # TODO
        # CUBLAS_GEMM_DEFAULT_TENSOR_OP

        for gemm, algo in algorithms.items():
            f.write(f'#define algo{gemm} CUBLAS_GEMM_{algo}_TENSOR_OP\n')

        f.write('\n\n')

        for name, dim in special_dims.items():
            f.write(f'#define sd{name} dim{dim}\n')

        f.write('\n\n')


    print('Compiling...')
    subprocess.run(f"nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -c --compiler-options -fPIC encoder_newest.cu -o encoder.o".split(' '))
    subprocess.run(f"nvcc -shared -o encoder_{instance_id}.so encoder.o".split(' '))
    print('Compiling done')

    lib = ctypes.CDLL(f'./encoder_{instance_id}.so')
    
    lib.init.argtypes = []
    lib.init.restype = ctypes.c_void_p

    lib.encoder_forward.argtypes = [ctypes.c_void_p] * (1 + 1 + len(layouts_params) + len(layout_forward_interm) + 1)

    lib.encoder_backward.argtypes = [ctypes.c_void_p] * (
        1 + 1 + len(layouts_params) + len(layout_backward_interm) + (
            len(layouts_params) + len(layout_forward_interm) + 1) + 1)

    lib.destroy.argtypes = [ctypes.c_void_p]

    return lib



torch_dtype_map = {
    torch.float16: 'half',
    torch.float32: 'float',
    torch.float64: 'double',
}



class Encoder(torch.nn.Module):

    last_instance_id = 0

    def __init__(self, B, S, H, P, torch_dtype, dropout_probability=0.5, enable_debug=False):
        super(Encoder, self).__init__()
        self.Q = 3
        self.B = B
        self.S = S
        self.H = H
        self.P = P
        self.I = H * P
        self.U = 4 * I
        self.J = S
        self.K = S

        self.torch_dtype = torch_dtype

        sizes = {s: getattr(self, s) for s in 'QBSHPIUJK'}

        for param, layout in layouts_params:
            sz = tuple(sizes[s] for s in layout)
            setattr(self, param, torch.nn.Parameter(torch.empty(*sz, dtype=self.torch_dtype)))


        self.params = tuple(getattr(self, k) for k, _ in layouts_params)

        self.reset_parameters()

        encoder_cpp = compile_module(sizes, torch_dtype_map[self.torch_dtype], dropout_probability, Encoder.last_instance_id)
        Encoder.last_instance_id += 1
        encoder_handle = encoder_cpp.init()

        self.debug = {}

        self.dropout_probability = dropout_probability

        class EncoderFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, X, *weights):
                Y = torch.empty_like(X, device='cuda')

                forward_interm = []
                for name, layout in layout_forward_interm:
                    sz = tuple(sizes[s] for s in layout)
                    forward_interm.append(torch.empty(*sz, device='cuda', dtype=self.torch_dtype, requires_grad=False))

                encoder_cpp.encoder_forward(
                    encoder_handle, 
                    X.contiguous().data_ptr(), 
                    *(t.contiguous().data_ptr() for t in weights),
                    *(t.contiguous().data_ptr() for t in forward_interm), 
                    Y.contiguous().data_ptr())
                ctx.save_for_backward(*weights, *forward_interm, X)

                if enable_debug:

                    libcudart.cudaDeviceSynchronize()
                    for idx, (name, layout) in enumerate(layout_forward_interm):
                        self.debug[name] = forward_interm[idx].detach().cpu().numpy()

                    for idx, (name, layout) in enumerate(layouts_params):
                        self.debug[name] = weights[idx].detach().cpu().numpy()

                return Y

            @staticmethod
            def backward(ctx, DY):
                DX = torch.empty_like(DY, device='cuda')

                backward_interm = []
                for name, layout in layout_backward_interm:
                    sz = tuple(sizes[s] for s in layout)
                    backward_interm.append(torch.empty(*sz, dtype=self.torch_dtype, device='cuda', requires_grad=False))


                param_gradients = []
                for param, layout in layouts_params:
                    sz = tuple(sizes[s] for s in layout)
                    param_gradients.append(torch.empty(*sz, dtype=self.torch_dtype, device='cuda', requires_grad=False))

                encoder_cpp.encoder_backward(
                    encoder_handle,
                    DY.cuda().contiguous().data_ptr(), 
                    *(t.contiguous().data_ptr() for t in param_gradients),
                    *(t.contiguous().data_ptr() for t in backward_interm),
                    *(t.contiguous().data_ptr() for t in ctx.saved_tensors),
                    DX.cuda().contiguous().data_ptr()
                    )

                if enable_debug:

                    libcudart.cudaDeviceSynchronize()
                    for idx, (name, layout) in enumerate(layout_backward_interm):
                        self.debug[name] = backward_interm[idx].detach().cpu().numpy()

                    for idx, (name, layout) in enumerate(layouts_params):
                        self.debug['D' + name] = param_gradients[idx].detach().cpu().numpy()

                return (DX, *param_gradients)

        self.func = EncoderFunction


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(2)
        for weight in self.parameters():
            tens = torch.empty_like(weight, dtype=torch.float32)
            tens.data.uniform_(-stdv, +stdv)
            with torch.no_grad():
                weight[:] = tens

    def forward(self, X):
        
        return self.func.apply(X, *self.params)

import time
import numpy as np

def isclose(a, b, atol=1e-4, rtol=1e-4):
    if a.shape != b.shape:
        print("Shape missmatch:", a.shape, b.shape)
        return False
    pattern = np.absolute(a - b) < (atol + rtol * np.absolute(b))
    if pattern.all():
        return True
    else:
        a_bad = np.extract(1 - pattern, a)
        b_bad = np.extract(1 - pattern, b)
        bad_ratio = a_bad.size * 1. / pattern.size
        print("Bad ratio", bad_ratio)
        # print(a)
        # print(b)
        print(a_bad)
        print(b_bad)
        return False
        # if bad_ratio < 0.05:
        #     return True
        # else:
        #     print("Bad ratio", bad_ratio)
        #     print(a_bad)
        #     print(b_bad)
        #     return False


if __name__ == '__main__':
    B = 8
    S = 8
    H = 16
    P = 32

    I = H * P

    torch_dtype = torch.float64

    X = torch.randn((B, S, I), dtype=torch_dtype, requires_grad=True).cuda()
    X.retain_grad()

    encoder = Encoder(B, S, H, P, torch_dtype=torch_dtype, enable_debug=True, dropout_probability=0).cuda()

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
    for (name, layout), pt_array in zip(layouts_params, encoder.params):
        numpy_params[name] = np.einsum(layout + '->' + numpy_layout[name], pt_array.detach().cpu().numpy())

    numpy_params['X'] = np.einsum(f'{layout_input[1]}->BJI', X.clone().detach().cpu().numpy())
    numpy_params['WK'] = numpy_params['WKQV'][0]
    numpy_params['WQ'] = numpy_params['WKQV'][1]
    numpy_params['WV'] = numpy_params['WKQV'][2]
    numpy_params['BQKV'] = np.stack((numpy_params['BKQV'][1], numpy_params['BKQV'][0], numpy_params['BKQV'][2]))
    numpy_params['WO'] = numpy_params['WO'].reshape((I, I))
    

    (y,
        i_attn_concat, i_attn_proj_q, i_attn_proj_k, i_attn_proj_v,
        i_attn_scaled_scores, i_attn_dropout_mask,
        i_norm1_mean, i_norm1_std, i_norm1_normed,
        i_linear1_dropout_mask, i_ff_dropout_mask,
        i_norm2_mean, i_norm2_std, i_norm2_normed,
        i_ff_resid, i_ff1, iff1_linear, i_normed1,
        i_attn_resid) = ref_encoder(
            numpy_params['X'], numpy_params['WQ'], numpy_params['WK'], numpy_params['WV'], numpy_params['WO'],
            numpy_params['BQKV'], numpy_params['BO'], 1/np.sqrt(P),
            numpy_params['S1'], numpy_params['B1'], numpy_params['S2'], numpy_params['B2'],
            numpy_params['LINW1'], numpy_params['LINB1'], numpy_params['LINW2'], numpy_params['LINB2'],
            0, 0, 0, activation='relu')

    encoder.train()
    Y = encoder.forward(X)

    # start forward pass check

    KK = encoder.debug['KKQQVV'][0]
    QQ = encoder.debug['KKQQVV'][1]
    VV = encoder.debug['KKQQVV'][2]

    WK = encoder.debug['WKQV'][0]
    WQ = encoder.debug['WKQV'][1]
    WV = encoder.debug['WKQV'][2]

    refWQ = numpy_params['WQ']
    print('WQ: ', isclose(np.einsum(f"{dict(layouts_params)['WKQV'][1:]}->HPI", WQ), refWQ))

    refWK = numpy_params['WK']
    print('WK: ', isclose(np.einsum(f"{dict(layouts_params)['WKQV'][1:]}->HPI", WK), refWK))

    refWV = numpy_params['WV']
    print('WV: ', isclose(np.einsum(f"{dict(layouts_params)['WKQV'][1:]}->HPI", WV), refWV))
    
    layoutQQ = dict(layout_forward_interm)['KKQQVV'][1:]
    refQQ = np.einsum(f"BHJP->{layoutQQ}", i_attn_proj_q)
    print('QQ: ', isclose(QQ, refQQ))

    layoutKK = layoutQQ.translate({ord('J'): 'K'})
    refKK = np.einsum(f"BHKP->{layoutKK}", i_attn_proj_k)
    print('KK: ', isclose(KK, refKK))

    refVV = np.einsum(f"BHKP->{layoutKK}", i_attn_proj_v)
    print('VV: ', isclose(VV, refVV))

    refALPHA = np.einsum(f"BHJK->{dict(layout_forward_interm)['ALPHA']}", i_attn_scaled_scores)
    ALPHA = encoder.debug['ALPHA']
    print('ALPHA: ', isclose(ALPHA, refALPHA))

    myY = np.einsum(f"{layout_output[1]}->BJI", Y.detach().cpu().numpy())
    print('Y: ', isclose(myY, y))

    # end forward pass check

    # run backward pass

    DY = torch.randn(y.shape, dtype=torch_dtype).cuda()

    dy = np.einsum(f"{layout_output[1]}->BJI", DY.detach().cpu().numpy())

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
        numpy_params['WQ'], numpy_params['WK'], numpy_params['WV'], numpy_params['WO'], 1/np.sqrt(P),
        numpy_params['S1'], numpy_params['B1'], numpy_params['S2'], numpy_params['B2'],
        numpy_params['LINW1'], numpy_params['LINB1'], numpy_params['LINW2'], numpy_params['LINB2'],
        0, 0, 0, activation='relu')

    Y.backward(DY)
    DX = X.grad.detach().cpu().numpy()

    # start backward pass check

    print('DX: ', isclose(np.einsum(f"{layout_input[1]}->BJI", DX), dx))
    einsumstr = f"{layout_input[1]},{dict(layout_backward_interm)['DKKQQVV']}->{dict(layouts_params)['WKQV']}"
    print(einsumstr)
    myDWKQV = np.einsum(einsumstr, numpy_params['X'], encoder.debug['DKKQQVV'])

    print('!!!!!!!!!!!!!!!', isclose(myDWKQV, encoder.debug['DWKQV']))

    DWK = encoder.debug['DWKQV'][0]
    DWQ = encoder.debug['DWKQV'][1]
    DWV = encoder.debug['DWKQV'][2]
    DWK = DWK.reshape(P, H, I)
    print('DWK: ', isclose(np.einsum(f"PHI->HPI", DWK), dattn_wk))
    print('DWQ: ', isclose(np.einsum(f"{dict(layouts_params)['WKQV'][1:]}->HPI", DWQ), dattn_wq))
    print('DWV: ', isclose(np.einsum(f"{dict(layouts_params)['WKQV'][1:]}->HPI", DWV), dattn_wv))

    DBKQV = encoder.debug['DBKQV']
    DBQKV = np.stack((DBKQV[1], DBKQV[0], DBKQV[2]))
    print('DBQKV: ', isclose(np.einsum(f"{dict(layouts)['BKQV']}->QHP", DBQKV), dattn_in_b))

    DWO = encoder.debug['DWO']
    refDWO = np.einsum(f"IHP->{dict(layouts_params)['WO']}", dattn_wo.reshape(I, H, P))
    print('DWO: ', isclose(DWO, refDWO))

    print('DBO: ', isclose(encoder.debug['DBO'], dattn_out_b))

    print('DS1: ', isclose(encoder.debug['DS1'], dnorm1_scale))

    print('DS2: ', isclose(encoder.debug['DS2'], dnorm2_scale))

    print('DB1: ', isclose(encoder.debug['DB1'], dnorm1_bias))

    print('DB2: ', isclose(encoder.debug['DB2'], dnorm2_bias))

    print('DLINB1: ', isclose(encoder.debug['DLINB1'], dlinear1_b))

    print('DLINW1: ', isclose(np.einsum(f"{dict(layouts)['LINW1']}->UI", encoder.debug['DLINW1']), dlinear1_w))

    print('DLINB2: ', isclose(encoder.debug['DLINB2'], dlinear2_b))

    print('DLINW2: ', isclose(np.einsum(f"{dict(layouts)['LINW2']}->IU", encoder.debug['DLINW2']), dlinear2_w))

    # end backward pass check

    # run performance test

    B = 96
    S = 128
    H = 16
    P = 64

    I = H * P

    torch_dtype = torch.float16
    X = torch.randn((B, S, I), dtype=torch_dtype, device='cuda')

    encoder = Encoder(B, S, H, P, torch_dtype=torch_dtype).cuda()

    reps = 10
    warmup = 5
    forward = np.zeros(reps-warmup)
    backward = np.zeros(reps-warmup)
    timer = CudaTimer(2)
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(reps):
        timer.start()
        Y = encoder(X)
        timer.next()

        (Y.sum()).backward()
        timer.next()
        timer.end()
        times = timer.get_times()
        if i >= warmup:
            forward[i-warmup] = times[0]
            backward[i-warmup] = times[1]
    torch.cuda.cudart().cudaProfilerStop()

    print('Forward: median: {:.6f} ms (min: {:.6f}, max: {:.6f}, stddev: {:.6f})'.format(np.median(forward), np.min(forward), np.max(forward), np.std(forward)))
    print('Backward: median: {:.6f} ms (min: {:.6f}, max: {:.6f}, stddev: {:.6f})'.format(np.median(backward), np.min(backward), np.max(backward), np.std(backward)))
