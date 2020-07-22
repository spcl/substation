"""PyTorch interface to Substation encoder layers."""

import math
import subprocess
import ctypes
import textwrap
import json
import torch
import hashlib
import pathlib
import os

_torch_dtype_map = {
    torch.float16: 'half',
    torch.float32: 'float',
    torch.float64: 'double'
}

class Encoder(torch.nn.Module):
    """Optimized BERT encoder layer."""

    def __init__(self, compilation_dir, batch_size, max_seq_len, num_heads, embedding_size,
                 layouts, torch_dtype,
                 softmax_dropout_probability, residual1_dropout_probability, activation_dropout_probability, residual2_dropout_probability,
                 enable_debug=False, profiling=False):
        super().__init__()

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.proj_size = self.embedding_size // self.num_heads
        self.intermediate_size = self.embedding_size * 4
        self.softmax_dropout_probability = softmax_dropout_probability
        self.residual1_dropout_probability = residual1_dropout_probability
        self.activation_dropout_probability = activation_dropout_probability
        self.residual2_dropout_probability = residual2_dropout_probability
        self.torch_dtype = torch_dtype
        self.debug = {}  # Will be used to capture intermediate tensors.
        self.layouts = layouts

        # This maps "human readable" sizses to the layout names.
        sizes = {
            'Q': 3,
            'B': self.batch_size,
            'S': self.max_seq_len,
            'H': self.num_heads,
            'P': self.proj_size,
            'I': self.embedding_size,
            'U': self.intermediate_size,
            'J': self.max_seq_len,
            'K': self.max_seq_len
        }

        # Set up parameters.
        for param, layout in layouts['params']:
            sz = tuple(sizes[s] for s in layout)
            setattr(self, param, torch.nn.Parameter(torch.empty(
                *sz, dtype=self.torch_dtype)))
        self.params = tuple(getattr(self, k) for k, _ in layouts['params'])
        self.reset_parameters()

        # Compile the optimized version.
        encoder_cpp = Encoder._compile_module(
            sizes, _torch_dtype_map[self.torch_dtype], layouts,
            self.softmax_dropout_probability,
            self.residual1_dropout_probability,
            self.activation_dropout_probability,
            self.residual2_dropout_probability,
            profiling, compilation_dir)
        encoder_handle = encoder_cpp.init()

        # Define the actual autograd function.
        class EncoderFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, X, *weights):
                # TODO: Should specify device better.
                Y = torch.empty_like(X, device='cuda')
                # Allocate space for intermediate data.
                forward_intermediate = []
                for name, layout in layouts['forward_interm']:
                    sz = tuple(sizes[s] for s in layout)
                    forward_intermediate.append(torch.empty(
                        *sz, device='cuda', dtype=self.torch_dtype,
                        requires_grad=False))

                # Run forward prop.
                encoder_cpp.encoder_forward(
                    encoder_handle,
                    X.contiguous().data_ptr(),
                    *(t.contiguous().data_ptr() for t in weights),
                    *(t.contiguous().data_ptr() for t in forward_intermediate),
                    Y.contiguous().data_ptr())
                ctx.save_for_backward(*weights, *forward_intermediate, X)

                if enable_debug:
                    torch.cuda.synchronize()
                    for idx, (name, _) in enumerate(layouts['forward_interm']):
                        self.debug[name] = forward_intermediate[idx].detach().cpu().numpy()
                    for idx, (name, _) in enumerate(layouts['params']):
                        self.debug[name] = weights[idx].detach().cpu().numpy()

                return Y


            @staticmethod
            def backward(ctx, DY):
                DX = torch.empty_like(DY, device='cuda')
                # Allocate space for intermediate data and parameter gradients.
                backward_intermediate = []
                for name, layout in layouts['backward_interm']:
                    sz = tuple(sizes[s] for s in layout)
                    backward_intermediate.append(torch.empty(
                        *sz, device='cuda', dtype=self.torch_dtype,
                        requires_grad=False))
                param_gradients = []
                for param, layout in layouts['params']:
                    sz = tuple(sizes[s] for s in layout)
                    param_gradients.append(torch.empty(
                        *sz, device='cuda', dtype=self.torch_dtype,
                        requires_grad=False))

                # Run backprop.
                encoder_cpp.encoder_backward(
                    encoder_handle,
                    DY.contiguous().data_ptr(),
                    *(t.contiguous().data_ptr() for t in param_gradients),
                    *(t.contiguous().data_ptr() for t in backward_intermediate),
                    *(t.contiguous().data_ptr() for t in ctx.saved_tensors),
                    DX.contiguous().data_ptr())

                if enable_debug:
                    torch.cuda.synchronize()
                    for idx, (name, _) in enumerate(layouts['backward_interm']):
                        self.debug[name] = backward_intermediate[idx].detach().cpu().numpy()
                    for idx, (name, _) in enumerate(layouts['params']):
                        self.debug['D' + name] = param_gradients[idx].detach().cpu().numpy()

                return (DX, *param_gradients)

        self.encoder_func = EncoderFunction


    def reset_parameters(self):
        stdev = 1.0 / math.sqrt(2)

        for weight in self.parameters():
            if self.torch_dtype == torch.float16:
                # half precision is not supported on CPU side,
                # so we need to initialize everything with float32 precision
                # and then copy to GPU
                t = torch.empty_like(weight, dtype=torch.float32)
                torch.nn.init.uniform_(t, -stdev, stdev)
                with torch.no_grad():
                    weight[:] = t
            else:
                with torch.no_grad():
                    torch.nn.init.uniform_(weight, -stdev, stdev)

    def forward(self, X):
        """Run the encoder."""
        return self.encoder_func.apply(X, *self.params)


    @staticmethod
    def _compile_module(sizes, datatype, layouts,
                        softmax_dropout_probability, residual1_dropout_probability, activation_dropout_probability, residual2_dropout_probability,
                        profiling, compilation_dir):

        # create compilation dir if it doesn't exist
        os.makedirs(compilation_dir, exist_ok=True)

        # caching mechanism
        module_params = {
            'sizes': sizes,
            'datatype': datatype,
            'layouts': layouts,
            'softmax_dropout_probability': softmax_dropout_probability,
            'residual1_dropout_probability': residual1_dropout_probability,
            'activation_dropout_probability': activation_dropout_probability,
            'residual2_dropout_probability': residual2_dropout_probability,
            'profiling': profiling
        }

        module_string = json.dumps(module_params, sort_keys=True)
        module_string = ''.join(c if c.isalnum() else '' for c in module_string)
        module_hash = hashlib.md5(module_string.encode()).hexdigest()

        compilation_hashes = os.path.join(compilation_dir, 'compilation_hashes.txt')

        try:
            with open(compilation_hashes, 'r') as f:
                known_hashes = json.load(f)
        except IOError:
            known_hashes = {}

        old_module_string = known_hashes.get(module_hash)
        compilation_required = old_module_string != module_string

        if compilation_required:
            known_hashes[module_hash] = module_string

        with open(compilation_hashes, 'w') as f:
            json.dump(known_hashes, f)

        module_compilation_dir = os.path.join(compilation_dir, module_hash)
        os.makedirs(module_compilation_dir, exist_ok=True)

        if compilation_required:
            """Generate and compile module."""

            # Write the layouts out.
            with open(os.path.join(module_compilation_dir, 'encoder_parameters.cuh'), 'w') as f:
                f.write('#pragma once\n\n')

                if profiling:
                    f.write('#define SUBSTATION_PROFILING\n\n')

                f.write('\n'.join([f'#define size{k} {v}' for k, v in sizes.items()]))
                f.write('\n\n')

                f.write(f'using Real = {datatype};\n\n')

                f.write(f'#define ENCODER_SOFTMAX_DROPOUT_PROBABILITY {softmax_dropout_probability}\n\n')
                f.write(f'#define ENCODER_RESIDUAL1_DROPOUT_PROBABILITY {residual1_dropout_probability}\n\n')
                f.write(f'#define ENCODER_ACTIVATION_DROPOUT_PROBABILITY {activation_dropout_probability}\n\n')
                f.write(f'#define ENCODER_RESIDUAL2_DROPOUT_PROBABILITY {residual2_dropout_probability}\n\n')

                f.write(f'#define ARRAY_X_DEF Real* g{layouts["input"][0]}\n\n')
                f.write(f'#define ARRAY_X g{layouts["input"][0]}\n\n')

                f.write('#define ARRAY_WEIGHTS_DEF ' + ', '.join(
                    [f'Real* g{k}' for k, v in layouts['params']]))
                f.write('\n')
                f.write('#define ARRAY_WEIGHTS ' + ', '.join(
                    [f'g{k}' for k, v in layouts['params']]))
                f.write('\n\n')

                f.write('#define ARRAY_FWD_INTERM_DEF ' + ', '.join(
                    [f'Real* g{k}' for k, v in layouts['forward_interm']]))
                f.write('\n')
                f.write('#define ARRAY_FWD_INTERM ' + ', '.join(
                    [f'g{k}' for k, v in layouts['forward_interm']]))
                f.write('\n\n')

                f.write(f'#define ARRAY_Y_DEF Real* g{layouts["output"][0]}\n')
                f.write(f'#define ARRAY_Y g{layouts["output"][0]}\n\n')

                f.write(f'#define ARRAY_D_Y_DEF Real* gD{layouts["output"][0]}\n')
                f.write(f'#define ARRAY_D_Y gD{layouts["output"][0]}\n\n')

                f.write('#define ARRAY_D_WEIGHTS_DEF ' + ', '.join(
                    [f'Real* gD{k}' for k, v in layouts['params']]))
                f.write('\n')
                f.write('#define ARRAY_D_WEIGHTS ' + ', '.join(
                    [f'gD{k}' for k, v in layouts['params']]))
                f.write('\n\n')

                f.write('#define ARRAY_BWD_INTERM_DEF ' + ', '.join(
                    [f'Real* g{k}' for k, v in layouts['backward_interm']]))
                f.write('\n')
                f.write('#define ARRAY_BWD_INTERM ' + ', '.join(
                    [f'g{k}' for k, v in layouts['backward_interm']]))
                f.write('\n\n')

                f.write(f'#define ARRAY_D_X_DEF Real* gD{layouts["input"][0]}\n')
                f.write(f'#define ARRAY_D_X gD{layouts["input"][0]}\n\n')

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

                all_layouts = dict(
                    layouts['forward_interm'] + layouts['backward_interm']
                    + layouts['params'] + [layouts['input'], layouts['output']]
                    + [('D' + k, v) for k, v in
                       [*layouts['params'], layouts['input'], layouts['output']]])

                assert all_layouts['KKQQVV'][0] == 'Q'
                KK_layout = all_layouts['KKQQVV'][1:]
                assert all_layouts['DKKQQVV'][0] == 'Q'
                DKK_layout = all_layouts['DKKQQVV'][1:]
                assert all_layouts['WKKWQQWVV'][0] == 'Q'
                WK_K_layout = all_layouts['WKKWQQWVV'][1:]
                assert all_layouts['BKQV'][0] == 'Q'
                BK_layout = all_layouts['BKQV'][1:]
                assert all_layouts['WKQV'][0] == 'Q'
                WK_layout = all_layouts['WKQV'][1:]

                # Add some additional definitions.
                all_layouts['WKKself'] = WK_K_layout
                all_layouts['WQQself'] = WK_K_layout
                all_layouts['WVVself'] = WK_K_layout
                all_layouts['BK'] = BK_layout
                all_layouts['BQ'] = BK_layout
                all_layouts['BV'] = BK_layout
                all_layouts['DBK'] = BK_layout
                all_layouts['DBQ'] = BK_layout
                all_layouts['DBV'] = BK_layout
                all_layouts['Q'] = all_layouts['X']
                all_layouts['K'] = all_layouts['X'].translate({ord('J'): 'K'})
                all_layouts['V'] = all_layouts['X'].translate({ord('J'): 'K'})
                all_layouts['KK'] = KK_layout.translate({ord('J'): 'K'})
                all_layouts['QQ'] = KK_layout
                all_layouts['VV'] = KK_layout.translate({ord('J'): 'K'})
                all_layouts['DKK'] = DKK_layout.translate({ord('J'): 'K'})
                all_layouts['DQQ'] = DKK_layout
                all_layouts['DVV'] = DKK_layout.translate({ord('J'): 'K'})
                all_layouts['KKself'] = KK_layout
                all_layouts['QQself'] = KK_layout
                all_layouts['VVself'] = KK_layout
                all_layouts['DKKself'] = DKK_layout
                all_layouts['DQQself'] = DKK_layout
                all_layouts['DVVself'] = DKK_layout
                all_layouts['WKK'] = WK_K_layout
                all_layouts['WQQ'] = WK_K_layout
                all_layouts['WVV'] = WK_K_layout
                all_layouts['WK'] = WK_layout
                all_layouts['WQ'] = WK_layout
                all_layouts['WV'] = WK_layout
                all_layouts['DWK'] = WK_layout
                all_layouts['DWQ'] = WK_layout
                all_layouts['DWV'] = WK_layout

                for k, v in all_layouts.items():
                    layout = ', '.join([f'dim{x}' for x in v])
                    layout_def = f'using l{k} = metal::list<{layout}>;\n'
                    f.write(layout_def)
                f.write('\n\n')

                for gemm, algo in layouts['algorithms'].items():
                    f.write(f'#define algo{gemm} CUBLAS_GEMM_{algo}_TENSOR_OP\n')
                f.write('\n\n')

                for name, dim in layouts['special_dims'].items():
                    f.write(f'#define sd{name} dim{dim}\n')
                f.write('\n\n')

            # Compile and load with ctypes.
            print('Compiling...')
            subprocess.run(['nvcc', '-O3',
                            '-gencode', 'arch=compute_61,code=sm_61',
                            '-gencode', 'arch=compute_70,code=sm_70',
                            '-c', '--compiler-options', '-fPIC',
                            '-I', module_compilation_dir,
                            'encoder.cu',
                            '-o', os.path.join(module_compilation_dir, 'encoder.o')])
            subprocess.run(['nvcc', '-shared',
                            os.path.join(module_compilation_dir, 'encoder.o'),
                            '-o', os.path.join(module_compilation_dir, 'encoder.so')])
            print('Compiling done')

        else:
            print('Compilation is not required. Using cached build.')

        lib = ctypes.CDLL(os.path.join(module_compilation_dir, './encoder.so'))
        lib.init.argtypes = []
        lib.init.restype = ctypes.c_void_p
        lib.encoder_forward.argtypes = [ctypes.c_void_p] * (
            1 + 1 + len(layouts['params']) + len(layouts['forward_interm']) + 1)
        lib.encoder_backward.argtypes = [ctypes.c_void_p] * (
            1 + 1 + len(layouts['params']) + len(layouts['backward_interm']) + (
                len(layouts['params']) + len(layouts['forward_interm']) + 1) + 1)
        lib.destroy.argtypes = [ctypes.c_void_p]

        return lib
