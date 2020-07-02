import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer
from copy import deepcopy
import time
import itertools
import argparse
import socket
import glob
import shutil
import sys
import pickle

# Temporary hack to switch things up for Lassen.
if 'lassen' in socket.gethostname():
    nvcc_gencode = '-gencode arch=compute_70,code=sm_70'
    check_correctness = True
    work_from_tmp = True
    tmp_dir = '/dev/shm/'
    output_to_file = True
else:
    nvcc_gencode = '-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70'
    check_correctness = True
    work_from_tmp = False
    output_to_file = False


def list_dependencies():
    return glob.glob('*.hpp') + glob.glob('*.cuh') + glob.glob('*.h')


def switch_to_tmp():
    for file in list_dependencies():
        shutil.copy(file, tmp_dir)
    os.chdir(tmp_dir)


def get_local_rank():
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        return 0


def get_local_size():
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    elif 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        return 1


def get_world_rank():
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        return 0


def get_world_size():
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        return 1


def get_gpu_helper():
    gpu_helper_basename = f'gpu_helper-{get_world_rank()}'
    if not os.path.exists(gpu_helper_basename + '.so'):
        gpu_mem_source = """
            #include "blocks.cuh"
            #include <cuda.h>
            #include <stdio.h>

            extern "C" {
                void* gpu_allocate(size_t size) {
                    void* ptr = nullptr;
                    CHECK(cudaMalloc(&ptr, size));
                    CHECK(cudaMemset(ptr, 0, size));
                    return ptr;
                }

                void gpu_free(void* ptr) {
                    CHECK(cudaFree(ptr));
                }

                void host_to_gpu(void* gpu, void* host, size_t size) {
                    CHECK(cudaMemcpy(gpu, host, size, cudaMemcpyHostToDevice));
                }

                void gpu_to_host(void* host, void* gpu, size_t size) {
                    CHECK(cudaMemcpy(host, gpu, size, cudaMemcpyDeviceToHost));
                }

                void device_synchronize() {
                    CHECK(cudaDeviceSynchronize());
                }

                int fast_allclose(float* a, float* b, size_t size, float atol, float rtol) {
                    for (size_t i = 0; i < size; ++i) {
                        if (fabs(a[i] - b[i]) > atol + rtol*fabs(b[i])) {
                            return 0;
                        }
                    }
                    return 1;
                }
            }
        """
        
        with open(gpu_helper_basename + '.cu', 'w') as f:
            f.write(gpu_mem_source)
            
        subprocess.run(f"nvcc -O3 {nvcc_gencode} -c --compiler-options -fPIC {gpu_helper_basename}.cu -o {gpu_helper_basename}.o".split(' '))
        subprocess.run(f"nvcc -shared -o {gpu_helper_basename}.so {gpu_helper_basename}.o".split(' '))
    
    lib = ctypes.CDLL(f'./{gpu_helper_basename}.so')
    
    lib.gpu_allocate.argtypes = [ctypes.c_size_t]
    lib.gpu_allocate.restype = ctypes.c_void_p
    
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    
    lib.host_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    
    lib.gpu_to_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

    lib.fast_allclose.argtypes = [ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_size_t,
                                  ctypes.c_float,
                                  ctypes.c_float]
    lib.fast_allclose.retype = [ctypes.c_int]
    
    return lib


def unload_lib(lib):
    cdll = ctypes.CDLL("libdl.so")
    cdll.dlclose.restype = ctypes.c_int
    cdll.dlclose.argtypes = [ctypes.c_void_p]
    res = cdll.dlclose(lib._handle)
    if res != 0:
        raise Exception("dlclose failed")


def get_kernel_name(params):
        # A hopefully-unique name for a kernel file.
        return f'temp-rank{get_world_rank()}-{hash(tuple(params.items()))}'


def filter_sizes(sizes, dim_names):
    new_sizes = []
    for size in sizes:
        new_sizes.append(
            dict([(name, size[name]) for name in dim_names]))
    return new_sizes


class Kernel:
    def __init__(self):
        pass

    def setup_gpu_helper(self):
        self.gpu_helper = get_gpu_helper()

    def build_kernel_async(self, kernel_basename, params):
        self.generate_source(**params, filename=kernel_basename + '.cu')
        return self.build(kernel_basename + '.cu', kernel_basename + '.so',
                          wait=False)

    def get_lib(self, kernel_basename):
        return self.load(kernel_basename + '.so')

    def clear_kernel(self, kernel_basename):
        os.unlink(kernel_basename + '.cu')
        os.unlink(kernel_basename + '.so')

    def build(self, source, output, wait=True):
        command = f"""
            nvcc -shared -O3
            {nvcc_gencode}
            -Xcompiler -fPIC
            {source} -o {output}
        """.format(source=source, output=output).split()
        if wait:
            subprocess.run(command)
        else:
            return subprocess.Popen(command)

    def load(self, lib):
        lib = ctypes.CDLL("./" + lib)
        lib.benchmark.argtypes = [ctypes.c_void_p for _ in self.input_mapping]
        return lib

    def all_permutations(self):
        for e in self.all_arrays:
            yield itertools.permutations(self.all_arrays[e])

    def _generate_layouts_list(self):
        permutations = itertools.product(*self.all_permutations())
        
        for p in permutations:
            yield {k: "".join(v) for k, v in zip(self.all_arrays.keys(), p)}

    def generate_layouts_list(self):
        # Split layouts among workers.
        layouts = list(self._generate_layouts_list())
        layouts_per_rank = [len(layouts) // get_world_size()] * get_world_size()
        remainder = len(layouts) % get_world_size()
        for i in range(remainder): layouts_per_rank[i] += 1
        layouts_prefix_sum, layouts_sum = [0], 0
        for i in range(get_world_size()):
            layouts_sum += layouts_per_rank[i]
            layouts_prefix_sum.append(layouts_sum)
        start_layout = layouts_prefix_sum[get_world_rank()]
        end_layout = layouts_prefix_sum[get_world_rank() + 1]
        print(f'{get_world_rank()}: Handling layouts for {start_layout}:{end_layout} of {len(layouts)}')
        sys.stdout.flush()
        return layouts[start_layout:end_layout]

    def allocate_gpu_arrays(self, sizes):
        gpu_arrays = {}
        for k in self.all_arrays:
            size = reduce(lambda a,b: a*b, [sizes[dim] for dims in self.all_arrays[k] for dim in dims])
            gpu_arrays[k] = self.gpu_helper.gpu_allocate(2 * size)
        return gpu_arrays

    def copy_input_arrays(self, arrays, gpu_arrays):
        for k in self.input_arrays:
            array = arrays[k]
            gpu_array = gpu_arrays[k]
            self.gpu_helper.host_to_gpu(gpu_array, array.ctypes.data, array.nbytes)

    def generate_arrays(self, layout, sizes):
        arrays = {}

        for k in self.input_arrays:
            size = [sizes[l] for l in layout[k]]
            arrays[k] = np.random.rand(*size).astype(np.float16)

        for k in self.output_arrays:
            size = [sizes[l] for l in layout[k]]
            arrays[k] = np.zeros(size, dtype=np.float16)

        return arrays

    def run_measurement(self, lib, kernel_params):

        warmup = 5
        repetitions = 25

        for _ in range(warmup):
            lib.benchmark(*kernel_params)
        self.gpu_helper.device_synchronize()
        
        start = timer()
        for _ in range(repetitions):
            lib.benchmark(*kernel_params)
        self.gpu_helper.device_synchronize()
        end = timer()
        runtime = (end - start) / repetitions
        return runtime

    
    def copy_output_arrays(self, arrays, gpu_arrays):
        for k in self.output_arrays:
            array = arrays[k]
            gpu_array = gpu_arrays[k]
            self.gpu_helper.gpu_to_host(array.ctypes.data, gpu_array, array.nbytes)

    def free_gpu_arrays(self, arrays):
        for v in arrays:
            self.gpu_helper.gpu_free(v)

    def allclose(self, a, b):
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        if a.size != b.size:
            return False
        return self.gpu_helper.fast_allclose(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.size, 1e-2, 1e-2)

def check_dropout_mask(mask):
    if (np.absolute(mask) < 1e-5).all():
        raise Exception("droupout mask is 0")
        return False

    if (np.absolute(mask - 1) < 1e-5).all():
        raise Exception("droupout mask is 1")
        return False

    pattern = np.logical_or(np.absolute(mask) < 1e-5, (np.absolute(mask - 1) < 1e-5))
    if not pattern.all():
        raise Exception("droupout mask is neither 0 nor 1")
        return False

class SoftmaxKernel(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'HBJK')
        self.input_arrays = dict(
            IN='HBJK'
        )
        self.output_arrays = dict(
            OUT='HBJK',
            DROP_MASK='HBJK',
            DROP='HBJK'
        )
        self.vector_dims = 'HBJK'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = ['IN', 'OUT', 'DROP_MASK', 'DROP']


    def generate_source(self, H, B, J, K, vec_dim, IN, OUT, DROP_MASK, DROP, filename):
        source = """
            #include "block_softmax.cuh"
        
            struct H { enum { value = %d }; };
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct K { enum { value = %d }; };
            using lIN = metal::list<%s>;
            using lOUT = metal::list<%s>;
            using lDROP_MASK = metal::list<%s>;
            using lDROP = metal::list<%s>;

            extern "C" {
                void benchmark(half* in, half* out, half* drop_mask, half* drop) {
                    GlobalRandomState grs;
                    Softmax<half, K, %s, lIN, lOUT, lDROP_MASK, lDROP>::run(in, out, drop_mask, drop, 0.5, grs, 0);
                }
            }
        """ % (H, B, J, K,
            ",".join(list(IN)),
            ",".join(list(OUT)),
            ",".join(list(DROP_MASK)),
            ",".join(list(DROP)),
            vec_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for vec_dim in self.vector_dims:
            for sizes in self.sizes:
                for layout in layouts:
                    # Hack to simplify cases:
                    if layout['DROP'] != layout['DROP_MASK']: continue
                    params = {**sizes, 'vec_dim': vec_dim, **layout}
                    yield params, sizes, layout

    @staticmethod
    def ref_softmax(x, axis):
        exps = np.exp(x - x.max(axis=axis, keepdims=True))
        return exps / exps.sum(axis=axis, keepdims=True)

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        check_dropout_mask(ca['DROP_MASK'])

        my = {}
        my['OUT'] = self.ref_softmax(ca['IN'], axis=-1)
        my['DROP'] = my['OUT'] * ca['DROP_MASK']

        for arr in ['OUT', 'DROP']:
            if not self.allclose(my[arr], ca[arr]):
                print(f'{arr} is wrong')
                return False        

        return True

class BiasActivationDropoutKernel(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'BJU')
        self.input_arrays = dict(
            SB1_LINW1='BJU',
            LINB1='U'
        )
        self.output_arrays = dict(
            DROP2='BJU',
            DROP2MASK='BJU',
            LIN1='BJU'
        )
        self.vec_dims = 'BJU'
        self.thread_dims = 'BJU'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = ['SB1_LINW1', 'DROP2', 'LINB1', 'LIN1', 'DROP2MASK']


    def generate_source(self, B, J, U, vec_dim, thread_dim, SB1_LINW1, DROP2, LINB1, LIN1, DROP2MASK, filename):
        source = """
            #include "block_bad.cuh"
        
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct U { enum { value = %d }; };
            using lSB1_LINW1 = metal::list<%s>;
            using lDROP2 = metal::list<%s>;
            using lLIN1 = metal::list<%s>;
            using lDROP2MASK = metal::list<%s>;

            extern "C" {
                void benchmark(half* gSB1_LINW1, half* gDROP2, half* gLINB1, half* gLIN1, half* gDROP2MASK) {
                    GlobalRandomState grs;
                    BiasActivationDropout<half, B, J, U, %s, %s, lSB1_LINW1, lDROP2, lLIN1, lDROP2MASK>::run(
                        gSB1_LINW1, gDROP2, gLINB1, gLIN1, gDROP2MASK, 0.5f, grs, 0);
                }
            }
        """ % (B, J, U,
            ",".join(list(SB1_LINW1)), 
            ",".join(list(DROP2)),
            ",".join(list(LIN1)),
            ",".join(list(DROP2MASK)),
            vec_dim,
            thread_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for sizes in self.sizes:
            for vec_dim in self.vec_dims:
                for thread_dim in self.thread_dims:
                    for layout in layouts:
                        params = {**sizes, 'vec_dim': vec_dim, 'thread_dim': thread_dim, **layout}
                        yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        check_dropout_mask(ca['DROP2MASK'])

        my = {}
        my['LIN1'] = ca['SB1_LINW1'] + np.broadcast_to(ca['LINB1'], shape=ca['SB1_LINW1'].shape)
        relu = np.maximum(0, my['LIN1'])
        my['DROP2'] = ca['DROP2MASK'] * relu

        for arr in ['LIN1', 'DROP2']:
            if not self.allclose(my[arr], ca[arr]):
                print(f'{arr} is wrong')
                return False

        return True
            

class BiasDropoutResidualLinearNorm(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'BJN')

        self.input_arrays = dict(
            DROP2_LINW2='BJN',
            SB1='BJN',
            S2='N',
            B2='N',
            LINB2='N'
        )
        self.output_arrays = dict(
            SB2='BJN',
            LN2='BJN',
            LIN2='BJN',
            LN2DIFF='BJN',
            LN2STD='BJ',
            DROP3MASK='BJN'
        )
        self.vector_dims = 'BJN'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = [
            'SB2', 'DROP2_LINW2', 'SB1', 'LN2',
            'S2', 'B2', 'LINB2', 'LIN2', 'LN2DIFF', 
            'LN2STD', 'DROP3MASK']


    def generate_source(self, B, J, N, vec_dim, SB2, DROP2_LINW2, SB1, LN2, LN2DIFF, LN2STD, LIN2, DROP3MASK, S2, B2, LINB2, filename):
        source = """
            #include "block_bdrln.cuh"
        
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct N { enum { value = %d }; };

            using lSB2 = metal::list<%s>;
            using lDROP2_LINW2 = metal::list<%s>;
            using lSB1 = metal::list<%s>;
            using lLN2 = metal::list<%s>;
            using lLN2DIFF = metal::list<%s>;
            using lLN2STD = metal::list<%s>;
            using lLIN2 = metal::list<%s>;
            using lDROP3MASK = metal::list<%s>;

            extern "C" {
                void benchmark(
                    half* gSB2, half* gDROP2_LINW2, half* gSB1, half* gLN2,
                    half* gS2, half* gB2, half* gLINB2, half* gLIN2, half* gLN2DIFF, 
                    half* gLN2STD, half* gDROP3MASK) 
                {
                    GlobalRandomState grs;
                    BiasDropoutResidualLinearNorm<half, true, N, %s, lSB2, lDROP2_LINW2, lSB1, lLN2, lLN2DIFF, lLN2STD, lLIN2, lDROP3MASK>::run(
                        gSB2, gDROP2_LINW2, gSB1, gLN2, gS2, gB2, gLINB2, gLIN2, gLN2DIFF, gLN2STD, gDROP3MASK, 0.5f, grs, 0);
                }
            }
        """ % (B, J, N, 
            ",".join(list(SB2)), 
            ",".join(list(DROP2_LINW2)),
            ",".join(list(SB1)),
            ",".join(list(LN2)), 
            ",".join(list(LN2DIFF)),
            ",".join(list(LN2STD)),
            ",".join(list(LIN2)),
            ",".join(list(DROP3MASK)),
            vec_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for vec_dim in self.vector_dims:
            for sizes in self.sizes:
                for layout in layouts:
                    params = {**sizes, 'vec_dim': vec_dim, **layout}
                    yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        check_dropout_mask(ca['DROP3MASK'])

        my = {}
        my['LIN2'] = ca['DROP2_LINW2'] + np.broadcast_to(ca['LINB2'], shape=ca['DROP2_LINW2'].shape)
        drop3 = my['LIN2'] * ca['DROP3MASK']
        resid2 = drop3 + ca['SB1']
        ln2mean = np.mean(resid2, axis=-1)
        my['LN2STD'] = 1 / np.std(resid2, axis=-1)
        my['LN2DIFF'] = resid2 - np.repeat(ln2mean[..., np.newaxis], resid2.shape[-1], axis=-1)
        my['LN2'] = my['LN2DIFF'] * np.repeat(my['LN2STD'][..., np.newaxis], resid2.shape[-1], axis=-1)
        my['SB2'] = my['LN2'] * ca['S2'] + ca['B2']

        for arr in ['LIN2', 'LN2STD', 'LN2DIFF', 'LN2', 'SB2']:
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                return False

        return True


class BackwardLayerNormResidualDropout(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'BJN')

        self.input_arrays = dict(
            DLN2='BJN',
            LN2STD='BJ',
            LN2DIFF='BJN',
            DROP3MASK='BJN'
        )
        self.output_arrays = dict(
            DRESID2='BJN',
            DLIN2='BJN'
        )
        self.vector_dims = 'BJN'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = [
            'DLN2', 'LN2STD', 'LN2DIFF', 'DROP3MASK', 'DRESID2', 'DLIN2']


    def generate_source(self, B, J, N, vec_dim, DLN2, LN2STD, LN2DIFF, DROP3MASK, DRESID2, DLIN2, filename):
        source = """
            #include "block_blnrd.cuh"
        
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct N { enum { value = %d }; };

            using lDLN2 = metal::list<%s>;
            using lLN2STD = metal::list<%s>;
            using lLN2DIFF = metal::list<%s>;
            using lDROP3MASK = metal::list<%s>;
            using lDRESID2 = metal::list<%s>;
            using lDLIN2 = metal::list<%s>;

            extern "C" {
                void benchmark(
                    half* gDLN2, half* gLN2STD, half* gLN2DIFF, half* gDROP3MASK,
                    half* gDRESID2, half* gDLIN2, curandStatePhilox4_32_10_t* curandStates) 
                {
                    BackwardLayerNormResidualDropout<half, N, %s, 
                        lDLN2, lLN2STD, lLN2DIFF, lDROP3MASK,
                        lDRESID2, lDLIN2
                    >::run(
                        gDLN2, gLN2STD, gLN2DIFF, gDROP3MASK,
                        gDRESID2, gDLIN2, 0
                    );
                }
            }
        """ % (B, J, N, 
            ",".join(list(DLN2)), 
            ",".join(list(LN2STD)),
            ",".join(list(LN2DIFF)),
            ",".join(list(DROP3MASK)), 
            ",".join(list(DRESID2)),
            ",".join(list(DLIN2)),
            vec_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for vec_dim in self.vector_dims:
            for sizes in self.sizes:
                for layout in layouts:
                    params = {**sizes, 'vec_dim': vec_dim, **layout}
                    yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        N = ca['DLN2'].shape[-1]
        resid2_a = ca['DLN2'] * np.repeat(ca['LN2STD'][..., np.newaxis], N, axis=-1)
        resid2_b = np.sum(ca['DLN2'], axis=-1) * ca['LN2STD'] / N
        resid2_b = np.repeat(resid2_b[..., np.newaxis], N, axis=-1)
        resid2_c = np.sum(ca['DLN2'] * ca['LN2DIFF'], axis=-1) * ca['LN2STD'] ** 3 / N
        resid2_c = np.repeat(resid2_c[..., np.newaxis], N, axis=-1) * ca['LN2DIFF']

        my = {}
        my['DRESID2'] = resid2_a - resid2_b - resid2_c
        my['DLIN2'] = my['DRESID2'] * ca['DROP3MASK']

        for arr in ['DRESID2', 'DLIN2']:
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                return False

        return True


class BackwardScaleBias(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'BJN')

        self.input_arrays = dict(
            LN2='BJN',
            S2='N',
            DSB2='BJN'
        )
        self.output_arrays = dict(
            DLN2='BJN',
            DS2='N',
            DB2='N'
        )
        self.vector_dims = 'BJN'
        self.warp_dims = 'BJ'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = [
            'LN2', 'S2', 'DSB2', 'DLN2', 'DS2', 'DB2']


    def generate_source(self, B, J, N, vec_dim, warp_dim, LN2, S2, DSB2, DLN2, DS2, DB2, filename):
        source = """
            #include "block_bsb_ebsb.cuh"
        
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct N { enum { value = %d }; };

            using lLN2 = metal::list<%s>;
            using lDLN2 = metal::list<%s>;
            using lDSB2 = metal::list<%s>;

            extern "C" {
                void benchmark(
                    half* gLN2, half* gS2, half* gDSB2,
                    half* gDLN2, half* gDS2, half* gDB2) 
                {
                    BackwardScaleBias<half, B, J, N, %s, %s, lLN2, lDLN2, lDSB2>::run(
                        gLN2, gS2, gDSB2,
                        gDLN2, gDS2, gDB2,
                        0);
                }
            }
        """ % (B, J, N, 
            ",".join(list(LN2)), 
            ",".join(list(DLN2)),
            ",".join(list(DSB2)),
            vec_dim,
            warp_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for vec_dim in self.vector_dims:
            for warp_dim in self.warp_dims:
                for sizes in self.sizes:
                    for layout in layouts:
                        params = {**sizes, 'vec_dim': vec_dim, 'warp_dim': warp_dim, **layout}
                        yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        my = {}
        my['DLN2'] = np.einsum("bji,i->bji", ca['DSB2'], ca['S2'])
        my['DS2'] = np.einsum("bji,bji->i", ca['DSB2'], ca['LN2'])
        my['DB2'] = np.einsum("bji->i", ca['DSB2'])

        for arr in ['DLN2', 'DB2', 'DS2']:
            if not self.allclose(my[arr], ca[arr]):
                print(my[arr])
                print('-' * 200)
                print(ca[arr])
                print("%s is wrong" % arr)
                return False

        return True


class BackwardDropoutReluLinearBias(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'BJU')

        self.input_arrays = dict(
            DDROP2='BJU',
            DROP2MASK='BJU',
            LIN1='BJU'
        )
        self.output_arrays = dict(
            DLIN1='BJU',
            DLINB1='U'
        )
        self.vector_dims = 'BJU'
        self.warp_dims = 'BJ'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = [
            'DDROP2', 'DROP2MASK', 'LIN1', 'DLIN1', 'DLINB1']


    def generate_source(self, B, J, U, vec_dim, warp_dim, DDROP2, DROP2MASK, LIN1, DLIN1, DLINB1, filename):
        source = """
            #include "block_bdrlb.cuh"
        
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct U { enum { value = %d }; };

            using lDDROP2 = metal::list<%s>;
            using lDROP2MASK = metal::list<%s>;
            using lLIN1 = metal::list<%s>;
            using lDLIN1 = metal::list<%s>;

            extern "C" {
                void benchmark(
                    half* DDROP2, half* DROP2MASK, half* LIN1,
                    half* DLIN1, half* DLINB1) 
                {
                    BackwardDropoutReluLinearBias<half,
                        B, J, U,
                        %s, %s,
                        lDDROP2, lDROP2MASK, lLIN1, lDLIN1
                    >::run(DDROP2, DROP2MASK, LIN1,
                        DLIN1, DLINB1, 0);
                }
            }
        """ % (B, J, U,
            ",".join(list(DDROP2)), 
            ",".join(list(DROP2MASK)),
            ",".join(list(LIN1)),
            ",".join(list(DLIN1)),
            vec_dim,
            warp_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for vec_dim in self.vector_dims:
            for warp_dim in self.warp_dims:
                for sizes in self.sizes:
                    for layout in layouts:
                        params = {**sizes, 'vec_dim': vec_dim, 'warp_dim': warp_dim, **layout}
                        yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        my = {}
        dact = ca['DDROP2'] * ca['DROP2MASK']
        my['DLIN1'] = (ca['LIN1'] > 0) * dact
        my['DLINB1'] = np.einsum("bju->u", my['DLIN1'])

        for arr in ['DLIN1', 'DLINB1']:
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                return False

        return True


class BackwardSoftmaxKernel(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'HBJK')

        self.input_arrays = dict(
            ALPHA='HBJK',
            DROP_MASK='HBJK',
            DROP='HBJK'
        )
        self.output_arrays = dict(
            DBETA='HBJK'
        )
        self.vector_dims = 'HBJK'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = [
            'ALPHA', 'DBETA', 'DROP_MASK', 'DROP']


    def generate_source(self, H, B, J, K, vec_dim, ALPHA, DBETA, DROP_MASK, DROP, filename):
        source = """
            #include "block_bs.cuh"
        
            struct H { enum { value = %d }; };
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct K { enum { value = %d }; };

            using lALPHA = metal::list<%s>;
            using lDBETA = metal::list<%s>;
            using lDROP_MASK = metal::list<%s>;
            using lDROP = metal::list<%s>;

            extern "C" {
                void benchmark(
                    half* gALPHA, half* gDBETA, half* gDROP_MASK, half* gDROP) 
                {
                    BackwardSoftmax<half, H, B, J, K, %s,
                        lALPHA, lDBETA, lDROP_MASK, lDROP>::run(gALPHA, gDBETA, gDROP_MASK, gDROP, 0);
                }
            }
        """ % (H, B, J, K,
            ",".join(list(ALPHA)),
            ",".join(list(DBETA)),
            ",".join(list(DROP_MASK)),
            ",".join(list(DROP)),
            vec_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for vec_dim in self.vector_dims:
            for sizes in self.sizes:
                for layout in layouts:
                    # Hack to simplify cases.
                    if layout['DROP'] != layout['DROP_MASK']: continue
                    params = {**sizes, 'vec_dim': vec_dim, **layout}
                    yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        my = {}
        dalpha = ca['DROP'] * ca['DROP_MASK']
        my['DBETA'] = ca['ALPHA'] * (dalpha - np.sum(ca['ALPHA'] * dalpha, axis=-1, keepdims=True))

        for arr in ['DBETA']:
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                return False

        return True


class ExtendedBackwardScaleBias(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'BJN')

        self.input_arrays = dict(
            LN1="BJN",
            S1="N",
            DRESID2="BJN",
            DLIN1_LINW1="BJN",
            DLIN2="BJN"
        )
        self.output_arrays = dict(
            DLINB2="N",
            DS1="N",
            DB1="N",
            DLN1="BJN"
        )
        self.vector_dims = 'BJN'
        self.warp_dims = 'BJ'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = [
            'LN1', 'S1', 'DRESID2', 'DLIN1_LINW1', 'DLIN2', 'DLN1', 'DS1', 'DB1', 'DLINB2']


    def generate_source(self, B, J, N, vec_dim, warp_dim, 
        LN1, S1, DRESID2, DLIN1_LINW1, DLIN2, DLN1, DS1, DB1, DLINB2, filename):
        source = """
            #include "block_bsb_ebsb.cuh"
        
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct N { enum { value = %d }; };

            using lLN1 = metal::list<%s>;
            using lDLN1 = metal::list<%s>;
            using lDRESID2 = metal::list<%s>;
            using lDLIN1_LINW1 = metal::list<%s>;
            using lDLIN2 = metal::list<%s>;

            extern "C" {
                void benchmark(
                    half* gLN1, half* gS1, half* gDRESID2, half* gDLIN1_LINW1, half* gDLIN2,
                    half* gDLN1, half* gDS1, half* gDB1, half* gDLINB2) 
                {
                    ExtendedBackwardScaleBias<half,
                        B, J, N,
                        %s, %s,
                        lLN1, lDLN1, lDRESID2, lDLIN1_LINW1, lDLIN2
                    >::run(gLN1, gS1, gDRESID2, gDLIN1_LINW1, gDLIN2,
                        gDLN1, gDS1, gDB1, gDLINB2, 0);
                }
            }
        """ % (B, J, N,
            ",".join(list(LN1)), 
            ",".join(list(DLN1)),
            ",".join(list(DRESID2)),
            ",".join(list(DLIN1_LINW1)), 
            ",".join(list(DLIN2)),
            vec_dim,
            warp_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for vec_dim in self.vector_dims:
            for warp_dim in self.warp_dims:
                for sizes in self.sizes:
                    for layout in layouts:
                        params = {**sizes, 'vec_dim': vec_dim, 'warp_dim': warp_dim, **layout}
                        yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        my = {}
        my['DLINB2'] = np.einsum("BJN->N", ca['DLIN2'])
        myDSB1 = ca['DRESID2'] + ca['DLIN1_LINW1']
        my['DS1'] = np.einsum("BJN,BJN->N", myDSB1, ca['LN1'])
        my['DB1'] = np.einsum("BJN->N", myDSB1)
        my['DLN1'] = np.einsum("BJN,N->BJN", myDSB1, ca['S1'])

        for arr in self.output_arrays.keys():
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                return False

        return True


class BackwardEncoderInput(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'BJN')

        self.input_arrays = dict(
            DXATT="BJN",
            DRESID1="BJN"
        )
        self.output_arrays = dict(
            DX="BJN"
        )
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = ['DXATT', 'DRESID1', 'DX']


    def generate_source(self, B, J, N, dt, dv, DXATT, DRESID1, DX, filename):
        source = """
            #include "block_bei.cuh"
        
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct N { enum { value = %d }; };

            using lDXATT = metal::list<%s>;
            using lDRESID1 = metal::list<%s>;
            using lDX = metal::list<%s>;

            extern "C" {
                void benchmark(half* gDXATT, half* gDRESID1, half* gDX) 
                {
                    BackwardEncoderInput<half, B, J, N, %s, %s, lDXATT, lDRESID1, lDX>::run(gDXATT, gDRESID1, gDX, 0);
                }
            }
        """ % (B, J, N,
            ",".join(list(DXATT)), 
            ",".join(list(DRESID1)),
            ",".join(list(DX)),
            dv,
            dt)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for dv in 'BJN':
            for dt in 'BJN':
                for sizes in self.sizes:
                    for layout in layouts:
                        params = {**sizes, 'dt': dt, 'dv': dv, **layout}
                        yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        my = {}
        my['DX'] = ca['DXATT'] + ca['DRESID1']

        for arr in self.output_arrays.keys():
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                return False

        return True


class AttentionInputBiases(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'HPBJ')

        self.input_arrays = dict(
            WKK="HPBJ",
            WVV="HPBJ",
            WQQ="HPBJ",
            BK="HP",
            BV="HP",
            BQ="HP",
        )
        self.output_arrays = dict(
            KK="HPBJ",
            VV="HPBJ",
            QQ="HPBJ",
        )
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.short_all_arrays = {'WKK': self.input_arrays['WKK'],
                                 'BK': self.input_arrays['BK'],
                                 'KK': self.output_arrays['KK']}
        self.input_mapping = [*self.input_arrays.keys(), *self.output_arrays.keys()]


    def generate_source(self, H, P, B, J, dv, dt, WKK, WVV, WQQ, BK, BV, BQ, KK, VV, QQ, filename):
        source = """
            #include "block_aib.cuh"

            struct H { enum { value = %d }; };
            struct P { enum { value = %d }; };
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };

            using lWKK = metal::list<%s>;
            using lWVV = metal::list<%s>;
            using lWQQ = metal::list<%s>;
            using lBK = metal::list<%s>;
            using lBV = metal::list<%s>;
            using lBQ = metal::list<%s>;
            using lKK = metal::list<%s>;
            using lVV = metal::list<%s>;
            using lQQ = metal::list<%s>;

            extern "C" {
                void benchmark(half* gWKK, half* gWVV, half* gWQQ, half* gBK, half* gBV, half* gBQ, half* gKK, half* gVV, half* gQQ) 
                {
                    AttentionInputBiases<half,
                        H, P, B, J,
                        %s, %s,
                        lWKK, lWVV, lWQQ,
                        lBK, lBV, lBQ,
                        lKK, lVV, lQQ
                    >::run(gWKK, gWVV, gWQQ, gBK, gBV, gBQ, gKK, gVV, gQQ, 0);
                }
            }
        """ % (H, P, B, J,
            ",".join(list(WKK)), 
            ",".join(list(WVV)),
            ",".join(list(WQQ)),
            ",".join(list(BK)), 
            ",".join(list(BV)),
            ",".join(list(BQ)),
            ",".join(list(KK)), 
            ",".join(list(VV)),
            ",".join(list(QQ)),
            dv, dt)

        with open(filename, 'w') as f:
            f.write(source)

    # Hacks to reduce the number of layouts.
    def all_permutations(self):
        for e in self.short_all_arrays:
            yield itertools.permutations(self.short_all_arrays[e])

    def _generate_layouts_list(self):
        permutations = itertools.product(*self.all_permutations())
        for p in permutations:
            layout = {k: "".join(v) for k, v in zip(self.short_all_arrays.keys(), p)}
            # Add back in the other cases.
            layout['WVV'] = layout['WKK']
            layout['WQQ'] = layout['WKK']
            layout['BV'] = layout['BK']
            layout['BQ'] = layout['BK']
            layout['VV'] = layout['KK']
            layout['QQ'] = layout['KK']
            yield layout

    def get_options(self):
        layouts = self.generate_layouts_list()
        for dv in 'BJ':
            for dt in 'BJ':
                for sizes in self.sizes:
                    for layout in layouts:
                        params = {**sizes, 'dv': dv, 'dt': dt, **layout}
                        yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        my = {}
        my['KK'] = ca['WKK'] + ca['BK'][:,:,np.newaxis,np.newaxis]
        my['VV'] = ca['WVV'] + ca['BV'][:,:,np.newaxis,np.newaxis]
        my['QQ'] = ca['WQQ'] + ca['BQ'][:,:,np.newaxis,np.newaxis]

        for arr in self.output_arrays.keys():
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                return False

        return True


class BackwardAttentionInputBiases(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'PHBJ')

        self.input_arrays = dict(
            DKK="PHBJ", DVV="PHBJ", DQQ="PHBJ"
        )
        self.output_arrays = dict(
            DBK="PH", DBV="PH", DBQ="PH"
        )
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = [*self.input_arrays.keys(), *self.output_arrays.keys()]


    def generate_source(self, P, H, B, J, dv, dw, DKK, DVV, DQQ, DBK, DBV, DBQ, filename):
        source = """
            #include "block_baib.cuh"
        
            struct P { enum { value = %d }; };
            struct H { enum { value = %d }; };
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };

            using lDKK = metal::list<%s>;
            using lDVV = metal::list<%s>;
            using lDQQ = metal::list<%s>;
            using lDBK = metal::list<%s>;
            using lDBV = metal::list<%s>;
            using lDBQ = metal::list<%s>;

            extern "C" {
                void benchmark(half* gDKK, half* gDVV, half* gDQQ, half* gDBK, half* gDBV, half* gDBQ) 
                {
                    BackwardAttentionInputBiases<half, 
                        P, H, B, J,
                        %s, %s,
                        lDKK, lDVV, lDQQ, 
                        lDBK, lDBV, lDBQ>::run(
                            gDKK, gDVV, gDQQ, gDBK, gDBV, gDBQ, 0);
                }
            }
        """ % (P, H, B, J,
            ",".join(list(DKK)), 
            ",".join(list(DVV)),
            ",".join(list(DQQ)),
            ",".join(list(DBK)), 
            ",".join(list(DBV)),
            ",".join(list(DBQ)),
            dv, dw)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for dv in 'PHBJ':
            for dw in 'BJ':
                for sizes in self.sizes:
                    for layout in layouts:
                        params = {**sizes, 'dv': dv, 'dw': dw, **layout}
                        yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        my = {}
        my['DBK'] = np.sum(ca['DKK'], axis=(2,3))
        my['DBV'] = np.sum(ca['DVV'], axis=(2,3))
        my['DBQ'] = np.sum(ca['DQQ'], axis=(2,3))

        for arr in self.output_arrays.keys():
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                print(my[arr])
                print(500 * '-')
                print(ca[arr])
                return False

        return True


class BackwardAttentionOutputBias(Kernel):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = filter_sizes(sizes, 'BJN')

        self.input_arrays = dict(
            DATT="BJN",
        )
        self.output_arrays = dict(
            DBO="N"
        )
        self.all_arrays = {**self.input_arrays, **self.output_arrays}
        self.input_mapping = [*self.input_arrays.keys(), *self.output_arrays.keys()]


    def generate_source(self, B, J, N, dv, dw, DATT, DBO, filename):
        source = """
            #include "block_baob.cuh"
        
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct N { enum { value = %d }; };

            using lDATT = metal::list<%s>;

            extern "C" {
                void benchmark(half* gDATT, half* gDBO) 
                {
                    BackwardAttnOutBias<half,
                        N, B, J,
                        %s, %s,
                        lDATT>::run(gDATT, gDBO, 0);
                }
            }
        """ % (B, J, N,
            ",".join(list(DATT)),
            dv, dw)

        with open(filename, 'w') as f:
            f.write(source)

    def get_options(self):
        layouts = self.generate_layouts_list()
        for dv in 'BJN':
            for dw in 'BJ':
                for sizes in self.sizes:
                    for layout in layouts:
                        params = {**sizes, 'dv': dv, 'dw': dw, **layout}
                        yield params, sizes, layout

    def compare_with_reference(self, arrays, layout):
        ca = {k: np.einsum(layout[k] + '->' + v, arrays[k].astype(np.float32)) for k, v in self.all_arrays.items()}

        my = {}
        my['DBO'] = np.sum(ca['DATT'], axis=(0, 1))

        for arr in self.output_arrays.keys():
            if not self.allclose(my[arr], ca[arr]):
                print("%s is wrong" % arr)
                print(my[arr])
                print(500 * '-')
                print(ca[arr])
                return False

        return True


def resume_from_remaining(output_basename, kernel):
    output_dir = os.path.dirname(output_basename)
    kernel_name = os.path.basename(output_basename)
    remaining_file = output_dir + '/' + kernel_name + '-remaining'
    if os.path.exists(remaining_file):
        with open(remaining_file, 'rb') as f:
            remaining = pickle.load(f)
    else:
        return None
    # Hack to get all the options.
    old_rank = get_world_rank()
    old_size = get_world_size()
    os.environ['MV2_COMM_WORLD_RANK'] = '0'
    os.environ['MV2_COMM_WORLD_SIZE'] = '1'
    opts = list(kernel.get_options())
    os.environ['MV2_COMM_WORLD_RANK'] = str(old_rank)
    os.environ['MV2_COMM_WORLD_SIZE'] = str(old_size)
    # Partition the remaining work.
    opts_per_rank = [len(remaining) // get_world_size()] * get_world_size()
    remainder = len(remaining) % get_world_size()
    for i in range(remainder): opts_per_rank[i] += 1
    remaining_prefix_sum, remaining_sum = [0], 0
    for i in range(get_world_size()):
        remaining_sum += opts_per_rank[i]
        remaining_prefix_sum.append(remaining_sum)
    start_task = remaining_prefix_sum[get_world_rank()]
    end_task = remaining_prefix_sum[get_world_rank() + 1]
    print(f'{get_world_rank()}: Resuming from remaining file, handling {start_task}:{end_task} of {len(remaining)}')
    opts_to_run = []
    for task in remaining[start_task:end_task]:
        opts_to_run.append(opts[task])
    return opts_to_run


def run_benchmark(kernel, output_basename, time_bm=False, compile_ahead=6,
                  resume=False):
    rank = get_world_rank()

    # Set up output file.
    if output_to_file:
        output_file = open(f'{output_basename}-bm{rank}.csv', 'a+' if resume else 'w')
    else:
        output_file = sys.stdout

    if work_from_tmp:
        switch_to_tmp()

    kernel.setup_gpu_helper()

    # Select the right GPU.
    os.putenv('CUDA_VISIBLE_DEVICES', str(get_local_rank()))

    # For compiling ahead.
    compiling_subprocesses = {}

    options = list(kernel.get_options())
    print(f'{get_world_rank()}: {len(options)} total configurations')

    # Determine how many cases we already did.
    # Note: This only works if resumed with the same number of processes.
    if resume:
        new_opts = resume_from_remaining(output_basename, kernel)
        if new_opts is not None:
            options = new_opts
        else:
            if not output_to_file:
                raise RuntimeError('Cannot resume from stdout')
            output_file.seek(0)
            done_already = output_file.readlines()  # Puts us back at the end.
            num_done = len(done_already)
            print(f'{get_world_rank()}: Resuming from {num_done}')
            options = options[num_done:]
    sys.stdout.flush()

    # Layouts are split internally.
    for i in range(len(options)):
        params, sizes, layout = options[i]

        if time_bm: all_start_time = time.perf_counter()

        if time_bm: start_time = time.perf_counter()
        # Submit this and future kernels to be compiled.
        # Not the most efficient way, but :shrug:.
        for future_opts in options[i:i+compile_ahead]:
            future_params = future_opts[0]
            kbn = get_kernel_name(future_params)
            if kbn not in compiling_subprocesses:
                subproc = kernel.build_kernel_async(kbn, future_params)
                compiling_subprocesses[kbn] = subproc

        kernel_basename = get_kernel_name(params)
        # Wait for asynchronous compilation to finish.
        compiling_subprocesses[kernel_basename].wait()
        # Remove future.
        del compiling_subprocesses[kernel_basename]
        lib = kernel.get_lib(kernel_basename)
        if time_bm: print(f'Build: {time.perf_counter() - start_time:.4f}')

        if time_bm: start_time = time.perf_counter()
        arrays = kernel.generate_arrays(layout, sizes)
        gpu_arrays = kernel.allocate_gpu_arrays(sizes)
        
        kernel.copy_input_arrays(arrays, gpu_arrays)

        kernel_arrays = []
        
        for name in kernel.input_mapping:
            kernel_arrays.append(gpu_arrays[name])

        runtime = kernel.run_measurement(lib, kernel_arrays)
        if time_bm: print(f'BM: {time.perf_counter() - start_time:.4f}')

        output_str = ""
        for k in params:
            v = params[k]
            output_str += "{k} = {v} ".format(k=k, v=v)
        output_str += "time = %lg s" % runtime

        kernel.copy_output_arrays(arrays, gpu_arrays)

        kernel.free_gpu_arrays(kernel_arrays)

        if time_bm: start_time = time.perf_counter()
        success = kernel.compare_with_reference(arrays, layout)
        if time_bm: print(f'Check: {time.perf_counter() - start_time:.4f}')

        if success:
            output_file.write(output_str + '\n')
        else:
            output_file.write(output_str + ' ERROR\n')

        if i % 1000 == 0:
            output_file.flush()

        unload_lib(lib)
        kernel.clear_kernel(kernel_basename)

        if time_bm: print(f'All: {time.perf_counter() - all_start_time:.4f}')

    if len(compiling_subprocesses):
        # Should never happen.
        print('Some subprocesses still unprocessed')

    if output_to_file:
        output_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-k', '--kernel', required=True,
        choices=['softmax', 'bad', 'bdrln', 'blnrd', 'bsb', 'bdrlb', 'bs',
                 'ebsb', 'bei', 'aib', 'baib', 'baob'],
        help='Kernel to benchmark')
    parser.add_argument(
        '--time', default=False, action='store_true',
        help='Time benchmarking process')
    parser.add_argument(
        '--size', default=None, type=str,
        help='Specify size')
    parser.add_argument(
        '--out-dir', default=None, type=str,
        help='Output directory')
    parser.add_argument(
        '--resume', default=False, action='store_true',
        help='Resume a prior run')
    args = parser.parse_args()

    if args.size:
        size_strs = args.size.split(',')
        size = dict()
        for s in size_strs:
            s = s.split('=')
            size[s[0]] = int(s[1])
        sizes = [size]
    else:
        # Default: BERT-large, seq len 512, batch size 8.
        sizes = [dict(H=16, B=8, J=512, K=512, U=4*16*64, N=16*64, P=64)]
    # For reference, other sizes:
    # BERT 512
    #dict(B=1, J=512, N=16 * 64),
    #dict(B=2, J=512, N=16 * 64),
    #dict(B=4, J=512, N=16 * 64),
    #dict(B=8, J=512, N=16 * 64),
    # BERT 128
    #dict(B=1, J=128, N=16 * 64),
    #dict(B=2, J=128, N=16 * 64),
    #dict(B=4, J=128, N=16 * 64),
    #dict(B=8, J=128, N=16 * 64),
    # GPT-2 med
    #dict(B=1, J=1024, N=16 * 64),
    #dict(B=2, J=1024, N=16 * 64),
    #dict(B=4, J=1024, N=16 * 64),
    #dict(B=8, J=1024, N=16 * 64),
    # GPT-2 xl
    #dict(B=1, J=1024, N=25 * 64),
    #dict(B=2, J=1024, N=25 * 64),
    #dict(B=4, J=1024, N=25 * 64),
    #dict(B=8, J=1024, N=25 * 64),
    # Megatron
    #dict(B=1, J=1024, N=32 * 96)

    if args.kernel == 'softmax':
        k = SoftmaxKernel(sizes)
    elif args.kernel == 'bad':
        k = BiasActivationDropoutKernel(sizes)
    elif args.kernel == 'bdrln':
        k = BiasDropoutResidualLinearNorm(sizes)
    elif args.kernel == 'blnrd':
        k = BackwardLayerNormResidualDropout(sizes)
    elif args.kernel == 'bsb':
        k = BackwardScaleBias(sizes)
    elif args.kernel == 'bdrlb':
        k = BackwardDropoutReluLinearBias(sizes)
    elif args.kernel == 'bs':
        k = BackwardSoftmaxKernel(sizes)
    elif args.kernel == 'ebsb':
        k = ExtendedBackwardScaleBias(sizes)
    elif args.kernel == 'bei':
        k = BackwardEncoderInput(sizes)
    elif args.kernel == 'aib':
        k = AttentionInputBiases(sizes)
    elif args.kernel == 'baib':
        k = BackwardAttentionInputBiases(sizes)
    elif args.kernel == 'baob':
        k = BackwardAttentionOutputBias(sizes)
    else:
        raise Exception("Wrong kernel name")

    output_basename = args.kernel
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        output_basename = args.out_dir + '/' + output_basename

    run_benchmark(k, output_basename, time_bm=args.time, resume=args.resume)
