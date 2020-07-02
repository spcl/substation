import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer
from copy import deepcopy
import itertools
import socket
import glob
import shutil

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


def ref_softmax(x, axis):
    exps = np.exp(x - x.max(axis=axis, keepdims=True))
    return exps / exps.sum(axis=axis, keepdims=True)


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
    # Would be better to use some synchronization, but this avoids adding
    # dependencies
    gpu_helper_basename = f'gpu_helper-{get_world_rank()}'
    if not os.path.exists(gpu_helper_basename + '.so'):
        gpu_mem_source = """
            #include <cuda.h>
            #include <stdio.h>
            #include <math.h>
        
            #define CHECK(expr) do {\
                auto err = (expr);\
                if (err != 0) {\
                    printf("ERROR %s %s:%d\\n", #expr, __FILE__, __LINE__); \
                    abort(); \
                }\
            } while(0)

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
                            printf("%zu: %f != %f\\n", i, a[i], b[i]);
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

class SoftmaxKernel:
    def __init__(self):
        self.sizes = [
            dict(H=16, B=8, J=512, K=512)
        ]
        self.input_arrays = dict(
            IN='HBJK'
        )
        self.output_arrays = dict(
            OUT='HBJK'
        )
        self.vector_dims = 'HBJK'
        self.all_arrays = {**self.input_arrays, **self.output_arrays}

    def setup_gpu_helper(self):
        self.gpu_helper = get_gpu_helper()

    def all_permutations(self):
        for e in self.all_arrays:
            yield itertools.permutations(self.all_arrays[e])

    def generate_layouts_list(self):

        permutations = itertools.product(*self.all_permutations())
        
        for p in permutations:
            yield {k: "".join(v) for k, v in zip(self.all_arrays.keys(), p)}

    def build(self, source, output):
        command = f"""
            nvcc -shared -O3
            {nvcc_gencode}
            -Xcompiler -fPIC
            {source} -o {output}
        """.format(source=source, output=output).split()
        subprocess.run(command)

    def load(self, lib):
        lib = ctypes.CDLL("./" + lib)
        lib.benchmark.argtypes = [ctypes.c_void_p, ctypes.c_void_p] 
        return lib


    def generate_source(self, H, B, J, K, vec_dim, layout_in, layout_out, filename):
        source = """
            #include "blocks.cuh"
        
            struct H { enum { value = %d }; };
            struct B { enum { value = %d }; };
            struct J { enum { value = %d }; };
            struct K { enum { value = %d }; };
            using lIN = metal::list<%s>;
            using lOUT = metal::list<%s>;

            extern "C" {
                void benchmark(half* in, half* out) {
                    Softmax<half, K, %s, lIN, lOUT>::run(in, out, 0);
                }
            }
        """ % (H, B, J, K, ",".join(list(layout_in)), ",".join(list(layout_out)), vec_dim)

        with open(filename, 'w') as f:
            f.write(source)

    def generate_arrays(self, layout, sizes):
        arrays = {}

        for k in self.input_arrays:
            size = [sizes[l] for l in layout[k]]
            arrays[k] = np.random.rand(*size).astype(np.float16)

        for k in self.output_arrays:
            size = [sizes[l] for l in layout[k]]
            arrays[k] = np.zeros(size, dtype=np.float16)

        return arrays

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

    def run_measurement(self, lib, gpu_arrays):

        warmup = 5
        repetitions = 100

        for _ in range(warmup):
            lib.benchmark(*gpu_arrays.values())
        self.gpu_helper.device_synchronize()
        
        start = timer()
        for _ in range(repetitions):
            lib.benchmark(*gpu_arrays.values())
        self.gpu_helper.device_synchronize()
        end = timer()
        time = (end - start) / repetitions
        return time

    def copy_output_arrays(self, arrays, gpu_arrays):
        for k in self.output_arrays:
            array = arrays[k]
            gpu_array = gpu_arrays[k]
            self.gpu_helper.gpu_to_host(array.ctypes.data, gpu_array, array.nbytes)

    def free_gpu_arrays(self, arrays):
        for a in arrays:
            v = arrays[a]
            self.gpu_helper.gpu_free(v)

    def compare_with_reference(self, arrays, layout):
        ref_A = np.einsum(layout['IN'] + '->' + self.all_arrays['IN'], arrays['IN'].astype(np.float32))
        ref_sm = ref_softmax(ref_A, axis=-1)
        custom_B = np.einsum(layout['OUT'] + '->' + self.all_arrays['OUT'], arrays['OUT'].astype(np.float32))
        ref_sm = np.ascontiguousarray(ref_sm)
        custom_B = np.ascontiguousarray(custom_B)
        return self.gpu_helper.fast_allclose(
            custom_B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ref_sm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ref_sm.size, 1e-4, 1e-4)
            

def run_bechmark(kernel):
    rank = get_world_rank()

    # Open before we change directories.
    if output_to_file:
        output_file = open(f'output-{rank}.csv', 'w')
    else:
        output_file = sys.stdout

    if work_from_tmp:
        switch_to_tmp()

    kernel.setup_gpu_helper()

    # Select the right GPU.
    os.putenv('CUDA_VISIBLE_DEVICES', str(get_local_rank()))

    layouts = list(kernel.generate_layouts_list())

    # Split layouts among workers.
    layouts_per_rank = [len(layouts) // get_world_size()] * get_world_size()
    remainder = len(layouts) % get_world_size()
    for i in range(remainder): layouts_per_rank[i] += 1
    layouts_prefix_sum, layouts_sum = [0], 0
    for i in range(get_world_size()):
        layouts_sum += layouts_per_rank[i]
        layouts_prefix_sum.append(layouts_sum)
    start_layout = layouts_prefix_sum[get_world_rank()]
    end_layout = layouts_prefix_sum[get_world_rank() + 1]
    print(f'{rank}: Handling layouts {start_layout}:{end_layout} of {len(layouts)}')

    for vec_dim in kernel.vector_dims:
        for sizes in kernel.sizes:
            for layout in layouts[start_layout:end_layout]:
                params = [*sizes.values(), vec_dim, *layout.values()]
                params_str = ' '.join([str(x) for x in params])

                kernel.generate_source(*params, f'temp-{rank}.cu')
                kernel.build(f'temp-{rank}.cu', f'temp-{rank}.so')
                lib = kernel.load(f'temp-{rank}.so')

                arrays = kernel.generate_arrays(layout, sizes)
                gpu_arrays = kernel.allocate_gpu_arrays(sizes)
                
                kernel.copy_input_arrays(arrays, gpu_arrays)

                time = kernel.run_measurement(lib, gpu_arrays)
                output_file.write(params_str + ' ' + str(time) + '\n')

                kernel.copy_output_arrays(arrays, gpu_arrays)

                kernel.free_gpu_arrays(gpu_arrays)

                if check_correctness:
                    success = kernel.compare_with_reference(arrays, layout)
                else:
                    success = True

                if not success:
                    output_file.write(params_str + ' ERROR\n')

                unload_lib(lib)
            # Flush occasionally.
            output_file.flush()

    if output_to_file:
        output_file.close()

if __name__ == '__main__':
    softmaxKernel = SoftmaxKernel()
    run_bechmark(softmaxKernel)
