#/usr/bin/env python

import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer

from substation.attention import softmax as ref_softmax

from generator import generate_softmax


def performance_softmax():

    dims = {'H': 16, 'B': 8, 'J': 512, 'K': 512}
    reduce_dim = 'K'
    
    reps = 100
    generate_softmax(dims, reduce_dim, 'softmax_perf.so', reps)
    
    lib = ctypes.cdll.LoadLibrary('./softmax_perf.so')
        
    A = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    ref_B = ref_softmax(A, dim=3)
    
    for dims_permutation_in in itertools.permutations(dims):
        for dims_permutation_out in itertools.permutations(dims):
            in_label = "".join(dims_permutation_in)
            out_label = "".join(dims_permutation_out)
            
            func_name = 'temp_%s_%s' % (in_label, out_label)
            
            softmax = getattr(lib, func_name)
            
            out_array_shape = list(map(lambda x: dims[x], dims_permutation_out))
            B = np.ascontiguousarray(np.zeros(out_array_shape), dtype='float16')
            
            temp_A = np.ascontiguousarray(np.einsum("HBJK->" + in_label, A), dtype='float16')
            
            A_raw = ctypes.c_void_p(temp_A.ctypes.data)
            B_raw = ctypes.c_void_p(B.ctypes.data)
            
            softmax.restype = ctypes.c_double
            
            time = softmax(A_raw, B_raw)
            print("softmax %s -> %s %f us" % (in_label, out_label, time))
            
            B = np.einsum(out_label + "->HBJK", B)
            
            if not np.allclose(B, ref_B, atol=1e-3, rtol=1e-2):
                raise Exception(in_label + " -> " + out_label)

if __name__ == '__main__':
    performance_softmax()

