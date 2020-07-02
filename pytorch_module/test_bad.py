#/usr/bin/env python

import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer

from substation.transformer import linear, relu, dropout

from generator import generate_bad

def ref_bad(inp, bias, probability):
    tmp1 = inp + bias
    tmp2 = relu(tmp1)
    res, _ = dropout(tmp2, probability)
    return res

def test_bad():
    
    dims = { 'B': 8, 'S': 512, 'U': 4 * 16 * 64 }
    reduce_dim = 'U'
    
    base_layout = "".join(dims.keys())
    
    generate_bad(dims, reduce_dim, 'bad_test.so')
    
    lib = ctypes.cdll.LoadLibrary('./bad_test.so')
    
    inp = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    bias = np.ascontiguousarray(np.random.rand(dims[reduce_dim]), dtype='float16')
    dropout_probability = 0
    ref_out = ref_bad(inp, bias, dropout_probability)
    
    for dims_permutation_out in itertools.permutations(dims):
        for dims_permutation_in in itertools.permutations(dims):
                out_label = "".join(dims_permutation_out)
                in_label = "".join(dims_permutation_in)
                
                func_name = 'temp_%s_%s' % (in_label, out_label)
                
                bad = getattr(lib, func_name)
                
                out_array_shape = list(map(lambda x: dims[x], dims_permutation_out))
                out = np.ascontiguousarray(np.zeros(out_array_shape), dtype='float16')
                
                temp_inp = np.ascontiguousarray(np.einsum(base_layout + "->" + in_label, inp), dtype='float16')
                
                bad(ctypes.c_void_p(temp_inp.ctypes.data),
                     ctypes.c_void_p(out.ctypes.data),
                     ctypes.c_void_p(bias.ctypes.data))
                
                out = np.einsum(out_label + "->" + base_layout, out)
                
                layouts = "IN %s OUT %s" % (in_label, out_label)
                
                if not np.allclose(out, ref_out, atol=1e-2, rtol=1e-1):
                    print(out)
                    print('---------------')
                    print(ref_out)
                    raise Exception(layouts)
                else:
                    print("OK", layouts)


if __name__ == '__main__':
    test_bad()
    print("All tests are passed")
