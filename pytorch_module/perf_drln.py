#/usr/bin/env python

import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer

from substation.transformer import dropout, layer_norm

from generator import generate_drln

def ref_drln(inp, resid, scale, bias, probability):
    tmp1, _ = dropout(inp, probability)
    tmp2 = tmp1 + resid
    res, _, _, _ = layer_norm(tmp2, scale, bias)
    return res

def performance_drln():

    dims = { 'B': 8, 'J': 512, 'N': 16 * 64 }
    reduce_dim = 'N'
    
    reps = 100
    
    base_layout = "".join(dims.keys())
    
    generate_drln(dims, reduce_dim, 'drln_perf.so', reps)
    
    lib = ctypes.cdll.LoadLibrary('./drln_perf.so')
        
    inp = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    resid = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    scale = np.ascontiguousarray(np.random.rand(dims[reduce_dim]), dtype='float16')
    bias = np.ascontiguousarray(np.random.rand(dims[reduce_dim]), dtype='float16')
    dropout_probability = 0
    ref_out = ref_drln(inp, resid, scale, bias, dropout_probability)
    
    for dims_permutation_out in itertools.permutations(dims):
        for dims_permutation_in in itertools.permutations(dims):
            for dims_permutation_resid in itertools.permutations(dims):
                out_label = "".join(dims_permutation_out)
                in_label = "".join(dims_permutation_in)
                resid_label = "".join(dims_permutation_resid)
                
                func_name = 'temp_%s_%s_%s' % (out_label, in_label, resid_label)
                
                drln = getattr(lib, func_name)
            
                out_array_shape = list(map(lambda x: dims[x], dims_permutation_out))
                out = np.ascontiguousarray(np.zeros(out_array_shape), dtype='float16')
                
                temp_inp = np.ascontiguousarray(np.einsum(base_layout+ "->" + in_label, inp), dtype='float16')
                temp_resid = np.ascontiguousarray(np.einsum(base_layout + "->" + resid_label, resid), dtype='float16')
                
                drln.restype = ctypes.c_double
                
                time = drln(ctypes.c_void_p(out.ctypes.data),
                     ctypes.c_void_p(temp_inp.ctypes.data),
                     ctypes.c_void_p(temp_resid.ctypes.data),
                     ctypes.c_void_p(scale.ctypes.data),
                     ctypes.c_void_p(bias.ctypes.data))
                
                layouts = "OUT %s IN %s RESID %s" % (out_label, in_label, resid_label)
                
                print("drln %s %f us" % (layouts, time))
                
                out = np.einsum(out_label + "->" + base_layout, out)
                
                if not np.allclose(out, ref_out, atol=1e-2, rtol=1e-1):
                    raise Exception(layouts)

if __name__ == '__main__':
    performance_drln()

