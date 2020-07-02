#/usr/bin/env python

import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer

from substation.transformer import layer_norm_backward_weights

from generator import generate_bsb


def ref_bsb(inp, scale, dout):
    dinp = dout * scale
    dinp2 = np.einsum("bji,i->bji", dout, scale)
    assert(np.allclose(dinp, dinp2, atol=1e-2, rtol=1e-1))
    dscale, dbias = layer_norm_backward_weights(dout, inp, True, True)
    return dinp, dscale, dbias

def perf_bsb():
    
    #dims = { 'B': 2, 'J': 32, 'N': 8 }
    dims = { 'B': 8, 'J': 512, 'N': 16 * 64 }
    reduce_dim = 'B'
    warp_reduce_dim = 'J'
    non_reduce_dim = 'N'
    
    base_layout = "".join(dims.keys())
    
    generate_bsb(dims, reduce_dim, warp_reduce_dim, 'perf_bsb.so', reps=100)
    
    lib = ctypes.cdll.LoadLibrary('./perf_bsb.so')
    
    inp = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    scale = np.ascontiguousarray(np.random.rand(dims[non_reduce_dim]), dtype='float16')
    dout = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    dscale = np.ascontiguousarray(np.random.rand(dims[non_reduce_dim]), dtype='float16')
    dbias = np.ascontiguousarray(np.random.rand(dims[non_reduce_dim]), dtype='float16')
    
    dropout_probability = 0
    ref_din, ref_dscale, ref_dbias = ref_bsb(inp, scale, dout)
    
    for dims_permutation_in in itertools.permutations(dims):
        for dims_permutation_din in itertools.permutations(dims):
            for dims_permutation_dout in itertools.permutations(dims):
                in_label = "".join(dims_permutation_in)
                din_label = "".join(dims_permutation_din)
                dout_label = "".join(dims_permutation_dout)
                
                func_name = 'temp_%s_%s_%s' % (in_label, din_label, dout_label)
                
                bsb = getattr(lib, func_name)
                
                din_array_shape = list(map(lambda x: dims[x], dims_permutation_din))
                din = np.ascontiguousarray(np.zeros(din_array_shape), dtype='float16')
                
                temp_inp = np.ascontiguousarray(np.einsum(base_layout + "->" + in_label, inp), dtype='float16')
                temp_dout = np.ascontiguousarray(np.einsum(base_layout + "->" + dout_label, dout), dtype='float16')
                
                bsb.restype = ctypes.c_double
                
                time = bsb(ctypes.c_void_p(temp_inp.ctypes.data),
                     ctypes.c_void_p(scale.ctypes.data),
                     ctypes.c_void_p(temp_dout.ctypes.data),
                     ctypes.c_void_p(din.ctypes.data),
                     ctypes.c_void_p(dscale.ctypes.data),
                     ctypes.c_void_p(dbias.ctypes.data))
                
                din = np.einsum(din_label + "->" + base_layout, din)
                
                layouts = "IN %s DIN %s DOUT %s" % (in_label, din_label, dout_label)
                
                print("bsb %s %f us" % (layouts, time))
                
                if not np.allclose(dbias, ref_dbias, atol=1e-2, rtol=1e-1):
                    print("DBIAS")
                    raise Exception(layouts)
                elif not np.allclose(dscale, ref_dscale, atol=1e-2, rtol=1e-1):
                    print("DSCALE")
                    raise Exception(layouts)
                elif not np.allclose(din, ref_din, atol=1e-2, rtol=1e-1):
                    print(din)
                    print('------------------------')
                    print(ref_din)
                    print("DIN")
                    raise Exception(layouts)
                else:
                    pass
                    #print("OK", layouts)


if __name__ == '__main__':
    perf_bsb()
