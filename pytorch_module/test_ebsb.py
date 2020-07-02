#/usr/bin/env python

import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer

from substation.transformer import layer_norm_backward_weights

from generator import generate_ebsb


def ref_ebsb(inp, scale, dout1, dout2, dlinear):
    dlinear_bias = dlinear.sum(axis=(0, 1))
    dout12 = dout1 + dout2
    dinp = dout12 * scale
    dscale, dbias = layer_norm_backward_weights(dout12, inp, True, True)
    return dinp, dscale, dbias, dout12, dlinear_bias

def unload_lib(lib):
    cdll = ctypes.CDLL("libdl.so")
    cdll.dlclose.restype = ctypes.c_int
    cdll.dlclose.argtypes = [ctypes.c_void_p]
    res = cdll.dlclose(lib._handle)
    if res != 0:
        raise Exception("dlclose failed")

def test_ebsb():
    
    dims = { 'B': 2, 'J': 32, 'N': 8 }
    #dims = { 'B': 8, 'J': 512, 'N': 16 * 64 }
    reduce_dim = 'B'
    warp_reduce_dim = 'J'
    non_reduce_dim = 'N'
    
    base_layout = "".join(dims.keys())    
    
    #half* IN, half* SCALE, half* DOUT1, half* DOUT2, half* DLINEAR,
    #half* DIN, half* DSCALE, half* DBIAS, half* DOUT12, half* DLINEAR_BIAS,
    
    #typename reduceDim, typename warpReduceDim, typename nonReduceDim,
    #typename in_layout,
    #typename din_layout,
    #typename dout1_layout,
    #typename dout2_layout,
    #typename dout12_layout,
    #typename dlinear_layout
    
    inp = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    scale = np.ascontiguousarray(np.random.rand(dims[non_reduce_dim]), dtype='float16')
    dout1 = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    dout2 = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    dlinear = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    #din = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    dscale = np.ascontiguousarray(np.random.rand(dims[non_reduce_dim]), dtype='float16')
    dbias = np.ascontiguousarray(np.random.rand(dims[non_reduce_dim]), dtype='float16')
    #dout12 = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    dlinear_bias = np.ascontiguousarray(np.random.rand(dims[non_reduce_dim]), dtype='float16')
    
    ref_din, ref_dscale, ref_dbias, ref_dout12, ref_dlinear_bias = ref_ebsb(inp, scale, dout1, dout2, dlinear)
    
    for dims_permutation_in in itertools.permutations(dims):
        for dims_permutation_din in itertools.permutations(dims):
            for dims_permutation_dout1 in itertools.permutations(dims):
                for dims_permutation_dout2 in itertools.permutations(dims):
                    for dims_permutation_dout12 in itertools.permutations(dims):
                        for dims_permutation_dlinear in itertools.permutations(dims):
                            
                            in_label = "".join(dims_permutation_in)
                            din_label = "".join(dims_permutation_din)
                            dout1_label = "".join(dims_permutation_dout1)
                            dout2_label = "".join(dims_permutation_dout2)
                            dout12_label = "".join(dims_permutation_dout12)
                            dlinear_label = "".join(dims_permutation_dlinear)
                            
                            func_name = 'temp_%s_%s_%s_%s_%s_%s' % (in_label, din_label, dout1_label, dout2_label, dout12_label, dlinear_label)
                            
                            #half* IN, half* SCALE, half* DOUT1, half* DOUT2, half* DLINEAR,
                            #half* DIN, half* DSCALE, half* DBIAS, half* DOUT12, half* DLINEAR_BIAS,
                            
                            din_array_shape = list(map(lambda x: dims[x], dims_permutation_din))
                            din = np.ascontiguousarray(np.zeros(din_array_shape), dtype='float16')
                            
                            dout12_array_shape = list(map(lambda x: dims[x], dims_permutation_dout12))
                            dout12 = np.ascontiguousarray(np.zeros(dout12_array_shape), dtype='float16')
                            
                            temp_in = np.ascontiguousarray(np.einsum(base_layout + "->" + in_label, inp), dtype='float16')
                            temp_dout1 = np.ascontiguousarray(np.einsum(base_layout + "->" + dout1_label, dout1), dtype='float16')
                            temp_dout2 = np.ascontiguousarray(np.einsum(base_layout + "->" + dout2_label, dout2), dtype='float16')
                            temp_dlinear = np.ascontiguousarray(np.einsum(base_layout + "->" + dlinear_label, dlinear), dtype='float16')
                                                        
                            generate_ebsb(dims, reduce_dim, warp_reduce_dim, non_reduce_dim,
                                          dims_permutation_in,
                                          dims_permutation_din,
                                          dims_permutation_dout1,
                                          dims_permutation_dout2,
                                          dims_permutation_dout12,
                                          dims_permutation_dlinear,
                                          'ebsb_test.so')
                            lib = ctypes.cdll.LoadLibrary('./ebsb_test.so')
                            ebsb = getattr(lib, func_name)
                            ebsb.argtypes = [ctypes.c_void_p] * 10
                            ebsb(temp_in.ctypes.data,
                                 scale.ctypes.data,
                                 temp_dout1.ctypes.data,
                                 temp_dout2.ctypes.data,
                                 temp_dlinear.ctypes.data,
                                 din.ctypes.data,
                                 dscale.ctypes.data,
                                 dbias.ctypes.data,
                                 dout12.ctypes.data,
                                 dlinear_bias.ctypes.data)
                            unload_lib(lib)
                            
                            din = np.einsum(din_label + "->" + base_layout, din)
                            dout12 = np.einsum(dout12_label + "->" + base_layout, dout12)
                            
                            layouts = "IN %s DIN %s DOUT1 %s DOUT2 %s DOUT12 %s DLINEAR %s" % (in_label, din_label, dout1_label, dout2_label, dout12_label, dlinear_label)
                            
                            #dinp, dscale, dbias, dout12, dlinear_bias
                            
                            if not np.allclose(din, ref_din, atol=1e-2, rtol=1e-1):
                                print("DINP")
                                raise Exception(layouts)
                            elif not np.allclose(dscale, ref_dscale, atol=1e-2, rtol=1e-1):
                                print("DSCALE")
                                raise Exception(layouts)
                            elif not np.allclose(dbias, ref_dbias, atol=1e-2, rtol=1e-1):
                                print("DBIAS")
                                raise Exception(layouts)
                            elif not np.allclose(dout12, ref_dout12, atol=1e-2, rtol=1e-1):
                                print("DOUT12")
                                raise Exception(layouts)
                            elif not np.allclose(dlinear_bias, ref_dlinear_bias, atol=1e-2, rtol=1e-1):
                                print("DLIENAR_BIAS")
                                raise Exception(layouts)
                            else:
                                print("OK", layouts)


if __name__ == '__main__':
    test_ebsb()
    print("All tests are passed")
