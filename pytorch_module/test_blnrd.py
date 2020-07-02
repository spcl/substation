#/usr/bin/env python

import subprocess
import ctypes
import numpy as np
import itertools
from functools import reduce
import os
from timeit import default_timer as timer

from substation.transformer import layer_norm_backward_data, dropout_backward_data

from generator import generate_blnrd

def ref_blnrd(dout, std, diff, drop_mask):
    # only diff = x - mean is used in layer_norm_backward_data, x and mean themselves are not used
    x = diff
    mean = 0
    d_ln_in = layer_norm_backward_data(x, dout, mean, np.repeat(std[:, :, np.newaxis], x.shape[-1], axis=2))
    nonzero_magic_value = 0.12345
    d_drop_in = dropout_backward_data(d_ln_in, nonzero_magic_value, drop_mask)
    return d_ln_in, d_drop_in

def test_blnrd():
    dims = { 'B': 2, 'J': 8, 'N': 32 }
    dims = { 'B': 8, 'J': 512, 'N': 16 * 64 }
    
    nothing_dim = 'B'
    vec_dim = 'J'
    reduce_dim = 'N'

    base_layout = "".join(dims.keys())
    base_std_layout = 'BJ'
    
    generate_blnrd(dims, nothing_dim, vec_dim, reduce_dim, 'blnrd_test.so')
    
    lib = ctypes.cdll.LoadLibrary('./blnrd_test.so')
    
    dout = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    std = np.ascontiguousarray(np.random.rand(dims[nothing_dim], dims[vec_dim]), dtype='float16') + 1.
    diff = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    drop_mask = np.ascontiguousarray(np.random.rand(*dims.values()), dtype='float16')
    
    ref_d_ln_in, ref_d_drop_in = ref_blnrd(dout, std, diff, drop_mask)
    
    for dims_permutation_dout in itertools.permutations(dims):
        for dims_permutation_std in itertools.permutations([nothing_dim, vec_dim]):
            for dims_permutation_diff in itertools.permutations(dims):
                for dims_permutation_drop_mask in itertools.permutations(dims):
                    for dims_permutation_d_ln_in in itertools.permutations(dims):
                        for dims_permutation_d_drop_in in itertools.permutations(dims):
                            dout_label = "".join(dims_permutation_dout)
                            std_label = "".join(dims_permutation_std)
                            diff_label = "".join(dims_permutation_diff)
                            drop_mask_label = "".join(dims_permutation_drop_mask)
                            d_ln_in_label = "".join(dims_permutation_d_ln_in)
                            d_drop_in_label = "".join(dims_permutation_d_drop_in)
                            
                            func_name =  'temp_%s_%s_%s_%s_%s_%s' % (dout_label, std_label, diff_label, drop_mask_label, d_ln_in_label, d_drop_in_label)
                            
                            blnrd = getattr(lib, func_name)
                            
                            d_ln_in_array_shape = list(map(lambda x: dims[x], dims_permutation_d_ln_in))
                            d_ln_in = np.ascontiguousarray(np.zeros(d_ln_in_array_shape), dtype='float16')
                            
                            d_drop_in_array_shape = list(map(lambda x: dims[x], dims_permutation_d_drop_in))
                            d_drop_in = np.ascontiguousarray(np.zeros(d_drop_in_array_shape), dtype='float16')
                            
                            temp_dout = np.ascontiguousarray(np.einsum(base_layout + "->" + dout_label, dout), dtype='float16')
                            temp_std = np.ascontiguousarray(np.einsum(base_std_layout + "->" + std_label, std), dtype='float16')
                            temp_diff = np.ascontiguousarray(np.einsum(base_layout + "->" + diff_label, diff), dtype='float16')
                            temp_drop_mask = np.ascontiguousarray(np.einsum(base_layout + "->" + drop_mask_label, drop_mask), dtype='float16')
                            
                            blnrd(ctypes.c_void_p(temp_dout.ctypes.data),
                                ctypes.c_void_p(temp_std.ctypes.data),
                                ctypes.c_void_p(temp_diff.ctypes.data),
                                ctypes.c_void_p(temp_drop_mask.ctypes.data),
                                ctypes.c_void_p(d_ln_in.ctypes.data),
                                ctypes.c_void_p(d_drop_in.ctypes.data))
                            
                            d_ln_in = np.einsum(d_ln_in_label + "->" + base_layout, d_ln_in)
                            d_drop_in = np.einsum(d_drop_in_label + "->" + base_layout, d_drop_in)
                            
                            layouts = "DOUT %s STD %s DIFF %s DROP_MASK %s D_LN_IN %s D_DROP_IN %s" % (dout_label, std_label, diff_label, drop_mask_label, d_ln_in_label, d_drop_in_label)
                            
                            if not np.allclose(d_ln_in, ref_d_ln_in, atol=1e-2, rtol=1e-1):
                                print(d_ln_in)
                                print('---------------')
                                print(ref_d_ln_in)
                                print("D_LN_IN")
                                raise Exception(layouts)
                            elif not np.allclose(d_drop_in, ref_d_drop_in, atol=1e-2, rtol=1e-1):
                                print(d_drop_in)
                                print('---------------')
                                print(ref_d_drop_in)
                                print("D_DROP_IN")
                                raise Exception(layouts)
                            else:
                                print("OK", layouts)
                                
                            break
                        break
                    break
                break
            break
        break


if __name__ == '__main__':
    test_blnrd()
    print("All tests are passed")
