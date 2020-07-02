""" Data types and support for DaCe/NumPy/PyTorch. """

# Avoid further imports to be imported along with this file
import dace as _dace
import numpy as _np
import torch as _torch

# Default type configuration
np_dtype = _np.float32

if np_dtype == _np.float32:
    dace_dtype = _dace.float32
    torch_dtype = _torch.float32
    c_dtype = 'float'
    cublas_gemm = 'cublasSgemm'
    exp_func = 'expf'
    max_func = 'fmaxf'
elif np_dtype == _np.float64:
    dace_dtype = _dace.float64
    torch_dtype = _torch.float64
    c_dtype = 'double'
    cublas_gemm = 'cublasDgemm'
    exp_func = 'exp'
    max_func = 'fmax'
elif np_dtype == _np.float16:
    dace_dtype = _dace.float16
    torch_dtype = _torch.float16
    c_dtype = 'half'
    cublas_gemm = 'cublasHgemm'
    exp_func = 'expf'  # hexp? h2exp?
    max_func = 'fmaxf'  # DACE_MAX? (a < b ? b : a)
else:
    raise TypeError('Unsuppoorted type %s' % np_dtype.__name__)
