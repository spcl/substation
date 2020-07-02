import ctypes

# Requires the .so be in LD_LIBRARY_PATH.
_cudart = ctypes.CDLL('libcudart.so')

start_cuda_profiling = _cudart.cudaProfilerStart
stop_cuda_profiling = _cudart.cudaProfilerStop
