from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gpu_wait',
    ext_modules=[
        CUDAExtension('gpu_wait', [
            'gpuwait.cpp',
            'wait_kernel.cu'
        ])],
    cmdclass={'build_ext': BuildExtension})

