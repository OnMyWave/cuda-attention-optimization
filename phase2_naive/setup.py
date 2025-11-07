"""
Setup script for building PyTorch CUDA extension
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA path
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

setup(
    name='attention_cuda',
    ext_modules=[
        CUDAExtension(
            name='attention_cuda',
            sources=[
                'attention_cuda.cpp',
                'attention_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_80,code=sm_80',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)