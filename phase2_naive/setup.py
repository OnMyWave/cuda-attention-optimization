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
                    '-arch=sm_70',  # Compute capability 7.0 (V100, Titan V)
                    '-arch=sm_75',  # Compute capability 7.5 (Turing: RTX 20 series, T4)
                    '-arch=sm_80',  # Compute capability 8.0 (Ampere: A100)
                    '-arch=sm_86',  # Compute capability 8.6 (Ampere: RTX 30 series)
                    '-arch=sm_89',  # Compute capability 8.9 (Ada Lovelace: RTX 40 series)
                    '-arch=sm_90',  # Compute capability 9.0 (Hopper: H100)
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
