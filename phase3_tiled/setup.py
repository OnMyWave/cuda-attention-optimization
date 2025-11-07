"""
Setup script for building PyTorch CUDA extension (Tiled version)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='attention_tiled',
    ext_modules=[
        CUDAExtension(
            name='attention_tiled',
            sources=[
                'attention_cuda.cpp',
                'attention_tiled.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_70',
                    '-arch=sm_75',
                    '-arch=sm_80',
                    '-arch=sm_86',
                    '-arch=sm_89',
                    '-arch=sm_90',
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
