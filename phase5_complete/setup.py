"""
Setup script for building Phase 5 CUDA extensions (LayerNorm, MLP)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='transformer_ops',
    ext_modules=[
        CUDAExtension(
            name='transformer_ops',
            sources=[
                'transformer_ops_cuda.cpp',
                'layer_norm.cu',
                'mlp.cu',
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
