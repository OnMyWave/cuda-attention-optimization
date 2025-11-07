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
            libraries=['cublas'],
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
