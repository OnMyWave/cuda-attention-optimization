"""
Setup script for building PyTorch CUDA extension (Fused version)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention_fused',
    ext_modules=[
        CUDAExtension(
            name='attention_fused',
            sources=[
                'attention_cuda.cpp',
                'attention_fused.cu',
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
