# CUDA Attention Optimization

A progressive implementation of Transformer attention mechanisms in CUDA, from baseline PyTorch to optimized CUDA kernels with Roofline analysis.

## Project Structure

```
cuda-attention-optimization/
├── README.md                    # This file
├── readme.md                    # Detailed implementation plan
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
│
├── phase1_baseline/             # PyTorch baseline
│   ├── pytorch_transformer.py   # Transformer implementation
│   ├── profile_pytorch.py       # Profiling script
│   ├── visualize_results.py     # Visualization
│   └── results/                 # Profiling results
│
├── phase2_naive/                # Naive CUDA implementation
│   ├── attention_kernel.cu      # CUDA kernels
│   ├── attention_cuda.cpp       # PyTorch binding
│   ├── setup.py                 # Build script
│   ├── test_correctness.py      # Correctness tests
│   ├── profile_naive.py         # Profiling
│   └── results/                 # Results
│
├── phase3_tiled/                # Tiled with shared memory
│   ├── attention_tiled.cu       # Tiled kernels
│   ├── attention_cuda.cpp       # PyTorch binding
│   ├── setup.py                 # Build script
│   ├── test_tiled.py            # Tests
│   └── results/                 # Results
│
├── phase4_optimized/            # Fused kernel
│   ├── attention_fused.cu       # Fused attention kernel
│   ├── attention_cuda.cpp       # PyTorch binding
│   ├── setup.py                 # Build script
│   ├── test_fused.py            # Tests
│   ├── profile_fused.py         # Profiling
│   └── results/                 # Results
│
├── phase5_complete/             # LayerNorm & MLP
│   ├── layer_norm.cu            # LayerNorm kernel
│   ├── mlp.cu                   # MLP kernel
│   ├── transformer_ops_cuda.cpp # PyTorch binding
│   ├── setup.py                 # Build script
│   ├── test_transformer_ops.py  # Tests
│   └── results/                 # Results
│
├── final/                       # Phase 6-7: Integration
│   ├── optimized_transformer.py # Complete optimized model
│   ├── benchmark.py             # End-to-end benchmarks
│   └── visualizations/          # Final plots
│
└── utils/                       # Utilities
    └── roofline.py              # Roofline analysis
```

## Quick Start

### 1. Environment Setup

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate cuda-opt

# Or using pip
pip install -r requirements.txt
```

### 2. Verify CUDA Installation

```bash
nvcc --version
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Phase 1: PyTorch Baseline

```bash
cd phase1_baseline

# Test the implementation
python3 pytorch_transformer.py

# Profile (requires PyTorch to be installed with CUDA support)
python3 profile_pytorch.py

# Visualize results
python3 visualize_results.py
```

### 4. Phase 2: Naive CUDA

```bash
cd phase2_naive

# Build the CUDA extension
python3 setup.py install

# Test correctness
python3 test_correctness.py

# Profile performance
python3 profile_naive.py
```

### 5. Phase 3: Tiled Implementation

```bash
cd phase3_tiled

# Build the CUDA extension
python3 setup.py install

# Test correctness
python3 test_tiled.py

# Profile performance
python3 profile_tiled.py
```

### 6. Phase 4: Fused Kernel

```bash
cd phase4_optimized

# Build the CUDA extension
python3 setup.py install

# Test correctness
python3 test_fused.py

# Profile and compare all implementations
python3 profile_fused.py
```

### 7. Phase 5: LayerNorm & MLP

```bash
cd phase5_complete

# Build the CUDA extension
python3 setup.py install

# Test correctness
python3 test_transformer_ops.py
```

### 8. Phase 6-7: End-to-End Benchmarking

```bash
cd final

# Test integrated optimized transformer
python3 optimized_transformer.py

# Run comprehensive benchmarks (requires all previous phases)
python3 benchmark.py
```

## Implementation Overview

### Phase 1: PyTorch Baseline
- Multi-head attention implementation in PyTorch
- Profiling with torch.profiler and Nsight Compute
- Baseline Roofline analysis

### Phase 2: Naive CUDA
- Three separate CUDA kernels:
  1. Q @ K^T matrix multiplication
  2. Softmax (per-row, numerically stable)
  3. Attention @ V matrix multiplication
- PyTorch C++ extension integration
- Correctness validation against PyTorch

### Phase 3: Shared Memory Tiling
- Tiled matrix multiplication for better memory reuse
- Optimized softmax with warp-level reductions
- Configurable tile size (default: 16x16)
- Improved operational intensity

### Phase 4: Fused Kernel ✓
- Kernel fusion (QK^T → Softmax → Attention@V)
- Online softmax algorithm
- Minimized global memory traffic
- No intermediate score matrix storage

### Phase 5: Complete Transformer Components ✓
- LayerNorm with warp-level reductions
- Fused MLP (Linear → GELU → Linear)
- Optimized for common dimensions (512, 768, 1024, 2048)

### Phase 6-7: Integration & Analysis ✓
- Complete optimized Transformer model
- End-to-end benchmarking
- Comprehensive performance analysis
- Scaling analysis across configurations

## Performance Metrics

Each phase includes:
- ✓ Execution time (ms)
- ✓ Throughput (GFLOPS)
- ✓ Memory bandwidth (GB/s)
- ✓ Operational intensity (FLOPs/Byte)
- ✓ Roofline plot
- ✓ Speedup comparison

## System Requirements

- CUDA Toolkit (12.0+)
- PyTorch (2.0+) with CUDA support
- Python 3.10+
- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)

## Profiling with Nsight Compute

```bash
# Comprehensive profiling
ncu --set full --export profile_output python3 profile_pytorch.py

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
python3 profile_naive.py
```

## Testing

All implementations are validated against PyTorch reference:
- Numerical correctness (`torch.allclose`)
- Multiple batch sizes and sequence lengths
- Edge cases and numerical stability

## Key Features

### Naive CUDA Kernels
- Simple per-element thread mapping
- Numerically stable softmax (max trick)
- Baseline for optimization comparison

### Tiled Kernels
- Shared memory for data reuse
- Reduced global memory traffic
- Warp-level primitives (`__shfl_down_sync`)
- Better cache utilization

### Roofline Analysis
- Identifies memory vs compute bottlenecks
- Tracks optimization progress
- Guides further optimization efforts

## Known Limitations

1. **FP32 only**: Current implementation uses single precision
2. **No masking**: Causal masking not yet implemented in CUDA kernels
3. **Fixed dimensions**: Optimized for specific tile sizes
4. **Inference only**: No backward pass

## Implementation Complete ✓

All 7 phases have been implemented:
- ✓ Phase 1: PyTorch Baseline
- ✓ Phase 2: Naive CUDA Kernels
- ✓ Phase 3: Tiled Implementation
- ✓ Phase 4: Fused Kernel
- ✓ Phase 5: LayerNorm & MLP
- ✓ Phase 6: Integration
- ✓ Phase 7: Comprehensive Benchmarking

## Future Enhancements

- [ ] Add FP16/BF16 support
- [ ] Implement FlashAttention-2 improvements
- [ ] Add causal masking support in CUDA
- [ ] Multi-head attention batching
- [ ] Backward pass for training
- [ ] Triton implementation comparison

## References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [CUTLASS](https://github.com/NVIDIA/cutlass)

## License

This project is for educational purposes.

## Authors

Implementation following the detailed plan in `readme.md`.
