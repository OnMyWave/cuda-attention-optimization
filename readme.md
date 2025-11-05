# CUDA Attention Optimization - Implementation Plan for Claude Code

## Project Overview
Implement and optimize Transformer attention mechanisms in CUDA, with progressive optimizations and Roofline analysis at each stage.

**Target:** Inference-only (forward pass)  
**Timeline:** 5 weeks  
**Primary Goal:** Understand GPU performance through systematic optimization

---

## Phase-by-Phase Implementation Guide

### PHASE 1: PyTorch Baseline (Week 1)
**Goal:** Working Transformer with profiling baseline

#### Tasks:
1. **Setup environment**
   - Install PyTorch, CUDA toolkit, Nsight Compute
   - Verify GPU availability and CUDA version
   - Test basic CUDA kernel compilation

2. **Implement PyTorch Transformer**
   - Small model: 4 layers, 512 hidden dim, 8 attention heads
   - Dataset: WikiText-2 (or synthetic data for simplicity)
   - Focus on inference, no training loop needed

3. **Profile PyTorch attention**
   - Use `torch.profiler` for basic timing
   - Use Nsight Compute for detailed metrics:
     ```bash
     ncu --set full --export baseline python profile_pytorch.py
     ```
   - Extract: GFLOPS, memory bandwidth, occupancy

4. **Calculate baseline Roofline**
   - Operational intensity = FLOPs / Bytes
   - Plot on Roofline (use matplotlib)

#### Deliverables:
```
phase1/
├── pytorch_transformer.py      # Basic Transformer implementation
├── profile_pytorch.py          # Profiling script
├── baseline_roofline.png       # Roofline plot
└── baseline_metrics.json       # Performance numbers
```

#### Success Criteria:
- [ ] Transformer runs correctly on GPU
- [ ] PyTorch attention profiled with Nsight Compute
- [ ] Baseline Roofline plot generated
- [ ] Know exact GFLOPS and bandwidth numbers

---

### PHASE 2: Naive CUDA Implementation (Week 2)
**Goal:** Working CUDA kernels with PyTorch integration

#### Tasks:
1. **Implement 3 separate CUDA kernels**
   
   **Kernel 1: Matrix Multiplication (Q @ K^T)**
   ```cuda
   __global__ void matmul_qk(
       float* Q,        // [batch, seq_len, head_dim]
       float* K,        // [batch, seq_len, head_dim]
       float* scores,   // [batch, seq_len, seq_len]
       int batch, int seq_len, int head_dim
   ) {
       // Each thread computes one element of scores
       // scores[i][j] = sum(Q[i][k] * K[j][k]) / sqrt(head_dim)
   }
   ```

   **Kernel 2: Softmax**
   ```cuda
   __global__ void softmax(
       float* scores,   // [batch, seq_len, seq_len]
       float* attn,     // [batch, seq_len, seq_len]
       int batch, int seq_len
   ) {
       // Per-row softmax
       // 1. Find max (for numerical stability)
       // 2. Compute exp(x - max)
       // 3. Sum and normalize
   }
   ```

   **Kernel 3: Attention @ V**
   ```cuda
   __global__ void matmul_av(
       float* attn,     // [batch, seq_len, seq_len]
       float* V,        // [batch, seq_len, head_dim]
       float* out,      // [batch, seq_len, head_dim]
       int batch, int seq_len, int head_dim
   ) {
       // out[i][d] = sum(attn[i][j] * V[j][d])
   }
   ```

2. **Create PyTorch C++ extension**
   ```python
   # setup.py
   from setuptools import setup
   from torch.utils.cpp_extension import BuildExtension, CUDAExtension
   
   setup(
       name='attention_cuda',
       ext_modules=[
           CUDAExtension('attention_cuda', [
               'attention_cuda.cpp',
               'attention_kernel.cu',
           ])
       ],
       cmdclass={'build_ext': BuildExtension}
   )
   ```

3. **Verify correctness**
   - Compare output with PyTorch (use `torch.allclose`)
   - Test with various batch sizes and sequence lengths
   - Check numerical stability

4. **Profile naive implementation**
   - Same Nsight Compute workflow
   - Compare against PyTorch baseline

#### Deliverables:
```
phase2/
├── attention_kernel.cu         # CUDA kernels
├── attention_cuda.cpp          # PyTorch binding
├── setup.py                    # Build script
├── test_correctness.py         # Validation
├── profile_naive.py            # Profiling script
├── naive_roofline.png          # Updated Roofline
└── naive_metrics.json          # Performance numbers
```

#### Success Criteria:
- [ ] CUDA kernels compile successfully
- [ ] Output matches PyTorch (within numerical tolerance)
- [ ] Can profile with Nsight Compute
- [ ] Know why it's slower than PyTorch (expected!)

---

### PHASE 3: Shared Memory Tiling (Week 3)
**Goal:** Optimize matrix multiplication with shared memory

#### Tasks:
1. **Implement tiled matmul**
   ```cuda
   #define TILE_SIZE 16  // Tune this
   
   __global__ void matmul_tiled(
       float* A, float* B, float* C,
       int M, int N, int K
   ) {
       __shared__ float As[TILE_SIZE][TILE_SIZE];
       __shared__ float Bs[TILE_SIZE][TILE_SIZE];
       
       int row = blockIdx.y * TILE_SIZE + threadIdx.y;
       int col = blockIdx.x * TILE_SIZE + threadIdx.x;
       
       float sum = 0.0f;
       
       // Loop over tiles
       for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
           // Load tile into shared memory
           if (row < M && t * TILE_SIZE + threadIdx.x < K)
               As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
           else
               As[threadIdx.y][threadIdx.x] = 0.0f;
           
           if (col < N && t * TILE_SIZE + threadIdx.y < K)
               Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
           else
               Bs[threadIdx.y][threadIdx.x] = 0.0f;
           
           __syncthreads();
           
           // Compute partial sum
           for (int k = 0; k < TILE_SIZE; k++)
               sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
           
           __syncthreads();
       }
       
       if (row < M && col < N)
           C[row * N + col] = sum;
   }
   ```

2. **Tune tile size**
   - Try TILE_SIZE = 8, 16, 32
   - Measure performance for each
   - Check shared memory usage (48KB limit on most GPUs)

3. **Apply to both matmuls**
   - Q @ K^T
   - Attention @ V

4. **Profile and analyze**
   - Measure shared memory bank conflicts
   - Check occupancy improvements
   - Calculate new operational intensity

#### Deliverables:
```
phase3/
├── attention_tiled.cu          # Tiled kernels
├── profile_tiled.py            # Profiling
├── tiled_roofline.png          # Updated Roofline
└── tile_size_comparison.png    # Performance vs tile size
```

#### Success Criteria:
- [ ] Tiled implementation faster than naive
- [ ] Operational intensity increased (visible on Roofline)
- [ ] Understand shared memory usage
- [ ] Know optimal tile size

---

### PHASE 4: Memory Coalescing (Week 4, Part 1)
**Goal:** Optimize memory access patterns

#### Tasks:
1. **Analyze memory access patterns**
   - Use Nsight Compute's memory analysis
   - Identify uncoalesced accesses
   - Check for bank conflicts

2. **Optimize layouts**
   - Consider transpose operations
   - Restructure thread indexing
   - Ensure consecutive threads access consecutive memory

3. **Compare before/after**
   - Memory throughput metrics
   - Global load efficiency
   - Global store efficiency

#### Deliverables:
```
phase4a/
├── attention_coalesced.cu      # Optimized access patterns
├── memory_analysis.txt         # Nsight Compute report
└── coalesced_metrics.json      # Performance improvements
```

---

### PHASE 5: Kernel Fusion (Week 4, Part 2)
**Goal:** Combine operations to minimize memory traffic

#### Tasks:
1. **Fuse QK^T → Softmax → Attention*V**
   ```cuda
   __global__ void fused_attention(
       float* Q, float* K, float* V, float* out,
       int batch, int seq_len, int head_dim
   ) {
       // Each block handles one query position
       // 1. Compute Q[i] @ K^T (keep in shared memory)
       // 2. Apply softmax (online algorithm)
       // 3. Multiply with V and write output
       // 
       // Key: Never write intermediate results to global memory
   }
   ```

2. **Implement online softmax**
   ```cuda
   // Compute softmax without storing all scores
   // Use max-trick for numerical stability
   // Running sum for normalization
   ```

3. **Optimize register usage**
   - Keep intermediate results in registers
   - Minimize shared memory usage
   - Balance occupancy vs register pressure

4. **Profile final version**
   - Should see minimal memory traffic
   - Higher operational intensity
   - Closer to compute-bound

#### Deliverables:
```
phase4b/
├── attention_fused.cu          # Fused kernel
├── profile_fused.py            # Final profiling
├── final_roofline.png          # Complete Roofline
└── optimization_summary.md     # What worked and why
```

---

### PHASE 6: Additional Components (Week 4, Optional)
**Goal:** Complete Transformer block

#### LayerNorm Implementation:
```cuda
__global__ void layer_norm(
    float* x,           // Input [batch, seq_len, hidden_dim]
    float* gamma,       // Scale [hidden_dim]
    float* beta,        // Shift [hidden_dim]
    float* out,         // Output [batch, seq_len, hidden_dim]
    int batch, int seq_len, int hidden_dim
) {
    // Each block handles one sequence position
    // 1. Compute mean (warp reduction)
    // 2. Compute variance (warp reduction)
    // 3. Normalize: (x - mean) / sqrt(var + eps)
    // 4. Scale and shift: out = gamma * normalized + beta
}
```

**Warp-level reduction:**
```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

#### MLP Implementation:
```cuda
__global__ void fused_mlp(
    float* x,           // [batch, seq_len, hidden_dim]
    float* W1,          // [hidden_dim, 4*hidden_dim]
    float* W2,          // [4*hidden_dim, hidden_dim]
    float* out,         // [batch, seq_len, hidden_dim]
    int batch, int seq_len, int hidden_dim
) {
    // Fuse: Linear1 → GELU → Linear2
    // Keep intermediate results in shared memory
}
```

**GELU activation:**
```cuda
__device__ float gelu(float x) {
    // Approximate: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c = 0.797884560804236f; // sqrt(2/pi)
    const float a = 0.044715f;
    float x_cubed = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + a * x_cubed)));
}
```

---

## PHASE 7: Integration and Analysis (Week 5)

### Tasks:
1. **Integrate all components**
   ```python
   class OptimizedTransformer(nn.Module):
       def __init__(self):
           self.attention = CustomAttention()  # Your CUDA
           self.layer_norm1 = CustomLayerNorm()  # Your CUDA or PyTorch
           self.mlp = CustomMLP()  # Your CUDA or PyTorch
           self.layer_norm2 = nn.LayerNorm()  # PyTorch is fine
   ```

2. **End-to-end benchmarking**
   - Measure full forward pass
   - Compare against pure PyTorch
   - Test with different sequence lengths (128, 256, 512, 1024)
   - Test with different batch sizes

3. **Generate final visualizations**
   ```python
   # Roofline plot with all variants
   # Speedup chart
   # Performance breakdown by component
   # Scaling analysis (seq_len vs throughput)
   ```

4. **Write report**
   - What worked and why
   - What didn't work and why
   - Lessons learned
   - Future optimizations

### Deliverables:
```
final/
├── optimized_transformer.py    # Complete implementation
├── benchmark.py                # End-to-end testing
├── visualizations/
│   ├── complete_roofline.png
│   ├── speedup_chart.png
│   ├── scaling_analysis.png
│   └── component_breakdown.png
├── report.pdf                  # Final writeup
└── README.md                   # How to run everything
```

---

## File Structure for Entire Project

```
cuda-attention-optimization/
├── README.md
├── IMPLEMENTATION_PLAN.md (this file)
├── environment.yml
├── requirements.txt
│
├── phase1_baseline/
│   ├── pytorch_transformer.py
│   ├── profile_pytorch.py
│   └── results/
│
├── phase2_naive/
│   ├── attention_kernel.cu
│   ├── attention_cuda.cpp
│   ├── setup.py
│   ├── test_correctness.py
│   └── results/
│
├── phase3_tiled/
│   ├── attention_tiled.cu
│   ├── setup.py
│   └── results/
│
├── phase4_optimized/
│   ├── attention_coalesced.cu
│   ├── attention_fused.cu
│   ├── setup.py
│   └── results/
│
├── phase5_complete/ (optional)
│   ├── layer_norm.cu
│   ├── mlp.cu
│   └── results/
│
├── final/
│   ├── optimized_transformer.py
│   ├── benchmark.py
│   ├── visualizations/
│   └── report/
│
└── utils/
    ├── profiling.py
    ├── roofline.py
    ├── visualization.py
    └── testing.py
```

---

## Key Implementation Details

### Attention Math Recap
```
Q, K, V: [batch, seq_len, head_dim]

scores = Q @ K^T / sqrt(head_dim)  # [batch, seq_len, seq_len]
attn = softmax(scores, dim=-1)     # [batch, seq_len, seq_len]
out = attn @ V                      # [batch, seq_len, head_dim]
```

### CUDA Kernel Launch Configuration

**Naive implementation:**
```cpp
dim3 block(16, 16);  // 256 threads per block
dim3 grid(
    (seq_len + block.x - 1) / block.x,
    (seq_len + block.y - 1) / block.y
);
matmul_qk<<<grid, block>>>(Q, K, scores, ...);
```

**Tiled implementation:**
```cpp
dim3 block(TILE_SIZE, TILE_SIZE);
dim3 grid(
    (N + TILE_SIZE - 1) / TILE_SIZE,
    (M + TILE_SIZE - 1) / TILE_SIZE
);
matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
```

**Fused implementation:**
```cpp
// Each block handles multiple query positions
dim3 block(256);  // Tune this
dim3 grid((seq_len + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK);
fused_attention<<<grid, block>>>(Q, K, V, out, ...);
```

### Profiling Commands

**Basic timing:**
```bash
python -m torch.utils.benchmark
```

**Nsight Compute (comprehensive):**
```bash
ncu --set full --export profile_output python script.py
```

**Nsight Compute (specific metrics):**
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum \
python script.py
```

### Roofline Calculation

**Operational Intensity:**
```
FLOPs = 2 * batch * seq_len * seq_len * head_dim  (for Q@K^T)
Bytes = (batch * seq_len * head_dim) * 4 * 2  (Q and K, float32)
      + (batch * seq_len * seq_len) * 4        (scores output)
Intensity = FLOPs / Bytes
```

**Plot:**
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(peak_gflops, peak_bandwidth_gb_s):
    intensities = np.logspace(-2, 2, 100)
    
    # Memory-bound ceiling
    memory_bound = intensities * peak_bandwidth_gb_s
    
    # Compute-bound ceiling
    compute_bound = np.ones_like(intensities) * peak_gflops
    
    # Roofline
    roofline = np.minimum(memory_bound, compute_bound)
    
    plt.loglog(intensities, roofline, 'k-', linewidth=2, label='Roofline')
    plt.loglog(measured_intensity, measured_gflops, 'ro', label='Measured')
    plt.xlabel('Operational Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (GFLOPS)')
    plt.legend()
    plt.grid(True)
    plt.savefig('roofline.png')
```

---

## Testing Strategy

### Unit Tests
```python
def test_matmul():
    # Small matrices
    Q = torch.randn(1, 4, 8, device='cuda')
    K = torch.randn(1, 4, 8, device='cuda')
    
    # PyTorch reference
    expected = Q @ K.transpose(-2, -1)
    
    # Your CUDA kernel
    actual = attention_cuda.matmul_qk(Q, K)
    
    assert torch.allclose(expected, actual, rtol=1e-4, atol=1e-5)

def test_softmax():
    scores = torch.randn(1, 4, 4, device='cuda')
    
    expected = torch.softmax(scores, dim=-1)
    actual = attention_cuda.softmax(scores)
    
    assert torch.allclose(expected, actual, rtol=1e-4, atol=1e-5)

def test_attention():
    batch, seq_len, head_dim = 2, 8, 64
    Q = torch.randn(batch, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, seq_len, head_dim, device='cuda')
    
    # PyTorch reference
    scores = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    expected = attn @ V
    
    # Your CUDA kernel
    actual = attention_cuda.forward(Q, K, V)
    
    assert torch.allclose(expected, actual, rtol=1e-3, atol=1e-4)
```

### Performance Tests
```python
import torch.utils.benchmark as benchmark

def benchmark_attention(batch, seq_len, head_dim):
    Q = torch.randn(batch, seq_len, head_dim, device='cuda')
    K = torch.randn(batch, seq_len, head_dim, device='cuda')
    V = torch.randn(batch, seq_len, head_dim, device='cuda')
    
    # PyTorch
    pytorch_time = benchmark.Timer(
        stmt='attention_pytorch(Q, K, V)',
        globals={'attention_pytorch': attention_pytorch, 'Q': Q, 'K': K, 'V': V}
    ).blocked_autorange()
    
    # Your CUDA
    cuda_time = benchmark.Timer(
        stmt='attention_cuda.forward(Q, K, V)',
        globals={'attention_cuda': attention_cuda, 'Q': Q, 'K': K, 'V': V}
    ).blocked_autorange()
    
    print(f"PyTorch: {pytorch_time.mean * 1000:.3f} ms")
    print(f"CUDA: {cuda_time.mean * 1000:.3f} ms")
    print(f"Speedup: {pytorch_time.mean / cuda_time.mean:.2f}x")
```

---

## Common Pitfalls and Solutions

### Issue 1: CUDA Extension Won't Compile
**Symptoms:** `nvcc: command not found` or compilation errors

**Solutions:**
```bash
# Check CUDA installation
nvcc --version

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue 2: Kernel Produces Wrong Results
**Symptoms:** `torch.allclose` fails

**Solutions:**
1. Test with small inputs (e.g., 2x2 matrices)
2. Print intermediate results
3. Check array indexing carefully
4. Verify memory layouts (row-major vs column-major)
5. Check for race conditions (missing `__syncthreads()`)

### Issue 3: Kernel is Slower Than Expected
**Symptoms:** No speedup over PyTorch

**Solutions:**
1. Check occupancy with Nsight Compute
2. Verify coalesced memory access
3. Check for bank conflicts in shared memory
4. Profile with `ncu --set full`
5. Compare against theoretical peak

### Issue 4: Out of Memory
**Symptoms:** CUDA out of memory errors

**Solutions:**
1. Reduce batch size
2. Reduce sequence length
3. Use smaller tile sizes
4. Check for memory leaks
5. Use `torch.cuda.empty_cache()`

### Issue 5: Numerical Instability in Softmax
**Symptoms:** NaN or inf values

**Solutions:**
```cuda
// Always use max-trick
float max_val = -INFINITY;
for (int i = 0; i < N; i++)
    max_val = fmaxf(max_val, scores[i]);

float sum = 0.0f;
for (int i = 0; i < N; i++) {
    scores[i] = expf(scores[i] - max_val);  // Subtract max
    sum += scores[i];
}

for (int i = 0; i < N; i++)
    scores[i] /= sum;
```

---

## Success Metrics by Phase

| Phase | Minimum Success | Good Success | Excellent Success |
|-------|----------------|--------------|-------------------|
| 1 | PyTorch runs | Profiled correctly | Roofline plotted |
| 2 | CUDA compiles | Output correct | Profiled |
| 3 | Tiled works | 1.5x speedup | 2x+ speedup |
| 4 | Coalesced works | 1.3x speedup | Fused works well |
| 5 | One extra op | LayerNorm works | Full block works |
| 6 | Report done | Good analysis | Publication-quality |

---

## Time Budget Estimation

**Aggressive (論文 準備中):**
- Phase 1: 2 days
- Phase 2: 3 days
- Phase 3: 3 days
- Phase 4: 3 days
- Phase 5: Skip
- Phase 6: 2 days

**Comfortable:**
- Phase 1: 3 days
- Phase 2: 4 days
- Phase 3: 4 days
- Phase 4: 5 days
- Phase 5: 3 days
- Phase 6: 3 days

**Comprehensive:**
- Phase 1: 4 days
- Phase 2: 5 days
- Phase 3: 5 days
- Phase 4: 7 days
- Phase 5: 5 days
- Phase 6: 4 days

---

## References for Implementation

1. **CUDA C Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/
2. **PyTorch C++ Extension Tutorial:** https://pytorch.org/tutorials/advanced/cpp_extension.html
3. **Nsight Compute User Guide:** https://docs.nvidia.com/nsight-compute/
4. **FlashAttention Code:** https://github.com/Dao-AILab/flash-attention
5. **CUTLASS (NVIDIA CUDA Templates):** https://github.com/NVIDIA/cutlass

---

## Final Checklist

Before moving to next phase:
- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Correctness verified against PyTorch
- [ ] Profiled with Nsight Compute
- [ ] Metrics documented
- [ ] Roofline plot updated
- [ ] Code committed to git

Before final submission:
- [ ] All phases completed (or documented why skipped)
- [ ] Complete Roofline analysis
- [ ] End-to-end benchmarks
- [ ] Report written
- [ ] Code cleaned and commented
- [ ] README with instructions
- [ ] Presentation slides prepared

---

## Emergency Fallback Plan

**If running out of time after Week 3:**
1. Stop at tiled implementation
2. Write thorough analysis of Phases 1-3
3. Explain what you would do in Phases 4-5
4. Focus on quality of Roofline analysis
5. Emphasize learning and understanding

**This is still a successful project!**

---

## Getting Started Command

```bash
# Clone your repo
git clone <your-repo>
cd cuda-attention-optimization

# Create environment
conda create -n cuda-opt python=3.10
conda activate cuda-opt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy pytest

# Start Phase 1
cd phase1_baseline
python pytorch_transformer.py
```

Good luck! 🚀
