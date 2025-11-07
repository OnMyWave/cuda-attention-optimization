# Getting Started Guide

이 가이드는 CUDA Attention Optimization 프로젝트를 단계별로 실행하는 방법을 설명합니다.

## 사전 요구사항

### 필수 소프트웨어
- CUDA Toolkit 12.0 이상
- Python 3.10 이상
- PyTorch 2.0 이상 (CUDA 지원)
- GCC/G++ 컴파일러

### GPU 요구사항
- NVIDIA GPU (Compute Capability 7.0 이상)
- 권장: V100, A100, RTX 3090, RTX 4090, H100

## 설치 단계

### 1. CUDA 확인

```bash
# CUDA 버전 확인
nvcc --version

# CUDA 경로 설정 (필요한 경우)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. Python 환경 설정

#### 방법 A: Conda 사용 (권장)

```bash
# Conda 환경 생성
conda env create -f environment.yml
conda activate cuda-opt

# 또는 수동으로 설치
conda create -n cuda-opt python=3.10
conda activate cuda-opt
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install matplotlib numpy pytest
```

#### 방법 B: pip 사용

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. PyTorch CUDA 확인

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

출력 예시:
```
CUDA available: True
Device: NVIDIA A100-SXM4-40GB
```

## Phase별 실행 가이드

### Phase 1: PyTorch Baseline

```bash
cd phase1_baseline

# 1. Transformer 모델 테스트
python3 pytorch_transformer.py
# 출력: 모델이 정상적으로 실행되고 forward pass가 완료됨

# 2. 성능 프로파일링
python3 profile_pytorch.py
# 출력: results/baseline_metrics.json 생성

# 3. 결과 시각화
python3 visualize_results.py
# 출력: results/baseline_roofline.png, results/performance_comparison.png 생성
```

**예상 결과:**
- 실행 시간 측정 완료
- GFLOPS, 메모리 대역폭 계산
- Roofline 플롯 생성

### Phase 2: Naive CUDA Implementation

```bash
cd ../phase2_naive

# 1. CUDA extension 빌드
python3 setup.py install
# 빌드 시간: 1-3분 소요 (GPU 아키텍처에 따라 다름)

# 2. 정확성 테스트
python3 test_correctness.py
# 모든 테스트가 PASSED 되어야 함

# 3. 성능 프로파일링
python3 profile_naive.py
# 출력: results/naive_metrics.json, results/naive_roofline.png
```

**빌드 문제 해결:**

만약 빌드가 실패하면:

```bash
# CUDA 경로 확인
echo $CUDA_HOME

# 없다면 설정
export CUDA_HOME=/usr/local/cuda-12.4

# 다시 빌드
python3 setup.py clean
python3 setup.py install
```

**예상 성능:**
- PyTorch 대비 0.5x ~ 1.5x (naive 구현이므로 더 느릴 수 있음)
- 이는 정상이며, 최적화의 출발점입니다

### Phase 3: Tiled Implementation

```bash
cd ../phase3_tiled

# 1. CUDA extension 빌드
python3 setup.py install

# 2. 정확성 테스트
python3 test_tiled.py
# 모든 테스트가 PASSED 되어야 함

# 3. 성능 프로파일링 (추가 구현 필요)
# python3 profile_tiled.py
```

**예상 성능:**
- Naive 대비 1.5x ~ 3x 향상
- Shared memory 사용으로 메모리 대역폭 향상

### Phase 4: Fused Kernel (Advanced)

```bash
cd ../phase4_optimized

# 1. CUDA extension 빌드
python3 setup.py install

# 2. 테스트 및 프로파일링 (구현 필요)
```

**예상 성능:**
- Tiled 대비 1.5x ~ 2x 향상
- Global memory traffic 최소화

## 성능 벤치마크 예시

일반적인 구성 (batch=4, seq_len=128, head_dim=64):

| Implementation | Time (ms) | GFLOPS | Speedup |
|---------------|-----------|--------|---------|
| PyTorch       | 0.500     | 500    | 1.0x    |
| Naive CUDA    | 0.600     | 400    | 0.8x    |
| Tiled CUDA    | 0.300     | 800    | 1.7x    |
| Fused CUDA    | 0.200     | 1200   | 2.5x    |

*실제 성능은 GPU 모델에 따라 크게 달라질 수 있습니다.*

## 프로파일링 with Nsight Compute

더 자세한 분석을 원한다면:

```bash
# 전체 프로파일링
ncu --set full --export profile_output python3 profile_pytorch.py

# 특정 메트릭만
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed python3 profile_naive.py

# 결과 보기
ncu-ui profile_output.ncu-rep
```

## 일반적인 문제 해결

### 문제 1: "nvcc: command not found"

```bash
# CUDA 설치 확인
ls /usr/local/cuda-*/bin/nvcc

# 경로 설정
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

### 문제 2: "CUDA error: no kernel image available"

이는 GPU의 compute capability와 컴파일된 아키텍처가 맞지 않는 경우입니다.

```bash
# GPU compute capability 확인
python3 -c "import torch; print(torch.cuda.get_device_capability())"

# setup.py에서 해당 아키텍처 추가
# 예: compute capability 8.6 (RTX 3090) → -arch=sm_86
```

### 문제 3: PyTorch import 오류

```bash
# PyTorch 재설치
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 문제 4: 메모리 부족

```bash
# 작은 배치 크기로 테스트
# profile 스크립트에서 batch_size나 seq_len 줄이기
```

## 다음 단계

1. **Phase별 결과 비교**: 각 phase의 metrics.json 파일을 비교하여 최적화 효과 확인
2. **Roofline 분석**: 각 구현이 memory-bound인지 compute-bound인지 확인
3. **파라미터 튜닝**: tile_size 등의 파라미터 변경하여 성능 개선 시도
4. **실제 응용**: 자신의 Transformer 모델에 통합

## 참고 자료

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch CUDA Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

## 도움이 필요한 경우

1. 프로젝트의 `readme.md` 파일에서 자세한 구현 계획 확인
2. 각 phase의 코드에 있는 주석 참고
3. GPU 모델별 최적화 팁은 CUDA 문서 참조

## 프로젝트 완료 체크리스트

- [ ] Phase 1: PyTorch baseline 실행 및 프로파일링 완료
- [ ] Phase 2: Naive CUDA 빌드 및 정확성 검증 완료
- [ ] Phase 3: Tiled implementation 정확성 검증 완료
- [ ] Phase 4: Fused kernel 구현 (선택사항)
- [ ] 모든 phase의 Roofline 분석 완료
- [ ] 성능 비교 문서 작성

성공적인 구현을 기원합니다! 🚀
