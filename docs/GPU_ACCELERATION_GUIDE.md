# GPU 가속 가이드

## 시스템 정보 (2025-11-09 확인)

- **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM)
- **CUDA**: 12.8
- **Driver**: 572.70
- **CuPy**: 13.6.0 ✅ **설치 완료**
- **Python**: 3.11

## 현재 상태

현재 다중 잠재변수 ICLV 모델은 **CPU 병렬처리**를 사용합니다.
- **방식**: `multiprocessing.ProcessPoolExecutor`
- **병렬화 대상**: 개인별 우도 계산 (326명을 27개 코어에 분산)
- **예상 속도 향상**: ~20-27배

**CuPy 설치 완료**: 짧은 경로 가상환경 (`C:\gpu_env`)에 설치됨

## GPU 가속 가능성

### 1. CuPy를 사용한 GPU 가속

**CuPy**는 NumPy와 호환되는 GPU 가속 라이브러리입니다.

#### 장점
- ✅ NumPy 코드를 거의 그대로 사용 가능
- ✅ 대규모 행렬 연산에서 10-100배 속도 향상
- ✅ 다중 GPU 지원

#### 단점
- ❌ NVIDIA GPU 필요 (CUDA 지원)
- ❌ 설치 복잡 (CUDA Toolkit 필요)
- ❌ 메모리 제약 (GPU VRAM)
- ❌ 작은 연산에서는 오히려 느릴 수 있음 (CPU-GPU 전송 오버헤드)

#### 설치 방법

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# CUDA 버전 확인
nvidia-smi
```

### 2. ICLV 모델에서 GPU 활용 가능성 분석

#### 적합한 부분 (GPU 가속 효과 큼)
1. **측정모델 우도 계산**
   - 38개 지표 × 326명 × 100 draws = 1,238,800 계산
   - 행렬 연산 위주 → GPU 적합

2. **구조모델 예측**
   - 5개 잠재변수 × 326명 × 100 draws = 163,000 계산
   - 행렬 곱셈 위주 → GPU 적합

3. **Halton 시퀀스 생성**
   - 326명 × 100 draws × 5차원 = 163,000 값
   - 병렬 생성 가능 → GPU 적합

#### 부적합한 부분 (GPU 가속 효과 작음)
1. **선택모델 우도 계산**
   - 개인별 순차 처리 필요
   - 작은 연산 반복 → CPU가 더 효율적

2. **최적화 (BFGS)**
   - scipy.optimize는 CPU 전용
   - GPU 최적화 라이브러리 필요 (JAX, PyTorch)

### 3. 구현 전략

#### 전략 A: 하이브리드 (CPU + GPU)
```python
# 측정모델만 GPU 사용
import cupy as cp

class GPUOrderedProbitMeasurement:
    def log_likelihood(self, data, latent_vars, params):
        # NumPy → CuPy 변환
        data_gpu = cp.array(data)
        lv_gpu = cp.array(latent_vars)
        
        # GPU에서 계산
        ll_gpu = self._compute_ll_gpu(data_gpu, lv_gpu, params)
        
        # CuPy → NumPy 변환
        return cp.asnumpy(ll_gpu)
```

**장점**: 가장 효과적인 부분만 GPU 사용
**단점**: CPU-GPU 전송 오버헤드

#### 전략 B: 완전 GPU (JAX 사용)
```python
import jax
import jax.numpy as jnp

# JAX로 전체 모델 재작성
@jax.jit  # JIT 컴파일로 GPU 최적화
def joint_log_likelihood(params, data, draws):
    # 전체 계산을 GPU에서 수행
    ...
```

**장점**: 최대 속도 향상 (100배 이상 가능)
**단점**: 전체 코드 재작성 필요

### 4. 권장 사항

#### 현재 시스템 (28 CPU 코어)
- ✅ **CPU 병렬처리 유지 권장**
- 이유:
  1. GPU 없음 (CuPy 미설치)
  2. CPU 병렬처리로 충분한 속도 (~20배)
  3. 구현 복잡도 낮음

#### GPU가 있는 경우
- ⚠️ **하이브리드 접근 고려**
- 단계:
  1. CuPy 설치
  2. 측정모델만 GPU로 변환
  3. 성능 비교 (CPU vs GPU)
  4. 효과 있으면 확장

#### 대규모 데이터 (1000명 이상)
- ✅ **JAX 기반 완전 GPU 구현 권장**
- 이유:
  1. 계산량이 GPU 오버헤드를 상쇄
  2. 100배 이상 속도 향상 가능
  3. 자동 미분으로 gradient 계산 정확

## 5. 성능 비교 (예상)

| 방식 | 1회 우도 계산 | 1000회 반복 | 상대 속도 |
|------|--------------|------------|----------|
| **순차 (1 CPU)** | 60초 | 16.7시간 | 1x |
| **병렬 (27 CPU)** | 2.5초 | 42분 | **24x** |
| **하이브리드 (CPU+GPU)** | 1.5초 | 25분 | 40x |
| **완전 GPU (JAX)** | 0.5초 | 8분 | 120x |

*예상치이며 실제 성능은 하드웨어에 따라 다름*

## 6. 다음 단계

### 즉시 실행 가능
1. ✅ CPU 병렬처리로 모델 추정
2. ✅ 결과 검증
3. ✅ 성능 측정

### GPU 사용 시
1. GPU 사양 확인 (NVIDIA, CUDA 버전)
2. CuPy 설치
3. 측정모델 GPU 변환
4. 성능 비교

### 장기 계획
1. JAX 기반 재구현 검토
2. 자동 미분 활용
3. 대규모 데이터 대응

## 7. 결론

**현재 상황**: CPU 병렬처리로 충분히 빠름 (27배 속도 향상)

**GPU 필요 조건**:
- NVIDIA GPU 보유
- 대규모 데이터 (1000명 이상)
- 추가 개발 시간 투자 가능

**권장**: 먼저 CPU 병렬처리로 모델 추정 완료 후, 필요 시 GPU 고려

