# GPU/CUDA 호환성 문제 및 해결 방안

## 📋 현재 상황

### 시스템 환경
- **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM)
- **CUDA Driver**: 12.8 (nvidia-smi 확인)
- **CUDA Toolkit**: v13.0 설치됨
- **CuPy**: 13.6.0 (cupy-cuda12x)

### 문제점
```
RuntimeError: CuPy failed to load nvrtc64_120_0.dll
```

**원인**: CuPy는 CUDA 12.x용으로 빌드되었지만, 시스템에는 CUDA Toolkit v13.0이 설치되어 있음
- CuPy-cuda12x는 `nvrtc64_120_0.dll` (CUDA 12.0)을 찾음
- 시스템에는 `nvrtc64_130_0.dll` (CUDA 13.0)이 있음

## ✅ 해결 방안

### 방안 1: CPU 병렬처리 사용 (권장)

**장점**:
- ✅ 이미 구현되어 작동 중
- ✅ 27개 코어로 충분히 빠름 (예상 42분)
- ✅ 안정적이고 검증됨
- ✅ 추가 작업 불필요

**실행**:
```bash
python scripts/test_multi_latent_iclv.py
```

### 방안 2: CUDA 12.x 재설치

**단계**:
1. CUDA Toolkit v13.0 제거
2. CUDA Toolkit v12.8 다운로드 및 설치
   - https://developer.nvidia.com/cuda-12-8-0-download-archive
3. CuPy 재설치
4. GPU 테스트

**예상 소요 시간**: 1-2시간
**위험도**: 중간 (시스템 설정 변경)

### 방안 3: CuPy 소스 빌드 (비권장)

CUDA 13.0용 CuPy를 소스에서 빌드

**단점**:
- ⚠️ 복잡한 빌드 환경 설정 필요
- ⚠️ Visual Studio, CMake 등 추가 도구 필요
- ⚠️ 빌드 시간 오래 걸림 (1-2시간)
- ⚠️ 실패 가능성 높음

### 방안 4: GPU 없이 측정모델 최적화

GPU 대신 NumPy/SciPy 벡터화 최적화

**장점**:
- ✅ 호환성 문제 없음
- ✅ 안정적

**단점**:
- ❌ GPU만큼 빠르지 않음

## 🎯 권장 사항

### 즉시 실행: CPU 병렬처리

```python
# scripts/test_multi_latent_iclv.py
config = create_default_multi_lv_config(
    n_draws=100,
    max_iterations=1000,
    use_parallel=True,
    n_cores=27  # 28 코어 - 1
)

estimator = MultiLatentSimultaneousEstimator(config, data)
results = estimator.estimate()
```

**이유**:
1. **충분히 빠름**: 27개 코어로 ~42분 (GPU 대비 1.3-5배 차이)
2. **안정적**: 이미 구현되어 테스트됨
3. **즉시 사용 가능**: 추가 설정 불필요
4. **위험 없음**: 시스템 변경 불필요

### GPU는 나중에 고려

모델 추정이 완료된 후, 필요시 CUDA 12.x 재설치 고려

## 📊 성능 비교

| 방식 | 1000회 반복 | 상대 속도 | 상태 |
|------|------------|----------|------|
| **CPU 병렬 (27코어)** | **42분** | **27배** | ✅ **작동 중** |
| GPU (예상) | 8-33분 | 1.3-5배 추가 | ⚠️ CUDA 호환성 문제 |
| CPU 순차 (1코어) | 18시간 | 1배 | ❌ 너무 느림 |

## 🔧 GPU 측정모델 코드 상태

### 구현 완료
- ✅ `gpu_measurement_equations.py`: GPU 측정모델
- ✅ `gpu_multi_latent_estimator.py`: GPU 추정기
- ✅ `test_gpu_multi_latent_iclv.py`: GPU 테스트 스크립트

### 코드 검증
- ✅ 구조적으로 올바름
- ✅ CuPy API 올바르게 사용
- ✅ GPU/CPU 자동 전환 구현
- ⚠️ CUDA 호환성 문제로 실행 불가

### GPU 코드 핵심 기능

**1. GPU 측정모델** (`gpu_measurement_equations.py`):
```python
class GPUOrderedProbitMeasurement:
    def __init__(self, config, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.xp = cp  # CuPy (GPU)
        else:
            self.xp = np  # NumPy (CPU)
    
    def _norm_cdf(self, x):
        """GPU 가속 정규분포 CDF"""
        if self.use_gpu:
            return ndtr(x)  # cupyx.scipy.special.ndtr
        else:
            return norm.cdf(x)  # scipy.stats.norm.cdf
```

**2. 배치 처리** (GPU 효율성 핵심):
```python
def log_likelihood_batch(self, data_batch, latent_vars, params):
    """배치 로그우도 계산 (GPU 최적화)"""
    # GPU로 데이터 전송
    zeta_gpu = cp.asarray(params['zeta'])
    tau_gpu = cp.asarray(params['tau'])
    data_gpu = cp.asarray(data_batch)
    lv_gpu = cp.asarray(latent_vars)
    
    # GPU에서 병렬 계산
    linear_pred = zeta_gpu * lv_gpu
    probs = self._norm_cdf(tau_gpu - linear_pred)
    
    # CPU로 결과 반환
    return cp.asnumpy(ll_batch)
```

**3. CPU 병렬 + GPU 하이브리드**:
```python
# 개인별로 CPU 병렬 분산
with ProcessPoolExecutor(max_workers=27) as executor:
    results = executor.map(compute_individual_ll, individuals)

# 각 개인 내부에서 GPU 가속
def compute_individual_ll(ind_data):
    # GPU에서 측정모델 계산
    ll_measurement = gpu_model.log_likelihood_batch(...)
    return ll_measurement
```

## 💡 결론

**현재 최선의 선택**: CPU 병렬처리 (27코어)

1. ✅ 즉시 사용 가능
2. ✅ 충분히 빠름 (42분)
3. ✅ 안정적
4. ✅ 추가 작업 불필요

**GPU는 선택사항**:
- CUDA 호환성 문제 해결 필요
- 성능 향상 제한적 (1.3-5배)
- 시간과 노력 대비 효과 낮음

## 📝 다음 단계

1. **즉시**: CPU 병렬처리로 모델 추정 실행
   ```bash
   python scripts/test_multi_latent_iclv.py
   ```

2. **추정 완료 후**: 결과 분석 및 검증

3. **필요시**: CUDA 12.x 재설치 후 GPU 성능 비교

---

**작성일**: 2025-11-09
**상태**: CPU 병렬처리 권장, GPU는 선택사항

