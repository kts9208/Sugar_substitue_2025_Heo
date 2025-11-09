# 측정모델 GPU 변환 가능성 분석

## 요약

✅ **CuPy 설치 완료**: `C:\gpu_env` (짧은 경로 가상환경)
⚠️ **CUDA 라이브러리 경로 문제**: cuRAND DLL 로드 실패
📊 **측정모델 GPU 변환**: 가능하지만 제한적 효과 예상

## 1. CuPy 설치 현황

### 설치 성공
```bash
# 짧은 경로 가상환경 생성
python -m venv /c/gpu_env

# CuPy 설치
/c/gpu_env/Scripts/pip.exe install cupy-cuda12x

# 설치 확인
CuPy version: 13.6.0
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce RTX 4060
GPU memory: 7.99 GB
```

### 현재 문제
```
ImportError: DLL load failed while importing curand: 지정된 모듈을 찾을 수 없습니다.
```

**원인**: CUDA Toolkit이 시스템에 설치되지 않았거나 PATH에 없음

**해결 방법**:
1. CUDA Toolkit 12.8 설치: https://developer.nvidia.com/cuda-downloads
2. 환경변수 설정: `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
3. PATH에 추가: `%CUDA_PATH%\bin`

## 2. 측정모델 GPU 변환 분석

### 현재 측정모델 구조

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/measurement_equations.py" mode="EXCERPT">
```python
class OrderedProbitMeasurement:
    def log_likelihood(self, data, latent_var, params):
        # 1. 파라미터 추출
        zeta = params['zeta']  # (n_indicators,)
        tau = params['tau']    # (n_indicators, n_thresholds)
        
        # 2. 선형 예측
        linear_pred = zeta * latent_var  # (n_obs, n_indicators)
        
        # 3. Ordered Probit 확률 계산
        for i, indicator in enumerate(self.indicators):
            y = data[indicator].values
            probs = self._ordered_probit_prob(y, linear_pred[:, i], tau[i])
            ll += np.log(probs + 1e-10)
```
</augment_code_snippet>

### GPU 변환 가능 부분

#### ✅ 높은 효과 예상
1. **선형 예측 계산**
   ```python
   # CPU (NumPy)
   linear_pred = zeta * latent_var  # (326, 38)
   
   # GPU (CuPy)
   import cupy as cp
   zeta_gpu = cp.array(zeta)
   lv_gpu = cp.array(latent_var)
   linear_pred_gpu = zeta_gpu * lv_gpu
   ```
   - **연산량**: 326명 × 38지표 × 100 draws = 1,238,800
   - **GPU 효과**: 10-50배 속도 향상

2. **정규분포 CDF 계산 (Φ)**
   ```python
   # CPU (scipy)
   from scipy.stats import norm
   prob = norm.cdf(upper) - norm.cdf(lower)
   
   # GPU (CuPy)
   from cupyx.scipy.special import ndtr  # CDF of standard normal
   prob_gpu = ndtr(upper_gpu) - ndtr(lower_gpu)
   ```
   - **연산량**: 326 × 38 × 5 (카테고리) × 100 = 6,194,000
   - **GPU 효과**: 20-100배 속도 향상

#### ⚠️ 낮은 효과 예상
1. **개인별 순차 처리**
   - 현재 구조: 326명을 순차적으로 처리
   - GPU는 대규모 병렬 연산에 적합
   - 작은 배치 반복은 CPU-GPU 전송 오버헤드 큼

2. **데이터 전송 오버헤드**
   ```python
   # CPU → GPU 전송
   data_gpu = cp.array(data)  # 시간 소요
   
   # GPU → CPU 전송
   result = cp.asnumpy(result_gpu)  # 시간 소요
   ```
   - 326명 × 100 draws = 32,600회 전송
   - 전송 시간이 계산 시간보다 클 수 있음

### GPU 변환 전략

#### 전략 A: 배치 처리 (권장)
```python
class GPUOrderedProbitMeasurement:
    def log_likelihood_batch(self, data_batch, latent_vars_batch, params):
        """
        여러 개인을 한번에 GPU로 처리
        
        Args:
            data_batch: (n_persons, n_obs_per_person, n_indicators)
            latent_vars_batch: (n_persons, n_draws)
            params: 파라미터
        """
        import cupy as cp
        
        # 한번에 GPU로 전송
        data_gpu = cp.array(data_batch)
        lv_gpu = cp.array(latent_vars_batch)
        zeta_gpu = cp.array(params['zeta'])
        tau_gpu = cp.array(params['tau'])
        
        # 배치 계산
        linear_pred = cp.einsum('ij,ik->ijk', lv_gpu, zeta_gpu)
        
        # Ordered Probit 확률 (벡터화)
        probs = self._ordered_probit_prob_vectorized(
            data_gpu, linear_pred, tau_gpu
        )
        
        # 로그우도
        ll = cp.sum(cp.log(probs + 1e-10), axis=(1, 2))
        
        # CPU로 반환
        return cp.asnumpy(ll)
```

**장점**:
- CPU-GPU 전송 최소화 (326회 → 1회)
- GPU 병렬 처리 최대 활용
- 예상 속도 향상: 10-30배

**단점**:
- 메모리 사용량 증가 (8GB VRAM 제한)
- 코드 복잡도 증가

#### 전략 B: 핵심 연산만 GPU (간단)
```python
class HybridOrderedProbitMeasurement:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy as cp
            self.xp = cp
        else:
            import numpy as np
            self.xp = np
    
    def _compute_probs(self, linear_pred, tau):
        """정규분포 CDF 계산만 GPU 사용"""
        if self.use_gpu:
            from cupyx.scipy.special import ndtr
            return ndtr(linear_pred)
        else:
            from scipy.stats import norm
            return norm.cdf(linear_pred)
```

**장점**:
- 최소한의 코드 변경
- 안정성 높음
- 예상 속도 향상: 2-5배

**단점**:
- GPU 효과 제한적
- 여전히 CPU-GPU 전송 오버헤드

## 3. 성능 예상

### 현재 (CPU 병렬)
- **1회 우도 계산**: 2.5초
- **1000회 반복**: 42분
- **병렬화**: 27 코어

### GPU 변환 후 (전략 A)
- **1회 우도 계산**: 0.5-1.0초 (예상)
- **1000회 반복**: 8-17분 (예상)
- **속도 향상**: 2.5-5배 (CPU 병렬 대비)

### GPU 변환 후 (전략 B)
- **1회 우도 계산**: 1.5-2.0초 (예상)
- **1000회 반복**: 25-33분 (예상)
- **속도 향상**: 1.3-1.7배 (CPU 병렬 대비)

## 4. 메모리 분석

### GPU VRAM 사용량 (전략 A)

```python
# 데이터 크기
n_persons = 326
n_obs_per_person = 18
n_indicators = 38
n_draws = 100
n_categories = 5

# 배열 크기 (float32 기준)
data_size = n_persons * n_obs_per_person * n_indicators * 4  # 0.9 MB
lv_size = n_persons * n_draws * 4  # 0.13 MB
linear_pred_size = n_persons * n_draws * n_indicators * 4  # 4.9 MB
probs_size = n_persons * n_obs_per_person * n_indicators * n_categories * 4  # 4.5 MB

# 총 VRAM 사용량
total_vram = data_size + lv_size + linear_pred_size + probs_size
# ≈ 10.5 MB (매우 작음!)
```

**결론**: VRAM 충분 (8GB 중 10MB만 사용)

## 5. 구현 우선순위

### 즉시 실행 (권장)
1. ✅ **CPU 병렬처리로 모델 추정**
   - 이미 구현됨
   - 충분히 빠름 (42분)
   - 안정적

### 단기 (1-2주)
2. ⏸️ **CUDA Toolkit 설치**
   - cuRAND DLL 문제 해결
   - CuPy 완전 작동 확인

3. ⏸️ **전략 B 구현 (간단)**
   - 핵심 연산만 GPU
   - 최소한의 코드 변경
   - 2-5배 속도 향상

### 중기 (1-2개월)
4. ⏸️ **전략 A 구현 (최적)**
   - 배치 처리
   - 10-30배 속도 향상
   - 코드 재구성 필요

### 장기 (3-6개월)
5. ⏸️ **JAX 기반 재구현**
   - 자동 미분
   - GPU 최적화
   - 100배 이상 속도 향상

## 6. 권장 사항

### 현재 상황
- ✅ CuPy 설치 완료
- ⚠️ CUDA Toolkit 미설치
- ✅ CPU 병렬처리 작동

### 권장 순서
1. **먼저 CPU 병렬처리로 모델 추정 완료**
   - 결과 검증
   - 파라미터 해석
   - 논문 작성

2. **필요 시 GPU 최적화**
   - CUDA Toolkit 설치
   - 전략 B 구현 (간단)
   - 성능 비교

3. **대규모 데이터 시 전략 A**
   - 1000명 이상
   - 배치 처리 구현

## 7. 다음 단계

### 즉시
```bash
# CPU 병렬처리로 모델 추정
python scripts/test_multi_latent_iclv.py
```

### CUDA Toolkit 설치 (선택)
1. https://developer.nvidia.com/cuda-downloads
2. CUDA 12.8 다운로드 및 설치
3. 환경변수 설정
4. CuPy 테스트

### GPU 변환 (선택)
1. 전략 B 구현 (간단)
2. 성능 비교
3. 효과 있으면 전략 A

## 결론

**현재 최선의 선택**: CPU 병렬처리 (27 코어)
- 이미 구현됨
- 충분히 빠름 (42분)
- 안정적
- 추가 작업 불필요

**GPU 활용**: 나중에 고려
- CUDA Toolkit 설치 필요
- 속도 향상 제한적 (2-5배)
- 구현 복잡도 증가
- 현재 우선순위 낮음

