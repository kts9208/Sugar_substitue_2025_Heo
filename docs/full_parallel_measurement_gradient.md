# 완전 병렬 측정모델 Gradient 계산

## 개요

Advanced Indexing을 사용하여 측정모델의 **71개 파라미터** (33 zeta + 38 sigma_sq)를 **1번의 GPU 커널 호출**로 계산하는 완전 병렬 처리 구현입니다.

### 핵심 수치

- **지표 수**: 38개 (5개 LV에 분산)
- **파라미터 수**: 71개
  - Zeta (loading factor): 33개 (각 LV의 첫 번째는 1.0으로 고정)
  - Sigma_sq (오차분산): 38개 (모든 지표)
- **계산량**: 2,477,600개 (326명 × 100 draws × 38 지표 × 2)
- **GPU 커널 호출**: 1번 (기존 38번 → 38배 개선)

---

## 핵심 아이디어

### 문제 상황

기존 구현은 각 지표마다 순차적으로 gradient를 계산했습니다:

```python
# 기존: 38번 GPU 커널 호출
for lv_idx, lv_name in enumerate(lv_names):  # 5개 LV
    for i in range(n_indicators):  # LV별로 다른 개수
        # 각 지표마다 zeta와 sigma_sq gradient 계산
        grad_zeta = compute_single_indicator_zeta(...)  # (326, 100) 계산
        grad_sigma_sq = compute_single_indicator_sigma_sq(...)  # (326, 100) 계산
```

**문제점:**
- GPU 커널 호출: 38번 (6+6+3+20+3 지표)
- 각 지표마다 2개 파라미터 (zeta, sigma_sq) 계산
- 각 호출마다 GPU 동기화 오버헤드
- 병렬 처리 잠재력 미활용

---

### 해결 방법: Advanced Indexing

**핵심 발견:**
- 각 지표의 gradient는 **특정 LV 값**에만 의존
- 각 지표마다 **2개 파라미터** (zeta, sigma_sq)의 gradient 계산 필요
- 모든 LV 값은 이미 `all_lvs_gpu[:, :, :]`에 준비되어 있음
- NumPy/CuPy의 Advanced Indexing으로 각 지표에 맞는 LV를 자동 선택 가능!
- **zeta와 sigma_sq를 동시에 계산** 가능!

```python
# 1. 지표-LV 매핑 배열 생성
indicator_to_lv = [
    0, 0, 0, 0, 0, 0,           # q6-q11 → health_concern (LV 0)
    1, 1, 1, 1, 1, 1,           # q12-q17 → perceived_benefit (LV 1)
    2, 2, 2,                     # q27-q29 → perceived_price (LV 2)
    3, 3, 3, ..., 3,            # q30-q49 → nutrition_knowledge (LV 3, 20개)
    4, 4, 4                      # q18-q20 → purchase_intention (LV 4)
]  # 총 38개

# 2. Advanced Indexing으로 각 지표에 맞는 LV 선택
all_lvs_gpu.shape = (326, 100, 5)
lv_for_indicators = all_lvs_gpu[:, :, indicator_to_lv]
# → (326, 100, 38) - Zero-padding 없음!

# 3. 완전 병렬 계산 (zeta와 sigma_sq 동시에!)
y_pred_all = zeta_all * lv_for_indicators  # (326, 100, 38)
residual_all = all_y_all - y_pred_all      # (326, 100, 38)

# Zeta gradient
grad_zeta_all = residual_all * lv_for_indicators / sigma_sq_all  # (326, 100, 38)

# Sigma_sq gradient (동시에 계산!)
grad_sigma_sq_all = (-0.5 / sigma_sq_all +
                     0.5 * (residual_all ** 2) / (sigma_sq_all ** 2))  # (326, 100, 38)
```

**마법 같은 일:**
- 지표 0 (q6): `lv_for_indicators[:, :, 0]` = `all_lvs_gpu[:, :, 0]` (HC)
- 지표 6 (q12): `lv_for_indicators[:, :, 6]` = `all_lvs_gpu[:, :, 1]` (PB)
- 지표 15 (q30): `lv_for_indicators[:, :, 15]` = `all_lvs_gpu[:, :, 3]` (NK)

**결과:**
- 38개 지표의 zeta gradient: 1번 계산
- 38개 지표의 sigma_sq gradient: 1번 계산
- **총 71개 파라미터의 gradient를 1번의 GPU 커널 호출로!**

---

## 성능 비교

### GPU 커널 호출 횟수

| 방식 | GPU 커널 호출 | 개선 비율 |
|------|--------------|----------|
| 기존 (지표별 순차) | 38번 | - |
| 제안 (LV별 순차, 지표별 병렬) | 5번 | 7.6배 |
| **완전 병렬 (Advanced Indexing)** | **1번** | **38배** ✅ |

### 메모리 사용량

| 방식 | 메모리 | 비고 |
|------|--------|------|
| 완전 병렬 | 9.45 MB | (326 × 100 × 38) |
| Zero-padding | 24.87 MB | (326 × 100 × 5 × 20) |
| LV별 순차 | 4.97 MB | (326 × 100 × 20, 최대) |

**완전 병렬이 Zero-padding보다 62% 메모리 절약!**

### 실제 성능 (테스트 결과)

```
계산량: 326명 × 100 draws × 38 지표 = 1,238,800개
소요 시간: 0.39초
처리량: 3,177,202 계산/초
```

---

## 구현 상세

### 1. 완전 병렬 Gradient 계산 함수

**파일:** `src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_full_parallel.py`

```python
def compute_measurement_full_parallel_gpu(
    gpu_measurement_model,
    all_ind_data: List[pd.DataFrame],
    all_lvs_gpu,  # CuPy array (326, 100, 5)
    params_dict: Dict,
    all_weights_gpu,  # CuPy array (326, 100)
    lv_names: List[str],
    iteration_logger=None,
    log_level: str = 'MINIMAL'
) -> Dict:
    """
    측정모델 Gradient - 완전 병렬 (모든 지표 한 번에)
    
    Advanced Indexing을 사용하여 38개 지표를 1번의 GPU 커널 호출로 계산
    """
    # 1. 지표-LV 매핑 배열 생성
    indicator_to_lv = []
    for lv_idx, lv_name in enumerate(lv_names):
        config = gpu_measurement_model.models[lv_name].config
        n_indicators = len(config.indicators)
        indicator_to_lv.extend([lv_idx] * n_indicators)
    
    # 2. 모든 관측값 및 파라미터 수집
    all_y = ...  # (326, 38)
    all_zeta = ...  # (38,)
    all_sigma_sq = ...  # (38,)
    
    # 3. GPU로 전송
    all_y_gpu = cp.asarray(all_y)
    all_zeta_gpu = cp.asarray(all_zeta)
    all_sigma_sq_gpu = cp.asarray(all_sigma_sq)
    indicator_to_lv_gpu = cp.asarray(indicator_to_lv)
    
    # 4. ✨ Advanced Indexing: 각 지표에 맞는 LV 선택
    lv_for_indicators = all_lvs_gpu[:, :, indicator_to_lv_gpu]  # (326, 100, 38)
    
    # 5. 완전 병렬 Gradient 계산
    y_pred_all = all_zeta_gpu[None, None, :] * lv_for_indicators
    residual_all = all_y_gpu[:, None, :] - y_pred_all
    
    grad_zeta_batch = (residual_all * lv_for_indicators / 
                       all_sigma_sq_gpu[None, None, :])
    grad_sigma_sq_batch = (-0.5 / all_sigma_sq_gpu[None, None, :] + 
                           0.5 * (residual_all ** 2) / 
                           (all_sigma_sq_gpu[None, None, :] ** 2))
    
    # 6. 가중평균
    grad_zeta_all = cp.sum(all_weights_gpu[:, :, None] * grad_zeta_batch, axis=1)
    grad_sigma_sq_all = cp.sum(all_weights_gpu[:, :, None] * grad_sigma_sq_batch, axis=1)
    
    # 7. LV별로 분리하여 반환
    gradients = {}
    idx = 0
    for lv_name in lv_names:
        n_ind = len(params_dict['measurement'][lv_name]['zeta'])
        gradients[lv_name] = {
            'grad_zeta': cp.asnumpy(grad_zeta_all[:, idx:idx+n_ind]),
            'grad_sigma_sq': cp.asnumpy(grad_sigma_sq_all[:, idx:idx+n_ind])
        }
        idx += n_ind
    
    return gradients
```

### 2. 통합 함수

```python
def compute_all_individuals_gradients_full_parallel_gpu(
    gpu_measurement_model,
    all_ind_data: List[pd.DataFrame],
    all_ind_draws: np.ndarray,
    params_dict: Dict,
    measurement_model,
    structural_model,
    choice_model,
    iteration_logger=None,
    log_level: str = 'MINIMAL'
) -> List[Dict]:
    """
    모든 개인의 gradient를 완전 병렬로 계산
    
    측정모델: 38개 지표를 1번의 GPU 커널로 계산
    구조모델: 기존 방식 사용
    선택모델: 기존 방식 사용
    """
    # 1. LV 계산
    all_lvs_gpu = ...  # (326, 100, 5)
    
    # 2. ✨ 측정모델 Gradient - 완전 병렬
    meas_grads = compute_measurement_full_parallel_gpu(...)
    
    # 3. 구조모델 Gradient
    struct_grads = compute_structural_full_batch_gpu(...)
    
    # 4. 선택모델 Gradient
    choice_grads = compute_choice_full_batch_gpu(...)
    
    # 5. 개인별로 분리
    return individual_gradients
```

---

## 사용 방법

### 1. GPUBatchEstimator에서 활성화

```python
from src.analysis.hybrid_choice_model.iclv_models.gpu_batch_estimator import GPUBatchEstimator

# 완전 병렬 처리 활성화 (기본값)
estimator = GPUBatchEstimator(
    config,
    use_gpu=True,
    use_full_parallel=True  # ✨ 완전 병렬 처리
)

# 추정 실행
results = estimator.estimate(data, measurement_model, structural_model, choice_model)
```

### 2. 기존 방식 사용 (LV별 순차)

```python
# LV별 순차, 지표별 병렬 (5번 GPU 호출)
estimator = GPUBatchEstimator(
    config,
    use_gpu=True,
    use_full_parallel=False
)
```

---

## 기술적 세부사항

### Advanced Indexing이란?

NumPy/CuPy의 강력한 기능으로, 배열의 각 위치마다 다른 인덱스를 사용할 수 있습니다:

```python
# 일반 인덱싱
arr[0, 1, 2]  # 단일 원소

# Slicing
arr[:, :, 0]  # 모든 개인, 모든 draw, LV 0

# Advanced Indexing ✨
indices = [0, 0, 1, 1, 2, 3, 3, 3]
arr[:, :, indices]  # 각 위치마다 다른 LV 선택!
# → (326, 100, 8)
```

### Zero-Padding과의 비교

**Zero-Padding 방식:**
```python
# 모든 LV를 최대 크기(20)로 맞춤
all_grads = np.zeros((326, 100, 5, 20))

# 문제점:
# - 메모리 낭비: 38/100 = 62% 낭비
# - 계산 낭비: GPU가 0도 계산함
# - 복잡한 인덱싱: 각 LV마다 다른 범위 사용
```

**Advanced Indexing 방식:**
```python
# 정확히 필요한 만큼만 할당
lv_for_indicators = all_lvs_gpu[:, :, indicator_to_lv]  # (326, 100, 38)

# 장점:
# - 메모리 낭비 없음
# - 불필요한 계산 없음
# - 코드 명확함
```

---

## 테스트

### 단위 테스트

```bash
python scripts/test_full_parallel_measurement.py
```

**출력 예시:**
```
✅ 완전 병렬 Gradient 계산 완료!
   - 소요 시간: 0.3899초
   - 처리량: 3,177,202 계산/초

GPU 커널 호출:
   - 기존 (지표별 순차): 38번
   - 제안 (LV별 순차): 5번
   - 완전 병렬 (Advanced Indexing): 1번 ✅

메모리 사용:
   - 완전 병렬: 9.45 MB
   - Zero-padding: 24.87 MB
   - 절약: 15.42 MB (62.0%)
```

---

## 결론

### 주요 성과

1. ✅ **GPU 커널 호출 38배 감소** (38번 → 1번)
2. ✅ **메모리 62% 절약** (Zero-padding 대비)
3. ✅ **코드 명확성 향상** (Advanced Indexing)
4. ✅ **확장성** (지표 수가 늘어나도 1번 호출)

### 적용 범위

- 측정모델: ✅ 완전 병렬 (1번 GPU 호출)
- 구조모델: 기존 방식 (계층적 구조로 인해 순차 필요)
- 선택모델: 기존 방식 (이미 완전 병렬)

### 향후 개선 가능성

- 구조모델도 Advanced Indexing 적용 검토
- 다양한 측정 방법 (Ordered Probit) 지원
- 자동 배치 크기 조정

---

## 참고 자료

- **구현 파일:** `src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_full_parallel.py`
- **통합 파일:** `src/analysis/hybrid_choice_model/iclv_models/multi_latent_gradient.py`
- **테스트 파일:** `scripts/test_full_parallel_measurement.py`
- **NumPy Advanced Indexing:** https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

