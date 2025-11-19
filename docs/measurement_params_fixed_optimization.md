# 측정모델 파라미터 고정 시 최적화

## 📋 개요

동시추정에서 **측정모델 파라미터를 고정**하는 경우, 불필요한 계산을 제거하여 성능을 크게 향상시킬 수 있습니다.

### 고정되는 파라미터

1. **ζ (zeta)**: 요인적재량 (Factor Loadings)
2. **σ² (sigma_sq)**: 오차분산 (Error Variance)

두 파라미터 모두 순차추정 1단계(SEM)에서 추정된 값을 사용합니다.

## 🔴 기존 문제점

측정모델 파라미터가 고정되어 있음에도 불구하고:

1. **우도 계산**: 매 iteration마다 측정모델 우도를 재계산
2. **그래디언트 계산**: 매 iteration마다 측정모델 그래디언트를 계산 (항상 0)
3. **메모리 낭비**: 동일한 값을 반복적으로 계산하여 메모리와 시간 낭비

### 예시: 100 iterations, 1000 individuals, 100 draws

- **측정모델 우도 계산 횟수**: 100 × 1000 × 100 = **10,000,000회**
- **측정모델 그래디언트 계산 횟수**: 100 × 1000 × 100 = **10,000,000회**

하지만 파라미터가 고정되어 있으므로:
- 우도는 **최초 1회만** 계산하면 됨
- 그래디언트는 **계산할 필요 없음** (항상 0)

## ✅ 최적화 방안

### 1. 측정모델 우도 캐싱

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_gpu_batch_estimator.py`

```python
# ✅ 측정모델 우도: 파라미터 고정 시 캐싱
if self._measurement_params_fixed:
    # 캐시 키: (개인 ID, draw 인덱스)
    cache_key = (ind_id, j)
    
    if self._cached_measurement_ll is None:
        self._cached_measurement_ll = {}
    
    if cache_key not in self._cached_measurement_ll:
        # 최초 1회만 계산
        ll_measurement = measurement_model.log_likelihood(
            ind_data, lv, param_dict['measurement']
        )
        self._cached_measurement_ll[cache_key] = ll_measurement
    else:
        # 캐시에서 가져오기
        ll_measurement = self._cached_measurement_ll[cache_key]
else:
    # 파라미터가 변하므로 매번 계산
    ll_measurement = measurement_model.log_likelihood(
        ind_data, lv, param_dict['measurement']
    )
```

### 2. 측정모델 그래디언트 계산 스킵

**파일**: `src/analysis/hybrid_choice_model/iclv_models/multi_latent_gradient.py`

```python
# ✅ 측정모델 파라미터 고정 시 그래디언트 계산 스킵
if self.measurement_params_fixed:
    # 측정모델 그래디언트를 0으로 설정 (파라미터 고정)
    grad_meas = {}
    for lv_name in self.measurement_grad.lv_names:
        config = self.measurement_grad.measurement_configs[lv_name]
        measurement_method = getattr(config, 'measurement_method', 'ordered_probit')
        
        n_ind = len(config.indicators)
        grad_meas[lv_name] = {'grad_zeta': np.zeros(n_ind)}
        
        if measurement_method == 'continuous_linear':
            grad_meas[lv_name]['grad_sigma_sq'] = np.zeros(n_ind)
        else:
            n_thresh = config.n_categories - 1
            grad_meas[lv_name]['grad_tau'] = np.zeros((n_ind, n_thresh))
else:
    # 파라미터가 변하므로 그래디언트 계산
    grad_meas = self.measurement_grad.compute_gradient(
        ind_data, latent_vars, params_dict['measurement']
    )
```

## 🚀 성능 향상 예상

### 계산 복잡도 비교

| 항목 | 기존 | 최적화 후 | 개선율 |
|------|------|----------|--------|
| 측정모델 우도 계산 | O(N × R × I) | O(N × R) | **~I배** |
| 측정모델 그래디언트 계산 | O(N × R × I) | O(1) | **~무한대** |

- N: 개인 수
- R: Halton draws 수
- I: Iteration 수

### 실제 예시 (N=1000, R=100, I=100)

- **우도 계산**: 10,000,000회 → 100,000회 (**100배 감소**)
- **그래디언트 계산**: 10,000,000회 → 0회 (**완전 제거**)

## 📝 사용 방법

### 자동 감지

`SimultaneousGPUBatchEstimator`는 초기 파라미터에 `measurement` 키가 있으면 자동으로 측정모델 파라미터 고정 모드를 활성화합니다.

```python
# estimate() 호출 시 initial_params에 measurement 포함
estimator.estimate(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_model=choice_model,
    initial_params=initial_params  # {'measurement': {...}, 'structural': {...}, 'choice': {...}}
)
```

### 로그 확인

최적화가 활성화되면 다음과 같은 로그가 출력됩니다:

```
✅ 측정모델 파라미터 고정 모드: 우도를 최초 1회만 계산하고 캐싱합니다.
✅ 측정모델 파라미터 고정: 그래디언트 계산 스킵
```

## 🔍 검증

### 우도 값 확인

최적화 전후 우도 값이 동일한지 확인:

```python
# 최적화 전
ll_before = estimator._joint_log_likelihood(params, ...)

# 최적화 후
ll_after = estimator._joint_log_likelihood(params, ...)

assert np.isclose(ll_before, ll_after)
```

### 그래디언트 값 확인

측정모델 그래디언트가 0인지 확인:

```python
gradients = joint_grad.compute_gradients(...)

for ind_grad in gradients:
    for lv_name, grad in ind_grad['measurement'].items():
        assert np.allclose(grad['grad_zeta'], 0.0)
        if 'grad_sigma_sq' in grad:
            assert np.allclose(grad['grad_sigma_sq'], 0.0)
```

## 📌 주의사항

1. **측정모델 파라미터가 실제로 고정되어 있는지 확인**
   - 순차추정 1단계에서 추정된 값을 사용하는 경우에만 적용
   - ζ (요인적재량)와 σ² (오차분산) 모두 순차추정에서 로드됨

2. **캐시 메모리 사용량**
   - 개인 수 × draws 수만큼 캐시 저장
   - 메모리가 부족한 경우 주의

3. **디버깅 시**
   - 최적화를 비활성화하려면 `_measurement_params_fixed = False`로 설정

## 📝 순차추정에서 측정모델 파라미터 로드

### lavaan 결과에서 파라미터 추출

**파일**: `scripts/test_gpu_batch_iclv.py`

```python
# ✅ zeta (요인적재량) 추출
# lavaan에서 '~' 연산자로 표현됨
row = meas_params[(meas_params['lval'] == indicator) &
                 (meas_params['op'] == '~') &
                 (meas_params['rval'] == lv_name)]

# ✅ sigma_sq (오차분산) 추출
# lavaan에서 '~~' 연산자로 표현됨 (자기 자신과의 공분산)
row = meas_params[(meas_params['lval'] == indicator) &
                 (meas_params['op'] == '~~') &
                 (meas_params['rval'] == indicator)]
```

### 로드 확인 로그

```
[INFO] health_concern 측정모델 파라미터 로드:
  - zeta (요인적재량): [1.0, 0.95, 0.98, ...]
  - sigma_sq (오차분산): [0.61, 0.50, 0.49, ...]
```

