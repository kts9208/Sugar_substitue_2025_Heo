# GPU Analytic Gradient 문제점 분석

## 🔍 Executive Summary

GPU Analytic Gradient 구현에서 발견된 **7가지 주요 문제점**:

| # | 문제 | 심각도 | 영향 | 상태 |
|---|------|--------|------|------|
| 1 | **Importance Weighting 누락** | 🔴 Critical | NaN, 잘못된 그래디언트 | 미구현 |
| 2 | **측정모델: 첫 번째 행만 사용** | 🔴 Critical | 대부분 데이터 무시 | 버그 |
| 3 | **단순 합산 (가중평균 아님)** | 🔴 Critical | 수학적 오류 | 버그 |
| 4 | **구조모델: 첫 번째 행만 사용** | 🟡 Major | 공변량 정보 손실 | 버그 |
| 5 | **선택모델: 순차 처리** | 🟡 Major | GPU 미활용 | 비효율 |
| 6 | **Likelihood 계산 누락** | 🔴 Critical | Weighting 불가능 | 미구현 |
| 7 | **수치 안정성 부족** | 🟡 Major | NaN 발생 가능 | 부분 구현 |

**결론**: 현재 구현은 **수학적으로 잘못되었으며**, 수정 없이는 사용 불가능합니다.

---

## 📋 문제점 상세 분석

### 문제 1: Importance Weighting 누락 🔴

#### 현재 구현 (GPU)

```python
# gpu_gradient_batch.py, Line 126-127
grad_zeta = cp.asnumpy(grad_zeta_batch.sum(axis=0))  # ❌ 단순 합산
grad_tau = cp.asnumpy(grad_tau_batch.sum(axis=0))    # ❌ 단순 합산
```

#### 올바른 구현 (CPU)

```python
# multi_latent_gradient.py, Line 379-384
# Importance weights 계산
total_likelihood = sum(draw_likelihoods)
weights = np.array(draw_likelihoods) / total_likelihood

# 가중평균 그래디언트 계산
for w, grad in zip(weights, draw_gradients):
    weighted_meas[lv_name]['grad_zeta'] += w * grad['measurement'][lv_name]['grad_zeta']
```

#### 문제점

**GPU 버전은 모든 draws의 그래디언트를 단순 합산합니다:**
```
grad_GPU = Σᵢ grad_i  # ❌ 잘못됨
```

**올바른 방법은 importance weighting입니다:**
```
grad_correct = Σᵢ wᵢ · grad_i  # ✅ 올바름
where wᵢ = L_i / Σⱼ L_j
```

#### 영향

1. **수학적 오류**: 시뮬레이션 기반 추정의 핵심 원리 위반
2. **편향된 그래디언트**: 우도가 낮은 draws도 동일한 가중치
3. **NaN 발생**: 극단적인 그래디언트 값 누적
4. **수렴 실패**: BFGS가 잘못된 방향으로 이동

#### 수정 방법

```python
# 1. 각 draw의 likelihood 계산
ll_batch = compute_likelihood_batch_gpu(ind_data, lvs_list, params)  # (n_draws,)

# 2. Importance weights 계산
weights = cp.exp(ll_batch) / cp.sum(cp.exp(ll_batch))  # (n_draws,)

# 3. 가중평균
grad_zeta = cp.asnumpy(cp.sum(weights[:, None] * grad_zeta_batch, axis=0))
grad_tau = cp.asnumpy(cp.sum(weights[:, None, None] * grad_tau_batch, axis=0))
```

---

### 문제 2: 측정모델 - 첫 번째 행만 사용 🔴

#### 현재 구현

```python
# gpu_gradient_batch.py, Line 74
first_row = ind_data.iloc[0]  # ❌ 첫 번째 행만

for i, indicator in enumerate(config.indicators):
    if indicator not in first_row.index:  # ❌ 첫 번째 행에서만 확인
        continue
    
    y = first_row[indicator]  # ❌ 첫 번째 행의 값만
```

#### 올바른 구현 (CPU)

```python
# multi_latent_gradient.py, Line 101-130
for idx in range(len(data)):  # ✅ 모든 행 순회
    row = data.iloc[idx]
    
    for i, indicator in enumerate(indicators):
        if indicator not in row.index:
            continue
        
        y = row[indicator]  # ✅ 각 행의 값
```

#### 문제점

**개인의 선택 상황이 18개인데, 첫 번째만 사용:**
```
개인 데이터:
  Row 0: q1=3, q2=4, q3=2, ...  ✅ 사용됨
  Row 1: q1=4, q2=3, q3=5, ...  ❌ 무시됨
  Row 2: q1=2, q2=5, q3=3, ...  ❌ 무시됨
  ...
  Row 17: q1=5, q2=2, q3=4, ... ❌ 무시됨

→ 94.4% (17/18) 데이터 손실!
```

#### 영향

1. **정보 손실**: 대부분의 측정 데이터 무시
2. **편향된 추정**: 첫 번째 선택 상황에만 의존
3. **잘못된 그래디언트**: 측정모델 파라미터 업데이트 오류
4. **수렴 불가능**: 잘못된 정보로 최적화

#### 수정 방법

```python
# 모든 선택 상황에 대해 그래디언트 계산
for idx in range(len(ind_data)):
    row = ind_data.iloc[idx]
    
    for i, indicator in enumerate(config.indicators):
        if indicator not in row.index:
            continue
        
        y = row[indicator]
        if pd.isna(y):
            continue
        
        # 그래디언트 계산 (현재 코드와 동일)
        ...
```

---

### 문제 3: 단순 합산 (가중평균 아님) 🔴

#### 현재 구현

```python
# gpu_gradient_batch.py
# 측정모델 (Line 126-127)
grad_zeta = cp.asnumpy(grad_zeta_batch.sum(axis=0))  # ❌ sum
grad_tau = cp.asnumpy(grad_tau_batch.sum(axis=0))    # ❌ sum

# 구조모델 (Line 195-198)
grad_gamma_lv = cp.dot(exo_lv_gpu.T, residual) / error_variance  # ❌ dot (sum)
grad_gamma_x = residual.sum() / error_variance * X_gpu           # ❌ sum

# 선택모델 (Line 297-303)
grad_intercept_total += cp.sum(sign * mills).item()  # ❌ sum
grad_beta_total += cp.dot(attr_gpu.T, sign * mills)  # ❌ sum
grad_lambda_total += cp.sum(sign * mills * lv).item()  # ❌ sum
```

#### 올바른 구현 (CPU)

```python
# multi_latent_gradient.py, Line 421-434
for w, grad in zip(weights, draw_gradients):  # ✅ 가중치 사용
    # 측정모델
    weighted_meas[lv_name]['grad_zeta'] += w * grad['measurement'][lv_name]['grad_zeta']
    weighted_meas[lv_name]['grad_tau'] += w * grad['measurement'][lv_name]['grad_tau']
    
    # 구조모델
    weighted_struct['grad_gamma_lv'] += w * grad['structural']['grad_gamma_lv']
    weighted_struct['grad_gamma_x'] += w * grad['structural']['grad_gamma_x']
    
    # 선택모델
    weighted_choice['grad_intercept'] += w * grad['choice']['grad_intercept']
    weighted_choice['grad_beta'] += w * grad['choice']['grad_beta']
    weighted_choice['grad_lambda'] += w * grad['choice']['grad_lambda']
```

#### 문제점

**시뮬레이션 기반 추정의 핵심 원리:**

```
E[∇ log L] = ∫ ∇ log L(θ|η) · f(η) dη
           ≈ (1/R) Σᵣ ∇ log L(θ|ηᵣ)  # ❌ 단순 평균 (Monte Carlo)
           ≈ Σᵣ wᵣ · ∇ log L(θ|ηᵣ)   # ✅ 가중평균 (Importance Sampling)

where wᵣ = L(θ|ηᵣ) / Σₛ L(θ|ηₛ)
```

**GPU 버전은 가중치 없이 합산:**
```
grad_GPU = Σᵣ grad_r  # ❌ 잘못됨
```

**올바른 방법:**
```
grad_correct = Σᵣ wᵣ · grad_r  # ✅ 올바름
```

#### 영향

1. **이론적 오류**: Importance sampling 원리 위반
2. **편향된 추정**: 모든 draws를 동등하게 취급
3. **비효율적**: 우도가 낮은 draws도 동일한 영향
4. **수렴 문제**: 잘못된 그래디언트 방향

---

### 문제 4: 구조모델 - 첫 번째 행만 사용 🟡

#### 현재 구현

```python
# gpu_gradient_batch.py, Line 178-179
first_row = ind_data.iloc[0]  # ❌ 첫 번째 행만
X = np.array([first_row[cov] if cov in first_row.index else 0.0 for cov in covariates])
```

#### 문제점

**공변량은 개인별로 동일하므로 큰 문제는 아니지만:**
- 첫 번째 행에 공변량이 없으면 0으로 처리
- 다른 행에 공변량이 있어도 무시

#### 영향

- **중간 수준**: 공변량은 보통 모든 행에 동일하게 존재
- **잠재적 버그**: 데이터 구조에 따라 문제 발생 가능

#### 수정 방법

```python
# 개인 수준 공변량 추출 (첫 번째 행 사용은 OK, 하지만 명시적으로)
# 또는 ind_data에서 개인 ID로 그룹화하여 추출
person_data = ind_data.groupby('person_id').first()  # 더 안전
X = np.array([person_data[cov] for cov in covariates])
```

---

### 문제 5: 선택모델 - 순차 처리 🟡

#### 현재 구현

```python
# gpu_gradient_batch.py, Line 275-303
for draw_idx in range(n_draws):  # ❌ 순차 처리
    lv = lv_gpu[draw_idx]
    
    V = intercept + cp.dot(attr_gpu, beta_gpu) + lambda_lv * lv
    prob = cp_ndtr(V)
    phi = cp_norm_pdf(V)
    
    # 그래디언트 누적
    grad_intercept_total += cp.sum(sign * mills).item()
    grad_beta_total += cp.dot(attr_gpu.T, sign * mills)
    grad_lambda_total += cp.sum(sign * mills * lv).item()
```

#### 문제점

**GPU의 장점을 활용하지 못함:**
- 100개 draws를 순차적으로 처리
- GPU 병렬 처리 미활용

#### 올바른 구현 (배치)

```python
# 모든 draws를 한 번에 처리
lv_batch = lv_gpu[:, None]  # (n_draws, 1)
attr_batch = attr_gpu[None, :, :]  # (1, n_situations, n_attributes)

# Broadcasting으로 모든 draws 동시 계산
V_batch = intercept + cp.dot(attr_batch, beta_gpu) + lambda_lv * lv_batch
# Shape: (n_draws, n_situations)

prob_batch = cp_ndtr(V_batch)
phi_batch = cp_norm_pdf(V_batch)

# 배치 그래디언트 계산
mills_batch = phi_batch / prob_batch
grad_intercept_batch = cp.sum(sign * mills_batch, axis=1)  # (n_draws,)
grad_beta_batch = cp.dot(mills_batch.T, attr_gpu)  # (n_draws, n_attributes)
grad_lambda_batch = cp.sum(sign * mills_batch * lv_batch, axis=1)  # (n_draws,)

# Importance weighting 적용
grad_intercept = cp.sum(weights * grad_intercept_batch)
grad_beta = cp.dot(weights, grad_beta_batch)
grad_lambda = cp.sum(weights * grad_lambda_batch)
```

#### 영향

- **성능 저하**: GPU 병렬 처리 미활용
- **속도**: 현재 구현도 느림 (순차 처리)

---

### 문제 6: Likelihood 계산 누락 🔴

#### 현재 구현

```python
# gpu_gradient_batch.py
# Likelihood 계산 코드 없음!
# Importance weights를 계산할 수 없음
```

#### 필요한 구현

```python
def compute_likelihood_batch_gpu(
    ind_data: pd.DataFrame,
    lvs_list: List[Dict[str, float]],
    params: Dict
) -> cp.ndarray:
    """
    각 draw의 likelihood 계산 (importance weighting용)
    
    Returns:
        (n_draws,) array of log-likelihoods
    """
    n_draws = len(lvs_list)
    ll_batch = cp.zeros(n_draws)
    
    for draw_idx in range(n_draws):
        lv = lvs_list[draw_idx]
        
        # 측정모델 우도
        ll_meas = compute_measurement_ll_gpu(ind_data, lv, params['measurement'])
        
        # 구조모델 우도
        ll_struct = compute_structural_ll_gpu(ind_data, lv, params['structural'])
        
        # 선택모델 우도
        ll_choice = compute_choice_ll_gpu(ind_data, lv, params['choice'])
        
        # 결합 우도
        ll_batch[draw_idx] = ll_meas + ll_struct + ll_choice
    
    return ll_batch
```

#### 영향

1. **Importance weighting 불가능**: 가중치 계산 불가
2. **단순 합산만 가능**: 수학적으로 잘못된 방법
3. **NaN 발생**: 극단적인 그래디언트 누적

---

### 문제 7: 수치 안정성 부족 🟡

#### 현재 구현

```python
# gpu_gradient_batch.py
# 일부만 구현됨

# 측정모델 (Line 111)
prob = cp.clip(prob, 1e-10, 1 - 1e-10)  # ✅ 구현됨

# 선택모델 (Line 283)
prob = cp.clip(prob, 1e-10, 1 - 1e-10)  # ✅ 구현됨

# 하지만:
# - log-likelihood 계산 시 log(0) 방지 없음
# - exp overflow/underflow 방지 없음
# - NaN 체크 없음
```

#### 필요한 개선

```python
# 1. Log-sum-exp trick for importance weights
def log_sum_exp(log_values):
    max_val = cp.max(log_values)
    return max_val + cp.log(cp.sum(cp.exp(log_values - max_val)))

ll_batch = compute_likelihood_batch_gpu(...)
log_weights = ll_batch - log_sum_exp(ll_batch)
weights = cp.exp(log_weights)

# 2. NaN 체크
if cp.any(cp.isnan(grad_zeta)):
    logger.warning("NaN detected in grad_zeta")
    grad_zeta = cp.nan_to_num(grad_zeta, nan=0.0)

# 3. Gradient clipping
grad_zeta = cp.clip(grad_zeta, -1e6, 1e6)
```

---

## 📊 문제점 요약표

| 문제 | CPU 구현 | GPU 구현 | 차이점 |
|------|---------|---------|--------|
| **Importance weighting** | ✅ 구현됨 | ❌ 누락 | GPU는 단순 합산 |
| **측정모델 데이터** | ✅ 모든 행 | ❌ 첫 행만 | 94% 데이터 손실 |
| **가중평균** | ✅ 가중평균 | ❌ 단순 합산 | 수학적 오류 |
| **구조모델 공변량** | ✅ 올바름 | ⚠️ 첫 행만 | 잠재적 버그 |
| **선택모델 배치** | ❌ 순차 | ❌ 순차 | 둘 다 비효율 |
| **Likelihood 계산** | ✅ 구현됨 | ❌ 누락 | Weighting 불가 |
| **수치 안정성** | ✅ 완전 | ⚠️ 부분 | NaN 위험 |

---

## 🔧 수정 우선순위

### Priority 1 (Critical) - 즉시 수정 필요

1. **Importance weighting 구현**
   - Likelihood 계산 함수 추가
   - Weights 계산 및 적용
   - 예상 작업: 2-3시간

2. **측정모델 모든 행 처리**
   - Loop 추가하여 모든 선택 상황 처리
   - 예상 작업: 1시간

3. **가중평균으로 변경**
   - 모든 sum을 weighted sum으로 변경
   - 예상 작업: 1시간

### Priority 2 (Major) - 성능 개선

4. **선택모델 배치 처리**
   - For loop 제거, broadcasting 사용
   - 예상 작업: 2시간

5. **수치 안정성 강화**
   - Log-sum-exp trick
   - NaN 체크 및 처리
   - Gradient clipping
   - 예상 작업: 1-2시간

### Priority 3 (Minor) - 코드 품질

6. **구조모델 공변량 추출 개선**
   - 더 안전한 방법 사용
   - 예상 작업: 30분

---

## 📈 수정 후 예상 성능

| 항목 | 현재 (버그) | 수정 후 |
|------|------------|---------|
| **정확도** | ❌ 잘못됨 | ✅ 올바름 |
| **속도** | NaN 에러 | ~22초/그래디언트 |
| **안정성** | ❌ 불안정 | ✅ 안정적 |
| **수렴** | ❌ 실패 | ✅ 성공 예상 |

---

## 💡 결론

**현재 GPU Analytic Gradient 구현은 사용 불가능합니다.**

**주요 이유:**
1. Importance weighting 누락 → 수학적 오류
2. 대부분 데이터 무시 → 정보 손실
3. 단순 합산 → 이론적 오류

**수정 예상 시간:** 8-12시간

**수정 후 이득:** 77분 → 22초 (210배 향상)

**권장사항:**
- 단기: Numerical gradient 사용 (안정적)
- 중기: GPU gradient 수정 (1-2주)
- 장기: 하이브리드 접근

