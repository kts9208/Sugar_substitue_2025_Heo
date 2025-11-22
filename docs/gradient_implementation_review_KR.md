# 그래디언트 구현 디테일 검토 결과

## 📋 검토 항목

1. ✅ **측정모델 그래디언트 차원 합산** (grad_meas_LV)
2. ✅ **선택모델 그래디언트 부호** (grad_choice_LV)
3. ✅ **구조모델 오차분산 그래디언트** (sigma_sq_PB)

---

## ① 측정모델 그래디언트 차원 합산 ✅ **올바름**

### 검토 내용
> **질문**: 38개 지표에서 오는 그래디언트를 모두 더해야(Sum) LV에 대한 총 그래디언트가 되는가?

### 코드 위치
`src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`
- Line 46-142: `compute_measurement_grad_wrt_lv_gpu()`

### 구현 확인

```python
# Line 72: 초기화
grad_ll_wrt_lv = cp.zeros(n_draws)  # (n_draws,) 형태

# Line 94-111: 각 지표별로 누적 합산
for i, indicator in enumerate(config.indicators):
    if measurement_method == 'continuous_linear':
        # ∂LL/∂LV = ζ_i * (y_i - ζ_i*LV) / σ²_i
        grad_ll_wrt_lv += zeta_gpu[i] * residual / sigma_sq_gpu[i]
    else:
        # Ordered Probit
        # ∂LL/∂LV = (φ_upper - φ_lower) / P * (-ζ_i)
        grad_ll_wrt_lv += (phi_upper - phi_lower) / prob * (-zeta_gpu[i])

# Line 141: 반환
return cp.asnumpy(grad_ll_wrt_lv)  # (n_draws,)
```

### 수식 검증

**Continuous Linear 방식:**
```
Y_i = α_i + ζ_i × LV + ε_i,  ε_i ~ N(0, σ²_i)

log L = Σ_i log P(Y_i | LV)
      = Σ_i [-0.5 log(2π σ²_i) - 0.5 (Y_i - α_i - ζ_i × LV)² / σ²_i]

∂ log L / ∂LV = Σ_i ∂ log P(Y_i | LV) / ∂LV
               = Σ_i ζ_i × (Y_i - α_i - ζ_i × LV) / σ²_i
```

**Ordered Probit 방식:**
```
V_i = ζ_i × LV
P(Y_i = k) = Φ(τ_k - V_i) - Φ(τ_{k-1} - V_i)

∂ log P(Y_i = k) / ∂LV = (φ(τ_k - V_i) - φ(τ_{k-1} - V_i)) / P(Y_i = k) × (-ζ_i)

∂ log L / ∂LV = Σ_i ∂ log P(Y_i | LV) / ∂LV
```

### ✅ 결론
- **올바른 구현**: 각 지표별로 `+=` 연산자를 사용하여 누적 합산
- **차원 확인**: 최종 결과는 `(n_draws,)` 형태
- **논리 검증**: 하나의 LV가 여러 지표에 영향을 주므로, 모든 지표로부터의 그래디언트를 합산하는 것이 올바름

---

## ② 선택모델 그래디언트 부호 ✅ **올바름**

### 검토 내용
> **질문**: `theta_chosen - sum(P_j * theta_j)` 부호가 올바른가?

### 코드 위치
`src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`
- Line 2227-2477: `_compute_multinomial_logit_gradient_gpu()`

### 구현 확인

```python
# Line 2394-2399: Gradient 계산
diff = y_batch_gpu[:, None, :, :] - P_batch  # (y - P)
weighted_diff = all_weights_gpu[:, :, None, None] * diff  # w_r × (y - P)

# Line 2435-2453: Theta 그래디언트
for (alt_name, lv_name), theta_val in theta_params.items():
    grad_theta = cp.zeros(n_individuals)
    lv_idx = lv_names.index(lv_name)
    
    for ind_idx in range(n_individuals):
        for cs_idx in range(n_choice_sets):
            for alt_idx in range(3):
                if (대안이 alt_name과 일치):
                    lv_values = all_lvs_gpu[ind_idx, :, lv_idx]  # (R,)
                    grad_theta[ind_idx] += cp.sum(
                        weighted_diff[ind_idx, :, cs_idx, alt_idx] * lv_values
                    )
    
    gradients[f'theta_{alt_name}_{lv_name}'] = cp.asnumpy(grad_theta)
```

### 수식 검증

**Multinomial Logit 그래디언트:**
```
V_j = asc_j + β × X + θ_j × LV + γ_j × LV × X

P(j) = exp(V_j) / Σ_k exp(V_k)

log L = Σ_t log P(y_t)

∂ log L / ∂θ_j = Σ_t [I(y_t = j) - P(j)] × LV
                = Σ_t [(y_t = j일 때 1, 아니면 0) - P(j)] × LV
```

**부호 검증:**
- ✅ 선택된 대안 (y = j): `I(y = j) = 1`, `1 - P(j) > 0` (P(j) < 1이므로)
- ✅ 선택 안 된 대안 (y ≠ j): `I(y = j) = 0`, `0 - P(j) < 0`
- ✅ `θ_j > 0`이고 LV가 효용을 높이면, 선택된 대안의 θ가 클 때 그래디언트는 양수

**예시:**
```
선택 상황: 일반당 선택 (y = sugar)
LV = PI = 2.0
theta_sugar_PI = 0.5
theta_sugar_free_PI = 0.3
P(sugar) = 0.6
P(sugar_free) = 0.3

∂ log L / ∂theta_sugar_PI = (1 - 0.6) × 2.0 = 0.8 > 0  ✅ 양수 (LV를 더 키워라!)
∂ log L / ∂theta_sugar_free_PI = (0 - 0.3) × 2.0 = -0.6 < 0  ✅ 음수 (LV를 줄여라!)
```

### ✅ 결론
- **올바른 구현**: `(y - P) × LV` 형태로 계산
- **부호 검증**: 논리적으로 올바름
- **가중평균**: Importance weighting 적용 (`w_r × gradient`)

---

## ③ 구조모델 오차분산 그래디언트 ✅ **현재는 추정 안 함 (고정)**

### 검토 내용
> **질문**: 구조모델 오차분산 (σ²_PB)도 추정하는가?

### 코드 위치
`src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`
- Line 594-826: `compute_structural_gradient_batch_gpu()`

### 구현 확인

```python
# Line 603: error_variance는 파라미터로 받지만 고정값
def compute_structural_gradient_batch_gpu(
    ...
    error_variance: float = 1.0,  # 고정값
    ...
):
```

**현재 구현:**
- ✅ `gamma` (경로계수)만 추정
- ✅ `error_variance`는 1.0으로 고정
- ✅ 오차분산에 대한 그래디언트 계산 없음

### 만약 추정한다면?

**수식:**
```
η_target = γ × η_predictor + ε,  ε ~ N(0, σ²)

log L = -0.5 log(2π σ²) - 0.5 (η_target - γ × η_predictor)² / σ²

∂ log L / ∂σ² = -0.5 / σ² + 0.5 (η_target - γ × η_predictor)² / σ⁴
```

**체인룰 적용:**
```
∂LL / ∂σ²_PB = Σ_r w_r × ∂LL_r / ∂σ²_PB

여기서:
∂LL_r / ∂σ²_PB = -0.5 / σ²_PB + 0.5 (PB - γ × HC)² / (σ²_PB)²
```

**구현 예시 (참고용):**
```python
# 잔차
residual = target_gpu - gamma_gpu * pred_gpu  # (n_draws,)

# ∂LL/∂σ² = Σ_r w_r × [-0.5 / σ² + 0.5 × residual² / σ⁴]
grad_sigma_sq = cp.sum(
    weights_gpu * (-0.5 / error_variance + 0.5 * (residual ** 2) / (error_variance ** 2))
)
```

### ✅ 결론
- **현재 구현**: 오차분산 고정 (1.0)
- **그래디언트**: 계산 안 함 (추정 대상 아님)
- **향후 확장**: 위 수식대로 구현하면 됨

---

## 📊 최종 검토 요약

| 항목 | 현재 구현 | 올바른지 | 비고 |
|------|----------|---------|------|
| **① 측정모델 차원 합산** | `grad_ll_wrt_lv += ...` | ✅ 올바름 | 38개 지표 누적 합산 |
| **② 선택모델 부호** | `(y - P) × LV` | ✅ 올바름 | Multinomial Logit 정확 |
| **③ 구조모델 오차분산** | 고정 (1.0) | ✅ 올바름 | 추정 안 함 |

---

## 🎯 핵심 포인트

### 1. 측정모델 그래디언트
```python
# ✅ 올바른 구현
grad_ll_wrt_lv = cp.zeros(n_draws)
for i in range(n_indicators):
    grad_ll_wrt_lv += ζ_i × (Y_i - ζ_i × LV) / σ²_i  # 누적 합산
```

### 2. 선택모델 그래디언트
```python
# ✅ 올바른 구현
diff = y - P  # (선택 지시자) - (선택 확률)
grad_theta = Σ_r w_r × Σ_t diff × LV  # 가중평균
```

### 3. 구조모델 오차분산
```python
# ✅ 현재는 고정
error_variance = 1.0  # 추정 안 함

# 만약 추정한다면:
# grad_sigma_sq = Σ_r w_r × [-0.5/σ² + 0.5×residual²/σ⁴]
```

---

## ✅ 결론

**모든 그래디언트 구현이 올바릅니다!**

1. ✅ 측정모델: 38개 지표의 그래디언트를 올바르게 합산
2. ✅ 선택모델: Multinomial Logit 그래디언트 부호 정확
3. ✅ 구조모델: 오차분산은 고정 (추정 안 함)

추가 수정이 필요하지 않습니다. 현재 구현은 이론적으로 정확합니다.

