# 구조모델 그래디언트 역전파 수정

## 📌 개요

구조모델 파라미터 그래디언트 계산에 **체인룰(Chain Rule) 역전파**를 올바르게 적용했습니다.

---

## ❌ **문제점: 잘못된 그래디언트 계산**

### **기존 구현 (삭제됨)**

```python
# ❌ 잘못된 구현: 구조모델 우도의 그래디언트만 계산
# ∂LL_structural/∂γ = Σ_r w_r * (target - μ)_r / σ² * predictor_r
weighted_residual = weights_gpu * residual / error_variance
grad_gamma = cp.sum(weighted_residual * pred_gpu)
```

**문제**:
1. **구조모델 우도 그래디언트만 계산**: `∂LL_structural/∂γ`
2. **역전파 누락**: 측정모델과 선택모델을 통한 역전파가 없음
3. **이론적 오류**: 구조모델 우도는 전체 우도에 포함되지 않음 (이전에 수정됨)

---

## ✅ **해결책: 체인룰 역전파 적용**

### **올바른 그래디언트 공식**

```
∂LL/∂γ_HC_to_PB = Σ_r w_r × ∂LL_r/∂γ_HC_to_PB 

∂LL_r/∂γ_HC_to_PB = ∂LL_measurement/∂PB × ∂PB/∂γ_HC_to_PB
                    + ∂LL_choice/∂PB × ∂PB/∂γ_HC_to_PB

where:
∂PB/∂γ_HC_to_PB = HC (예측변수 값)
```

### **새 구현**

```python
# ✅ 올바른 구현: 체인룰 역전파

# 1. ∂LL_measurement/∂target 계산
grad_ll_meas_wrt_target = compute_measurement_grad_wrt_lv_gpu(
    gpu_measurement_model,
    ind_data,
    lvs_list,
    params['measurement'],
    target
)

# 2. ∂LL_choice/∂target 계산
grad_ll_choice_wrt_target = compute_choice_grad_wrt_lv_gpu(
    ind_data,
    lvs_list,
    params['choice'],
    target,
    choice_attributes
)

# 3. 총 그래디언트: ∂LL/∂target
grad_ll_wrt_target = grad_ll_meas_wrt_target + grad_ll_choice_wrt_target

# 4. 체인룰: ∂LL/∂γ = Σ_r w_r × (∂LL/∂target)_r × (∂target/∂γ)_r
# ∂target/∂γ = predictor
grad_gamma = cp.sum(weights_gpu * grad_ll_wrt_target * pred_gpu)
```

---

## 🔧 **추가된 헬퍼 함수**

### **1. `compute_measurement_grad_wrt_lv_gpu()`**

측정모델 우도의 잠재변수에 대한 그래디언트 계산

```python
def compute_measurement_grad_wrt_lv_gpu(
    gpu_measurement_model,
    ind_data: pd.DataFrame,
    lvs_list: List[Dict[str, float]],
    params_measurement: Dict,
    target_lv: str
) -> np.ndarray:
    """
    ∂LL_measurement/∂LV for each draw
    
    Returns:
        (n_draws,) array
    """
```

**계산 로직**:

- **Continuous Linear**: `∂LL/∂LV = Σᵢ ζᵢ * (yᵢ - ζᵢ*LV) / σᵢ²`
- **Ordered Probit**: `∂LL/∂LV = Σᵢ (φ_upper - φ_lower) / P * (-ζᵢ)`

### **2. `compute_choice_grad_wrt_lv_gpu()`**

선택모델 우도의 잠재변수에 대한 그래디언트 계산

```python
def compute_choice_grad_wrt_lv_gpu(
    ind_data: pd.DataFrame,
    lvs_list: List[Dict[str, float]],
    params_choice: Dict,
    target_lv: str,
    choice_attributes: List[str]
) -> np.ndarray:
    """
    ∂LL_choice/∂LV for each draw
    
    Returns:
        (n_draws,) array
    """
```

**계산 로직**:

- **Binary Probit**: `∂LL_choice/∂LV = Σ_situations (sign * mills * λ)`
  - `mills = φ(V) / P(choice)`
  - `sign = +1 if choice=1, -1 if choice=0`

---

## 📊 **수정된 파일**

### **`gpu_gradient_batch.py`**

#### **1. 헬퍼 함수 추가** (라인 46-245)
- `compute_measurement_grad_wrt_lv_gpu()`: 측정모델 역전파
- `compute_choice_grad_wrt_lv_gpu()`: 선택모델 역전파

#### **2. 구조모델 그래디언트 수정** (라인 646-810)

**계층적 구조** (라인 696-747):
```python
# ✅ 역전파 적용
grad_ll_wrt_target = grad_ll_meas_wrt_target + grad_ll_choice_wrt_target
grad_gamma = cp.sum(weights_gpu * grad_ll_wrt_target * pred_gpu)
```

**병렬 구조** (라인 751-816):
```python
# ✅ 역전파 적용
grad_ll_wrt_endo = grad_ll_meas_wrt_endo + grad_ll_choice_wrt_endo
grad_gamma_lv = cp.dot(exo_lv_gpu.T, weights_gpu * grad_ll_wrt_endo)
```

#### **3. 함수 호출 수정** (라인 1225-1241)
```python
grad_struct = compute_structural_gradient_batch_gpu(
    ...,
    params_dict,  # ✅ 전체 파라미터 전달
    ...,
    gpu_measurement_model=gpu_measurement_model,  # ✅ 역전파용
    choice_model=choice_model  # ✅ 역전파용
)
```

#### **4. 완전 병렬 버전 수정** (라인 1831-1919)
```python
def compute_structural_full_batch_gpu(
    all_ind_data,
    all_lvs_gpu,
    params_dict,  # ✅ 전체 파라미터
    ...,
    choice_model,  # ✅ 역전파용
    gpu_measurement_model  # ✅ 역전파용
):
    # 개인별로 역전파 계산
    for ind_idx, ind_data in enumerate(all_ind_data):
        grad_ll_wrt_target = grad_ll_meas + grad_ll_choice
        grad_gamma = cp.sum(weights * grad_ll_wrt_target * pred_values)
```

---

## 🎯 **이론적 근거**

### **ICLV 모델 우도**

```
L = ∫ P(Choice | LV, X) × P(Indicators | LV) × P(LV | X) dLV
```

### **시뮬레이션 기반 근사**

```
L ≈ (1/R) Σᵣ P(Choice | LVᵣ, X) × P(Indicators | LVᵣ)
```

여기서 `LVᵣ = γ * X + ηᵣ` (구조모델에서 생성)

### **그래디언트 계산**

```
∂ log L / ∂γ = ∂/∂γ log[(1/R) Σᵣ P(Choice | LVᵣ) × P(Indicators | LVᵣ)]
```

**체인룰 적용**:

```
∂ log L / ∂γ = Σᵣ w_r × [∂ log P(Choice | LVᵣ) / ∂LVᵣ + ∂ log P(Indicators | LVᵣ) / ∂LVᵣ] × ∂LVᵣ / ∂γ
```

여기서:
- `w_r`: Importance weight
- `∂LVᵣ / ∂γ = X` (예측변수)

---

## 🚀 **예상 효과**

### **1. 정확한 그래디언트**
- ✅ 이론적으로 올바른 그래디언트 계산
- ✅ 측정모델과 선택모델의 정보가 구조모델 파라미터 추정에 반영됨

### **2. 더 나은 수렴**
- ✅ 올바른 그래디언트 방향
- ✅ 더 빠른 수렴 속도
- ✅ 더 정확한 파라미터 추정

### **3. 일관성**
- ✅ 우도 계산과 그래디언트 계산이 일관됨
- ✅ 수치적 그래디언트와 해석적 그래디언트가 일치

---

**작성일**: 2025-11-22  
**작성자**: Sugar Substitute Research Team

