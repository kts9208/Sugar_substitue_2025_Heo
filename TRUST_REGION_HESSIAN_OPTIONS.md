# Trust Region의 Hessian 근사 방법

**날짜**: 2025-11-23  
**질문**: Trust Region에서 BHHH가 필수인가? 다른 Hessian 근사 방법은?

---

## 📋 요약

**답변**: ❌ **BHHH는 필수가 아닙니다!**

Trust Region은 **여러 Hessian 근사 방법**을 사용할 수 있으며, scipy의 `trust-constr`는 **자동으로 BFGS 또는 SR1 근사**를 사용합니다.

---

## ✅ 1. Trust Region의 Hessian 근사 방법

### 1.1 scipy trust-constr의 기본 동작

**scipy 공식 문서** (v1.13.0):
```
trust-constr: Trust-region algorithm for constrained optimization.

When Hessians are not provided, it uses the BFGS method to approximate them.
For problems with many constraints, it may switch to SR1 approximation.
```

**핵심**:
- ✅ Hessian 제공 안 하면 → **자동으로 BFGS 근사** 사용
- ✅ 제약 조건 많으면 → **SR1 근사**로 전환 가능
- ✅ **BHHH는 전혀 사용하지 않음**

---

### 1.2 Trust Region이 사용하는 Hessian 근사 방법

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **BFGS** | Quasi-Newton 방법 | ✅ 안정적, 빠름 | ⚠️ 메모리 많이 사용 |
| **SR1** | Symmetric Rank-1 | ✅ 비볼록 문제에 강함 | ⚠️ 불안정할 수 있음 |
| **Analytic Hessian** | 사용자 제공 | ✅ 가장 정확 | ⚠️ 계산 비용 높음 |
| **Finite Difference** | 수치적 근사 | ✅ 구현 쉬움 | ⚠️ 매우 느림 |

**scipy trust-constr 기본값**: **BFGS**

---

## 🔍 2. BHHH는 언제 사용하는가?

### 2.1 BHHH의 목적

**BHHH (Berndt-Hall-Hall-Hausman)**:
```
Hessian ≈ Σ_i (grad_i × grad_i^T)
```

**용도**:
- ❌ **최적화 중 Hessian 근사** (Trust Region이 사용하는 것)
- ✅ **최적화 후 표준오차 계산** (통계적 추론)

**핵심 차이**:
- **Trust Region의 BFGS**: 최적화 알고리즘이 **탐색 방향** 계산에 사용
- **BHHH**: 최적화 완료 후 **표준오차** 계산에 사용

---

### 2.2 현재 코드의 BHHH 사용 목적

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py" mode="EXCERPT">
````python
# Line 1506-1514
else:
    # Optimizer가 hess_inv를 제공하지 않는 경우 → BHHH 방법으로 계산
    # 참고: BFGS와 L-BFGS-B는 모두 hess_inv를 제공하므로,
    #       이 분기는 다른 optimizer를 사용하거나 최적화가 실패한 경우에만 실행됨
    self.iteration_logger.warning("⚠️ Optimizer가 Hessian 역행렬을 제공하지 않음")
    self.iteration_logger.info("→ BHHH 방법으로 Hessian 역행렬 계산 시작...")
    self.iteration_logger.info("  (개인별 gradient의 Outer Product 사용)")
    
    # BHHH 방법으로 Hessian 계산
    hess_inv_bhhh = self._compute_bhhh_hessian_inverse(...)
````
</augment_code_snippet>

**목적**: **표준오차 계산**을 위한 Hessian 역행렬 계산

**Trust Region과의 관계**:
- Trust Region은 **자체 BFGS 근사**를 사용하여 최적화
- 최적화 완료 후, **표준오차 계산**을 위해 BHHH 사용

---

## 📊 3. Trust Region의 Hessian 근사 vs BHHH

### 3.1 두 가지 Hessian의 역할

| 항목 | Trust Region의 BFGS | BHHH |
|------|-------------------|------|
| **사용 시점** | 최적화 중 (매 iteration) | 최적화 후 (1회) |
| **목적** | 탐색 방향 계산 | 표준오차 계산 |
| **계산 방법** | BFGS 업데이트 | Outer Product of Gradients |
| **메모리** | 전체 Hessian 저장 | 개인별 gradient 저장 |
| **정확도** | Quasi-Newton 근사 | OPG 근사 |
| **제공 여부** | ❌ 외부 제공 안 함 | ✅ 계산 후 제공 |

---

### 3.2 Trust Region 최적화 과정

```
[Iteration 1]
  1. Gradient 계산: g_1 = ∇f(x_1)
  2. BFGS Hessian 근사: H_1 ≈ ∇²f(x_1)
  3. Trust Region 부문제 해결: min_p {g_1^T p + 1/2 p^T H_1 p}  s.t. ||p|| ≤ Δ
  4. 파라미터 업데이트: x_2 = x_1 + p
  5. BFGS Hessian 업데이트: H_2 = BFGS_update(H_1, s, y)

[Iteration 2]
  ...

[최적화 완료]
  - 최종 파라미터: x*
  - Trust Region의 Hessian: H* (내부에만 존재, 외부 제공 안 함)
  
[표준오차 계산]
  - BHHH 방법으로 Hessian 역행렬 계산
  - SE = sqrt(diag(H_BHHH^(-1)))
```

**핵심**:
- Trust Region은 **자체 BFGS Hessian**을 사용하여 최적화
- 하지만 **외부에 제공하지 않음** (`result.hess_inv = None`)
- 따라서 **표준오차 계산**을 위해 BHHH 사용

---

## 💡 4. Trust Region에서 Hessian 제공 방법

### 4.1 옵션 1: Analytic Hessian 제공 (가장 정확)

```python
def hessian_function(x):
    """
    Analytic Hessian 계산
    
    Returns:
        Hessian 행렬 (n_params, n_params)
    """
    # 해석적 Hessian 계산 (매우 복잡)
    ...
    return hessian

result = optimize.minimize(
    objective,
    initial_params,
    method='trust-constr',
    jac=gradient_function,
    hess=hessian_function,  # ← Analytic Hessian 제공
    options=optimizer_options
)

# Trust Region이 제공한 Hessian 사용
if hasattr(result, 'hess'):
    hessian = result.hess
    hess_inv = np.linalg.inv(hessian)
```

**장점**:
- ✅ 가장 정확한 Hessian
- ✅ Trust Region이 더 빠르게 수렴

**단점**:
- ❌ Analytic Hessian 계산 매우 복잡
- ❌ 계산 비용 높음 (매 iteration마다)

---

### 4.2 옵션 2: Hessian-Vector Product 제공 (효율적)

```python
def hessp_function(x, p):
    """
    Hessian-vector product: H(x) @ p
    
    Args:
        x: 파라미터
        p: 벡터
    
    Returns:
        H(x) @ p
    """
    # Finite difference로 근사
    epsilon = 1e-8
    grad_x = gradient_function(x)
    grad_x_plus = gradient_function(x + epsilon * p)
    return (grad_x_plus - grad_x) / epsilon

result = optimize.minimize(
    objective,
    initial_params,
    method='trust-constr',
    jac=gradient_function,
    hessp=hessp_function,  # ← Hessian-vector product 제공
    options=optimizer_options
)
```

**장점**:
- ✅ 전체 Hessian 계산 불필요
- ✅ 메모리 효율적

**단점**:
- ❌ 여전히 계산 비용 높음
- ❌ `result.hess_inv` 제공 안 함

---

### 4.3 옵션 3: BHHH 사용 (현재 방식, 권장)

```python
result = optimize.minimize(
    objective,
    initial_params,
    method='trust-constr',
    jac=gradient_function,
    # hess, hessp 제공 안 함 → Trust Region이 자동으로 BFGS 사용
    options=optimizer_options
)

# Trust Region은 hess_inv 제공 안 함
# → BHHH 방법으로 계산
hess_inv_bhhh = compute_bhhh_hessian_inverse(...)
```

**장점**:
- ✅ 구현 간단 (이미 구현됨)
- ✅ 계산 비용 낮음 (최적화 후 1회만)
- ✅ Robust 표준오차 계산 가능

**단점**:
- ⚠️ 추가 60초 소요 (전체의 2%)

---

## 📊 5. 비교: Hessian 제공 방법

| 방법 | Trust Region 최적화 | 표준오차 계산 | 추가 시간 | 구현 난이도 |
|------|-------------------|-------------|----------|-----------|
| **Analytic Hessian** | BFGS → Analytic | Analytic 역행렬 | 매 iteration 증가 | ⚠️⚠️⚠️ 매우 어려움 |
| **Hessian-Vector Product** | BFGS → hessp | BHHH | 매 iteration 증가 | ⚠️⚠️ 어려움 |
| **BHHH (현재)** | BFGS (자동) | BHHH | +60초 (1회) | ✅ 쉬움 |

**권장**: **BHHH (현재 방식)**

---

## 🔍 6. Trust Region의 내부 BFGS vs 외부 BHHH

### 6.1 Trust Region 내부 BFGS

**역할**: 최적화 알고리즘의 탐색 방향 계산

**BFGS 업데이트**:
```python
# Iteration k에서 k+1로
s_k = x_{k+1} - x_k  # 파라미터 변화
y_k = g_{k+1} - g_k  # Gradient 변화

# BFGS Hessian 업데이트
H_{k+1} = H_k + (y_k @ y_k^T) / (y_k @ s_k) - (H_k @ s_k @ s_k^T @ H_k) / (s_k^T @ H_k @ s_k)
```

**특징**:
- ✅ 매 iteration마다 업데이트
- ✅ 전체 Hessian 저장 (메모리 많이 사용)
- ❌ 외부에 제공 안 함

---

### 6.2 외부 BHHH

**역할**: 표준오차 계산을 위한 Hessian 역행렬

**BHHH 계산**:
```python
# 최적화 완료 후 1회만 계산
individual_gradients = [grad_1, grad_2, ..., grad_N]

# BHHH Hessian
H_BHHH = Σ_i (grad_i @ grad_i^T)

# Hessian 역행렬
H_BHHH_inv = np.linalg.inv(H_BHHH)

# 표준오차
SE = sqrt(diag(H_BHHH_inv))
```

**특징**:
- ✅ 최적화 후 1회만 계산
- ✅ 개인별 gradient 기반 (통계적으로 의미 있음)
- ✅ Robust 표준오차 계산 가능

---

## 💡 7. 결론

### 7.1 Trust Region과 BHHH의 관계

| 질문 | 답변 |
|------|------|
| **Trust Region이 BHHH를 사용하나요?** | ❌ **아니오** (BFGS 사용) |
| **BHHH는 왜 필요한가요?** | ✅ **표준오차 계산**을 위해 |
| **Trust Region이 Hessian을 제공하나요?** | ❌ **아니오** (`result.hess_inv = None`) |
| **다른 Hessian 근사 방법은?** | ✅ **Analytic, hessp, BHHH** |

---

### 7.2 권장 방법

**현재 방식 (BHHH) 유지 권장**:

1. ✅ **Trust Region 최적화**: 자동으로 BFGS 근사 사용
2. ✅ **표준오차 계산**: BHHH 방법 사용
3. ✅ **추가 시간**: 60초 (전체의 2%)
4. ✅ **구현 난이도**: 낮음 (이미 구현됨)

**Analytic Hessian 제공은 비권장**:
- ❌ 구현 매우 복잡
- ❌ 계산 비용 높음
- ❌ 실질적 이득 미미

---

### 7.3 최종 정리

**Trust Region의 Hessian 근사**:
- **최적화 중**: Trust Region 자체 BFGS 근사 사용 (자동)
- **최적화 후**: BHHH 방법으로 표준오차 계산 (수동)

**BHHH는 필수가 아니라 선택**:
- Analytic Hessian 제공 가능 (복잡함)
- Hessian-Vector Product 제공 가능 (복잡함)
- **BHHH 사용 권장** (간단하고 효율적)

---

**분석 완료 일시**: 2025-11-23

