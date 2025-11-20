# Hessian 계산 로직 상세 설명

**날짜**: 2025-11-20  
**작성자**: AI Assistant

---

## 📋 개요

현재 시스템에서 사용하는 **Hessian 계산 방법**을 설명합니다:

1. **L-BFGS-B/BFGS의 Hessian 근사** (최적화 중 자동 계산) ✅ **주 방법**
2. **BHHH 방법** (Fallback - optimizer가 hess_inv를 제공하지 않을 때만)

---

## 🔵 1. L-BFGS-B의 Hessian 근사 (최적화 중)

### 1.1 알고리즘 개요

L-BFGS-B는 **Limited-memory BFGS**로, 전체 Hessian 행렬을 저장하지 않고 **최근 m개의 (s, y) 쌍**만 저장합니다.

```python
# L-BFGS-B 의사 코드
def lbfgs_b_optimizer(fun, x0, jac, bounds, maxiter=200):
    """
    L-BFGS-B 최적화 알고리즘
    
    Args:
        fun: 목적 함수 (우도 함수)
        x0: 초기 파라미터
        jac: Gradient 함수
        bounds: 파라미터 bounds
    """
    # 초기화
    x = x0
    m = 10  # 메모리 크기 (최근 10개 쌍만 저장)
    s_history = []  # 파라미터 변화 이력
    y_history = []  # Gradient 변화 이력
    
    for k in range(maxiter):
        # 1. Gradient 계산
        g = jac(x)  # ← 우리의 analytic gradient 함수 호출
        
        # 2. 탐색 방향 계산 (Two-loop recursion)
        # H^(-1) · g를 명시적으로 계산하지 않고 암묵적으로 계산
        p = two_loop_recursion(s_history, y_history, g)
        
        # 3. Line search (Wolfe 조건)
        alpha = line_search(fun, jac, x, p, g, bounds)
        
        # 4. 파라미터 업데이트
        x_new = x + alpha * p
        g_new = jac(x_new)
        
        # 5. (s, y) 쌍 저장
        s = x_new - x  # 파라미터 변화
        y = g_new - g  # Gradient 변화
        
        # 6. 메모리 관리 (최근 m개만 유지)
        if len(s_history) >= m:
            s_history.pop(0)
            y_history.pop(0)
        
        s_history.append(s)
        y_history.append(y)
        
        # 7. 수렴 체크
        if converged(g_new):
            break
        
        x = x_new
    
    # ❌ Hessian 역행렬을 반환하지 않음!
    return OptimizeResult(x=x, fun=fun(x), jac=g, success=True)
```

### 1.2 Two-Loop Recursion

L-BFGS-B의 핵심은 **Two-loop recursion**으로 H^(-1) · g를 계산하는 것입니다:

```python
def two_loop_recursion(s_history, y_history, g):
    """
    Two-loop recursion으로 H^(-1) · g 계산
    
    전체 Hessian 역행렬을 만들지 않고 암묵적으로 계산
    """
    m = len(s_history)
    q = g.copy()
    alpha = np.zeros(m)
    rho = np.zeros(m)
    
    # First loop (backward)
    for i in range(m-1, -1, -1):
        rho[i] = 1.0 / np.dot(y_history[i], s_history[i])
        alpha[i] = rho[i] * np.dot(s_history[i], q)
        q = q - alpha[i] * y_history[i]
    
    # Initial Hessian approximation
    if m > 0:
        gamma = np.dot(s_history[-1], y_history[-1]) / np.dot(y_history[-1], y_history[-1])
    else:
        gamma = 1.0
    
    r = gamma * q
    
    # Second loop (forward)
    for i in range(m):
        beta = rho[i] * np.dot(y_history[i], r)
        r = r + s_history[i] * (alpha[i] - beta)
    
    return -r  # 탐색 방향 (음수)
```

### 1.3 L-BFGS-B의 hess_inv 반환

#### ✅ L-BFGS-B는 `hess_inv`를 반환합니다!

```python
result = scipy.optimize.minimize(..., method='L-BFGS-B')
print(type(result.hess_inv))
# <class 'scipy.optimize._lbfgsb_py.LbfgsInvHessProduct'>

# numpy 배열로 변환
hess_inv_array = result.hess_inv.todense()
```

**특징**:
- 타입: `LbfgsInvHessProduct` 객체 (BFGS는 `numpy.ndarray`)
- 메모리 효율: 전체 행렬을 저장하지 않고 (s, y) 쌍만 저장
- 사용 방법: `todense()` 메서드로 numpy 배열로 변환 가능
- 벡터 곱: `hess_inv @ v` 연산 지원

### 1.4 문제점

#### ❌ 문제: y_k/s_k 비율이 클 때 불안정
- `ρ = 1/(s_k^T · y_k)`가 매우 작아짐 (y_k가 클 때)
- Two-loop recursion에서 수치적 불안정 발생
- 탐색 방향이 0이 되는 현상
- **하지만 최적화가 성공하면 hess_inv는 정상적으로 제공됨**

---

## 🟢 2. BHHH 방법의 Hessian 계산 (Fallback)

### 2.1 이론적 배경

**BHHH (Berndt-Hall-Hall-Hausman)** 방법은 Maximum Likelihood Estimation에서 Hessian을 근사하는 방법입니다.

**사용 시점**: Optimizer가 `hess_inv`를 제공하지 않을 때만 사용 (BFGS/L-BFGS-B는 제공함)

#### 정확한 Hessian (2차 미분)
```
H = ∂²LL/∂θ∂θ^T = Σ_i ∂²LL_i/∂θ∂θ^T
```

#### BHHH 근사 (1차 미분만 사용)
```
H ≈ Σ_i (∂LL_i/∂θ) × (∂LL_i/∂θ)^T
  = Σ_i (grad_i × grad_i^T)
  = OPG (Outer Product of Gradients)
```

### 2.2 구현 코드

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/bhhh_calculator.py" mode="EXCERPT">
````python
def compute_bhhh_hessian(
    self,
    individual_gradients: List[np.ndarray],
    for_minimization: bool = True
) -> np.ndarray:
    """
    개인별 gradient로부터 BHHH Hessian 계산
    
    BHHH = Σ_i (grad_i × grad_i^T)
    """
    n_individuals = len(individual_gradients)
    n_params = len(individual_gradients[0])
    
    # BHHH Hessian 초기화
    hessian_bhhh = np.zeros((n_params, n_params))
    
    # Σ_i (grad_i × grad_i^T)
    for i, grad in enumerate(individual_gradients):
        # Outer product: grad_i × grad_i^T
        outer_prod = np.outer(grad, grad)
        hessian_bhhh += outer_prod
    
    # 최소화 문제의 경우 음수 부호
    if for_minimization:
        hessian_bhhh = -hessian_bhhh
    
    return hessian_bhhh
````
</augment_code_snippet>

### 2.3 Hessian 역행렬 계산

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/bhhh_calculator.py" mode="EXCERPT">
````python
def compute_hessian_inverse(
    self,
    hessian: np.ndarray,
    regularization: float = 1e-8
) -> np.ndarray:
    """
    Hessian 역행렬 계산 (정규화 포함)
    """
    n_params = hessian.shape[0]
    
    # 정규화 (수치 안정성)
    hessian_reg = hessian + regularization * np.eye(n_params)
    
    # 역행렬 계산
    hess_inv = np.linalg.inv(hessian_reg)
    
    return hess_inv
````
</augment_code_snippet>

### 2.4 장점

✅ **계산 효율성**: 2차 미분 불필요 (1차 미분만 사용)  
✅ **전체 Hessian**: 모든 파라미터 간 상관관계 포함  
✅ **표준오차 계산**: SE = sqrt(diag(H^(-1)))  
✅ **Robust SE**: Sandwich estimator 계산 가능

---

## 🔄 3. 현재 시스템의 Hessian 계산 흐름

### 3.1 최적화 중 (L-BFGS-B가 자동으로 Hessian 근사)

```
Iteration 1:
  x0 → g0 = jac(x0)
  p0 = two_loop_recursion([], [], g0) = -g0  (첫 iteration은 gradient descent)
  x1 = x0 + alpha * p0

  s0 = x1 - x0
  y0 = g1 - g0
  저장: s_history = [s0], y_history = [y0]

Iteration 2:
  x1 → g1 = jac(x1)
  p1 = two_loop_recursion([s0], [y0], g1)  ← Hessian 근사 사용
  x2 = x1 + alpha * p1

  s1 = x2 - x1
  y1 = g2 - g1
  저장: s_history = [s0, s1], y_history = [y0, y1]

...

❌ 문제 발생 (Iteration 3):
  y_k/s_k 비율이 690으로 매우 큼
  → ρ = 1/(s_k^T · y_k)가 매우 작음
  → two_loop_recursion에서 수치적 불안정
  → p = 0 (탐색 방향이 0)
  → 최적화 중단

✅ 최적화 성공 시:
  result.hess_inv = LbfgsInvHessProduct(s_history, y_history)
  → todense()로 numpy 배열로 변환
  → 표준오차 계산
```

### 3.2 최적화 후 Hessian 처리

```python
# 우리 코드의 실제 로직
if hasattr(result, 'hess_inv') and result.hess_inv is not None:
    # ✅ L-BFGS-B/BFGS가 제공한 hess_inv 사용
    if hasattr(result.hess_inv, 'todense'):
        # L-BFGS-B: LbfgsInvHessProduct → numpy array
        hess_inv_array = result.hess_inv.todense()
    else:
        # BFGS: 이미 numpy array
        hess_inv_array = result.hess_inv

    # 표준오차 계산
    SE = sqrt(diag(hess_inv_array))

else:
    # ❌ Optimizer가 hess_inv를 제공하지 않음 (드문 경우)
    # → BHHH 방법으로 계산 (Fallback)

    개인별 gradient 계산:
      for each individual i:
        grad_i = compute_individual_gradient(x_final, data_i)

    BHHH Hessian 계산:
      H = Σ_i (grad_i × grad_i^T)

    Hessian 역행렬:
      H_inv = inv(H + 1e-8 * I)

    표준오차:
      SE = sqrt(diag(H_inv))
```

---

## 📊 4. 문제 진단: 왜 탐색 방향이 0이 되는가?

### 4.1 수치적 분석

**Iteration #2 데이터**:
- s_k norm: 0.747
- y_k norm: 515.4
- **비율: 690.2**
- s_k^T · y_k: 319.4
- **ρ = 1/319.4 = 0.00313**

**Two-loop recursion에서**:
```python
# First loop
rho = 1 / (y^T · s) = 0.00313
alpha = rho * (s^T · q) = 0.00313 * (매우 큰 값)
q = q - alpha * y = q - (매우 큰 값) * y
```

→ `q`가 매우 작아짐 또는 0에 가까워짐  
→ `r = gamma * q ≈ 0`  
→ **탐색 방향 p ≈ 0**

### 4.2 근본 원인

1. **Gradient 크기 불균형**
   - 구조모델: ~0.01
   - 선택모델: ~100~600
   - **10,000배 차이**

2. **파라미터 스케일 불균형**
   - 모든 스케일이 1.0으로 고정
   - 불균형 해소 불가

3. **Hessian 근사의 ill-conditioning**
   - y_k/s_k 비율이 매우 큼
   - Hessian이 매우 큰 고유값을 가짐
   - 역행렬이 거의 0

---

## 💡 5. 해결책

### 5.1 파라미터 스케일링

```python
# Gradient 크기에 따라 자동 스케일링
scale_factors = 1.0 / np.maximum(np.abs(initial_gradient), 1.0)
x_scaled = x * scale_factors
```

### 5.2 Hessian 주기적 리셋

```python
# 10 iterations마다 Hessian을 초기값(I)으로 리셋
if iteration % 10 == 0:
    s_history.clear()
    y_history.clear()
```

### 5.3 Trust Region 방법

```python
# L-BFGS-B 대신 Trust Region 사용
optimizer = 'trust-constr'
```

---

## 📝 결론

### ✅ L-BFGS-B는 `hess_inv`를 제공합니다!

현재 시스템은:
- **L-BFGS-B/BFGS**: 최적화 중 Hessian 근사 계산 → `hess_inv` 제공 ✅
  - L-BFGS-B: `LbfgsInvHessProduct` 객체 (todense()로 변환)
  - BFGS: `numpy.ndarray`
- **BHHH**: Fallback (optimizer가 hess_inv를 제공하지 않을 때만)

### 🔴 현재 문제

문제는 **L-BFGS-B의 Hessian 근사가 ill-conditioned** 상태가 되어 탐색 방향이 0이 되는 것입니다.
- 최적화가 중단됨 → `hess_inv`를 받을 수 없음
- 해결책: **파라미터 스케일링 + Hessian 리셋**

### 📌 코드 수정 사항

1. **잘못된 주석 수정**: "L-BFGS-B는 hess_inv 제공 안 함" → "L-BFGS-B도 hess_inv 제공"
2. **로깅 명확화**: L-BFGS-B vs BFGS 구분
3. **BHHH는 Fallback**: optimizer가 hess_inv를 제공하지 않을 때만 사용

