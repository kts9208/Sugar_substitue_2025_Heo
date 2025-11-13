# BFGS vs BHHH 호환성 분석

## 📋 요약

**질문**: 현재 BFGS 모듈이 어떻게 구현되어 있는지, BHHH 기법으로 변경하려면 호환이 힘든지 검토

**답변**: 
- ✅ **BFGS → BHHH 변경은 완전히 호환 가능**
- ✅ **이미 BHHH 구현 완료** (`docs/early_stopping_hessian_optimization.md` 참조)
- ✅ **BFGS는 최적화 알고리즘, BHHH는 Hessian 근사 방법** (서로 다른 역할)
- ✅ **함께 사용 가능**: BFGS로 최적화 + BHHH로 표준오차 계산

---

## 1. 현재 BFGS 구현 구조

### **1.1. 최적화 프레임워크**

**위치**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

```python
# Line 1234-1242
result = optimize.minimize(
    early_stopping_wrapper.objective,  # 목적 함수 (negative log-likelihood)
    initial_params_scaled,              # 초기 파라미터
    method=self.config.estimation.optimizer,  # 'BFGS' 또는 'L-BFGS-B'
    jac=jac_function,                   # Gradient 함수 (analytic)
    bounds=bounds if self.config.estimation.optimizer == 'L-BFGS-B' else None,
    callback=early_stopping_wrapper.callback,  # Iteration callback
    options=optimizer_options           # BFGS 옵션
)
```

**BFGS 옵션** (Line 1207-1217):
```python
if self.config.estimation.optimizer == 'BFGS':
    optimizer_options = {
        'maxiter': 200,  # Major iteration 최대 횟수
        'ftol': 1e-3,    # 함수값 상대적 변화 0.1% 이하면 종료
        'gtol': 1e-3,    # 그래디언트 norm 허용 오차
        'c1': 1e-4,      # Armijo 조건 파라미터 (scipy 기본값)
        'c2': 0.9,       # Curvature 조건 파라미터 (scipy 기본값)
        'disp': True
    }
```

### **1.2. BFGS 내부 동작 원리**

**Scipy의 BFGS 구현** (`scipy.optimize._minimize._minimize_bfgs`):

```python
def _minimize_bfgs(fun, x0, jac, callback, ...):
    # 1. 초기화
    x = x0
    H = np.eye(n)  # Hessian 역행렬 초기값 (단위 행렬)

    for k in range(maxiter):
        # 2. Gradient 계산
        g = jac(x)  # ← 우리의 analytic gradient 함수 호출

        # 3. 탐색 방향 계산
        p = -H @ g  # H^{-1} × (-g)

        # 4. Line search (Wolfe 조건)
        alpha = line_search(fun, jac, x, p, g, ...)

        # 5. 파라미터 업데이트
        x_new = x + alpha * p

        # 6. Hessian 역행렬 업데이트 (BFGS 공식)
        s = x_new - x
        y = jac(x_new) - g

        # BFGS 업데이트 공식
        rho = 1.0 / (y.T @ s)
        H = (I - rho * s @ y.T) @ H @ (I - rho * y @ s.T) + rho * s @ s.T

        # 7. Callback 호출
        if callback is not None:
            callback(x_new)  # ⚠️ x_new만 전달, H는 전달 안 됨!

        # 8. 수렴 체크
        if converged:
            break

        x = x_new

    # 9. 결과 반환
    return OptimizeResult(x=x, hess_inv=H, ...)  # ✅ 정상 종료 시에만 H 반환
```

**핵심 포인트**:
1. ✅ **BFGS는 Hessian 역행렬을 근사**하여 탐색 방향 계산
2. ✅ **매 iteration마다 H 업데이트** (s_k, y_k 사용)
3. ❌ **Callback에서 H 접근 불가** (x_new만 전달)
4. ✅ **정상 종료 시 `result.hess_inv`로 H 반환**

### **1.3. 현재 Hessian 사용 방식**

**표준오차 계산** (Line 1296-1372):

```python
if self.config.estimation.calculate_se:
    # BFGS의 hess_inv가 있으면 사용
    if hasattr(result, 'hess_inv') and result.hess_inv is not None:
        self.logger.info("Hessian 역행렬: BFGS에서 자동 제공 (추가 계산 0회)")

        hess_inv = result.hess_inv
        if hasattr(hess_inv, 'todense'):
            hess_inv_array = hess_inv.todense()
        else:
            hess_inv_array = hess_inv

        # Hessian 역행렬 저장
        self.hessian_inv_matrix = np.array(hess_inv_array)

        # 대각 원소 = 분산 근사
        diag_elements = np.diag(hess_inv_array)

        # 표준오차 = sqrt(분산)
        standard_errors = np.sqrt(np.abs(diag_elements))
    else:
        # L-BFGS-B는 hess_inv 제공 안 함
        self.logger.warning("Hessian 역행렬 없음 (L-BFGS-B는 hess_inv 제공 안 함)")
        self.hessian_inv_matrix = None
```

**문제점**:
- ❌ **조기 종료 시 `result.hess_inv` 없음** (StopIteration 예외)
- ❌ **L-BFGS-B는 hess_inv 제공 안 함** (메모리 제한)
- ❌ **BFGS의 H는 근사값** (실제 Hessian과 차이 있음)

---

## 2. BHHH 방법 개요

### **2.1. BHHH란?**

**BHHH (Berndt-Hall-Hall-Hausman, 1974)**:
- Maximum Likelihood Estimation에서 Hessian을 근사하는 방법
- **개인별 gradient의 outer product 합**으로 Hessian 근사

### **2.2. 이론적 배경**

**정확한 Hessian**:
```
H = ∂²LL/∂θ∂θ^T = Σ_i ∂²LL_i/∂θ∂θ^T
```

**BHHH 근사**:
```
H_BHHH ≈ -Σ_i (∂LL_i/∂θ) × (∂LL_i/∂θ)^T
        = -Σ_i (grad_i × grad_i^T)
```

**Information Matrix Equality** (MLE 이론):
```
E[-∂²LL/∂θ∂θ^T] = E[(∂LL/∂θ) × (∂LL/∂θ)^T]
```

**의미**:
- 최적점 근처에서 BHHH 근사는 **Fisher Information Matrix**와 동일
- 표준오차 계산에 적합

### **2.3. BHHH 구현 (이미 완료)**

**위치**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py` (조기 종료 후)

```python
# 조기 종료 후 Hessian 역행렬 계산 (BHHH 방법)
if self.config.estimation.calculate_se:
    self.logger.info("조기 종료 후 Hessian 역행렬 계산 중 (BHHH 방법)...")

    n_params = len(early_stopping_wrapper.best_x)

    # 1. 개인별 gradient 계산
    individual_gradients = []
    param_dict = self._unpack_parameters(
        early_stopping_wrapper.best_x,
        measurement_model,
        structural_model,
        choice_model
    )

    # 2. 각 개인에 대해 gradient 계산 (최대 50명)
    for i, (person_id, ind_data) in enumerate(data.groupby('person_id')):
        if i >= 50:  # 샘플링
            break

        ind_draws = halton_draws[i] if i < len(halton_draws) else halton_draws[0]

        # 개인별 gradient 계산 (GPU 배치 처리)
        grad_dict = self.joint_grad.compute_individual_gradient(
            ind_data=ind_data,
            ind_draws=ind_draws,
            params_dict=param_dict,
            measurement_model=measurement_model,
            structural_model=structural_model,
            choice_model=choice_model
        )

        # Gradient를 벡터로 변환
        grad_vector = self._pack_gradient(
            grad_dict,
            measurement_model,
            structural_model,
            choice_model
        )

        individual_gradients.append(grad_vector)

    # 3. BHHH Hessian 계산: H = -Σ (g_i × g_i^T)
    hessian_bhhh = np.zeros((n_params, n_params))
    for grad in individual_gradients:
        hessian_bhhh -= np.outer(grad, grad)  # 음수 (최대화 → 최소화)

    # 4. Hessian 역행렬 계산
    hess_inv = np.linalg.inv(hessian_bhhh)

    # 5. 표준오차 계산
    standard_errors = np.sqrt(np.diag(hess_inv))
```

**계산 비용**:
- 우도 계산: **0회** (gradient만 계산)
- Gradient 계산: **50회** (개인별)
- 소요 시간: **~1.5분** (GPU 사용 시)

---

## 3. BFGS vs BHHH 비교

### **3.1. 역할 차이**

| 항목 | BFGS | BHHH |
|------|------|------|
| **목적** | 최적화 알고리즘 | Hessian 근사 방법 |
| **사용 시점** | 파라미터 추정 중 | 추정 완료 후 (표준오차 계산) |
| **입력** | 목적 함수 + Gradient | 개인별 Gradient |
| **출력** | 최적 파라미터 + Hessian 역행렬 | Hessian 역행렬 |
| **Hessian 업데이트** | 매 iteration (s_k, y_k 사용) | 1회 (최적점에서) |

### **3.2. Hessian 근사 정확도**

| 방법 | 정확도 | 조건 |
|------|--------|------|
| **BFGS** | 중간 | Ill-conditioned 시 부정확 |
| **BHHH** | 높음 | 최적점 근처에서 정확 |
| **수치적 방법** | 가장 높음 | 계산 비용 매우 큼 |

**현재 문제 (Iteration #9 특이행렬)**:
- BFGS Hessian이 **ill-conditioned** → 특이행렬
- BHHH는 **개인별 gradient 직접 사용** → 더 안정적

### **3.3. 계산 비용 비교**

| 방법 | 우도 계산 | Gradient 계산 | 소요 시간 | Hessian 크기 |
|------|-----------|---------------|-----------|--------------|
| **BFGS (자동)** | 0회 | 0회 | 0초 | 전체 (80×80) |
| **BHHH** | 0회 | 50회 | 1.5분 | 전체 (80×80) |
| **수치적 (대각)** | 41,209회 | 0회 | 10.5일 | 대각만 (80개) |

**BHHH 장점**:
- ✅ BFGS보다 **더 정확** (실제 gradient 사용)
- ✅ 수치적 방법보다 **10,080배 빠름**
- ✅ **전체 Hessian** 계산 (상관관계 포함)

---

## 4. 호환성 분석

### **4.1. BFGS → BHHH 변경 가능성**

**질문**: BFGS를 BHHH로 완전히 대체할 수 있는가?

**답변**: ❌ **불가능** (역할이 다름)

**이유**:
1. **BFGS**: 최적화 알고리즘 (파라미터 추정)
2. **BHHH**: Hessian 근사 방법 (표준오차 계산)

**올바른 질문**: BFGS의 Hessian 근사를 BHHH로 대체할 수 있는가?

**답변**: ✅ **가능** (이미 구현됨)

### **4.2. 현재 구현 상태**

**시나리오 1: 정상 종료**
```python
result = optimize.minimize(..., method='BFGS', ...)

if result.success:
    # BFGS의 Hessian 역행렬 사용
    hess_inv = result.hess_inv
    standard_errors = np.sqrt(np.diag(hess_inv))
```

**시나리오 2: 조기 종료**
```python
# BFGS 조기 종료 (result.hess_inv 없음)

# BHHH 방법으로 Hessian 계산
hess_inv = compute_bhhh_hessian(individual_gradients)
standard_errors = np.sqrt(np.diag(hess_inv))
```

**결론**: ✅ **이미 호환 가능하게 구현됨**

### **4.3. 최적화 알고리즘 변경 옵션**

**현재 지원하는 방법**:
1. **BFGS**: Hessian 역행렬 근사 (무제한 메모리)
2. **L-BFGS-B**: 제한된 메모리 BFGS (bounds 지원)
3. **Nelder-Mead**: Gradient 불필요 (느림)

**추가 가능한 방법**:
1. **Trust Region**: Ill-conditioned Hessian에 강함
2. **Newton-CG**: 정확한 Hessian 사용 (BHHH 제공 가능)
3. **Custom BHHH Optimizer**: BHHH로 Hessian 근사하는 최적화

---

## 5. BHHH를 최적화에 사용하는 방법

### **5.1. Custom Optimizer 구현**

**아이디어**: BFGS 대신 BHHH로 Hessian을 근사하는 Newton 방법

```python
def bhhh_optimizer(func, grad_func, x0, individual_grad_func, data, ...):
    """
    BHHH 방법을 사용한 Newton 최적화

    Args:
        func: 목적 함수 (negative log-likelihood)
        grad_func: 전체 gradient 함수
        x0: 초기 파라미터
        individual_grad_func: 개인별 gradient 함수
        data: 데이터
    """
    x = x0

    for iteration in range(max_iter):
        # 1. 전체 gradient 계산
        g = grad_func(x)

        # 2. 개인별 gradient 계산
        individual_grads = []
        for person_id, ind_data in data.groupby('person_id'):
            grad_i = individual_grad_func(x, ind_data)
            individual_grads.append(grad_i)

        # 3. BHHH Hessian 계산
        H_bhhh = np.zeros((len(x), len(x)))
        for grad_i in individual_grads:
            H_bhhh -= np.outer(grad_i, grad_i)

        # 4. Newton 방향 계산
        try:
            H_inv = np.linalg.inv(H_bhhh)
            p = -H_inv @ g
        except np.linalg.LinAlgError:
            # Hessian 특이행렬 → Gradient descent
            p = -g

        # 5. Line search
        alpha = line_search(func, grad_func, x, p, g)

        # 6. 파라미터 업데이트
        x = x + alpha * p

        # 7. 수렴 체크
        if np.linalg.norm(g) < gtol:
            break

    return x, H_inv
```

**장점**:
- ✅ **매 iteration마다 정확한 Hessian** 사용
- ✅ **Ill-conditioned 문제에 강함**
- ✅ **표준오차 자동 계산**

**단점**:
- ❌ **계산 비용 높음** (매 iteration마다 개인별 gradient 계산)
- ❌ **메모리 사용 많음** (전체 Hessian 저장)

### **5.2. 하이브리드 접근**

**권장 방법**: BFGS로 최적화 + BHHH로 표준오차 계산

```python
# 1. BFGS로 빠르게 최적화
result = optimize.minimize(
    func,
    x0,
    method='BFGS',
    jac=grad_func,
    options={'maxiter': 200}
)

# 2. BHHH로 정확한 표준오차 계산
hess_inv_bhhh = compute_bhhh_hessian(
    result.x,
    individual_grad_func,
    data
)

standard_errors = np.sqrt(np.diag(hess_inv_bhhh))
```

**장점**:
- ✅ **빠른 수렴** (BFGS)
- ✅ **정확한 표준오차** (BHHH)
- ✅ **이미 구현됨**

---

## 6. 현재 문제 해결 방안

### **6.1. Iteration #9 특이행렬 문제**

**문제**: BFGS Hessian이 특이행렬 → 탐색 방향 계산 불가

**해결 방안 1: Trust Region 방법**
```python
result = optimize.minimize(
    func,
    x0,
    method='trust-ncg',  # Trust Region Newton-CG
    jac=grad_func,
    hess=bhhh_hessian_func,  # BHHH Hessian 제공
    options={'maxiter': 200}
)
```

**해결 방안 2: BHHH Optimizer (Custom)**
- 위의 5.1 참조
- 매 iteration마다 BHHH Hessian 계산

**해결 방안 3: Hessian Reset**
```python
# BFGS callback에서 Hessian 리셋
def callback(xk):
    if iteration % 5 == 0:
        # Hessian을 Identity로 리셋
        # (scipy BFGS는 직접 리셋 불가능)
        pass
```

**권장**: **Trust Region + BHHH Hessian**

---

## 7. 결론

### **7.1. 호환성 요약**

| 질문 | 답변 |
|------|------|
| BFGS를 BHHH로 대체 가능? | ❌ 역할이 다름 (최적화 vs Hessian 근사) |
| BFGS Hessian을 BHHH로 대체 가능? | ✅ 가능 (이미 구현됨) |
| BHHH를 최적화에 사용 가능? | ✅ 가능 (Custom Optimizer 필요) |
| 현재 구현 호환성? | ✅ 완전 호환 (조기 종료 시 BHHH 사용) |

### **7.2. 권장 사항**

**현재 상황 (Iteration #9 특이행렬)**:

1. ✅ **Trust Region + BHHH Hessian** (가장 안정적)
   ```python
   method='trust-ncg'
   hess=bhhh_hessian_func
   ```

2. ✅ **L-BFGS-B + BHHH 표준오차** (현재 방식 유지)
   ```python
   method='L-BFGS-B'
   # 조기 종료 후 BHHH로 표준오차 계산
   ```

3. ⚠️ **Custom BHHH Optimizer** (구현 필요, 계산 비용 높음)

**최종 권장**: **Option 1 (Trust Region + BHHH)**

### **7.3. 구현 난이도**

| 방법 | 구현 난이도 | 계산 비용 | 안정성 |
|------|------------|----------|--------|
| Trust Region + BHHH | ★★☆☆☆ (쉬움) | ★★★☆☆ (중간) | ★★★★★ (최고) |
| L-BFGS-B + BHHH SE | ★☆☆☆☆ (매우 쉬움) | ★★☆☆☆ (낮음) | ★★★☆☆ (중간) |
| Custom BHHH Optimizer | ★★★★☆ (어려움) | ★★★★☆ (높음) | ★★★★★ (최고) |

**다음 단계**: Trust Region 방법 구현 시도?

<function_calls>
<invoke name="view">
<parameter name="path">src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py
