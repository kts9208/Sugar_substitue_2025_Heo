# Hessian 계산 로직 요약

**날짜**: 2025-11-20  
**로그 파일**: `simultaneous_estimation_log_20251120_192842.txt`

---

## 🎯 핵심 요약

### ✅ L-BFGS-B는 `hess_inv`를 제공합니다!

현재 시스템의 Hessian 계산:

| 방법 | 타입 | 사용 시점 | 상태 |
|------|------|-----------|------|
| **L-BFGS-B** | `LbfgsInvHessProduct` | 최적화 성공 시 자동 제공 | ✅ 주 방법 |
| **BFGS** | `numpy.ndarray` | 최적화 성공 시 자동 제공 | ✅ 주 방법 |
| **BHHH** | `numpy.ndarray` | Fallback (hess_inv 없을 때) | ⚠️ 드물게 사용 |

**현재 문제**: L-BFGS-B의 Hessian 근사가 ill-conditioned → 최적화 중단 → hess_inv를 받을 수 없음

---

## 📐 1. L-BFGS-B의 Hessian 근사 (최적화 중)

### 알고리즘

```python
# L-BFGS-B 핵심 로직
for iteration in range(maxiter):
    # 1. Gradient 계산
    g = jac(x)  # Analytic gradient
    
    # 2. 탐색 방향 계산 (Two-loop recursion)
    # H^(-1) · g를 암묵적으로 계산 (전체 H 저장 안 함)
    p = two_loop_recursion(s_history, y_history, g)
    
    # 3. Line search
    alpha = line_search(fun, jac, x, p, g)
    
    # 4. 파라미터 업데이트
    x_new = x + alpha * p
    g_new = jac(x_new)
    
    # 5. (s, y) 쌍 저장 (최근 10개만)
    s = x_new - x  # 파라미터 변화
    y = g_new - g  # Gradient 변화
    
    s_history.append(s)
    y_history.append(y)
```

### Two-Loop Recursion

```python
def two_loop_recursion(s_history, y_history, g):
    """
    H^(-1) · g를 암묵적으로 계산
    
    전체 Hessian 행렬을 만들지 않고
    최근 m개의 (s, y) 쌍만 사용
    """
    q = g.copy()
    
    # First loop (backward)
    for i in reversed(range(m)):
        rho[i] = 1.0 / (y[i]^T · s[i])  # ← 문제 발생 지점!
        alpha[i] = rho[i] * (s[i]^T · q)
        q = q - alpha[i] * y[i]
    
    # Initial Hessian approximation
    gamma = (s[-1]^T · y[-1]) / (y[-1]^T · y[-1])
    r = gamma * q
    
    # Second loop (forward)
    for i in range(m):
        beta = rho[i] * (y[i]^T · r)
        r = r + s[i] * (alpha[i] - beta)
    
    return -r  # 탐색 방향
```

### 문제 발생 메커니즘

**Iteration #2 실제 데이터**:
```
s_k norm: 0.747
y_k norm: 515.4
비율: 690.2  ← 매우 큼!

s_k^T · y_k: 319.4
ρ = 1 / 319.4 = 0.00313
```

**Two-loop recursion 실행**:
```python
# First loop
rho = 0.00313
alpha = 0.00313 * (s^T · q)  # s^T · q가 매우 큼
q = q - alpha * y  # y가 매우 크므로 q가 급격히 감소

# 결과
q ≈ 0  # q가 거의 0이 됨
r = gamma * q ≈ 0
p = -r ≈ 0  # 탐색 방향이 0!
```

**결과**:
- ❌ 탐색 방향 `d norm = 0.000000`
- ❌ 파라미터 업데이트 불가
- ❌ 최적화 중단

---

## 🟢 2. L-BFGS-B의 hess_inv 제공

### L-BFGS-B는 hess_inv를 반환합니다!

```python
result = scipy.optimize.minimize(..., method='L-BFGS-B')

# ✅ hess_inv 존재
print(type(result.hess_inv))
# <class 'scipy.optimize._lbfgsb_py.LbfgsInvHessProduct'>

# numpy 배열로 변환
hess_inv_array = result.hess_inv.todense()
```

**특징**:
- 타입: `LbfgsInvHessProduct` (BFGS는 `numpy.ndarray`)
- 메모리: (s, y) 쌍만 저장 (전체 행렬 저장 안 함)
- 변환: `todense()` 메서드로 numpy 배열로 변환
- 연산: `hess_inv @ v` 벡터 곱 지원

---

## 🔵 3. BHHH Hessian 계산 (Fallback)

### 사용 시점

**Optimizer가 hess_inv를 제공하지 않을 때만 사용** (BFGS/L-BFGS-B는 제공함)

### 이론

**정확한 Hessian** (2차 미분):
```
H = ∂²LL/∂θ∂θ^T = Σ_i ∂²LL_i/∂θ∂θ^T
```

**BHHH 근사** (1차 미분만):
```
H ≈ Σ_i (grad_i × grad_i^T)
  = OPG (Outer Product of Gradients)
```

### 구현

```python
def compute_bhhh_hessian(individual_gradients):
    """
    개인별 gradient로부터 BHHH Hessian 계산
    """
    n_params = len(individual_gradients[0])
    hessian = np.zeros((n_params, n_params))
    
    # Σ_i (grad_i × grad_i^T)
    for grad_i in individual_gradients:
        hessian += np.outer(grad_i, grad_i)
    
    # 최소화 문제이므로 음수
    hessian = -hessian
    
    return hessian

def compute_hessian_inverse(hessian):
    """
    Hessian 역행렬 계산 (정규화 포함)
    """
    # 정규화 (수치 안정성)
    hessian_reg = hessian + 1e-8 * np.eye(n_params)
    
    # 역행렬
    hess_inv = np.linalg.inv(hessian_reg)
    
    return hess_inv

def compute_standard_errors(hess_inv):
    """
    표준오차 계산
    """
    # SE = sqrt(diag(H^(-1)))
    variances = np.diag(hess_inv)
    se = np.sqrt(np.abs(variances))
    
    return se
```

### 장점

✅ **계산 효율**: 2차 미분 불필요  
✅ **전체 Hessian**: 모든 상관관계 포함  
✅ **표준오차**: 정확한 SE 계산  
✅ **안정성**: 정규화로 수치 안정성 확보

---

## 🔴 3. 현재 문제 진단

### 3.1 L-BFGS-B의 실패 원인

| 원인 | 설명 | 영향 |
|------|------|------|
| **Gradient 불균형** | 구조모델(~0.01) vs 선택모델(~600) | y_k norm이 매우 큼 |
| **파라미터 스케일** | 모두 1.0으로 고정 | 불균형 해소 불가 |
| **y_k/s_k 비율** | 690.2 (매우 큼) | ρ가 작아짐 |
| **Two-loop 불안정** | ρ * (큰 값) 계산 | q ≈ 0 |
| **탐색 방향 0** | p = -r ≈ 0 | 최적화 중단 |

### 3.2 성분별 분석 (Iteration #2)

| 성분 | s_k | y_k | 비율 | 파라미터 |
|------|-----|-----|------|----------|
| [0] | 8.08e-05 | 6.94e-18 | 0.00 | gamma_HC_PB |
| [1] | -2.07e-05 | 3.71e-06 | 0.18 | gamma_PB_PI |
| [2] | 0.266 | -7.45 | 28.0 | asc_sugar |
| [3] | 0.294 | 172.9 | **589.0** | asc_sugar_free ⚠️ |
| [4] | 0.350 | 44.7 | 127.8 | beta_health_label |

**핵심 발견**:
- 성분 [3] (asc_sugar_free)의 비율이 **589**로 극단적
- 구조모델 파라미터는 거의 변화 없음
- 선택모델 gradient가 과도하게 큼

---

## 💡 4. 해결책

### 4.1 파라미터 스케일링 활성화

```python
# config 수정
config.estimation.use_parameter_scaling = True  # 현재 False

# 효과
scale_factors = 1.0 / np.maximum(np.abs(initial_gradient), 1.0)
# → Gradient 크기에 따라 자동 스케일링
# → y_k/s_k 비율 감소
```

### 4.2 Hessian 주기적 리셋

```python
# 10 iterations마다 Hessian 초기화
if iteration % 10 == 0:
    s_history.clear()
    y_history.clear()
    # → H = I로 리셋
    # → ill-conditioning 방지
```

### 4.3 Trust Region 방법

```python
# L-BFGS-B 대신 Trust Region 사용
config.estimation.optimizer = 'trust-constr'

# 효과
# → 파라미터 변화를 제한
# → Hessian 근사 문제에 덜 민감
```

---

## 📊 5. 코드 위치

### L-BFGS-B 최적화
- **파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`
- **라인**: 1305-1340
- **함수**: `estimate()`

### BHHH Hessian 계산
- **파일**: `src/analysis/hybrid_choice_model/iclv_models/bhhh_calculator.py`
- **라인**: 51-164 (compute_bhhh_hessian)
- **라인**: 166-215 (compute_hessian_inverse)
- **라인**: 217-260 (compute_standard_errors)

---

## 📝 결론

### ✅ L-BFGS-B는 `hess_inv`를 제공합니다!

**코드 수정 사항**:
1. ✅ 잘못된 주석 수정: "L-BFGS-B는 hess_inv 제공 안 함" → 삭제
2. ✅ 로깅 명확화: L-BFGS-B vs BFGS 구분
3. ✅ BHHH는 Fallback: optimizer가 hess_inv를 제공하지 않을 때만 사용

**현재 문제**:
- L-BFGS-B의 Hessian 근사가 ill-conditioned 상태
- y_k/s_k 비율이 690으로 극단적으로 큼
- Two-loop recursion에서 탐색 방향이 0이 됨
- **최적화가 중단되어 hess_inv를 받을 수 없음**

**해결 방법**:
1. 파라미터 스케일링 활성화 (1순위)
2. Hessian 주기적 리셋 (1순위)
3. Trust Region 방법 (2순위)

**참고 문서**:
- `docs/HESSIAN_CALCULATION_LOGIC_EXPLAINED.md` - 상세 설명 (업데이트됨)
- `results/HESSIAN_CONVERGENCE_ISSUE_REPORT_20251120.md` - 진단 보고서
- `scripts/test_lbfgsb_hess_inv.py` - L-BFGS-B hess_inv 테스트

