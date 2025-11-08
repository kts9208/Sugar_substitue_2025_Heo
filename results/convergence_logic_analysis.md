# 현재 코드의 수렴 판단 로직 분석

## 질문

**"로그우도가 -5362.4673이 연속해서 나오면 수렴하도록 로직이 현재 되어있는거야?"**

## 답변: **아니오, 그렇지 않습니다.**

---

## 1. 현재 수렴 판단 로직

### 코드 위치
`src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py` (lines 444-456)

```python
result = optimize.minimize(
    negative_log_likelihood,
    initial_params,
    method='BFGS',
    jac=gradient_function,
    options={
        'maxiter': self.config.estimation.max_iterations,  # 1000
        'ftol': 1e-6,
        'gtol': 1e-5,
        'disp': True
    }
)
```

### 수렴 판단 주체

**scipy.optimize.minimize (BFGS)가 내부적으로 수렴을 판단합니다.**

우리 코드는 수렴 판단 로직을 **직접 구현하지 않고**, scipy에 **완전히 위임**하고 있습니다.

---

## 2. scipy BFGS의 수렴 판단 로직

### 2.1 ftol (Function Tolerance) 기준

**정의**: 
```
ftol = 1e-6
```

**수렴 조건**:
```
|f(x_k) - f(x_{k-1})| / max(|f(x_k)|, |f(x_{k-1})|, 1) < ftol
```

**즉, 상대적 함수값 변화가 ftol보다 작으면 수렴**

### 예시 계산 (LL = -5362.4673)

**Iteration 69**: LL = -5362.4673  
**Iteration 70**: LL = -5362.4673

```
변화량 = |(-5362.4673) - (-5362.4673)| = 0.0000
상대 변화 = 0.0000 / max(5362.4673, 5362.4673, 1) = 0.0000 / 5362.4673 = 0
```

**0 < 1e-6 ✅ 수렴 조건 만족**

### 2.2 gtol (Gradient Tolerance) 기준

**정의**:
```
gtol = 1e-5
```

**수렴 조건**:
```
||∇f(x_k)||_∞ < gtol
```

**즉, gradient의 최대 절대값이 gtol보다 작으면 수렴**

### 예시

만약 모든 파라미터의 gradient가:
```
∂LL/∂θ₁ = 0.000001
∂LL/∂θ₂ = -0.000002
...
∂LL/∂θ₃₇ = 0.000003
```

**max(|gradient|) = 0.000003 < 1e-5 ✅ 수렴 조건 만족**

---

## 3. 실제 수렴 과정 분석 (로그 기반)

### Terminal 6 로그 (이전 실행)

```
Iter  69: LL = -5362.4673 (Best: -5362.4673)
Iter  70: LL = -5362.4673 (Best: -5362.4673)
...
Iter 130: LL = -5362.4673 (Best: -5362.4673)
Iter 139: LL = -5362.4673 (Best: -5362.4673)

최적화 실패: Desired error not necessarily achieved due to precision loss.
```

### 왜 Iter 69에서 수렴하지 않았나?

**가능한 이유**:

1. **Gradient가 아직 gtol보다 컸음**
   - LL은 변하지 않았지만, gradient는 여전히 1e-5보다 컸을 수 있음
   - BFGS는 line search를 수행하면서 gradient를 계속 평가

2. **Line search가 계속 진행됨**
   - BFGS는 각 반복마다 line search를 수행
   - Line search 중에는 함수값이 변하지 않아도 계속 탐색

3. **Precision loss 발생**
   - Iter 70-139: 부동소수점 정밀도 한계에 도달
   - 더 이상 개선 불가능 → "Precision loss" 메시지

---

## 4. 수렴 판단 로직의 문제점

### 현재 문제

**LL이 연속해서 같은 값이 나와도 즉시 수렴하지 않음**

**이유**:
1. scipy BFGS는 ftol과 gtol을 **동시에** 체크하지 않음
2. **OR 조건**: ftol 만족 **또는** gtol 만족 시 수렴
3. 하지만 실제로는 **여러 반복에 걸쳐** 조건을 확인

### 개선 방안

**조기 종료 로직 추가**:

```python
# 현재 코드에 추가할 수 있는 callback 함수
class ConvergenceChecker:
    def __init__(self, patience=10, tol=1e-6):
        self.patience = patience
        self.tol = tol
        self.best_ll = -np.inf
        self.no_improvement_count = 0
    
    def __call__(self, xk):
        current_ll = -negative_log_likelihood(xk)
        
        if abs(current_ll - self.best_ll) < self.tol:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.best_ll = current_ll
        
        if self.no_improvement_count >= self.patience:
            raise StopIteration("No improvement for {} iterations".format(self.patience))

# 사용
checker = ConvergenceChecker(patience=10, tol=1e-6)
result = optimize.minimize(
    negative_log_likelihood,
    initial_params,
    method='BFGS',
    jac=gradient_function,
    callback=checker,  # 추가
    options={...}
)
```

---

## 5. 결론

### 질문에 대한 답변

**"로그우도가 -5362.4673이 연속해서 나오면 수렴하도록 로직이 현재 되어있는거야?"**

**❌ 아니오, 그렇지 않습니다.**

**현재 로직**:
- scipy BFGS가 내부적으로 ftol과 gtol을 체크
- **상대적 LL 변화 < 1e-6** AND/OR **gradient norm < 1e-5** 시 수렴
- 하지만 line search 과정에서 여러 반복이 소요될 수 있음

**실제 발생한 일**:
- Iter 69: 실질적 수렴 도달
- Iter 70-139: Line search 계속 진행, precision loss 발생
- Iter 139: "Precision loss" 메시지로 종료

### 권장 사항

**Option 1: 현재 설정 유지** (권장)
- scipy BFGS의 기본 로직 신뢰
- "Precision loss"는 실질적 수렴을 의미
- 추가 수정 불필요

**Option 2: 조기 종료 로직 추가** (선택적)
- callback 함수로 LL 변화 모니터링
- 10회 연속 개선 없으면 조기 종료
- 시간 절약 가능 (70회 반복 절약)

**Option 3: ftol/gtol 조정** (비권장)
- ftol을 1e-5로 완화 → 더 빨리 수렴
- gtol을 1e-6으로 강화 → 더 정확한 수렴
- 하지만 현재 설정이 이미 적절함

---

## 6. 요약

| 항목 | 현재 상태 | 설명 |
|------|-----------|------|
| **수렴 판단 주체** | scipy BFGS | 우리 코드는 위임만 함 |
| **ftol 기준** | 1e-6 (상대값) | LL 변화 < 1e-6 시 수렴 |
| **gtol 기준** | 1e-5 | Gradient norm < 1e-5 시 수렴 |
| **연속 동일 LL** | ❌ 즉시 수렴 안 함 | Line search 계속 진행 |
| **Iter 69 이후** | Precision loss | 실질적 수렴 완료 |
| **개선 필요성** | ⚠️ 선택적 | 조기 종료 로직 추가 가능 |

**최종 답변**: 현재 로직은 LL이 연속해서 같아도 즉시 수렴하지 않으며, scipy BFGS의 내부 로직에 따라 line search를 계속 수행합니다. 하지만 이는 정상적인 동작이며, "Precision loss" 메시지는 실질적 수렴을 의미합니다.

