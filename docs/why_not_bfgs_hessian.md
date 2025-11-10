# 왜 BFGS의 Hessian 역행렬을 조기 종료 시 사용할 수 없는가?

## 🎯 질문

**방안 1**: BFGS의 누적된 Hessian 역행렬 활용 (이미 계산됨, 추가 비용 0)

이 방법을 사용하지 못하는 이유는?

---

## 📋 답변 요약

**BFGS의 Hessian 역행렬은 조기 종료 시 접근할 수 없습니다.**

**이유**:
1. ❌ **Callback에서 접근 불가**: `scipy.optimize.minimize`의 callback은 파라미터(`xk`)만 전달받음
2. ❌ **중간 상태 저장 불가**: BFGS 내부 상태(Hessian 역행렬)는 private 변수
3. ❌ **조기 종료 시 미완성**: `StopIteration` 예외 발생 시 `result` 객체가 생성되지 않음
4. ✅ **정상 종료 시만 가능**: `optimize.minimize`가 정상 종료되어야 `result.hess_inv` 접근 가능

---

## 🔍 상세 분석

### 1. **scipy.optimize.minimize의 BFGS 구조**

#### BFGS 알고리즘 개요
```python
# BFGS 내부 구조 (의사 코드)
def _minimize_bfgs(fun, x0, jac, callback, ...):
    # 초기화
    x = x0
    H = np.eye(n)  # Hessian 역행렬 초기값 (단위 행렬)
    
    for k in range(maxiter):
        # 1. Gradient 계산
        g = jac(x)
        
        # 2. 탐색 방향 계산
        p = -H @ g
        
        # 3. Line search
        alpha = line_search(...)
        
        # 4. 파라미터 업데이트
        x_new = x + alpha * p
        
        # 5. Hessian 역행렬 업데이트 (BFGS 공식)
        s = x_new - x
        y = jac(x_new) - g
        H = H + ...  # BFGS update formula
        
        # 6. Callback 호출
        if callback is not None:
            callback(x_new)  # ⚠️ x_new만 전달, H는 전달 안 됨!
        
        # 7. 수렴 체크
        if converged:
            break
        
        x = x_new
    
    # 8. 결과 반환
    return OptimizeResult(x=x, hess_inv=H, ...)  # ✅ 정상 종료 시에만 H 반환
```

**핵심 문제**:
- `H` (Hessian 역행렬)는 함수 내부 지역 변수
- Callback은 `x_new`만 받음 (`H`는 받지 못함)
- 정상 종료 시에만 `OptimizeResult`에 `hess_inv` 포함

---

### 2. **조기 종료 시 문제점**

#### 현재 구현
```python
class EarlyStoppingWrapper:
    def objective(self, x):
        self.func_call_count += 1
        current_ll = self.func(x)
        
        # LL 개선 체크
        if current_ll < self.best_ll - self.tol:
            self.best_ll = current_ll
            self.best_x = x.copy()
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # 조기 종료 조건
        if self.no_improvement_count >= self.patience:
            self.early_stopped = True
            raise StopIteration("조기 종료")  # ⚠️ 예외 발생!
        
        return current_ll
    
    def callback(self, xk):
        # ❌ 여기서 Hessian 역행렬에 접근할 수 없음!
        # callback은 xk만 받음
        pass

# 최적화 실행
try:
    result = optimize.minimize(
        early_stopping_wrapper.objective,
        initial_params,
        method='BFGS',
        jac=jac_function,
        callback=early_stopping_wrapper.callback,
        options={'maxiter': 200}
    )
    
    # ✅ 정상 종료 시: result.hess_inv 사용 가능
    if hasattr(result, 'hess_inv'):
        hess_inv = result.hess_inv
        
except StopIteration as e:
    # ❌ 조기 종료 시: result 객체가 생성되지 않음!
    # BFGS 내부의 H는 접근 불가능
    # 여기서 Hessian 역행렬을 얻을 방법이 없음!
    pass
```

**문제점**:
1. `StopIteration` 예외 발생 → `optimize.minimize` 중단
2. `OptimizeResult` 객체 생성 안 됨
3. BFGS 내부의 `H` (Hessian 역행렬)는 함수 스코프 내에서 소멸
4. **접근 불가능!**

---

### 3. **시도해본 해결 방법들**

#### 시도 1: Callback에서 저장
```python
def callback(self, xk):
    # ❌ 실패: callback은 xk만 받음, hess_inv는 받지 못함
    # scipy.optimize.minimize의 BFGS callback 시그니처:
    # callback(xk) - xk는 현재 파라미터
    pass
```

**결과**: ❌ 불가능 (callback 시그니처 제한)

---

#### 시도 2: Monkey patching으로 내부 변수 접근
```python
# BFGS 내부 함수를 수정하여 H를 외부에 저장
import scipy.optimize._optimize as opt_module

original_bfgs = opt_module._minimize_bfgs

def patched_bfgs(fun, x0, args, jac, callback, **kwargs):
    # ❌ 실패: scipy 내부 구조가 복잡하고 버전마다 다름
    # 유지보수 불가능
    pass

opt_module._minimize_bfgs = patched_bfgs
```

**결과**: ❌ 불가능 (복잡성, 유지보수성 문제)

---

#### 시도 3: 정상 종료 후 hess_inv 저장
```python
try:
    result = optimize.minimize(...)
    
    # ✅ 정상 종료 시에만 작동
    if hasattr(result, 'hess_inv'):
        early_stopping_wrapper.best_hess_inv = result.hess_inv
        
except StopIteration as e:
    # ❌ 조기 종료 시: best_hess_inv는 None
    # 여기서 Hessian을 다시 계산해야 함
    pass
```

**결과**: ⚠️ 부분적 성공 (정상 종료 시만 작동)

---

### 4. **왜 L-BFGS-B도 안 되는가?**

L-BFGS-B는 제한된 메모리 BFGS로, **전체 Hessian 역행렬을 저장하지 않습니다**.

```python
# L-BFGS-B 구조
def _minimize_lbfgsb(fun, x0, ...):
    # Limited memory: 최근 m개 (s, y) 쌍만 저장
    m = 10  # 기본값
    s_history = []  # 최근 m개의 s = x_new - x
    y_history = []  # 최근 m개의 y = g_new - g
    
    # Hessian 역행렬을 명시적으로 저장하지 않음!
    # 대신 (s, y) 쌍으로부터 암묵적으로 계산
    
    for k in range(maxiter):
        # Two-loop recursion으로 H @ g 계산
        # 전체 H 행렬을 만들지 않음!
        p = two_loop_recursion(s_history, y_history, g)
        ...
    
    # ❌ result.hess_inv 없음!
    return OptimizeResult(x=x, ...)
```

**L-BFGS-B의 `result` 객체**:
```python
result = optimize.minimize(..., method='L-BFGS-B')
print(hasattr(result, 'hess_inv'))  # False!
```

**결과**: ❌ L-BFGS-B는 `hess_inv`를 제공하지 않음

---

## 💡 대안: BHHH 방법

### **왜 BHHH를 선택했는가?**

| 방법 | 추가 우도 계산 | 추가 그래디언트 계산 | 소요 시간 | Hessian 크기 | 구현 난이도 |
|------|---------------|---------------------|-----------|--------------|------------|
| **BFGS hess_inv** | 0회 | 0회 | 0초 | 전체 | ❌ 불가능 |
| **수치적 (대각)** | 41,209회 | 0회 | 10.5일 | 대각만 | 쉬움 |
| **수치적 (전체)** | 8,363,618회 | 0회 | 2,128일 | 전체 | 쉬움 |
| **Analytic (대각)** | 0회 | 202회 | 5시간 | 대각만 | 중간 |
| **BHHH** | 0회 | 50회 | 75분 | 전체 | 쉬움 |

**BHHH의 장점**:
1. ✅ **추가 우도 계산 0회**
2. ✅ **그래디언트 계산 50회만** (이미 구현된 함수 사용)
3. ✅ **전체 Hessian 행렬** (상관관계 포함)
4. ✅ **구현 간단** (개인별 gradient의 outer product)
5. ✅ **이론적으로 타당** (MLE에서 asymptotically equivalent)

---

## 🎯 결론

### **방안 1을 사용할 수 없는 이유**

1. **기술적 제약**:
   - ❌ scipy의 callback은 파라미터만 전달
   - ❌ BFGS 내부 상태는 private
   - ❌ 조기 종료 시 `result` 객체 미생성

2. **구조적 한계**:
   - ❌ L-BFGS-B는 `hess_inv` 제공 안 함
   - ❌ Monkey patching은 유지보수 불가능
   - ❌ scipy 내부 구조 수정 불가능

3. **실용적 문제**:
   - ❌ 정상 종료 시에만 작동 (조기 종료 시 실패)
   - ❌ 조기 종료가 목적인데 정상 종료를 기다려야 함
   - ❌ 모순적인 접근

### **최종 선택: BHHH 방법**

**이유**:
- ✅ 조기 종료와 완벽히 호환
- ✅ 추가 우도 계산 0회
- ✅ 75분 소요 (수치적 방법의 201배 빠름)
- ✅ 전체 Hessian 행렬 (더 정확)
- ✅ 구현 간단, 유지보수 쉬움

**Trade-off**:
- 그래디언트 계산 50회 필요 (75분)
- 하지만 수치적 방법(10.5일)보다 훨씬 빠름
- BFGS hess_inv(0초)보다 느리지만, **접근 불가능하므로 의미 없음**

---

## 📚 참고 자료

### **scipy.optimize.minimize 문서**
- Callback signature: `callback(xk)` - 파라미터만 전달
- BFGS: `result.hess_inv` 제공 (정상 종료 시)
- L-BFGS-B: `result.hess_inv` 제공 안 함

### **BHHH 방법 (Berndt-Hall-Hall-Hausman, 1974)**
- 논문: "Estimation and Inference in Nonlinear Structural Models"
- Hessian ≈ Σ_i (grad_i × grad_i^T)
- MLE에서 asymptotically equivalent to true Hessian
- 계산 효율적, 수치적으로 안정적

### **실무적 선택**
- 대부분의 통계 소프트웨어 (Stata, R, Python statsmodels)는 BHHH 또는 유사한 방법 사용
- 조기 종료 시 Hessian 재계산은 표준적인 접근
- BFGS hess_inv 접근은 정상 종료 시에만 가능 (조기 종료와 양립 불가)

