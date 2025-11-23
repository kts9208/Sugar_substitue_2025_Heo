# Trust Region 알고리즘 도입 방안

**날짜**: 2025-11-23  
**목적**: 동시추정에서 Trust Region 알고리즘 사용을 위한 기존 코드 활용 및 호환성 처리 방안

---

## 📋 요약

현재 동시추정 코드는 **L-BFGS-B** 알고리즘을 사용하고 있으며, **Trust Region** 알고리즘으로 전환하기 위해서는 **최소한의 코드 수정**만 필요합니다.

**핵심**: 기존 코드의 **else 분기**가 이미 Trust Region을 포함한 모든 optimizer를 지원하도록 설계되어 있습니다!

---

## ✅ 1. 현재 코드 구조 분석

### 1.1 Optimizer 분기 구조

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py" mode="EXCERPT">
````python
# Line 1291-1390: Optimizer별 분기
if self.config.estimation.optimizer == 'BHHH':
    # BHHH 전용 로직
    ...
elif self.config.estimation.optimizer == 'BFGS':
    # BFGS 전용 로직
    ...
elif self.config.estimation.optimizer == 'L-BFGS-B':
    # L-BFGS-B 전용 로직 (현재 사용 중)
    ...
else:
    # ✅ 모든 다른 optimizer (Trust Region 포함!)
    optimizer_options = {
        'maxiter': 200,
        'disp': True
    }
    
    result = optimize.minimize(
        early_stopping_wrapper.objective,
        initial_params_scaled,
        method=self.config.estimation.optimizer,  # ✅ 동적으로 optimizer 선택
        jac=jac_function,
        callback=early_stopping_wrapper.callback,
        options=optimizer_options
    )
````
</augment_code_snippet>

**핵심 발견**:
- ✅ **else 분기**가 이미 존재하여 모든 optimizer를 지원
- ✅ `method=self.config.estimation.optimizer`로 동적 선택
- ✅ Analytic gradient (`jac=jac_function`) 제공
- ✅ Callback 지원
- ✅ Parameter scaling 지원

---

## ✅ 2. Trust Region 알고리즘 선택

### 2.1 scipy.optimize.minimize의 Trust Region 방법

| Method | 이름 | Gradient 필요 | Hessian 필요 | Bounds 지원 |
|--------|------|--------------|-------------|-----------|
| **trust-constr** | Trust Region Constrained | ✅ 필수 | ⚠️ 선택 (없으면 근사) | ✅ 지원 |
| trust-ncg | Trust Region Newton-CG | ✅ 필수 | ✅ 필수 | ❌ 미지원 |
| trust-krylov | Trust Region Krylov | ✅ 필수 | ✅ 필수 | ❌ 미지원 |
| trust-exact | Trust Region Exact | ✅ 필수 | ✅ 필수 | ❌ 미지원 |

**권장**: **`trust-constr`**
- ✅ Gradient만 필요 (Hessian은 선택사항)
- ✅ Bounds 지원 (파라미터 범위 제약)
- ✅ 제약 조건 지원 (필요 시)
- ✅ Hessian 없으면 자동으로 SR1 또는 BFGS 근사 사용

---

### 2.2 trust-constr의 Hessian 처리

**scipy 공식 문서**:
```
trust-constr: Trust-region algorithm for constrained optimization.
- If Hessian is not provided, it uses SR1 or BFGS approximation.
- Supports bounds and general constraints.
```

**우리 코드에서**:
- ✅ Analytic gradient 제공 (`jac=jac_function`)
- ❌ Hessian 제공 안 함 (`hess=None`)
- → **trust-constr가 자동으로 BFGS 근사 사용**

---

## ✅ 3. 기존 코드 활용 방안

### 3.1 최소 수정 방안 (권장)

**수정 위치**: `scripts/test_gpu_batch_iclv.py`

**변경 전**:
```python
config = create_sugar_substitute_multi_lv_config(
    ...
    optimizer='L-BFGS-B',  # ← 현재
    ...
)
```

**변경 후**:
```python
config = create_sugar_substitute_multi_lv_config(
    ...
    optimizer='trust-constr',  # ← Trust Region으로 변경
    ...
)
```

**끝!** 다른 코드 수정 불필요!

---

### 3.2 작동 원리

1. **Config 생성**:
   ```python
   config.estimation.optimizer = 'trust-constr'
   ```

2. **Estimator에서 분기**:
   ```python
   # simultaneous_estimator_fixed.py, Line 1377-1390
   else:  # ← 'trust-constr'는 여기로 진입
       optimizer_options = {
           'maxiter': 200,
           'disp': True
       }
       
       result = optimize.minimize(
           early_stopping_wrapper.objective,
           initial_params_scaled,
           method='trust-constr',  # ← 동적으로 설정됨
           jac=jac_function,       # ← Analytic gradient 제공
           callback=early_stopping_wrapper.callback,
           options=optimizer_options
       )
   ```

3. **scipy가 자동 처리**:
   - Hessian 없음 → BFGS 근사 자동 사용
   - Bounds 있으면 자동 적용
   - Trust Region 알고리즘 실행

---

## ✅ 4. 호환성 처리

### 4.1 Gradient 호환성

**현재 코드**:
- ✅ Analytic gradient 이미 구현됨
- ✅ `jac=jac_function`으로 제공
- ✅ Trust Region이 그대로 사용 가능

**검증**:
```python
# simultaneous_estimator_fixed.py, Line 1387
jac=jac_function,  # ← Trust Region이 사용
```

---

### 4.2 Bounds 호환성

**현재 코드**:
- ✅ Bounds 이미 정의됨
- ⚠️ L-BFGS-B 분기에서만 전달됨

**수정 필요**:
```python
# simultaneous_estimator_fixed.py, Line 1383-1390
else:
    optimizer_options = {
        'maxiter': 200,
        'disp': True
    }
    
    result = optimize.minimize(
        early_stopping_wrapper.objective,
        initial_params_scaled,
        method=self.config.estimation.optimizer,
        jac=jac_function,
        bounds=bounds,  # ← 추가 필요!
        callback=early_stopping_wrapper.callback,
        options=optimizer_options
    )
```

---

### 4.3 Callback 호환성

**현재 코드**:
- ✅ Callback 이미 구현됨
- ✅ `callback=early_stopping_wrapper.callback`로 제공
- ✅ Trust Region이 그대로 사용 가능

**검증**:
```python
# simultaneous_estimator_fixed.py, Line 1023-1088
def callback(self, xk):
    """
    BFGS callback - 매 Major iteration마다 호출됨
    Trust Region도 동일하게 작동
    """
    ...
```

---

### 4.4 Hessian 역행렬 호환성

**trust-constr의 Hessian 역행렬 제공 여부**:
- ❌ `result.hess_inv` 제공 안 함
- ✅ 대신 BHHH 방법으로 계산 (이미 구현됨)

**현재 코드**:
```python
# simultaneous_estimator_fixed.py, Line 1506-1520
if hasattr(result, 'hess_inv') and result.hess_inv is not None:
    # L-BFGS-B, BFGS가 제공한 hess_inv 사용
    ...
else:
    # ✅ Trust Region은 여기로 진입
    # → BHHH 방법으로 Hessian 역행렬 계산
    hess_inv_bhhh = self._compute_bhhh_hessian_inverse(...)
```

**결론**: ✅ 호환성 문제 없음 (BHHH fallback 이미 구현됨)

---

### 4.5 Parameter Scaling 호환성

**현재 코드**:
- ✅ Parameter scaling 이미 구현됨
- ✅ 모든 optimizer에 동일하게 적용
- ✅ Trust Region이 그대로 사용 가능

**검증**:
```python
# simultaneous_estimator_fixed.py, Line 1385
initial_params_scaled,  # ← 스케일된 파라미터 사용
```

---

## 📝 5. 구현 단계

### 5.1 단계 1: 최소 수정 (즉시 테스트 가능)

**파일**: `scripts/test_gpu_batch_iclv.py`

**수정**:
```python
# Line 193
optimizer='trust-constr',  # L-BFGS-B → trust-constr
```

**실행**:
```bash
python scripts/test_gpu_batch_iclv.py
```

**예상 결과**:
- ✅ 정상 실행
- ⚠️ Bounds가 전달되지 않아 경고 가능
- ✅ BHHH 방법으로 Hessian 역행렬 계산

---

### 5.2 단계 2: Bounds 추가 (권장)

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**수정 위치**: Line 1383-1390

**변경 전**:
```python
else:
    optimizer_options = {
        'maxiter': 200,
        'disp': True
    }
    
    result = optimize.minimize(
        early_stopping_wrapper.objective,
        initial_params_scaled,
        method=self.config.estimation.optimizer,
        jac=jac_function,
        callback=early_stopping_wrapper.callback,
        options=optimizer_options
    )
```

**변경 후**:
```python
else:
    optimizer_options = {
        'maxiter': 200,
        'disp': True
    }
    
    result = optimize.minimize(
        early_stopping_wrapper.objective,
        initial_params_scaled,
        method=self.config.estimation.optimizer,
        jac=jac_function,
        bounds=bounds,  # ← 추가
        callback=early_stopping_wrapper.callback,
        options=optimizer_options
    )
```

---

### 5.3 단계 3: Trust Region 전용 옵션 추가 (선택사항)

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**추가 위치**: Line 1377 이후

**코드**:
```python
elif self.config.estimation.optimizer == 'trust-constr':
    optimizer_options = {
        'maxiter': 200,
        'disp': True,
        'gtol': 1e-5,      # Gradient tolerance
        'xtol': 1e-6,      # Parameter tolerance
        'barrier_tol': 1e-8,  # Barrier parameter tolerance
        'initial_tr_radius': 1.0,  # Initial trust region radius
        'max_tr_radius': 1000.0,   # Maximum trust region radius
        'verbose': 2       # Verbosity level (0, 1, 2, 3)
    }
    
    self.iteration_logger.info(
        f"Trust Region (trust-constr) 옵션:\n"
        f"  - maxiter: {optimizer_options['maxiter']}\n"
        f"  - gtol: {optimizer_options['gtol']}\n"
        f"  - xtol: {optimizer_options['xtol']}\n"
        f"  - initial_tr_radius: {optimizer_options['initial_tr_radius']}\n"
        f"  - Hessian: BFGS 근사 (자동)\n"
        f"\n"
        f"  💡 Trust Region은 평탄한 영역에서 더 안정적입니다."
    )
    
    result = optimize.minimize(
        early_stopping_wrapper.objective,
        initial_params_scaled,
        method='trust-constr',
        jac=jac_function,
        bounds=bounds,
        callback=early_stopping_wrapper.callback,
        options=optimizer_options
    )
```

---

## 📊 6. 비교: L-BFGS-B vs Trust Region

| 항목 | L-BFGS-B | trust-constr |
|------|----------|--------------|
| **알고리즘** | Quasi-Newton (Limited Memory) | Trust Region |
| **Hessian 근사** | Limited Memory BFGS (m=10) | BFGS (전체 메모리) |
| **Bounds 지원** | ✅ 지원 | ✅ 지원 |
| **제약 조건** | ❌ 미지원 | ✅ 지원 |
| **평탄한 영역** | ⚠️ 불안정 | ✅ 안정적 |
| **Hessian 부정확 시** | ⚠️ 탐색 방향 0 가능 | ✅ Trust Radius로 제한 |
| **수렴 속도** | ✅ 빠름 | ⚠️ 느림 |
| **메모리 사용** | ✅ 적음 (m=10) | ⚠️ 많음 (전체) |
| **hess_inv 제공** | ✅ 제공 (LbfgsInvHessProduct) | ❌ 미제공 |
| **기존 코드 호환** | ✅ 완벽 | ✅ 완벽 (BHHH fallback) |

---

## 💡 7. 권장 사항

### 7.1 즉시 시도 가능

1. **파라미터 스케일링 활성화** (가장 우선)
   ```python
   use_parameter_scaling = True
   ```

2. **Trust Region 시도** (L-BFGS-B 실패 시)
   ```python
   optimizer='trust-constr'
   ```

---

### 7.2 단계별 접근

**Phase 1**: 파라미터 스케일링만 활성화
- L-BFGS-B + Parameter Scaling
- 예상: Hessian 근사 정확도 향상

**Phase 2**: Trust Region 시도
- trust-constr + Parameter Scaling
- 예상: 평탄한 영역에서 더 안정적

**Phase 3**: 초기값 개선
- 순차추정 2단계 결과 사용
- 예상: 더 나은 local minimum 탐색

---

## 📋 8. 요약

### ✅ 기존 코드 활용

| 항목 | 상태 | 설명 |
|------|------|------|
| **Optimizer 분기** | ✅ 준비됨 | else 분기가 모든 optimizer 지원 |
| **Analytic Gradient** | ✅ 준비됨 | jac=jac_function 제공 |
| **Bounds** | ⚠️ 추가 필요 | else 분기에 bounds 추가 |
| **Callback** | ✅ 준비됨 | callback 이미 구현됨 |
| **Parameter Scaling** | ✅ 준비됨 | 모든 optimizer에 적용 |
| **Hessian 역행렬** | ✅ 준비됨 | BHHH fallback 이미 구현됨 |

### ✅ 호환성

- ✅ **Gradient**: 완벽 호환
- ⚠️ **Bounds**: 1줄 추가 필요
- ✅ **Callback**: 완벽 호환
- ✅ **Hessian**: BHHH fallback으로 자동 처리
- ✅ **Parameter Scaling**: 완벽 호환

### ✅ 최소 수정

**1줄 수정**으로 즉시 테스트 가능:
```python
optimizer='trust-constr'  # scripts/test_gpu_batch_iclv.py, Line 193
```

**권장 수정** (1줄 추가):
```python
bounds=bounds,  # simultaneous_estimator_fixed.py, Line 1388
```

---

**Trust Region 도입 방안 분석 완료!** 🎯

