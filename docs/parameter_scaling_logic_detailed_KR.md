# 동시추정 파라미터 스케일링 로직 상세 설명

## 📋 개요

동시추정 코드는 **Apollo R 패키지 스타일의 파라미터 스케일링**을 사용하여 최적화 안정성을 향상시킵니다.

**핵심 아이디어:**
- 파라미터를 내부 스케일(Internal Scale)로 변환하여 최적화
- 그래디언트 크기를 균형있게 조정
- 수치 안정성 향상

---

## 1. 파라미터 스케일링 개념

### 1.1 External vs Internal 파라미터

**External 파라미터 (θ_external):**
- 실제 모델 파라미터 (해석 가능한 값)
- 예: `asc_sugar = 1.5`, `beta_price = -0.5`

**Internal 파라미터 (θ_internal):**
- 최적화에 사용되는 스케일링된 값
- 예: `asc_sugar_internal = 1.5 / 1.0 = 1.5`

### 1.2 스케일링 공식

**파라미터 스케일링:**
```
θ_internal = θ_external / scale
θ_external = θ_internal × scale
```

**그래디언트 스케일링 (체인룰):**
```
∂LL/∂θ_internal = ∂LL/∂θ_external × scale
```

**이유:**
```
θ_external = θ_internal × scale
∂θ_external/∂θ_internal = scale

체인룰:
∂LL/∂θ_internal = (∂LL/∂θ_external) × (∂θ_external/∂θ_internal)
                 = (∂LL/∂θ_external) × scale
```

---

## 2. ParameterScaler 클래스

### 2.1 클래스 구조

**파일 위치:** `src/analysis/hybrid_choice_model/iclv_models/parameter_scaler.py`

**주요 메서드:**
1. `__init__()`: 스케일 팩터 초기화
2. `scale_parameters()`: External → Internal
3. `unscale_parameters()`: Internal → External
4. `scale_gradient()`: 그래디언트 스케일링

### 2.2 초기화 로직

```python
class ParameterScaler:
    def __init__(self, initial_params, param_names, custom_scales=None, logger=None):
        self.param_names = param_names
        self.scales = np.ones(len(initial_params))
        
        if custom_scales is not None:
            # ✅ Custom scales 사용 (gradient 균형 최적화)
            for i, name in enumerate(param_names):
                if name in custom_scales:
                    self.scales[i] = custom_scales[name]
                else:
                    # Apollo 방식: abs(initial_value)
                    value = initial_params[i]
                    if abs(value) > 1e-10:
                        self.scales[i] = abs(value)
                    else:
                        self.scales[i] = 1.0
        else:
            # Apollo R 방식 (기본)
            for i, value in enumerate(initial_params):
                if abs(value) > 1e-10:
                    self.scales[i] = abs(value)
                else:
                    self.scales[i] = 1.0
```

**Apollo R 방식:**
- 초기값이 0이 아닌 파라미터: `scale = abs(initial_value)`
- 초기값이 0인 파라미터: `scale = 1.0` (스케일링 안 함)

**Custom Scales 방식:**
- 각 파라미터별로 수동 설정된 스케일 사용
- 그래디언트 크기 균형 최적화

---

## 3. 커스텀 스케일 설정

### 3.1 _get_custom_scales() 메서드

**파일 위치:** `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`
**라인:** 1886-1982

**목표:** 모든 internal gradient를 50~1,000 범위로 조정

### 3.2 파라미터별 커스텀 스케일

#### ① 측정모델 파라미터 (동시추정에서는 고정)

**요인적재량 (zeta):**
```python
if name.startswith('zeta_'):
    if 'health_concern' in name:
        custom_scales[name] = 0.024
    elif 'perceived_benefit' in name:
        custom_scales[name] = 0.050
    elif 'perceived_price' in name:
        custom_scales[name] = 0.120
    elif 'nutrition_knowledge' in name:
        custom_scales[name] = 0.022
    elif 'purchase_intention' in name:
        custom_scales[name] = 0.083
    else:
        custom_scales[name] = 0.05  # 기본값
```

**오차분산 (sigma_sq):**
```python
elif name.startswith('sigma_sq_'):
    if 'health_concern' in name:
        custom_scales[name] = 0.034
    elif 'perceived_benefit' in name:
        custom_scales[name] = 0.036
    elif 'perceived_price' in name:
        custom_scales[name] = 0.023
    elif 'nutrition_knowledge' in name:
        custom_scales[name] = 0.046
    elif 'purchase_intention' in name:
        custom_scales[name] = 0.026
    else:
        custom_scales[name] = 0.03  # 기본값
```

#### ② 구조모델 파라미터 (추정 대상)

**경로계수 (gamma):**
```python
elif name.startswith('gamma_'):
    # ✅ 구조모델 그래디언트가 극도로 작은 문제 해결
    # 잠재변수가 표준정규분포 (평균 ≈ 0)로 생성되어 그래디언트 ≈ 0
    # → 더 큰 스케일 팩터로 그래디언트를 증폭
    custom_scales[name] = 50.0  # 0.5 → 50.0 (100배 증가)
```

**특징:**
- ⚠️ **매우 큰 스케일 (50.0)**
- 이유: LV가 표준정규분포 (평균 ≈ 0) → 그래디언트 매우 작음
- 효과: 그래디언트를 100배 증폭하여 최적화 가능하게 함

#### ③ 선택모델 파라미터 (추정 대상)

**Beta (속성 계수):**
```python
elif name.startswith('beta_'):
    if name == 'beta_intercept':
        custom_scales[name] = 0.290
    elif name == 'beta_sugar_free':
        custom_scales[name] = 0.230
    elif name == 'beta_health_label':
        custom_scales[name] = 0.220
    elif name == 'beta_price':
        custom_scales[name] = 0.056  # ⚠️ 가장 작은 스케일
    else:
        custom_scales[name] = 0.2  # 기본값
```

**Lambda (LV 계수):**
```python
elif name.startswith('lambda_'):
    if name == 'lambda_main':
        custom_scales[name] = 0.890
    elif name == 'lambda_mod_perceived_price':
        custom_scales[name] = 0.470
    elif name == 'lambda_mod_nutrition_knowledge':
        custom_scales[name] = 1.200
    else:
        custom_scales[name] = 0.5  # 기본값
```

**ASC (대안별 상수) - 현재 모델:**
```python
# 현재 모델에서는 asc_sugar, asc_sugar_free 사용
# 커스텀 스케일 설정 없음 → Apollo 방식 사용
# scale = abs(initial_value) = abs(0.1) = 0.1
```

---

## 4. 스케일링 적용 흐름

### 4.1 초기화 단계

**파일:** `simultaneous_estimator_fixed.py`
**라인:** 505-527

```python
# 1. Custom scales 생성
custom_scales = self._get_custom_scales(param_names)

# 2. ParameterScaler 초기화
self.param_scaler = ParameterScaler(
    initial_params=initial_params,
    param_names=param_names,
    custom_scales=custom_scales,
    logger=self.iteration_logger
)

# 3. 초기 파라미터 스케일링 (External → Internal)
initial_params_scaled = self.param_scaler.scale_parameters(initial_params)
```

### 4.2 최적화 단계

**우도 계산 (Likelihood):**
```python
def negative_log_likelihood_func(params_scaled):
    # 1. Internal → External 변환
    params_external = param_context.unscale_parameters(params_scaled)
    
    # 2. External 파라미터로 우도 계산
    ll = compute_likelihood(params_external)
    
    return -ll
```

**그래디언트 계산 (Gradient):**
```python
def gradient_func(params_scaled):
    # 1. Internal → External 변환
    params_external = param_context.unscale_parameters(params_scaled)
    
    # 2. External 파라미터로 그래디언트 계산
    grad_external = compute_gradient(params_external)
    
    # 3. 그래디언트 스케일링 (External → Internal)
    grad_internal = param_context.scale_gradient(grad_external)
    
    return -grad_internal
```

### 4.3 최적화 실행

```python
# L-BFGS-B 최적화 (Internal 파라미터 공간에서)
result = minimize(
    fun=negative_log_likelihood_func,
    x0=initial_params_scaled,  # Internal 파라미터
    jac=gradient_func,
    method='L-BFGS-B',
    ...
)

# 최종 결과 언스케일링 (Internal → External)
final_params_external = self.param_scaler.unscale_parameters(result.x)
```

---

## 5. 스케일링 효과 예시

### 5.1 구조모델 파라미터 (gamma)

**초기값:**
```
gamma_HC_to_PB = 0.1 (External)
```

**스케일링:**
```
scale = 50.0
gamma_HC_to_PB_internal = 0.1 / 50.0 = 0.002
```

**그래디언트:**
```
grad_external = 0.01 (매우 작음)
grad_internal = 0.01 × 50.0 = 0.5 (증폭됨)
```

**효과:**
- ✅ 그래디언트가 50배 증폭
- ✅ 최적화 알고리즘이 파라미터 업데이트 가능

---

### 5.2 선택모델 파라미터 (beta_price)

**초기값:**
```
beta_price = 0.1 (External)
```

**스케일링:**
```
scale = 0.056
beta_price_internal = 0.1 / 0.056 = 1.786
```

**그래디언트:**
```
grad_external = 2000 (매우 큼, 가격 스케일 때문)
grad_internal = 2000 × 0.056 = 112 (감소됨)
```

**효과:**
- ✅ 그래디언트가 1/18로 감소
- ✅ 다른 파라미터와 균형

---

## 6. 스케일 팩터 요약표

| 파라미터 | 커스텀 스케일 | 효과 | 비고 |
|----------|--------------|------|------|
| **gamma_*** | 50.0 | 그래디언트 증폭 (×50) | 구조모델 |
| **beta_price** | 0.056 | 그래디언트 감소 (×0.056) | 가격 변수 |
| **beta_health_label** | 0.220 | 그래디언트 감소 (×0.22) | 이진 변수 |
| **lambda_main** | 0.890 | 그래디언트 감소 (×0.89) | LV 주효과 |
| **zeta_*** | 0.022~0.120 | 그래디언트 감소 | 측정모델 (고정) |
| **sigma_sq_*** | 0.023~0.046 | 그래디언트 감소 | 측정모델 (고정) |

---

## 7. 스케일링의 장단점

### 7.1 장점

1. ✅ **그래디언트 균형**
   - 모든 파라미터의 그래디언트를 비슷한 크기로 조정
   - L-BFGS-B 최적화 안정성 향상

2. ✅ **수치 안정성**
   - Hessian 행렬의 조건수(condition number) 감소
   - 수치 오차 감소

3. ✅ **수렴 속도**
   - 균형잡힌 그래디언트로 더 빠른 수렴

### 7.2 단점

1. ⚠️ **복잡성 증가**
   - 스케일링/언스케일링 로직 추가
   - 디버깅 어려움

2. ⚠️ **수동 튜닝 필요**
   - 커스텀 스케일 값을 수동으로 설정
   - 모델마다 다른 스케일 필요

3. ⚠️ **해석 주의**
   - Internal 파라미터는 해석 불가
   - 항상 External 파라미터로 변환 필요

---

## 8. 주요 코드 위치

| 항목 | 파일 | 라인 |
|------|------|------|
| **ParameterScaler 클래스** | `parameter_scaler.py` | 16-202 |
| **커스텀 스케일 설정** | `simultaneous_estimator_fixed.py` | 1886-1982 |
| **스케일러 초기화** | `simultaneous_estimator_fixed.py` | 505-527 |
| **우도 계산 (언스케일링)** | `simultaneous_estimator_fixed.py` | 2709 |
| **그래디언트 스케일링** | `simultaneous_estimator_fixed.py` | 716, 2803, 2859 |

---

## 9. 사용 예시

### 9.1 스케일링 활성화

```python
# config에서 설정
config.estimation.use_parameter_scaling = True

# 자동으로 커스텀 스케일 적용
estimator.estimate(...)
```

### 9.2 스케일링 비활성화

```python
# config에서 설정
config.estimation.use_parameter_scaling = False

# 모든 스케일을 1.0으로 설정 (스케일링 안 함)
estimator.estimate(...)
```

---

## 10. 결론

**동시추정 파라미터 스케일링:**
1. ✅ Apollo R 스타일 기반
2. ✅ 커스텀 스케일로 그래디언트 균형 최적화
3. ✅ 구조모델 그래디언트 증폭 (×50)
4. ✅ 선택모델 그래디언트 감소 (가격: ×0.056)
5. ✅ 최적화 안정성 및 수렴 속도 향상

**핵심 공식:**
```
θ_internal = θ_external / scale
∂LL/∂θ_internal = ∂LL/∂θ_external × scale
```

**추가 질문이 있으시면 말씀해주세요!**

