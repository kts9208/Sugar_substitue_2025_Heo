# 동시추정 리팩토링 계획

## 📋 전제 조건

**동시추정은 항상 측정모델 파라미터를 고정한다**

이는 설정이 아니라 동시추정의 정의입니다.

---

## 🎯 리팩토링 가능 요소

### 1. `_measurement_params_fixed` 플래그 제거

**현재 (불필요한 조건문)**:
```python
if self._measurement_params_fixed:
    # 측정모델 고정 로직
else:
    # 전체 파라미터 최적화 로직
```

**개선 후 (조건문 제거)**:
```python
# 항상 측정모델 고정
# 측정모델 고정 로직만 실행
```

**영향받는 코드**:
- Line 438-441: JointGradient 초기화
- Line 461-488: 초기 파라미터 분리
- Line 533: ParameterContext 생성
- Line 647: Bounds 계산

---

### 2. ParameterContext 단순화

**현재**:
```python
class ParameterContext:
    def __init__(self, ..., fixed_measurement_params: Optional[np.ndarray] = None):
        self.measurement_params_fixed = (fixed_measurement_params is not None)
        
        if self.measurement_params_fixed:
            # 고정 로직
        else:
            # 전체 최적화 로직
```

**개선 후**:
```python
class ParameterContext:
    def __init__(self, ..., fixed_measurement_params: np.ndarray):
        # fixed_measurement_params는 필수 (Optional 제거)
        # 조건문 제거 (항상 고정)
        self.fixed_measurement_params = fixed_measurement_params
        self.n_measurement = len(fixed_measurement_params)
```

---

### 3. 파라미터 이름 생성 단순화

**현재**:
```python
if self._measurement_params_fixed:
    param_names = self.param_manager.get_parameter_names(
        ..., exclude_measurement=True
    )
else:
    param_names = self.param_manager.get_parameter_names(...)
```

**개선 후**:
```python
# 항상 측정모델 제외
param_names = self.param_manager.get_parameter_names(
    ..., exclude_measurement=True
)
```

---

### 4. Bounds 계산 단순화

**현재**:
```python
bounds = self._get_parameter_bounds(
    ..., exclude_measurement=self._measurement_params_fixed
)
```

**개선 후**:
```python
# 항상 측정모델 제외
bounds = self._get_parameter_bounds(
    ..., exclude_measurement=True
)
```

---

### 5. 초기 파라미터 처리 단순화

**현재**:
```python
initial_params_full = self._get_initial_parameters(...)

if self._measurement_params_fixed:
    fixed_measurement_params, initial_params_opt = \
        self.param_manager.split_measurement_params(initial_params_full, ...)
    initial_params = initial_params_opt
    self._fixed_measurement_params = fixed_measurement_params
else:
    initial_params = initial_params_full
    self._fixed_measurement_params = None
```

**개선 후**:
```python
initial_params_full = self._get_initial_parameters(...)

# 항상 측정모델 파라미터 분리
fixed_measurement_params, initial_params_opt = \
    self.param_manager.split_measurement_params(initial_params_full, ...)

# 최적화 파라미터만 사용
initial_params = initial_params_opt
self._fixed_measurement_params = fixed_measurement_params
```

---

### 6. ParameterManager 메서드 단순화

**현재**:
```python
def get_parameter_names(self, ..., exclude_measurement: bool = False):
    names = []
    
    if not exclude_measurement:
        names.extend(self._get_measurement_param_names(...))
    
    names.extend(self._get_structural_param_names(...))
    names.extend(self._get_choice_param_names(...))
    
    return names
```

**개선 후 (동시추정 전용 메서드)**:
```python
def get_optimized_parameter_names(self, structural_model, choice_model):
    """동시추정용: 구조모델 + 선택모델 파라미터만"""
    names = []
    names.extend(self._get_structural_param_names(structural_model))
    names.extend(self._get_choice_param_names(choice_model))
    return names
```

---

## 📊 리팩토링 효과

### Before
- 조건문: 6개
- 불필요한 분기: 6개
- Optional 파라미터: 3개
- 코드 복잡도: 높음

### After
- 조건문: 0개
- 불필요한 분기: 0개
- Optional 파라미터: 0개
- 코드 복잡도: 낮음

---

## 🚀 추가 개선 사항

### 1. 클래스 분리

**현재**: `SimultaneousEstimator`가 측정모델 고정/비고정 모두 처리

**개선**: 동시추정 전용 클래스 생성
```python
class SimultaneousEstimatorWithFixedMeasurement:
    """
    동시추정 전용 Estimator
    
    전제:
    - 측정모델 파라미터는 CFA 결과로 고정
    - 구조모델 + 선택모델만 추정
    """
```

### 2. 초기값 검증 강화

```python
def estimate(self, ..., initial_params):
    # CFA 결과 필수 검증
    if 'measurement' not in initial_params:
        raise ValueError(
            "동시추정은 CFA 결과가 필수입니다!\n"
            "initial_params에 'measurement' 키를 포함해야 합니다."
        )
    
    # 측정모델 파라미터 완전성 검증
    self._validate_measurement_params(initial_params['measurement'], ...)
```

### 3. 메서드 이름 명확화

**Before**:
- `get_parameter_names(exclude_measurement=True)`
- `get_parameter_bounds(exclude_measurement=True)`

**After**:
- `get_optimized_parameter_names()` (측정모델 제외가 기본)
- `get_optimized_parameter_bounds()` (측정모델 제외가 기본)

---

## 📝 구현 우선순위

### Phase 1: 핵심 단순화 (높은 우선순위)
1. ✅ API 단순화 (`measurement_params_fixed` 파라미터 제거)
2. ⬜ 조건문 제거 (6개 조건문 → 0개)
3. ⬜ ParameterContext 단순화 (Optional 제거)

### Phase 2: 메서드 정리 (중간 우선순위)
4. ⬜ 파라미터 이름/bounds 생성 단순화
5. ⬜ 초기 파라미터 처리 단순화

### Phase 3: 구조 개선 (낮은 우선순위)
6. ⬜ ParameterManager 전용 메서드 추가
7. ⬜ 초기값 검증 강화
8. ⬜ 메서드 이름 명확화

---

## ⚠️ 주의사항

### 1. 순차추정과의 호환성

순차추정(Sequential Estimation)은 측정모델도 추정하므로, 리팩토링 시 순차추정 코드에 영향을 주지 않도록 주의해야 합니다.

**해결 방안**:
- `SimultaneousEstimator`와 `SequentialEstimator`를 완전히 분리
- 공통 로직은 별도 모듈로 추출

### 2. 기존 테스트 코드

기존 테스트 코드가 `measurement_params_fixed=True`를 명시적으로 전달하는 경우, 이를 제거해야 합니다.

### 3. 문서 업데이트

리팩토링 후 다음 문서를 업데이트해야 합니다:
- `docs/measurement_params_fixed_optimization.md`
- API 문서
- 사용 예제

---

## 🎯 최종 목표

```python
# ✅ 간결하고 명확한 API
estimator = SimultaneousGPUBatchEstimator(...)

result = estimator.estimate(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_model=choice_model,
    initial_params={
        'measurement': {...},  # CFA 결과 (필수)
        'structural': {...},
        'choice': {...}
    }
)

# ✅ 내부 구현
# - 조건문 없음 (항상 측정모델 고정)
# - Optional 파라미터 없음 (명확한 타입)
# - 단순하고 읽기 쉬운 코드
```


