# Optimizer 선정 로직 통합 방안

**날짜**: 2025-11-23  
**문제**: Optimizer 선정이 여러 곳에서 이루어져 충돌 발생

---

## 📋 문제 분석

### 1. 현재 문제점

**증상**:
- Config에서 `optimizer='trust-constr'` 설정
- 실제 실행 시 `Nelder-Mead` 사용됨

**원인**:
```python
# Line 432, 706
use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B', 'BHHH']
```

- `trust-constr`는 리스트에 없음 → `use_gradient = False`
- `use_gradient = False` → Nelder-Mead 사용 (Line 1620-1634)

---

### 2. Optimizer 선정 로직 위치

| 위치 | 라인 | 역할 | 문제 |
|------|------|------|------|
| **Line 432** | `use_gradient = ...` | Gradient calculator 초기화 여부 결정 | ❌ Trust Region 미포함 |
| **Line 706** | `use_gradient = ...` | Optimizer 분기 결정 | ❌ Trust Region 미포함 |
| **Line 1231** | `if use_gradient:` | Gradient-based vs Gradient-free 분기 | ❌ 잘못된 분기 |
| **Line 1291-1392** | Optimizer별 분기 | 실제 optimize.minimize() 호출 | ✅ 정상 (else 분기 존재) |

---

## 🔧 통합 방안

### 방안 1: Gradient-based Optimizer 리스트 확장 (권장)

**핵심 아이디어**:
- `use_gradient` 결정 로직을 **Gradient-free optimizer 리스트**로 변경
- Trust Region, Newton-CG 등 모든 gradient-based optimizer 자동 지원

**수정 위치**: Line 432, 706

**Before**:
```python
use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B', 'BHHH']
```

**After**:
```python
# Gradient-free optimizer 리스트 (명시적)
GRADIENT_FREE_OPTIMIZERS = ['Nelder-Mead', 'Powell', 'COBYLA']

# Gradient-based optimizer 자동 판단
use_gradient = self.config.estimation.optimizer not in GRADIENT_FREE_OPTIMIZERS
```

**장점**:
- ✅ 새로운 gradient-based optimizer 추가 시 자동 지원
- ✅ 명시적이고 유지보수 쉬움
- ✅ Trust Region, Newton-CG, SLSQP 등 모두 자동 지원

**단점**:
- ⚠️ 잘못된 optimizer 이름 입력 시 gradient-based로 간주됨

---

### 방안 2: Optimizer 타입 명시적 정의

**핵심 아이디어**:
- Optimizer를 타입별로 명시적으로 분류
- Config에서 optimizer 타입 검증

**수정 위치**: `iclv_config.py`, Line 432, 706

**EstimationConfig 수정**:
```python
# iclv_config.py
GRADIENT_BASED_OPTIMIZERS = [
    'BFGS', 'L-BFGS-B', 'BHHH', 
    'trust-constr', 'trust-ncg', 'trust-exact', 'trust-krylov',
    'Newton-CG', 'CG', 'SLSQP', 'dogleg'
]

GRADIENT_FREE_OPTIMIZERS = [
    'Nelder-Mead', 'Powell', 'COBYLA'
]

@dataclass
class EstimationConfig:
    optimizer: str = 'BFGS'
    
    def __post_init__(self):
        """Optimizer 검증"""
        all_optimizers = GRADIENT_BASED_OPTIMIZERS + GRADIENT_FREE_OPTIMIZERS
        if self.optimizer not in all_optimizers:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer}. "
                f"Supported: {all_optimizers}"
            )
    
    def is_gradient_based(self) -> bool:
        """Gradient-based optimizer 여부"""
        return self.optimizer in GRADIENT_BASED_OPTIMIZERS
```

**simultaneous_estimator_fixed.py 수정**:
```python
# Line 432, 706
use_gradient = self.config.estimation.is_gradient_based()
```

**장점**:
- ✅ 가장 명시적이고 안전
- ✅ Optimizer 검증 자동화
- ✅ 잘못된 optimizer 이름 조기 발견

**단점**:
- ⚠️ 새 optimizer 추가 시 리스트 업데이트 필요

---

### 방안 3: Scipy Optimizer 메타데이터 활용

**핵심 아이디어**:
- Scipy의 optimizer 정보를 동적으로 확인
- Gradient 필요 여부 자동 판단

**수정 위치**: Line 432, 706

**구현**:
```python
def requires_gradient(optimizer_name: str) -> bool:
    """
    Optimizer가 gradient를 필요로 하는지 확인
    
    Scipy의 minimize() 함수 시그니처를 확인하여 판단
    """
    # Gradient-free optimizer (명시적)
    gradient_free = ['Nelder-Mead', 'Powell', 'COBYLA']
    
    if optimizer_name in gradient_free:
        return False
    
    # 나머지는 모두 gradient-based로 간주
    # (Trust Region, BFGS, L-BFGS-B, Newton-CG, CG, SLSQP 등)
    return True

# Line 432, 706
use_gradient = requires_gradient(self.config.estimation.optimizer)
```

**장점**:
- ✅ 간단하고 확장 가능
- ✅ 새 optimizer 자동 지원

**단점**:
- ⚠️ Scipy 버전 변경 시 영향 받을 수 있음

---

## 🎯 권장 방안: **방안 2 (명시적 정의)**

**이유**:
1. ✅ **안전성**: Optimizer 검증으로 오타 방지
2. ✅ **명시성**: 지원하는 optimizer 명확히 문서화
3. ✅ **유지보수**: 새 optimizer 추가 시 한 곳만 수정
4. ✅ **확장성**: 향후 optimizer별 특수 처리 가능

---

## 📝 구현 단계

### Step 1: `iclv_config.py` 수정

**파일**: `src/analysis/hybrid_choice_model/iclv_models/iclv_config.py`

**추가 내용** (Line 150 이전):
```python
# Optimizer 분류
GRADIENT_BASED_OPTIMIZERS = [
    # Quasi-Newton methods
    'BFGS', 'L-BFGS-B',
    
    # Newton methods
    'Newton-CG', 'CG',
    
    # Trust Region methods
    'trust-constr', 'trust-ncg', 'trust-exact', 'trust-krylov', 'dogleg',
    
    # Sequential Quadratic Programming
    'SLSQP',
    
    # Custom methods
    'BHHH'  # Berndt-Hall-Hall-Hausman (Newton-CG with OPG Hessian)
]

GRADIENT_FREE_OPTIMIZERS = [
    'Nelder-Mead',  # Simplex method
    'Powell',       # Powell's method
    'COBYLA'        # Constrained Optimization BY Linear Approximation
]
```

**EstimationConfig 수정** (Line 153-195):
```python
@dataclass
class EstimationConfig:
    """추정 설정"""
    
    # 추정 방법
    method: Literal['simultaneous', 'sequential'] = 'simultaneous'
    
    # 시뮬레이션 설정
    n_draws: int = 1000
    draw_type: Literal['halton', 'random', 'mlhs'] = 'halton'
    scramble_halton: bool = True
    
    # 최적화 설정
    optimizer: str = 'BFGS'
    max_iterations: int = 2000
    convergence_tolerance: float = 1e-6

    # Gradient 설정 (Apollo 방식)
    use_analytic_gradient: bool = True  # True: analytic gradient, False: numerical gradient

    # Parameter Scaling 설정
    use_parameter_scaling: bool = True  # True: parameter scaling 활성화, False: 비활성화

    # Data Standardization 설정
    standardize_choice_attributes: bool = True
    
    # Gradient 로깅 설정
    gradient_log_level: Literal['MINIMAL', 'MODERATE', 'DETAILED'] = 'DETAILED'

    # 병렬처리 설정
    use_parallel: bool = False
    n_cores: Optional[int] = None

    # 표준오차 계산
    calculate_se: bool = True
    se_method: Literal['hessian', 'bootstrap', 'robust'] = 'hessian'

    # 부트스트랩 설정
    n_bootstrap: int = 500

    # 조기 종료 설정
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_tol: float = 1e-6
    
    def __post_init__(self):
        """Optimizer 검증"""
        all_optimizers = GRADIENT_BASED_OPTIMIZERS + GRADIENT_FREE_OPTIMIZERS
        if self.optimizer not in all_optimizers:
            import warnings
            warnings.warn(
                f"Unknown optimizer: '{self.optimizer}'. "
                f"Supported optimizers: {all_optimizers}. "
                f"Assuming gradient-based optimizer."
            )
    
    def is_gradient_based(self) -> bool:
        """Gradient-based optimizer 여부 확인"""
        return self.optimizer not in GRADIENT_FREE_OPTIMIZERS
```

---

### Step 2: `simultaneous_estimator_fixed.py` 수정

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**수정 1**: Line 432
```python
# Before
use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B', 'BHHH']

# After
use_gradient = self.config.estimation.is_gradient_based()
```

**수정 2**: Line 706
```python
# Before
use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B', 'BHHH']

# After
use_gradient = self.config.estimation.is_gradient_based()
```

---

## ✅ 검증 방법

### 1. Trust Region 테스트
```python
config = create_sugar_substitute_multi_lv_config(
    optimizer='trust-constr',
    use_analytic_gradient=True,
    se_method='robust'
)

# 예상 결과:
# - use_gradient = True
# - Analytic gradient 초기화
# - Trust Region optimizer 사용
# - Sandwich Estimator 계산
```

### 2. Nelder-Mead 테스트
```python
config = create_sugar_substitute_multi_lv_config(
    optimizer='Nelder-Mead'
)

# 예상 결과:
# - use_gradient = False
# - Gradient calculator 초기화 안 함
# - Nelder-Mead optimizer 사용
```

### 3. 잘못된 Optimizer 테스트
```python
config = create_sugar_substitute_multi_lv_config(
    optimizer='INVALID_OPTIMIZER'
)

# 예상 결과:
# - Warning 메시지 출력
# - Gradient-based로 간주 (fallback)
```

---

## 📊 지원 Optimizer 목록

| Optimizer | 타입 | Gradient | Hessian | Bounds | Constraints |
|-----------|------|----------|---------|--------|-------------|
| **BFGS** | Quasi-Newton | ✅ | Approx | ❌ | ❌ |
| **L-BFGS-B** | Quasi-Newton | ✅ | Approx | ✅ | ❌ |
| **trust-constr** | Trust Region | ✅ | Approx | ✅ | ✅ |
| **trust-ncg** | Trust Region | ✅ | ✅ | ❌ | ❌ |
| **Newton-CG** | Newton | ✅ | ✅ | ❌ | ❌ |
| **SLSQP** | SQP | ✅ | Approx | ✅ | ✅ |
| **BHHH** | Custom | ✅ | OPG | ❌ | ❌ |
| **Nelder-Mead** | Simplex | ❌ | ❌ | ✅ | ❌ |
| **Powell** | Direction Set | ❌ | ❌ | ✅ | ❌ |
| **COBYLA** | Linear Approx | ❌ | ❌ | ✅ | ✅ |

---

**분석 완료 일시**: 2025-11-23

