# 🎯 Simultaneous 추정용 선택모델 구현 완료

**작성일**: 2025-11-05  
**목적**: King (2022) Apollo R 코드 기반 Binary Probit 선택모델 구현  
**상태**: ✅ 구현 완료, 테스트 통과

---

## ✅ 구현 요약

### **핵심 성과**

King (2022) Apollo R 코드를 최대한 반영하여 **ICLV 동시 추정용 Binary Probit 선택모델**을 구현했습니다.

**구현 파일**:
- `src/analysis/hybrid_choice_model/iclv_models/choice_equations.py` (신규 생성)

**주요 클래스**:
- `BinaryProbitChoice`: Binary Probit 선택모델
- `ChoiceConfig`: 선택모델 설정

**기능**:
1. ✅ 로그우도 계산 (`log_likelihood`)
2. ✅ 선택 확률 예측 (`predict_probabilities`)
3. ✅ 선택 예측 (`predict`)
4. ✅ WTP 계산 (`calculate_wtp`)
5. ✅ 초기 파라미터 생성 (`get_initial_params`)

---

## 📊 모델 사양

### **1. 수학적 모델**

```
효용함수:
V = intercept + β*X + λ*LV

선택 확률:
P(Yes) = Φ(V)
P(No) = 1 - Φ(V)

여기서:
- V: 효용 (Utility)
- X: 선택 속성 (e.g., price, quality)
- β: 속성 계수 (Attribute coefficients)
- λ: 잠재변수 계수 (Latent variable coefficient)
- LV: 잠재변수 (Latent Variable)
- Φ: 표준정규 누적분포함수
```

### **2. Apollo R 코드 대응**

**King (2022) Apollo R 코드**:
```r
op_settings = list(
  outcomeOrdered = Q6ResearchResponse,
  V = intercept + b_bid*Q6Bid + lambda*LV,
  tau = list(-100, 0),
  componentName = "choice",
  coding = c(-1, 0, 1)
)
P[['choice']] = apollo_op(op_settings, functionality)
```

**Python 구현**:
```python
from src.analysis.hybrid_choice_model.iclv_models import (
    BinaryProbitChoice,
    ChoiceConfig
)

# 설정
config = ChoiceConfig(
    choice_attributes=['bid', 'quality'],
    choice_type='binary',
    price_variable='bid'
)

# 모델 생성
model = BinaryProbitChoice(config)

# 파라미터
params = {
    'intercept': 0.5,
    'beta': np.array([-2.0, 0.3]),  # [β_bid, β_quality]
    'lambda': 1.5
}

# 로그우도 계산
ll = model.log_likelihood(data, lv, params)

# 확률 예측
probs = model.predict_probabilities(data, lv, params)
```

---

## 🔧 사용 방법

### **1. 기본 사용법**

```python
import numpy as np
import pandas as pd
from src.analysis.hybrid_choice_model.iclv_models import (
    BinaryProbitChoice,
    ChoiceConfig
)

# 1. 설정 생성
config = ChoiceConfig(
    choice_attributes=['price', 'quality'],
    choice_type='binary',
    price_variable='price'
)

# 2. 모델 생성
model = BinaryProbitChoice(config)

# 3. 데이터 준비
data = pd.DataFrame({
    'price': [0.5, 1.0, 1.5],
    'quality': [0.3, 0.5, 0.7],
    'choice': [1, 1, 0]  # 0 or 1
})

# 4. 잠재변수
lv = np.array([0.5, 0.0, -0.5])

# 5. 파라미터
params = {
    'intercept': 0.5,
    'beta': np.array([-2.0, 0.3]),
    'lambda': 1.5
}

# 6. 로그우도 계산
ll = model.log_likelihood(data, lv, params)
print(f"로그우도: {ll:.4f}")

# 7. 확률 예측
probs = model.predict_probabilities(data, lv, params)
print(f"선택 확률: {probs}")

# 8. 선택 예측
predictions = model.predict(data, lv, params)
print(f"예측 선택: {predictions}")
```

### **2. WTP 계산**

```python
# WTP 계산
wtp_quality = model.calculate_wtp(params, 'quality')
print(f"WTP for Quality: {wtp_quality:.4f}")

# 이론적 WTP = -β_quality / β_price
# = -0.3 / (-2.0) = 0.15
```

### **3. Simultaneous 추정과 통합**

```python
from src.analysis.hybrid_choice_model.iclv_models import (
    OrderedProbitMeasurement,
    LatentVariableRegression,
    BinaryProbitChoice,
    SimultaneousEstimator,
    create_iclv_config
)

# 1. 설정 생성
config = create_iclv_config(
    latent_variable='health_concern',
    indicators=['hc_1', 'hc_2', 'hc_3'],
    sociodemographics=['age', 'gender', 'income'],
    choice_attributes=['price', 'quality'],
    price_variable='price',
    n_draws=1000
)

# 2. 모델 생성
measurement_model = OrderedProbitMeasurement(config.measurement)
structural_model = LatentVariableRegression(config.structural)
choice_model = BinaryProbitChoice(config.choice)

# 3. 동시 추정
estimator = SimultaneousEstimator(config)
results = estimator.estimate(
    data,
    measurement_model,
    structural_model,
    choice_model
)

# 4. 결과 확인
print(f"로그우도: {results['log_likelihood']:.2f}")
print(f"선택모델 파라미터:")
print(f"  절편: {results['parameters']['choice']['intercept']:.4f}")
print(f"  β: {results['parameters']['choice']['beta']}")
print(f"  λ: {results['parameters']['choice']['lambda']:.4f}")
```

---

## ✅ 검증 결과

### **1. Apollo R 코드 검증**

**테스트 케이스**:
```python
# 파라미터
intercept = 0.5
β_bid = -2.0
λ = 1.5

# 테스트 케이스
케이스 1: Bid=0.0, LV=0.0 → V=0.5 → P(Yes)=0.6915 ✅
케이스 2: Bid=1.0, LV=0.0 → V=-1.5 → P(Yes)=0.0668 ✅
케이스 3: Bid=0.0, LV=1.0 → V=2.0 → P(Yes)=0.9772 ✅
케이스 4: Bid=1.0, LV=1.0 → V=0.0 → P(Yes)=0.5000 ✅
```

**결과**: ✅ Apollo R 코드와 수치적으로 완벽히 일치

### **2. WTP 계산 검증**

```python
β_price = -2.0
β_quality = 0.6

WTP = -β_quality / β_price = 0.3000 ✅
```

**결과**: ✅ 이론적 WTP와 일치

### **3. 시각화 검증**

**생성된 파일**:
1. `tests/binary_probit_price_sensitivity.png`
   - 가격에 따른 선택 확률 변화
   - 잠재변수 수준별 비교

2. `tests/binary_probit_lv_effect.png`
   - 잠재변수에 따른 선택 확률 변화
   - λ 값별 비교

**결과**: ✅ 예상대로 작동

---

## 🔄 기존 시스템과의 통합

### **1. ICLV 모듈 구조**

```
src/analysis/hybrid_choice_model/iclv_models/
├── __init__.py                    # 모듈 export
├── iclv_config.py                 # 설정 클래스
├── measurement_equations.py       # 측정모델 (OrderedProbitMeasurement)
├── structural_equations.py        # 구조모델 (LatentVariableRegression)
├── choice_equations.py            # 선택모델 (BinaryProbitChoice) ✨ 신규
├── simultaneous_estimator.py      # 동시 추정기
└── wtp_calculator.py              # WTP 계산기
```

### **2. 중복 방지**

**기존 선택모델과의 차이**:

| 항목 | 기존 선택모델 | ICLV 선택모델 |
|------|--------------|--------------|
| **위치** | `choice_models/` | `iclv_models/` |
| **목적** | Sequential 추정 | Simultaneous 추정 |
| **인터페이스** | `fit(data)` | `log_likelihood(data, lv, params)` |
| **잠재변수** | 외부에서 계산 | 동시 추정 중 생성 |
| **사용처** | 단독 선택모델 분석 | ICLV 통합 분석 |

**결론**: ✅ 중복 없음, 목적이 다름

---

## 📝 API 문서

### **BinaryProbitChoice 클래스**

#### **메서드**

**1. `__init__(config: ChoiceConfig)`**
- 모델 초기화
- Args: `config` - 선택모델 설정

**2. `log_likelihood(data, lv, params) -> float`**
- 로그우도 계산
- Args:
  - `data`: 선택 데이터 (DataFrame)
  - `lv`: 잠재변수 값 (ndarray or scalar)
  - `params`: 파라미터 딕셔너리
- Returns: 로그우도 값

**3. `predict_probabilities(data, lv, params) -> ndarray`**
- 선택 확률 예측
- Args: 위와 동일
- Returns: 선택 확률 배열

**4. `predict(data, lv, params, threshold=0.5) -> ndarray`**
- 선택 예측
- Args: 위와 동일 + `threshold`
- Returns: 예측 선택 (0 or 1)

**5. `calculate_wtp(params, attribute) -> float`**
- WTP 계산
- Args:
  - `params`: 파라미터 딕셔너리
  - `attribute`: WTP를 계산할 속성
- Returns: WTP 값

**6. `get_initial_params(data) -> Dict`**
- 초기 파라미터 생성
- Args: `data` - 선택 데이터
- Returns: 초기 파라미터 딕셔너리

---

## 🎯 다음 단계

### **완료된 컴포넌트**

1. ✅ 측정모델 (OrderedProbitMeasurement)
2. ✅ 구조모델 (LatentVariableRegression)
3. ✅ 선택모델 (BinaryProbitChoice) ← **신규 완료**
4. ✅ 동시 추정기 (SimultaneousEstimator)

### **남은 작업**

1. ⏳ 실제 데이터로 전체 ICLV 동시 추정 테스트
2. ⏳ WTP 계산기 완성 (Conditional/Unconditional)
3. ⏳ 결과 분석 및 시각화
4. ⏳ 문서화 완성

---

## 📚 참고 자료

### **King (2022) 논문**
- King, P. M. (2022). Willingness-to-pay for precautionary control of microplastics.
- Journal of Environmental Economics and Policy.
- https://doi.org/10.1080/21606544.2022.2146757

### **Apollo R 패키지**
- http://www.apollochoicemodelling.com/
- Ordered Probit 함수: `apollo_op()`

### **관련 문서**
- `docs/ICLV_R_TO_PYTHON_VALIDATION.md`
- `docs/STRUCTURAL_MODEL_COMPARISON_ANALYSIS.md`
- `docs/ORDERED_PROBIT_IMPLEMENTATION.md`

---

**보고서 작성일**: 2025-11-05  
**작성자**: Sugar Substitute Research Team  
**상태**: ✅ 구현 완료, 테스트 통과

