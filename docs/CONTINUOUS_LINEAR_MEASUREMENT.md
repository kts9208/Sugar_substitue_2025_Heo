# Continuous Linear Measurement (연속형 선형 측정모델)

## 📋 개요

**Continuous Linear Measurement**는 리커트 척도를 **연속형 변수**로 간주하여 잠재변수를 측정하는 **구조방정식 모형(SEM)** 방식의 측정모델입니다.

기존의 **Ordered Probit** 방법과 **독립적으로 선택 가능**하며, 두 방법을 **혼합하여 사용**할 수도 있습니다.

---

## 🎯 주요 특징

### 1. **수학적 모델**

#### **Continuous Linear (연속형 선형)**
```
Y_i = ζ_i * LV + ε_i
ε_i ~ N(0, σ²_i)
```

- **파라미터**:
  - `ζ` (zeta): 요인적재량 (factor loadings)
  - `σ²` (sigma_sq): 오차분산 (error variances)

- **로그우도**:
```
LL = Σ_i [ -0.5 * log(2π * σ²_i) - 0.5 * (Y_i - ζ_i * LV)² / σ²_i ]
```

#### **Ordered Probit (순서형 프로빗)**
```
P(Y_i = k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
```

- **파라미터**:
  - `ζ` (zeta): 요인적재량
  - `τ` (tau): 임계값 (thresholds)

---

### 2. **파라미터 수 비교**

| 측정 방법 | 파라미터 (3개 지표) | 파라미터 (38개 지표) |
|----------|-------------------|---------------------|
| **Continuous Linear** | 5개 | 71개 |
| - zeta | 2개 (첫 번째 고정) | 33개 (각 LV 첫 번째 고정) |
| - sigma_sq | 3개 | 38개 |
| **Ordered Probit** | 15개 | 190개 |
| - zeta | 3개 | 38개 |
| - tau | 12개 (3 × 4) | 152개 (38 × 4) |
| **감소량** | **10개 (66.7%)** | **119개 (62.6%)** |

---

### 3. **장단점 비교**

| 항목 | Continuous Linear | Ordered Probit |
|------|------------------|----------------|
| **파라미터 수** | ✅ 적음 (62% 감소) | ❌ 많음 |
| **추정 속도** | ✅ 빠름 | ❌ 느림 |
| **메모리 사용** | ✅ 적음 | ❌ 많음 |
| **이론적 정확성** | ⚠️ 리커트를 연속형으로 가정 | ✅ 리커트를 순서형으로 처리 |
| **실무 적용** | ✅ SEM에서 널리 사용 | ⚠️ 계산 복잡도 높음 |

---

## 🔧 사용 방법

### 1. **기본 사용 (Continuous Linear - 디폴트)**

```python
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import MeasurementConfig

# Continuous Linear 측정모델 (디폴트)
config = MeasurementConfig(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    n_categories=5,
    measurement_method='continuous_linear'  # 디폴트 (생략 가능)
)
```

### 2. **Ordered Probit 사용**

```python
# Ordered Probit 측정모델
config = MeasurementConfig(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    n_categories=5,
    measurement_method='ordered_probit'  # 명시적 지정
)
```

### 3. **혼합 사용 (다중 잠재변수)**

```python
# 잠재변수별로 다른 측정 방법 사용 가능
measurement_configs = {
    'health_concern': MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        measurement_method='continuous_linear'  # 연속형 선형
    ),
    'perceived_benefit': MeasurementConfig(
        latent_variable='perceived_benefit',
        indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
        measurement_method='ordered_probit'  # 순서형 프로빗
    ),
    'perceived_price': MeasurementConfig(
        latent_variable='perceived_price',
        indicators=['q27', 'q28', 'q29'],
        measurement_method='continuous_linear'  # 연속형 선형
    ),
    # ... 나머지 잠재변수
}
```

---

## 🏗️ 독립성 및 호환성

### 1. **독립성 (Independence)**

두 측정 방법은 **완전히 독립적**으로 구현되어 있습니다:

#### **클래스 구조**
```
measurement_equations.py
├── OrderedProbitMeasurement      # 순서형 프로빗
└── ContinuousLinearMeasurement   # 연속형 선형 (새로 추가)

gpu_measurement_equations.py
├── GPUOrderedProbitMeasurement      # GPU 순서형 프로빗
└── GPUContinuousLinearMeasurement   # GPU 연속형 선형 (새로 추가)
```

#### **파라미터 구조**
```python
# Continuous Linear
params = {
    'zeta': np.array([...]),      # 요인적재량
    'sigma_sq': np.array([...])   # 오차분산
}

# Ordered Probit
params = {
    'zeta': np.array([...]),      # 요인적재량
    'tau': np.array([[...], ...]) # 임계값
}
```

---

### 2. **호환성 (Compatibility)**

두 측정 방법은 **동일한 인터페이스**를 공유하여 완벽하게 호환됩니다:

#### **공통 인터페이스**
```python
class MeasurementModel:
    def __init__(self, config: MeasurementConfig)
    def initialize_parameters(self) -> Dict[str, np.ndarray]
    def log_likelihood(self, data, latent_var, params) -> float
    def get_n_parameters(self) -> int
    def get_parameter_bounds(self) -> List[Tuple[float, float]]
```

#### **자동 선택 메커니즘**
```python
# multi_latent_measurement.py
for lv_name, config in measurement_configs.items():
    method = config.measurement_method
    
    if method == 'continuous_linear':
        model = ContinuousLinearMeasurement(config)
    elif method == 'ordered_probit':
        model = OrderedProbitMeasurement(config)
```

#### **파라미터 처리 (gpu_batch_estimator.py)**
```python
# 초기화
if method == 'continuous_linear':
    # zeta + sigma_sq
elif method == 'ordered_probit':
    # zeta + tau

# Bounds
if method == 'continuous_linear':
    # zeta: [-10, 10], sigma_sq: [0.01, 100]
elif method == 'ordered_probit':
    # zeta: [0.1, 10], tau: [-10, 10]

# 언팩
if method == 'continuous_linear':
    params = {'zeta': ..., 'sigma_sq': ...}
elif method == 'ordered_probit':
    params = {'zeta': ..., 'tau': ...}
```

---

### 3. **혼합 사용 예시**

```python
# 5개 잠재변수에 대해 서로 다른 측정 방법 사용
measurement_configs = {
    'health_concern': MeasurementConfig(
        measurement_method='continuous_linear'  # 연속형
    ),
    'perceived_benefit': MeasurementConfig(
        measurement_method='ordered_probit'     # 순서형
    ),
    'perceived_price': MeasurementConfig(
        measurement_method='continuous_linear'  # 연속형
    ),
    'nutrition_knowledge': MeasurementConfig(
        measurement_method='continuous_linear'  # 연속형
    ),
    'purchase_intention': MeasurementConfig(
        measurement_method='ordered_probit'     # 순서형
    )
}

# MultiLatentMeasurement가 자동으로 적절한 모델 선택
model = MultiLatentMeasurement(measurement_configs)

# 파라미터 수 자동 계산
# - health_concern: 5개 (continuous_linear)
# - perceived_benefit: 30개 (ordered_probit)
# - perceived_price: 5개 (continuous_linear)
# - nutrition_knowledge: 39개 (continuous_linear)
# - purchase_intention: 15개 (ordered_probit)
# 총: 94개
```

---

## 📊 성능 비교

### 1. **파라미터 수 (5개 잠재변수, 38개 지표)**

| 측정 방법 | 파라미터 수 | 감소량 |
|----------|-----------|--------|
| **All Continuous Linear** | 71개 | 62.6% ↓ |
| **All Ordered Probit** | 190개 | - |
| **혼합 (3 CL + 2 OP)** | ~130개 | 31.6% ↓ |

### 2. **추정 시간 (예상)**

| 측정 방법 | 추정 시간 | 개선 |
|----------|----------|------|
| **All Continuous Linear** | ~2-3분 | 50-70% ↓ |
| **All Ordered Probit** | ~5-10분 | - |
| **혼합** | ~3-6분 | 30-40% ↓ |

### 3. **메모리 사용량 (예상)**

| 측정 방법 | 메모리 | 개선 |
|----------|--------|------|
| **All Continuous Linear** | ~2-3GB | 40-50% ↓ |
| **All Ordered Probit** | ~4-5GB | - |
| **혼합** | ~3-4GB | 20-30% ↓ |

---

## 🧪 테스트

```bash
# 기본 기능 테스트
python scripts/test_continuous_linear_measurement.py

# 전체 ICLV 모델 테스트
python scripts/test_gpu_batch_iclv.py
```

---

## 📝 요약

1. ✅ **Continuous Linear**와 **Ordered Probit**은 **완전히 독립적**으로 구현
2. ✅ **동일한 인터페이스**를 공유하여 **완벽하게 호환**
3. ✅ **잠재변수별로 다른 측정 방법** 선택 가능
4. ✅ **파라미터 수 62.6% 감소** (All Continuous Linear)
5. ✅ **추정 시간 50-70% 단축** (예상)
6. ✅ **GPU 가속 지원** (CuPy)

---

## 🔗 관련 파일

- `src/analysis/hybrid_choice_model/iclv_models/iclv_config.py`: 설정
- `src/analysis/hybrid_choice_model/iclv_models/measurement_equations.py`: CPU 구현
- `src/analysis/hybrid_choice_model/iclv_models/gpu_measurement_equations.py`: GPU 구현
- `src/analysis/hybrid_choice_model/iclv_models/multi_latent_measurement.py`: 다중 잠재변수 관리
- `src/analysis/hybrid_choice_model/iclv_models/gpu_batch_estimator.py`: 파라미터 처리
- `scripts/test_continuous_linear_measurement.py`: 테스트

