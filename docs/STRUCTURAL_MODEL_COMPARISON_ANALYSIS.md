# 📊 구조모델 비교 분석: 기존 시스템 vs King (2022) ICLV

**작성일**: 2025-11-04  
**목적**: 구조모델 구현 전 기존 시스템 활용 가능성 검토  
**결론**: ⚠️ 부분 활용 가능, 새로운 구현 필요

---

## ✅ 핵심 결론

### **기존 시스템 활용 가능성: 30%**

| 항목 | 기존 시스템 | King (2022) ICLV | 호환성 |
|------|-------------|------------------|--------|
| **모델 유형** | semopy 경로분석 | 회귀 + 확률분포 | ⚠️ 부분 |
| **추정 방법** | 최대우도 (ML) | 시뮬레이션 (MSL) | ❌ 불가 |
| **오차항** | 고정 분산 | 시뮬레이션 draws | ❌ 불가 |
| **통합 방식** | Sequential | Simultaneous | ❌ 불가 |
| **회귀 계수** | ✅ 활용 가능 | ✅ 동일 개념 | ✅ 가능 |

**권장 사항**: 
- ✅ 기존 semopy를 **Sequential 방식 초기값**으로 활용
- ❌ 기존 semopy를 **Simultaneous 추정**에 직접 사용 불가
- ✅ **새로운 LatentVariableRegression 클래스 구현 필요**

---

## 📋 1. 기존 시스템 구조모델 분석

### **1.1 semopy 경로분석 (Path Analysis)**

**파일**: `src/analysis/path_analysis/`

**핵심 기능**:
```python
from semopy import Model

# 모델 스펙 정의
model_spec = """
# 측정모델
health_concern =~ q6 + q7 + q8 + q9 + q10 + q11
perceived_benefit =~ q12 + q13 + q14 + q15 + q16 + q17

# 구조모델 (경로분석)
perceived_benefit ~ age + gender + income + health_concern
purchase_intention ~ perceived_benefit + perceived_price
"""

# 모델 적합
model = Model(model_spec)
model.fit(data)

# 결과 추출
params = model.inspect()
```

**특징**:
1. **측정모델 + 구조모델 통합**
   - CFA (Confirmatory Factor Analysis)
   - Path Analysis (경로분석)
   - 동시에 추정

2. **최대우도 추정 (ML)**
   - 정규분포 가정
   - 고정 오차 분산
   - 결정론적 추정

3. **연속형 데이터 가정**
   - 리커트 척도를 연속형으로 처리
   - 정규분포 가정

---

### **1.2 기존 시스템의 구조방정식**

**모델**:
```
LV = γ₀ + γ₁*X₁ + γ₂*X₂ + ... + ε

여기서:
- LV: 잠재변수 (예: perceived_benefit)
- X: 사회인구학적 변수 (age, gender, income)
- γ: 회귀계수
- ε: 오차항 (정규분포, 고정 분산)
```

**추정 방법**:
- semopy의 최대우도 추정
- 모든 파라미터 동시 추정
- 오차 분산 고정 (σ² = 1 - R²)

**장점**:
- ✅ 간단하고 빠름
- ✅ 표준오차 자동 계산
- ✅ 적합도 지수 제공 (CFI, TLI, RMSEA)

**단점**:
- ❌ 리커트 척도를 연속형으로 처리
- ❌ 선택모델과 통합 불가
- ❌ 시뮬레이션 기반 추정 불가

---

## 📊 2. King (2022) ICLV 구조모델 분석

### **2.1 Apollo R 코드 기반 구조모델**

**모델**:
```r
# Apollo R 코드 (King 2022)
apollo_randCoeff = function(apollo_beta, apollo_inputs) {
    randcoeff = list()
    
    # 구조방정식: 사회인구학적 변수 → 잠재변수
    randcoeff[["LV"]] = gamma_age * age + 
                        gamma_gender * gender + 
                        gamma_income * income + 
                        eta
    
    return(randcoeff)
}

# eta는 표준정규분포에서 시뮬레이션
# eta ~ N(0, 1)
```

**특징**:
1. **시뮬레이션 기반 추정**
   - Halton draws 사용
   - 각 개인마다 R개의 draws
   - 몬테카를로 적분

2. **확률분포 명시**
   ```
   LV = γ*X + η
   η ~ N(0, σ²)
   
   P(LV|X) = (1/√(2πσ²)) * exp(-(LV - γ*X)²/(2σ²))
   ```

3. **동시 추정 (Simultaneous)**
   - 측정모델 + 구조모델 + 선택모델
   - 결합 우도함수 최대화
   ```
   L = ∏ᵢ ∫ P(Choice|LV) × P(Indicators|LV) × P(LV|X) dLV
   
   시뮬레이션 근사:
   L ≈ ∏ᵢ (1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)
   ```

---

### **2.2 필요한 메서드**

**LatentVariableRegression 클래스**:

```python
class LatentVariableRegression:
    """
    ICLV 구조모델: X → LV
    
    King (2022) Apollo R 코드 기반
    """
    
    def __init__(self, config: StructuralConfig):
        self.config = config
        self.sociodemographics = config.sociodemographics
        self.error_variance = config.error_variance
    
    def predict(self, data: pd.DataFrame, params: Dict, 
                draw: float) -> np.ndarray:
        """
        잠재변수 예측 (시뮬레이션 기반)
        
        LV = γ*X + σ*draw
        
        Args:
            data: 사회인구학적 변수 데이터
            params: {'gamma': np.ndarray}  # 회귀계수
            draw: 표준정규분포 draw (Halton sequence)
        
        Returns:
            잠재변수 값 (n_obs,)
        """
        gamma = params['gamma']
        X = data[self.sociodemographics].values
        
        # 선형 예측
        lv_mean = X @ gamma
        
        # 오차항 추가 (시뮬레이션)
        lv = lv_mean + np.sqrt(self.error_variance) * draw
        
        return lv
    
    def log_likelihood(self, data: pd.DataFrame, lv: np.ndarray,
                      params: Dict, draw: float) -> float:
        """
        구조모델 로그우도
        
        P(LV|X) ~ N(γ*X, σ²)
        
        Args:
            data: 사회인구학적 변수 데이터
            lv: 잠재변수 값
            params: {'gamma': np.ndarray}
            draw: 표준정규분포 draw
        
        Returns:
            로그우도 값
        """
        gamma = params['gamma']
        X = data[self.sociodemographics].values
        
        # 평균
        lv_mean = X @ gamma
        
        # 로그우도 (정규분포)
        ll = -0.5 * np.log(2 * np.pi * self.error_variance)
        ll -= 0.5 * ((lv - lv_mean) ** 2) / self.error_variance
        
        return np.sum(ll)
    
    def fit(self, data: pd.DataFrame, latent_var: np.ndarray) -> Dict:
        """
        구조모델 단독 추정 (Sequential 방식용)
        
        OLS 회귀분석
        
        Args:
            data: 사회인구학적 변수 데이터
            latent_var: 잠재변수 값 (측정모델에서 추정)
        
        Returns:
            {'gamma': np.ndarray, 'sigma': float}
        """
        X = data[self.sociodemographics].values
        y = latent_var
        
        # OLS 추정
        gamma = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # 잔차 분산
        residuals = y - X @ gamma
        sigma = np.std(residuals)
        
        return {
            'gamma': gamma,
            'sigma': sigma,
            'r_squared': 1 - (np.var(residuals) / np.var(y))
        }
```

---

## 🔄 3. 기존 시스템 vs King (2022) 비교

### **3.1 모델 방정식 비교**

| 항목 | 기존 semopy | King (2022) ICLV |
|------|-------------|------------------|
| **방정식** | `LV = γ*X + ε` | `LV = γ*X + η` |
| **오차항** | `ε ~ N(0, σ²)` 고정 | `η ~ N(0, σ²)` 시뮬레이션 |
| **추정** | 최대우도 (ML) | 시뮬레이션 (MSL) |
| **회귀계수** | γ (동일) | γ (동일) |

**핵심 차이**: 오차항 처리 방식

---

### **3.2 추정 방법 비교**

#### **기존 semopy (Sequential)**

```python
# 1단계: 측정모델 (CFA)
cfa_model = Model("LV =~ q1 + q2 + q3")
cfa_model.fit(data)
factor_scores = cfa_model.predict_factors(data)

# 2단계: 구조모델 (회귀분석)
path_model = Model("LV ~ age + gender + income")
path_model.fit(data)
gamma = path_model.inspect()

# 3단계: 선택모델
choice_model.fit(data, latent_var=factor_scores)
```

**특징**:
- 단계별 추정
- 각 단계 독립적
- 빠르고 간단

---

#### **King (2022) ICLV (Simultaneous)**

```python
# 동시 추정
def joint_log_likelihood(params):
    ll = 0
    
    for individual in data:
        # Halton draws
        draws = halton_sequence(n_draws)
        
        sim_ll = 0
        for draw in draws:
            # 구조모델: LV 시뮬레이션
            lv = structural_model.predict(individual, params, draw)
            
            # 측정모델 우도
            ll_measurement = measurement_model.log_likelihood(
                individual, lv, params
            )
            
            # 선택모델 우도
            ll_choice = choice_model.log_likelihood(
                individual, lv, params
            )
            
            # 구조모델 우도
            ll_structural = structural_model.log_likelihood(
                individual, lv, params, draw
            )
            
            # 결합 우도
            sim_ll += exp(ll_measurement + ll_choice + ll_structural)
        
        ll += log(sim_ll / n_draws)
    
    return ll

# 최적화
result = minimize(lambda p: -joint_log_likelihood(p), initial_params)
```

**특징**:
- 모든 모델 동시 추정
- 시뮬레이션 기반
- 복잡하지만 정확

---

### **3.3 장단점 비교**

| 항목 | 기존 semopy | King (2022) ICLV |
|------|-------------|------------------|
| **장점** | • 간단하고 빠름<br>• 표준오차 자동<br>• 적합도 지수 제공 | • 정확한 추정<br>• 선택모델 통합<br>• 이론적 엄밀성 |
| **단점** | • 리커트 척도 부적합<br>• 선택모델 통합 불가<br>• 단계별 오차 누적 | • 복잡함<br>• 계산 비용 높음<br>• 수렴 어려움 |
| **적용** | Sequential 추정 | Simultaneous 추정 |

---

## 🎯 4. 기존 시스템 활용 방안

### **방안 1: Sequential 추정 초기값으로 활용 ✅ 권장**

**개념**:
```python
# 1단계: semopy로 초기 추정
from semopy import Model

model_spec = """
perceived_benefit =~ q12 + q13 + q14 + q15 + q16 + q17
perceived_benefit ~ age + gender + income
"""

model = Model(model_spec)
model.fit(data)

# 회귀계수 추출
params = model.inspect()
gamma_initial = params[params['op'] == '~']['Estimate'].values

# 2단계: ICLV 동시 추정의 초기값으로 사용
iclv_initial_params = {
    'gamma': gamma_initial,  # semopy에서 추출
    'zeta': np.ones(n_indicators),
    'tau': np.array([-2, -1, 1, 2]),
    'beta': np.zeros(n_attributes),
    'lambda': 1.0
}

# 3단계: ICLV 동시 추정
estimator = SimultaneousEstimator(config)
results = estimator.estimate(
    data, 
    measurement_model, 
    structural_model, 
    choice_model,
    initial_params=iclv_initial_params
)
```

**장점**:
- ✅ 좋은 초기값 제공
- ✅ 수렴 속도 향상
- ✅ 기존 코드 재사용

**단점**:
- ⚠️ 여전히 새로운 구현 필요

---

### **방안 2: Sequential 추정 전용 모드 ✅ 가능**

**개념**:
```python
class LatentVariableRegression:
    def fit_sequential(self, data, latent_var):
        """
        Sequential 방식: semopy 활용
        """
        # semopy 모델 생성
        sociodem_vars = " + ".join(self.sociodemographics)
        model_spec = f"LV ~ {sociodem_vars}"
        
        # 데이터 준비
        data_with_lv = data.copy()
        data_with_lv['LV'] = latent_var
        
        # semopy 적합
        model = Model(model_spec)
        model.fit(data_with_lv)
        
        # 결과 추출
        params = model.inspect()
        gamma = params[params['op'] == '~']['Estimate'].values
        
        return {'gamma': gamma}
    
    def fit_simultaneous(self, ...):
        """
        Simultaneous 방식: 새로운 구현
        """
        # 시뮬레이션 기반 추정
        pass
```

**장점**:
- ✅ Sequential 방식에서 semopy 활용
- ✅ 기존 코드 최대 활용

**단점**:
- ⚠️ Simultaneous 방식은 여전히 새로운 구현 필요

---

### **방안 3: 완전히 새로운 구현 ✅ 최종 권장**

**이유**:
1. **ICLV 동시 추정 필수**
   - semopy는 선택모델과 통합 불가
   - 시뮬레이션 기반 추정 불가

2. **코드 일관성**
   - OrderedProbitMeasurement와 동일한 스타일
   - SimultaneousEstimator와 완벽한 통합

3. **확장성**
   - 다양한 분포 지원 가능
   - 고급 기능 추가 용이

**구현 계획**:
```python
class LatentVariableRegression:
    """
    완전히 새로운 구현
    
    King (2022) Apollo R 코드 기반
    """
    
    def __init__(self, config):
        # 설정 초기화
        pass
    
    def predict(self, data, params, draw):
        # LV = γ*X + σ*draw
        pass
    
    def log_likelihood(self, data, lv, params, draw):
        # P(LV|X) ~ N(γ*X, σ²)
        pass
    
    def fit(self, data, latent_var):
        # Sequential 방식: OLS
        # (semopy 사용 가능)
        pass
```

---

## 📊 5. 최종 권장 사항

### **✅ 권장: 방안 3 (새로운 구현) + 방안 1 (초기값 활용)**

**구현 전략**:

1. **LatentVariableRegression 클래스 새로 구현**
   - King (2022) Apollo R 코드 기반
   - OrderedProbitMeasurement와 동일한 스타일
   - 시뮬레이션 기반 추정 지원

2. **semopy를 초기값 생성에 활용**
   - Sequential 추정으로 좋은 초기값 생성
   - Simultaneous 추정의 수렴 속도 향상

3. **Sequential 모드 지원**
   - `fit()` 메서드에서 semopy 활용 가능
   - 빠른 프로토타이핑 지원

---

### **구현 우선순위**

**P0 (최우선)**:
```python
class LatentVariableRegression:
    def predict(self, data, params, draw):
        # ICLV 동시 추정 필수
        pass
    
    def log_likelihood(self, data, lv, params, draw):
        # ICLV 동시 추정 필수
        pass
```

**P1 (높음)**:
```python
    def fit(self, data, latent_var):
        # Sequential 방식 지원
        # semopy 활용 가능
        pass
```

**P2 (중간)**:
```python
    def get_initial_params_from_semopy(self, data, latent_var):
        # semopy로 초기값 생성
        pass
```

---

## 📝 6. 구현 예시

### **6.1 기본 구현**

```python
class LatentVariableRegression:
    """
    ICLV 구조모델
    
    Model:
        LV = γ*X + η
        η ~ N(0, σ²)
    """
    
    def __init__(self, config: StructuralConfig):
        self.config = config
        self.sociodemographics = config.sociodemographics
        self.error_variance = config.error_variance
        self.logger = logging.getLogger(__name__)
    
    def predict(self, data: pd.DataFrame, params: Dict, 
                draw: float) -> np.ndarray:
        """
        잠재변수 예측 (시뮬레이션)
        
        LV = γ*X + σ*draw
        """
        gamma = params['gamma']
        X = data[self.sociodemographics].values
        
        # 선형 예측
        lv_mean = X @ gamma
        
        # 오차항 추가
        lv = lv_mean + np.sqrt(self.error_variance) * draw
        
        return lv
    
    def log_likelihood(self, data: pd.DataFrame, lv: np.ndarray,
                      params: Dict, draw: float) -> float:
        """
        구조모델 로그우도
        
        P(LV|X) ~ N(γ*X, σ²)
        """
        gamma = params['gamma']
        X = data[self.sociodemographics].values
        
        lv_mean = X @ gamma
        
        # 정규분포 로그우도
        ll = -0.5 * np.log(2 * np.pi * self.error_variance)
        ll -= 0.5 * ((lv - lv_mean) ** 2) / self.error_variance
        
        return np.sum(ll)
    
    def fit(self, data: pd.DataFrame, latent_var: np.ndarray) -> Dict:
        """
        Sequential 방식 추정 (OLS)
        """
        X = data[self.sociodemographics].values
        y = latent_var
        
        # OLS
        gamma = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # 잔차 분산
        residuals = y - X @ gamma
        sigma = np.std(residuals)
        
        return {
            'gamma': gamma,
            'sigma': sigma,
            'r_squared': 1 - (np.var(residuals) / np.var(y))
        }
```

---

### **6.2 semopy 초기값 활용**

```python
    def get_initial_params_from_semopy(self, data: pd.DataFrame,
                                      latent_var: np.ndarray) -> Dict:
        """
        semopy로 초기값 생성
        """
        from semopy import Model
        
        # 모델 스펙
        sociodem_vars = " + ".join(self.sociodemographics)
        model_spec = f"LV ~ {sociodem_vars}"
        
        # 데이터 준비
        data_with_lv = data.copy()
        data_with_lv['LV'] = latent_var
        
        # semopy 적합
        model = Model(model_spec)
        model.fit(data_with_lv)
        
        # 파라미터 추출
        params = model.inspect()
        gamma = params[params['op'] == '~']['Estimate'].values
        
        return {'gamma': gamma}
```

---

## ✅ 최종 결론

### **기존 시스템 활용 가능성: 30%**

**활용 가능**:
- ✅ Sequential 추정 초기값 생성
- ✅ Sequential 모드 `fit()` 메서드
- ✅ 회귀계수 개념 동일

**활용 불가**:
- ❌ ICLV 동시 추정
- ❌ 시뮬레이션 기반 추정
- ❌ 선택모델 통합

**최종 권장**:
1. **새로운 LatentVariableRegression 클래스 구현** (필수)
2. **semopy를 초기값 생성에 활용** (선택)
3. **Sequential 모드 지원** (선택)

**예상 작업량**: 1-2일

---

**보고서 작성일**: 2025-11-04  
**작성자**: Sugar Substitute Research Team  
**상태**: ✅ 분석 완료, 구현 대기

