# 📊 잠재변수 모델 구현 현황 보고서

**작성일**: 2025-11-04  
**검토 범위**: 측정모델 + 구조모델 (잠재변수 모델 전체)  
**상태**: ⚠️ 부분 완성

---

## ✅ 핵심 결론

### **측정모델 (Measurement Model): 100% 완성 ✅**
- **파일**: `src/analysis/hybrid_choice_model/iclv_models/measurement_equations.py`
- **클래스**: `OrderedProbitMeasurement`
- **상태**: 완전히 구현되고 테스트 완료

### **구조모델 (Structural Model): 0% 완성 ❌**
- **파일**: `src/analysis/hybrid_choice_model/iclv_models/structural_equations.py`
- **클래스**: `LatentVariableRegression`
- **상태**: **파일 자체가 존재하지 않음**

### **전체 잠재변수 모델: 50% 완성 ⚠️**
- 측정모델만 완성
- 구조모델 미구현
- **ICLV 동시 추정 불가능**

---

## 📋 상세 분석

### **1. 측정모델 (Measurement Model) ✅**

#### **1.1 구현 상태**

**파일**: `src/analysis/hybrid_choice_model/iclv_models/measurement_equations.py` (329 lines)

**클래스**: `OrderedProbitMeasurement`

**핵심 메서드**:
```python
class OrderedProbitMeasurement:
    def __init__(self, config: MeasurementConfig)
    
    def log_likelihood(self, data, latent_var, params) -> float
        # ✅ 완전 구현
        # King (2022) Apollo R 코드와 100% 동일
        # 검증 완료: 차이 0.0000000000
    
    def predict_probabilities(self, latent_var, params) -> Dict
        # ✅ 완전 구현
        # 각 범주의 확률 예측
    
    def predict(self, latent_var, params) -> pd.DataFrame
        # ✅ 완전 구현
        # 가장 높은 확률의 범주 예측
    
    def fit(self, data, initial_params) -> Dict
        # ✅ 완전 구현
        # Sequential 방식용 단독 추정
```

**모델 방정식**:
```
P(Y_i = k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)

여기서:
- Y_i: 관측지표 (1-5 for 5-point Likert scale)
- τ: 임계값 (thresholds) - 범주 경계
- ζ: 요인적재량 (factor loadings)
- LV: 잠재변수 (latent variable)
- Φ: 표준정규 누적분포함수
```

**Apollo R 코드 대응**:
```r
# Apollo R 코드
op_settings = list(
    outcomeOrdered = Q13,
    V = zeta_Q13 * LV,
    tau = c(tau_Q13_1, tau_Q13_2, tau_Q13_3, tau_Q13_4),
    componentName = "indic_Q13"
)
P[["indic_Q13"]] = apollo_op(op_settings, functionality)
```

**검증 결과**:
- ✅ Apollo R 코드와 수치적 동일성 확인 (차이: 0.0000000000)
- ✅ 실제 데이터 테스트 완료 (5개 요인, 300명)
- ✅ 역코딩 데이터 테스트 완료 (8.59% 개선)

---

#### **1.2 테스트 현황**

**테스트 파일**:
1. `tests/test_ordered_probit_measurement.py` - 단위 테스트 ✅
2. `tests/test_ordered_probit_integration.py` - Apollo 동일성 검증 ✅
3. `tests/test_ordered_probit_real_data.py` - 실제 데이터 테스트 ✅
4. `tests/test_ordered_probit_reversed_data.py` - 역코딩 데이터 테스트 ✅

**테스트 결과**:
- ✅ 모든 테스트 통과
- ✅ 로그우도 계산 정확성 검증
- ✅ 확률 예측 정확성 검증
- ✅ 역코딩 효과 검증 (8.59% 개선)

---

### **2. 구조모델 (Structural Model) ❌**

#### **2.1 구현 상태**

**파일**: `src/analysis/hybrid_choice_model/iclv_models/structural_equations.py`

**상태**: **파일이 존재하지 않음** ❌

**예상 클래스**: `LatentVariableRegression`

**필요한 메서드**:
```python
class LatentVariableRegression:
    """
    구조모델: 사회인구학적 변수 → 잠재변수
    
    Model:
        LV = γ*X + η
    
    여기서:
        - LV: 잠재변수
        - X: 사회인구학적 변수 (age, gender, income, etc.)
        - γ: 회귀계수
        - η: 오차항 (정규분포)
    """
    
    def __init__(self, config: StructuralConfig):
        # 설정 초기화
        pass
    
    def predict(self, data, params, draw) -> np.ndarray:
        """
        잠재변수 예측
        
        LV = γ*X + σ*draw
        
        Args:
            data: 사회인구학적 변수 데이터
            params: 회귀계수 (gamma)
            draw: 오차항 draw (Halton sequence)
        
        Returns:
            잠재변수 값
        """
        pass
    
    def log_likelihood(self, data, lv, params, draw) -> float:
        """
        구조모델 로그우도
        
        P(LV|X) ~ N(γ*X, σ²)
        
        Args:
            data: 사회인구학적 변수 데이터
            lv: 잠재변수 값
            params: 회귀계수
            draw: 오차항 draw
        
        Returns:
            로그우도 값
        """
        pass
    
    def fit(self, data, latent_var) -> Dict:
        """
        구조모델 단독 추정 (Sequential 방식용)
        
        OLS 회귀분석
        
        Args:
            data: 사회인구학적 변수 데이터
            latent_var: 잠재변수 값 (측정모델에서 추정)
        
        Returns:
            추정 결과 (gamma, sigma)
        """
        pass
```

---

#### **2.2 Apollo R 코드 참조**

**King (2022) Apollo R 코드**:
```r
# 구조방정식 정의
LV = gamma_age * age + 
     gamma_gender * gender + 
     gamma_income * income + 
     eta

# eta는 표준정규분포
eta ~ N(0, 1)

# 동시 추정에서 사용
apollo_randCoeff = function(apollo_beta, apollo_inputs) {
    randcoeff = list()
    randcoeff[["LV"]] = gamma_age * age + 
                        gamma_gender * gender + 
                        gamma_income * income + 
                        eta
    return(randcoeff)
}
```

---

#### **2.3 구현 필요성**

**구조모델이 없으면**:
1. ❌ ICLV 동시 추정 불가능
2. ❌ 사회인구학적 변수의 간접효과 추정 불가능
3. ❌ Unconditional WTP 계산 불가능
4. ❌ 모집단 평균 효과 추정 불가능

**구조모델이 있으면**:
1. ✅ 사회인구학적 변수 → 잠재변수 → 선택 경로 분석
2. ✅ 직접효과 vs 간접효과 분해
3. ✅ 개인별 이질성 모델링
4. ✅ 정책 시뮬레이션 가능

---

### **3. 관련 파일 현황**

#### **3.1 존재하는 파일**

| 파일 | 상태 | 크기 | 설명 |
|------|------|------|------|
| `measurement_equations.py` | ✅ 완성 | 329 lines | Ordered Probit 측정모델 |
| `simultaneous_estimator.py` | ⚠️ 부분 | 386 lines | 동시 추정 엔진 (구조모델 필요) |
| `iclv_config.py` | ✅ 완성 | 200+ lines | 설정 클래스 |
| `__init__.py` | ⚠️ 오류 | 93 lines | import 오류 (파일 누락) |

#### **3.2 누락된 파일**

| 파일 | 상태 | 우선순위 | 설명 |
|------|------|----------|------|
| `structural_equations.py` | ❌ 없음 | **최우선** | 구조모델 (필수) |
| `iclv_analyzer.py` | ❌ 없음 | 높음 | 메인 분석기 |
| `wtp_calculator.py` | ❌ 없음 | 중간 | WTP 계산기 |

---

### **4. SimultaneousEstimator 분석**

#### **4.1 현재 구현 상태**

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator.py` (386 lines)

**클래스**: `SimultaneousEstimator`

**핵심 메서드**:
```python
class SimultaneousEstimator:
    def estimate(self, data, measurement_model, structural_model, choice_model):
        # ⚠️ 부분 구현
        # structural_model 파라미터를 받지만 실제 파일 없음
        pass
    
    def _joint_log_likelihood(self, params, measurement_model, 
                             structural_model, choice_model):
        # ⚠️ 부분 구현
        # 구조모델 메서드 호출하지만 실제 구현 없음
        
        # Line 195: 구조모델 예측 (미구현)
        lv = structural_model.predict(ind_data, param_dict['structural'], draw)
        
        # Line 208: 구조모델 로그우도 (미구현)
        ll_structural = structural_model.log_likelihood(
            ind_data, lv, param_dict['structural'], draw
        )
```

**문제점**:
- `structural_model.predict()` 호출하지만 `LatentVariableRegression` 클래스 없음
- `structural_model.log_likelihood()` 호출하지만 메서드 없음
- **실행 시 AttributeError 발생 예상**

---

#### **4.2 결합 우도함수**

**이론적 정의**:
```
L = ∏ᵢ ∫ P(Choice|LV) × P(Indicators|LV) × P(LV|X) dLV

시뮬레이션 근사:
L ≈ ∏ᵢ (1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)

여기서:
- P(Choice|LV): 선택모델 우도
- P(Indicators|LV): 측정모델 우도 ✅ (구현됨)
- P(LV|X): 구조모델 우도 ❌ (미구현)
```

**현재 상태**:
- ✅ `P(Indicators|LV)`: `measurement_model.log_likelihood()` 구현됨
- ❌ `P(LV|X)`: `structural_model.log_likelihood()` 미구현
- ⚠️ `P(Choice|LV)`: `choice_model.log_likelihood()` 부분 구현

---

### **5. __init__.py Import 오류**

#### **5.1 현재 Import 문**

**파일**: `src/analysis/hybrid_choice_model/iclv_models/__init__.py`

```python
# Line 31-35: 존재하지 않는 모듈 import
from .iclv_analyzer import (
    ICLVAnalyzer,
    ICLVResults,
    run_iclv_analysis
)

# Line 42-45: 존재하지 않는 모듈 import
from .structural_equations import (
    LatentVariableRegression,
    estimate_structural_model
)

# Line 53-57: 존재하지 않는 모듈 import
from .wtp_calculator import (
    WTPCalculator,
    calculate_conditional_wtp,
    calculate_unconditional_wtp
)
```

**결과**:
- `import` 시 `ModuleNotFoundError` 발생
- 모듈 전체 사용 불가능
- 테스트에서 직접 import 우회 필요

---

#### **5.2 테스트에서의 우회 방법**

**현재 테스트 코드**:
```python
# tests/test_ordered_probit_reversed_data.py

# 직접 파일 경로로 import (우회)
measurement_equations_path = project_root / "src" / "analysis" / \
    "hybrid_choice_model" / "iclv_models" / "measurement_equations.py"
import importlib.util
spec = importlib.util.spec_from_file_location("measurement_equations", 
                                               measurement_equations_path)
measurement_equations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement_equations)
OrderedProbitMeasurement = measurement_equations.OrderedProbitMeasurement
```

**문제점**:
- 정상적인 import 불가능
- 모든 테스트에서 우회 필요
- 유지보수 어려움

---

## 🎯 구현 우선순위

### **최우선 (P0): structural_equations.py 구현**

**이유**:
1. ICLV 동시 추정의 핵심 컴포넌트
2. SimultaneousEstimator가 이미 호출하고 있음
3. 없으면 전체 ICLV 시스템 작동 불가

**예상 작업량**: 1-2일

**구현 내용**:
- `LatentVariableRegression` 클래스
- `predict()` 메서드: LV = γ*X + σ*draw
- `log_likelihood()` 메서드: P(LV|X) ~ N(γ*X, σ²)
- `fit()` 메서드: OLS 회귀분석

---

### **높음 (P1): iclv_analyzer.py 구현**

**이유**:
1. 사용자 친화적 인터페이스
2. 전체 ICLV 파이프라인 통합
3. 문서화된 예제 코드에서 사용

**예상 작업량**: 1일

**구현 내용**:
- `ICLVAnalyzer` 클래스
- `fit()` 메서드: 전체 추정 파이프라인
- `ICLVResults` 클래스: 결과 저장
- `run_iclv_analysis()` 헬퍼 함수

---

### **중간 (P2): wtp_calculator.py 구현**

**이유**:
1. WTP 계산은 최종 목표
2. 구조모델 완성 후 구현 가능
3. Conditional/Unconditional WTP 모두 필요

**예상 작업량**: 1일

**구현 내용**:
- `WTPCalculator` 클래스
- `calculate_conditional_wtp()`: 개인별 LV 조건부
- `calculate_unconditional_wtp()`: 모집단 평균

---

## ✅ 완성된 부분 요약

### **측정모델 (OrderedProbitMeasurement)**

| 항목 | 상태 |
|------|------|
| **클래스 구현** | ✅ 완성 |
| **log_likelihood()** | ✅ 완성 |
| **predict_probabilities()** | ✅ 완성 |
| **predict()** | ✅ 완성 |
| **fit()** | ✅ 완성 |
| **Apollo R 동일성** | ✅ 검증 (차이 0.0) |
| **실제 데이터 테스트** | ✅ 완료 (5개 요인) |
| **역코딩 데이터 테스트** | ✅ 완료 (8.59% 개선) |

---

## ❌ 미완성 부분 요약

### **구조모델 (LatentVariableRegression)**

| 항목 | 상태 |
|------|------|
| **파일 존재** | ❌ 없음 |
| **클래스 구현** | ❌ 없음 |
| **predict()** | ❌ 없음 |
| **log_likelihood()** | ❌ 없음 |
| **fit()** | ❌ 없음 |

### **기타 누락 파일**

| 파일 | 상태 |
|------|------|
| `iclv_analyzer.py` | ❌ 없음 |
| `wtp_calculator.py` | ❌ 없음 |

---

## 📌 최종 결론

### **잠재변수 모델 완성도: 50%**

| 컴포넌트 | 완성도 | 상태 |
|----------|--------|------|
| **측정모델** | 100% | ✅ 완전 구현 |
| **구조모델** | 0% | ❌ 파일 없음 |
| **전체** | **50%** | ⚠️ 부분 완성 |

---

### **ICLV 시스템 작동 가능 여부**

| 기능 | 가능 여부 | 이유 |
|------|-----------|------|
| **측정모델 단독 추정** | ✅ 가능 | OrderedProbitMeasurement 완성 |
| **Sequential 추정** | ⚠️ 부분 가능 | 구조모델 OLS로 대체 가능 |
| **Simultaneous 추정** | ❌ 불가능 | 구조모델 필수 |
| **Conditional WTP** | ⚠️ 부분 가능 | 잠재변수 고정 시 |
| **Unconditional WTP** | ❌ 불가능 | 구조모델 필수 |

---

### **다음 단계**

**즉시 조치 필요**:
1. **structural_equations.py 구현** (최우선)
   - `LatentVariableRegression` 클래스
   - King (2022) Apollo R 코드 기반
   - 예상 작업량: 1-2일

2. **__init__.py 수정**
   - 누락된 파일 import 제거 또는 주석 처리
   - 정상적인 import 가능하도록 수정

3. **iclv_analyzer.py 구현** (높은 우선순위)
   - 사용자 친화적 인터페이스
   - 전체 파이프라인 통합

4. **wtp_calculator.py 구현** (중간 우선순위)
   - Conditional/Unconditional WTP
   - 최종 목표 달성

---

**보고서 작성일**: 2025-11-04  
**검토자**: Sugar Substitute Research Team  
**상태**: ⚠️ 구조모델 구현 필요

