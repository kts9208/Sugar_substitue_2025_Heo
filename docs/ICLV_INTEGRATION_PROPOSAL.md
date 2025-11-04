# 🎯 ICLV 모델 통합 제안서
## PhDHybridChoiceModelPaper 코드 분석 및 현재 프로젝트 연계 방안

---

## 📊 1. 참조 논문 분석 요약

### 논문 정보
- **제목**: "Willingness-to-pay for precautionary control of microplastics, a comparison of hybrid choice models"
- **저자**: Dr Peter King (University of Kent)
- **게재지**: Journal of Environmental Economics and Policy (JEEP)
- **DOI**: https://doi.org/10.1080/21606544.2022.2146757
- **GitHub**: https://github.com/pmpk20/PhDHybridChoiceModelPaper

### 핵심 방법론: ICLV (Integrated Choice and Latent Variable) 모델

#### 모델 구조
```
ICLV 모델 = 측정모델 (Measurement Model) + 구조모델 (Structural Model) + 선택모델 (Choice Model)

1. 측정모델 (Measurement Equations)
   - 잠재변수(LV) → 관측지표(Indicators)
   - 예: 위험인식(LV) → Q13, Q14, Q15 (5점 척도)
   - Ordered Probit 모델 사용

2. 구조모델 (Structural Equations)
   - 사회인구학적 변수 → 잠재변수
   - 예: LV = γ₁*Age + γ₂*Gender + ... + η (오차항)

3. 선택모델 (Choice Model)
   - 효용함수에 잠재변수 포함
   - V = intercept + β_bid*Bid + λ*LV + β*X
   - Binary Choice (Yes/No) - Ordered Probit
```

#### 사용 패키지: Apollo (R)
- **특징**: 
  - 동시 추정 (Simultaneous Estimation)
  - Halton Draws를 이용한 시뮬레이션 (1000 draws)
  - 패널 데이터 처리
  - Unconditional & Conditional WTP 계산

---

## 🔍 2. 현재 프로젝트 vs 참조 논문 비교

### 2.1 공통점 ✅

| 항목 | 현재 프로젝트 | 참조 논문 |
|------|--------------|----------|
| **잠재변수 모델링** | SEM (semopy) | ICLV (apollo) |
| **선택모델** | MNL, RPL 지원 | Ordered Probit |
| **데이터 통합** | DCE + SEM 통합 | DCE + Indicators 통합 |
| **요인점수 계산** | 평균 기반 | 동시 추정 |
| **모듈화 설계** | 팩토리 패턴 | 함수형 |

### 2.2 차이점 및 개선 기회 🎯

| 항목 | 현재 프로젝트 | 참조 논문 | 개선 방향 |
|------|--------------|----------|----------|
| **추정 방법** | 2단계 (Sequential) | 동시 추정 (Simultaneous) | ⭐ 동시 추정 구현 |
| **측정모델** | CFA (연속형) | Ordered Probit | ⭐ Ordered Probit 추가 |
| **구조모델** | 경로분석 | 잠재변수 회귀 | ⭐ 구조방정식 강화 |
| **WTP 계산** | 기본 계산 | Conditional/Unconditional | ⭐ 고급 WTP 계산 |
| **시뮬레이션** | 기본 | Halton Draws | ⭐ Halton Draws 구현 |
| **사회인구학적 변수** | 선택모델만 | 양쪽 모두 | ⭐ 이중 통합 |

---

## 💡 3. 구체적 통합 제안

### 제안 1: ICLV 전용 모듈 추가 ⭐⭐⭐⭐⭐

#### 구현 위치
```
src/analysis/hybrid_choice_model/
├── iclv_models/                        # 새로 추가
│   ├── __init__.py
│   ├── iclv_analyzer.py               # ICLV 메인 분석기
│   ├── measurement_equations.py       # 측정방정식 (Ordered Probit)
│   ├── structural_equations.py        # 구조방정식
│   ├── simultaneous_estimator.py      # 동시 추정 엔진
│   └── wtp_calculator.py              # WTP 계산기
```

#### 핵심 기능
```python
class ICLVAnalyzer:
    """
    ICLV (Integrated Choice and Latent Variable) 모델 분석기
    
    참조: King (2022) - Microplastics WTP study
    """
    
    def __init__(self, config: ICLVConfig):
        self.measurement_model = OrderedProbitMeasurement()
        self.structural_model = LatentVariableRegression()
        self.choice_model = OrderedProbitChoice()
        self.estimator = SimultaneousEstimator()
    
    def fit(self, data: pd.DataFrame, 
            indicators: List[str],
            sociodemographics: List[str],
            choice_attributes: List[str]):
        """
        동시 추정 수행
        
        1. 측정방정식: LV → Indicators (Ordered Probit)
        2. 구조방정식: Sociodemographics → LV
        3. 선택방정식: Attributes + LV → Choice
        """
        # 동시 우도함수 최대화
        results = self.estimator.maximize_joint_likelihood(
            measurement_eq=self.measurement_model,
            structural_eq=self.structural_model,
            choice_eq=self.choice_model,
            data=data
        )
        
        return results
    
    def calculate_wtp(self, results, method='conditional'):
        """
        WTP 계산
        
        - Conditional: 개인별 잠재변수 조건부
        - Unconditional: 모집단 평균
        """
        if method == 'conditional':
            return self._conditional_wtp(results)
        else:
            return self._unconditional_wtp(results)
```

### 제안 2: Ordered Probit 측정모델 구현 ⭐⭐⭐⭐

#### 현재 문제
- 현재는 연속형 CFA만 지원
- 리커트 척도 데이터를 연속형으로 처리

#### 개선 방안
```python
class OrderedProbitMeasurement:
    """
    Ordered Probit 측정모델
    
    리커트 척도 데이터를 올바르게 모델링
    """
    
    def __init__(self, n_categories: int = 5):
        self.n_categories = n_categories
        self.thresholds = None  # τ (tau) 파라미터
        self.loadings = None    # ζ (zeta) 파라미터
    
    def fit(self, indicators: pd.DataFrame, latent_var: str):
        """
        측정모델 추정
        
        P(Y_i = k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
        
        여기서:
        - Y_i: 관측지표 (1~5)
        - τ: 임계값 (thresholds)
        - ζ: 요인적재량 (loadings)
        - LV: 잠재변수
        - Φ: 표준정규 누적분포함수
        """
        # 최대우도 추정
        pass
```

### 제안 3: 동시 추정 엔진 구현 ⭐⭐⭐⭐⭐

#### 핵심 아이디어
```python
class SimultaneousEstimator:
    """
    ICLV 모델 동시 추정
    
    참조: Apollo 패키지의 동시 추정 방법론
    """
    
    def maximize_joint_likelihood(self, 
                                  measurement_eq,
                                  structural_eq,
                                  choice_eq,
                                  data: pd.DataFrame,
                                  n_draws: int = 1000,
                                  draw_type: str = 'halton'):
        """
        결합 우도함수 최대화
        
        L = ∏ᵢ ∫ P(Choice|LV) × P(Indicators|LV) × P(LV|X) dLV
        
        시뮬레이션 기반 추정:
        1. Halton Draws 생성 (η ~ N(0,1))
        2. 각 draw에 대해 우도 계산
        3. 평균하여 시뮬레이션 우도 계산
        4. 최대화
        """
        
        # Halton Draws 생성
        draws = self._generate_halton_draws(n_draws, data.shape[0])
        
        # 시뮬레이션 우도 계산
        def joint_log_likelihood(params):
            ll = 0
            for draw in draws:
                # 구조방정식으로 LV 계산
                lv = structural_eq.predict(data, params, draw)
                
                # 측정모델 우도
                ll_measurement = measurement_eq.log_likelihood(data, lv, params)
                
                # 선택모델 우도
                ll_choice = choice_eq.log_likelihood(data, lv, params)
                
                ll += ll_measurement + ll_choice
            
            return ll / n_draws  # 평균
        
        # 최적화
        result = scipy.optimize.minimize(
            lambda p: -joint_log_likelihood(p),
            initial_params,
            method='BFGS'
        )
        
        return result
    
    def _generate_halton_draws(self, n_draws: int, n_individuals: int):
        """Halton 시퀀스 생성 (준난수)"""
        # scipy.stats.qmc.Halton 사용
        from scipy.stats import qmc
        sampler = qmc.Halton(d=1, scramble=True)
        draws = sampler.random(n=n_draws * n_individuals)
        return draws.reshape(n_individuals, n_draws)
```

### 제안 4: 사회인구학적 변수 이중 통합 ⭐⭐⭐

#### 참조 논문의 접근
```R
# 구조방정식 (Structural Equation)
LV = gamma_Age*Age + gamma_Gender*Gender + ... + eta

# 선택방정식 (Choice Equation)
V = intercept + b_bid*Bid + lambda*LV + 
    beta_Age*Age + beta_Gender*Gender + ...
```

#### 현재 프로젝트 적용
```python
class DualSociodemographicIntegration:
    """
    사회인구학적 변수를 구조모델과 선택모델 양쪽에 통합
    
    장점:
    1. 직접효과 (선택모델): β*X → Choice
    2. 간접효과 (구조모델): γ*X → LV → Choice
    3. 총효과 = 직접효과 + 간접효과
    """
    
    def build_model_spec(self, 
                        latent_vars: List[str],
                        sociodemographics: List[str],
                        choice_attributes: List[str]):
        """
        이중 통합 모델 스펙 생성
        """
        
        # 구조방정식
        structural_eqs = []
        for lv in latent_vars:
            eq = f"{lv} ~ " + " + ".join(sociodemographics)
            structural_eqs.append(eq)
        
        # 선택방정식
        choice_eq = "choice ~ " + " + ".join(choice_attributes) + \
                    " + " + " + ".join(latent_vars) + \
                    " + " + " + ".join(sociodemographics)
        
        return "\n".join(structural_eqs + [choice_eq])
```

### 제안 5: 고급 WTP 계산 ⭐⭐⭐⭐

#### Conditional vs Unconditional WTP
```python
class AdvancedWTPCalculator:
    """
    고급 WTP 계산기
    
    참조: King (2022) WTP 계산 방법
    """
    
    def calculate_conditional_wtp(self, model_results, individual_data):
        """
        조건부 WTP (Conditional WTP)
        
        개인별 잠재변수 값을 조건으로 한 WTP
        
        WTP_i = -intercept/β_price + λ*LV_i/β_price
        
        여기서 LV_i는 개인 i의 조건부 잠재변수 값
        """
        # 개인별 잠재변수 추정 (Posterior mean)
        lv_conditional = self._estimate_conditional_lv(
            model_results, individual_data
        )
        
        # WTP 계산
        intercept = model_results.params['intercept']
        beta_price = model_results.params['b_bid']
        lambda_lv = model_results.params['lambda']
        
        wtp = (-intercept + lambda_lv * lv_conditional) / beta_price
        
        return wtp
    
    def calculate_unconditional_wtp(self, model_results, n_simulations=10000):
        """
        무조건부 WTP (Unconditional WTP)
        
        모집단 평균 WTP
        
        E[WTP] = -intercept/β_price + λ*E[LV]/β_price
        
        시뮬레이션 기반 계산
        """
        # 잠재변수 분포에서 샘플링
        lv_samples = self._simulate_lv_distribution(
            model_results, n_simulations
        )
        
        # 각 샘플에 대해 WTP 계산
        intercept = model_results.params['intercept']
        beta_price = model_results.params['b_bid']
        lambda_lv = model_results.params['lambda']
        
        wtp_samples = (-intercept + lambda_lv * lv_samples) / beta_price
        
        # 평균 및 신뢰구간
        return {
            'mean': np.mean(wtp_samples),
            'median': np.median(wtp_samples),
            'std': np.std(wtp_samples),
            'ci_lower': np.percentile(wtp_samples, 2.5),
            'ci_upper': np.percentile(wtp_samples, 97.5)
        }
```

---

## 🚀 4. 구현 로드맵

### Phase 1: 기초 구현 (1-2주)
- [ ] Ordered Probit 측정모델 구현
- [ ] 기본 ICLV 분석기 구조 설계
- [ ] 데이터 형식 정의 및 검증

### Phase 2: 핵심 기능 (2-3주)
- [ ] 동시 추정 엔진 구현
- [ ] Halton Draws 생성기
- [ ] 사회인구학적 변수 이중 통합

### Phase 3: 고급 기능 (1-2주)
- [ ] Conditional/Unconditional WTP 계산
- [ ] 모델 비교 기능
- [ ] 결과 시각화

### Phase 4: 통합 및 테스트 (1주)
- [ ] 기존 모듈과 통합
- [ ] 단위 테스트 작성
- [ ] 문서화

---

## 📝 5. 사용 예시

### 설탕 대체재 연구 적용
```python
from src.analysis.hybrid_choice_model.iclv_models import ICLVAnalyzer

# 데이터 준비
dce_data = pd.read_csv("data/dce_data.csv")
survey_data = pd.read_csv("data/survey_data.csv")

# ICLV 분석기 설정
config = ICLVConfig(
    # 측정모델
    latent_variable='health_concern',
    indicators=['health_concern_1', 'health_concern_2', 'health_concern_3'],
    indicator_type='ordered',  # 5점 척도
    
    # 구조모델
    sociodemographics=['age', 'gender', 'income', 'education'],
    include_in_choice=True,  # 선택모델에도 포함
    
    # 선택모델
    choice_attributes=['price', 'sugar_content', 'health_label'],
    choice_type='binary',  # Yes/No
    
    # 추정 설정
    estimation_method='simultaneous',
    n_draws=1000,
    draw_type='halton'
)

# 분석 실행
analyzer = ICLVAnalyzer(config)
results = analyzer.fit(dce_data, survey_data)

# WTP 계산
wtp_conditional = analyzer.calculate_wtp(results, method='conditional')
wtp_unconditional = analyzer.calculate_wtp(results, method='unconditional')

print(f"평균 WTP: {wtp_unconditional['mean']:.2f}원")
print(f"95% CI: [{wtp_unconditional['ci_lower']:.2f}, {wtp_unconditional['ci_upper']:.2f}]")
```

---

## 🎯 6. 기대 효과

### 학술적 기여
1. **방법론적 엄밀성**: 동시 추정으로 일관된 모수 추정
2. **모델 비교**: Sequential vs Simultaneous 비교 가능
3. **출판 가능성**: ICLV 방법론 적용으로 논문 수준 향상

### 실무적 가치
1. **정확한 WTP 추정**: 조건부/무조건부 WTP 계산
2. **정책 시뮬레이션**: 다양한 시나리오 분석
3. **시장 세분화**: 개인별 선호 이질성 파악

### 기술적 발전
1. **모듈 확장**: 기존 시스템에 ICLV 추가
2. **재사용성**: 다른 연구에도 적용 가능
3. **Python 생태계**: R Apollo의 Python 구현

---

## 📚 7. 참고 자료

### 핵심 논문
1. King, P. M. (2022). Willingness-to-pay for precautionary control of microplastics. JEEP.
2. Ben-Akiva et al. (2002). Hybrid choice models. Marketing Letters.
3. Train, K. (2009). Discrete Choice Methods with Simulation.

### 코드 참조
- GitHub: https://github.com/pmpk20/PhDHybridChoiceModelPaper
- Apollo 패키지: http://www.apollochoicemodelling.com/

### 현재 프로젝트 문서
- `HYBRID_CHOICE_MODEL_GUIDE.md`
- `HYBRID_CHOICE_MODEL_IMPLEMENTATION_SUMMARY.md`

---

**작성일**: 2025-11-03  
**작성자**: Sugar Substitute Research Team  
**버전**: 1.0

