# 📊 King (2022) vs 현재 프로젝트 상세 비교

## 🎯 Executive Summary

| 항목 | King (2022) | 현재 프로젝트 | 통합 후 |
|------|-------------|--------------|---------|
| **언어/플랫폼** | R (Apollo) | Python (semopy) | Python (통합) |
| **모델 유형** | ICLV | Hybrid Choice | ICLV + Hybrid |
| **추정 방법** | Simultaneous | Sequential | Both |
| **측정모델** | Ordered Probit | CFA (연속형) | Both |
| **시뮬레이션** | Halton Draws | 기본 | Halton Draws |
| **WTP 계산** | Conditional/Unconditional | 기본 | Advanced |

---

## 📚 1. 연구 배경 비교

### King (2022) - 마이크로플라스틱 연구

**연구 질문**
- 마이크로플라스틱 통제를 위한 지불의사액(WTP)은?
- 위험인식이 WTP에 미치는 영향은?
- 사회인구학적 특성의 직간접 효과는?

**데이터 구조**
```
개인 수: ~500명
선택 상황: Binary choice (Yes/No)
가격 범위: £0-£100
잠재변수: 위험인식 (Risk Perception)
관측지표: 3개 (Q13, Q14, Q15) - 5점 척도
사회인구학적: 9개 변수
```

**핵심 발견**
- Simultaneous 추정이 Sequential보다 우수
- 위험인식이 WTP에 유의한 영향
- 사회인구학적 변수의 이중 효과 확인

### 현재 프로젝트 - 설탕 대체재 연구

**연구 질문**
- 설탕 대체재에 대한 소비자 선호는?
- 건강관심도가 구매의도에 미치는 영향은?
- 다양한 요인들의 구조적 관계는?

**데이터 구조**
```
개인 수: TBD
선택 상황: Multinomial choice (여러 대안)
속성: 가격, 설탕함량, 건강라벨, 브랜드
잠재변수: 5개 (건강관심도, 지각된유익성, 구매의도, 지각된가격, 영양지식)
관측지표: 각 7개 - 7점 척도
사회인구학적: 4개 변수 (age, gender, income, education)
```

---

## 🔬 2. 방법론 상세 비교

### 2.1 모델 구조

#### King (2022) ICLV 모델

```
측정방정식 (Measurement Equations):
  P(Q13=k) = Φ(τ₁₃,ₖ - ζ₁₃*LV) - Φ(τ₁₃,ₖ₋₁ - ζ₁₃*LV)
  P(Q14=k) = Φ(τ₁₄,ₖ - ζ₁₄*LV) - Φ(τ₁₄,ₖ₋₁ - ζ₁₄*LV)
  P(Q15=k) = Φ(τ₁₅,ₖ - ζ₁₅*LV) - Φ(τ₁₅,ₖ₋₁ - ζ₁₅*LV)

구조방정식 (Structural Equation):
  LV = γ₁*Age + γ₂*Gender + ... + γ₉*Consequentiality + η
  η ~ N(0, σ²)

선택방정식 (Choice Equation):
  V = α + β_bid*Bid + λ*LV + β₁*Age + β₂*Gender + ... + β₉*Consequentiality
  P(Yes) = Φ(V)

동시 추정:
  L = ∏ᵢ ∫ P(Choice|LV) × P(Q13,Q14,Q15|LV) × P(LV|X) dLV
  
시뮬레이션:
  L ≈ ∏ᵢ (1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)
  R = 1000 Halton draws
```

#### 현재 프로젝트 (기존)

```
측정모델 (CFA):
  health_concern =~ hc_1 + hc_2 + ... + hc_7
  perceived_benefit =~ pb_1 + pb_2 + ... + pb_7
  ...

구조모델 (SEM):
  purchase_intention ~ health_concern + perceived_benefit + ...
  
선택모델 (MNL):
  V_j = β₁*price_j + β₂*sugar_j + β₃*label_j + β₄*brand_j
  P(j) = exp(V_j) / Σₖ exp(V_k)

순차 추정:
  1단계: SEM 추정 → 요인점수 계산
  2단계: 요인점수를 선택모델에 포함하여 추정
```

### 2.2 추정 방법 비교

| 측면 | Sequential (현재) | Simultaneous (King 2022) |
|------|------------------|-------------------------|
| **단계** | 2단계 분리 | 1단계 통합 |
| **일관성** | 비일관적 (inconsistent) | 일관적 (consistent) |
| **효율성** | 비효율적 | 효율적 |
| **표준오차** | 과소추정 가능 | 정확 |
| **계산 복잡도** | 낮음 | 높음 |
| **구현 난이도** | 쉬움 | 어려움 |

**Sequential의 문제점**
1. 1단계 추정 오차가 2단계로 전파
2. 표준오차가 과소추정됨
3. 파라미터 추정치가 편향될 수 있음

**Simultaneous의 장점**
1. 모든 정보를 동시에 활용
2. 일관된 추정치
3. 정확한 표준오차
4. 통계적으로 더 효율적

---

## 💻 3. 코드 구조 비교

### 3.1 King (2022) R 코드 구조

```r
# Apollo 패키지 사용
library(apollo)

# 1. 데이터 준비
database <- read.csv("data.csv")

# 2. Apollo 설정
apollo_control = list(
  modelName = "Q1ICLV",
  indivID = "ID",
  mixing = TRUE,
  nCores = 10
)

# 3. 파라미터 초기값
apollo_beta = c(
  intercept = 0,
  b_bid = 0,
  lambda = 1,
  gamma_Age = 0,
  gamma_Gender = 0,
  # ...
  zeta_Q13 = 1,
  zeta_Q14 = 1,
  zeta_Q15 = 1,
  tau_Q13_1 = -2,
  # ...
)

# 4. Halton Draws 설정
apollo_draws = list(
  interDrawsType = "halton",
  interNDraws = 1000,
  interNormDraws = c("eta")
)

# 5. 구조방정식 정의
apollo_randCoeff = function(apollo_beta, apollo_inputs) {
  randcoeff = list()
  randcoeff[["LV"]] = gamma_Age*Age + gamma_Gender*Gender + ... + eta
  return(randcoeff)
}

# 6. 우도함수 정의
apollo_probabilities = function(apollo_beta, apollo_inputs, functionality="estimate") {
  # 측정모델
  P[["indic_Q13"]] = apollo_op(op_settings1, functionality)
  P[["indic_Q14"]] = apollo_op(op_settings2, functionality)
  P[["indic_Q15"]] = apollo_op(op_settings3, functionality)
  
  # 선택모델
  P[['choice']] = apollo_op(op_settings, functionality)
  
  # 결합
  P = apollo_combineModels(P, apollo_inputs, functionality)
  P = apollo_panelProd(P, apollo_inputs, functionality)
  P = apollo_avgInterDraws(P, apollo_inputs, functionality)
  
  return(P)
}

# 7. 추정
model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

# 8. WTP 계산
unconditionals = apollo_unconditionals(model, apollo_probabilities, apollo_inputs)
conditionals = apollo_conditionals(model, apollo_probabilities, apollo_inputs)
```

### 3.2 현재 프로젝트 Python 코드 구조

```python
# 기존 Sequential 방식
from src.analysis.factor_analysis import FactorAnalyzer
from src.analysis.multinomial_logit import MultinomialLogitAnalyzer

# 1. SEM 분석
factor_analyzer = FactorAnalyzer(config)
sem_results = factor_analyzer.fit_model(survey_data, model_spec)

# 2. 요인점수 계산
factor_scores = factor_analyzer.calculate_factor_scores(survey_data)

# 3. DCE 데이터와 병합
integrated_data = pd.merge(dce_data, factor_scores, on='individual_id')

# 4. 선택모델 추정
mnl_analyzer = MultinomialLogitAnalyzer(config)
choice_results = mnl_analyzer.fit(integrated_data)
```

### 3.3 제안하는 ICLV Python 코드 구조

```python
# 새로운 Simultaneous 방식
from src.analysis.hybrid_choice_model.iclv_models import (
    ICLVAnalyzer,
    create_iclv_config
)

# 1. 설정
config = create_iclv_config(
    latent_variable='health_concern',
    indicators=['hc_1', 'hc_2', 'hc_3'],
    sociodemographics=['age', 'gender', 'income'],
    choice_attributes=['price', 'sugar_content'],
    n_draws=1000,
    draw_type='halton'
)

# 2. 분석 (한 번에!)
analyzer = ICLVAnalyzer(config)
results = analyzer.fit(integrated_data)

# 3. WTP 계산
wtp_conditional = analyzer.calculate_wtp(method='conditional')
wtp_unconditional = analyzer.calculate_wtp(method='unconditional')
```

---

## 📈 4. 예상 성능 비교

### 4.1 통계적 성능

| 지표 | Sequential | Simultaneous | 개선 |
|------|-----------|--------------|------|
| **Log-Likelihood** | -1000 | -950 | +50 |
| **AIC** | 2050 | 1950 | -100 |
| **BIC** | 2150 | 2050 | -100 |
| **표준오차** | 과소추정 | 정확 | ✓ |
| **파라미터 편향** | 있음 | 없음 | ✓ |

### 4.2 계산 성능

| 측면 | Sequential | Simultaneous |
|------|-----------|--------------|
| **실행 시간** | 1분 | 10-30분 |
| **메모리 사용** | 낮음 | 높음 |
| **수렴 안정성** | 높음 | 중간 |
| **초기값 민감도** | 낮음 | 높음 |

---

## 🎯 5. 통합 전략

### Phase 1: 기초 구현 (완료)
- ✅ ICLV 모듈 구조 설계
- ✅ 설정 시스템 구현
- ✅ Halton Draws 생성기
- ✅ 동시 추정 프레임워크

### Phase 2: 핵심 기능 (진행 중)
- 🔄 Ordered Probit 측정모델
- 🔄 구조방정식 모델
- 🔄 동시 우도함수 최적화
- 🔄 WTP 계산기

### Phase 3: 검증 및 테스트
- ⏳ King (2022) 재현 테스트
- ⏳ Sequential vs Simultaneous 비교
- ⏳ 설탕 대체재 데이터 적용
- ⏳ 성능 벤치마크

### Phase 4: 문서화 및 배포
- ⏳ 사용자 가이드 작성
- ⏳ API 문서화
- ⏳ 예제 노트북 작성
- ⏳ 논문 작성 지원

---

## 💡 6. 주요 개선 사항

### 6.1 방법론적 개선

1. **동시 추정 지원**
   - 일관된 파라미터 추정
   - 정확한 표준오차
   - 통계적 효율성 향상

2. **Ordered Probit 측정모델**
   - 리커트 척도 데이터의 올바른 처리
   - 범주형 데이터 모델링

3. **Halton Draws 시뮬레이션**
   - 준난수 사용으로 정확도 향상
   - 적은 draws로도 안정적 추정

4. **고급 WTP 계산**
   - Conditional WTP (개인별)
   - Unconditional WTP (모집단)
   - 신뢰구간 계산

### 6.2 기술적 개선

1. **모듈화 설계**
   - 재사용 가능한 컴포넌트
   - 확장 가능한 구조
   - 테스트 용이성

2. **Python 생태계 활용**
   - NumPy, SciPy 최적화
   - Pandas 데이터 처리
   - Matplotlib 시각화

3. **기존 시스템과 통합**
   - 기존 모듈 재사용
   - 일관된 인터페이스
   - 하위 호환성

---

## 📊 7. 실제 적용 시나리오

### 시나리오 1: 기본 분석
```python
# King (2022) 스타일 분석
config = create_king2022_config()
analyzer = ICLVAnalyzer(config)
results = analyzer.fit(data)
```

### 시나리오 2: 모델 비교
```python
# Sequential vs Simultaneous
results_seq = run_sequential_analysis(data)
results_sim = run_simultaneous_analysis(data)
comparison = compare_models(results_seq, results_sim)
```

### 시나리오 3: 정책 시뮬레이션
```python
# 건강 라벨 정책 효과
baseline = predict_choice(health_label=0)
policy = predict_choice(health_label=1)
impact = calculate_policy_impact(baseline, policy)
```

---

## 🎓 8. 학술적 기여

### 현재 프로젝트의 강점
1. **포괄적 분석**: 5개 잠재변수 동시 모델링
2. **다양한 모델**: MNL, RPL, Mixed Logit 지원
3. **Python 구현**: 접근성 및 재현성 향상

### King (2022)의 강점
1. **엄밀한 방법론**: Simultaneous 추정
2. **검증된 접근**: 출판된 연구
3. **실무 적용**: 정책 시뮬레이션

### 통합 후 기대 효과
1. **방법론적 우수성**: Simultaneous + 다중 잠재변수
2. **기술적 혁신**: Python ICLV 구현
3. **실용적 가치**: 정책 분석 도구

---

**작성일**: 2025-11-03  
**작성자**: Sugar Substitute Research Team  
**버전**: 1.0

