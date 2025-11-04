# 🚀 ICLV 모델 구현 예시 및 사용 가이드

## 📋 목차
1. [기본 사용법](#1-기본-사용법)
2. [King (2022) 재현](#2-king-2022-재현)
3. [설탕 대체재 연구 적용](#3-설탕-대체재-연구-적용)
4. [Sequential vs Simultaneous 비교](#4-sequential-vs-simultaneous-비교)
5. [고급 기능](#5-고급-기능)

---

## 1. 기본 사용법

### 1.1 간단한 ICLV 분석

```python
import pandas as pd
from src.analysis.hybrid_choice_model.iclv_models import (
    create_iclv_config,
    ICLVAnalyzer
)

# 데이터 로드
data = pd.read_csv("data/integrated_data.csv")

# 설정 생성
config = create_iclv_config(
    latent_variable='health_concern',
    indicators=['hc_1', 'hc_2', 'hc_3'],
    sociodemographics=['age', 'gender', 'income'],
    choice_attributes=['price', 'sugar_content'],
    price_variable='price',
    n_draws=1000
)

# 분석 실행
analyzer = ICLVAnalyzer(config)
results = analyzer.fit(data)

# 결과 확인
print(f"Log-Likelihood: {results.log_likelihood:.2f}")
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")

# WTP 계산
wtp = analyzer.calculate_wtp(method='unconditional')
print(f"평균 WTP: {wtp['mean']:.2f}원")
```

### 1.2 단계별 분석

```python
# 1단계: 데이터 준비
from src.analysis.hybrid_choice_model.iclv_models import (
    prepare_iclv_data,
    validate_iclv_data
)

# 데이터 검증
validation_result = validate_iclv_data(data, config)
if not validation_result['valid']:
    print("데이터 검증 실패:", validation_result['errors'])
    exit()

# 2단계: 모델 구성요소 생성
from src.analysis.hybrid_choice_model.iclv_models import (
    OrderedProbitMeasurement,
    LatentVariableRegression,
    OrderedProbitChoice
)

measurement_model = OrderedProbitMeasurement(config.measurement)
structural_model = LatentVariableRegression(config.structural)
choice_model = OrderedProbitChoice(config.choice)

# 3단계: 동시 추정
from src.analysis.hybrid_choice_model.iclv_models import (
    SimultaneousEstimator
)

estimator = SimultaneousEstimator(config)
results = estimator.estimate(
    data,
    measurement_model,
    structural_model,
    choice_model
)

# 4단계: 결과 분석
print("\n=== 측정모델 결과 ===")
print("요인적재량 (zeta):")
for i, ind in enumerate(config.measurement.indicators):
    zeta = results['parameters']['measurement']['zeta'][i]
    print(f"  {ind}: {zeta:.3f}")

print("\n=== 구조모델 결과 ===")
print("사회인구학적 변수 계수 (gamma):")
for i, var in enumerate(config.structural.sociodemographics):
    gamma = results['parameters']['structural']['gamma'][i]
    print(f"  {var}: {gamma:.3f}")

print("\n=== 선택모델 결과 ===")
print(f"잠재변수 계수 (lambda): {results['parameters']['choice']['lambda']:.3f}")
print("속성 계수 (beta):")
for i, attr in enumerate(config.choice.choice_attributes):
    beta = results['parameters']['choice']['beta'][i]
    print(f"  {attr}: {beta:.3f}")
```

---

## 2. King (2022) 재현

### 2.1 마이크로플라스틱 WTP 연구 재현

```python
from src.analysis.hybrid_choice_model.iclv_models import (
    create_king2022_config,
    ICLVAnalyzer
)

# King (2022) 스타일 설정
config = create_king2022_config(
    latent_variable='risk_perception',
    indicators=[
        'Q13_current_threat',
        'Q14_future_threat', 
        'Q15_environment_threat'
    ],
    sociodemographics=[
        'age', 'gender', 'distance', 'income',
        'experts', 'bp', 'charity', 'certainty', 'consequentiality'
    ],
    choice_attributes=['bid'],
    n_draws=1000,
    draw_type='halton'
)

# 데이터 로드 (King 2022 형식)
data = pd.read_csv("data/microplastics_data.csv")

# 분석 실행
analyzer = ICLVAnalyzer(config)
results = analyzer.fit(data)

# WTP 계산 (King 2022 방식)
# Conditional WTP
wtp_conditional = analyzer.calculate_wtp(method='conditional')
print(f"Conditional WTP 중앙값: {wtp_conditional['median']:.2f}")

# Unconditional WTP
wtp_unconditional = analyzer.calculate_wtp(method='unconditional')
print(f"Unconditional WTP 중앙값: {wtp_unconditional['median']:.2f}")

# 결과 저장 (Apollo 스타일)
analyzer.save_results(
    output_dir='results/king2022_replication',
    format='apollo_style'
)
```

### 2.2 사회인구학적 변수 이중 통합

```python
# King (2022)의 핵심: 사회인구학적 변수를 양쪽에 포함

config = create_king2022_config(
    # ... 기본 설정 ...
    include_in_choice=True  # 선택모델에도 포함
)

# 결과 해석
results = analyzer.fit(data)

# 직접효과 (선택모델)
direct_effects = results['parameters']['choice']['beta_sociodem']

# 간접효과 (구조모델 → 잠재변수 → 선택)
gamma = results['parameters']['structural']['gamma']
lambda_lv = results['parameters']['choice']['lambda']
indirect_effects = gamma * lambda_lv

# 총효과
total_effects = direct_effects + indirect_effects

print("\n=== 효과 분해 ===")
for i, var in enumerate(config.structural.sociodemographics):
    print(f"{var}:")
    print(f"  직접효과: {direct_effects[i]:.3f}")
    print(f"  간접효과: {indirect_effects[i]:.3f}")
    print(f"  총효과: {total_effects[i]:.3f}")
```

---

## 3. 설탕 대체재 연구 적용

### 3.1 기본 설정

```python
from src.analysis.hybrid_choice_model.iclv_models import (
    create_sugar_substitute_config,
    ICLVAnalyzer
)

# 설탕 대체재 연구용 설정
config = create_sugar_substitute_config(
    latent_variable='health_concern',
    indicators=[
        'health_concern_1', 'health_concern_2', 'health_concern_3',
        'health_concern_4', 'health_concern_5', 'health_concern_6',
        'health_concern_7'
    ],
    sociodemographics=['age', 'gender', 'income', 'education'],
    choice_attributes=['price', 'sugar_content', 'health_label', 'brand'],
    n_categories=7,  # 7점 척도
    choice_type='multinomial',  # 다항선택
    n_draws=1000
)

# 데이터 로드
dce_data = pd.read_csv("data/processed/dce/choice_data.csv")
survey_data = pd.read_csv("data/processed/survey/health_concern.csv")

# 데이터 통합
from src.analysis.hybrid_choice_model.data_integration import (
    integrate_dce_sem_data
)

integrated_data = integrate_dce_sem_data(
    dce_data, 
    survey_data,
    individual_id='respondent_id'
)

# 분석 실행
analyzer = ICLVAnalyzer(config)
results = analyzer.fit(integrated_data)

# 결과 저장
analyzer.save_results(
    output_dir='results/current/iclv_analysis',
    include_plots=True
)
```

### 3.2 다중 잠재변수 모델

```python
# 여러 잠재변수를 동시에 모델링

from src.analysis.hybrid_choice_model.iclv_models import (
    MultiLatentICLVAnalyzer
)

# 설정
config = {
    'latent_variables': {
        'health_concern': {
            'indicators': ['hc_1', 'hc_2', 'hc_3', 'hc_4', 'hc_5', 'hc_6', 'hc_7'],
            'n_categories': 7
        },
        'perceived_benefit': {
            'indicators': ['pb_1', 'pb_2', 'pb_3', 'pb_4', 'pb_5', 'pb_6', 'pb_7'],
            'n_categories': 7
        },
        'nutrition_knowledge': {
            'indicators': ['nk_1', 'nk_2', 'nk_3', 'nk_4'],
            'n_categories': 7
        }
    },
    'sociodemographics': ['age', 'gender', 'income', 'education'],
    'choice_attributes': ['price', 'sugar_content', 'health_label', 'brand'],
    'n_draws': 1000
}

# 분석
analyzer = MultiLatentICLVAnalyzer(config)
results = analyzer.fit(integrated_data)

# 잠재변수 간 상관관계
correlations = analyzer.calculate_lv_correlations()
print("\n=== 잠재변수 상관관계 ===")
print(correlations)
```

---

## 4. Sequential vs Simultaneous 비교

### 4.1 Sequential 추정 (기존 방법)

```python
from src.analysis.hybrid_choice_model import (
    HybridChoiceAnalyzer,
    create_default_config
)

# Sequential 설정
config_seq = create_default_config()
config_seq.estimation.method = 'sequential'

# 분석
analyzer_seq = HybridChoiceAnalyzer(config_seq)
results_seq = analyzer_seq.analyze(dce_data, survey_data)

print("=== Sequential 추정 결과 ===")
print(f"Log-Likelihood: {results_seq.model_fit['log_likelihood']:.2f}")
print(f"AIC: {results_seq.model_fit['aic']:.2f}")
```

### 4.2 Simultaneous 추정 (ICLV)

```python
from src.analysis.hybrid_choice_model.iclv_models import (
    ICLVAnalyzer,
    create_iclv_config
)

# Simultaneous 설정
config_sim = create_iclv_config(
    latent_variable='health_concern',
    indicators=['hc_1', 'hc_2', 'hc_3'],
    sociodemographics=['age', 'gender'],
    choice_attributes=['price', 'sugar_content'],
    estimation_method='simultaneous',
    n_draws=1000
)

# 분석
analyzer_sim = ICLVAnalyzer(config_sim)
results_sim = analyzer_sim.fit(integrated_data)

print("=== Simultaneous 추정 결과 ===")
print(f"Log-Likelihood: {results_sim.log_likelihood:.2f}")
print(f"AIC: {results_sim.aic:.2f}")
```

### 4.3 비교 분석

```python
from src.analysis.hybrid_choice_model.iclv_models import (
    compare_estimation_methods
)

# 비교
comparison = compare_estimation_methods(
    sequential_results=results_seq,
    simultaneous_results=results_sim,
    data=integrated_data
)

print("\n=== 추정 방법 비교 ===")
print(f"Sequential LL: {comparison['sequential']['log_likelihood']:.2f}")
print(f"Simultaneous LL: {comparison['simultaneous']['log_likelihood']:.2f}")
print(f"LL 차이: {comparison['ll_difference']:.2f}")
print(f"AIC 차이: {comparison['aic_difference']:.2f}")

# 파라미터 비교
print("\n=== 파라미터 비교 ===")
for param in comparison['parameter_comparison']:
    print(f"{param['name']}:")
    print(f"  Sequential: {param['sequential']:.3f}")
    print(f"  Simultaneous: {param['simultaneous']:.3f}")
    print(f"  차이: {param['difference']:.3f}")
```

---

## 5. 고급 기능

### 5.1 부트스트랩 표준오차

```python
config = create_iclv_config(
    # ... 기본 설정 ...
    calculate_se=True,
    se_method='bootstrap',
    n_bootstrap=500
)

analyzer = ICLVAnalyzer(config)
results = analyzer.fit(data)

# 부트스트랩 신뢰구간
print("\n=== 부트스트랩 신뢰구간 ===")
for param_name, ci in results.bootstrap_ci.items():
    print(f"{param_name}: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
```

### 5.2 정책 시뮬레이션

```python
# 건강 라벨 정책의 효과 예측

# 기준 시나리오
baseline_scenario = {
    'price': 2000,
    'sugar_content': 50,
    'health_label': 0,
    'brand': 'A'
}

# 정책 시나리오 (건강 라벨 추가)
policy_scenario = {
    'price': 2000,
    'sugar_content': 50,
    'health_label': 1,
    'brand': 'A'
}

# 선택 확률 예측
prob_baseline = analyzer.predict_choice_probability(baseline_scenario)
prob_policy = analyzer.predict_choice_probability(policy_scenario)

print(f"기준 선택 확률: {prob_baseline:.3f}")
print(f"정책 선택 확률: {prob_policy:.3f}")
print(f"증가율: {(prob_policy - prob_baseline) / prob_baseline * 100:.1f}%")
```

### 5.3 개인별 잠재변수 추정

```python
# Conditional 잠재변수 값 추정

conditional_lv = analyzer.estimate_conditional_latent_variables(data)

# 개인별 값 확인
print("\n=== 개인별 잠재변수 (상위 5명) ===")
print(conditional_lv.head())

# 분포 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(conditional_lv['health_concern'], bins=50, alpha=0.7)
plt.xlabel('Health Concern (Latent Variable)')
plt.ylabel('Frequency')
plt.title('Distribution of Conditional Latent Variables')
plt.savefig('results/lv_distribution.png')
```

### 5.4 모델 진단

```python
# 모델 적합도 진단

diagnostics = analyzer.run_diagnostics(results)

print("\n=== 모델 진단 ===")
print(f"수렴 여부: {diagnostics['converged']}")
print(f"Hessian 양정부호: {diagnostics['hessian_positive_definite']}")
print(f"파라미터 유의성: {diagnostics['n_significant_params']}/{diagnostics['n_params']}")

# 잔차 분석
residuals = analyzer.calculate_residuals(results, data)
plt.figure(figsize=(10, 6))
plt.scatter(residuals['fitted'], residuals['residual'], alpha=0.5)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.savefig('results/residual_plot.png')
```

---

## 📊 결과 해석 가이드

### 측정모델 해석
- **요인적재량 (ζ)**: 잠재변수가 관측지표에 미치는 영향
  - 값이 클수록 해당 지표가 잠재변수를 잘 측정
  - 일반적으로 0.5 이상이면 양호

### 구조모델 해석
- **구조계수 (γ)**: 사회인구학적 변수가 잠재변수에 미치는 영향
  - 양수: 해당 변수가 증가하면 잠재변수도 증가
  - 음수: 해당 변수가 증가하면 잠재변수 감소

### 선택모델 해석
- **잠재변수 계수 (λ)**: 잠재변수가 선택에 미치는 영향
  - 양수: 잠재변수가 높을수록 선택 확률 증가
  - 통계적 유의성이 중요

### WTP 해석
- **Conditional WTP**: 개인별 특성을 고려한 WTP
- **Unconditional WTP**: 모집단 평균 WTP
- 일반적으로 Unconditional WTP가 정책 분석에 더 유용

---

**작성일**: 2025-11-03  
**버전**: 1.0

