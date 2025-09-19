# Path Analysis Module

semopy를 사용한 경로분석(Path Analysis) 모듈입니다. 구조방정식모델링(SEM)의 핵심 기능인 경로분석을 통해 변수 간의 인과관계와 매개효과를 분석할 수 있습니다.

## 🎯 주요 기능

### 1. 경로분석 모델 구축
- **단순 매개모델**: X → M → Y
- **다중 매개모델**: X → M1,M2,... → Y  
- **순차 매개모델**: X → M1 → M2 → Y
- **사용자 정의 구조모델**: 복합적인 경로 관계

### 2. 효과 분석
- **직접효과**: 독립변수가 종속변수에 미치는 직접적 영향
- **간접효과**: 매개변수를 통한 간접적 영향
- **총효과**: 직접효과 + 간접효과
- **매개효과 분석**: Sobel test, Bootstrap 신뢰구간

### 3. 모델 평가
- **적합도 지수**: CFI, TLI, RMSEA, SRMR 등
- **경로계수 유의성**: Z-test, p-value
- **모델 비교**: AIC, BIC 기준

### 4. 결과 시각화
- **경로 다이어그램**: semopy 내장 기능 활용
- **효과 차트**: 직접/간접/총효과 비교
- **적합도 지수 차트**: 기준선과 함께 표시
- **매개효과 분석 차트**: Sobel test 결과

## 📦 모듈 구조

```
path_analysis/
├── __init__.py              # 패키지 초기화 및 주요 함수 export
├── config.py                # 설정 클래스 및 사전 정의된 모델 템플릿
├── model_builder.py         # 경로분석 모델 구축
├── path_analyzer.py         # 핵심 분석 엔진
├── effects_calculator.py    # 효과 계산 (직접/간접/총효과)
├── results_exporter.py      # 결과 저장 (CSV, JSON, 요약보고서)
├── visualizer.py           # 시각화 (차트, 다이어그램)
├── example_usage.py        # 사용 예제
└── README.md               # 이 파일
```

## 🚀 빠른 시작

### 1. 기본 사용법

```python
from path_analysis import analyze_path_model, create_path_model

# 단순 매개모델 생성
model_spec = create_path_model(
    model_type='simple_mediation',
    independent_var='health_concern',
    mediator_var='perceived_benefit', 
    dependent_var='purchase_intention'
)

# 분석 실행
variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
results = analyze_path_model(model_spec, variables)

print("분석 완료!")
```

### 2. 다중 매개모델

```python
# 다중 매개모델 생성
model_spec = create_path_model(
    model_type='multiple_mediation',
    independent_var='health_concern',
    mediator_vars=['perceived_benefit', 'nutrition_knowledge'],
    dependent_var='purchase_intention'
)

# 분석 실행
variables = ['health_concern', 'perceived_benefit', 'nutrition_knowledge', 'purchase_intention']
results = analyze_path_model(model_spec, variables)
```

### 3. 효과 분석

```python
from path_analysis import EffectsCalculator

# 효과 계산기 생성
effects_calc = EffectsCalculator(results['model_object'])

# 모든 효과 계산
effects = effects_calc.calculate_all_effects(
    independent_var='health_concern',
    dependent_var='purchase_intention',
    mediator_vars=['perceived_benefit']
)

print(f"직접효과: {effects['direct_effects']['coefficient']:.4f}")
print(f"간접효과: {effects['indirect_effects']['total_indirect_effect']:.4f}")
print(f"총효과: {effects['total_effects']['total_effect']:.4f}")
```

### 4. 결과 저장 및 시각화

```python
from path_analysis import export_path_results, PathAnalysisVisualizer

# 결과 저장
saved_files = export_path_results(results, filename_prefix="my_analysis")

# 시각화
visualizer = PathAnalysisVisualizer()
viz_files = visualizer.create_comprehensive_visualization(results)
```

## 📊 지원하는 모델 유형

### 1. 단순 매개모델 (Simple Mediation)
```
X → M → Y
X -------→ Y (직접효과)
```

### 2. 다중 매개모델 (Multiple Mediation)  
```
    → M1 →
X          Y
    → M2 →
X -------→ Y (직접효과)
```

### 3. 순차 매개모델 (Serial Mediation)
```
X → M1 → M2 → Y
X -------------→ Y (직접효과)
```

### 4. 사용자 정의 모델 (Custom Model)
```python
# 복합적인 경로 관계 정의 가능
variables = ['X1', 'X2', 'M1', 'M2', 'Y']
paths = [
    ('X1', 'M1'),
    ('X2', 'M1'), 
    ('M1', 'M2'),
    ('M2', 'Y'),
    ('X1', 'Y')
]
```

## ⚙️ 설정 옵션

### 기본 설정
```python
from path_analysis import create_default_path_config

config = create_default_path_config(
    estimator='MLW',           # 추정방법
    optimizer='SLSQP',         # 최적화 알고리즘
    standardized=True,         # 표준화 계수
    bootstrap_samples=1000,    # 부트스트랩 샘플 수
    confidence_level=0.95      # 신뢰수준
)
```

### 매개효과 분석용 설정
```python
from path_analysis import create_mediation_config

config = create_mediation_config(
    bootstrap_samples=5000,    # 매개효과는 더 많은 부트스트랩 필요
    include_bootstrap_ci=True,
    calculate_effects=True
)
```

## 📈 결과 해석

### 적합도 지수 기준
- **CFI/TLI**: > 0.95 (우수), > 0.90 (수용가능)
- **RMSEA**: < 0.06 (우수), < 0.08 (수용가능)  
- **SRMR**: < 0.08 (우수), < 0.10 (수용가능)
- **Chi-square**: 비유의적(p > 0.05)이 바람직

### 매개효과 유형
- **완전매개**: 간접효과 유의, 직접효과 비유의
- **부분매개**: 간접효과와 직접효과 모두 유의
- **매개효과 없음**: 간접효과 비유의

## 🔧 고급 사용법

### 1. 모델 비교
```python
# 여러 모델의 적합도 비교
models = {
    'direct': direct_model_results,
    'mediation': mediation_model_results
}

for name, results in models.items():
    aic = results['fit_indices']['aic']
    print(f"{name} model AIC: {aic:.2f}")
```

### 2. Bootstrap 신뢰구간
```python
# 매개효과의 Bootstrap 신뢰구간 계산 (향후 구현 예정)
bootstrap_ci = effects_calc.bootstrap_mediation_ci(
    independent_var='X',
    dependent_var='Y', 
    mediator_vars=['M'],
    n_bootstrap=5000
)
```

### 3. 다중집단 분석 (향후 구현 예정)
```python
# 집단별 경로계수 차이 검정
multigroup_results = analyze_multigroup_path_model(
    model_spec=model_spec,
    data=data,
    group_var='gender'
)
```

## 📋 요구사항

### 필수 패키지
- `semopy >= 2.0.0`: 구조방정식모델링
- `pandas >= 1.3.0`: 데이터 처리
- `numpy >= 1.20.0`: 수치 계산
- `matplotlib >= 3.3.0`: 기본 시각화
- `seaborn >= 0.11.0`: 고급 시각화
- `scipy >= 1.7.0`: 통계 계산

### 선택적 패키지
- `graphviz`: 경로 다이어그램 생성 (semopy 내장 기능)

## 📝 사용 예제

자세한 사용 예제는 `example_usage.py` 파일을 참조하세요:

1. **단순 매개모델**: 건강관심도 → 지각된혜택 → 구매의도
2. **다중 매개모델**: 건강관심도 → [지각된혜택, 영양지식] → 구매의도  
3. **사용자 정의 모델**: 복합적인 경로 관계
4. **효과 비교**: 여러 매개변수의 효과 크기 비교
5. **모델 비교**: 직접효과 모델 vs 매개모델

## 🤝 기여하기

이 모듈은 Sugar Substitute Research Team에서 개발되었습니다. 
버그 리포트나 기능 제안은 이슈로 등록해 주세요.

## 📄 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다.

---

**Author**: Sugar Substitute Research Team  
**Date**: 2025-09-07  
**Version**: 1.0.0
