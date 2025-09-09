# Moderation Analysis Module

semopy를 사용한 조절효과 분석(Moderation Analysis) 모듈입니다.

## 📋 개요

이 모듈은 설탕 대체재 연구를 위한 조절효과 분석 기능을 제공합니다. semopy를 기반으로 하여 상호작용항을 통한 조절효과를 분석하고, 단순기울기 분석, 조건부 효과 계산 등의 고급 기능을 포함합니다.

### 주요 특징

- **독립적 설계**: 다른 모듈과 완전히 독립적으로 작동
- **semopy 기반**: 강력한 구조방정식 모델링 라이브러리 사용
- **포괄적 분석**: 조절효과 검정, 단순기울기 분석, 조건부 효과 계산
- **자동화된 시각화**: 상호작용 플롯, 단순기울기 그래프, 히트맵 생성
- **결과 저장**: CSV, JSON, 요약보고서 자동 생성
- **재사용성**: 모듈화된 설계로 높은 재사용성

## 🚀 설치 및 요구사항

### 필수 라이브러리

```bash
pip install semopy pandas numpy scipy matplotlib seaborn scikit-learn
```

### 선택적 라이브러리

```bash
pip install jupyter  # 노트북 환경에서 사용 시
```

## 📁 모듈 구조

```
moderation_analysis/
├── __init__.py              # 패키지 초기화 및 주요 함수 export
├── config.py                # 분석 설정 및 기본 설정
├── data_loader.py           # 데이터 로딩 및 전처리
├── interaction_builder.py   # 상호작용항 생성 및 모델 구축
├── moderation_analyzer.py   # 핵심 조절효과 분석 엔진
├── results_exporter.py      # 결과 저장 (CSV, JSON, 보고서)
├── visualizer.py           # 시각화 (그래프, 차트)
├── example_usage.py        # 사용 예제
├── test_moderation_analysis.py  # 단위 테스트
└── README.md               # 이 파일
```

## 🎯 주요 클래스

### 1. ModerationAnalyzer
조절효과 분석의 핵심 클래스입니다.

```python
from moderation_analysis import ModerationAnalyzer

analyzer = ModerationAnalyzer()
results = analyzer.analyze_moderation_effects(
    independent_var='health_concern',
    dependent_var='purchase_intention',
    moderator_var='nutrition_knowledge'
)
```

### 2. ModerationDataLoader
5개 요인 데이터를 로드하고 조절효과 분석을 위해 준비합니다.

```python
from moderation_analysis import ModerationDataLoader

loader = ModerationDataLoader()
data = loader.prepare_moderation_data(
    'health_concern', 'purchase_intention', 'nutrition_knowledge'
)
```

### 3. InteractionBuilder
상호작용항 생성 및 조절효과 모델을 구축합니다.

```python
from moderation_analysis import InteractionBuilder

builder = InteractionBuilder()
interaction_data = builder.create_interaction_terms(
    data, 'health_concern', 'nutrition_knowledge'
)
```

### 4. ModerationVisualizer
조절효과 분석 결과를 시각화합니다.

```python
from moderation_analysis import ModerationVisualizer

visualizer = ModerationVisualizer()
plot_files = visualizer.visualize_comprehensive_analysis(data, results)
```

## 📖 사용 방법

### 1. 기본 사용법

```python
from moderation_analysis import analyze_moderation_effects, export_moderation_results

# 조절효과 분석
results = analyze_moderation_effects(
    independent_var='health_concern',
    dependent_var='purchase_intention',
    moderator_var='nutrition_knowledge'
)

# 결과 저장
saved_files = export_moderation_results(results)
print(f"저장된 파일들: {list(saved_files.keys())}")
```

### 2. 사용자 정의 설정

```python
from moderation_analysis import create_custom_moderation_config, analyze_moderation_effects

# 사용자 정의 설정
config = create_custom_moderation_config(
    results_dir="my_moderation_results",
    bootstrap_samples=1000,
    confidence_level=0.99,
    center_variables=True
)

# 분석 실행
results = analyze_moderation_effects(
    independent_var='perceived_benefit',
    dependent_var='purchase_intention',
    moderator_var='perceived_price'
)
```

### 3. 단계별 상세 분석

```python
from moderation_analysis import (
    load_moderation_data, create_interaction_terms, 
    calculate_simple_slopes, visualize_moderation_analysis
)

# 1. 데이터 로드
data = load_moderation_data(
    'health_concern', 'purchase_intention', 'nutrition_knowledge'
)

# 2. 상호작용항 생성
interaction_data = create_interaction_terms(
    data, 'health_concern', 'nutrition_knowledge'
)

# 3. 단순기울기 분석
simple_slopes = calculate_simple_slopes(
    'health_concern', 'purchase_intention', 'nutrition_knowledge', 
    interaction_data
)

# 4. 시각화
plot_files = visualize_moderation_analysis(interaction_data, results)
```

## 🔍 분석 결과 해석

### 조절효과 검정 결과

```python
moderation_test = results['moderation_test']
print(f"상호작용 계수: {moderation_test['interaction_coefficient']:.4f}")
print(f"P값: {moderation_test['p_value']:.4f}")
print(f"유의성: {'유의함' if moderation_test['significant'] else '유의하지 않음'}")
print(f"해석: {moderation_test['interpretation']}")
```

### 단순기울기 분석 결과

```python
simple_slopes = results['simple_slopes']
for level, slope_info in simple_slopes.items():
    print(f"{level}: 기울기={slope_info['simple_slope']:.4f}, "
          f"P값={slope_info['p_value']:.4f}")
```

## 📊 시각화 기능

### 1. 조절효과 플롯
독립변수와 종속변수 간의 관계가 조절변수 수준에 따라 어떻게 달라지는지 보여줍니다.

### 2. 단순기울기 그래프
조절변수의 각 수준(Low, Mean, High)에서 독립변수의 효과를 막대그래프로 표시합니다.

### 3. 상호작용 히트맵
독립변수와 조절변수의 모든 조합에서 종속변수의 예측값을 히트맵으로 표시합니다.

## 📁 결과 파일

분석 결과는 `moderation_analysis_results/` 디렉토리에 저장됩니다:

- `*_coefficients_*.csv`: 회귀계수 테이블
- `*_simple_slopes_*.csv`: 단순기울기 분석 결과
- `*_conditional_effects_*.csv`: 조건부 효과 결과
- `*_fit_indices_*.csv`: 모델 적합도 지수
- `*_full_results_*.json`: 전체 결과 (JSON 형태)
- `*_summary_report_*.txt`: 요약 보고서
- `moderation_plot_*.png`: 조절효과 시각화
- `simple_slopes_*.png`: 단순기울기 그래프
- `interaction_heatmap_*.png`: 상호작용 히트맵

## 🧪 테스트

```bash
# 전체 테스트 실행
python moderation_analysis/test_moderation_analysis.py

# 특정 테스트 클래스 실행
python -m unittest moderation_analysis.test_moderation_analysis.TestModerationAnalyzer
```

## 📚 예제 실행

```bash
# 모든 예제 실행
python moderation_analysis/example_usage.py

# 특정 예제 실행
python moderation_analysis/example_usage.py 1  # 기본 조절효과 분석
python moderation_analysis/example_usage.py 2  # 사용자 정의 설정
python moderation_analysis/example_usage.py 3  # 단계별 상세 분석
python moderation_analysis/example_usage.py 4  # 포괄적 시각화
python moderation_analysis/example_usage.py 5  # 다중 조절효과 분석
```

## 🚀 메인 스크립트 실행

```bash
# 기본 조절효과 분석 실행
python run_moderation_analysis.py

# 사용자 정의 분석 실행
python run_moderation_analysis.py --custom
```

## 🔧 설정 옵션

### ModerationAnalysisConfig 주요 옵션

- `data_dir`: 데이터 디렉토리 경로
- `results_dir`: 결과 저장 디렉토리 경로
- `estimator`: 추정 방법 ("ML", "GLS", "WLS", "ULS")
- `standardized`: 표준화 계수 사용 여부
- `bootstrap_samples`: 부트스트래핑 샘플 수
- `confidence_level`: 신뢰수준
- `center_variables`: 변수 중심화 여부
- `simple_slopes_values`: 단순기울기 분석 수준

## 📈 지원하는 분석

1. **조절효과 검정**: 상호작용항의 유의성 검정
2. **단순기울기 분석**: 조절변수 수준별 독립변수 효과
3. **조건부 효과**: 조절변수 값에 따른 효과 변화
4. **모델 적합도**: CFI, TLI, RMSEA, SRMR 등
5. **부트스트래핑**: 신뢰구간 계산 (향후 확장 예정)

## 🎯 사용 가능한 요인

- `health_concern`: 건강관심도 (q6~q11)
- `perceived_benefit`: 지각된 혜택 (q16~q17)
- `purchase_intention`: 구매의도 (q18~q19)
- `perceived_price`: 지각된 가격 (q20~q21)
- `nutrition_knowledge`: 영양지식 (q30~q49)

## ⚠️ 주의사항

1. **데이터 요구사항**: `processed_data/survey_data/` 디렉토리에 5개 요인별 CSV 파일이 있어야 합니다.
2. **semopy 의존성**: semopy 라이브러리가 설치되어 있어야 합니다.
3. **메모리 사용량**: 큰 데이터셋의 경우 메모리 사용량이 클 수 있습니다.
4. **수렴 문제**: 복잡한 모델의 경우 수렴하지 않을 수 있습니다.

## 🔄 버전 정보

- **Version**: 1.0.0
- **Author**: Sugar Substitute Research Team
- **Date**: 2025-09-09

## 📞 지원

문제가 발생하거나 기능 요청이 있으시면 이슈를 등록해 주세요.
