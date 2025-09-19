# Factor Analysis Module

semopy를 이용한 확인적 요인분석(CFA) 및 Factor Loading 분석을 위한 모듈입니다.

## 📋 개요

이 모듈은 전처리된 설문조사 데이터를 사용하여 확인적 요인분석을 수행하고, factor loading 값을 계산하여 CSV 파일로 저장하는 기능을 제공합니다.

### 주요 특징

- **기존 모듈 활용**: 기존 `FactorConfig`와 전처리된 데이터를 적극 활용
- **semopy 기반**: 강력한 구조방정식 모델링 라이브러리 사용
- **재사용성**: 모듈화된 설계로 높은 재사용성
- **확장성**: 새로운 요인이나 분석 방법 쉽게 추가 가능
- **종합적 결과**: Factor loadings, 적합도 지수, 표준화 결과 등 포함

## 🚀 설치 및 요구사항

### 필수 라이브러리

```bash
pip install pandas numpy semopy
```

### 선택적 라이브러리

```bash
pip install matplotlib seaborn  # 시각화용 (향후 확장)
```

## 📁 모듈 구조

```
factor_analysis/
├── __init__.py              # 패키지 초기화
├── config.py                # 분석 설정 및 모델 스펙 생성
├── data_loader.py           # CSV 파일 로딩
├── factor_analyzer.py       # semopy 기반 요인분석
├── results_exporter.py      # 결과 CSV 저장
├── test_factor_analysis.py  # 단위 테스트
├── example_usage.py         # 사용 예제
└── README.md               # 이 파일
```

## 🎯 주요 클래스

### 1. FactorDataLoader
전처리된 요인별 CSV 파일들을 로딩합니다.

```python
from factor_analysis import FactorDataLoader

loader = FactorDataLoader()
data = loader.load_single_factor('health_concern')
```

### 2. SemopyAnalyzer
semopy를 사용한 확인적 요인분석을 수행합니다.

```python
from factor_analysis import SemopyAnalyzer, FactorAnalysisConfig

config = FactorAnalysisConfig(estimator='ML', standardized=True)
analyzer = SemopyAnalyzer(config)
```

### 3. FactorResultsExporter
분석 결과를 다양한 형태의 CSV 파일로 저장합니다.

```python
from factor_analysis import FactorResultsExporter

exporter = FactorResultsExporter("results/")
exporter.export_comprehensive_results(results)
```

## 📖 사용 방법

### 1. 기본 사용법

```python
from factor_analysis import analyze_factor_loading, export_factor_results

# 단일 요인 분석
results = analyze_factor_loading('health_concern')

# 결과 저장
saved_files = export_factor_results(results)
print(f"저장된 파일들: {list(saved_files.keys())}")
```

### 2. 다중 요인 분석

```python
# 여러 요인 동시 분석
factors = ['health_concern', 'perceived_benefit', 'purchase_intention']
results = analyze_factor_loading(factors)

# 요인별 loadings 확인
loadings = results['factor_loadings']
for factor in loadings['Factor'].unique():
    factor_loadings = loadings[loadings['Factor'] == factor]
    print(f"\n{factor} Factor Loadings:")
    print(factor_loadings[['Item', 'Loading', 'P_value']].to_string(index=False))
```

### 3. 사용자 정의 설정

```python
from factor_analysis import create_custom_config, FactorAnalyzer

# 사용자 정의 설정
config = create_custom_config(
    estimator='ML',
    optimizer='L-BFGS-B',
    max_iterations=2000,
    confidence_level=0.99
)

# 설정을 사용한 분석
analyzer = FactorAnalyzer(config=config)
results = analyzer.analyze_single_factor('health_concern')
```

## 📊 결과 해석

### Factor Loadings 테이블

| Factor | Item | Loading | SE | Z_value | P_value | Significant |
|--------|------|---------|----|---------|---------|-----------| 
| health_concern | q6 | 0.75 | 0.05 | 15.0 | 0.000 | True |
| health_concern | q7 | 0.68 | 0.06 | 11.3 | 0.000 | True |

### 적합도 지수

- **CFI (Comparative Fit Index)**: ≥ 0.95 (Excellent), ≥ 0.90 (Good)
- **TLI (Tucker-Lewis Index)**: ≥ 0.95 (Excellent), ≥ 0.90 (Good)  
- **RMSEA (Root Mean Square Error)**: ≤ 0.05 (Excellent), ≤ 0.08 (Good)
- **SRMR (Standardized Root Mean Square Residual)**: ≤ 0.05 (Excellent), ≤ 0.08 (Good)

## 🔧 고급 사용법

### 모델 스펙 직접 생성

```python
from factor_analysis import create_factor_model_spec

# 단일 요인 모델
spec = create_factor_model_spec(single_factor='health_concern')
print(spec)

# 다중 요인 모델 (상관관계 허용)
spec = create_factor_model_spec(
    factor_names=['health_concern', 'perceived_benefit'],
    allow_correlations=True
)
```

### 결과 개별 저장

```python
from factor_analysis import FactorResultsExporter

exporter = FactorResultsExporter("my_results/")

# Factor loadings만 저장
loadings_file = exporter.export_factor_loadings(results)

# 적합도 지수만 저장  
fit_file = exporter.export_fit_indices(results)

# 요약 보고서 저장
summary_file = exporter.export_summary_report(results)
```

## 🧪 테스트

```bash
# 단위 테스트 실행
python factor_analysis/test_factor_analysis.py

# 사용 예제 실행
python factor_analysis/example_usage.py
```

## 📝 분석 가능한 요인들

현재 분석 가능한 요인들 (DCE 변수와 인구통계학적 변수 제외):

- `health_concern`: 소비자의 건강관심도
- `perceived_benefit`: substitute의 지각된 유익성  
- `purchase_intention`: substitute의 구매의도
- `perceived_price`: 인지된 가격수준
- `nutrition_knowledge`: 소비자의 영양지식 수준

## ⚠️ 주의사항

1. **데이터 요구사항**: 전처리된 CSV 파일들이 `processed_data/survey_data/` 디렉토리에 있어야 합니다.
2. **semopy 설치**: `pip install semopy` 명령으로 semopy를 설치해야 합니다.
3. **샘플 크기**: 요인분석을 위해서는 충분한 샘플 크기가 필요합니다 (일반적으로 변수당 5-10개 관측치).
4. **결측치**: 기본적으로 listwise deletion을 사용합니다.

## 🔄 기존 모듈과의 연계

- `processed_data/modules/survey_data_preprocessor.py`의 `FactorConfig` 활용
- 전처리된 CSV 파일들을 직접 로딩하여 중복 방지
- 기존 요인 정의와 문항 구성을 그대로 사용

## 📈 향후 확장 계획

- [ ] 탐색적 요인분석(EFA) 기능 추가
- [ ] 시각화 기능 (factor loading plot, path diagram)
- [ ] 다집단 분석 기능
- [ ] 종단 요인분석 기능
- [ ] 베이지안 요인분석 옵션
