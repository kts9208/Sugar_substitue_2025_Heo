# Sugar Substitute Research - Statistical Analysis System

설탕 대체재에 대한 소비자 행동 연구를 위한 종합적인 통계 분석 시스템입니다.

## 프로젝트 개요

이 프로젝트는 설탕 대체재에 대한 소비자 선택 실험 및 설문조사 데이터를 분석하기 위한 모듈화된 통계 분석 시스템입니다. 구조방정식모델링(SEM), 요인분석, 경로분석, 다항로짓모델 등 다양한 고급 통계 기법을 제공합니다.

## 주요 특징

- **모듈화된 설계**: 각 분석 기법별로 독립적인 모듈 구성
- **재사용성**: 다양한 연구 데이터에 적용 가능한 범용적 구조
- **확장성**: 새로운 분석 기법 추가 용이
- **가독성**: 명확한 함수명과 상세한 문서화
- **검증**: 포괄적인 테스트 코드 및 결과 검증

## 핵심 분석 모듈

### 📊 1. 데이터 전처리 (Data Preprocessing)
- **위치**: `processed_data/modules/`
- **기능**: 원본 Excel 데이터를 분석 가능한 형태로 변환
- **주요 모듈**: 설문조사 전처리, DCE 전처리, 역문항 처리

### 🔬 2. 요인분석 (Factor Analysis)
- **위치**: `factor_analysis/`
- **기능**: semopy 기반 확인적 요인분석(CFA)
- **주요 기능**: Factor Loading, 모델 적합도, 신뢰도 계산

### 📈 3. 경로분석 (Path Analysis)
- **위치**: `path_analysis/`
- **기능**: 구조방정식모델링(SEM) 기반 경로분석
- **주요 기능**: 매개효과, 직간접효과, 경로계수 분석

### 🎯 4. 다항로짓모델 (Multinomial Logit)
- **위치**: `multinomial_logit/`
- **기능**: DCE 데이터 기반 선택모델 분석
- **주요 기능**: 선택확률, 한계효과, 민감도 분석

## 주요 실행 스크립트

### 🚀 핵심 분석 실행
```bash
# 요인분석 실행
python run_factor_analysis.py

# 신뢰도 분석 실행
python run_reliability_analysis.py

# 통합 신뢰도 분석 (역문항 처리 포함)
python run_integrated_reliability_analysis.py

# 경로분석 실행
python run_path_analysis_5factors.py

# 판별타당도 검증
python run_discriminant_validity_analysis.py

# 상관관계 분석 및 시각화
python run_semopy_correlations.py
python run_correlation_visualization.py

# 다항로짓모델 분석
python multinomial_logit/mnl_analysis.py
```

### 📋 필요한 라이브러리
```bash
pip install semopy statsmodels pandas numpy scipy matplotlib seaborn networkx
```

## 데이터 구조

### 📁 주요 데이터 디렉토리
- `Raw data/`: 원본 Excel 데이터 파일
- `processed_data/survey_data/`: 전처리된 설문조사 데이터 (5개 요인별 CSV)
- `processed_data/dce_data/`: DCE 실험 데이터
- `processed_data/modules/`: 데이터 전처리 모듈

### 📊 분석 결과 디렉토리
- `factor_analysis_results/`: 요인분석 결과
- `path_analysis_results/`: 경로분석 결과
- `discriminant_validity_results/`: 판별타당도 결과
- `correlation_visualization_results/`: 상관관계 시각화
- `reliability_analysis_results/`: 신뢰도 분석 결과

## 분석 파이프라인

### 🔄 표준 분석 순서
1. **데이터 전처리** (`processed_data/modules/`)
   - 원본 Excel → CSV 변환
   - 역문항 처리
   - 데이터 검증

2. **요인분석** (`factor_analysis/`)
   - 확인적 요인분석(CFA)
   - Factor Loading 계산
   - 모델 적합도 평가

3. **신뢰도 검증** (`factor_analysis/reliability_calculator.py`)
   - Cronbach's Alpha
   - Composite Reliability (CR)
   - Average Variance Extracted (AVE)

4. **판별타당도 검증** (`discriminant_validity_analyzer.py`)
   - Fornell-Larcker 기준
   - 상관관계 vs AVE 제곱근 비교

5. **경로분석** (`path_analysis/`)
   - 구조방정식모델링
   - 매개효과 분석
   - 직간접효과 계산

6. **선택실험분석** (`multinomial_logit/`)
   - 다항로짓모델
   - 선택확률 예측
   - 민감도 분석

## 주요 연구 결과

1. **무설탕 제품 선호**: 무설탕 제품이 일반당 제품 대비 2.22배 높은 선택 확률
2. **건강라벨 역설**: 건강라벨이 있는 제품이 오히려 1.70배 낮은 선택 확률
3. **가격 프리미엄**: 가격이 높을수록 선택 확률이 증가 (4.38배)
4. **대안 효과**: 대안 A와 B 간 유의한 차이 없음

### 모델 적합도

- **Log-Likelihood**: -2201.52
- **AIC**: 4413.05
- **BIC**: 4443.70
- **Pseudo R-squared**: 0.054
Sugar substitute research project 2025 - Heo
