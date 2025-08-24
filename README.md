# Multinomial Logit Model for DCE Analysis

DCE(Discrete Choice Experiment) 데이터를 사용한 Multinomial Logit Model 구현 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 설탕 대체재에 대한 소비자 선택 실험 데이터를 분석하기 위해 Multinomial Logit Model을 구현합니다. StatsModels 라이브러리를 적극적으로 활용하여 모듈화된 구조로 설계되었습니다.

## 주요 특징

- **모듈화된 설계**: 단일 책임 원칙에 따른 함수 분리
- **재사용성**: 다른 DCE 데이터에도 적용 가능한 범용적 구조
- **가독성**: 명확한 함수명과 상세한 문서화
- **유지보수성**: 설정 기반 모델 구성
- **검증**: 포괄적인 테스트 코드 포함

## 프로젝트 구조

```
multinomial_logit/
├── __init__.py              # 패키지 초기화
├── data_loader.py           # DCE 데이터 로딩 모듈
├── data_preprocessor.py     # 데이터 전처리 모듈
├── model_config.py          # 모델 설정 관리 모듈
├── model_estimator.py       # 모델 추정 모듈
└── results_analyzer.py      # 결과 분석 모듈

tests/
└── test_multinomial_logit.py # 테스트 코드

main_analysis.py             # 메인 실행 스크립트
README.md                    # 프로젝트 문서
```

## 설치 및 실행

### 필요한 라이브러리

```bash
pip install statsmodels pandas numpy scipy
```

### 실행 방법

1. **메인 분석 실행**:
```bash
python main_analysis.py
```

2. **테스트 실행**:
```bash
python tests/test_multinomial_logit.py
```

## 데이터 요구사항

프로젝트는 `processed_data/dce_data/` 디렉토리에 다음 파일들이 있어야 합니다:

- `dce_choice_matrix.csv`: 선택 매트릭스 데이터
- `dce_attribute_data.csv`: 속성 데이터
- `dce_choice_summary.csv`: 선택 요약 데이터
- `dce_choice_sets_config.csv`: 선택 세트 설정
- `dce_raw_data.csv`: 원본 데이터
- `dce_validation_report.csv`: 검증 보고서

## 분석 결과

### 주요 발견사항

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
