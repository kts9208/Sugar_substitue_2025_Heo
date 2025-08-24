# Survey Data Preprocessor

설문조사 raw data를 통계계산 모듈에 적용하기 쉽게 전처리하는 Python 모듈입니다.

## 주요 기능

- Excel 파일에서 설문조사 데이터 로딩
- 요인별 문항 그룹핑 (8개 요인)
- 통계 분석용 데이터 형태로 변환
- CSV 파일로 요인별 데이터 내보내기
- 포괄적인 유닛 테스트 포함

## 요인 구성

| 요인명 | 문항 번호 | 설명 |
|--------|-----------|------|
| demographics_1 | q1-q5 | 인구통계학적 변수 (1차) |
| health_concern | q6-q11 | 소비자의 건강관심도 |
| perceived_benefit | q12-q17 | substitute의 지각된 유익성 |
| purchase_intention | q18-q20 | substitute의 구매의도 |
| dce_variables | q21-q26 | DCE 변수 (추후 별도 처리예정) |
| perceived_price | q27-q29 | 인지된 가격수준 |
| nutrition_knowledge | q30-q49 | 소비자의 영양지식 수준 |
| demographics_2 | q50-q56 | 인구통계학적 변수 (2차) |

## 설치 및 요구사항

```bash
pip install pandas openpyxl
```

## 기본 사용법

### 1. 간단한 사용 예제

```python
from survey_data_preprocessor import SurveyDataPreprocessor

# 전처리기 초기화
preprocessor = SurveyDataPreprocessor("Raw data/Sugar_substitue_Raw data_250730.xlsx")

# 데이터 로딩 및 준비
preprocessor.load_and_prepare_data()

# 특정 요인 데이터 가져오기
health_data = preprocessor.get_factor_data('health_concern')
print(f"건강관심도 데이터 형태: {health_data.shape}")

# 모든 요인 데이터 가져오기
all_factors = preprocessor.get_all_factors_data()

# CSV 파일로 내보내기
preprocessor.export_factor_data('processed_data')
```

### 2. 요인 정보 확인

```python
from survey_data_preprocessor import FactorConfig

# 모든 요인 목록
factors = FactorConfig.get_all_factors()

# 특정 요인의 문항들
health_questions = FactorConfig.get_factor_questions('health_concern')

# 요인 설명
description = FactorConfig.get_factor_description('health_concern')
```

### 3. 통계 분석 준비

```python
# 건강관심도 요인 데이터
health_data = preprocessor.get_factor_data('health_concern')

# 기술통계
stats = health_data.iloc[:, 1:].describe()  # 'no' 컬럼 제외

# 상관관계 분석
correlation = health_data.iloc[:, 1:].corr()

# 요인분석용 행렬 (numpy array)
matrix = health_data.iloc[:, 1:].values
```

## 클래스 구조

### SurveyDataPreprocessor (메인 클래스)
- `load_and_prepare_data()`: 데이터 로딩 및 준비
- `get_factor_data(factor_name)`: 특정 요인 데이터 반환
- `get_all_factors_data()`: 모든 요인 데이터 반환
- `get_summary()`: 요인 요약 정보 반환
- `export_factor_data(output_dir)`: CSV 파일로 내보내기

### DataLoader
- Excel 파일 로딩 및 기본 검증
- 시트 이름 확인 기능

### FactorGrouper
- 요인별 문항 그룹핑
- 데이터 유효성 검증

### FactorConfig
- 요인 정의 및 설정 관리
- 문항 번호 매핑

## 출력 파일

`export_factor_data()` 실행 시 생성되는 파일들:

```
processed_data/
├── demographics_1.csv      # 인구통계학적 변수 (1차)
├── health_concern.csv      # 소비자의 건강관심도
├── perceived_benefit.csv   # substitute의 지각된 유익성
├── purchase_intention.csv  # substitute의 구매의도
├── dce_variables.csv       # DCE 변수
├── perceived_price.csv     # 인지된 가격수준
├── nutrition_knowledge.csv # 소비자의 영양지식 수준
├── demographics_2.csv      # 인구통계학적 변수 (2차)
└── factors_summary.csv     # 요인 요약 정보
```

## 테스트 실행

```bash
python test_survey_preprocessor.py
```

모든 테스트는 실제 raw data를 사용하여 수행됩니다.

## 예제 실행

```bash
python example_usage.py
```

## 설계 원칙

- **재사용성**: 모듈화된 클래스 구조
- **가독성**: 명확한 메서드명과 문서화
- **단일책임**: 각 클래스는 하나의 책임만 담당
- **확장성**: 새로운 요인 추가 용이
- **유지보수성**: 설정과 로직 분리

## 에러 처리

- 파일 존재 여부 확인
- Excel 파일 형식 검증
- 데이터 유효성 검사
- 누락된 문항에 대한 경고

## 로깅

모든 주요 작업에 대해 INFO 레벨 로깅을 제공합니다:
- 데이터 로딩 상태
- 요인별 그룹핑 결과
- 파일 내보내기 상태

## 주의사항

1. Excel 파일은 'DATA' 시트에 데이터가 있어야 합니다.
2. 문항 번호는 q1, q2_1, q3, ... 형태여야 합니다.
3. 'no' 컬럼이 응답자 식별용으로 사용됩니다.
4. 누락된 문항이 있을 경우 경고 메시지가 출력됩니다.

## 라이선스

Sugar Substitute Research Team - 2025
