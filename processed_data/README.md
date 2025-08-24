# Sugar Substitute Research - Data Preprocessing Modules

설문조사 raw data를 통계 분석에 적합한 형태로 전처리하는 Python 모듈 모음입니다.

## 📁 프로젝트 구조

```
processed_data/
├── modules/                    # 핵심 모듈
│   ├── survey_data_preprocessor.py    # 설문조사 전처리 모듈
│   └── dce_preprocessor.py            # DCE 전처리 모듈
├── tests/                      # 유닛 테스트
│   ├── test_survey_preprocessor.py    # 설문조사 모듈 테스트
│   └── test_dce_preprocessor.py       # DCE 모듈 테스트
├── examples/                   # 사용 예제
│   ├── example_usage.py               # 설문조사 모듈 예제
│   └── dce_example_usage.py           # DCE 모듈 예제
├── docs/                       # 문서
│   ├── PREPROCESSOR_README.md         # 설문조사 모듈 문서
│   └── DCE_README.md                  # DCE 모듈 문서
├── survey_data/                # 설문조사 전처리 결과
│   ├── demographics_1.csv             # 인구통계학적 변수 (1차)
│   ├── health_concern.csv             # 소비자의 건강관심도
│   ├── perceived_benefit.csv          # substitute의 지각된 유익성
│   ├── purchase_intention.csv         # substitute의 구매의도
│   ├── dce_variables.csv              # DCE 변수
│   ├── perceived_price.csv            # 인지된 가격수준
│   ├── nutrition_knowledge.csv        # 소비자의 영양지식 수준
│   ├── demographics_2.csv             # 인구통계학적 변수 (2차)
│   └── factors_summary.csv            # 요인 요약 정보
└── dce_data/                   # DCE 전처리 결과
    ├── dce_raw_data.csv               # 원본 DCE 데이터 (넘버링 수정됨)
    ├── dce_choice_summary.csv         # 문항별 선택 요약 통계
    ├── dce_choice_matrix.csv          # 선택 매트릭스 (로짓 모델용)
    ├── dce_attribute_data.csv         # 속성별 선택 데이터
    ├── dce_choice_sets_config.csv     # DCE 실험 설계 정보
    └── dce_validation_report.csv      # 데이터 검증 보고서
```

## 🚀 빠른 시작

### 1. 설문조사 데이터 전처리

```python
import sys
sys.path.append('modules')
from survey_data_preprocessor import SurveyDataPreprocessor

# 전처리기 초기화
preprocessor = SurveyDataPreprocessor("../Raw data/Sugar_substitue_Raw data_250730.xlsx")

# 데이터 로딩 및 준비
preprocessor.load_and_prepare_data()

# 특정 요인 데이터 가져오기
health_data = preprocessor.get_factor_data('health_concern')

# 모든 요인 데이터 내보내기
preprocessor.export_factor_data('survey_data')
```

### 2. DCE 데이터 전처리

```python
import sys
sys.path.append('modules')
from dce_preprocessor import DCEProcessor

# DCE 프로세서 초기화
processor = DCEProcessor("../Raw data/Sugar_substitue_Raw data_250730.xlsx")

# 데이터 로딩 및 준비
processor.load_and_prepare_data()

# 선택 요약 통계
summary = processor.get_choice_summary()

# 속성 선호도 분석
analysis = processor.get_attribute_analysis()

# 모든 DCE 데이터 내보내기
processor.export_dce_data('dce_data')
```

## 📊 주요 기능

### 설문조사 전처리 모듈
- **8개 요인별 문항 그룹핑**: 인구통계학적 변수, 건강관심도, 지각된 유익성 등
- **통계 분석용 데이터 변환**: 요인분석, 회귀분석 등에 적합한 형태
- **포괄적 데이터 검증**: 누락 데이터 확인 및 보고
- **다양한 내보내기 형태**: CSV, 요약 통계, 상관관계 분석

### DCE 전처리 모듈
- **6개 선택 상황 분석**: q21-q26 문항의 선택 패턴 분석
- **속성별 선호도 계산**: 무설탕(70.5%), 건강라벨(59.0%) 선호율
- **로짓 모델용 데이터**: Choice Matrix 형태로 변환
- **자동 데이터 정제**: 넘버링 오류 자동 감지 및 수정

## 🧪 테스트 실행

```bash
# 설문조사 모듈 테스트 (24개 테스트)
python tests/test_survey_preprocessor.py

# DCE 모듈 테스트 (29개 테스트)
python tests/test_dce_preprocessor.py
```

## 📖 예제 실행

```bash
# 설문조사 모듈 예제
python examples/example_usage.py

# DCE 모듈 예제
python examples/dce_example_usage.py
```

## 📈 분석 결과 요약

### 설문조사 데이터 (300명)
- **8개 요인**: 57개 문항을 8개 요인으로 그룹핑
- **완전한 데이터**: 누락값 없는 깨끗한 데이터
- **통계 분석 준비**: 기술통계, 상관관계, 요인분석 가능

### DCE 데이터 (300명, 6개 선택 상황)
- **무설탕 선호율**: 70.5% (1,198/1,699 선택)
- **건강라벨 선호율**: 59.0% (1,003/1,699 선택)
- **평균 선택 가격**: 2,376원
- **선택 안함 비율**: 평균 5.6%

## 🔧 기술적 특징

### 설계 원칙
- **재사용성**: 모듈화된 클래스 구조
- **가독성**: 명확한 메서드명과 문서화
- **단일책임**: 각 클래스는 하나의 책임만 담당
- **확장성**: 새로운 요인/분석 추가 용이
- **유지보수성**: 설정과 로직 분리

### 품질 보증
- **53개 유닛 테스트**: 모든 기능에 대한 포괄적 테스트
- **실제 데이터 검증**: raw data만을 사용한 테스트
- **자동 오류 처리**: 데이터 이상 자동 감지 및 수정
- **포괄적 로깅**: 모든 처리 과정 추적 가능

## 📋 요구사항

```bash
pip install pandas openpyxl numpy
```

## 🎯 사용 사례

### 1. 기술통계 분석
```python
# 건강관심도 기술통계
health_data = preprocessor.get_factor_data('health_concern')
stats = health_data.iloc[:, 1:].describe()
```

### 2. 상관관계 분석
```python
# 지각된 유익성 상관관계
benefit_data = preprocessor.get_factor_data('perceived_benefit')
correlation = benefit_data.iloc[:, 1:].corr()
```

### 3. 로짓 모델 분석
```python
# DCE 로짓 모델용 데이터
choice_matrix = processor.get_choice_matrix()
# 종속변수: chosen, 독립변수: sugar_free, has_health_label, price
```

### 4. 시장 점유율 시뮬레이션
```python
# 속성별 선호도 기반 시장 예측
analysis = processor.get_attribute_analysis()
sugar_pref = analysis['sugar_free_preference']['preference_rate']
```

## 📞 지원

- **상세 문서**: `docs/` 폴더의 각 모듈별 README 참조
- **예제 코드**: `examples/` 폴더의 실행 가능한 예제
- **테스트 코드**: `tests/` 폴더의 유닛 테스트 참조

## 📝 라이선스

Sugar Substitute Research Team - 2025

---

**주의사항**: 모든 경로는 `processed_data` 폴더를 기준으로 설정되어 있습니다. 다른 위치에서 실행할 경우 경로를 적절히 수정해주세요.
