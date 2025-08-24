# DCE (Discrete Choice Experiment) Preprocessor

DCE 변수를 위한 전용 전처리 및 분석 모듈입니다. 기존 설문조사 전처리 모듈과 독립적으로 구현되었습니다.

## 주요 기능

- DCE 실험 설계 정보 관리
- 선택 데이터 로딩 및 검증
- 선택 패턴 분석
- 속성별 선호도 분석
- 다양한 형태의 데이터 내보내기
- 포괄적인 유닛 테스트 포함

## DCE 실험 설계

본 연구의 DCE는 6개 선택 상황(q21-q26)으로 구성되며, 각 상황에서 응답자는 다음 중 선택합니다:
- 제품 A (값: 1)
- 제품 B (값: 2)  
- 모두 선택 안함 (값: 3)

### 제품 속성

각 제품은 3가지 속성으로 구성됩니다:
- **당 종류**: 일반당 vs 무설탕
- **건강라벨**: 있음 vs 없음
- **가격**: 2000원, 2500원, 3000원

### 선택 상황 구성

| 문항 | 제품 A | 제품 B |
|------|--------|--------|
| q21 | 무설탕, 건강라벨 있음, 2500원 | 일반당, 건강라벨 없음, 2000원 |
| q22 | 일반당, 건강라벨 없음, 3000원 | 무설탕, 건강라벨 있음, 2500원 |
| q23 | 무설탕, 건강라벨 없음, 2000원 | 무설탕, 건강라벨 있음, 3000원 |
| q24 | 일반당, 건강라벨 있음, 2000원 | 일반당, 건강라벨 없음, 2500원 |
| q25 | 무설탕, 건강라벨 없음, 2500원 | 일반당, 건강라벨 있음, 3000원 |
| q26 | 일반당, 건강라벨 있음, 3000원 | 무설탕, 건강라벨 없음, 2000원 |

## 설치 및 요구사항

```bash
pip install pandas openpyxl numpy
```

## 기본 사용법

### 1. 간단한 사용 예제

```python
from dce_preprocessor import DCEProcessor

# DCE 프로세서 초기화
processor = DCEProcessor("Raw data/Sugar_substitue_Raw data_250730.xlsx")

# 데이터 로딩 및 준비
processor.load_and_prepare_data()

# 선택 요약 통계
summary = processor.get_choice_summary()
print(summary[['question_id', 'product_a_share', 'product_b_share', 'neither_share']])

# 속성 선호도 분석
analysis = processor.get_attribute_analysis()
print(f"무설탕 선호율: {analysis['sugar_free_preference']['preference_rate']:.3f}")

# 모든 데이터 내보내기
processor.export_dce_data('dce_processed_data')
```

### 2. 실험 설계 정보 확인

```python
from dce_preprocessor import DCEConfig

# 모든 선택 상황 정보
choice_sets = DCEConfig.get_choice_sets_summary()

# 특정 문항 정보
q21_set = DCEConfig.get_choice_set('q21')
print(f"q21: {q21_set}")
```

### 3. 고급 분석

```python
# 선택 매트릭스 (로짓 모델용)
choice_matrix = processor.get_choice_matrix()

# 개별 응답자 패턴 분석
analyzer = processor.analyzer
analyzer.prepare_analysis_data()
patterns = analyzer.get_choice_patterns()
```

## 클래스 구조

### DCEProcessor (메인 클래스)
- `load_and_prepare_data()`: 데이터 로딩 및 준비
- `get_choice_summary()`: 선택 요약 통계
- `get_choice_matrix()`: 선택 매트릭스 반환
- `get_attribute_analysis()`: 속성 선호도 분석
- `export_dce_data(output_dir)`: 모든 데이터 내보내기

### DCEConfig
- DCE 실험 설계 정보 관리
- 제품 속성 정의
- 선택 상황 구성 정보

### DCEDataLoader
- Excel 파일에서 DCE 데이터 로딩
- 데이터 검증 및 넘버링 오류 수정

### DCEPreprocessor
- 선택 데이터를 분석 가능한 형태로 변환
- 선택 매트릭스 생성
- 속성 효과 데이터 생성

### DCEAnalyzer
- 선택 패턴 분석
- 속성 선호도 계산
- 개별 응답자 패턴 분석

## 출력 파일

`export_dce_data()` 실행 시 생성되는 파일들:

```
dce_processed_data/
├── dce_raw_data.csv              # 원본 DCE 데이터 (넘버링 수정됨)
├── dce_choice_summary.csv        # 문항별 선택 요약 통계
├── dce_choice_matrix.csv         # 선택 매트릭스 (로짓 모델용)
├── dce_attribute_data.csv        # 속성별 선택 데이터
├── dce_choice_sets_config.csv    # DCE 실험 설계 정보
└── dce_validation_report.csv     # 데이터 검증 보고서
```

## 분석 결과 예시

### 선택 요약 (Choice Summary)
- 각 문항별 제품 A, B, 선택안함의 비율
- 제품 설명과 함께 제공

### 속성 선호도 분석
- **무설탕 선호율**: 70.5% (실제 선택에서 무설탕 제품을 선택한 비율)
- **건강라벨 선호율**: 59.0% (건강라벨이 있는 제품을 선택한 비율)
- **평균 선택 가격**: 선택된 제품들의 평균 가격

## 테스트 실행

```bash
python test_dce_preprocessor.py
```

모든 테스트는 실제 raw data를 사용하여 수행됩니다.

## 예제 실행

```bash
python dce_example_usage.py
```

## 특별 기능

### 1. 넘버링 오류 자동 수정
- 중복된 응답자 ID 자동 감지
- 새로운 ID로 자동 재할당
- 데이터 손실 없이 처리

### 2. 다양한 분석 형태 지원
- **Choice Matrix**: 로짓/프로빗 모델용 형태
- **Attribute Data**: 속성별 효과 분석용
- **Summary Statistics**: 기술통계 및 시각화용

### 3. 유연한 데이터 구조
- Enum을 활용한 타입 안전성
- Dataclass를 활용한 구조화된 데이터
- 확장 가능한 설정 관리

## 통계 분석 연계

생성된 데이터는 다음과 같은 통계 분석에 활용 가능합니다:

### 1. 로짓/프로빗 모델
```python
# choice_matrix.csv 사용
# 종속변수: chosen
# 독립변수: sugar_free, has_health_label, price
```

### 2. 혼합 로짓 모델
```python
# 개별 응답자 이질성 분석
# respondent_id를 그룹 변수로 활용
```

### 3. 시장 점유율 시뮬레이션
```python
# 속성 조합별 선택 확률 예측
# 새로운 제품의 시장 점유율 추정
```

## 주의사항

1. DCE 데이터는 q21-q26 문항만 포함됩니다.
2. 선택값은 1(제품A), 2(제품B), 3(선택안함)이어야 합니다.
3. 넘버링 오류가 발견되면 자동으로 수정됩니다.
4. 'no' 컬럼이 응답자 식별용으로 사용됩니다.

## 라이선스

Sugar Substitute Research Team - 2025
