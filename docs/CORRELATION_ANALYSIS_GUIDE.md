# 통합 상관관계 분석 가이드

## 개요

통합 상관관계 분석 모듈은 1단계 SEM 변수와 2단계 선택모델 변수를 모두 포함하는 포괄적인 상관관계 분석을 제공합니다.

### 주요 기능

1. **잠재변수 지표 간 상관관계** (1단계 SEM)
   - 각 잠재변수별 관측지표 간 상관관계 분석
   - 측정모델의 타당성 확인

2. **잠재변수 간 상관관계** (1단계 SEM)
   - semopy를 사용한 CFA 모델 기반 상관계수 추출
   - 잠재변수 간 관계 파악

3. **선택모델 속성변수 간 상관관계** (2단계)
   - 선택 속성 간 상관관계 분석
   - 다중공선성 확인

4. **사회인구통계변수 간 상관관계** (2단계)
   - 사회인구학적 변수 간 상관관계 분석

5. **잠재변수-속성변수 간 상관관계** (1단계-2단계 연결)
   - 잠재변수와 선택 속성 간 관계 파악

6. **잠재변수-사회인구통계변수 간 상관관계** (1단계-2단계 연결)
   - 잠재변수와 사회인구학적 변수 간 관계 파악

7. **전체 상관관계 행렬**
   - 모든 변수를 포함하는 통합 상관관계 행렬

---

## 사용 방법

### 1. 기본 사용법

```python
from src.analysis.hybrid_choice_model.iclv_models.sem_estimator import SEMEstimator
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

# SEMEstimator 생성
sem_estimator = SEMEstimator()

# 선택모델 설정
choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price'],
    choice_type='binary',
    price_variable='price',
    all_lvs_as_main=True,
    main_lvs=['health_concern', 'perceived_benefit', 'perceived_price',
              'nutrition_knowledge', 'purchase_intention']
)

# 통합 상관관계 분석 실행
results = sem_estimator.analyze_correlations(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_config=choice_config,
    save_path='results/correlations'
)
```

### 2. 독립 실행

```python
from src.analysis.hybrid_choice_model.iclv_models.correlation_analyzer import IntegratedCorrelationAnalyzer

# 분석기 생성
analyzer = IntegratedCorrelationAnalyzer()

# 분석 실행
results = analyzer.analyze_all_correlations(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_config=choice_config,
    factor_scores=factor_scores,  # 선택사항
    save_path='results/correlations'
)

# 요약 출력
analyzer.print_summary()
```

---

## 결과 구조

### 반환값

```python
{
    'indicator_correlations': Dict[str, pd.DataFrame],  # 지표 간 상관관계
    'latent_correlations': pd.DataFrame,  # 잠재변수 간 상관관계
    'attribute_correlations': pd.DataFrame,  # 속성변수 간 상관관계
    'sociodem_correlations': pd.DataFrame,  # 사회인구통계변수 간 상관관계
    'lv_attribute_correlations': pd.DataFrame,  # 잠재변수-속성변수 간
    'lv_sociodem_correlations': pd.DataFrame,  # 잠재변수-사회인구통계변수 간
    'full_correlation_matrix': pd.DataFrame,  # 전체 상관관계 행렬
    'summary': Dict  # 요약 통계
}
```

### 저장 파일

결과는 다음 파일로 저장됩니다:

1. `indicator_corr_{lv_name}_{timestamp}.csv` - 각 잠재변수별 지표 간 상관관계
2. `latent_correlations_{timestamp}.csv` - 잠재변수 간 상관관계
3. `attribute_correlations_{timestamp}.csv` - 속성변수 간 상관관계
4. `sociodem_correlations_{timestamp}.csv` - 사회인구통계변수 간 상관관계
5. `lv_attribute_correlations_{timestamp}.csv` - 잠재변수-속성변수 간 상관관계
6. `lv_sociodem_correlations_{timestamp}.csv` - 잠재변수-사회인구통계변수 간 상관관계
7. `full_correlation_matrix_{timestamp}.csv` - 전체 상관관계 행렬
8. `correlation_summary_{timestamp}.json` - 요약 통계 (JSON)

---

## 예제 실행

```bash
# 예제 스크립트 실행
python examples/correlation_analysis_example.py
```

---

## 활용 방안

### 1. 모델 진단

- **다중공선성 확인**: 속성변수 간 높은 상관관계 확인
- **측정모델 타당성**: 지표 간 상관관계가 적절한지 확인
- **구조모델 설정**: 잠재변수 간 상관관계를 바탕으로 경로 설정

### 2. 변수 선택

- **변수 제거**: 높은 상관관계를 보이는 변수 중 하나 제거
- **변수 추가**: 낮은 상관관계를 보이는 변수 추가 고려

### 3. 모델 개선

- **경로 추가**: 유의한 상관관계를 보이는 잠재변수 간 경로 추가
- **조절효과 탐색**: 잠재변수-속성변수 간 상관관계를 바탕으로 조절효과 탐색

---

## 주의사항

1. **데이터 요구사항**
   - 통합 데이터 (SEM 지표 + 선택모델 변수 포함)
   - 개인별 unique 데이터 자동 추출

2. **semopy 필요**
   - 잠재변수 간 상관관계 분석에 semopy 라이브러리 필요
   - 설치: `pip install semopy`

3. **계산 시간**
   - 대규모 데이터의 경우 몇 분 소요될 수 있음

---

## 문의

문제가 발생하거나 추가 기능이 필요한 경우 개발팀에 문의하세요.

