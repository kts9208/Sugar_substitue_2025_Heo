# 통합 상관관계 분석 모듈 구현 요약

## 📋 개요

1단계 SEM 변수와 2단계 선택모델 변수를 모두 포함하는 통합 상관관계 분석 모듈을 구현했습니다.

**구현 날짜**: 2025-11-16

---

## 🎯 구현 목표

기존에는 1단계 SEM 추정 시 잠재변수 간 상관관계만 확인할 수 있었습니다. 이를 확장하여:

1. **1단계 SEM 변수**: 잠재변수 지표, 잠재변수 간 상관관계
2. **2단계 선택모델 변수**: 속성변수, 사회인구통계변수
3. **1단계-2단계 연결**: 잠재변수와 선택모델 변수 간 상관관계

모든 변수를 포괄하는 통합 상관관계 분석을 제공합니다.

---

## 📁 구현 파일

### 1. 핵심 모듈

**`src/analysis/hybrid_choice_model/iclv_models/correlation_analyzer.py`** (신규)
- `IntegratedCorrelationAnalyzer` 클래스
- 7가지 상관관계 분석 기능
- 결과 저장 및 요약 출력

### 2. 통합 기능

**`src/analysis/hybrid_choice_model/iclv_models/sem_estimator.py`** (수정)
- `analyze_correlations()` 메서드 추가
- SEMEstimator에서 직접 상관관계 분석 실행 가능

### 3. 예제 및 문서

- **`examples/correlation_analysis_example.py`**: 실제 데이터 사용 예제
- **`scripts/test_correlation_analysis.py`**: 샘플 데이터 테스트 스크립트
- **`docs/CORRELATION_ANALYSIS_GUIDE.md`**: 사용 가이드

---

## 🔧 주요 기능

### 1. 잠재변수 지표 간 상관관계
- 각 잠재변수별 관측지표 간 상관관계 분석
- 측정모델의 타당성 확인

### 2. 잠재변수 간 상관관계
- semopy CFA 모델 기반 상관계수 추출
- 표준화된 추정값 사용

### 3. 선택모델 속성변수 간 상관관계
- 선택 속성 간 상관관계 분석
- 다중공선성 확인

### 4. 사회인구통계변수 간 상관관계
- 사회인구학적 변수 간 상관관계 분석

### 5. 잠재변수-속성변수 간 상관관계
- 잠재변수와 선택 속성 간 관계 파악

### 6. 잠재변수-사회인구통계변수 간 상관관계
- 잠재변수와 사회인구학적 변수 간 관계 파악

### 7. 전체 상관관계 행렬
- 모든 변수를 포함하는 통합 상관관계 행렬

---

## 💻 사용 방법

### 방법 1: SEMEstimator 사용 (권장)

```python
from src.analysis.hybrid_choice_model.iclv_models.sem_estimator import SEMEstimator
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

# SEMEstimator 생성
sem_estimator = SEMEstimator()

# 선택모델 설정
choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price'],
    choice_type='binary',
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

### 방법 2: 독립 실행

```python
from src.analysis.hybrid_choice_model.iclv_models.correlation_analyzer import IntegratedCorrelationAnalyzer

analyzer = IntegratedCorrelationAnalyzer()

results = analyzer.analyze_all_correlations(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_config=choice_config,
    save_path='results/correlations'
)

analyzer.print_summary()
```

---

## 📊 결과 구조

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

1. `indicator_corr_{lv_name}_{timestamp}.csv` - 각 잠재변수별 지표 간 상관관계
2. `latent_correlations_{timestamp}.csv` - 잠재변수 간 상관관계
3. `attribute_correlations_{timestamp}.csv` - 속성변수 간 상관관계
4. `sociodem_correlations_{timestamp}.csv` - 사회인구통계변수 간 상관관계
5. `lv_attribute_correlations_{timestamp}.csv` - 잠재변수-속성변수 간 상관관계
6. `lv_sociodem_correlations_{timestamp}.csv` - 잠재변수-사회인구통계변수 간 상관관계
7. `full_correlation_matrix_{timestamp}.csv` - 전체 상관관계 행렬
8. `correlation_summary_{timestamp}.json` - 요약 통계 (JSON)

---

## 🧪 테스트

### 샘플 데이터 테스트

```bash
python scripts/test_correlation_analysis.py
```

### 실제 데이터 예제

```bash
python examples/correlation_analysis_example.py
```

---

## 📈 활용 방안

### 1. 모델 진단
- 다중공선성 확인
- 측정모델 타당성 검증
- 구조모델 경로 설정 근거

### 2. 변수 선택
- 높은 상관관계 변수 제거
- 낮은 상관관계 변수 추가 고려

### 3. 모델 개선
- 유의한 상관관계 기반 경로 추가
- 조절효과 탐색

---

## ⚠️ 주의사항

1. **semopy 필요**: 잠재변수 간 상관관계 분석에 필요
2. **데이터 요구사항**: 통합 데이터 (SEM 지표 + 선택모델 변수 포함)
3. **계산 시간**: 대규모 데이터의 경우 몇 분 소요 가능

---

## 🔄 기존 코드와의 통합

기존 `SEMEstimator`의 `fit_cfa_only()` 메서드는 그대로 유지되며, 새로운 `analyze_correlations()` 메서드가 추가되었습니다.

기존 코드에 영향을 주지 않으면서 새로운 기능을 제공합니다.

---

## 📚 참고 문서

- `docs/CORRELATION_ANALYSIS_GUIDE.md`: 상세 사용 가이드
- `examples/correlation_analysis_example.py`: 실제 사용 예제
- `scripts/test_correlation_analysis.py`: 테스트 스크립트

