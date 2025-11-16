# 순차추정 단계별 실행 가이드

## 📋 개요

순차추정을 1단계와 2단계로 분리하여 실행할 수 있습니다.
1단계에서 잠재변수 간 관계를 확인한 후, 2단계에서 선택모델을 추정합니다.

### 장점
- ✅ **1단계 결과 검토 가능**: 잠재변수 간 경로계수 확인 후 진행
- ✅ **세션 분리 가능**: 1단계 실행 → 결과 검토 → 2단계 실행
- ✅ **실험 효율성**: 1단계 고정 → 여러 선택모델 테스트
- ✅ **재현성 보장**: 요인점수 파일 저장 → 동일한 값으로 재실행

---

## 🔄 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│ 1단계: 측정모델 + 구조모델 (SEM)                              │
├─────────────────────────────────────────────────────────────┤
│ - 측정모델 추정 (CFA)                                         │
│ - 구조모델 추정 (경로분석)                                     │
│ - 요인점수 추출 및 표준화                                      │
│ - 결과 저장: stage1_results.pkl                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    결과 파일 검토
                    - 경로계수 확인
                    - 적합도 지수 확인
                    - 요인점수 분포 확인
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2단계: 선택모델                                               │
├─────────────────────────────────────────────────────────────┤
│ - 요인점수 로드 (stage1_results.pkl)                         │
│ - 선택모델 추정 (Multinomial Logit)                          │
│ - 최종 결과 출력                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 사용 방법

### 방법 1: 예제 스크립트 사용 (권장)

#### 1단계 실행
```bash
python examples/sequential_stage1_example.py
```

**출력:**
- `results/stage1_results.pkl`: 1단계 전체 결과
- `logs/stage1_estimation.log`: 추정 로그
- 콘솔: 경로계수, 적합도 지수, 요인점수 통계

#### 2단계 실행
```bash
python examples/sequential_stage2_with_extended_model.py
```

**출력:**
- `results/sequential_stage_wise/{모델명}_parameters.csv`: 파라미터 추정치 (계수, 표준오차, t값, p값)
- `results/sequential_stage_wise/{모델명}_fit.csv`: 적합도 (로그우도, AIC, BIC)
- `logs/stage2_estimation.log`: 추정 로그
- 콘솔: 선택모델 파라미터, AIC, BIC

---

### 방법 2: Python 코드에서 직접 사용

#### 1단계: 측정모델 + 구조모델

```python
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import MultiLatentStructural
from src.analysis.hybrid_choice_model.config import MultiLatentConfig

# 설정 및 모델 생성
config = MultiLatentConfig(...)
measurement_model = MultiLatentMeasurement(config)
structural_model = MultiLatentStructural(config)
estimator = SequentialEstimator(config)

# 1단계 실행
stage1_results = estimator.estimate_stage1_only(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    save_path='results/stage1_results.pkl',  # 결과 저장
    log_file='logs/stage1.log'
)

# 결과 확인
print(stage1_results['paths'])  # 잠재변수 간 경로계수
print(stage1_results['fit_indices'])  # 적합도 지수
```

#### 2단계: 선택모델

```python
from src.analysis.hybrid_choice_model.iclv_models.choice_model import MultinomialLogitChoice

# 선택모델 생성
choice_model = MultinomialLogitChoice(...)

# 2단계 실행 (파일에서 요인점수 로드)
stage2_results = estimator.estimate_stage2_only(
    data=data,
    choice_model=choice_model,
    factor_scores='results/stage1_results.pkl',  # 파일 경로
    log_file='logs/stage2.log'
)

# 또는 메모리에서 직접 전달
stage2_results = estimator.estimate_stage2_only(
    data=data,
    choice_model=choice_model,
    factor_scores=stage1_results['factor_scores'],  # 딕셔너리
    log_file='logs/stage2.log'
)
```

---

## 📊 1단계 결과 구조

`estimate_stage1_only()` 반환값:

```python
{
    'sem_results': Dict,  # SEM 전체 결과
    'factor_scores': {  # 요인점수 (표준화됨)
        'purchase_intention': np.ndarray,
        'perceived_price': np.ndarray,
        ...
    },
    'paths': pd.DataFrame,  # 잠재변수 간 경로계수
    'loadings': pd.DataFrame,  # 요인적재량
    'fit_indices': {  # 적합도 지수
        'CFI': float,
        'TLI': float,
        'RMSEA': float,
        'SRMR': float
    },
    'log_likelihood': float,
    'save_path': str  # 저장 경로 (저장한 경우)
}
```

---

## 💾 저장/로드 메서드

### 전체 결과 저장/로드

```python
# 저장
SequentialEstimator.save_stage1_results(
    results=stage1_results,
    path='results/stage1_results.pkl'
)

# 로드
loaded_results = SequentialEstimator.load_stage1_results(
    path='results/stage1_results.pkl'
)
```

### 요인점수만 저장/로드 (경량)

```python
# 저장
SequentialEstimator.save_factor_scores(
    factor_scores=stage1_results['factor_scores'],
    path='results/factor_scores.pkl'
)

# 로드
factor_scores = SequentialEstimator.load_factor_scores(
    path='results/factor_scores.pkl'
)
```

---

## 🔍 1단계 결과 검토 체크리스트

### 1. 경로계수 (Paths)
- [ ] 모든 경로가 유의한가?
- [ ] 경로 방향이 이론과 일치하는가?
- [ ] 경로 크기가 합리적인가?

### 2. 적합도 지수
- [ ] CFI ≥ 0.90
- [ ] TLI ≥ 0.90
- [ ] RMSEA ≤ 0.08
- [ ] SRMR ≤ 0.08

### 3. 요인점수
- [ ] 평균 ≈ 0, 표준편차 ≈ 1 (표준화 확인)
- [ ] NaN/Inf 없음
- [ ] 분포가 합리적인가?

---

## ⚠️ 주의사항

1. **요인점수 표준화**: 1단계에서 자동으로 Z-score 표준화됨
2. **파일 형식**: `.pkl` (pickle) 형식 사용 권장
3. **데이터 일관성**: 1단계와 2단계에서 동일한 데이터 사용
4. **설정 일관성**: 1단계와 2단계에서 동일한 `config` 사용

---

## 🎯 활용 사례

### 사례 1: 여러 선택모델 비교
```python
# 1단계 1회 실행
stage1_results = estimator.estimate_stage1_only(...)

# 2단계 여러 번 실행 (다른 선택모델)
for choice_model in [model1, model2, model3]:
    results = estimator.estimate_stage2_only(
        data, choice_model, stage1_results['factor_scores']
    )
```

### 사례 2: 세션 분리
```bash
# Day 1: 1단계 실행
python run_stage1.py
# → results/stage1_results.pkl 생성

# Day 2: 결과 검토 후 2단계 실행
python run_stage2.py
```

---

## 📚 참고

- 전체 추정 (1단계 + 2단계 통합): `estimator.estimate()`
- API 문서: `docs/API_REFERENCE.md`
- ICLV 가이드: `docs/ICLV_COMPLETE_SYSTEM_GUIDE.md`

