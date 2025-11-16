# 순차추정 부트스트래핑 가이드

## 📋 개요

순차추정(Sequential Estimation)에서 3가지 부트스트래핑 모드를 지원합니다:

1. **Stage 1 Only**: 1단계(SEM)만 부트스트래핑
2. **Stage 2 Only**: 2단계(선택모델)만 부트스트래핑 (요인점수 고정)
3. **Both Stages**: 1+2단계 전체 부트스트래핑

---

## 🎯 사용 목적

### 1. Stage 1 Only (SEM 부트스트래핑)
- **목적**: 측정모델과 구조모델 파라미터의 신뢰구간 추정
- **사용 시기**: 
  - SEM 파라미터의 불확실성 평가
  - 경로계수의 유의성 검정
  - 요인적재량의 안정성 확인

### 2. Stage 2 Only (선택모델 부트스트래핑)
- **목적**: 선택모델 파라미터의 신뢰구간 추정 (요인점수 고정)
- **사용 시기**:
  - 1단계 결과를 고정하고 2단계만 재추정
  - 선택모델 파라미터의 불확실성만 평가
  - 계산 시간 절약 (1단계 재추정 불필요)

### 3. Both Stages (전체 부트스트래핑)
- **목적**: 순차추정 전체의 불확실성 전파 평가
- **사용 시기**:
  - 1단계 불확실성이 2단계에 미치는 영향 평가
  - 전체 모델의 신뢰구간 추정
  - 가장 정확한 표준오차 추정 (하지만 계산 시간 많이 소요)

---

## 🚀 사용법

### 기본 사용

```python
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import (
    bootstrap_stage1_only,
    bootstrap_stage2_only,
    bootstrap_both_stages
)

# 1. Stage 1 Only
results_stage1 = bootstrap_stage1_only(
    data=data,
    measurement_model=measurement_config,
    structural_model=structural_config,
    n_bootstrap=100,
    n_workers=4,
    confidence_level=0.95,
    random_seed=42,
    show_progress=True
)

# 2. Stage 2 Only
results_stage2 = bootstrap_stage2_only(
    choice_data=choice_data,
    factor_scores=factor_scores,  # 1단계에서 추출한 요인점수
    choice_model=choice_config,
    n_bootstrap=100,
    n_workers=4,
    confidence_level=0.95,
    random_seed=42,
    show_progress=True
)

# 3. Both Stages
results_both = bootstrap_both_stages(
    data=data,
    measurement_model=measurement_config,
    structural_model=structural_config,
    choice_model=choice_config,
    n_bootstrap=100,
    n_workers=4,
    confidence_level=0.95,
    random_seed=42,
    show_progress=True
)
```

### 클래스 사용

```python
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import SequentialBootstrap

# 부트스트래퍼 생성
bootstrapper = SequentialBootstrap(
    n_bootstrap=100,
    n_workers=4,
    confidence_level=0.95,
    random_seed=42,
    show_progress=True
)

# 1단계만 실행
results_stage1 = bootstrapper.run_stage1_bootstrap(
    data=data,
    measurement_model=measurement_config,
    structural_model=structural_config
)

# 2단계만 실행
results_stage2 = bootstrapper.run_stage2_bootstrap(
    choice_data=choice_data,
    factor_scores=factor_scores,
    choice_model=choice_config
)

# 전체 실행
results_both = bootstrapper.run_both_stages_bootstrap(
    data=data,
    measurement_model=measurement_config,
    structural_model=structural_config,
    choice_model=choice_config
)
```

---

## 📊 결과 구조

모든 부트스트래핑 함수는 다음 구조의 딕셔너리를 반환합니다:

```python
{
    'bootstrap_estimates': List[Dict],  # 각 샘플의 파라미터 추정치
    'confidence_intervals': pd.DataFrame,  # 파라미터별 신뢰구간
    'bootstrap_statistics': pd.DataFrame,  # 평균, 표준편차 등
    'n_successful': int,  # 성공한 샘플 수
    'n_failed': int,  # 실패한 샘플 수
    'mode': str  # 'stage1', 'stage2', 'both'
}
```

### 신뢰구간 DataFrame

| parameter | mean | lower_ci | upper_ci | significant |
|-----------|------|----------|----------|-------------|
| zeta_PI_0 | 1.000 | 1.000 | 1.000 | False |
| gamma_HC_to_PB | 0.305 | 0.198 | 0.412 | True |
| asc_sugar | 1.458 | 0.801 | 2.115 | True |
| beta_price | -0.562 | -1.225 | 0.101 | False |

### 부트스트랩 통계량 DataFrame

| parameter | mean | std | median | min | max |
|-----------|------|-----|--------|-----|-----|
| gamma_HC_to_PB | 0.305 | 0.054 | 0.303 | 0.180 | 0.430 |
| asc_sugar | 1.458 | 0.335 | 1.452 | 0.750 | 2.200 |

---

## 💡 예제 실행

```bash
# 모든 예제 실행
python examples/bootstrap_sequential_example.py --mode all

# 1단계만
python examples/bootstrap_sequential_example.py --mode stage1

# 2단계만
python examples/bootstrap_sequential_example.py --mode stage2

# 전체
python examples/bootstrap_sequential_example.py --mode both
```

---

## ⚙️ 파라미터 설명

| 파라미터 | 설명 | 기본값 | 권장값 |
|---------|------|--------|--------|
| `n_bootstrap` | 부트스트랩 샘플 수 | 100 | 500-1000 |
| `n_workers` | 병렬 작업 수 | CPU-1 | 4-8 |
| `confidence_level` | 신뢰수준 | 0.95 | 0.95 |
| `random_seed` | 랜덤 시드 | 42 | 임의 |
| `show_progress` | 진행 상황 표시 | True | True |

---

## 📈 계산 시간 비교

| 모드 | 샘플당 시간 | 100샘플 예상 시간 |
|------|------------|------------------|
| Stage 1 Only | ~5초 | ~8분 |
| Stage 2 Only | ~2초 | ~3분 |
| Both Stages | ~7초 | ~12분 |

*4 workers 기준, 실제 시간은 데이터 크기와 모델 복잡도에 따라 다름*

---

## 🔍 주의사항

1. **Stage 2 Only 사용 시**:
   - 요인점수가 고정되므로 1단계 불확실성이 반영되지 않음
   - 표준오차가 과소추정될 수 있음
   - 빠른 탐색용으로 적합

2. **Both Stages 사용 시**:
   - 가장 정확한 신뢰구간 추정
   - 계산 시간이 가장 오래 걸림
   - 최종 분석에 권장

3. **병렬 처리**:
   - Windows에서는 `if __name__ == "__main__":` 블록 필수
   - 메모리 사용량 주의 (워커 수 × 데이터 크기)

---

## 📝 참고문헌

- Train, K. E. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press.
- Bhat, C. R., & Dubey, S. K. (2014). A new estimation approach to integrate latent psychological constructs in choice modeling. *Transportation Research Part B*, 67, 68-85.


