# 순차추정 부트스트래핑 가이드

## ⚠️ 중요 업데이트 (2025-11-23)

**항상 1+2단계 통합 부트스트래핑을 사용하세요!**

- `bootstrap_both_stages()` 함수만 사용 권장
- `bootstrap_stage1_only()`, `bootstrap_stage2_only()`는 deprecated
- 1단계의 불확실성을 2단계 신뢰구간에 반영하는 것이 이론적으로 올바름

---

## 📋 개요

순차추정(Sequential Estimation)에서 **1+2단계 통합 부트스트래핑**을 수행합니다:

- 각 부트스트랩 샘플마다 1단계(SEM) → 2단계(선택모델)를 순차 실행
- 1단계의 불확실성을 2단계 신뢰구간에 반영
- 이론적으로 올바른 순차추정 표준오차 제공

---

## 🎯 왜 Both Stages만 사용해야 하는가?

### ✅ Both Stages (1+2단계 통합) - 권장

**장점**:
- ✅ 1단계의 불확실성이 2단계 신뢰구간에 반영됨
- ✅ 이론적으로 올바른 표준오차 추정
- ✅ 보수적이고 정확한 신뢰구간
- ✅ 논문 발표에 적합

**단점**:
- ⚠️ 계산 시간이 오래 걸림 (각 샘플마다 1+2단계 모두 추정)

### ❌ Stage 1 Only / Stage 2 Only - Deprecated

**문제점**:
- ❌ 1단계의 불확실성이 2단계에 반영되지 않음
- ❌ 신뢰구간이 과소추정될 위험
- ❌ 이론적으로 부정확한 표준오차
- ❌ 논문 심사에서 지적받을 가능성

---

## 🚀 사용법 (권장)

### ✅ 기본 사용 - Both Stages만 사용

```python
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_both_stages

# ✅ 권장: 1+2단계 통합 부트스트래핑
results = bootstrap_both_stages(
    data=data,
    measurement_model=measurement_config,
    structural_model=structural_config,
    choice_model=choice_config,
    n_bootstrap=1000,  # 권장: 1000 이상
    n_workers=6,       # CPU 코어 수에 맞게 조정
    confidence_level=0.95,
    random_seed=42,
    show_progress=True
)
```

### ⚠️ Deprecated 함수들 (사용 금지)

```python
# ❌ 사용하지 마세요 - Deprecated
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import (
    bootstrap_stage1_only,  # ❌ Deprecated
    bootstrap_stage2_only   # ❌ Deprecated
)

# 이 함수들을 사용하면 DeprecationWarning이 발생합니다.
```

### 클래스 사용 (고급)

```python
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import SequentialBootstrap

# 부트스트래퍼 생성
bootstrapper = SequentialBootstrap(
    n_bootstrap=1000,  # 권장: 1000 이상
    n_workers=6,
    confidence_level=0.95,
    random_seed=42,
    show_progress=True
)

# ✅ 권장: 1+2단계 통합 실행
results = bootstrapper.run_both_stages_bootstrap(
    data=data,
    measurement_model=measurement_config,
    structural_model=structural_config,
    choice_model=choice_config
)

# ❌ Deprecated: 사용하지 마세요
# results_stage1 = bootstrapper.run_stage1_bootstrap(...)  # Deprecated
# results_stage2 = bootstrapper.run_stage2_bootstrap(...)  # Deprecated
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

2. **Both Stages 사용 (필수)**:
   - ✅ 항상 이 방법을 사용하세요
   - ✅ 가장 정확한 신뢰구간 추정
   - ⚠️ 계산 시간이 오래 걸림 (각 샘플마다 1+2단계 모두 추정)
   - 📌 최종 분석 및 논문 발표에 필수

3. **병렬 처리**:
   - Windows에서는 `if __name__ == "__main__":` 블록 필수
   - 메모리 사용량 주의 (워커 수 × 데이터 크기)
   - 권장 워커 수: CPU 코어 수 - 1

---

## 🎯 권장사항 요약

### ✅ DO (해야 할 것)

1. **항상 `bootstrap_both_stages()` 사용**
   - 1단계의 불확실성을 2단계에 반영
   - 이론적으로 올바른 표준오차

2. **충분한 부트스트랩 샘플 수**
   - 최소 1000회 이상 권장
   - 안정적인 신뢰구간 추정

3. **병렬 처리 활용**
   - `n_workers=6` 이상 권장
   - 계산 시간 대폭 단축

### ❌ DON'T (하지 말아야 할 것)

1. **`bootstrap_stage1_only()` 사용 금지**
   - Deprecated
   - 1단계 불확실성이 2단계에 반영 안 됨

2. **`bootstrap_stage2_only()` 사용 금지**
   - Deprecated
   - 신뢰구간 과소추정 위험

3. **적은 샘플 수 사용 금지**
   - 100회 미만은 불안정
   - 최소 1000회 이상 권장

---

## 📝 참고문헌

- Train, K. E. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press.
- Bhat, C. R., & Dubey, S. K. (2014). A new estimation approach to integrate latent psychological constructs in choice modeling. *Transportation Research Part B*, 67, 68-85.
- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.


