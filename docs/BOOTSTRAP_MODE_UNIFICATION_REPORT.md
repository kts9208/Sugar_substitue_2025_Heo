# 부트스트래핑 모드 통합 보고서

**날짜**: 2025-11-23  
**작성자**: ICLV Team  
**목적**: 순차추정 부트스트래핑을 항상 Both 모드로 통합

---

## 📋 변경 요약

### 변경 사항

**이전**: 3가지 부트스트래핑 모드 지원
- `mode='stage1'`: 1단계만 부트스트래핑
- `mode='stage2'`: 2단계만 부트스트래핑 (요인점수 고정)
- `mode='both'`: 1+2단계 통합 부트스트래핑

**이후**: 항상 Both 모드 사용 권장
- ✅ `bootstrap_both_stages()` 함수만 사용 권장
- ⚠️ `bootstrap_stage1_only()`, `bootstrap_stage2_only()`는 deprecated
- 📌 1단계의 불확실성을 2단계 신뢰구간에 반영

---

## 🎯 변경 이유

### 1. 이론적 타당성

**문제점**:
- Stage 1 Only: 1단계 불확실성이 2단계에 반영 안 됨
- Stage 2 Only: 요인점수를 고정하여 신뢰구간 과소추정

**해결책**:
- Both Stages: 각 부트스트랩 샘플마다 1→2단계 순차 실행
- 1단계의 불확실성이 2단계 신뢰구간에 자동 반영
- 이론적으로 올바른 순차추정 표준오차

### 2. 학술적 요구사항

- 논문 심사에서 1단계 불확실성 반영 요구
- 보수적이고 정확한 신뢰구간 필요
- 재현 가능한 연구 결과

---

## 🔧 코드 변경 내역

### 1. `bootstrap_sequential.py` 수정

#### 1.1 모듈 Docstring 업데이트

```python
"""
순차추정 부트스트래핑 모듈

⚠️ 중요: 항상 1+2단계 통합 부트스트래핑을 수행합니다.
- 각 부트스트랩 샘플마다 1단계(SEM) → 2단계(선택모델)를 순차적으로 실행
- 1단계의 불확실성을 2단계 신뢰구간에 반영
- 이론적으로 올바른 순차추정 부트스트래핑
"""
```

#### 1.2 클래스 Docstring 업데이트

```python
class SequentialBootstrap:
    """
    순차추정 부트스트래핑 클래스
    
    ⚠️ 항상 1+2단계 통합 부트스트래핑을 수행합니다.
    
    Note: run_stage1_bootstrap, run_stage2_bootstrap은 deprecated되었습니다.
          항상 run_both_stages_bootstrap을 사용하세요.
    """
```

#### 1.3 Deprecated 메서드에 경고 추가

```python
def run_stage1_bootstrap(self, ...):
    """
    ⚠️ DEPRECATED: 이 메서드는 더 이상 권장되지 않습니다.
    대신 run_both_stages_bootstrap()을 사용하세요.
    """
    import warnings
    warnings.warn(
        "run_stage1_bootstrap()은 deprecated되었습니다. "
        "1단계의 불확실성을 2단계에 반영하려면 run_both_stages_bootstrap()을 사용하세요.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... 기존 코드
```

#### 1.4 편의 함수에 경고 추가

```python
def bootstrap_stage1_only(...):
    """
    ⚠️ DEPRECATED: 이 함수는 더 이상 권장되지 않습니다.
    대신 bootstrap_both_stages()를 사용하세요.
    """
    import warnings
    warnings.warn(...)
    # ... 기존 코드
```

#### 1.5 권장 함수 Docstring 강화

```python
def bootstrap_both_stages(...):
    """
    ✅ 권장: 1+2단계 전체 부트스트래핑 (편의 함수)
    
    이 방법은 1단계의 불확실성을 2단계 신뢰구간에 반영하므로
    이론적으로 올바른 순차추정 표준오차를 제공합니다.
    """
```

### 2. `BOOTSTRAP_SEQUENTIAL_GUIDE.md` 업데이트

#### 2.1 상단에 중요 공지 추가

```markdown
## ⚠️ 중요 업데이트 (2025-11-23)

**항상 1+2단계 통합 부트스트래핑을 사용하세요!**

- `bootstrap_both_stages()` 함수만 사용 권장
- `bootstrap_stage1_only()`, `bootstrap_stage2_only()`는 deprecated
```

#### 2.2 사용법 섹션 재구성

- ✅ 권장 방법 강조
- ❌ Deprecated 함수 명시

#### 2.3 권장사항 섹션 추가

- DO / DON'T 명확히 구분
- 이론적 근거 설명

---

## 📊 영향 분석

### 기존 코드 호환성

✅ **하위 호환성 유지**:
- 기존 `bootstrap_stage1_only()`, `bootstrap_stage2_only()` 함수는 여전히 작동
- DeprecationWarning만 발생
- 기존 코드 즉시 수정 불필요

⚠️ **권장 조치**:
- 새 코드는 `bootstrap_both_stages()` 사용
- 기존 코드는 점진적으로 마이그레이션

### 성능 영향

⚠️ **계산 시간 증가**:
- Both 모드는 각 샘플마다 1+2단계 모두 추정
- Stage 2 Only 대비 약 2~3배 시간 소요

✅ **정확도 향상**:
- 이론적으로 올바른 신뢰구간
- 보수적이고 정확한 표준오차

---

## 🎯 사용자 가이드

### 권장 사용법

```python
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_both_stages

# ✅ 권장: 항상 이 방법 사용
results = bootstrap_both_stages(
    data=data,
    measurement_model=config.measurement_configs,
    structural_model=config.structural,
    choice_model=choice_config,
    n_bootstrap=1000,  # 최소 1000회 이상
    n_workers=6,
    confidence_level=0.95,
    random_seed=42
)
```

### 마이그레이션 가이드

**이전 코드**:
```python
# ❌ Deprecated
results = bootstrap_stage2_only(
    choice_data=data,
    factor_scores=factor_scores,
    choice_model=choice_config,
    n_bootstrap=1000
)
```

**새 코드**:
```python
# ✅ 권장
results = bootstrap_both_stages(
    data=data,  # 전체 데이터 사용
    measurement_model=config.measurement_configs,
    structural_model=config.structural,
    choice_model=choice_config,
    n_bootstrap=1000
)
```

---

## 📝 결론

1. **항상 `bootstrap_both_stages()` 사용**
2. **1단계 불확실성을 2단계에 반영**
3. **이론적으로 올바른 표준오차 추정**
4. **논문 발표에 적합한 신뢰구간**

---

**문의**: ICLV Team

