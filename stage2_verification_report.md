# 2단계 선택모델 요인점수 사용 검증 보고서

**날짜**: 2025-11-16
**검증 대상**: `examples/sequential_stage2_with_extended_model.py` 실행 결과

---

## ✅ 검증 결과 요약

**모든 검증 항목 통과!** 2단계 선택모델이 1단계 결과를 올바르게 사용했습니다.

---

## 📋 검증 항목

### 1. 데이터 매칭 검증 ✅

| 항목 | 1단계 | 2단계 | 일치 여부 |
|------|-------|-------|-----------|
| 개인 수 | 326 | 326 | ✅ 일치 |
| 전체 행 수 | - | 5,904 | - |
| 개인당 선택 세트 수 | - | 18.1 | - |

**결론**: 1단계 요인점수(326명)와 2단계 데이터(326명)의 개인 수가 정확히 일치합니다.

---

### 2. 요인점수 표준화 검증 ✅

| 잠재변수 | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| health_concern | -0.0000 | 1.0000 | -4.0247 | 1.8954 |
| perceived_benefit | 0.0000 | 1.0000 | -3.1460 | 2.7232 |
| perceived_price | -0.0000 | 1.0000 | -3.1057 | 2.1889 |
| nutrition_knowledge | 0.0000 | 1.0000 | -3.1632 | 2.3560 |
| **purchase_intention** | **-0.0000** | **1.0000** | **-2.6930** | **1.6813** |

**결론**: 모든 요인점수가 평균 0, 표준편차 1로 올바르게 표준화되었습니다.

---

### 3. 요인점수 확장 검증 ✅

**확장 과정**:
- **확장 전**: (326,) - 개인별 1개 값
- **확장 후**: (5,904,) - 선택 세트별 값 (개인당 18.1개 반복)
- **확장 방법**: `respondent_id` 기준 매핑

**확장 후 통계**:
- Mean: 0.0036 (≈0, 정상)
- Std: 0.9980 (≈1, 정상)

**개인별 일관성 확인** (예: ID=1):
- 요인점수: -0.1490
- 선택 세트 수: 18
- 확장된 값: 모두 -0.1490으로 동일 ✅

**결론**: 요인점수가 `respondent_id` 기준으로 올바르게 확장되었으며, 각 개인의 모든 선택 세트에서 동일한 요인점수가 사용되었습니다.

---

### 4. 선택모델 효용함수 검증 ✅

**효용함수 구조** (`choice_equations.py` 173-206번 라인):

```python
# 일반당 대안 (sugar)
V_sugar = ASC_sugar + β₁·sugar_free + β₂·health_label + β₃·price 
          + θ_sugar_PI · PI_factor_score

# 무설탕 대안 (sugar_free)
V_sugar_free = ASC_sugar_free + β₁·sugar_free + β₂·health_label + β₃·price 
               + θ_sugar_free_PI · PI_factor_score

# Opt-out 대안
V_optout = 0 (reference)
```

**추정된 파라미터**:
- `theta_sugar_purchase_intention` = -0.383 (p=0.713, n.s.)
- `theta_sugar_free_purchase_intention` = -0.180 (p=0.814, n.s.)

**결론**: PI 요인점수가 효용함수에 올바르게 포함되었으며, 대안별로 다른 계수(`theta_sugar`, `theta_sugar_free`)가 추정되었습니다.

---

### 5. 코드 흐름 검증 ✅

**1단계 → 2단계 데이터 전달 경로**:

```
1. 1단계 추정 (sequential_stage1_example.py)
   └─> 요인점수 저장: results/sequential_stage_wise/stage1_results.pkl
       └─> factor_scores = {
               'purchase_intention': np.ndarray (326,),
               'health_concern': np.ndarray (326,),
               ...
           }

2. 2단계 추정 (sequential_stage2_with_extended_model.py)
   └─> SequentialEstimator.estimate_stage2_only()
       └─> load_stage1_results()
           └─> factor_scores 로드 (326,)

       └─> MultinomialLogitChoice.fit()
           └─> 요인점수 확장 (326,) → (5904,)
               └─> respondent_id 기준 매핑

           └─> 효용함수 계산
               └─> V[i] += theta * lv_arrays['purchase_intention'][i // n_alternatives]
```

**결론**: 1단계 결과가 올바르게 로드되고, 2단계에서 정확히 사용되었습니다.

---

## 🔍 추가 확인 사항

### 샘플 데이터 확인 (첫 10행)

| respondent_id | choice | sugar_free | health_label | price | PI_factor_score |
|---------------|--------|------------|--------------|-------|-----------------|
| 1 | 1 | 1.0 | 1.0 | 2.5 | -0.1490 |
| 1 | 0 | 0.0 | 0.0 | 2.0 | -0.1490 |
| 1 | 0 | NaN | NaN | NaN | -0.1490 |
| 1 | 0 | 0.0 | 0.0 | 3.0 | -0.1490 |
| 1 | 1 | 1.0 | 1.0 | 2.5 | -0.1490 |
| ... | ... | ... | ... | ... | ... |

**관찰**:
- 동일한 개인(ID=1)의 모든 선택 세트에서 PI 요인점수가 -0.1490으로 동일 ✅
- Opt-out 대안(choice=0, NaN 속성)에도 동일한 요인점수 적용 ✅

---

## 📊 최종 결론

### ✅ 검증 통과 항목 (5/5)

1. ✅ **개인 수 일치**: 1단계(326명) = 2단계(326명)
2. ✅ **요인점수 표준화**: Mean≈0, Std≈1
3. ✅ **요인점수 확장 정상**: (326,) → (5,904,)
4. ✅ **개인별 요인점수 일관성**: 각 개인의 모든 선택 세트에서 동일한 값 사용
5. ✅ **효용함수 포함**: PI 요인점수가 대안별 효용함수에 올바르게 포함됨

### 🎯 종합 평가

**2단계 선택모델이 1단계 측정모델/구조모델의 결과를 완벽하게 사용했습니다.**

- 요인점수 로드: 정상 ✅
- 요인점수 확장: 정상 ✅
- 효용함수 계산: 정상 ✅
- 파라미터 추정: 정상 ✅

---

**검증 스크립트**: `verify_stage2_factor_scores.py`  
**검증 일시**: 2025-11-16 20:59

