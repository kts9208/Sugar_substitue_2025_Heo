# 2단계 선택모델 추정 사용 예시

## 📋 개요

`sequential_stage2_with_extended_model.py`는 **설정 변수만 수정**하면 다양한 선택모델을 자동으로 추정할 수 있습니다.

---

## 🎯 핵심 개념

### 3가지 설정 변수

1. **`MAIN_LVS`**: 잠재변수 주효과
   - 효용함수에 직접 영향: V = β·X + **λ·LV**

2. **`MODERATION_LVS`**: 조절효과
   - 잠재변수 간 상호작용: LV₁ × LV₂

3. **`LV_ATTRIBUTE_INTERACTIONS`**: LV-속성 상호작용
   - 잠재변수와 속성의 상호작용: LV × Attribute

---

## 📝 실전 예시

### 예시 1: Base Model → Base + PI 주효과

**변경 전:**
```python
MAIN_LVS = []
```

**변경 후:**
```python
MAIN_LVS = ['purchase_intention']
```

**효과:**
- 모델: `Base Model` → `Base Model + PI 주효과`
- 파일명: `stage2_base_model_*.csv` → `stage2_PI_*.csv`
- 효용함수: V = β₁·health_label + β₂·price → V = β₁·health_label + β₂·price + **λ·PI**

---

### 예시 2: PI 주효과 → PI + NK 주효과

**변경 전:**
```python
MAIN_LVS = ['purchase_intention']
```

**변경 후:**
```python
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
```

**효과:**
- 모델: `Base Model + PI 주효과` → `Base Model + PI + NK 주효과`
- 파일명: `stage2_PI_*.csv` → `stage2_PI_NK_*.csv`
- 효용함수: V = ... + λ₁·PI → V = ... + λ₁·PI + **λ₂·NK**

---

### 예시 3: 주효과 → 주효과 + 상호작용

**변경 전:**
```python
MAIN_LVS = ['purchase_intention']
LV_ATTRIBUTE_INTERACTIONS = []
```

**변경 후:**
```python
MAIN_LVS = ['purchase_intention']
LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'price')]
```

**효과:**
- 모델: `Base Model + PI 주효과` → `Base Model + PI 주효과 + LV-Attr 상호작용 1개`
- 파일명: `stage2_PI_*.csv` → `stage2_PI_1int_*.csv`
- 효용함수: V = ... + λ·PI → V = ... + λ·PI + **γ·(PI × price)**

---

### 예시 4: 복합 모델 (모든 효과 포함)

```python
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
MODERATION_LVS = [('perceived_price', 'nutrition_knowledge')]
LV_ATTRIBUTE_INTERACTIONS = [
    ('purchase_intention', 'price'),
    ('nutrition_knowledge', 'health_label')
]
```

**효과:**
- 모델: `Base Model + PI + NK 주효과 + 조절효과 1개 + LV-Attr 상호작용 2개`
- 파일명: `stage2_PI_NK_2int_*.csv`
- 효용함수:
  ```
  V = β₁·health_label + β₂·price
    + λ₁·PI + λ₂·NK                    (주효과)
    + δ·(PP × NK)                       (조절효과)
    + γ₁·(PI × price)                   (상호작용 1)
    + γ₂·(NK × health_label)            (상호작용 2)
  ```

---

## 🔄 모델 비교 워크플로우

### 단계별 모델 확장

```python
# Step 1: Base Model
MAIN_LVS = []
# → 실행 → 결과 확인

# Step 2: Base + PI
MAIN_LVS = ['purchase_intention']
# → 실행 → 결과 비교 (AIC, BIC)

# Step 3: Base + PI + NK
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
# → 실행 → 결과 비교

# Step 4: Base + PI + NK + 상호작용
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'price')]
# → 실행 → 최종 모델 선택
```

---

## 📊 자동 생성되는 출력

### 콘솔 출력 예시

```
✅ 선택모델 설정:
   - 모델 유형: Base Model + PI + NK 주효과 + LV-Attr 상호작용 1개
   - 선택 속성: health_label, price
   - 잠재변수 주효과: 구매의도(PI), 영양지식(NK)
   - 조절효과: 없음
   - LV-Attribute 상호작용: 1개
      * purchase_intention × price
```

### 파일명 자동 생성

| 설정 | 파일명 |
|------|--------|
| Base Model | `stage2_base_model_*.csv` |
| Base + PI | `stage2_PI_*.csv` |
| Base + PI + NK | `stage2_PI_NK_*.csv` |
| Base + PI + 상호작용 1개 | `stage2_PI_1int_*.csv` |
| Base + PI + NK + 상호작용 2개 | `stage2_PI_NK_2int_*.csv` |

---

## ✅ 체크리스트

실행 전 확인사항:

- [ ] 1단계 결과 파일이 존재하는가? (`STAGE1_RESULT_FILE`)
- [ ] 잠재변수 이름이 정확한가? (오타 확인)
- [ ] 속성 이름이 정확한가? (오타 확인)
- [ ] 상호작용 설정이 의도한 대로인가?

---

**Author**: ICLV Team  
**Date**: 2025-01-16

