# 2단계 선택모델 추정 빠른 가이드

## 🎯 사용법

`sequential_stage2_with_extended_model.py` 파일의 **main() 함수 상단 설정만 수정**하면 됩니다!

---

## 📝 설정 변수

```python
# 📌 잠재변수 주효과 (원하는 잠재변수만 추가)
MAIN_LVS = []  # ✅ 여기만 수정!

# 📌 조절효과 (잠재변수 2개 세트)
MODERATION_LVS = []  # ✅ 여기만 수정!

# 📌 LV-Attribute 상호작용 (잠재변수-속성 2개 세트)
LV_ATTRIBUTE_INTERACTIONS = []  # ✅ 여기만 수정!
```

---

## 💡 예시

### 1️⃣ Base Model (잠재변수 없음)
```python
MAIN_LVS = []
MODERATION_LVS = []
LV_ATTRIBUTE_INTERACTIONS = []
```
**출력:** `2단계 추정: 선택모델 (Base Model)`

---

### 2️⃣ Base + PI 주효과
```python
MAIN_LVS = ['purchase_intention']
MODERATION_LVS = []
LV_ATTRIBUTE_INTERACTIONS = []
```
**출력:** `2단계 추정: 선택모델 (Base Model + PI 주효과)`

---

### 3️⃣ Base + PI + NK 주효과
```python
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
MODERATION_LVS = []
LV_ATTRIBUTE_INTERACTIONS = []
```
**출력:** `2단계 추정: 선택모델 (Base Model + PI + NK 주효과)`

---

### 4️⃣ Base + PI 주효과 + PI×price 상호작용
```python
MAIN_LVS = ['purchase_intention']
MODERATION_LVS = []
LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'price')]
```
**출력:** `2단계 추정: 선택모델 (Base Model + PI 주효과 + LV-Attr 상호작용 1개)`

---

### 5️⃣ 복합 모델 (주효과 + 조절효과 + 상호작용)
```python
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
MODERATION_LVS = [('perceived_price', 'nutrition_knowledge')]
LV_ATTRIBUTE_INTERACTIONS = [
    ('purchase_intention', 'price'),
    ('nutrition_knowledge', 'health_label')
]
```
**출력:** `2단계 추정: 선택모델 (Base Model + PI + NK 주효과 + 조절효과 1개 + LV-Attr 상호작용 2개)`

---

## 📚 사용 가능한 변수

### 잠재변수 (Latent Variables)
- `'purchase_intention'` (PI): 구매의도
- `'nutrition_knowledge'` (NK): 영양지식
- `'perceived_benefit'` (PB): 건강유익성
- `'perceived_price'` (PP): 가격수준
- `'health_concern'` (HC): 건강관심도

### 선택 속성 (Choice Attributes)
- `'health_label'`: 건강 라벨
- `'price'`: 가격

---

## 🚀 실행

```bash
python examples/sequential_stage2_with_extended_model.py
```

---

## 📁 결과 파일

자동으로 생성되는 파일명:
- `results/sequential_stage_wise/{모델명}_parameters.csv`
- `results/sequential_stage_wise/{모델명}_fit.csv`

파일명 예시:
- `stage2_base_model_parameters.csv` (Base Model)
- `stage2_PI_parameters.csv` (Base + PI)
- `stage2_PI_NK_parameters.csv` (Base + PI + NK)
- `stage2_PI_NK_1int_parameters.csv` (Base + PI + NK + 상호작용 1개)

---

## ✅ 장점

1. **간단한 설정**: 리스트만 수정하면 끝!
2. **자동 메시지**: 모델 유형이 자동으로 출력됨
3. **자동 파일명**: 모델 설정에 맞는 파일명 자동 생성
4. **오류 방지**: 설정 불일치 걱정 없음

---

**Author**: ICLV Team  
**Date**: 2025-01-16

