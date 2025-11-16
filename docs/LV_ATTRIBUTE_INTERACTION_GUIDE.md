# LV-Attribute 상호작용 가이드

## 개요

선택모델의 효용함수에 **잠재변수(LV)와 선택속성(Attribute) 간 상호작용항**을 추가하는 기능입니다.

### 효용함수

기존 모델:
```
V = intercept + β·X + Σ(λ_i · LV_i)
```

상호작용 추가 모델:
```
V = intercept + β·X + Σ(λ_i · LV_i) + Σ(γ_ij · LV_i · X_j)
```

여기서:
- `β`: 선택속성 계수
- `λ_i`: 잠재변수 i의 주효과 계수
- `γ_ij`: 잠재변수 i와 선택속성 j의 상호작용 계수

---

## 사용법

### 1. ChoiceConfig 설정

```python
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price'],
    choice_type='multinomial',
    all_lvs_as_main=True,
    main_lvs=['purchase_intention', 'nutrition_knowledge'],
    # ✅ LV-Attribute 상호작용 정의
    lv_attribute_interactions=[
        {'lv': 'purchase_intention', 'attribute': 'price'},
        {'lv': 'purchase_intention', 'attribute': 'health_label'},
        {'lv': 'nutrition_knowledge', 'attribute': 'health_label'}
    ]
)
```

### 2. 선택모델 생성

```python
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice

model = MultinomialLogitChoice(config)
```

### 3. 순차추정 예제

```python
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator

# 2단계 추정 실행
stage2_results = estimator.estimate_stage2_only(
    data=data,
    choice_model=model,
    factor_scores='results/stage1_results.pkl'
)
```

---

## 파라미터 해석

### 추정된 파라미터

```python
params = {
    # 속성 계수
    'beta_sugar_free': -0.5,
    'beta_health_label': 0.3,
    'beta_price': -0.001,
    
    # 잠재변수 주효과
    'lambda_purchase_intention': 1.0,
    'lambda_nutrition_knowledge': 0.5,
    
    # ✅ LV-Attribute 상호작용
    'gamma_purchase_intention_price': -0.0002,
    'gamma_purchase_intention_health_label': 0.4,
    'gamma_nutrition_knowledge_health_label': 0.3
}
```

### 해석

1. **`gamma_purchase_intention_price = -0.0002`**
   - 구매의도(PI)가 높을수록 가격에 더 민감해짐 (음수)
   - PI가 1 증가하면 가격 계수가 0.0002 감소

2. **`gamma_purchase_intention_health_label = 0.4`**
   - 구매의도(PI)가 높을수록 건강라벨을 더 선호함 (양수)
   - PI가 1 증가하면 건강라벨 계수가 0.4 증가

3. **`gamma_nutrition_knowledge_health_label = 0.3`**
   - 영양지식(NK)이 높을수록 건강라벨을 더 선호함 (양수)
   - NK가 1 증가하면 건강라벨 계수가 0.3 증가

---

## 효용함수 예시

개인 A (PI=0.5, NK=0.8)가 다음 제품을 평가할 때:
- sugar_free = 1
- health_label = 1
- price = 1000

```
V = 0.0 (intercept)
  + (-0.5 × 1 + 0.3 × 1 + -0.001 × 1000)  # 속성 효과
  + (1.0 × 0.5 + 0.5 × 0.8)                # LV 주효과
  + (-0.0002 × 0.5 × 1000)                 # PI × price
  + (0.4 × 0.5 × 1)                        # PI × health_label
  + (0.3 × 0.8 × 1)                        # NK × health_label
  = 0.04
```

---

## 실행 예제

```bash
# 2단계 추정 실행 (상호작용 포함)
python examples/sequential_stage2_with_extended_model.py
```

출력 예시:
```
[선택모델 파라미터]

  <속성 계수 (beta)>
    beta_intercept: 0.0000
    beta_sugar_free: -0.5000
    beta_health_label: 0.3000
    beta_price: -0.0010

  <잠재변수 주효과 (lambda)>
    lambda_purchase_intention: 1.0000
    lambda_nutrition_knowledge: 0.5000

  <LV-Attribute 상호작용 (gamma)>
    gamma_purchase_intention_price: -0.0002
    gamma_purchase_intention_health_label: 0.4000
    gamma_nutrition_knowledge_health_label: 0.3000
```

---

## 주의사항

1. **상호작용 항목은 `main_lvs`에 포함된 잠재변수만 사용 가능**
2. **속성은 `choice_attributes`에 포함된 것만 사용 가능**
3. **초기값은 자동으로 0.0으로 설정됨** (주효과보다 작게)
4. **opt-out 대안에는 상호작용이 적용되지 않음** (효용 = 0)

---

## 테스트

```bash
# 단위 테스트 실행
python tests/test_lv_attribute_interaction.py
```

---

## 참고

- 기존 LV-LV 조절효과: `moderation_enabled=True`
- 새로운 LV-Attribute 상호작용: `lv_attribute_interactions=[...]`
- 두 기능은 독립적으로 사용 가능

