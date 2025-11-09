# 구조모델 변경 제안: 5개 잠재변수 계층 구조

**날짜**: 2025-11-09  
**목적**: 4개 잠재변수 → 구매의도 → 선택모델 구조 구현

---

## 📊 현재 상황

### **데이터 구조**

5개 잠재변수가 데이터에 존재:

1. **건강관심도** (Health Concern): Q6-Q11 (6개 문항)
2. **건강유익성** (Perceived Benefit): Q12-Q17 (6개 문항)
3. **가격수준** (Perceived Price): Q27-Q29 (3개 문항)
4. **구매의도** (Purchase Intention): Q18-Q20 (3개 문항)
5. **영양지식** (Nutrition Knowledge): Q30-Q49 (20개 문항)

### **현재 구조모델**

```
사회인구학적 변수 → 건강관심도 → 선택모델
(age, gender, income)
```

**문제점**:
- ❌ 5개 잠재변수 중 1개만 사용
- ❌ 잠재변수 간 관계 무시
- ❌ 이론적 구조 반영 안됨

---

## 🎯 목표 구조

### **이론적 배경**

**Theory of Planned Behavior (TPB)** 기반:

```
외생변수 → 태도/인식 → 의도 → 행동
```

### **제안 구조**

```
[계층 0] 외생변수 (사회인구학적)
├─ age_std
├─ gender
├─ income_std
└─ education_level

[계층 1] 1차 잠재변수 (태도/인식)
├─ 건강관심도 = f(age, gender, income, education)
├─ 건강유익성 = f(건강관심도)
├─ 가격수준 = f(income)
└─ 영양지식 = f(age, education)

[계층 2] 2차 잠재변수 (의도)
└─ 구매의도 = f(건강유익성, 가격수준, 영양지식, 건강관심도)

[계층 3] 선택모델 (행동)
└─ 선택 = f(설탕함량, 건강라벨, 가격, 구매의도)
```

---

## 🔧 구현 방안

### **방안 1: 단순 계층 구조 (권장)**

#### **개념**

각 잠재변수를 독립적으로 추정하되, 구매의도만 다른 잠재변수의 영향을 받음

#### **장점**
- ✅ 구현 간단
- ✅ 기존 코드 재사용 가능
- ✅ 추정 안정적
- ✅ 해석 명확

#### **단점**
- ⚠️ 잠재변수 간 상관관계 무시
- ⚠️ 동시 추정 아님 (Sequential)

#### **구현 방법**

**Step 1**: 각 잠재변수 독립 추정

```python
# 1. 건강관심도 (1차 LV)
config_hc = ICLVConfig(
    measurement=MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
    ),
    structural=StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std', 'education_level']
    )
)

# 2. 건강유익성 (1차 LV)
config_pb = ICLVConfig(
    measurement=MeasurementConfig(
        latent_variable='perceived_benefit',
        indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17']
    ),
    structural=StructuralConfig(
        sociodemographics=['health_concern_score']  # 건강관심도 요인점수
    )
)

# 3. 가격수준 (1차 LV)
config_pp = ICLVConfig(
    measurement=MeasurementConfig(
        latent_variable='perceived_price',
        indicators=['q27', 'q28', 'q29']
    ),
    structural=StructuralConfig(
        sociodemographics=['income_std']
    )
)

# 4. 영양지식 (1차 LV)
config_nk = ICLVConfig(
    measurement=MeasurementConfig(
        latent_variable='nutrition_knowledge',
        indicators=[f'q{i}' for i in range(30, 50)]
    ),
    structural=StructuralConfig(
        sociodemographics=['age_std', 'education_level']
    )
)

# 5. 구매의도 (2차 LV) - 최종 잠재변수
config_pi = ICLVConfig(
    measurement=MeasurementConfig(
        latent_variable='purchase_intention',
        indicators=['q18', 'q19', 'q20']
    ),
    structural=StructuralConfig(
        sociodemographics=[
            'perceived_benefit_score',  # 건강유익성 요인점수
            'perceived_price_score',    # 가격수준 요인점수
            'nutrition_knowledge_score', # 영양지식 요인점수
            'health_concern_score'      # 건강관심도 요인점수
        ]
    ),
    choice=ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price']
    )
)
```

**Step 2**: 순차 추정

```python
# 1단계: 건강관심도 추정 → 요인점수 저장
results_hc = estimate_iclv(data, config_hc)
data['health_concern_score'] = results_hc['factor_scores']

# 2단계: 건강유익성 추정 (건강관심도 사용)
results_pb = estimate_iclv(data, config_pb)
data['perceived_benefit_score'] = results_pb['factor_scores']

# 3단계: 가격수준 추정
results_pp = estimate_iclv(data, config_pp)
data['perceived_price_score'] = results_pp['factor_scores']

# 4단계: 영양지식 추정
results_nk = estimate_iclv(data, config_nk)
data['nutrition_knowledge_score'] = results_nk['factor_scores']

# 5단계: 구매의도 + 선택모델 동시 추정
results_final = estimate_iclv(data, config_pi)
```

---

### **방안 2: 완전 동시 추정 (Full Simultaneous)**

#### **개념**

5개 잠재변수 + 선택모델을 모두 동시에 추정

#### **장점**
- ✅ 이론적으로 가장 엄밀
- ✅ 효율적 추정 (최대우도)
- ✅ 표준오차 정확

#### **단점**
- ❌ 구현 매우 복잡
- ❌ 계산 시간 매우 길음
- ❌ 수렴 어려움
- ❌ 디버깅 어려움

#### **구현 난이도**: ⭐⭐⭐⭐⭐ (매우 어려움)

---

### **방안 3: 하이브리드 방식 (권장)**

#### **개념**

1차 잠재변수는 독립 추정, 구매의도만 동시 추정

#### **장점**
- ✅ 구현 난이도 적절
- ✅ 계산 시간 합리적
- ✅ 안정적 수렴
- ✅ 해석 명확

#### **단점**
- ⚠️ 완전 동시 추정은 아님

#### **구현 방법**

```python
# Phase 1: 1차 잠재변수 독립 추정 (측정모델만)
for lv in ['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge']:
    # 측정모델만 추정 (CFA)
    results = estimate_measurement_model(data, lv)
    data[f'{lv}_score'] = results['factor_scores']

# Phase 2: 구매의도 + 선택모델 동시 추정 (ICLV)
config_final = ICLVConfig(
    measurement=MeasurementConfig(
        latent_variable='purchase_intention',
        indicators=['q18', 'q19', 'q20']
    ),
    structural=StructuralConfig(
        sociodemographics=[
            'perceived_benefit_score',
            'perceived_price_score',
            'nutrition_knowledge_score',
            'health_concern_score'
        ]
    ),
    choice=ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price']
    )
)

results_final = estimate_iclv_simultaneous(data, config_final)
```

---

## 📋 구현 단계별 계획

### **방안 3 (하이브리드) 기준**

#### **Phase 1: 1차 잠재변수 추정 (2-3시간)**

1. 측정모델 추정 스크립트 작성
2. 4개 잠재변수 순차 추정
3. 요인점수 계산 및 저장

#### **Phase 2: 데이터 통합 (30분)**

1. 요인점수를 integrated_data.csv에 추가
2. 검증

#### **Phase 3: 구매의도 + 선택모델 설정 (1시간)**

1. 구조모델 설정 수정
2. 선택모델 설정 확인

#### **Phase 4: 최종 추정 (1-2시간)**

1. 구매의도 + 선택모델 동시 추정
2. 결과 검증

**총 예상 시간**: 4-6시간

---

## 🎯 최종 권장 방안

### **방안 3: 하이브리드 방식**

**이유**:
1. ✅ 구현 난이도와 이론적 엄밀성의 균형
2. ✅ 계산 시간 합리적
3. ✅ 안정적 수렴 가능
4. ✅ 기존 코드 최대한 재사용
5. ✅ 단계별 검증 가능

### **구현 우선순위**

**P0 (필수)**:
1. 측정모델 추정 스크립트
2. 요인점수 계산
3. 구조모델 설정 수정

**P1 (중요)**:
1. 최종 ICLV 추정
2. 결과 검증

**P2 (선택)**:
1. 완전 동시 추정 (향후 연구)

---

## 💡 대안: 간소화 방안

### **방안 4: 구매의도만 사용 (가장 간단)**

#### **개념**

구매의도만 잠재변수로 사용, 나머지는 관측변수로 처리

#### **구조**

```
사회인구학적 → 구매의도 → 선택
                    ↑
        (건강관심도, 유익성, 가격, 영양지식 평균값)
```

#### **장점**
- ✅ 매우 간단
- ✅ 빠른 구현 (1-2시간)
- ✅ 안정적

#### **단점**
- ❌ 측정오차 무시
- ❌ 이론적 엄밀성 낮음

---

## 📊 비교표

| 방안 | 구현 난이도 | 계산 시간 | 이론적 엄밀성 | 권장도 |
|------|------------|----------|--------------|--------|
| 방안 1: 단순 계층 | ⭐⭐ | 2-3시간 | ⭐⭐⭐ | ⭐⭐⭐ |
| 방안 2: 완전 동시 | ⭐⭐⭐⭐⭐ | 10+ 시간 | ⭐⭐⭐⭐⭐ | ⭐ |
| 방안 3: 하이브리드 | ⭐⭐⭐ | 4-6시간 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 방안 4: 간소화 | ⭐ | 1-2시간 | ⭐⭐ | ⭐⭐ |

---

## 🤔 의사결정 질문

1. **연구 목적**: 논문 출판? 실무 분석?
2. **시간 제약**: 얼마나 시간을 투자할 수 있나?
3. **이론적 엄밀성**: 얼마나 중요한가?
4. **계산 자원**: 긴 추정 시간 감당 가능한가?

---

## 📌 다음 단계

어떤 방안을 선택하시겠습니까?

- **방안 1**: 단순하지만 안정적
- **방안 3**: 균형잡힌 접근 (권장)
- **방안 4**: 빠른 프로토타입

선택하시면 구체적인 구현 계획을 제시하겠습니다.

