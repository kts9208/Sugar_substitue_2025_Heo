# ICLV 잠재변수 검토 보고서

**작성일**: 2025-11-05  
**프로젝트**: Sugar Substitute 2025 (대체당 연구)  
**목적**: 데이터에 존재하는 모든 잠재변수 확인 및 ICLV 모델 설정 수정

---

## 📊 핵심 발견

### ✅ **5개 잠재변수 모두 존재**

이전 보고서에서는 **건강관심도 1개**만 확인했으나, 실제로는 **5개의 잠재변수**가 모두 데이터에 존재합니다.

| 잠재변수 | 설문 문항 | 지표 수 | 척도 | 데이터 파일 | 상태 |
|---------|----------|--------|------|------------|------|
| **건강관심도** | Q6-Q11 | 6개 | 5점 | `health_concern.csv` | ✅ 존재 |
| **건강유익성** | Q12-Q17 | 6개 | 5점 | `perceived_benefit.csv` | ✅ 존재 |
| **가격수준** | Q27-Q29 | 3개 | 5점 | `perceived_price.csv` | ✅ 존재 |
| **구매의도** | Q18-Q20 | 3개 | 5점 | `purchase_intention.csv` | ✅ 존재 |
| **영양지식** | Q30-Q49 | 20개 | 5점 | `nutrition_knowledge.csv` | ✅ 존재 |

---

## 📋 잠재변수 상세 정보

### **1. 건강관심도 (Health Concern)**

**설문 문항**: Q6-Q11  
**지표 수**: 6개  
**척도**: 5점 Likert (1=전혀 그렇지 않다 ~ 5=매우 그렇다)  
**데이터**: `data/processed/survey/health_concern.csv`

**문항 내용**:
- Q6: 저는 제 건강에 대해 자주 되돌아봅니다.
- Q7: 저는 제 건강에 대해 매우 신경을 씁니다.
- Q8: 저는 건강 상태의 변화를 잘 알아차립니다.
- Q9: 저는 보통 제 건강에 대해 의식하고 있습니다.
- Q10: 저는 제 건강 상태에 대한 책임감을 느낍니다.
- Q11: 저는 하루를 보내는 동안 제 건강 상태를 인지하고 있습니다.

---

### **2. 건강유익성 (Perceived Benefit)**

**설문 문항**: Q12-Q17  
**지표 수**: 6개  
**척도**: 5점 Likert (일부 역코딩)  
**데이터**: `data/processed/survey/perceived_benefit.csv`

**문항 내용**:
- Q12: 대체당은 소비자에게 많은 이점을 가져다줍니다.
- Q13: 식품에서 대체당 사용으로 얻는 이점은 과대평가되어 있습니다. (역코딩)
- Q14: 대체당 사용은 개인적으로 저에게 혜택을 줍니다.
- Q15: 대체당은 불필요한 칼로리를 줄이도록 해줍니다.
- Q16: 만약 대체당이 없다면, 많은 다이어트 제품들이 생산될 수 없었을 것입니다.
- Q17: 대체당은 후회 없이 마음껏 즐길 수 있게 해줍니다.

---

### **3. 가격수준 (Perceived Price)**

**설문 문항**: Q27-Q29  
**지표 수**: 3개  
**척도**: 5점 Likert (일부 역코딩)  
**데이터**: `data/processed/survey/perceived_price.csv`

**문항 내용**:
- Q27: 대체당 식품의 가격은 높습니다.
- Q28: 대체당 식품의 가격은 낮습니다. (역코딩)
- Q29: 대체당 식품은 비쌉니다.

---

### **4. 구매의도 (Purchase Intention)**

**설문 문항**: Q18-Q20  
**지표 수**: 3개  
**척도**: 5점 Likert  
**데이터**: `data/processed/survey/purchase_intention.csv`

**문항 내용**:
- Q18: 앞으로 저는 대체당 제품을 구매할 의도가 있습니다.
- Q19: 다음번 쇼핑할 때 저는 대체당 제품을 구매할 것입니다.
- Q20: 저는 앞으로 대체당 제품을 구매할 계획이 있습니다.

---

### **5. 영양지식 수준 (Nutrition Knowledge)**

**설문 문항**: Q30-Q49  
**지표 수**: 20개  
**척도**: 5점 Likert (일부 역코딩)  
**데이터**: `data/processed/survey/nutrition_knowledge.csv`

**문항 내용**: 영양 관련 지식 문항 20개 (예: 지방, 칼로리, 건강한 식단 등)

---

## 🎯 수정된 ICLV 모델 설정

### **기존 설정 (잘못됨)**

```python
# ❌ 건강관심도 1개만 사용
measurement_config = MeasurementConfig(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    n_categories=5  # 5점 척도
)
```

### **수정된 설정 (올바름)**

```python
# ✅ 5개 잠재변수 모두 사용
measurement_configs = [
    # 1. 건강관심도
    MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        indicator_type='ordered',
        n_categories=5
    ),
    
    # 2. 건강유익성
    MeasurementConfig(
        latent_variable='perceived_benefit',
        indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
        indicator_type='ordered',
        n_categories=5
    ),
    
    # 3. 가격수준
    MeasurementConfig(
        latent_variable='perceived_price',
        indicators=['q27', 'q28', 'q29'],
        indicator_type='ordered',
        n_categories=5
    ),
    
    # 4. 구매의도
    MeasurementConfig(
        latent_variable='purchase_intention',
        indicators=['q18', 'q19', 'q20'],
        indicator_type='ordered',
        n_categories=5
    ),
    
    # 5. 영양지식
    MeasurementConfig(
        latent_variable='nutrition_knowledge',
        indicators=['q30', 'q31', 'q32', 'q33', 'q34', 'q35', 
                    'q36', 'q37', 'q38', 'q39', 'q40', 'q41',
                    'q42', 'q43', 'q44', 'q45', 'q46', 'q47',
                    'q48', 'q49'],
        indicator_type='ordered',
        n_categories=5
    )
]
```

---

## 📊 데이터 통합 수정

### **기존 통합 (부족)**

```python
# ❌ 건강관심도만 통합
df_health = pd.read_csv('data/processed/survey/health_concern.csv')
df_integrated = df_dce.merge(df_health, on='respondent_id', how='left')
```

### **수정된 통합 (완전)**

```python
# ✅ 5개 잠재변수 모두 통합
df_health = pd.read_csv('data/processed/survey/health_concern.csv')
df_benefit = pd.read_csv('data/processed/survey/perceived_benefit.csv')
df_price = pd.read_csv('data/processed/survey/perceived_price.csv')
df_purchase = pd.read_csv('data/processed/survey/purchase_intention.csv')
df_nutrition = pd.read_csv('data/processed/survey/nutrition_knowledge.csv')

# 순차적 병합
df_integrated = df_dce.merge(df_health, on='respondent_id', how='left')
df_integrated = df_integrated.merge(df_benefit, on='respondent_id', how='left')
df_integrated = df_integrated.merge(df_price, on='respondent_id', how='left')
df_integrated = df_integrated.merge(df_purchase, on='respondent_id', how='left')
df_integrated = df_integrated.merge(df_nutrition, on='respondent_id', how='left')
```

---

## 🔧 구조모델 수정

### **잠재변수 간 관계 설정**

```python
# 구조모델: 사회인구학적 변수 → 잠재변수
structural_config = StructuralConfig(
    # 건강관심도 = f(age, gender, income, education)
    health_concern_covariates=['age_std', 'gender', 'income_std', 'education_level'],
    
    # 건강유익성 = f(건강관심도, 영양지식)
    perceived_benefit_covariates=['health_concern', 'nutrition_knowledge'],
    
    # 가격수준 = f(income)
    perceived_price_covariates=['income_std'],
    
    # 구매의도 = f(건강유익성, 가격수준, 건강관심도)
    purchase_intention_covariates=['perceived_benefit', 'perceived_price', 'health_concern']
)
```

---

## 🎯 선택모델 수정

### **잠재변수를 선택모델에 포함**

```python
# 선택모델: 건강 라벨 + 가격 + 잠재변수 → 선택
choice_config = ChoiceConfig(
    choice_attributes=['health_label', 'price'],
    price_variable='price',
    choice_type='binary',
    
    # 잠재변수 포함
    latent_variables=['health_concern', 'perceived_benefit', 
                      'perceived_price', 'purchase_intention'],
    lv_in_choice=True
)
```

---

## 📁 데이터 파일 현황

### **측정모델 데이터 (5개)**

| 파일 | 행 수 | 컬럼 수 | 상태 |
|------|-------|---------|------|
| `health_concern.csv` | 300 | 7 (no + 6 지표) | ✅ 존재 |
| `perceived_benefit.csv` | 300 | 7 (no + 6 지표) | ✅ 존재 |
| `perceived_price.csv` | 300 | 4 (no + 3 지표) | ✅ 존재 |
| `purchase_intention.csv` | 300 | 4 (no + 3 지표) | ✅ 존재 |
| `nutrition_knowledge.csv` | 300 | 21 (no + 20 지표) | ✅ 존재 |

---

## ✅ 최종 결론

### **이전 보고서 오류**

❌ **건강관심도 1개만 확인** → 불완전한 ICLV 모델

### **수정된 현황**

✅ **5개 잠재변수 모두 확인** → 완전한 ICLV 모델 가능

### **총 지표 수**

- 건강관심도: 6개
- 건강유익성: 6개
- 가격수준: 3개
- 구매의도: 3개
- 영양지식: 20개
- **총 38개 지표**

### **다음 단계**

1. ✅ 데이터 통합 스크립트 수정 (`integrate_iclv_data.py`)
2. ✅ ICLV 모델 설정 수정 (`run_iclv_estimation.py`)
3. ✅ 5개 잠재변수 모두 포함한 동시추정 실행

---

**보고 완료** ✅

