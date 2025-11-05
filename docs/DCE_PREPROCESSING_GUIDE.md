# 📊 DCE 전처리 완전 가이드

**작성일**: 2025-11-05  
**목적**: DCE 데이터 구조 파악 및 ICLV 선택모델용 전처리 방법 제안

---

## 🔍 1. DCE 데이터 구조 분석 결과

### **1.1 현재 데이터 형식**

**원본 데이터**: `data/raw/Sugar_substitue_Raw data_250730.xlsx`

```
   no  q21  q22  q23  q24  q25  q26
0   1    1    2    1    1    1    2
1   3    1    2    2    1    2    1
2   5    1    2    1    3    1    2
```

### **1.2 변수 의미 (LABEL 시트 기반)**

| 변수 | 라벨 | 의미 추정 | 값 범위 |
|------|------|----------|---------|
| **q21** | 제품 A | 제품 A 선택 세트 1 | 1-3 |
| **q22** | 제품 B | 제품 B 선택 세트 1 | 1-3 |
| **q23** | 제품 A | 제품 A 선택 세트 2 | 1-3 |
| **q24** | 제품 A | 제품 A 선택 세트 3 | 1-3 |
| **q25** | 제품 A | 제품 A 선택 세트 4 | 1-3 |
| **q26** | 제품 B | 제품 B 선택 세트 4 | 1-3 |

### **1.3 값 분포 분석**

```
q21: 1(222명), 2(63명), 3(15명)  → 대부분 1 선택
q22: 1(26명), 2(264명), 3(10명)  → 대부분 2 선택
q23: 1(168명), 2(117명), 3(15명) → 1과 2 혼재
q24: 1(260명), 2(12명), 3(28명)  → 대부분 1 선택
q25: 1(214명), 2(70명), 3(16명)  → 대부분 1 선택
q26: 1(70명), 2(213명), 3(17명)  → 대부분 2 선택
```

**패턴 해석**:
- 값 1, 2, 3은 **선택지 코드**로 추정
- 각 질문마다 3개 대안 중 1개 선택 (Multinomial Choice)
- 제품 A/B는 **선택 세트** 또는 **속성 조합**을 의미

---

## 🎯 2. DCE 전처리란?

### **2.1 정의**

**DCE 전처리**는 원본 설문 데이터(q21-q26)를 **ICLV 선택모델이 요구하는 형식**으로 변환하는 과정입니다.

### **2.2 변환 목표**

**현재 형식** (Wide Format):
```
respondent_id  q21  q22  q23  q24  q25  q26
1              1    2    1    1    1    2
3              1    2    2    1    2    1
```

**목표 형식** (Long Format for ICLV):
```
respondent_id  choice_set  alternative  price  sugar  label  choice
1              1           0            2000   0      1      1
1              1           1            2500   25     0      0
1              1           2            3000   50     1      0
1              2           0            2000   0      0      0
1              2           1            2500   25     1      1
...
```

### **2.3 필요한 정보**

DCE 전처리를 위해 **반드시 필요한 정보**:

1. ✅ **선택 세트 수**: 몇 개의 선택 상황? (예: 6개)
2. ✅ **대안 수**: 각 세트당 몇 개 대안? (예: 3개)
3. ❌ **속성 정의**: 가격, 설탕 함량, 건강 라벨 등
4. ❌ **속성 수준**: 각 속성의 값 (예: 가격 2000/2500/3000원)
5. ❌ **실험 설계**: 어떤 조합이 제시되었는지
6. ❌ **선택 변수**: 1/2/3 중 무엇이 "선택함"인지

**현재 상태**: ❌ **3-6번 정보 부족** → 설문지 확인 필요!

---

## 📋 3. DCE 전처리 시나리오

### **시나리오 A: 설문지 정보 확보 후 (권장) ⭐⭐⭐⭐⭐**

#### **3.1 필요한 설문지 정보**

1. **DCE 질문 구조**
   ```
   예시:
   Q21. 다음 두 제품 중 어느 것을 선택하시겠습니까?
   
   제품 A: 가격 2,000원, 설탕 0%, 건강 라벨 있음
   제품 B: 가격 2,500원, 설탕 25%, 건강 라벨 없음
   제품 C: 구매하지 않음
   
   1) 제품 A
   2) 제품 B
   3) 구매하지 않음
   ```

2. **속성 수준표**
   ```
   | 속성 | 수준 1 | 수준 2 | 수준 3 |
   |------|--------|--------|--------|
   | 가격 | 2,000원 | 2,500원 | 3,000원 |
   | 설탕 함량 | 0% | 25% | 50% |
   | 건강 라벨 | 있음 | 없음 | - |
   ```

3. **실험 설계 매트릭스**
   ```
   | 선택 세트 | 대안 | 가격 | 설탕 | 라벨 |
   |----------|------|------|------|------|
   | 1 | A | 2000 | 0 | 1 |
   | 1 | B | 2500 | 25 | 0 |
   | 1 | C | - | - | - |
   ```

#### **3.2 전처리 프로세스**

```python
# Step 1: 설문지 정보 기반 설계 매트릭스 생성
design_matrix = create_design_matrix(
    choice_sets=6,  # q21-q26 = 6개 선택 세트
    alternatives=3,  # 각 세트당 3개 대안
    attributes={
        'price': [2000, 2500, 3000],
        'sugar': [0, 25, 50],
        'label': [0, 1]
    }
)

# Step 2: 응답 데이터와 설계 매트릭스 결합
dce_long = merge_responses_with_design(
    responses=df[['no', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26']],
    design_matrix=design_matrix
)

# Step 3: 선택 변수 생성
dce_long['choice'] = create_choice_indicator(
    dce_long['alternative'],
    dce_long['selected_alternative']
)

# Step 4: ICLV 형식으로 변환
dce_iclv = convert_to_iclv_format(dce_long)
```

**예상 결과**:
```
respondent_id  choice_set  price  sugar  label  choice
1              1           2000   0      1      1
1              1           2500   25     0      0
1              1           3000   50     0      0
1              2           2000   0      0      0
1              2           2500   25     1      1
...
```

---

### **시나리오 B: 설문지 없이 역추정 (차선책) ⭐⭐⭐**

설문지를 확보할 수 없는 경우, **데이터 패턴 분석**으로 역추정:

#### **3.1 역추정 가능한 정보**

1. ✅ **선택 세트 수**: 6개 (q21-q26)
2. ✅ **대안 수**: 3개 (값 범위 1-3)
3. ⚠️ **속성**: 추정 필요
4. ⚠️ **수준**: 추정 필요

#### **3.2 역추정 방법**

```python
# 1. 값 분포 분석
# q21: 1(222), 2(63), 3(15) → 대안 1이 가장 매력적
# q22: 1(26), 2(264), 3(10) → 대안 2가 가장 매력적
# → 각 선택 세트마다 속성 조합이 다름

# 2. 상관관계 분석
# q23-q25 높은 상관 (0.47) → 같은 속성 패턴?
# q22-q26 높은 상관 (0.37) → 제품 B 관련?

# 3. 가정 기반 설계 매트릭스 생성
# 가정: 3개 속성 (가격, 설탕, 라벨), 각 3수준
assumed_design = create_assumed_design(
    n_sets=6,
    n_alternatives=3,
    n_attributes=3
)
```

**한계**: ❌ 정확도 낮음, 해석 어려움

---

## 🔧 4. 구현 방법 제안

### **4.1 즉시 실행 가능한 분석 (설문지 확인 전)**

#### **Step 1: 기본 구조 파악**

```bash
# 이미 실행 완료!
python scripts/analyze_dce_structure.py
```

**결과**:
- ✅ 6개 선택 세트 확인
- ✅ 각 3개 대안 확인
- ✅ 값 분포 파악
- ❌ 속성 정보 부족

#### **Step 2: 추가 분석 스크립트**

```python
# scripts/analyze_dce_patterns.py
# 목적: 선택 패턴 분석으로 속성 추정

# 1. 선택 일관성 분석
# 2. 대안 간 선호도 비교
# 3. 응답자 특성별 선택 패턴
# 4. 가능한 속성 조합 추정
```

---

### **4.2 설문지 확보 후 구현 (권장)**

#### **Step 1: 설계 매트릭스 생성**

```python
# scripts/create_dce_design_matrix.py

import pandas as pd
import numpy as np

def create_design_matrix():
    """
    설문지 정보 기반 DCE 설계 매트릭스 생성
    
    입력: 설문지에서 확인한 속성 및 수준
    출력: design_matrix.csv
    """
    
    # 예시: 설문지에서 확인한 정보
    design = {
        'choice_set': [],
        'alternative': [],
        'price': [],
        'sugar_content': [],
        'health_label': []
    }
    
    # 선택 세트 1 (q21)
    design['choice_set'].extend([1, 1, 1])
    design['alternative'].extend([1, 2, 3])
    design['price'].extend([2000, 2500, 3000])  # 설문지 확인 필요!
    design['sugar_content'].extend([0, 25, 50])  # 설문지 확인 필요!
    design['health_label'].extend([1, 0, 0])     # 설문지 확인 필요!
    
    # 선택 세트 2-6 반복...
    
    df_design = pd.DataFrame(design)
    df_design.to_csv('data/processed/dce/design_matrix.csv', index=False)
    
    return df_design
```

#### **Step 2: 응답 데이터 변환**

```python
# scripts/preprocess_dce_data.py

def preprocess_dce_data():
    """
    DCE 응답 데이터를 ICLV 형식으로 변환
    
    입력: 
    - data/raw/Sugar_substitue_Raw data_250730.xlsx (q21-q26)
    - data/processed/dce/design_matrix.csv
    
    출력:
    - data/processed/dce/dce_long_format.csv
    """
    
    # 1. 원본 데이터 로드
    df = pd.read_excel('data/raw/Sugar_substitue_Raw data_250730.xlsx', 
                       sheet_name='DATA')
    
    # 2. 설계 매트릭스 로드
    design = pd.read_csv('data/processed/dce/design_matrix.csv')
    
    # 3. Long format 변환
    dce_long = []
    
    for idx, row in df.iterrows():
        respondent_id = row['no']
        
        # 각 선택 세트 처리
        for choice_set in range(1, 7):
            q_col = f'q{20 + choice_set}'
            selected = row[q_col]  # 1, 2, or 3
            
            # 해당 선택 세트의 모든 대안 추가
            set_design = design[design['choice_set'] == choice_set]
            
            for _, alt_row in set_design.iterrows():
                dce_long.append({
                    'respondent_id': respondent_id,
                    'choice_set': choice_set,
                    'alternative': alt_row['alternative'],
                    'price': alt_row['price'],
                    'sugar_content': alt_row['sugar_content'],
                    'health_label': alt_row['health_label'],
                    'choice': 1 if alt_row['alternative'] == selected else 0
                })
    
    df_long = pd.DataFrame(dce_long)
    df_long.to_csv('data/processed/dce/dce_long_format.csv', index=False)
    
    return df_long
```

#### **Step 3: ICLV 통합**

```python
# scripts/integrate_iclv_data.py

def integrate_iclv_data():
    """
    측정모델 + 구조모델 + 선택모델 데이터 통합
    
    출력: data/processed/iclv/integrated_data.csv
    """
    
    # 1. 각 데이터 로드
    health_concern = pd.read_csv('data/processed/survey/health_concern.csv')
    sociodem = load_sociodemographic_data()
    dce = pd.read_csv('data/processed/dce/dce_long_format.csv')
    
    # 2. respondent_id 기준 병합
    # health_concern: no → respondent_id
    health_concern = health_concern.rename(columns={'no': 'respondent_id'})
    
    # 3. DCE와 병합 (respondent_id 기준)
    integrated = dce.merge(health_concern, on='respondent_id', how='left')
    integrated = integrated.merge(sociodem, on='respondent_id', how='left')
    
    # 4. 저장
    integrated.to_csv('data/processed/iclv/integrated_data.csv', index=False)
    
    return integrated
```

---

## 📊 5. 전처리 후 ICLV 동시추정

### **5.1 데이터 준비 완료 후**

```python
from src.analysis.hybrid_choice_model.iclv_models import (
    OrderedProbitMeasurement,
    LatentVariableRegression,
    BinaryProbitChoice,
    SimultaneousEstimator,
    create_iclv_config
)

# 1. 통합 데이터 로드
integrated_data = pd.read_csv('data/processed/iclv/integrated_data.csv')

# 2. 설정
config = create_iclv_config(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    sociodemographics=['age_std', 'gender', 'income_std', 'education_level'],
    choice_attributes=['price', 'sugar_content', 'health_label'],
    price_variable='price',
    n_categories=7,
    choice_type='binary',  # 각 대안별 선택 여부
    n_draws=1000
)

# 3. 모델 생성
measurement_model = OrderedProbitMeasurement(config.measurement)
structural_model = LatentVariableRegression(config.structural)
choice_model = BinaryProbitChoice(config.choice)

# 4. 동시 추정
estimator = SimultaneousEstimator(config)
results = estimator.estimate(
    integrated_data,
    measurement_model,
    structural_model,
    choice_model
)

# 5. 결과 분석
print("측정모델 결과:", results['measurement'])
print("구조모델 결과:", results['structural'])
print("선택모델 결과:", results['choice'])
print("WTP:", results['wtp'])
```

---

## 🎯 6. 다음 단계 (우선순위)

### **우선순위 1: 설문지 확인 ⭐⭐⭐⭐⭐**

**필요한 정보**:
1. ✅ DCE 질문 원문 (Q21-Q26)
2. ✅ 각 질문의 선택지 설명
3. ✅ 속성 및 수준 정의
4. ✅ 실험 설계 매트릭스 (있는 경우)

**확인 방법**:
- 설문지 PDF/문서 확인
- 연구 계획서 확인
- DCE 설계 담당자에게 문의

### **우선순위 2: 설계 매트릭스 생성**

설문지 확인 후:
```bash
python scripts/create_dce_design_matrix.py
```

### **우선순위 3: DCE 전처리**

```bash
python scripts/preprocess_dce_data.py
```

### **우선순위 4: 데이터 통합**

```bash
python scripts/integrate_iclv_data.py
```

### **우선순위 5: ICLV 동시추정**

```bash
python scripts/run_iclv_estimation.py
```

---

## 📝 요약

### **DCE 전처리란?**
원본 설문 데이터(q21-q26)를 ICLV 선택모델이 요구하는 Long Format으로 변환하는 과정

### **현재 상태**
- ✅ 기본 구조 파악 완료 (6개 선택 세트, 각 3개 대안)
- ❌ 속성 정보 부족 (설문지 확인 필요)

### **필요한 작업**
1. **설문지 확인** (가장 중요!)
2. 설계 매트릭스 생성
3. 응답 데이터 변환
4. 데이터 통합

### **예상 소요 시간**
- 설문지 확인: 1-2시간
- 설계 매트릭스 생성: 2-3시간
- 전처리 스크립트 작성: 3-4시간
- 데이터 통합 및 검증: 2-3시간
- **총 8-12시간 (1-2일)**

---

**작성일**: 2025-11-05  
**상태**: ✅ 분석 완료, 설문지 확인 대기

