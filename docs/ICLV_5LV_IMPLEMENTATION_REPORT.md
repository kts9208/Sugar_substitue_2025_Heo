# ICLV 5개 잠재변수 구현 완료 보고서

**작성일**: 2025-11-05  
**프로젝트**: Sugar Substitute 2025 (대체당 연구)  
**목적**: 5개 잠재변수를 포함한 ICLV 데이터 통합 및 모델 설정 구현

---

## 📊 핵심 성과

### ✅ **구현 완료 항목**

| 항목 | 상태 | 결과 |
|------|------|------|
| **데이터 통합** | ✅ 완료 | 5개 잠재변수, 38개 지표 통합 |
| **ICLV 설정** | ✅ 완료 | 측정/구조/선택 모델 설정 |
| **데이터 검증** | ✅ 완료 | 모든 지표 및 변수 확인 |
| **코드 중복 제거** | ✅ 완료 | 기존 기능과 중복 없음 |

---

## 🔧 수정된 코드

### **1. 데이터 통합 스크립트** (`scripts/integrate_iclv_data.py`)

#### **수정 전 (건강관심도 1개만)**

```python
def load_health_concern_data():
    """건강관심도 데이터 로드"""
    df_health = pd.read_csv('data/processed/survey/health_concern.csv')
    return df_health

def integrate_data(df_dce, df_health, df_sociodem):
    """3개 데이터 소스 통합"""
    df_merged = df_dce.merge(df_health, on='respondent_id', how='left')
    df_integrated = df_merged.merge(df_sociodem, on='respondent_id', how='left')
    return df_integrated
```

#### **수정 후 (5개 잠재변수 모두)**

```python
def load_latent_variable_data():
    """5개 잠재변수 데이터 로드"""
    latent_vars = {}
    
    # 1. 건강관심도 (Q6-Q11)
    df_health = pd.read_csv('data/processed/survey/health_concern.csv')
    latent_vars['health_concern'] = df_health
    
    # 2. 건강유익성 (Q12-Q17)
    df_benefit = pd.read_csv('data/processed/survey/perceived_benefit.csv')
    latent_vars['perceived_benefit'] = df_benefit
    
    # 3. 가격수준 (Q27-Q29)
    df_price = pd.read_csv('data/processed/survey/perceived_price.csv')
    latent_vars['perceived_price'] = df_price
    
    # 4. 구매의도 (Q18-Q20)
    df_purchase = pd.read_csv('data/processed/survey/purchase_intention.csv')
    latent_vars['purchase_intention'] = df_purchase
    
    # 5. 영양지식 (Q30-Q49)
    df_nutrition = pd.read_csv('data/processed/survey/nutrition_knowledge.csv')
    latent_vars['nutrition_knowledge'] = df_nutrition
    
    return latent_vars

def integrate_data(df_dce, latent_vars, df_sociodem):
    """DCE + 5개 잠재변수 + 사회인구학적 데이터 통합"""
    df_merged = df_dce.copy()
    
    # 5개 잠재변수 순차 병합
    for lv_name, df_lv in latent_vars.items():
        df_merged = df_merged.merge(df_lv, on='respondent_id', how='left')
    
    # 사회인구학적 병합
    df_integrated = df_merged.merge(df_sociodem, on='respondent_id', how='left')
    
    return df_integrated
```

#### **검증 강화**

```python
def validate_integration(df_integrated, df_dce):
    """통합 데이터 검증"""
    
    # 5개 잠재변수 지표 검증
    health_cols = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']  # 건강관심도
    benefit_cols = ['q12', 'q13', 'q14', 'q15', 'q16', 'q17']  # 건강유익성
    purchase_cols = ['q18', 'q19', 'q20']  # 구매의도
    price_cols = ['q27', 'q28', 'q29']  # 가격수준
    nutrition_cols = [f'q{i}' for i in range(30, 50)]  # 영양지식
    
    # 모든 지표 존재 확인
    for col in health_cols + benefit_cols + purchase_cols + price_cols + nutrition_cols:
        assert col in df_integrated.columns, f"{col} 컬럼 누락"
    
    print(f"   ✓ 총 38개 지표 모두 존재")
```

---

### **2. ICLV 모델 설정 스크립트** (`scripts/run_iclv_estimation.py`)

#### **수정 전 (건강관심도 1개만)**

```python
def create_iclv_config():
    """ICLV 모델 설정 생성"""
    
    measurement_config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        n_categories=7
    )
    
    structural_config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std', 'education_level']
    )
    
    choice_config = ChoiceConfig(
        choice_attributes=['health_label', 'price']
    )
    
    return ICLVConfig(
        measurement=measurement_config,
        structural=structural_config,
        choice=choice_config
    )
```

#### **수정 후 (5개 잠재변수 모두)**

```python
def create_iclv_config():
    """ICLV 모델 설정 생성 (5개 잠재변수)"""
    
    configs = {}
    
    # 1. 건강관심도 (Q6-Q11)
    configs['health_concern'] = {
        'measurement': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['age_std', 'gender', 'income_std', 'education_level']
        )
    }
    
    # 2. 건강유익성 (Q12-Q17)
    configs['perceived_benefit'] = {
        'measurement': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['health_concern']  # 건강관심도의 영향
        )
    }
    
    # 3. 구매의도 (Q18-Q20)
    configs['purchase_intention'] = {
        'measurement': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['perceived_benefit', 'perceived_price']
        )
    }
    
    # 4. 가격수준 (Q27-Q29)
    configs['perceived_price'] = {
        'measurement': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['income_std']
        )
    }
    
    # 5. 영양지식 (Q30-Q49)
    configs['nutrition_knowledge'] = {
        'measurement': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=[f'q{i}' for i in range(30, 50)],
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['age_std', 'education_level']
        )
    }
    
    # 선택모델 설정 (공통)
    choice_config = ChoiceConfig(
        choice_attributes=['health_label', 'price'],
        price_variable='price',
        choice_type='binary',
        lv_in_choice=True
    )
    
    return {
        'latent_variables': configs,
        'choice': choice_config,
        'estimation': {'n_draws': 500, 'seed': 42, 'method': 'simultaneous'}
    }
```

---

## 📊 실행 결과

### **1. 데이터 통합 결과**

```
================================================================================
ICLV 데이터 통합 (5개 잠재변수)
================================================================================

[2] 잠재변수 데이터 로드 중...
   [2-1] 건강관심도...
      - 299명, 지표: ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
   [2-2] 건강유익성...
      - 299명, 지표: ['q12', 'q13', 'q14', 'q15', 'q16', 'q17']
   [2-3] 가격수준...
      - 299명, 지표: ['q27', 'q28', 'q29']
   [2-4] 구매의도...
      - 299명, 지표: ['q18', 'q19', 'q20']
   [2-5] 영양지식...
      - 299명, 지표: ['q30', ..., 'q49']

   - 총 5개 잠재변수 로드 완료

[4] 데이터 통합 중...
   - Step 1: + health_concern 병합... (5,400행 × 15컬럼)
   - Step 2: + perceived_benefit 병합... (5,400행 × 21컬럼)
   - Step 3: + perceived_price 병합... (5,400행 × 24컬럼)
   - Step 4: + purchase_intention 병합... (5,400행 × 27컬럼)
   - Step 5: + nutrition_knowledge 병합... (5,400행 × 47컬럼)
   - Step 6: + 사회인구학적 병합... (5,400행 × 58컬럼)

[5] 통합 데이터 검증 중...
   ✓ 건강관심도 (6개): ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
   ✓ 건강유익성 (6개): ['q12', 'q13', 'q14', 'q15', 'q16', 'q17']
   ✓ 구매의도 (3개): ['q18', 'q19', 'q20']
   ✓ 가격수준 (3개): ['q27', 'q28', 'q29']
   ✓ 영양지식 (20개): q30-q49
   ✓ 총 38개 지표 모두 존재

ICLV 데이터 통합 완료! (5개 잠재변수, 38개 지표)
```

### **2. 데이터 검증 결과**

```
================================================================================
ICLV 통합 데이터 검증 (5개 잠재변수)
================================================================================

[2] 잠재변수 지표 확인...
   ✓ 건강관심도: 6개 지표 모두 존재
   ✓ 건강유익성: 6개 지표 모두 존재
   ✓ 구매의도: 3개 지표 모두 존재
   ✓ 가격수준: 3개 지표 모두 존재
   ✓ 영양지식: 20개 지표 모두 존재

   총 38개 지표 확인 완료

[6] ICLV 추정용 데이터 준비...
   - 구매안함 제외: 5,400행 → 3,600행
   - income_std 결측치를 평균(-0.009)으로 대체

   최종 ICLV 데이터:
   - 행 수: 3,600
   - 응답자 수: 299
   - 선택 세트: 6개
   - 대안 수: 2개

   선택 분포:
   - 제품 A: 960회 (56.5%)
   - 제품 B: 739회 (43.5%)
```

---

## 📁 생성/수정된 파일

### **수정된 파일 (2개)**

1. **`scripts/integrate_iclv_data.py`**
   - 5개 잠재변수 데이터 로드 함수 추가
   - 순차 병합 로직 구현
   - 38개 지표 검증 강화

2. **`scripts/run_iclv_estimation.py`**
   - 5개 잠재변수 설정 생성
   - 잠재변수 간 구조모델 관계 정의
   - 데이터 검증 강화

### **생성된 파일 (2개)**

1. **`scripts/test_iclv_config.py`**
   - 5개 잠재변수 통합 데이터 검증
   - ICLV 모델 설정 정보 출력

2. **`docs/ICLV_5LV_IMPLEMENTATION_REPORT.md`** (본 파일)
   - 구현 완료 보고서

### **생성된 데이터 (1개)**

1. **`data/processed/iclv/integrated_data.csv`**
   - 5,400행 × 58컬럼
   - 5개 잠재변수, 38개 지표 포함

---

## 🎯 ICLV 모델 구조

### **측정모델 (5개)**

| 잠재변수 | 지표 수 | 문항 번호 | 척도 |
|---------|--------|----------|------|
| 건강관심도 | 6개 | Q6-Q11 | 5점 |
| 건강유익성 | 6개 | Q12-Q17 | 5점 |
| 구매의도 | 3개 | Q18-Q20 | 5점 |
| 가격수준 | 3개 | Q27-Q29 | 5점 |
| 영양지식 | 20개 | Q30-Q49 | 5점 |
| **총계** | **38개** | - | - |

### **구조모델 (잠재변수 간 관계)**

```
사회인구학적 변수 → 건강관심도
건강관심도 → 건강유익성
소득 → 가격수준
건강유익성 + 가격수준 → 구매의도
연령 + 교육 → 영양지식
```

### **선택모델**

```
선택 = f(건강 라벨, 가격, 잠재변수들)
```

---

## ✅ 최종 결론

### **구현 완료 사항**

1. ✅ **5개 잠재변수 데이터 통합** (38개 지표)
2. ✅ **ICLV 모델 설정 생성** (측정/구조/선택)
3. ✅ **데이터 검증 완료** (모든 지표 및 변수 확인)
4. ✅ **코드 중복 제거** (기존 기능과 중복 없음)

### **데이터 현황**

- **통합 데이터**: 5,400행 × 58컬럼
- **ICLV 추정용**: 3,600행 (구매안함 제외)
- **응답자 수**: 299명
- **선택 세트**: 6개
- **대안 수**: 2개 (제품 A, 제품 B)

### **다음 단계**

1. ⏳ SimultaneousEstimator 클래스 완전 구현
2. ⏳ 5개 잠재변수 동시추정 실행
3. ⏳ 추정 결과 분석 및 해석
4. ⏳ WTP 계산 및 정책 시사점 도출

---

**구현 완료** ✅  
**보고 일시**: 2025-11-05

