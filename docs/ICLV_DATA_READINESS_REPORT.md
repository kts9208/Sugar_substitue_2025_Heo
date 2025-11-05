# 📊 대체당 데이터 ICLV 동시추정 가능성 분석 보고서

**작성일**: 2025-11-05  
**분석 대상**: 대체당 연구 데이터  
**목적**: 측정모델, 구조모델, 선택모델을 이용한 동시추정 가능성 판단

---

## ✅ 핵심 결론

### **동시추정 가능 여부: ⚠️ 부분 가능 (선택모델 데이터 부족)**

| 컴포넌트 | 데이터 상태 | 가능 여부 | 비고 |
|---------|-----------|----------|------|
| **측정모델** | ✅ 완비 | ✅ 가능 | 건강관심도 6문항 (7점 척도) |
| **구조모델** | ✅ 완비 | ✅ 가능 | 사회인구학적 변수 13개 |
| **선택모델** | ❌ 부족 | ❌ 불가능 | DCE 데이터 미처리 상태 |
| **전체 동시추정** | ⚠️ 부분 | ⚠️ 조건부 가능 | DCE 데이터 전처리 필요 |

---

## 📋 데이터 현황 상세 분석

### **1. 측정모델 데이터 ✅ 완비**

#### **1.1 데이터 위치**
```
data/processed/survey/health_concern.csv
```

#### **1.2 데이터 구조**
- **응답자 수**: 300명
- **지표 수**: 6개 (q6, q7, q8, q9, q10, q11)
- **척도**: 7점 Likert 척도 (1-7)
- **결측치**: 없음

#### **1.3 샘플 데이터**
```
   no  q6  q7  q8  q9  q10  q11
0   1   4   4   3   3    4    3
1   3   4   4   3   4    3    3
2   5   4   4   4   4    3    3
```

#### **1.4 ICLV 설정**
```python
measurement_config = MeasurementConfig(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    indicator_type='ordered',
    n_categories=7  # 7점 척도
)
```

#### **1.5 가능성 평가**
✅ **완벽하게 준비됨**
- Ordered Probit 측정모델에 적합
- 충분한 지표 수 (6개)
- 적절한 척도 (7점)
- 결측치 없음

---

### **2. 구조모델 데이터 ✅ 완비**

#### **2.1 데이터 위치**
```
data/raw/Sugar_substitue_Raw data_250730.xlsx (DATA 시트)
```

#### **2.2 사회인구학적 변수**

| 변수명 (원본) | 변수명 (표준) | 설명 | 사용 가능 |
|--------------|--------------|------|----------|
| **q1** | gender | 성별 (0: 남성, 1: 여성) | ✅ |
| **q2_1** | age | 나이 (연속형) | ✅ |
| **q52** | income | 소득 (5개 범주) | ✅ |
| **q53** | education | 교육수준 (6개 범주) | ✅ |
| **q54** | diabetes | 당뇨병 여부 | ✅ |
| **q55** | family_diabetes | 가족 당뇨병 | ✅ |
| **q56** | sugar_substitute_usage | 설탕 대체재 사용 빈도 | ✅ |

**총 사용 가능 변수**: 7개 (핵심 4개 + 추가 3개)

#### **2.3 데이터 로딩 방법**
```python
from src.analysis.hybrid_choice_model.data_integration import load_sociodemographic_data

# 사회인구학적 데이터 로드
sociodem_data = load_sociodemographic_data()

# 자동 전처리 포함:
# - age → age_std (표준화)
# - income (범주형) → income_continuous → income_std
# - gender (0/1 그대로)
# - education → education_level
```

#### **2.4 ICLV 설정**
```python
structural_config = StructuralConfig(
    sociodemographics=['age_std', 'gender', 'income_std', 'education_level'],
    include_in_choice=True  # 선택모델에도 포함
)
```

#### **2.5 가능성 평가**
✅ **완벽하게 준비됨**
- 충분한 사회인구학적 변수 (7개)
- 자동 전처리 기능 구현됨
- 표준화 완료
- 결측치 처리 완료

---

### **3. 선택모델 데이터 ❌ 부족**

#### **3.1 현재 상태**
- **DCE 데이터 위치**: `data/raw/Sugar_substitue_Raw data_250730.xlsx`
- **DCE 변수**: q21, q22, q23, q24, q25, q26
- **전처리 상태**: ❌ **미처리**
- **선택 데이터 형식**: ❌ **미정의**

#### **3.2 원본 데이터 구조**
```
   no  q21  q22  q23  q24  q25  q26
0   1    1    2    1    1    1    2
1   3    1    2    2    1    2    1
2   5    1    2    1    3    1    2
```

**문제점**:
1. ❌ 선택 변수가 무엇인지 불명확 (q21-q26 중 어느 것?)
2. ❌ 가격 변수 미확인
3. ❌ 속성 변수 미확인
4. ❌ 선택 세트 구조 미확인
5. ❌ DCE 데이터 전처리 스크립트 없음

#### **3.3 필요한 데이터 구조**

**ICLV 선택모델에 필요한 형식**:
```python
# 개인별 1개 선택 (Binary Choice)
data = pd.DataFrame({
    'respondent_id': [1, 2, 3, ...],
    'price': [2000, 2500, 3000, ...],      # 가격
    'sugar_content': [0, 25, 50, ...],     # 설탕 함량
    'health_label': [0, 1, 0, ...],        # 건강 라벨
    'choice': [1, 0, 1, ...]               # 선택 (0 or 1)
})
```

또는

```python
# 선택 세트별 여러 대안 (Multinomial Choice)
data = pd.DataFrame({
    'respondent_id': [1, 1, 1, 2, 2, 2, ...],
    'choice_set': [1, 1, 1, 1, 1, 1, ...],
    'alternative': [0, 1, 2, 0, 1, 2, ...],
    'price': [2000, 2500, 3000, ...],
    'sugar_content': [0, 25, 50, ...],
    'choice': [1, 0, 0, 0, 1, 0, ...]      # 각 대안별 선택 여부
})
```

#### **3.4 가능성 평가**
❌ **현재 불가능 (전처리 필요)**
- DCE 데이터 구조 미확인
- 선택 변수 미정의
- 속성 변수 미정의
- 전처리 스크립트 필요

---

## 🔧 동시추정 가능 시나리오

### **시나리오 A: DCE 데이터 전처리 후 (권장) ⭐⭐⭐⭐⭐**

#### **필요 작업**:
1. ✅ DCE 데이터 구조 파악 (q21-q26 의미 확인)
2. ✅ 선택 변수 정의
3. ✅ 속성 변수 추출 (가격, 설탕 함량, 건강 라벨 등)
4. ✅ DCE 데이터 전처리 스크립트 작성
5. ✅ 데이터 통합 (설문 + 사회인구학적 + DCE)

#### **예상 코드**:
```python
from src.analysis.hybrid_choice_model.iclv_models import (
    OrderedProbitMeasurement,
    LatentVariableRegression,
    BinaryProbitChoice,
    SimultaneousEstimator,
    create_iclv_config
)

# 1. 설정
config = create_iclv_config(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    sociodemographics=['age_std', 'gender', 'income_std', 'education_level'],
    choice_attributes=['price', 'sugar_content', 'health_label'],
    price_variable='price',
    n_categories=7,
    choice_type='binary',
    n_draws=1000
)

# 2. 데이터 로드
health_concern = pd.read_csv("data/processed/survey/health_concern.csv")
sociodem_data = load_sociodemographic_data()
dce_data = load_dce_data()  # ← 전처리 필요!

# 3. 데이터 통합
integrated_data = integrate_data(health_concern, sociodem_data, dce_data)

# 4. 모델 생성
measurement_model = OrderedProbitMeasurement(config.measurement)
structural_model = LatentVariableRegression(config.structural)
choice_model = BinaryProbitChoice(config.choice)

# 5. 동시 추정
estimator = SimultaneousEstimator(config)
results = estimator.estimate(
    integrated_data,
    measurement_model,
    structural_model,
    choice_model
)
```

#### **예상 소요 시간**: 1-2일
- DCE 데이터 구조 파악: 2-4시간
- 전처리 스크립트 작성: 4-6시간
- 데이터 통합 및 검증: 2-4시간

---

### **시나리오 B: 측정모델 + 구조모델만 (즉시 가능) ⭐⭐⭐**

#### **가능한 분석**:
```python
# 1. 측정모델 단독 분석
measurement_model = OrderedProbitMeasurement(measurement_config)
measurement_results = measurement_model.fit(health_concern)

# 2. 구조모델 단독 분석
# 잠재변수 계산 (측정모델 결과 사용)
latent_var = measurement_results['factor_scores']

# 사회인구학적 데이터와 병합
merged_data = sociodem_data.copy()
merged_data['latent_var'] = latent_var

# 구조모델 추정
structural_model = LatentVariableRegression(structural_config)
structural_results = structural_model.fit(merged_data, latent_var)
```

#### **한계**:
- ❌ 선택모델 없음
- ❌ 완전한 ICLV 동시추정 불가능
- ✅ 측정모델 + 구조모델 Sequential 추정만 가능

---

## 📊 데이터 요약

### **사용 가능한 데이터**

| 데이터 유형 | 파일 | 관측치 | 변수 수 | 상태 |
|-----------|------|--------|---------|------|
| **건강관심도** | health_concern.csv | 300 | 6 | ✅ 완비 |
| **사회인구학적** | Raw data (DATA 시트) | 300 | 13 | ✅ 완비 |
| **DCE** | Raw data (q21-q26) | 300 | 6 | ❌ 미처리 |

### **필요한 추가 작업**

1. ⏳ **DCE 데이터 구조 파악**
   - q21-q26 각 변수의 의미 확인
   - 선택 변수 식별
   - 속성 변수 식별

2. ⏳ **DCE 데이터 전처리**
   - 선택 데이터 추출
   - 속성 데이터 추출
   - Long format 변환 (필요시)

3. ⏳ **데이터 통합**
   - 설문 + 사회인구학적 + DCE 병합
   - respondent_id 기준 매칭

---

## 🎯 권장 사항

### **즉시 실행 가능**
1. ✅ 측정모델 단독 분석
2. ✅ 구조모델 단독 분석
3. ✅ Sequential 추정 (측정 → 구조)

### **DCE 데이터 전처리 후 가능**
1. ⏳ 선택모델 추가
2. ⏳ 완전한 ICLV 동시추정
3. ⏳ WTP 계산

### **다음 단계**
1. **DCE 데이터 구조 확인**
   - 설문지 확인
   - q21-q26 변수 의미 파악
   - 선택 실험 설계 확인

2. **DCE 전처리 스크립트 작성**
   - 선택 변수 추출
   - 속성 변수 추출
   - 데이터 형식 변환

3. **통합 데이터 생성**
   - 3개 데이터 소스 병합
   - 결측치 처리
   - 최종 검증

---

## 📝 결론

### **현재 상태**
- ✅ 측정모델 데이터: 완비
- ✅ 구조모델 데이터: 완비
- ❌ 선택모델 데이터: 미처리

### **동시추정 가능 여부**
- ⚠️ **조건부 가능**: DCE 데이터 전처리 완료 시
- ✅ **부분 가능**: 측정모델 + 구조모델만 (즉시)

### **필요 작업**
1. DCE 데이터 구조 파악 (2-4시간)
2. DCE 전처리 스크립트 작성 (4-6시간)
3. 데이터 통합 (2-4시간)

**총 예상 소요 시간**: 1-2일

---

**보고서 작성일**: 2025-11-05  
**작성자**: Sugar Substitute Research Team  
**상태**: ✅ 분석 완료

