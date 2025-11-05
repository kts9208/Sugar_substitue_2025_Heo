# 📊 사회인구학적 데이터 통합 분석

**작성일**: 2025-11-04  
**목적**: 기존 코드에서 사회인구학적 데이터 처리 기능 확인 및 통합 방안 제시

---

## ✅ 핵심 결론

### **기존 시스템에 사회인구학적 데이터 처리 기능: 부분적으로 존재 ⚠️**

| 항목 | 상태 | 설명 |
|------|------|------|
| **원본 데이터 존재** | ✅ 있음 | `data/raw/Sugar_substitue_Raw data_250730.xlsx` |
| **사회인구학적 변수** | ✅ 있음 | q1 (gender), q2_1 (age), q52 (income), q53 (education) 등 |
| **데이터 로더** | ❌ 없음 | 사회인구학적 변수 전용 로더 없음 |
| **전처리 모듈** | ❌ 없음 | 사회인구학적 변수 전처리 모듈 없음 |
| **통합 기능** | ⚠️ 부분 | DCE-SEM 통합은 있으나 사회인구학적 변수 미포함 |

---

## 📋 원본 데이터 분석

### **1. 데이터 파일 위치**

**파일**: `data/raw/Sugar_substitue_Raw data_250730.xlsx`

**시트 구조**:
- `DATA`: 원본 설문 데이터 (300 obs × 58 variables)
- `LABEL`: 변수 레이블 (한글 설명)
- `CODE`: 변수 코딩 정보

---

### **2. 사회인구학적 변수 목록**

LABEL 시트 분석 결과:

| 변수명 | 설명 | 예시 값 | 비고 |
|--------|------|---------|------|
| **q1** | 성별 | 0) 남성, 1) 여성 | 이진 변수 |
| **q2_1** | 나이 (연속형) | 32, 40, 28, ... | 실제 나이 |
| **q3** | 연령대 (범주형) | 1) 만20-29세, 2) 만30-39세, ... | 5개 범주 |
| **q4** | 연령대 (간단) | 1) 20대, 2) 30대, ... | 4개 범주 |
| **q5** | 거주지역 | 1) 서울특별시, 8) 경기도, ... | 17개 지역 |
| **q51** | 직업 | 1) 전문직, 6) 사무직, 12) 학생, ... | 14개 직업 |
| **q51_14** | 직업 기타 | 기초생활수급자, 보건, ... | 자유 응답 |
| **q52** | 소득 | 1) 200만원 미만, 2) 200-300만원, ... | 5개 범주 |
| **q53** | 교육수준 | 1) 고졸 미만, 2) 고졸, 4) 대학 졸업, ... | 6개 범주 |
| **q54** | 당뇨병 여부 | 1) 예, 0) 아니오, 2) 아니오 | 이진 변수 |
| **q55** | 가족 당뇨병 | 1) 예, 0) 아니오, 2) 아니오 | 이진 변수 |
| **q56** | 설탕 대체재 사용 빈도 | 1) 항상, 2) 자주 함, 3) 가끔 함, 4) 거의 안 함 | 4개 범주 |

---

### **3. 데이터 샘플**

**첫 3개 관측치**:

```
   no  q1  q2_1  q3  q4  q5  ...  q51  q51_14  q52  q53  q54  q55  q56
0   1   0    32   2   2   1  ...   13     NaN    4    1    2    2    3
1   3   0    40   3   3   1  ...    6     NaN    4    0    2    3    3
2   5   1    28   1   1   8  ...    6     NaN    2    1    2    2    2
```

**해석**:
- 관측치 1: 남성, 32세, 30대, 서울, 무직, 대학 졸업, 당뇨병 없음, 가끔 사용
- 관측치 2: 남성, 40세, 40대, 서울, 사무직, 대학 졸업, 당뇨병 없음, 가끔 사용
- 관측치 3: 여성, 28세, 20대, 경기도, 사무직, 고졸, 당뇨병 없음, 자주 사용

---

## 🔍 기존 코드 분석

### **1. 데이터 로더 현황**

#### **A. 설문 데이터 로더** ✅

**파일**: `src/analysis/factor_analysis/data_loader.py`

**기능**:
- 요인별 CSV 파일 로드 (health_concern, perceived_benefit 등)
- 결측치 처리
- 데이터 검증

**한계**:
- ❌ 사회인구학적 변수 미포함
- ❌ 원본 Excel 파일 직접 로드 불가

---

#### **B. DCE 데이터 로더** ✅

**파일**: `src/analysis/utility_function/data_loader/dce_loader.py`

**기능**:
- DCE 선택 데이터 로드
- 속성 데이터 로드
- 선택 매트릭스 생성

**한계**:
- ❌ 사회인구학적 변수 미포함

---

#### **C. SEM 데이터 로더** ✅

**파일**: `src/analysis/utility_function/data_loader/sem_loader.py`

**기능**:
- SEM 결과 로드 (구조경로, 적합도 등)
- 요인 효과 처리

**한계**:
- ❌ 사회인구학적 변수 미포함

---

### **2. 데이터 통합 현황**

#### **A. Hybrid Data Integrator** ⚠️

**파일**: `src/analysis/hybrid_choice_model/data_integration/hybrid_data_integrator.py`

**기능**:
- DCE 데이터 + SEM 데이터 통합
- 개체 ID 기준 병합
- 잠재변수 통합

**한계**:
- ❌ 사회인구학적 변수 통합 기능 없음
- ⚠️ 구조는 있으나 실제 사용 안 됨

**코드 예시**:
```python
def _integrate_data(self, dce_data: pd.DataFrame, sem_data: pd.DataFrame, **kwargs) -> IntegrationResult:
    """데이터 통합"""
    # 개체 ID 기준 병합
    # 잠재변수 목록
    latent_variables = self.config.data.latent_variables
    
    return self.data_integrator.integrate_data(dce_data, sem_data, latent_variables)
```

---

### **3. 테스트 코드 분석**

#### **A. 합성 데이터 생성 함수** ✅

**파일**: `tests/test_structural_equations_real_data.py`

**함수**: `create_synthetic_sociodemographics()`

```python
def create_synthetic_sociodemographics(n_obs: int = 300) -> pd.DataFrame:
    """
    합성 사회인구학적 변수 생성
    
    실제 데이터에 사회인구학적 변수가 없으므로 합성 데이터 생성
    """
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.normal(40, 15, n_obs),  # 평균 40세, 표준편차 15
        'gender': np.random.binomial(1, 0.5, n_obs),  # 0: 남성, 1: 여성
        'income': np.random.normal(5, 2, n_obs),  # 평균 500만원 (단위: 100만원)
        'education': np.random.choice([1, 2, 3, 4], n_obs)  # 1: 고졸, 2: 전문대, 3: 대졸, 4: 대학원
    })
    
    # 표준화
    data['age_std'] = (data['age'] - data['age'].mean()) / data['age'].std()
    data['income_std'] = (data['income'] - data['income'].mean()) / data['income'].std()
    
    return data
```

**의미**:
- ✅ 구조모델 테스트용 합성 데이터 생성
- ❌ 실제 데이터 사용 안 함
- ⚠️ 이 함수가 존재한다는 것은 실제 사회인구학적 데이터가 없었음을 의미

---

## 📊 기존 시스템 활용 가능성

### **활용 가능한 컴포넌트**

| 컴포넌트 | 활용도 | 설명 |
|----------|--------|------|
| **FactorDataLoader** | 30% | 구조 참고 가능, 사회인구학적 변수용 수정 필요 |
| **HybridDataIntegrator** | 50% | 통합 로직 재사용 가능, 사회인구학적 변수 추가 필요 |
| **create_synthetic_sociodemographics** | 20% | 변수 구조 참고, 실제 데이터 로드로 대체 필요 |

---

## 🎯 통합 방안

### **방안 1: 새로운 SociodemographicLoader 클래스 생성** (권장 ✅)

**장점**:
- ✅ 명확한 책임 분리
- ✅ 재사용성 높음
- ✅ 기존 코드 영향 최소화

**단점**:
- ⚠️ 새로운 파일 생성 필요
- ⚠️ 작업량 중간

**구현 예시**:
```python
# src/analysis/hybrid_choice_model/data_integration/sociodemographic_loader.py

class SociodemographicLoader:
    """사회인구학적 변수 로더"""
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
    
    def load_data(self) -> pd.DataFrame:
        """원본 Excel 파일에서 사회인구학적 변수 로드"""
        df = pd.read_excel(self.raw_data_path, sheet_name='DATA')
        
        # 사회인구학적 변수 선택
        sociodem_vars = ['no', 'q1', 'q2_1', 'q52', 'q53']
        sociodem_data = df[sociodem_vars].copy()
        
        # 변수명 변경
        sociodem_data = sociodem_data.rename(columns={
            'no': 'respondent_id',
            'q1': 'gender',
            'q2_1': 'age',
            'q52': 'income',
            'q53': 'education'
        })
        
        return sociodem_data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """사회인구학적 변수 전처리"""
        processed = data.copy()
        
        # 나이 표준화
        processed['age_std'] = (processed['age'] - processed['age'].mean()) / processed['age'].std()
        
        # 소득 코딩 (범주형 → 연속형)
        income_mapping = {
            1: 1.5,  # 200만원 미만 → 150만원
            2: 2.5,  # 200-300만원 → 250만원
            3: 3.5,  # 300-400만원 → 350만원
            4: 4.5,  # 400-500만원 → 450만원
            5: 6.0   # 600만원 이상 → 600만원
        }
        processed['income_continuous'] = processed['income'].map(income_mapping)
        processed['income_std'] = (processed['income_continuous'] - processed['income_continuous'].mean()) / processed['income_continuous'].std()
        
        # 성별 (0: 남성, 1: 여성) - 그대로 사용
        
        # 교육수준 (1: 고졸 미만, 2: 고졸, 3: 대학 재학, 4: 대학 졸업, 5: 대학원 재학, 6: 대학원 졸업)
        # 그대로 사용 또는 더미 변수화
        
        return processed
```

---

### **방안 2: HybridDataIntegrator 확장** (차선책 ⚠️)

**장점**:
- ✅ 기존 통합 로직 활용
- ✅ 파일 수 증가 없음

**단점**:
- ❌ 클래스 복잡도 증가
- ❌ 단일 책임 원칙 위반

**구현 예시**:
```python
# src/analysis/hybrid_choice_model/data_integration/hybrid_data_integrator.py

class HybridDataIntegrator:
    def load_sociodemographic_data(self, raw_data_path: str) -> pd.DataFrame:
        """사회인구학적 변수 로드"""
        # 방안 1과 동일한 로직
        pass
    
    def integrate_data(self, dce_data, sem_data, sociodem_data, latent_variables):
        """DCE + SEM + 사회인구학적 데이터 통합"""
        # 기존 로직 + 사회인구학적 변수 병합
        pass
```

---

### **방안 3: 기존 FactorDataLoader 수정** (비권장 ❌)

**장점**:
- ✅ 새 파일 불필요

**단점**:
- ❌ 책임 범위 불명확
- ❌ 요인 데이터와 사회인구학적 데이터는 성격이 다름
- ❌ 유지보수 어려움

---

## 📝 최종 권장 사항

### **권장 방안: 방안 1 (SociodemographicLoader 클래스 생성)**

**구현 단계**:

1. **SociodemographicLoader 클래스 생성** (우선순위: P0)
   - 파일: `src/analysis/hybrid_choice_model/data_integration/sociodemographic_loader.py`
   - 기능: 원본 Excel 로드, 변수 선택, 전처리

2. **HybridDataIntegrator 수정** (우선순위: P1)
   - 사회인구학적 데이터 통합 기능 추가
   - 3-way merge: DCE + SEM + Sociodemographic

3. **테스트 코드 수정** (우선순위: P1)
   - `create_synthetic_sociodemographics()` → 실제 데이터 로드로 대체
   - 통합 테스트 추가

4. **문서화** (우선순위: P2)
   - 사용 예시 추가
   - 변수 코딩 문서화

---

## 🔧 구현 예시

### **사용 예시**

```python
from src.analysis.hybrid_choice_model.data_integration import (
    SociodemographicLoader,
    HybridDataIntegrator
)

# 1. 사회인구학적 데이터 로드
sociodem_loader = SociodemographicLoader(
    raw_data_path="data/raw/Sugar_substitue_Raw data_250730.xlsx"
)
sociodem_data = sociodem_loader.load_data()
sociodem_data = sociodem_loader.preprocess_data(sociodem_data)

# 2. 요인 데이터 로드 (기존)
perceived_benefit = pd.read_csv("data/processed/survey/perceived_benefit_reversed.csv")
latent_var = perceived_benefit[indicator_cols].mean(axis=1).values

# 3. 데이터 병합
merged_data = sociodem_data.copy()
merged_data['latent_var'] = latent_var

# 4. 구조모델 추정
from src.analysis.hybrid_choice_model.iclv_models import estimate_structural_model

results = estimate_structural_model(
    merged_data,
    merged_data['latent_var'].values,
    sociodemographics=['age_std', 'gender', 'income_std']
)

print(f"R²: {results['r_squared']:.4f}")
print(f"회귀계수: {results['gamma']}")
```

---

## ✅ 결론

### **기존 시스템 활용 가능성: 50%**

| 항목 | 상태 |
|------|------|
| **원본 데이터** | ✅ 존재 (Excel 파일) |
| **데이터 로더** | ❌ 없음 (새로 구현 필요) |
| **전처리 로직** | ⚠️ 부분 (참고 가능) |
| **통합 로직** | ⚠️ 부분 (확장 필요) |

### **즉시 조치 필요**

1. **SociodemographicLoader 클래스 구현** (최우선)
2. **HybridDataIntegrator 확장** (높은 우선순위)
3. **테스트 코드 수정** (중간 우선순위)

### **예상 작업량**

- **SociodemographicLoader**: 0.5일
- **HybridDataIntegrator 수정**: 0.5일
- **테스트 및 검증**: 0.5일
- **총 예상 시간**: 1.5일

---

**사회인구학적 데이터 통합 분석이 완료되었습니다!** 📊

