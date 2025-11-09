# LL (Choice) 값 도출 및 CSV 저장 구현 계획

**날짜**: 2025-11-09  
**목적**: 선택모델만의 로그우도(LL (Choice))를 계산하여 CSV에 저장  
**현재 상태**: LL (Choice) = N/A

---

## 📋 목차

1. [LL (Choice)의 의미](#1-ll-choice의-의미)
2. [현재 코드 구조 분석](#2-현재-코드-구조-분석)
3. [구현 방법](#3-구현-방법)
4. [수정이 필요한 파일](#4-수정이-필요한-파일)
5. [구현 단계](#5-구현-단계)
6. [예상 소요 시간](#6-예상-소요-시간)

---

## 1. LL (Choice)의 의미

### **정의**

**LL (Choice)**: 선택모델만의 로그우도

- **포함**: 선택모델 파라미터 (β, intercept, λ)
- **제외**: 측정모델 (ζ, τ), 구조모델 (γ)

### **수식**

**전체 ICLV 모델**:
```
LL (full) = Σᵢ log[(1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)]
```

**선택모델만**:
```
LL (choice) = Σᵢ Σₜ log[P(Choiceᵢₜ|Xᵢₜ, LVᵢ)]
```

여기서:
- `i`: 개인 인덱스
- `t`: 선택 상황 인덱스
- `LVᵢ`: 개인 i의 잠재변수 (추정된 값 사용)

### **용도**

1. **모델 비교**: ICLV vs. 일반 선택모델
2. **적합도 평가**: 선택모델 부분의 설명력
3. **잠재변수 효과**: LL (full) - LL (choice) = 측정모델 + 구조모델 기여도

---

## 2. 현재 코드 구조 분석

### **2.1 현재 LL 계산 구조**

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

#### **전체 모델 LL 계산**

**함수**: `_joint_log_likelihood()` (698-767행)

```python
def _joint_log_likelihood(self, params, measurement_model, 
                         structural_model, choice_model):
    """
    결합 로그우도 계산
    log L ≈ Σᵢ log[(1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)]
    """
    # 파라미터 분해
    param_dict = self._unpack_parameters(...)
    
    # 개인별 LL 계산
    for ind_id in individual_ids:
        person_ll = self._compute_individual_likelihood(...)
        total_ll += person_ll
    
    return total_ll
```

#### **개인별 LL 계산**

**함수**: `_compute_individual_likelihood()` (636-696행)

```python
def _compute_individual_likelihood(self, ind_id, ind_data, ind_draws,
                                   param_dict, measurement_model,
                                   structural_model, choice_model):
    """개인별 우도 계산"""
    draw_lls = []
    
    for draw in ind_draws:
        # 1. 구조모델: LV = γ*X + η
        lv = structural_model.predict(ind_data, param_dict['structural'], draw)
        
        # 2. 측정모델 우도: P(Indicators|LV)
        ll_measurement = measurement_model.log_likelihood(...)
        
        # 3. 선택모델 우도: P(Choice|X, LV)
        choice_set_lls = []
        for idx in range(len(ind_data)):
            ll_choice_t = choice_model.log_likelihood(...)  # ← 여기!
            choice_set_lls.append(ll_choice_t)
        ll_choice = sum(choice_set_lls)
        
        # 4. 구조모델 우도: P(LV|X)
        ll_structural = structural_model.log_likelihood(...)
        
        # 5. 결합 로그우도
        draw_ll = ll_measurement + ll_choice + ll_structural
        draw_lls.append(draw_ll)
    
    # logsumexp로 평균
    person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))
    return person_ll
```

### **2.2 선택모델 LL 계산**

**파일**: `src/analysis/hybrid_choice_model/iclv_models/choice_equations.py`

**클래스**: `BinaryProbitChoice`

**함수**: `log_likelihood()` (98-176행)

```python
def log_likelihood(self, data, lv, params):
    """
    선택모델 로그우도
    P(Choice|X, LV) = Φ(V) if choice=1, 1-Φ(V) if choice=0
    V = intercept + β*X + λ*LV
    """
    intercept = params['intercept']
    beta = params['beta']
    lambda_lv = params['lambda']
    
    # 선택 속성 추출
    X = data[self.choice_attributes].values
    choice = data['choice'].values
    
    # 효용 계산
    V = intercept + X @ beta + lambda_lv * lv_array
    
    # 확률 계산
    prob_yes = norm.cdf(V)
    
    # 로그우도
    ll = np.sum(choice * np.log(prob_yes) + 
                (1 - choice) * np.log(1 - prob_yes))
    
    return ll
```

---

## 3. 구현 방법

### **방법 1: 추정된 파라미터로 선택모델 LL 재계산** (권장)

#### **개념**

1. 전체 ICLV 모델 추정 완료 후
2. 추정된 파라미터 사용:
   - 선택모델 파라미터: β, intercept, λ
   - 잠재변수: 각 개인의 평균 LV (Halton draws 평균)
3. 선택모델만의 LL 계산

#### **장점**

- ✅ 기존 코드 재사용 가능
- ✅ 구현 간단
- ✅ 추가 추정 불필요

#### **단점**

- ⚠️ 잠재변수를 어떻게 정의할지 결정 필요
  - 옵션 A: Halton draws 평균
  - 옵션 B: 구조모델 예측값 (γ*X)
  - 옵션 C: 사후 평균 (posterior mean)

---

### **방법 2: 선택모델만 별도 추정**

#### **개념**

1. 잠재변수를 고정 (추정된 값 사용)
2. 선택모델 파라미터만 재추정
3. 재추정된 파라미터로 LL 계산

#### **장점**

- ✅ 선택모델의 "순수한" 적합도
- ✅ 잠재변수 효과 명확히 분리

#### **단점**

- ❌ 추가 추정 필요 (시간 소요)
- ❌ 구현 복잡
- ❌ 해석 복잡 (두 가지 β 값 존재)

---

### **방법 3: 추정 과정에서 LL (Choice) 기록**

#### **개념**

1. `_compute_individual_likelihood()` 함수 수정
2. `ll_choice` 값을 별도로 누적
3. 최종 LL (Choice) 반환

#### **장점**

- ✅ 정확한 LL (Choice) 값
- ✅ 추가 계산 불필요

#### **단점**

- ❌ 기존 코드 수정 필요
- ❌ 재추정 필요 (이미 완료된 추정 무효화)

---

## 4. 수정이 필요한 파일

### **방법 1 구현 시** (권장)

#### **4.1 새 파일 생성**

**파일**: `scripts/calculate_ll_choice.py` (새로 생성)

**기능**:
- 추정 결과 로드
- 잠재변수 계산
- 선택모델 LL 계산
- 결과 출력

**예상 코드 구조**:
```python
# 1. 추정 결과 로드
results = load_estimation_results()
params = results['parameters']

# 2. 데이터 로드
data = pd.read_csv('data/processed/iclv/integrated_data.csv')

# 3. 잠재변수 계산
lv_values = calculate_latent_variables(data, params)

# 4. 선택모델 LL 계산
ll_choice = calculate_choice_ll(data, lv_values, params['choice'])

# 5. 출력
print(f"LL (Choice): {ll_choice:.2f}")
```

#### **4.2 기존 파일 수정**

**파일**: `scripts/test_iclv_full_data.py`

**수정 위치**: 330-352행 (Estimation statistics 섹션)

**수정 내용**:
```python
# LL (Choice) 계산
ll_choice = calculate_choice_ll_from_results(results, data)

# Estimation statistics 추가
stats_list = [
    ...
    {'Coefficient': 'BIC', 'Estimate': f"{results['bic']:.2f}",
     'Std. Err.': 'LL (Choice)', 'P. Value': f"{ll_choice:.2f}"}  # ← 수정
]
```

#### **4.3 유틸리티 함수 추가**

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**추가 함수**:

```python
def calculate_choice_log_likelihood(self, params_dict, lv_method='posterior_mean'):
    """
    선택모델만의 로그우도 계산
    
    Args:
        params_dict: 추정된 파라미터 딕셔너리
        lv_method: 잠재변수 계산 방법
            - 'posterior_mean': 사후 평균 (Halton draws 평균)
            - 'structural': 구조모델 예측값 (γ*X)
            - 'zero': LV = 0 (잠재변수 효과 제거)
    
    Returns:
        선택모델 로그우도
    """
    # 1. 잠재변수 계산
    if lv_method == 'posterior_mean':
        lv_values = self._calculate_posterior_mean_lv(params_dict)
    elif lv_method == 'structural':
        lv_values = self._calculate_structural_lv(params_dict)
    elif lv_method == 'zero':
        lv_values = np.zeros(len(self.data))
    
    # 2. 선택모델 LL 계산
    choice_model = BinaryProbitChoice(self.config.choice)
    
    total_ll = 0.0
    individual_ids = self.data[self.config.individual_id_column].unique()
    
    for ind_id in individual_ids:
        ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
        ind_lv = lv_values[self.data[self.config.individual_id_column] == ind_id].mean()
        
        # 개인의 모든 선택 상황에 대한 LL
        for idx in range(len(ind_data)):
            ll_t = choice_model.log_likelihood(
                ind_data.iloc[idx:idx+1],
                ind_lv,
                params_dict['choice']
            )
            total_ll += ll_t
    
    return total_ll

def _calculate_posterior_mean_lv(self, params_dict):
    """사후 평균 잠재변수 계산"""
    draws = self.halton_generator.get_draws()
    individual_ids = self.data[self.config.individual_id_column].unique()
    
    lv_values = np.zeros(len(self.data))
    
    for i, ind_id in enumerate(individual_ids):
        ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
        ind_draws = draws[i, :]
        
        # 각 draw에 대한 LV 계산
        lv_draws = []
        for draw in ind_draws:
            lv = self.structural_model.predict(
                ind_data, params_dict['structural'], draw
            )
            lv_draws.append(lv)
        
        # 평균
        lv_mean = np.mean(lv_draws)
        
        # 개인의 모든 관측치에 동일한 LV 할당
        mask = self.data[self.config.individual_id_column] == ind_id
        lv_values[mask] = lv_mean
    
    return lv_values

def _calculate_structural_lv(self, params_dict):
    """구조모델 예측값 (γ*X)"""
    individual_ids = self.data[self.config.individual_id_column].unique()
    lv_values = np.zeros(len(self.data))
    
    for ind_id in individual_ids:
        ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
        
        # LV = γ*X (오차항 제외)
        lv = self.structural_model.predict(
            ind_data, params_dict['structural'], error_draw=0.0
        )
        
        mask = self.data[self.config.individual_id_column] == ind_id
        lv_values[mask] = lv
    
    return lv_values
```

---

## 5. 구현 단계

### **단계 1: 유틸리티 함수 추가** (30분)

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**작업**:
1. `calculate_choice_log_likelihood()` 함수 추가
2. `_calculate_posterior_mean_lv()` 함수 추가
3. `_calculate_structural_lv()` 함수 추가

**테스트**:
```python
# 간단한 테스트
estimator = SimultaneousEstimator(config)
estimator.data = data
ll_choice = estimator.calculate_choice_log_likelihood(params_dict)
print(f"LL (Choice): {ll_choice:.2f}")
```

---

### **단계 2: 계산 스크립트 생성** (20분)

**파일**: `scripts/calculate_ll_choice.py` (새로 생성)

**작업**:
1. 추정 결과 로드
2. LL (Choice) 계산
3. 결과 출력 및 저장

**예상 코드**:
```python
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator
# ... (import 생략)

# 1. 데이터 로드
data = pd.read_csv(project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv')

# 2. 설정 및 모델 생성
config = ICLVConfig(...)
estimator = SimultaneousEstimator(config)
estimator.data = data

# 3. 추정 결과 로드
results_df = pd.read_csv(project_root / 'results' / 'iclv_full_data_results.csv')
params_dict = extract_params_from_csv(results_df)

# 4. LL (Choice) 계산
ll_choice = estimator.calculate_choice_log_likelihood(params_dict, lv_method='posterior_mean')

print(f"LL (Choice): {ll_choice:.2f}")

# 5. 파일에 저장
with open(project_root / 'results' / 'll_choice.txt', 'w') as f:
    f.write(f"{ll_choice:.2f}")
```

---

### **단계 3: CSV 업데이트 스크립트 수정** (10분)

**파일**: `scripts/update_csv_with_ll_choice.py` (새로 생성)

**작업**:
1. `ll_choice.txt` 파일 읽기
2. CSV 파일 업데이트
3. 결과 확인

**예상 코드**:
```python
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent

# 1. LL (Choice) 값 읽기
with open(project_root / 'results' / 'll_choice.txt', 'r') as f:
    ll_choice = f.read().strip()

# 2. CSV 로드
csv_file = project_root / 'results' / 'iclv_full_data_results.csv'
df = pd.read_csv(csv_file)

# 3. BIC 행 찾기
bic_row_idx = df[df['Coefficient'] == 'BIC'].index[0]

# 4. LL (Choice) 값 업데이트
df.loc[bic_row_idx, 'P. Value'] = ll_choice

# 5. 저장
df.to_csv(csv_file, index=False, encoding='utf-8-sig')

print(f"✅ LL (Choice) 값 추가 완료: {ll_choice}")
```

---

### **단계 4: 메인 스크립트 수정** (10분)

**파일**: `scripts/test_iclv_full_data.py`

**수정 위치**: 330-352행

**작업**:
1. LL (Choice) 계산 로직 추가
2. Estimation statistics에 포함

**수정 코드**:
```python
# LL (Choice) 계산
try:
    ll_choice = estimator.calculate_choice_log_likelihood(
        results['parameters'], 
        lv_method='posterior_mean'
    )
    ll_choice_str = f"{ll_choice:.2f}"
except Exception as e:
    print(f"   ⚠️  LL (Choice) 계산 실패: {e}")
    ll_choice_str = 'N/A'

# Estimation statistics 추가
stats_list = [
    {'Coefficient': '', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
    {'Coefficient': 'Estimation statistics', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
    {'Coefficient': 'Iterations', 'Estimate': results.get('n_iterations', 'N/A'),
     'Std. Err.': 'LL (start)', 'P. Value': initial_ll},
    {'Coefficient': 'AIC', 'Estimate': f"{results['aic']:.2f}",
     'Std. Err.': 'LL (final, whole model)', 'P. Value': f"{results['log_likelihood']:.2f}"},
    {'Coefficient': 'BIC', 'Estimate': f"{results['bic']:.2f}",
     'Std. Err.': 'LL (Choice)', 'P. Value': ll_choice_str}  # ← 수정
]
```

---

### **단계 5: 테스트 및 검증** (30분)

**작업**:
1. `calculate_ll_choice.py` 실행
2. 결과 확인
3. CSV 업데이트 확인
4. 값의 타당성 검증

**검증 기준**:
- LL (Choice) < LL (final) ✅ (선택모델만이므로 더 작아야 함)
- LL (Choice)가 합리적인 범위 ✅ (예: -3000 ~ -6000)

---

## 6. 예상 소요 시간

| 단계 | 작업 | 예상 시간 |
|------|------|----------|
| 1 | 유틸리티 함수 추가 | 30분 |
| 2 | 계산 스크립트 생성 | 20분 |
| 3 | CSV 업데이트 스크립트 | 10분 |
| 4 | 메인 스크립트 수정 | 10분 |
| 5 | 테스트 및 검증 | 30분 |
| **합계** | | **1.5-2시간** |

---

## 7. 대안: 간소화된 방법

### **방법: 선택모델 LL만 빠르게 계산**

**장점**: 
- ✅ 매우 빠름 (10-20분)
- ✅ 기존 코드 거의 수정 없음

**단점**:
- ⚠️ 잠재변수를 0으로 가정 (λ 효과 제거)
- ⚠️ 정확한 LL (Choice)가 아님

**구현**:
```python
# scripts/quick_ll_choice.py
import pandas as pd
import numpy as np
from scipy.stats import norm

# 데이터 로드
data = pd.read_csv('data/processed/iclv/integrated_data.csv')

# 파라미터 로드
results_df = pd.read_csv('results/iclv_full_data_results.csv')
beta_price = results_df[results_df['Coefficient'] == 'β_price']['Estimate'].values[0]
beta_health = results_df[results_df['Coefficient'] == 'β_health_label']['Estimate'].values[0]
intercept = results_df[results_df['Coefficient'] == 'β_Intercept']['Estimate'].values[0]

# 효용 계산 (LV = 0 가정)
V = intercept + beta_price * data['price'] + beta_health * data['health_label']

# 확률
prob_yes = norm.cdf(V)
prob_yes = np.clip(prob_yes, 1e-10, 1 - 1e-10)

# LL
ll_choice = np.sum(
    data['choice'] * np.log(prob_yes) +
    (1 - data['choice']) * np.log(1 - prob_yes)
)

print(f"LL (Choice, LV=0): {ll_choice:.2f}")
```

---

## 8. 권장 사항

### **최종 권장: 방법 1 (추정된 파라미터로 재계산)**

**이유**:
1. ✅ 정확한 LL (Choice) 값
2. ✅ 구현 난이도 적절
3. ✅ 소요 시간 합리적 (1.5-2시간)
4. ✅ 향후 재사용 가능

### **잠재변수 계산 방법: posterior_mean**

**이유**:
1. ✅ ICLV 모델의 표준 방법
2. ✅ 불확실성 반영 (Halton draws 평균)
3. ✅ 해석 명확

---

## 9. 요약

### **수정이 필요한 파일**

| 파일 | 작업 | 난이도 |
|------|------|--------|
| `src/.../simultaneous_estimator_fixed.py` | 유틸리티 함수 3개 추가 | 중간 |
| `scripts/calculate_ll_choice.py` | 새 파일 생성 | 쉬움 |
| `scripts/update_csv_with_ll_choice.py` | 새 파일 생성 | 쉬움 |
| `scripts/test_iclv_full_data.py` | 10줄 수정 | 쉬움 |

### **핵심 구현 내용**

1. **잠재변수 계산**: Halton draws 평균 사용
2. **선택모델 LL**: 기존 `choice_model.log_likelihood()` 재사용
3. **CSV 업데이트**: BIC 행의 P. Value 열에 LL (Choice) 추가

### **예상 결과**

```
행 43: BIC, 11790.69, LL (Choice), -4500.00 (예상값)
```

---

**보고서 작성일**: 2025-11-09  
**상태**: 구현 대기 중

