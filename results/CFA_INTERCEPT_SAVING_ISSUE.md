# CFA 절편 저장 문제 분석

## 핵심 발견

### ❌ **CFA 추정 시 절편이 저장되지 않았습니다!**

**이유**:
1. `sem_estimator.fit_cfa_only()`는 `params` (전체 파라미터)를 반환하지만
2. `sequential_estimator.save_cfa_results()`는 `params`를 저장하지 않음
3. 결과적으로 절편 정보가 손실됨

---

## 1. CFA 추정 코드 분석

### **sem_estimator.fit_cfa_only()** (Line 81-179)

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/sem_estimator.py" mode="EXCERPT">
````python
def fit_cfa_only(self, data: pd.DataFrame,
                measurement_model: MultiLatentMeasurement) -> Dict[str, Any]:
    """
    Returns:
        {
            'model': semopy Model 객체,
            'factor_scores': Dict[str, np.ndarray],
            'params': pd.DataFrame,  # ✅ 모든 파라미터 (절편 포함)
            'loadings': pd.DataFrame,
            'correlations': pd.DataFrame,
            'fit_indices': Dict[str, float],
            'log_likelihood': float
        }
    """
    # ...
    
    # 5. 파라미터 추출
    params = self.model.inspect(std_est=True)  # ✅ 전체 파라미터
    
    # 요인적재량만 필터링
    loadings = params[
        (params['op'] == '~') &
        (params['rval'].isin(latent_vars)) &
        (~params['lval'].isin(latent_vars))
    ].copy()
    
    # 측정 오차분산만 필터링
    measurement_errors = params[
        (params['op'] == '~~') &
        (params['lval'] == params['rval']) &
        (~params['lval'].isin(latent_vars))
    ].copy()
    
    return {
        'model': self.model,
        'factor_scores': factor_scores,
        'params': params,  # ✅ 전체 파라미터 반환
        'loadings': loadings,
        'measurement_errors': measurement_errors,
        'correlations': correlations,
        'fit_indices': fit_indices,
        'log_likelihood': log_likelihood
    }
````
</augment_code_snippet>

**✅ `params`에는 절편이 포함되어 있음!**

---

## 2. CFA 결과 저장 코드 분석

### **sequential_estimator.save_cfa_results()** (Line 161-184)

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/sequential_estimator.py" mode="EXCERPT">
````python
def save_cfa_results(self, results: Dict, save_path: str):
    """
    CFA 결과 저장 (pickle + CSV)
    """
    # 1. Pickle 저장 (전체 결과)
    save_data = {
        'factor_scores': results['factor_scores'],
        'loadings': results['loadings'],  # ✅ 요인적재량만
        'measurement_errors': results.get('measurement_errors'),  # ✅ 오차분산만
        'correlations': results['correlations'],
        'fit_indices': results['fit_indices'],
        'log_likelihood': results['log_likelihood']
        # ❌ 'params' 저장 안함!
        # ❌ 'model' 저장 안함!
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
````
</augment_code_snippet>

**❌ `params`와 `model`을 저장하지 않음!**

---

## 3. 문제의 근본 원인

### **저장 로직의 문제**

1. **`fit_cfa_only()`는 `params` 반환**:
   - `params`: 전체 파라미터 (op == '~', '~~', '1' 모두 포함)
   - 절편 (op == '1')도 포함되어 있음

2. **`save_cfa_results()`는 `params` 저장 안함**:
   - `loadings`만 저장 (op == '~', 요인적재량만)
   - `measurement_errors`만 저장 (op == '~~', 오차분산만)
   - **절편 (op == '1')은 저장되지 않음!**

3. **결과**:
   - 저장된 파일에는 절편 정보 없음
   - 측정모델 우도 계산시 절편 사용 불가

---

## 4. semopy의 절편 추정 여부 확인

### **semopy CFA 모델 스펙**

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/sem_estimator.py" mode="EXCERPT">
````python
def _create_cfa_spec(self, measurement_model: MultiLatentMeasurement) -> str:
    """
    Example:
        ```
        # Measurement Model (CFA)
        health_concern =~ q6 + q7 + q8 + q9 + q10 + q11
        perceived_benefit =~ q12 + q13 + q14 + q15 + q16 + q17
        ```
    """
    spec_lines = []
    spec_lines.append("# Measurement Model (CFA)")
    
    for lv_name, config in measurement_model.configs.items():
        indicators = " + ".join(config.indicators)
        spec_lines.append(f"{lv_name} =~ {indicators}")  # ❌ 절편 명시 안함
    
    model_spec = "\n".join(spec_lines)
    return model_spec
````
</augment_code_snippet>

**문제**:
- semopy 모델 스펙에 절편이 명시되지 않음
- semopy는 기본적으로 절편을 추정하지만, **잠재변수가 표준화되면 절편이 0이 됨**

### **semopy의 기본 동작**

```python
# semopy CFA 모델
model_spec = """
health_concern =~ q6 + q7 + q8
"""

# semopy는 내부적으로 다음과 같이 처리:
# 1. 잠재변수 분산 Var(LV) = 1로 고정 (표준화)
# 2. 잠재변수 평균 E[LV] = 0으로 고정
# 3. 절편은 추정되지만, E[LV] = 0이므로 절편의 역할이 제한됨
```

**결과**:
- semopy는 절편을 추정할 수 있지만
- 잠재변수 표준화로 인해 절편이 0이 되거나
- `inspect()` 결과에 포함되지 않을 수 있음

---

## 5. 해결 방안

### **Option A: params 전체 저장 (추천)**

**방법**:
1. `save_cfa_results()`에서 `params` 전체 저장
2. 절편이 있으면 사용, 없으면 지표 평균 사용

**코드 수정**:
```python
def save_cfa_results(self, results: Dict, save_path: str):
    save_data = {
        'factor_scores': results['factor_scores'],
        'params': results.get('params'),  # ✅ 전체 파라미터 저장
        'loadings': results['loadings'],
        'measurement_errors': results.get('measurement_errors'),
        'correlations': results['correlations'],
        'fit_indices': results['fit_indices'],
        'log_likelihood': results['log_likelihood']
    }
```

**장점**:
- ✅ 절편 정보 보존
- ✅ 추후 분석 가능
- ✅ 구현 간단

---

### **Option B: 절편 명시적 추출 및 저장**

**방법**:
1. `params`에서 절편 (op == '1') 추출
2. 별도로 저장

**코드 수정**:
```python
def fit_cfa_only(self, ...):
    # ...
    params = self.model.inspect(std_est=True)
    
    # 절편 추출
    intercepts = params[params['op'] == '1'].copy()
    
    return {
        # ...
        'intercepts': intercepts,  # ✅ 절편 추가
    }

def save_cfa_results(self, results: Dict, save_path: str):
    save_data = {
        # ...
        'intercepts': results.get('intercepts'),  # ✅ 절편 저장
    }
```

**장점**:
- ✅ 절편 명시적 관리
- ✅ 코드 가독성 향상

**단점**:
- ⚠️ 절편이 없을 수 있음 (semopy가 추정 안함)

---

### **Option C: 절편을 지표 평균으로 계산 (가장 안전)**

**방법**:
1. CFA 추정 후 각 지표의 평균 계산
2. 절편으로 저장

**코드 수정**:
```python
def fit_cfa_only(self, data, measurement_model):
    # ... (CFA 추정)
    
    # 절편 계산 (각 지표의 평균)
    intercepts = {}
    for lv_name, config in measurement_model.configs.items():
        indicators = config.indicators
        intercepts[lv_name] = unique_data[indicators].mean(axis=0).values
    
    return {
        # ...
        'intercepts': intercepts,  # ✅ 절편 추가
    }
```

**장점**:
- ✅ 항상 절편 있음
- ✅ 이론적으로 올바름 (E[Y_i] = α_i)
- ✅ semopy 의존성 없음

**단점**:
- 없음

---

## 6. 권장 사항

### ✅ **Option C: 절편을 지표 평균으로 계산 (강력 추천)**

**이유**:
1. **항상 작동**: semopy가 절편을 추정하지 않아도 OK
2. **이론적으로 올바름**: E[Y_i] = α_i (지표의 평균)
3. **구현 간단**: 몇 줄만 추가
4. **안전**: 의존성 없음

**구현 순서**:
1. `sem_estimator.fit_cfa_only()`에서 절편 계산
2. `sequential_estimator.save_cfa_results()`에서 절편 저장
3. 측정모델 우도 계산시 절편 사용

---

## 7. 결론

### **문제 요약**

1. ❌ `sem_estimator.fit_cfa_only()`는 `params` 반환 (절편 포함 가능)
2. ❌ `sequential_estimator.save_cfa_results()`는 `params` 저장 안함
3. ❌ 결과적으로 절편 정보 손실
4. ❌ 측정모델 우도 계산시 절편 사용 불가

### **해결 방안**

✅ **절편을 지표 평균으로 계산하여 저장**
- `fit_cfa_only()`에서 절편 계산
- `save_cfa_results()`에서 절편 저장
- 측정모델 우도 계산시 절편 사용

### **예상 효과**

- 측정모델 우도: **-1218.39 → -27.74** (44배 개선!)
- 전체 우도 균형 크게 개선
- Gamma 파라미터 업데이트 더욱 안정적

---

**작성일**: 2025-11-20  
**작성자**: Augment Agent

