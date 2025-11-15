# SEMEstimator 계산 로직 vs 요구사항 비교 분석

## 📋 요구사항

### 1. 측정모델
- **요인적재량 (Factor Loadings)**: 각 잠재변수의 지표들에 대한 λ
- **측정 오차분산 (Measurement Error Variance)**: 각 지표의 오차분산 θ

### 2. 구조모델
- **경로계수 (Path Coefficients)**: HC→PB, PB→PI 관계 (2개)
- **구조 오차분산 (Structural Error Variance)**: 내생 잠재변수의 오차분산 ψ
- **외생 잠재변수 분산 (Exogenous LV Variance)**: HC의 분산 φ

### 3. 요인점수 계산
- **Factor Score Regression**: 요인점수 회귀하여 잠재변수 점수 계산

---

## 🔍 현재 SEMEstimator 계산 로직

### 1. 모델 스펙 생성 (`_create_sem_spec()`)

```python
# Measurement Model (CFA)
health_concern =~ q6 + q7 + q8 + q9 + q10 + q11
perceived_benefit =~ q12 + q13 + q14 + q15 + q16 + q17
purchase_intention =~ q18 + q19 + q20

# Structural Model (Path Analysis)
perceived_benefit ~ health_concern
purchase_intention ~ perceived_benefit
```

### 2. semopy 추정 (`fit()`)

```python
# 기존 SemopyAnalyzer 재사용
results = self.analyzer.fit_model(data, model_spec)
self.model = self.analyzer.model
```

### 3. 파라미터 추출 (`fit()`)

```python
params = self.model.inspect()
loadings = params[params['op'] == '=~'].copy()  # 요인적재량
paths = params[params['op'] == '~'].copy()      # 경로계수
```

**⚠️ 문제**: semopy는 `=~` 대신 `~`를 사용하므로 loadings가 비어있음!

### 4. 요인점수 추출 (`_extract_factor_scores()`)

```python
# 방법 1: semopy.predict_factors() (우선)
factor_scores_df = self.model.predict_factors(data)

# 방법 2: Bartlett 수동 계산 (fallback)
# Factor Score = (Λ'Λ)^(-1) Λ' X
```

---

## ❌ 현재 문제점

### 1. **파라미터 추출 오류**

| 파라미터 | 요구사항 | 현재 SEMEstimator | 상태 |
|---------|---------|------------------|------|
| **요인적재량** | ✅ 필요 | ❌ `op == '=~'` 필터링 실패 | **누락** |
| **측정 오차분산** | ✅ 필요 | ❌ 추출 안 함 | **누락** |
| **경로계수** | ✅ 필요 | ⚠️ `op == '~'` (loadings 포함) | **혼재** |
| **구조 오차분산** | ✅ 필요 | ❌ 추출 안 함 | **누락** |
| **외생 LV 분산** | ✅ 필요 | ❌ 추출 안 함 | **누락** |

### 2. **semopy `inspect()` 결과 구조**

실제 semopy는 다음과 같이 반환합니다:

```
   lval  op rval  Estimate  Std. Err   z-value   p-value
0   LV2   ~  LV1 -0.765042  0.605082 -1.264361  0.206101  # 구조 경로
1    q1   ~  LV1  1.000000         -         -         -  # 요인적재량 (고정)
2    q2   ~  LV1 -1.134824  0.921292 -1.231774  0.218034  # 요인적재량
3    q3   ~  LV1 -0.566684  0.503682 -1.125082  0.260554  # 요인적재량
7   LV1  ~~  LV1  0.321094  0.318901  1.006874  0.313995  # LV 분산
9    q1  ~~   q1  1.624488  0.362699  4.478885  0.000008  # 측정 오차분산
```

**핵심 발견**:
- ✅ 요인적재량: `op == '~'` AND `rval`이 잠재변수
- ✅ 경로계수: `op == '~'` AND `lval`, `rval` 모두 잠재변수
- ✅ 측정 오차분산: `op == '~~'` AND `lval == rval` AND `lval`이 관측변수
- ✅ 구조 오차분산: `op == '~~'` AND `lval == rval` AND `lval`이 내생 잠재변수
- ✅ 외생 LV 분산: `op == '~~'` AND `lval == rval` AND `lval`이 외생 잠재변수

---

## ✅ 필요한 수정사항

### 1. **파라미터 추출 로직 수정**

```python
def _extract_parameters(self, measurement_model, structural_model):
    """
    모든 파라미터 추출 (측정모델 + 구조모델)
    
    Returns:
        {
            'loadings': pd.DataFrame,           # 요인적재량
            'measurement_errors': pd.DataFrame, # 측정 오차분산
            'paths': pd.DataFrame,              # 경로계수
            'structural_errors': pd.DataFrame,  # 구조 오차분산
            'lv_variances': pd.DataFrame        # 잠재변수 분산
        }
    """
    params = self.model.inspect()
    
    # 잠재변수 목록
    latent_vars = list(measurement_model.configs.keys())
    
    # 1. 요인적재량: op == '~' AND rval이 잠재변수
    loadings = params[
        (params['op'] == '~') & 
        (params['rval'].isin(latent_vars))
    ].copy()
    
    # 2. 경로계수: op == '~' AND lval, rval 모두 잠재변수
    paths = params[
        (params['op'] == '~') & 
        (params['lval'].isin(latent_vars)) &
        (params['rval'].isin(latent_vars))
    ].copy()
    
    # 3. 측정 오차분산: op == '~~' AND lval == rval AND lval이 관측변수
    measurement_errors = params[
        (params['op'] == '~~') & 
        (params['lval'] == params['rval']) &
        (~params['lval'].isin(latent_vars))
    ].copy()
    
    # 4. 구조 오차분산: op == '~~' AND lval == rval AND lval이 내생 잠재변수
    endogenous_lvs = [structural_model.endogenous_lv]
    if structural_model.is_hierarchical:
        for path in structural_model.hierarchical_paths:
            endogenous_lvs.append(path['target'])
    
    structural_errors = params[
        (params['op'] == '~~') & 
        (params['lval'] == params['rval']) &
        (params['lval'].isin(endogenous_lvs))
    ].copy()
    
    # 5. 외생 잠재변수 분산: op == '~~' AND lval == rval AND lval이 외생 잠재변수
    exogenous_lvs = structural_model.exogenous_lvs
    lv_variances = params[
        (params['op'] == '~~') & 
        (params['lval'] == params['rval']) &
        (params['lval'].isin(exogenous_lvs))
    ].copy()
    
    return {
        'loadings': loadings,
        'measurement_errors': measurement_errors,
        'paths': paths,
        'structural_errors': structural_errors,
        'lv_variances': lv_variances
    }
```

### 2. **`fit()` 메서드 수정**

```python
def fit(self, data, measurement_model, structural_model):
    # ... (기존 코드)
    
    # 5. 파라미터 추출 (수정)
    extracted_params = self._extract_parameters(measurement_model, structural_model)
    
    return {
        'model': self.model,
        'factor_scores': factor_scores,
        'params': params,  # 전체 파라미터
        'loadings': extracted_params['loadings'],
        'measurement_errors': extracted_params['measurement_errors'],
        'paths': extracted_params['paths'],
        'structural_errors': extracted_params['structural_errors'],
        'lv_variances': extracted_params['lv_variances'],
        'fit_indices': results.get('fit_indices', {}),
        'log_likelihood': log_likelihood
    }
```

---

## 📊 수정 후 예상 결과

| 파라미터 | 추출 방법 | 예시 |
|---------|---------|------|
| **요인적재량** | `op == '~'` & `rval` in LVs | `q6 ~ health_concern: 1.000` |
| **측정 오차분산** | `op == '~~'` & `lval == rval` & 관측변수 | `q6 ~~ q6: 0.523` |
| **경로계수** | `op == '~'` & `lval`, `rval` in LVs | `perceived_benefit ~ health_concern: 0.456` |
| **구조 오차분산** | `op == '~~'` & `lval == rval` & 내생 LV | `perceived_benefit ~~ perceived_benefit: 0.789` |
| **외생 LV 분산** | `op == '~~'` & `lval == rval` & 외생 LV | `health_concern ~~ health_concern: 1.234` |

---

## 🎯 결론

**현재 SEMEstimator는 요구사항의 일부만 충족**:
- ✅ 요인점수 계산: 완벽 구현
- ⚠️ 파라미터 추출: 불완전 (loadings 필터링 오류, 분산 미추출)

**필요한 작업**:
1. `_extract_parameters()` 메서드 추가
2. `fit()` 반환값에 분산 파라미터 추가
3. 테스트 스크립트에서 검증

