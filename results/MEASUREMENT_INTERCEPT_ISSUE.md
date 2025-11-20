# 측정모델 절편(Intercept) 문제 분석

## 핵심 발견

### ❌ **현재 측정모델에 절편이 없습니다!**

**문제**:
- CFA 결과에 절편 파라미터 없음 (op == '1' 개수: 0)
- 측정모델 우도 계산시 절편 사용 안함: `Y_pred = ζ * LV`
- 올바른 공식: `Y_pred = α + ζ * LV`

---

## 1. 현재 상황 확인

### **CFA 파라미터**

```python
# CFA 결과 확인
loadings_df['op'].value_counts()
# 결과:
# ~    38  (요인적재량만 있음)
# 1     0  (절편 없음!)
```

### **측정모델 우도 계산 코드**

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/gpu_measurement_equations.py" mode="EXCERPT">
````python
# 예측값: Y_pred = ζ * LV
y_pred = zeta[i] * lv_gpu  # (n_batch,)

# 잔차
residual = y_obs - y_pred  # (n_batch,)
````
</augment_code_snippet>

**문제**: 절편 α가 없음!

---

## 2. 절편이 없을 때의 문제

### **이론적 문제**

**현재 모델 (절편 없음)**:
```
Y_i = ζ_i * LV + ε_i
LV ~ N(0, 1) (표준화)

E[Y_i] = ζ_i * E[LV] = ζ_i * 0 = 0
```

**실제 데이터**:
```
지표(q6-q49): 1-5점 리커트
관측 평균: 3-4
```

**불일치**:
```
예측 평균: 0
관측 평균: 3-4
→ 큰 잔차 → 매우 낮은 우도!
```

### **실제 계산 예시 (지표 q6)**

**데이터**:
- q6 평균: **3.7607**
- q6 표준편차: 0.8569

**CFA 파라미터**:
- ζ = 1.0000
- σ² = 0.2161
- α = **없음**

**예측 (LV = 0.5)**:
```
Y_pred = ζ * LV = 1.0 * 0.5 = 0.5

residual = Y_obs - Y_pred = 3.7607 - 0.5 = 3.2607 (매우 큼!)

ll = -0.5 * log(2π * 0.2161) - 0.5 * (3.2607)² / 0.2161
   = -0.5 * log(1.356) - 0.5 * 10.63 / 0.2161
   = -0.153 - 24.60
   = -24.75 (매우 낮음!)
```

---

## 3. 왜 절편이 없는가?

### **semopy의 동작 방식**

**일반적인 CFA 모델**:
```
Y_i = α_i + ζ_i * LV + ε_i
```

**semopy의 표준화**:
- 잠재변수 분산 Var(LV) = 1로 고정
- 잠재변수 평균 E[LV] = 0으로 고정
- **절편 α_i는 추정되지만, E[LV] = 0이므로 절편의 역할이 제한됨**

### **문제의 근본 원인**

1. **CFA는 표준화된 LV 사용**:
   - E[LV] = 0, Var(LV) = 1
   - 절편 α_i는 지표의 평균을 나타냄
   - 하지만 semopy는 절편을 명시적으로 저장하지 않음

2. **동시추정에서도 표준화된 LV 사용**:
   - LV ~ N(0, 1) (Halton 샘플링)
   - 하지만 절편 α_i를 사용하지 않음
   - **스케일 불일치!**

---

## 4. 올바른 측정모델

### **이론적 공식**

```
Y_i = α_i + ζ_i * LV + ε_i

여기서:
  α_i: 절편 (지표 i의 평균)
  ζ_i: 요인적재량
  LV: 잠재변수 (표준화, E[LV] = 0, Var[LV] = 1)
  ε_i ~ N(0, σ²_i): 측정오차
```

### **기대값**

```
E[Y_i] = α_i + ζ_i * E[LV] + E[ε_i]
       = α_i + ζ_i * 0 + 0
       = α_i
```

**✅ 절편 α_i = 지표의 평균**

### **예측 계산 (절편 있음)**

**지표 q6, LV = 0.5**:
```
α = 3.7607 (q6의 평균)
ζ = 1.0000
LV = 0.5

Y_pred = α + ζ * LV = 3.7607 + 1.0 * 0.5 = 4.2607

residual = Y_obs - Y_pred = 3.7607 - 4.2607 = -0.5 (작음!)

ll = -0.5 * log(2π * 0.2161) - 0.5 * (-0.5)² / 0.2161
   = -0.153 - 0.58
   = -0.73 (훨씬 높음!)
```

**비교**:
- 절편 없음: ll = **-24.75**
- 절편 있음: ll = **-0.73**
- **차이: 24.02** (엄청난 차이!)

---

## 5. 해결 방안

### **Option A: 절편을 지표 평균으로 설정 (추천)**

**방법**:
1. 각 지표의 평균 계산
2. 측정모델 우도 계산시 절편 사용

**코드 수정**:
```python
# 절편 계산 (각 지표의 평균)
alpha = data[indicators].mean(axis=0).values  # (n_indicators,)

# 예측값 계산
y_pred = alpha[i] + zeta[i] * lv_gpu  # ✅ 절편 추가!

# 잔차
residual = y_obs - y_pred
```

**장점**:
- ✅ 구현 간단
- ✅ 이론적으로 올바름
- ✅ 우도 크게 개선
- ✅ 추가 파라미터 없음 (고정값)

**단점**:
- 없음

---

### **Option B: CFA에서 절편 명시적 추정**

**방법**:
1. semopy 모델 스펙에 절편 추가
2. 절편 파라미터 추출
3. 측정모델 우도 계산시 사용

**semopy 모델 스펙**:
```python
# 현재
model_spec = """
health_concern =~ q6 + q7 + q8 + q9 + q10 + q11
"""

# 수정 (절편 명시)
model_spec = """
health_concern =~ q6 + q7 + q8 + q9 + q10 + q11
q6 ~ 1
q7 ~ 1
...
"""
```

**장점**:
- ✅ 이론적으로 완벽
- ✅ CFA와 일관성

**단점**:
- ⚠️ CFA 재추정 필요
- ⚠️ 구현 복잡

---

### **Option C: 지표 중심화 (Centering)**

**방법**:
1. 각 지표를 평균 중심화: `Y_i_centered = Y_i - mean(Y_i)`
2. 중심화된 지표로 CFA 재추정
3. 측정모델 우도 계산시 중심화된 지표 사용

**공식**:
```
Y_i_centered = Y_i - α_i
Y_i_centered = ζ_i * LV + ε_i
```

**장점**:
- ✅ 절편 불필요
- ✅ 이론적으로 올바름

**단점**:
- ⚠️ 데이터 재전처리 필요
- ⚠️ CFA 재추정 필요
- ⚠️ 해석 복잡

---

## 6. 권장 사항

### ✅ **Option A: 절편을 지표 평균으로 설정 (강력 추천)**

**이유**:
1. **구현 간단**: 코드 몇 줄만 수정
2. **이론적으로 올바름**: E[Y_i] = α_i
3. **우도 크게 개선**: 지표당 평균 우도 -24.75 → -0.73
4. **추가 파라미터 없음**: 고정값 사용

**예상 효과**:
- 측정모델 우도: **-1218.39 → -27.74** (지표당 -32.06 → -0.73)
- 전체 우도 균형 크게 개선
- Gamma 파라미터 업데이트 더욱 안정적

---

## 7. 구현 계획

### **1단계: 절편 계산**

```python
# 각 지표의 평균 계산
def compute_intercepts(data: pd.DataFrame, indicators: List[str]) -> np.ndarray:
    """각 지표의 평균 계산 (절편)"""
    return data[indicators].mean(axis=0).values
```

### **2단계: 측정모델 우도 계산 수정**

```python
# gpu_measurement_equations.py 수정
def log_likelihood_batch(self, data_batch, latent_vars, params):
    zeta = cp.asarray(params['zeta'])
    sigma_sq = cp.asarray(params['sigma_sq'])
    alpha = cp.asarray(params['alpha'])  # ✅ 절편 추가
    
    for i in range(self.n_indicators):
        y_obs = data_gpu[:, i]
        
        # ✅ 절편 포함
        y_pred = alpha[i] + zeta[i] * lv_gpu
        
        residual = y_obs - y_pred
        ll_i = -0.5 * cp.log(2 * cp.pi * sigma_sq[i]) - 0.5 * (residual ** 2) / sigma_sq[i]
        ll_batch = ll_batch + ll_i
```

### **3단계: CFA 결과에 절편 추가**

```python
# sem_estimator.py 수정
def fit_cfa_only(self, data, measurement_model):
    # ... (기존 CFA 추정)
    
    # 절편 계산
    intercepts = {}
    for lv_name, config in measurement_model.configs.items():
        indicators = config.indicators
        intercepts[lv_name] = data[indicators].mean(axis=0).values
    
    return {
        'loadings': loadings,
        'measurement_errors': errors,
        'intercepts': intercepts,  # ✅ 절편 추가
        ...
    }
```

---

## 8. 결론

### **현재 문제**

❌ 측정모델에 절편이 없음
- CFA 결과: 절편 파라미터 없음
- 우도 계산: `Y_pred = ζ * LV` (절편 없음)
- 결과: 예측 평균 0 vs 관측 평균 3-4 → 큰 잔차 → 매우 낮은 우도

### **해결 방안**

✅ **절편을 지표 평균으로 설정**
- 각 지표의 평균을 절편으로 사용
- 우도 계산: `Y_pred = α + ζ * LV`
- 예상 효과: 측정모델 우도 **-1218.39 → -27.74** (44배 개선!)

### **다음 단계**

1. 절편 계산 함수 구현
2. 측정모델 우도 계산 수정
3. CFA 결과에 절편 추가
4. 테스트 및 검증

---

**작성일**: 2025-11-20  
**작성자**: Augment Agent

