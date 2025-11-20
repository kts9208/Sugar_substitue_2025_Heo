# 측정모델 절편 구현 완료

## 📊 최종 결과

### ✅ **절편이 측정모델 우도 계산에 성공적으로 추가되었습니다!**

---

## 1. 수정된 파일

### **1.1 CFA 추정 (절편 생성)**

#### `src/analysis/hybrid_choice_model/iclv_models/sem_estimator.py`
- **Line 145-185**: 절편을 각 지표의 평균으로 계산
- 절편 DataFrame 생성 (semopy params 형식)
- `op='~'`, `rval='1'`로 저장

#### `src/analysis/hybrid_choice_model/iclv_models/sequential_estimator.py`
- **Line 180**: pickle 파일에 절편 저장
- **Line 143**: results 딕셔너리에 절편 추가
- **Line 217**: CSV 파일에 절편 저장 (`param_type='intercept'`)

### **1.2 측정모델 우도 계산 (절편 사용)**

#### `src/analysis/hybrid_choice_model/iclv_models/gpu_measurement_equations.py`

**`log_likelihood()` 메서드** (Line 461-532):
```python
def log_likelihood(self, data, latent_var, params):
    alpha = params.get('alpha', None)  # ✅ 절편 (선택적)
    
    # ✅ 예측값: Y_pred = α + ζ * LV
    if alpha_gpu is not None:
        y_pred = alpha_gpu[i] + zeta_gpu[i] * latent_var_gpu
    else:
        y_pred = zeta_gpu[i] * latent_var_gpu  # 하위 호환성
```

**`log_likelihood_batch()` 메서드** (Line 538-601):
```python
def log_likelihood_batch(self, data_batch, latent_vars, params):
    # ✅ 절편 (선택적)
    if 'alpha' in params and params['alpha'] is not None:
        alpha = cp.asarray(params['alpha'])
    else:
        alpha = None
    
    # ✅ 예측값: Y_pred = α + ζ * LV
    if alpha is not None:
        y_pred = alpha[i] + zeta[i] * lv_gpu
    else:
        y_pred = zeta[i] * lv_gpu  # 하위 호환성
```

### **1.3 동시추정 (절편 로드)**

#### `scripts/test_gpu_batch_iclv.py`

**측정모델에 CFA 결과 로드** (Line 235-303):
```python
intercepts_df = cfa_results.get('intercepts', None)  # ✅ 절편 로드

# ✅ alpha (절편)
alpha_values = []
if intercepts_df is not None:
    for indicator in indicators:
        row = intercepts_df[(intercepts_df['lval'] == indicator) &
                           (intercepts_df['op'] == '~') &
                           (intercepts_df['rval'] == '1')]
        alpha_values.append(float(row['Estimate'].iloc[0]))

model.config.alpha = np.array(alpha_values)  # ✅ 절편 추가
```

**초기값 설정** (Line 337-403):
```python
intercepts_df = cfa_results.get('intercepts', None)  # ✅ 절편 로드

measurement_dict[lv_name] = {
    'zeta': np.array(zeta_values),
    'sigma_sq': np.array(sigma_sq_values),
    'alpha': np.array(alpha_values)  # ✅ 절편 추가
}
```

#### `src/analysis/hybrid_choice_model/iclv_models/parameter_manager.py`

**파라미터 딕셔너리 생성** (Line 252-279):
```python
alpha = getattr(model.config, 'alpha', None)  # ✅ 절편 추가

param_dict['measurement'][lv_name] = {
    'zeta': zeta,
    'sigma_sq': sigma_sq,
    'alpha': alpha  # ✅ 절편 추가
}
```

---

## 2. 테스트 결과

### **테스트 스크립트**: `scripts/test_measurement_with_intercepts.py`

### **결과 (health_concern, LV=0.5)**

| 항목 | 절편 없음 | 절편 포함 | 개선 |
|------|----------|----------|------|
| **로그우도** | -142.21 | -3.29 | **138.92** |
| **지표당 평균** | -23.70 | -0.55 | **23.15** |
| **개선 비율** | - | - | **43배!** |

### **상세 분석 (q6)**

| 항목 | 절편 없음 | 절편 포함 |
|------|----------|----------|
| **Y_pred** | 0.50 | 4.26 |
| **residual** | 3.50 | -0.26 |
| **로그우도** | -28.50 | -0.31 |
| **개선** | - | **28.19** |

---

## 3. 절편 값

### **health_concern 절편 (α)**

| 지표 | 절편 | 의미 |
|------|------|------|
| q6 | 3.76 | 지표 평균 |
| q7 | 3.65 | 지표 평균 |
| q8 | 3.64 | 지표 평균 |
| q9 | 3.80 | 지표 평균 |
| q10 | 3.89 | 지표 평균 |
| q11 | 3.58 | 지표 평균 |

**평균**: 3.72 (1-5점 리커트 척도)

---

## 4. 하위 호환성

### **절편이 없는 경우에도 작동**

```python
# 절편 없음 (기존 코드)
params = {
    'zeta': np.array([...]),
    'sigma_sq': np.array([...])
}
ll = model.log_likelihood(data, lv, params)  # ✅ 작동

# 절편 포함 (새 코드)
params = {
    'zeta': np.array([...]),
    'sigma_sq': np.array([...]),
    'alpha': np.array([...])  # ✅ 절편 추가
}
ll = model.log_likelihood(data, lv, params)  # ✅ 작동
```

---

## 5. 다음 단계

### ✅ **동시추정 실행**

1. **CFA 결과 확인**: `results/sequential_stage_wise/cfa_results.pkl`에 절편 포함 확인
2. **동시추정 실행**: `scripts/test_gpu_batch_iclv.py` 실행
3. **우도 확인**: 측정모델 우도가 크게 개선되었는지 확인
4. **스케일링 재평가**: 측정모델 우도가 개선되면 스케일링 불필요할 수도 있음

---

## 6. 예상 효과

### **측정모델 우도 개선**

- **이전**: 지표당 평균 -23.70
- **현재**: 지표당 평균 -0.55
- **개선**: **43배!**

### **전체 우도 균형**

- 측정모델 우도가 크게 개선되어 구조모델/선택모델과 균형 맞춤
- 스케일링 (÷38) 불필요해질 가능성 높음
- gamma 파라미터 업데이트 더 원활해질 것으로 예상

---

## 7. 결론

### **성공적으로 완료!**

✅ **절편이 측정모델 우도 계산에 추가됨**
- CFA 결과에서 절편 로드
- 우도 계산에 절편 사용
- 하위 호환성 유지

✅ **우도 크게 개선**
- 지표당 평균: -23.70 → -0.55 (43배 개선!)
- 전체 우도 균형 크게 개선 예상

✅ **다음 단계 준비 완료**
- 동시추정 실행 가능
- 스케일링 재평가 필요

---

**작성일**: 2025-11-20  
**작성자**: Augment Agent

