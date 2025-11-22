# 동시추정 빠른 참조 가이드

## 🎯 한눈에 보는 동시추정

### 파라미터 구조
```
param_dict = {
    'measurement': {  # 고정 (CFA 결과)
        'HC': {'zeta': [...], 'sigma_sq': [...], 'alpha': [...]},
        'PB': {'zeta': [...], 'sigma_sq': [...], 'alpha': [...]},
        'PP': {'zeta': [...], 'sigma_sq': [...], 'alpha': [...]},
        'NK': {'zeta': [...], 'sigma_sq': [...], 'alpha': [...]},
        'PI': {'zeta': [...], 'sigma_sq': [...], 'alpha': [...]}
    },
    'structural': {  # 추정 대상
        'gamma_HC_to_PB': 0.1,
        'gamma_PB_to_PI': 0.1
    },
    'choice': {  # 추정 대상
        'asc_sugar': 0.1,
        'asc_sugar_free': 0.1,
        'beta_health_label': 0.1,
        'beta_price': 0.1,
        'theta_sugar_PI': 0.1,
        'theta_sugar_free_PI': 0.1,
        'gamma_sugar_PI_health_label': 0.1,
        'gamma_sugar_free_PI_health_label': 0.1
    }
}
```

---

## 📊 모델별 파라미터 및 그래디언트

### 1. 측정모델 (고정)

#### 파라미터
- **zeta (ζ)**: 요인적재량
- **sigma_sq (σ²)**: 오차분산
- **alpha (α)**: 절편

#### 우도 함수
```
P(I_j | η) = N(I_j | α_j + ζ_j × η, σ²_j)
```

#### 그래디언트 (계산만, 최적화 안 함)
```
∂LL/∂ζ_j = Σ_r w_r × (I_j - α_j - ζ_j × η_r) × η_r / σ²_j
∂LL/∂σ²_j = Σ_r w_r × [-1/(2σ²_j) + (I_j - α_j - ζ_j × η_r)² / (2σ⁴_j)]
```

---

### 2. 구조모델 (추정)

#### 파라미터
- **gamma (γ)**: 경로 계수
  - `gamma_HC_to_PB`: HC → PB
  - `gamma_PB_to_PI`: PB → PI

#### 구조 방정식
```
η_target = γ × η_predictor + ε
ε ~ N(0, 1)
```

#### 그래디언트
```
∂LL/∂γ = Σ_r w_r × [
    ∂P(I|η)/∂η × ∂η/∂γ +
    ∂P(Y|η)/∂η × ∂η/∂γ +
    ∂P(η)/∂γ
]

여기서:
∂η_target/∂γ = η_predictor
∂P(η)/∂γ = (η_target - γ × η_predictor) × η_predictor
```

---

### 3. 선택모델 (추정)

#### 파라미터
- **asc**: 대안별 상수
- **beta (β)**: 속성 계수
- **theta (θ)**: 대안별 LV 계수
- **gamma (γ)**: 대안별 LV-속성 상호작용

#### 효용 함수
```
V_sugar = asc_sugar + β_hl × hl + β_p × p + θ_sugar_PI × PI + γ_sugar_PI_hl × PI × hl
V_sugar_free = asc_sf + β_hl × hl + β_p × p + θ_sf_PI × PI + γ_sf_PI_hl × PI × hl
V_opt_out = 0

P(j) = exp(V_j) / Σ_k exp(V_k)
```

#### 그래디언트
```
∂LL/∂asc_j = Σ_r w_r × Σ_t [(y_t == j) - P(j|η_r)]

∂LL/∂β_attr = Σ_r w_r × Σ_t Σ_j [(y_t == j) - P(j|η_r)] × x_attr

∂LL/∂θ_j_LV = Σ_r w_r × Σ_t [(y_t == j) - P(j|η_r)] × η_r[LV]

∂LL/∂γ_j_LV_attr = Σ_r w_r × Σ_t [(y_t == j) - P(j|η_r)] × η_r[LV] × x_attr
```

---

## 🔄 계산 흐름

### 개인별 우도 계산
```
1. Halton Draws 생성: ε_r ~ N(0, 1)
2. 잠재변수 예측: η_r = f(γ, X, ε_r)
3. 측정모델 우도: P(I | η_r)
4. 선택모델 우도: P(Y | η_r)
5. 구조모델 우도: P(η_r)
6. 결합 우도: LL_r = log[P(I) × P(Y) × P(η)]
7. 개인 우도: LL_i = log[(1/R) Σ_r exp(LL_r)]
```

### 개인별 그래디언트 계산
```
1. 모든 draws의 우도 계산: LL_r (r=1,...,R)
2. Importance Weights: w_r = exp(LL_r) / Σ_r' exp(LL_r')
3. 가중평균 그래디언트: ∂LL/∂θ = Σ_r w_r × ∂LL_r/∂θ
```

### 전체 그래디언트
```
total_grad = Σ_i grad_i
```

---

## 💻 주요 함수 호출 순서

### 우도 계산
```python
# 1. 파라미터 언팩
param_dict = _unpack_parameters(params)

# 2. 개인별 우도 계산
for ind_id in individual_ids:
    # 2.1 Draws 가져오기
    ind_draws = halton_generator.get_draws()[i]
    
    # 2.2 각 draw별 처리
    for draw in ind_draws:
        # 2.2.1 LV 예측
        lv = structural_model.predict(ind_data, draw, param_dict['structural'])
        
        # 2.2.2 우도 계산
        ll_meas = measurement_model.log_likelihood(ind_data, lv, param_dict['measurement'])
        ll_choice = choice_model.log_likelihood(ind_data, lv, param_dict['choice'])
        ll_struct = structural_model.log_likelihood(ind_data, lv, draw, param_dict['structural'])
        
        # 2.2.3 결합 우도
        draw_ll = ll_meas + ll_choice + ll_struct
    
    # 2.3 개인 우도
    person_ll = logsumexp(draw_lls) - log(n_draws)
    
# 3. 전체 우도
total_ll = sum(person_lls)
```

### 그래디언트 계산
```python
# 1. 개인별 그래디언트
for ind_id in individual_ids:
    # 1.1 LV 예측 (모든 draws)
    lvs_list = [structural_model.predict(...) for draw in ind_draws]
    
    # 1.2 우도 계산 (모든 draws)
    ll_batch = compute_joint_likelihood_batch_gpu(...)
    
    # 1.3 Importance Weights
    weights = compute_importance_weights_gpu(ll_batch)
    
    # 1.4 가중평균 그래디언트
    grad_meas = compute_measurement_gradient_batch_gpu(..., weights)
    grad_struct = compute_structural_gradient_batch_gpu(..., weights)
    grad_choice = compute_choice_gradient_batch_gpu(..., weights)
    
    ind_grad = {'measurement': grad_meas, 'structural': grad_struct, 'choice': grad_choice}

# 2. 전체 그래디언트 합산
total_grad = sum(ind_grads)

# 3. 벡터 변환
grad_vector = _pack_gradient(total_grad)
```

---

## 📁 주요 파일 위치

### 실행 스크립트
- `scripts/test_gpu_batch_iclv.py`

### 추정기
- `src/analysis/hybrid_choice_model/iclv_models/simultaneous_gpu_batch_estimator.py`
- `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

### 파라미터 관리
- `src/analysis/hybrid_choice_model/iclv_models/parameter_manager.py`

### 그래디언트 계산
- `src/analysis/hybrid_choice_model/iclv_models/multi_latent_gradient.py`
- `src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`

### 모델
- `src/analysis/hybrid_choice_model/iclv_models/multi_latent_measurement.py`
- `src/analysis/hybrid_choice_model/iclv_models/multi_latent_structural.py`
- `src/analysis/hybrid_choice_model/iclv_models/choice_equations.py`

---

## 🔍 디버깅 팁

### 파라미터 확인
```python
# 파라미터 딕셔너리 출력
print(param_dict.keys())  # ['measurement', 'structural', 'choice']
print(param_dict['structural'])  # {'gamma_HC_to_PB': 0.45, ...}
print(param_dict['choice'])  # {'asc_sugar': 0.12, ...}
```

### 우도 확인
```python
# 각 모델별 우도 출력
print(f"측정모델 우도: {ll_measurement:.4f}")
print(f"선택모델 우도: {ll_choice:.4f}")
print(f"구조모델 우도: {ll_structural:.4f}")
print(f"결합 우도: {ll_total:.4f}")
```

### 그래디언트 확인
```python
# 그래디언트 크기 확인
print(f"구조모델 그래디언트: {grad_dict['structural']}")
print(f"선택모델 그래디언트: {grad_dict['choice']}")
print(f"그래디언트 노름: {np.linalg.norm(grad_vector)}")
```

