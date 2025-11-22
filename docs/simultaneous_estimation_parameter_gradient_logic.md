# 동시추정 코드 파라미터 및 그래디언트 계산 로직 설명

## 📋 목차
1. [전체 구조 개요](#전체-구조-개요)
2. [파라미터 계산 로직](#파라미터-계산-로직)
3. [그래디언트 계산 로직](#그래디언트-계산-로직)
4. [모델별 상세 설명](#모델별-상세-설명)

---

## 1. 전체 구조 개요

### 1.1 동시추정의 정의
```
✅ 동시추정 (Simultaneous Estimation):
   - 측정모델 파라미터: CFA 결과로 고정 (추정 안 함)
   - 구조모델 + 선택모델: 동시 추정
```

### 1.2 주요 클래스 구조
```
test_gpu_batch_iclv.py (실행 스크립트)
    ↓
SimultaneousGPUBatchEstimator (GPU 배치 처리)
    ↓ 상속
SimultaneousEstimatorFixed (동시추정 기본 로직)
    ↓ 사용
ParameterManager (파라미터 관리)
MultiLatentJointGradient (그래디언트 계산)
```

---

## 2. 파라미터 계산 로직

### 2.1 파라미터 구조

동시추정에서 다루는 파라미터는 3가지 모델로 구성됩니다:

```python
param_dict = {
    'measurement': {  # 측정모델 (고정, 추정 안 함)
        'HC': {'zeta': array, 'sigma_sq': array, 'alpha': array},
        'PB': {'zeta': array, 'sigma_sq': array, 'alpha': array},
        'PP': {'zeta': array, 'sigma_sq': array, 'alpha': array},
        'NK': {'zeta': array, 'sigma_sq': array, 'alpha': array},
        'PI': {'zeta': array, 'sigma_sq': array, 'alpha': array}
    },
    'structural': {  # 구조모델 (추정 대상)
        'gamma_HC_to_PB': float,
        'gamma_PB_to_PI': float,
        ...
    },
    'choice': {  # 선택모델 (추정 대상)
        'asc_sugar': float,
        'asc_sugar_free': float,
        'beta_health_label': float,
        'beta_price': float,
        'theta_sugar_PI': float,
        'theta_sugar_free_PI': float,
        ...
    }
}
```

### 2.2 파라미터 초기화 과정

#### Step 1: CFA 결과 로드 (측정모델)
```python
# test_gpu_batch_iclv.py, Line 233-300
pkl_path = project_root / 'results' / 'sequential_stage_wise' / 'cfa_results.pkl'

with open(pkl_path, 'rb') as f:
    cfa_results = pickle.load(f)

# 각 잠재변수별로 측정모델 파라미터 설정
for lv_name, model in measurement_model.models.items():
    # zeta (요인적재량) 추출
    zeta_values = []
    for indicator in indicators:
        row = loadings_df[(loadings_df['lval'] == indicator) &
                         (loadings_df['op'] == '~') &
                         (loadings_df['rval'] == lv_name)]
        zeta_values.append(float(row['Estimate'].iloc[0]))
    
    # sigma_sq (오차분산) 추출
    sigma_sq_values = []
    for indicator in indicators:
        row = errors_df[(errors_df['lval'] == indicator) &
                       (errors_df['op'] == '~~') &
                       (errors_df['rval'] == indicator)]
        sigma_sq_values.append(float(row['Estimate'].iloc[0]))
    
    # alpha (절편) 추출
    alpha_values = []
    for indicator in indicators:
        row = intercepts_df[(intercepts_df['lval'] == indicator) &
                           (intercepts_df['op'] == '~') &
                           (intercepts_df['rval'] == '1')]
        alpha_values.append(float(row['Estimate'].iloc[0]))
    
    # 측정모델 config에 설정
    model.config.zeta = np.array(zeta_values)
    model.config.sigma_sq = np.array(sigma_sq_values)
    model.config.alpha = np.array(alpha_values)
```

#### Step 2: 구조모델 초기화 (0.1)
```python
# test_gpu_batch_iclv.py, Line 405-416
structural_dict = {}
for path in config.structural.hierarchical_paths:
    target_lv = path['target']
    predictors = path['predictors']
    
    for pred_lv in predictors:
        param_name = f'gamma_{pred_lv}_to_{target_lv}'
        structural_dict[param_name] = 0.1
        # 예: gamma_HC_to_PB = 0.1
```

#### Step 3: 선택모델 초기화 (0.1)
```python
# test_gpu_batch_iclv.py, Line 417-453
choice_dict = {}
alternatives = ['sugar', 'sugar_free']  # opt-out 제외

# ASC (Alternative-Specific Constants)
for alt in alternatives:
    choice_dict[f'asc_{alt}'] = 0.1

# beta (속성 계수)
for attr in config.choice.choice_attributes:
    choice_dict[f'beta_{attr}'] = 0.1

# theta (LV 주효과) - 각 대안별로
for lv in config.choice.main_lvs:
    for alt in alternatives:
        choice_dict[f'theta_{alt}_{lv}'] = 0.1

# gamma (LV-속성 상호작용) - 각 대안별로
for interaction in config.choice.lv_attribute_interactions:
    lv = interaction['lv']
    attr = interaction['attribute']
    for alt in alternatives:
        choice_dict[f'gamma_{alt}_{lv}_{attr}'] = 0.1
```

### 2.3 파라미터 관리 (ParameterManager)

`ParameterManager` 클래스는 파라미터의 순서를 보장하고 딕셔너리 ↔ 배열 변환을 담당합니다.

#### 파라미터 이름 생성
```python
# parameter_manager.py, Line 46-84
def get_parameter_names(self, measurement_model, structural_model,
                       choice_model, exclude_measurement: bool = False):
    names = []
    
    # 1. 측정모델 (exclude_measurement=True이면 제외)
    if not exclude_measurement:
        names.extend(self._get_measurement_param_names(measurement_model))
    
    # 2. 구조모델
    names.extend(self._get_structural_param_names(structural_model))
    
    # 3. 선택모델
    names.extend(self._get_choice_param_names(choice_model))
    
    return names
```

#### 배열 → 딕셔너리 변환 (동시추정 전용)
```python
# parameter_manager.py, Line 223-309
def array_to_dict_optimized(self, param_array, param_names,
                            measurement_model, structural_model, choice_model):
    """
    동시추정 전용: 최적화 파라미터 배열을 딕셔너리로 변환
    측정모델 파라미터는 measurement_model 객체에서 직접 추출
    """
    param_dict = {
        'measurement': {},
        'structural': {},
        'choice': {}
    }

    # ✅ 측정모델: measurement_model.models[lv_name].config에서 추출
    for lv_name, model in measurement_model.models.items():
        zeta = model.config.zeta
        sigma_sq = model.config.sigma_sq
        alpha = model.config.alpha

        param_dict['measurement'][lv_name] = {
            'zeta': zeta,
            'sigma_sq': sigma_sq,
            'alpha': alpha
        }

    # ✅ 구조모델 + 선택모델: 배열에서 추출
    for i, name in enumerate(param_names):
        value = param_array[i]

        if name.startswith('gamma_') and '_to_' in name:
            param_dict['structural'][name] = value
        elif name.startswith('asc_'):
            param_dict['choice'][name] = value
        elif name.startswith('theta_'):
            param_dict['choice'][name] = value
        elif name.startswith('beta_'):
            param_dict['choice'][name] = value
        elif name.startswith('gamma_') and ('_sugar_' in name or '_sugar_free_' in name):
            param_dict['choice'][name] = value

    return param_dict
```

---

## 3. 그래디언트 계산 로직

### 3.1 그래디언트 계산 흐름

동시추정에서는 **측정모델 파라미터가 고정**되어 있으므로, 구조모델과 선택모델 파라미터에 대한 그래디언트만 계산합니다.

```
전체 로그우도 함수:
LL = Σ_i log[ (1/R) Σ_r P(Y_i, I_i | η_ir, θ) ]

여기서:
- i: 개인 인덱스
- r: draw 인덱스
- Y_i: 개인 i의 선택 데이터
- I_i: 개인 i의 지표 데이터
- η_ir: draw r에서의 잠재변수 값
- θ: 파라미터 (구조모델 + 선택모델)

P(Y_i, I_i | η_ir, θ) = P(I_i | η_ir) × P(Y_i | η_ir, θ) × P(η_ir | θ)
                       = 측정모델 우도 × 선택모델 우도 × 구조모델 우도
```

### 3.2 Importance Weighting

그래디언트 계산 시 **Importance Weighting**을 사용하여 각 draw의 기여도를 가중평균합니다.

```python
# gpu_gradient_batch.py, Line 1211-1212
ll_batch = compute_joint_likelihood_batch_gpu(...)  # 각 draw의 우도
weights = compute_importance_weights_gpu(ll_batch)  # 가중치 계산

# Importance weights 계산 (Apollo 방식)
weights[r] = exp(ll_batch[r]) / Σ_r' exp(ll_batch[r'])
```

### 3.3 개인별 그래디언트 계산

#### Step 1: 잠재변수 예측
```python
# simultaneous_gpu_batch_estimator.py, Line 550-593
lvs_list = []
for j in range(n_draws):
    draw = ind_draws[j]

    # 계층적 구조
    n_first_order = len(structural_model.exogenous_lvs)
    exo_draws = draw[:n_first_order]  # 1차 LV draws

    # 2차+ LV 오차항
    higher_order_draws = {}
    higher_order_lvs = structural_model.get_higher_order_lvs()
    for i, lv_name in enumerate(higher_order_lvs):
        higher_order_draws[lv_name] = draw[n_first_order + i]

    # 구조모델로 잠재변수 예측
    lv = structural_model.predict(
        ind_data, exo_draws, param_dict['structural'],
        higher_order_draws=higher_order_draws
    )

    lvs_list.append(lv)
```

#### Step 2: 결합 우도 계산
```python
# gpu_gradient_batch.py, Line 1201-1209
ll_batch = compute_joint_likelihood_batch_gpu(
    gpu_measurement_model,
    ind_data,
    lvs_list,
    ind_draws,
    params_dict,
    structural_model,
    choice_model
)
```

#### Step 3: Importance Weights 계산
```python
# gpu_gradient_batch.py, Line 1212
weights = compute_importance_weights_gpu(ll_batch, individual_id=ind_idx)
```

#### Step 4: 가중평균 그래디언트 계산
```python
# gpu_gradient_batch.py, Line 1215-1269
# 측정모델 그래디언트 (고정이므로 계산만 하고 최적화에 사용 안 함)
grad_meas = compute_measurement_gradient_batch_gpu(
    gpu_measurement_model,
    ind_data,
    lvs_list,
    params_dict['measurement'],
    weights
)

# 구조모델 그래디언트
grad_struct = compute_structural_gradient_batch_gpu(
    ind_data,
    lvs_list,
    exo_draws_list,
    params_dict['structural'],
    covariates,
    structural_model.endogenous_lv,
    structural_model.exogenous_lvs,
    weights
)

# 선택모델 그래디언트
grad_choice = compute_choice_gradient_batch_gpu(
    ind_data,
    lvs_list,
    params_dict['choice'],
    structural_model.endogenous_lv,
    choice_model.config.choice_attributes,
    weights
)
```

### 3.4 전체 그래디언트 집계

```python
# simultaneous_estimator_fixed.py, Line 2270-2300
# 모든 개인의 그래디언트 계산
all_grad_dicts = self.joint_grad.compute_gradients(
    all_ind_data=all_ind_data,
    all_ind_draws=all_ind_draws,
    params_dict=param_dict,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_model=choice_model
)

# 개인별 그래디언트 합산
total_grad_dict = {'measurement': {}, 'structural': {}, 'choice': {}}
for grad_dict in all_grad_dicts:
    # 구조모델 그래디언트 합산
    for key, value in grad_dict['structural'].items():
        if key not in total_grad_dict['structural']:
            total_grad_dict['structural'][key] = 0.0
        total_grad_dict['structural'][key] += value

    # 선택모델 그래디언트 합산
    for key, value in grad_dict['choice'].items():
        if key not in total_grad_dict['choice']:
            total_grad_dict['choice'][key] = 0.0
        total_grad_dict['choice'][key] += value
```

---

## 4. 모델별 상세 설명

### 4.1 측정모델 (Measurement Model)

#### 파라미터
- **zeta (ζ)**: 요인적재량 (Factor Loading)
- **sigma_sq (σ²)**: 오차분산 (Error Variance)
- **alpha (α)**: 절편 (Intercept)

#### 우도 함수
```
P(I_i | η_i) = Π_j N(I_ij | α_j + ζ_j × η_i, σ²_j)

여기서:
- I_ij: 개인 i의 지표 j 값
- α_j: 지표 j의 절편
- ζ_j: 지표 j의 요인적재량
- η_i: 개인 i의 잠재변수 값
- σ²_j: 지표 j의 오차분산
```

#### 그래디언트 (고정이므로 계산만 함)
```python
# gpu_gradient_batch.py, Line 365-591
def compute_measurement_gradient_batch_gpu(...):
    """
    측정모델 그래디언트 계산 (가중평균)

    ∂LL/∂ζ_j = Σ_r w_r × ∂log P(I_i | η_ir)/∂ζ_j
    ∂LL/∂σ²_j = Σ_r w_r × ∂log P(I_i | η_ir)/∂σ²_j
    """
    # GPU 배치 처리로 모든 draws를 한 번에 계산
    ...
```

### 4.2 구조모델 (Structural Model)

#### 파라미터
- **gamma (γ)**: 경로 계수 (Path Coefficient)
  - 예: `gamma_HC_to_PB` (건강관심 → 인지된 혜택)

#### 구조 방정식
```
η_target = γ × η_predictor + ε

여기서:
- η_target: 목표 잠재변수 (예: PB)
- η_predictor: 예측 잠재변수 (예: HC)
- γ: 경로 계수
- ε: 오차항 (표준정규분포)
```

#### 계층적 구조 예시
```
현재 모델: HC → PB → PI

1차 LV (외생): HC, PP, NK
2차 LV (내생): PB, PI

경로:
- gamma_HC_to_PB: HC → PB
- gamma_PB_to_PI: PB → PI
```

#### 그래디언트 계산
```python
# gpu_gradient_batch.py, Line 594-826
def compute_structural_gradient_batch_gpu(...):
    """
    구조모델 그래디언트 계산 (체인룰 역전파)

    ✅ 올바른 그래디언트:
    ∂LL/∂γ_HC_to_PB = Σ_r w_r × ∂LL_r/∂γ_HC_to_PB

    ∂LL_r/∂γ_HC_to_PB = ∂LL_measurement/∂PB × ∂PB/∂γ_HC_to_PB
                        + ∂LL_choice/∂PB × ∂PB/∂γ_HC_to_PB
                        + ∂LL_structural/∂γ_HC_to_PB

    여기서:
    - ∂PB/∂γ_HC_to_PB = HC (예측변수 값)
    - ∂LL_structural/∂γ_HC_to_PB = (PB - γ × HC) × HC / σ²
    """

    # 계층적 구조 처리
    for path in hierarchical_paths:
        predictor = path['predictors'][0]
        target = path['target']
        param_name = f'gamma_{predictor}_to_{target}'

        # 체인룰 적용
        grad_from_measurement = ...  # 측정모델로부터의 역전파
        grad_from_choice = ...       # 선택모델로부터의 역전파
        grad_from_structural = ...   # 구조모델 자체의 그래디언트

        grad_dict[param_name] = (grad_from_measurement +
                                 grad_from_choice +
                                 grad_from_structural)
```

### 4.3 선택모델 (Choice Model)

#### 파라미터
- **asc (Alternative-Specific Constant)**: 대안별 상수
  - `asc_sugar`: 일반당 상수
  - `asc_sugar_free`: 무설탕 상수
  - opt-out은 기준 대안 (파라미터 없음)

- **beta (β)**: 속성 계수
  - `beta_health_label`: 건강 라벨 효과
  - `beta_price`: 가격 효과

- **theta (θ)**: 대안별 잠재변수 계수
  - `theta_sugar_PI`: 일반당 선택에 대한 PI 효과
  - `theta_sugar_free_PI`: 무설탕 선택에 대한 PI 효과

- **gamma (γ)**: 대안별 LV-속성 상호작용
  - `gamma_sugar_PI_health_label`: 일반당에서 PI × health_label 상호작용
  - `gamma_sugar_free_PI_health_label`: 무설탕에서 PI × health_label 상호작용

#### 효용 함수 (Multinomial Logit)
```
V_sugar = asc_sugar + β_health_label × health_label + β_price × price
          + θ_sugar_PI × PI
          + γ_sugar_PI_health_label × PI × health_label

V_sugar_free = asc_sugar_free + β_health_label × health_label + β_price × price
               + θ_sugar_free_PI × PI
               + γ_sugar_free_PI_health_label × PI × health_label

V_opt_out = 0  (기준 대안)

P(alternative = j) = exp(V_j) / Σ_k exp(V_k)
```

#### 그래디언트 계산
```python
# gpu_gradient_batch.py, Line 828-1095
def compute_choice_gradient_batch_gpu(...):
    """
    선택모델 그래디언트 계산 (가중평균 + 배치 처리)

    ∂LL/∂θ = Σ_r w_r × Σ_t ∂log P(y_it | η_ir)/∂θ

    여기서:
    - t: 선택 상황 인덱스
    - ∂log P(y_it | η_ir)/∂θ = (y_it - P(y_it | η_ir)) × x_it
    """

    # GPU 배치 처리
    # 모든 draws × 모든 선택 상황을 한 번에 계산

    # ASC 그래디언트
    for alt in ['sugar', 'sugar_free']:
        grad_dict[f'asc_{alt}'] = Σ_r w_r × Σ_t (y_it == alt) - P(alt | η_ir)

    # Beta 그래디언트
    for attr in choice_attributes:
        grad_dict[f'beta_{attr}'] = Σ_r w_r × Σ_t Σ_j (y_it == j) - P(j | η_ir) × x_it[attr]

    # Theta 그래디언트
    for lv in main_lvs:
        for alt in ['sugar', 'sugar_free']:
            grad_dict[f'theta_{alt}_{lv}'] = Σ_r w_r × Σ_t ((y_it == alt) - P(alt | η_ir)) × η_ir[lv]

    # Gamma 그래디언트 (상호작용)
    for interaction in lv_attribute_interactions:
        lv = interaction['lv']
        attr = interaction['attribute']
        for alt in ['sugar', 'sugar_free']:
            grad_dict[f'gamma_{alt}_{lv}_{attr}'] = (
                Σ_r w_r × Σ_t ((y_it == alt) - P(alt | η_ir)) × η_ir[lv] × x_it[attr]
            )
```

---

## 5. GPU 배치 처리 최적화

### 5.1 완전 병렬화 (Full Parallelization)

```python
# simultaneous_gpu_batch_estimator.py, Line 343-368
if self.use_gpu and self.use_full_parallel:
    # 모든 개인 데이터 준비
    all_ind_data = []
    for ind_id in individual_ids:
        ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
        all_ind_data.append(ind_data)

    # ✅ 모든 개인 × 모든 draws를 한 번에 GPU로 계산
    total_ll = gpu_gradient_batch.compute_all_individuals_likelihood_full_batch_gpu(
        self.gpu_measurement_model,
        all_ind_data,
        draws,
        param_dict,
        structural_model,
        choice_model,
        use_scaling=True  # 최적화 중에는 스케일링 사용
    )
```

### 5.2 메모리 관리

```python
# simultaneous_gpu_batch_estimator.py, Line 200-208
self.memory_monitor = MemoryMonitor(
    cpu_threshold_mb=self.memory_monitor_cpu_threshold_mb,
    gpu_threshold_mb=self.memory_monitor_gpu_threshold_mb,
    auto_cleanup=True
)

# 우도 계산 전후로 메모리 체크 및 정리
mem_info = self.memory_monitor.check_and_cleanup("우도 계산")
```

---

## 6. 최적화 과정

### 6.1 L-BFGS-B 최적화

```python
# simultaneous_estimator_fixed.py
# L-BFGS-B 알고리즘 사용
# - Analytic Gradient 사용
# - Parameter Scaling 적용
# - Bounds 설정

optimizer_result = scipy.optimize.minimize(
    fun=objective_function,      # -LL (minimize)
    x0=initial_params,           # 초기값
    method='L-BFGS-B',
    jac=gradient_function,       # Analytic gradient
    bounds=bounds,               # 파라미터 범위
    options={'maxiter': MAX_ITERATIONS}
)
```

### 6.2 파라미터 스케일링

```python
# parameter_scaler.py
# 파라미터 크기 불균형 해결
# - 측정모델: 0.1 ~ 10 범위
# - 구조모델: -5 ~ 5 범위
# - 선택모델: -10 ~ 10 범위

scaled_params = scaler.scale_parameters(params)
unscaled_params = scaler.unscale_parameters(scaled_params)
```

---

## 7. 결과 처리

### 7.1 언스케일링된 우도 계산

```python
# simultaneous_gpu_batch_estimator.py, Line 869-925
# ✅ 최적화는 스케일링된 우도로 수행
# ✅ 최종 우도는 언스케일링하여 AIC/BIC 계산

unscaled_ll = gpu_gradient_batch.compute_all_individuals_likelihood_full_batch_gpu(
    ...,
    use_scaling=False  # 언스케일링
)

# AIC, BIC 재계산
results['log_likelihood'] = unscaled_ll
results['aic'] = -2 * unscaled_ll + 2 * k
results['bic'] = -2 * unscaled_ll + k * np.log(n)
```

### 7.2 파라미터 통계 추출

```python
# test_gpu_batch_iclv.py, Line 569-748
# 측정모델: CFA 결과에서 추출
# 구조모델 + 선택모델: 동시추정 결과에서 추출

param_list = []

# 측정모델 (CFA 결과)
for _, row in loadings_df.iterrows():
    param_list.append({
        'Coefficient': f'ζ_{lv_name}_{indicator}',
        'Estimate': row['Estimate'],
        'Std. Err.': row['Std. Err'],
        'P. Value': row['p-value']
    })

# 구조모델 (동시추정 결과)
for key, value in stats['structural'].items():
    param_list.append({
        'Coefficient': f'γ_{key.replace("gamma_", "")}',
        'Estimate': value['estimate'],
        'Std. Err.': value.get('std_error', '-'),
        'P. Value': value.get('p_value', '-')
    })

# 선택모델 (동시추정 결과)
for param_name, param_stats in stats['choice'].items():
    param_list.append({
        'Coefficient': param_name,
        'Estimate': param_stats['estimate'],
        'Std. Err.': param_stats.get('std_error', '-'),
        'P. Value': param_stats.get('p_value', '-')
    })
```

---

## 8. 요약

### 8.1 파라미터 계산 로직
1. **측정모델**: CFA 결과에서 로드 (고정)
2. **구조모델**: 0.1로 초기화 → 최적화
3. **선택모델**: 0.1로 초기화 → 최적화

### 8.2 그래디언트 계산 로직
1. **개인별 처리**: 각 개인의 모든 draws에 대해 계산
2. **Importance Weighting**: 각 draw의 우도에 따라 가중치 부여
3. **가중평균**: 가중치를 적용하여 그래디언트 평균
4. **전체 합산**: 모든 개인의 그래디언트를 합산

### 8.3 주요 특징
- ✅ **GPU 배치 처리**: 모든 개인 × 모든 draws를 한 번에 계산
- ✅ **Analytic Gradient**: 수치 미분 대신 해석적 그래디언트 사용
- ✅ **Parameter Scaling**: 파라미터 크기 불균형 해결
- ✅ **Memory Management**: 자동 메모리 모니터링 및 정리
- ✅ **Hierarchical Structure**: 계층적 잠재변수 구조 지원
- ✅ **Alternative-Specific Model**: 대안별 파라미터 지원
```

