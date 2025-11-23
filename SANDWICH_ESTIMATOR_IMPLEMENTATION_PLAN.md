# Sandwich Estimator (Huber-White) 구축 방안

**날짜**: 2025-11-23  
**목적**: Trust Region 최적화 후 수치적 Hessian + BHHH를 결합한 Robust SE 계산

---

## 📋 요약

**핵심 아이디어**: 
- ✅ BHHH 계산: ~60초 (이미 구현됨)
- ✅ 수치적 Hessian: Gradient 기반 근사로 **대폭 단축 가능**
- ✅ Sandwich Estimator: **이미 구현됨!**

**예상 소요 시간**: **~5-10분** (기존 10.5일 → 5분)

---

## 🔍 1. Sandwich Estimator 이론

### 1.1 공식

**Sandwich Estimator (Huber-White Robust SE)**:

```
Var(θ) = H^(-1) @ B @ H^(-1)

여기서:
- H: Hessian 행렬 (Expected Information)
- B: BHHH 행렬 (Observed Information, OPG)
- Var(θ): Robust 공분산 행렬

Robust SE = sqrt(diag(Var(θ)))
```

---

### 1.2 왜 Robust한가?

**일반 SE** (Hessian만 사용):
```
SE = sqrt(diag(H^(-1)))
```
- ⚠️ 가정: 모델이 올바르게 지정됨 (correctly specified)
- ⚠️ 가정: 오차가 독립동일분포 (i.i.d.)

**Robust SE** (Sandwich):
```
Robust SE = sqrt(diag(H^(-1) @ B @ H^(-1)))
```
- ✅ 모델 오지정에 robust
- ✅ 이분산성에 robust
- ✅ 클러스터링에 robust

---

## ✅ 2. 현재 구현 상태

### 2.1 이미 구현된 함수

<augment_code_snippet path="src/analysis/hybrid_choice_model/iclv_models/bhhh_calculator.py" mode="EXCERPT">
````python
# Line 260-315
def compute_robust_standard_errors(
    self,
    hessian_bhhh: np.ndarray,
    hessian_numerical: np.ndarray,
    regularization: float = 1e-8
) -> np.ndarray:
    """
    Robust 표준오차 계산 (Sandwich estimator)
    
    Var(θ) = H^(-1) @ BHHH @ H^(-1)
    SE = sqrt(diag(Var(θ)))
    
    여기서:
    - H: 수치적 Hessian (또는 BFGS Hessian)
    - BHHH: BHHH Hessian
    """
    n_params = hessian_bhhh.shape[0]
    
    # 수치적 Hessian 역행렬
    hess_num_reg = hessian_numerical + regularization * np.eye(n_params)
    hess_num_inv = np.linalg.inv(hess_num_reg)
    
    # Sandwich estimator: H^(-1) @ BHHH @ H^(-1)
    variance_matrix = hess_num_inv @ hessian_bhhh @ hess_num_inv
    
    # Robust 표준오차
    robust_se = np.sqrt(np.abs(np.diag(variance_matrix)))
    
    return robust_se
````
</augment_code_snippet>

**핵심**: ✅ **이미 완벽하게 구현되어 있음!**

---

## 🚀 3. 효율적인 수치적 Hessian 계산 방법

### 3.1 문제점: scipy.optimize.approx_fprime는 너무 느림

**기존 방법** (10.5일 소요):
```python
# approx_fprime 사용
# 우도 계산 41,209회 필요
```

---

### 3.2 해결책: Gradient 기반 Hessian 근사

**핵심 아이디어**: 
- ✅ 우도 계산 대신 **Gradient 계산** 사용
- ✅ Analytic gradient 이미 구현됨
- ✅ GPU 배치 처리 활용

**공식**:
```
H[i,j] ≈ (∂g_j/∂θ_i) = (g_j(θ + ε*e_i) - g_j(θ)) / ε

여기서:
- g_j: j번째 gradient 성분
- θ: 파라미터 벡터
- e_i: i번째 단위 벡터
- ε: 작은 perturbation
```

---

### 3.3 구현 방법

```python
def compute_numerical_hessian_from_gradient(
    params: np.ndarray,
    gradient_function,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Gradient 함수로부터 수치적 Hessian 계산
    
    Args:
        params: 파라미터 벡터 (n_params,)
        gradient_function: Gradient 계산 함수
        epsilon: Perturbation 크기
    
    Returns:
        Hessian 행렬 (n_params, n_params)
    """
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))
    
    # 기준 gradient 계산
    grad_0 = gradient_function(params)
    
    # 각 파라미터에 대해
    for i in range(n_params):
        # Perturbation
        params_plus = params.copy()
        params_plus[i] += epsilon
        
        # Perturbed gradient 계산
        grad_plus = gradient_function(params_plus)
        
        # Hessian i번째 행 계산
        hessian[i, :] = (grad_plus - grad_0) / epsilon
    
    # 대칭화 (수치 오차 보정)
    hessian = (hessian + hessian.T) / 2
    
    return hessian
```

---

### 3.4 계산 비용

**Gradient 계산 횟수**:
```
기준 gradient: 1회
Perturbed gradient: 202회 (파라미터 수)
총 계산: 203회
```

**소요 시간** (Gradient 1회 = ~2초 가정):
```
203회 × 2초 = 406초 ≈ 6.8분
```

**비교**:
| 방법 | 계산 횟수 | 소요 시간 |
|------|----------|----------|
| **우도 기반** (scipy) | 41,209회 우도 | 10.5일 |
| **Gradient 기반** (제안) | 203회 gradient | **~7분** |
| **속도 향상** | - | **2,160배** |

---

## 📊 4. 전체 Sandwich Estimator 계산 과정

### 4.1 단계별 계산

```python
def compute_sandwich_estimator_efficient(
    optimal_params: np.ndarray,
    gradient_function,
    individual_gradient_function,
    individual_ids: np.ndarray
) -> Dict:
    """
    효율적인 Sandwich Estimator 계산
    
    Returns:
        {
            'hessian_numerical': 수치적 Hessian,
            'hessian_bhhh': BHHH Hessian,
            'variance_matrix': Sandwich 공분산 행렬,
            'robust_se': Robust 표준오차,
            'se_hessian': Hessian 기반 SE,
            'se_bhhh': BHHH 기반 SE
        }
    """
    # 1. BHHH 계산 (~60초)
    print("1. BHHH Hessian 계산 중...")
    individual_gradients = []
    for ind_id in individual_ids:
        grad_i = individual_gradient_function(ind_id, optimal_params)
        individual_gradients.append(grad_i)
    
    hessian_bhhh = compute_bhhh_hessian(individual_gradients)
    print(f"   완료: {len(individual_gradients)}명")
    
    # 2. 수치적 Hessian 계산 (~7분)
    print("2. 수치적 Hessian 계산 중...")
    hessian_numerical = compute_numerical_hessian_from_gradient(
        optimal_params,
        gradient_function,
        epsilon=1e-5
    )
    print(f"   완료: {hessian_numerical.shape}")
    
    # 3. Sandwich Estimator 계산 (~1초)
    print("3. Sandwich Estimator 계산 중...")
    bhhh_calc = BHHHCalculator()
    robust_se = bhhh_calc.compute_robust_standard_errors(
        hessian_bhhh,
        hessian_numerical,
        regularization=1e-8
    )
    
    # 4. 비교용 SE 계산
    se_hessian = np.sqrt(np.abs(np.diag(np.linalg.inv(hessian_numerical))))
    se_bhhh = np.sqrt(np.abs(np.diag(np.linalg.inv(hessian_bhhh))))
    
    # 5. Sandwich 공분산 행렬
    h_inv = np.linalg.inv(hessian_numerical)
    variance_matrix = h_inv @ hessian_bhhh @ h_inv
    
    return {
        'hessian_numerical': hessian_numerical,
        'hessian_bhhh': hessian_bhhh,
        'variance_matrix': variance_matrix,
        'robust_se': robust_se,
        'se_hessian': se_hessian,
        'se_bhhh': se_bhhh
    }
```

---

### 4.2 소요 시간 요약

| 단계 | 작업 | 소요 시간 |
|------|------|----------|
| 1 | BHHH 계산 (328명) | ~60초 |
| 2 | 수치적 Hessian (203회 gradient) | ~7분 |
| 3 | Sandwich 계산 (행렬 곱셈) | ~1초 |
| **총 소요 시간** | - | **~8분** |

**비교**:
- 기존 (우도 기반): 10.5일
- 제안 (Gradient 기반): **8분**
- **속도 향상**: **1,890배**

---

## 💡 5. 구현 위치

### 5.1 새로운 함수 추가

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**추가할 함수**:

```python
def _compute_numerical_hessian_from_gradient(
    self,
    optimal_params: np.ndarray,
    measurement_model,
    structural_model,
    choice_model,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Gradient 함수로부터 수치적 Hessian 계산
    
    우도 계산 대신 gradient 계산 사용 → 2,160배 빠름
    """
    n_params = len(optimal_params)
    hessian = np.zeros((n_params, n_params))
    
    # 기준 gradient 계산
    grad_0 = self._joint_gradient(
        optimal_params,
        measurement_model,
        structural_model,
        choice_model
    )
    
    self.iteration_logger.info(
        f"수치적 Hessian 계산 시작 (Gradient 기반)\n"
        f"  파라미터 수: {n_params}\n"
        f"  Gradient 계산 횟수: {n_params + 1}회\n"
        f"  예상 소요 시간: ~{(n_params + 1) * 2 / 60:.1f}분"
    )
    
    # 각 파라미터에 대해
    for i in range(n_params):
        if i % 10 == 0:
            self.iteration_logger.info(f"  진행: {i}/{n_params}")
        
        # Perturbation
        params_plus = optimal_params.copy()
        params_plus[i] += epsilon
        
        # Perturbed gradient 계산
        grad_plus = self._joint_gradient(
            params_plus,
            measurement_model,
            structural_model,
            choice_model
        )
        
        # Hessian i번째 행
        hessian[i, :] = (grad_plus - grad_0) / epsilon
    
    # 대칭화
    hessian = (hessian + hessian.T) / 2
    
    self.iteration_logger.info("수치적 Hessian 계산 완료")
    
    return hessian
```

---

### 5.2 Sandwich Estimator 계산 함수

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**추가할 함수**:

```python
def _compute_sandwich_estimator(
    self,
    optimal_params: np.ndarray,
    measurement_model,
    structural_model,
    choice_model
) -> Dict:
    """
    Sandwich Estimator (Huber-White Robust SE) 계산
    
    Returns:
        {
            'hessian_numerical': 수치적 Hessian,
            'hessian_bhhh': BHHH Hessian,
            'variance_matrix': Sandwich 공분산 행렬,
            'robust_se': Robust 표준오차,
            'se_hessian': Hessian 기반 SE,
            'se_bhhh': BHHH 기반 SE
        }
    """
    self.iteration_logger.info("\n" + "="*80)
    self.iteration_logger.info("Sandwich Estimator (Huber-White Robust SE) 계산 시작")
    self.iteration_logger.info("="*80)
    
    # 1. BHHH Hessian 계산
    self.iteration_logger.info("\n[1/3] BHHH Hessian 계산 중...")
    hessian_bhhh_inv = self._compute_bhhh_hessian_inverse(
        optimal_params,
        measurement_model,
        structural_model,
        choice_model
    )
    
    # BHHH Hessian 복원 (역행렬의 역행렬)
    hessian_bhhh = np.linalg.inv(hessian_bhhh_inv)
    
    # 2. 수치적 Hessian 계산
    self.iteration_logger.info("\n[2/3] 수치적 Hessian 계산 중...")
    hessian_numerical = self._compute_numerical_hessian_from_gradient(
        optimal_params,
        measurement_model,
        structural_model,
        choice_model
    )
    
    # 3. Sandwich Estimator 계산
    self.iteration_logger.info("\n[3/3] Sandwich Estimator 계산 중...")
    bhhh_calc = BHHHCalculator(logger=self.iteration_logger)
    robust_se = bhhh_calc.compute_robust_standard_errors(
        hessian_bhhh,
        hessian_numerical,
        regularization=1e-8
    )
    
    # 비교용 SE 계산
    se_hessian = bhhh_calc.compute_standard_errors(
        np.linalg.inv(hessian_numerical)
    )
    se_bhhh = bhhh_calc.compute_standard_errors(hessian_bhhh_inv)
    
    # Sandwich 공분산 행렬
    h_inv = np.linalg.inv(hessian_numerical + 1e-8 * np.eye(len(optimal_params)))
    variance_matrix = h_inv @ hessian_bhhh @ h_inv
    
    self.iteration_logger.info("\n" + "="*80)
    self.iteration_logger.info("Sandwich Estimator 계산 완료")
    self.iteration_logger.info("="*80)
    
    return {
        'hessian_numerical': hessian_numerical,
        'hessian_bhhh': hessian_bhhh,
        'variance_matrix': variance_matrix,
        'robust_se': robust_se,
        'se_hessian': se_hessian,
        'se_bhhh': se_bhhh
    }
```

---

## 📊 6. 사용 방법

### 6.1 Trust Region 최적화 후 호출

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**수정 위치**: Line 1506-1514 (else 분기)

```python
else:
    # Optimizer가 hess_inv를 제공하지 않는 경우
    self.iteration_logger.warning("⚠️ Optimizer가 Hessian 역행렬을 제공하지 않음")
    
    # 옵션 1: BHHH만 사용 (빠름, ~60초)
    if self.config.estimation.se_method == 'bhhh':
        self.iteration_logger.info("→ BHHH 방법으로 Hessian 역행렬 계산")
        hess_inv_bhhh = self._compute_bhhh_hessian_inverse(...)
        self.hessian_inv_matrix = hess_inv_bhhh
    
    # 옵션 2: Sandwich Estimator 사용 (robust, ~8분)
    elif self.config.estimation.se_method == 'robust':
        self.iteration_logger.info("→ Sandwich Estimator (Robust SE) 계산")
        sandwich_results = self._compute_sandwich_estimator(
            result.x,
            measurement_model,
            structural_model,
            choice_model
        )
        self.hessian_inv_matrix = sandwich_results['variance_matrix']
        self.robust_se = sandwich_results['robust_se']
        self.se_hessian = sandwich_results['se_hessian']
        self.se_bhhh = sandwich_results['se_bhhh']
```

---

### 6.2 Config 설정

**파일**: `scripts/test_gpu_batch_iclv.py`

```python
config = create_sugar_substitute_multi_lv_config(
    ...
    optimizer='trust-constr',
    calculate_se=True,
    se_method='robust',  # 'bhhh', 'robust', 'hessian'
    ...
)
```

---

## 📋 7. 결과 저장

### 7.1 CSV 파일에 3가지 SE 저장

```python
results_df = pd.DataFrame({
    'parameter': param_names,
    'estimate': optimal_params,
    'se_hessian': se_hessian,      # 수치적 Hessian 기반
    'se_bhhh': se_bhhh,            # BHHH 기반
    'se_robust': robust_se,        # Sandwich (Robust)
    't_stat_robust': optimal_params / robust_se,
    'p_value_robust': 2 * (1 - stats.norm.cdf(np.abs(optimal_params / robust_se)))
})
```

---

## 📊 8. 예상 결과

### 8.1 SE 비교

| 파라미터 | SE (Hessian) | SE (BHHH) | SE (Robust) | 차이 |
|---------|-------------|-----------|------------|------|
| gamma_HC_to_PB | 0.123 | 0.145 | 0.156 | +27% |
| beta_price | 0.089 | 0.091 | 0.098 | +10% |
| theta_HC | 0.234 | 0.267 | 0.289 | +23% |

**일반적 패턴**:
- Robust SE ≥ BHHH SE ≥ Hessian SE
- 모델 오지정 시 Robust SE가 더 큼

---

## 💡 9. 최종 권장

### 9.1 상황별 권장

| 상황 | 권장 방법 | 소요 시간 | 이유 |
|------|----------|----------|------|
| **일반적인 경우** | BHHH | ~60초 | 빠르고 충분히 정확 |
| **모델 오지정 의심** | **Sandwich** | **~8분** | **Robust SE 필요** |
| **논문 제출용** | **Sandwich** | **~8분** | **더 보수적인 SE** |
| **시간 부족** | BHHH | ~60초 | 최소한의 SE |

---

### 9.2 최종 결론

**Sandwich Estimator 구축 방안**:
1. ✅ **이미 구현됨** (`compute_robust_standard_errors`)
2. ✅ **Gradient 기반 수치적 Hessian** 추가 (~7분)
3. ✅ **총 소요 시간**: ~8분 (기존 10.5일 → 8분)
4. ✅ **속도 향상**: **1,890배**

**권장**: ✅ **Sandwich Estimator 구현 강력 권장**

---

**분석 완료 일시**: 2025-11-23

---

## ✅ 10. 구현 완료!

### 10.1 추가된 함수

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

1. **`_compute_numerical_hessian_from_gradient`** (Line 2993-3088)
   - Gradient 기반 수치적 Hessian 계산
   - 203회 gradient 계산 (~7분)
   - 대칭화 처리 포함

2. **`_compute_sandwich_estimator`** (Line 3232-3361)
   - BHHH + 수치적 Hessian 결합
   - Sandwich 공분산 행렬 계산
   - 3가지 SE 비교 (Hessian, BHHH, Robust)

---

### 10.2 사용 방법

**Config 설정**:
```python
# scripts/test_gpu_batch_iclv.py
config = create_sugar_substitute_multi_lv_config(
    ...
    optimizer='trust-constr',
    calculate_se=True,
    se_method='robust',  # 'bhhh', 'robust', 'hessian'
    ...
)
```

**호출 위치**: `simultaneous_estimator_fixed.py` Line 1506-1514

```python
else:
    # Optimizer가 hess_inv를 제공하지 않는 경우
    if self.config.estimation.se_method == 'robust':
        # Sandwich Estimator 사용
        sandwich_results = self._compute_sandwich_estimator(
            result.x,
            measurement_model,
            structural_model,
            choice_model
        )
        if sandwich_results is not None:
            self.hessian_inv_matrix = sandwich_results['variance_matrix']
            self.robust_se = sandwich_results['robust_se']
            self.se_hessian = sandwich_results['se_hessian']
            self.se_bhhh = sandwich_results['se_bhhh']
    else:
        # BHHH만 사용 (기본)
        hess_inv_bhhh = self._compute_bhhh_hessian_inverse(...)
        self.hessian_inv_matrix = hess_inv_bhhh
```

---

### 10.3 예상 출력

```
================================================================================
Sandwich Estimator (Huber-White Robust SE) 계산 시작
================================================================================

[1/3] BHHH Hessian 계산 중...
  BHHH 계산: 328명 사용 (전체 328명 중)
  ...
  BHHH Hessian shape: (202, 202)

[2/3] 수치적 Hessian 계산 중...
  파라미터 수: 202
  Gradient 계산 횟수: 203회
  예상 소요 시간: ~6.8분
  ...
  수치적 Hessian shape: (202, 202)

[3/3] Sandwich Estimator 계산 중...
  Robust 표준오차 계산 완료: 범위 [1.23e-02, 4.56e-01]

================================================================================
Sandwich Estimator 계산 완료
================================================================================
  SE (Hessian) 범위: [1.12e-02, 3.89e-01]
  SE (BHHH) 범위: [1.18e-02, 4.12e-01]
  SE (Robust) 범위: [1.23e-02, 4.56e-01]

  평균 SE 비율:
    Robust / Hessian: 1.15
    Robust / BHHH: 1.08
    BHHH / Hessian: 1.06
================================================================================
```

---

### 10.4 결과 CSV 저장

**3가지 SE 모두 저장**:
```python
results_df = pd.DataFrame({
    'parameter': param_names,
    'estimate': optimal_params,
    'se_hessian': se_hessian,      # 수치적 Hessian 기반
    'se_bhhh': se_bhhh,            # BHHH 기반
    'se_robust': robust_se,        # Sandwich (Robust)
    't_stat_robust': optimal_params / robust_se,
    'p_value_robust': 2 * (1 - stats.norm.cdf(np.abs(optimal_params / robust_se)))
})
```

---

## 🎯 최종 요약

| 항목 | 내용 |
|------|------|
| **구현 상태** | ✅ **완료** |
| **추가 함수** | 2개 (수치적 Hessian, Sandwich) |
| **소요 시간** | ~8분 (BHHH 60초 + 수치적 Hessian 7분) |
| **속도 향상** | 1,890배 (기존 10.5일 → 8분) |
| **SE 종류** | 3가지 (Hessian, BHHH, Robust) |
| **권장 사용** | 논문 제출, 모델 오지정 의심 시 |

**구현 완료!** 🎉

