# 수치적 Hessian 계산 시 Gradient 계산 방법

**날짜**: 2025-11-23  
**질문**: 수치적 Hessian 계산 시 사용하는 gradient는 어떻게 계산되는가?

---

## 📋 요약

**답변**: ✅ **Analytic Gradient** 사용 (수치적 미분 아님!)

- ✅ **계산 방법**: GPU 배치 처리로 analytic gradient 계산
- ✅ **정확도**: 매우 높음 (수치 오차 없음)
- ✅ **속도**: 빠름 (~2초/회)
- ✅ **구현**: `self._joint_gradient()` → `self.joint_grad.compute_gradients()`

---

## 🔍 1. Gradient 계산 흐름

### 1.1 호출 체인

```
_compute_numerical_hessian_from_gradient()
  ↓
self._joint_gradient(params, ...)
  ↓
self._compute_gradient(params, ...)
  ↓
self.joint_grad.compute_gradients(...)  # MultiLatentJointGradient
  ↓
GPU Batch Analytic Gradient 계산
```

---

### 1.2 코드 위치

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**수치적 Hessian 계산** (Line 3028-3057):
```python
# 기준 gradient 계산
grad_0 = self._joint_gradient(
    optimal_params,
    measurement_model,
    structural_model,
    choice_model
)

# 각 파라미터에 대해
for i in range(n_params):
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
    
    # Hessian i번째 행 계산
    hessian[i, :] = (grad_plus - grad_0) / epsilon
```

---

## 🔍 2. `_joint_gradient()` 함수

### 2.1 함수 정의

**파일**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**Line 2246-2361**: `_compute_gradient()` 함수

```python
def _compute_gradient(self, params: np.ndarray,
                     measurement_model,
                     structural_model,
                     choice_model) -> np.ndarray:
    """
    순수한 analytic gradient 계산 (상태 의존성 제거)
    
    Args:
        params: 파라미터 벡터 (unscaled, external)
        measurement_model: 측정모델
        structural_model: 구조모델
        choice_model: 선택모델
    
    Returns:
        gradient 벡터 (negative gradient for minimization)
    """
    # 파라미터 딕셔너리로 변환
    param_dict = self._unpack_parameters(
        params, measurement_model, structural_model, choice_model
    )
    
    # 다중 잠재변수 여부 확인
    is_multi_latent = isinstance(self.config, MultiLatentConfig)
    
    if is_multi_latent:
        # 개인 데이터 준비
        individual_ids = self.data[self.config.individual_id_column].unique()
        
        all_ind_data = []
        all_ind_draws = []
        
        for ind_id in individual_ids:
            ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
            ind_idx = np.where(individual_ids == ind_id)[0][0]
            ind_draws = self.halton_generator.get_draws()[ind_idx]
            
            all_ind_data.append(ind_data)
            all_ind_draws.append(ind_draws)
        
        # 🎯 단일 진입점으로 gradient 계산
        all_grad_dicts = self.joint_grad.compute_gradients(
            all_ind_data=all_ind_data,
            all_ind_draws=all_ind_draws,
            params_dict=param_dict,
            measurement_model=measurement_model,
            structural_model=structural_model,
            choice_model=choice_model,
            iteration_logger=self.iteration_logger,
            log_level='MINIMAL'
        )
        
        # 모든 개인의 gradient 합산
        total_grad_dict = ...  # 합산 로직
        grad_dict = total_grad_dict
    
    # 그래디언트 벡터로 변환
    grad_vector = self._pack_gradient(grad_dict, measurement_model, structural_model, choice_model)
    
    # Negative gradient (minimize -LL)
    return -grad_vector
```

---

### 2.2 핵심: `self.joint_grad.compute_gradients()`

**파일**: `src/analysis/hybrid_choice_model/iclv_models/multi_latent_gradient.py`

**MultiLatentJointGradient 클래스**:
```python
class MultiLatentJointGradient:
    """
    다중 잠재변수 결합 그래디언트 계산
    
    GPU 배치 처리 지원
    """
    
    def compute_gradients(
        self,
        all_ind_data,
        all_ind_draws,
        params_dict,
        measurement_model,
        structural_model,
        choice_model,
        iteration_logger=None,
        log_level='MINIMAL'
    ):
        """
        모든 개인의 analytic gradient 계산
        
        GPU 배치 처리 사용
        """
        if self.use_gpu and self.gpu_measurement_model is not None:
            # ✨ 완전 병렬 처리 (Advanced Indexing)
            if self.use_full_parallel:
                return self.gpu_grad_full.compute_all_individuals_gradients_full_parallel_gpu(
                    self.gpu_measurement_model,
                    all_ind_data,
                    all_ind_draws,
                    params_dict,
                    measurement_model,
                    structural_model,
                    choice_model,
                    iteration_logger=iteration_logger,
                    log_level=log_level
                )
            else:
                # 기존 완전 GPU batch 모드
                return self.gpu_grad.compute_all_individuals_gradients_full_batch_gpu(...)
        else:
            # CPU 모드
            return self._compute_gradients_cpu(...)
```

---

## 🔍 3. Analytic Gradient 계산 방법

### 3.1 GPU 배치 처리

**파일**: `src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`

**핵심 함수들**:

1. **측정모델 Gradient**:
```python
def compute_measurement_gradient_batch_gpu(
    gpu_measurement_model,
    ind_data,
    lvs_list,
    params_dict,
    weights,
    iteration_logger=None,
    log_level='MINIMAL'
):
    """
    측정모델 analytic gradient 계산 (GPU batch)
    
    ∂LL_meas/∂zeta, ∂LL_meas/∂sigma_sq
    """
    # GPU에서 analytic gradient 계산
    # 수치 미분 사용 안 함!
    ...
```

2. **구조모델 Gradient**:
```python
def compute_structural_gradient_batch_gpu(
    ind_data,
    lvs_list,
    exo_draws_list,
    params_dict,
    covariates,
    endogenous_lv,
    exogenous_lvs,
    weights
):
    """
    구조모델 analytic gradient 계산 (GPU batch)
    
    ∂LL_struct/∂gamma, ∂LL_struct/∂sigma_eta
    """
    # GPU에서 analytic gradient 계산
    ...
```

3. **선택모델 Gradient**:
```python
def compute_choice_gradient_batch_gpu(
    ind_data,
    lvs_list,
    params_dict,
    endogenous_lv,
    choice_attributes,
    weights
):
    """
    선택모델 analytic gradient 계산 (GPU batch)
    
    ∂LL_choice/∂beta, ∂LL_choice/∂theta
    """
    # Probit gradient: Mills ratio 사용
    # Analytic formula!
    
    # Mills ratio 계산
    mills_batch = phi_batch / prob_final
    sign_batch = cp.where(choices_batch == 1, 1.0, -1.0)
    
    # Weighted mills
    weighted_mills = all_weights_gpu[:, :, None] * sign_batch * mills_batch
    
    # Gradient 계산
    gradients = {}
    gradients['intercept'] = cp.sum(weighted_mills, axis=(1, 2))
    gradients['beta'] = cp.sum(weighted_mills[:, :, :, None] * attr_batch, axis=(1, 2))
    gradients['theta'] = ...  # LV coefficient gradient
    
    return gradients
```

---

### 3.2 Analytic Gradient 공식

**측정모델** (정규분포):
```
∂LL_meas/∂zeta_k = Σ_i (y_ik - alpha_k - zeta_k * LV_i) * LV_i / sigma_k^2
∂LL_meas/∂sigma_k^2 = Σ_i [-1/(2*sigma_k^2) + (y_ik - mu_ik)^2 / (2*sigma_k^4)]
```

**구조모델** (정규분포):
```
∂LL_struct/∂gamma = Σ_i (LV_endo_i - mu_struct_i) * X_i / sigma_eta^2
∂LL_struct/∂sigma_eta^2 = Σ_i [-1/(2*sigma_eta^2) + (LV_endo_i - mu_struct_i)^2 / (2*sigma_eta^4)]
```

**선택모델** (Probit):
```
∂LL_choice/∂beta = Σ_i Σ_t Mills_ratio_it * X_it
∂LL_choice/∂theta = Σ_i Σ_t Mills_ratio_it * LV_i

여기서 Mills_ratio = φ(V) / Φ(V)  (선택=1)
                   = -φ(V) / (1-Φ(V))  (선택=0)
```

---

## ✅ 4. 핵심 정리

### 4.1 Gradient 계산 방법

| 항목 | 내용 |
|------|------|
| **계산 방법** | ✅ **Analytic Gradient** |
| **수치 미분 사용** | ❌ **사용 안 함** |
| **GPU 사용** | ✅ **GPU 배치 처리** |
| **정확도** | ✅ **매우 높음** (수치 오차 없음) |
| **속도** | ✅ **빠름** (~2초/회) |

---

### 4.2 수치적 Hessian vs Analytic Gradient

**수치적 Hessian 계산**:
```
H[i,j] ≈ (g_j(θ + ε*e_i) - g_j(θ)) / ε

여기서 g_j(θ)는 ANALYTIC gradient의 j번째 성분
```

**핵심**:
- ✅ **Hessian**: 수치적 근사 (gradient의 차분)
- ✅ **Gradient**: Analytic 계산 (미분 공식 사용)

---

### 4.3 왜 Analytic Gradient를 사용하는가?

**장점**:
1. ✅ **정확도 높음**: 수치 오차 없음
2. ✅ **속도 빠름**: GPU 배치 처리
3. ✅ **안정성**: 수치 미분의 epsilon 선택 문제 없음
4. ✅ **이미 구현됨**: 최적화에서 사용 중

**비교**:
| 방법 | 정확도 | 속도 | 구현 난이도 |
|------|--------|------|------------|
| **Analytic Gradient** | ✅ 높음 | ✅ 빠름 | ⚠️ 높음 |
| **수치적 Gradient** | ⚠️ 보통 | ❌ 느림 | ✅ 낮음 |

---

## 📊 5. 계산 비용

### 5.1 Gradient 1회 계산

**구성**:
- 328명 개인
- 100 Halton draws
- GPU 배치 처리

**소요 시간**: ~2초

---

### 5.2 수치적 Hessian 계산

**Gradient 계산 횟수**:
```
기준 gradient: 1회
Perturbed gradient: 202회 (파라미터 수)
총 계산: 203회
```

**총 소요 시간**:
```
203회 × 2초 = 406초 ≈ 6.8분
```

---

## 🎯 6. 최종 답변

### 질문: 수치적 Hessian 계산 시 사용하는 gradient는 어떻게 계산되는가?

**답변**:

✅ **Analytic Gradient 사용**

1. **계산 방법**: 
   - `self._joint_gradient()` 호출
   - `self.joint_grad.compute_gradients()` 실행
   - GPU 배치 처리로 analytic gradient 계산

2. **Analytic 공식**:
   - 측정모델: 정규분포 미분 공식
   - 구조모델: 정규분포 미분 공식
   - 선택모델: Probit Mills ratio 공식

3. **장점**:
   - ✅ 정확도 매우 높음 (수치 오차 없음)
   - ✅ 속도 빠름 (~2초/회)
   - ✅ 이미 구현되어 있음

4. **수치적 Hessian**:
   - Hessian만 수치적 근사 (gradient의 차분)
   - Gradient 자체는 analytic 계산

---

**핵심**: 
- ❌ **수치적 Gradient 사용 안 함**
- ✅ **Analytic Gradient 사용**
- ✅ **GPU 배치 처리로 빠르고 정확**

---

**분석 완료 일시**: 2025-11-23

