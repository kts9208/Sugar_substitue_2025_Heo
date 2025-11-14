# Gradient Validation 호환성 검토 보고서

**작성일**: 2025-11-14  
**대상 코드**: `test_gpu_batch_iclv.py`에서 사용되는 analytic gradient 계산  
**검증 도구**: `scipy.optimize.check_grad`

---

## 1. 요약

### 현재 상황
- `test_gpu_batch_iclv.py`는 **계층적 구조 + 조절효과 모델**을 사용
- 기존 gradient 검증 스크립트 (`validate_analytic_gradient.py`, `validate_gradient_simple.py`)는 **병렬 구조 + 기본 모델**을 가정
- **호환성 문제**: 구조모델과 선택모델의 파라미터 구조가 다름

### 호환성 상태

| 컴포넌트 | 현재 모델 (test_gpu_batch_iclv.py) | 검증 스크립트 | 호환성 |
|---------|-----------------------------------|--------------|--------|
| **측정모델** | Continuous Linear (5개 LV) | ✅ 지원 | ✅ 호환 |
| **구조모델** | 계층적 구조 (HC→PB→PI) | ❌ 병렬 구조만 지원 | ❌ **불호환** |
| **선택모델** | 조절효과 (λ_main + λ_mod) | ❌ 기본 모델만 지원 | ❌ **불호환** |

---

## 2. 상세 분석

### 2.1 측정모델 Gradient - ✅ 호환

#### 현재 구현 (`MultiLatentMeasurementGradient`)
```python
def compute_gradient(self, data, latent_vars, params):
    """
    Args:
        latent_vars: {lv_name: lv_value}  # 5개 LV
        params: {lv_name: {'zeta': ..., 'sigma_sq': ...}}
    
    Returns:
        {lv_name: {'grad_zeta': ..., 'grad_sigma_sq': ...}}
    """
```

#### 검증 스크립트 호환성
- ✅ `validate_analytic_gradient.py`는 다중 LV 지원
- ✅ Continuous Linear 방식 지원
- ✅ 파라미터 구조 일치

---

### 2.2 구조모델 Gradient - ❌ 불호환

#### 현재 모델 (계층적 구조)
```python
# test_gpu_batch_iclv.py
structural_config = MultiLatentStructuralConfig(
    endogenous_lv='purchase_intention',
    exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
    hierarchical_paths=[
        {'target': 'perceived_benefit', 'predictors': ['health_concern']},
        {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
    ]
)

# 파라미터 구조:
params['structural'] = {
    'gamma_health_concern_to_perceived_benefit': 0.5,
    'gamma_perceived_benefit_to_purchase_intention': 0.3
}
```

#### 검증 스크립트 (병렬 구조)
```python
# validate_gradient_simple.py (라인 174-181)
structural_config = MultiLatentStructuralConfig(
    endogenous_lv='purchase_intention',
    exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
    hierarchical_paths=None,  # ✅ 병렬 구조 사용
)

# 파라미터 구조:
params['structural'] = {
    'gamma_lv': [0.1, 0.2, 0.3, 0.4],  # 4개 외생 LV
    'gamma_x': []  # 공변량 없음
}
```

#### 문제점
1. **파라미터 키 불일치**:
   - 계층적: `gamma_{pred}_to_{target}` (개별 스칼라)
   - 병렬: `gamma_lv` (벡터), `gamma_x` (벡터)

2. **Gradient 계산 로직 차이**:
   - 계층적: 경로별 개별 계산
   - 병렬: 모든 외생 LV를 한번에 계산

3. **`_pack_gradient()` 불일치**:
   ```python
   # 계층적 구조 (SimultaneousEstimator._pack_gradient, 라인 2200-2230)
   for path in structural_model.hierarchical_paths:
       param_key = f'gamma_{pred}_to_{target}'
       gradient_list.append(np.array([grad_dict['structural'][f'grad_{param_key}']]))
   
   # 병렬 구조
   gradient_list.append(grad_dict['structural']['grad_gamma_lv'])
   gradient_list.append(grad_dict['structural']['grad_gamma_x'])
   ```

---

### 2.3 선택모델 Gradient - ❌ 불호환

#### 현재 모델 (조절효과)
```python
# test_gpu_batch_iclv.py
choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price'],
    moderation_enabled=True,  # 디폴트
    main_lv='purchase_intention',
    moderator_lvs=['perceived_price', 'nutrition_knowledge']
)

# 파라미터 구조:
params['choice'] = {
    'intercept': 0.0,
    'beta': [0.1, 0.2, -0.3],
    'lambda_main': 0.5,
    'lambda_mod_perceived_price': 0.1,
    'lambda_mod_nutrition_knowledge': 0.2
}
```

#### 검증 스크립트 (기본 모델)
```python
# validate_gradient_simple.py (라인 184-187)
choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price'],
    moderation_enabled=False  # ✅ 조절효과 비활성화
)

# 파라미터 구조:
params['choice'] = {
    'intercept': 0.0,
    'beta': [0.1, 0.2, -0.3],
    'lambda': 0.5  # 단일 lambda
}
```

#### 문제점
1. **파라미터 키 불일치**:
   - 조절효과: `lambda_main`, `lambda_mod_{lv_name}`
   - 기본: `lambda`

2. **Gradient 계산 로직 차이**:
   - 조절효과: 주효과 + 상호작용 항
   - 기본: 단순 선형 효과

3. **`_pack_gradient()` 불일치**:
   ```python
   # 조절효과 (SimultaneousEstimator._pack_gradient, 라인 2240-2260)
   gradient_list.append(np.array([grad_dict['choice']['grad_lambda_main']]))
   for mod_lv in choice_model.moderator_lvs:
       gradient_list.append(np.array([grad_dict['choice'][f'grad_lambda_mod_{mod_lv}']]))
   
   # 기본 모델
   gradient_list.append(np.array([grad_dict['choice']['grad_lambda']]))
   ```

---

## 3. 호환성 확보 방안

### 방안 A: 검증 스크립트 수정 (권장)

**장점**:
- 실제 사용 모델(계층적 + 조절효과)을 검증
- 프로덕션 코드 변경 없음

**단점**:
- 검증 스크립트 수정 필요

**수정 사항**:
1. `validate_analytic_gradient.py` 수정:
   - 계층적 구조 활성화
   - 조절효과 활성화

2. `validate_gradient_simple.py` 수정:
   - 계층적 구조 활성화
   - 조절효과 활성화

### 방안 B: test_gpu_batch_iclv.py 수정

**장점**:
- 기존 검증 스크립트 재사용

**단점**:
- 실제 사용하려는 모델 구조 변경
- 연구 목적과 불일치

---

## 4. 구체적 수정 방법 (방안 A)

### 4.1 `validate_analytic_gradient.py` 수정

**파일**: `scripts/validate_analytic_gradient.py`

**수정 위치**: 라인 60-120 (설정 부분)

**변경 전**:
```python
# 구조모델 설정 (병렬 구조)
structural_config = MultiLatentStructuralConfig(
    endogenous_lv='purchase_intention',
    exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
    covariates=[],
    hierarchical_paths=None,  # 병렬 구조
    error_variance=1.0
)

# 선택모델 설정 (기본 모델)
choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price'],
    moderation_enabled=False
)
```

**변경 후**:
```python
# 구조모델 설정 (계층적 구조) - test_gpu_batch_iclv.py와 동일
structural_config = MultiLatentStructuralConfig(
    endogenous_lv='purchase_intention',
    exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
    covariates=[],
    hierarchical_paths=[
        {'target': 'perceived_benefit', 'predictors': ['health_concern']},
        {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
    ],
    error_variance=1.0
)

# 선택모델 설정 (조절효과) - test_gpu_batch_iclv.py와 동일
choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price'],
    moderation_enabled=True,  # 디폴트 사용
    # main_lv='purchase_intention',  # 디폴트
    # moderator_lvs=['perceived_price', 'nutrition_knowledge']  # 디폴트
)
```

### 4.2 `validate_gradient_simple.py` 수정

**파일**: `scripts/validate_gradient_simple.py`

**수정 위치**: 라인 174-187

**동일한 변경 적용**

---

## 5. 검증 절차

### 5.1 수정 후 실행 순서

1. **검증 스크립트 수정**:
   ```bash
   # validate_analytic_gradient.py 수정
   # validate_gradient_simple.py 수정
   ```

2. **간단한 검증 실행** (빠른 확인):
   ```bash
   python scripts/validate_gradient_simple.py
   ```

3. **상세 검증 실행** (전체 확인):
   ```bash
   python scripts/validate_analytic_gradient.py
   ```

### 5.2 예상 결과

**성공 기준**:
- `check_grad` error < 1e-3 (0.1% 이내)
- 파라미터별 상대 오차 < 1% (대부분)

**실패 시 조치**:
1. Gradient 계산 로직 확인
2. 파라미터 packing/unpacking 순서 확인
3. Importance weighting 적용 확인

---

## 6. 추가 고려사항

### 6.1 GPU vs CPU Gradient

현재 `test_gpu_batch_iclv.py`는:
- `use_analytic_gradient=True` (CPU gradient 사용)
- GPU는 우도 계산만 사용

**검증 범위**:
- ✅ CPU analytic gradient 검증 (현재 목표)
- ⏳ GPU analytic gradient 검증 (향후 과제)

### 6.2 계층적 구조 Gradient 구현 확인

**확인 필요**:
1. `MultiLatentStructuralGradient.compute_gradient()` - 계층적 구조 지원 여부
2. `gpu_gradient_batch.py` - 계층적 구조 GPU gradient 지원 여부

**현재 상태** (코드 검토 결과):
- ✅ CPU gradient: 계층적 구조 지원 (`multi_latent_gradient.py`)
- ✅ GPU gradient: 계층적 구조 지원 (`gpu_gradient_batch.py`, 라인 447-516)

### 6.3 조절효과 Gradient 구현 확인

**확인 필요**:
1. `ChoiceGradient.compute_gradient()` - 조절효과 지원 여부
2. `gpu_gradient_batch.py` - 조절효과 GPU gradient 지원 여부

**현재 상태** (코드 검토 결과):
- ❌ **CPU gradient: 조절효과 미지원** (`gradient_calculator.py`, 라인 316-399)
  - `ChoiceGradient.compute_gradient()`: `lambda` 파라미터만 지원
  - `lambda_main`, `lambda_mod_{lv}` 파라미터 미지원
- ✅ GPU gradient: 조절효과 지원 (`gpu_gradient_batch.py`, 라인 1500-1650)

**문제점**:
- CPU analytic gradient는 조절효과를 지원하지 않음
- `test_gpu_batch_iclv.py`에서 `use_analytic_gradient=True`를 사용하면 **파라미터 키 오류 발생**
  - `KeyError: 'lambda'` (실제로는 `lambda_main`이 필요)

---

## 7. 결론 및 권장사항

### 7.1 즉시 조치 필요 사항

1. **CPU Gradient 조절효과 지원 추가**:
   - `gradient_calculator.py`의 `ChoiceGradient` 클래스 수정
   - 조절효과 모델 gradient 계산 로직 추가

2. **검증 스크립트 수정**:
   - 계층적 구조 + 조절효과 모델로 변경
   - `test_gpu_batch_iclv.py`와 동일한 설정 사용

### 7.2 검증 우선순위

1. **1순위**: CPU gradient 조절효과 지원 추가
2. **2순위**: 검증 스크립트 수정 및 실행
3. **3순위**: GPU gradient 검증 (별도 스크립트)

### 7.3 최종 권장사항

**현재 상황**:
- `test_gpu_batch_iclv.py`는 조절효과 모델을 사용하지만, CPU analytic gradient는 조절효과를 지원하지 않음
- 따라서 **현재는 numerical gradient를 사용해야 함**

**단기 해결책**:
```python
# test_gpu_batch_iclv.py
estimation_config = EstimationConfig(
    optimizer='BHHH',
    use_analytic_gradient=False,  # ✅ Numerical gradient 사용
    # ...
)
```

**장기 해결책**:
1. CPU gradient에 조절효과 지원 추가
2. 검증 스크립트로 정확성 확인
3. Analytic gradient 활성화

---

**작성자**: Augment Agent  
**검토 필요**: CPU gradient 조절효과 구현 확인

