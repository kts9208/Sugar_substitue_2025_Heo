# Gradient 검증 호환성 검토 요약

**작성일**: 2025-11-14  
**요청**: test_gpu_batch_iclv 코드의 analytic gradient를 scipy.optimize.check_grad로 검증  
**결과**: ❌ **현재 호환 불가능** (수정 필요)

---

## 핵심 문제

### 1. 구조모델 Gradient - ❌ 불호환

**현재 모델** (`test_gpu_batch_iclv.py`):
- 계층적 구조 (HC → PB → PI)
- 파라미터: `gamma_health_concern_to_perceived_benefit`, `gamma_perceived_benefit_to_purchase_intention`

**검증 스크립트** (`validate_analytic_gradient.py`, `validate_gradient_simple.py`):
- 병렬 구조 (HC, PB, PP, NK → PI)
- 파라미터: `gamma_lv` (벡터), `gamma_x` (벡터)

**호환성**: ❌ 파라미터 구조 불일치

---

### 2. 선택모델 Gradient - ❌ 불호환 (더 심각)

**현재 모델** (`test_gpu_batch_iclv.py`):
- 조절효과 모델 (PI × PP, PI × NK)
- 파라미터: `lambda_main`, `lambda_mod_perceived_price`, `lambda_mod_nutrition_knowledge`

**CPU Gradient 구현** (`gradient_calculator.py`):
- 기본 모델만 지원
- 파라미터: `lambda` (단일 값)
- **조절효과 미지원**

**호환성**: ❌ **CPU gradient가 조절효과를 지원하지 않음**

---

## 상세 분석

### CPU vs GPU Gradient 지원 현황

| 기능 | CPU Gradient | GPU Gradient | test_gpu_batch_iclv.py |
|------|-------------|-------------|------------------------|
| **측정모델** | | | |
| - Continuous Linear | ✅ 지원 | ✅ 지원 | ✅ 사용 |
| - Ordered Probit | ✅ 지원 | ✅ 지원 | ❌ 미사용 |
| **구조모델** | | | |
| - 병렬 구조 | ✅ 지원 | ✅ 지원 | ❌ 미사용 |
| - 계층적 구조 | ✅ 지원 | ✅ 지원 | ✅ 사용 |
| **선택모델** | | | |
| - 기본 모델 | ✅ 지원 | ✅ 지원 | ❌ 미사용 |
| - 조절효과 | ❌ **미지원** | ✅ 지원 | ✅ 사용 |

**결론**: CPU gradient는 조절효과를 지원하지 않으므로, `test_gpu_batch_iclv.py`에서 `use_analytic_gradient=True`를 사용할 수 없음.

---

## 코드 증거

### CPU Gradient - 조절효과 미지원

**파일**: `src/analysis/hybrid_choice_model/iclv_models/gradient_calculator.py`

```python
class ChoiceGradient:
    """
    선택모델 (Binary Probit) 그래디언트 계산
    
    Model: P(choice=1) = Φ(V), V = β*X + λ*LV  # ← 기본 모델만
    """
    
    def compute_gradient(self, data, lv, params, choice_attributes):
        """
        Args:
            params: {'intercept': float, 'beta': np.ndarray, 'lambda': float}
                                                                # ↑ 단일 lambda만
        Returns:
            {'grad_intercept': float, 'grad_beta': np.ndarray, 'grad_lambda': float}
        """
        intercept = params['intercept']
        beta = params['beta']
        lambda_lv = params['lambda']  # ← 'lambda' 키만 지원
        
        # ... (조절효과 계산 로직 없음)
        
        return {
            'grad_intercept': grad_intercept,
            'grad_beta': grad_beta,
            'grad_lambda': grad_lambda  # ← 단일 lambda만 반환
        }
```

### GPU Gradient - 조절효과 지원

**파일**: `src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`

```python
def compute_choice_gradient_full_batch_gpu(...):
    """
    선택모델 Gradient - 완전 GPU Batch (조절효과 지원)
    """
    # 조절효과 모델
    if moderation_enabled:
        # lambda_main
        gradients['grad_lambda_main'] = ...
        
        # lambda_mod_{lv_name}
        for mod_lv in moderator_lvs:
            gradients[f'grad_lambda_mod_{mod_lv}'] = ...
    else:
        # 기본 모델
        gradients['grad_lambda'] = ...
```

---

## 해결 방안

### 방안 1: CPU Gradient에 조절효과 지원 추가 (권장)

**장점**:
- 실제 사용 모델(조절효과)을 analytic gradient로 검증 가능
- 향후 CPU 모드에서도 조절효과 사용 가능

**단점**:
- `gradient_calculator.py` 수정 필요 (약 50-100 라인)

**작업량**: 중간 (2-3시간)

---

### 방안 2: 검증 스크립트를 기본 모델로 변경

**장점**:
- 코드 수정 최소화

**단점**:
- 실제 사용 모델(조절효과)을 검증하지 못함
- 연구 목적과 불일치

**작업량**: 적음 (30분)

---

### 방안 3: 현재는 Numerical Gradient 사용 (임시 해결책)

**장점**:
- 즉시 사용 가능
- 코드 수정 불필요

**단점**:
- 속도 느림 (analytic gradient 대비 10-100배)
- Gradient 정확성 검증 불가

**작업량**: 없음 (설정만 변경)

```python
# test_gpu_batch_iclv.py
estimation_config = EstimationConfig(
    optimizer='BHHH',
    use_analytic_gradient=False,  # ✅ Numerical gradient 사용
    # ...
)
```

---

## 권장 조치

### 즉시 조치 (현재 실행 가능하게)

1. **`test_gpu_batch_iclv.py` 설정 변경**:
   ```python
   use_analytic_gradient=False  # Numerical gradient 사용
   ```

2. **실행 확인**:
   ```bash
   python scripts/test_gpu_batch_iclv.py
   ```

### 단기 조치 (1-2일 내)

1. **CPU Gradient에 조절효과 지원 추가**:
   - `gradient_calculator.py`의 `ChoiceGradient` 클래스 수정
   - GPU gradient 구현 참고 (`gpu_gradient_batch.py`)

2. **검증 스크립트 수정**:
   - 계층적 구조 + 조절효과 모델로 변경
   - `test_gpu_batch_iclv.py`와 동일한 설정 사용

3. **Gradient 검증 실행**:
   ```bash
   python scripts/validate_gradient_simple.py
   python scripts/validate_analytic_gradient.py
   ```

### 장기 조치 (1주일 내)

1. **GPU Gradient 검증**:
   - GPU analytic gradient 정확성 검증
   - CPU vs GPU gradient 비교

2. **통합 테스트**:
   - 전체 모델 추정 + gradient 검증
   - 수렴 속도 비교 (analytic vs numerical)

---

## 기술적 세부사항

### CPU Gradient 조절효과 구현 가이드

**수정 파일**: `src/analysis/hybrid_choice_model/iclv_models/gradient_calculator.py`

**수정 위치**: `ChoiceGradient.compute_gradient()` (라인 335-399)

**추가 로직**:
```python
def compute_gradient(self, data, lv, params, choice_attributes):
    """
    선택모델 그래디언트 계산 (조절효과 지원)
    
    기본 모델:
        V = β*X + λ*LV
    
    조절효과 모델:
        V = β*X + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)
    """
    intercept = params['intercept']
    beta = params['beta']
    
    # ✅ 조절효과 여부 확인
    if 'lambda_main' in params:
        # 조절효과 모델
        lambda_main = params['lambda_main']
        
        # LV가 딕셔너리인지 확인 (다중 LV)
        if isinstance(lv, dict):
            lv_main = lv[self.main_lv]  # 주 LV
            
            # 조절효과 계산
            moderation_term = 0.0
            for mod_lv_name in self.moderator_lvs:
                lambda_mod = params[f'lambda_mod_{mod_lv_name}']
                lv_mod = lv[mod_lv_name]
                moderation_term += lambda_mod * lv_main * lv_mod
            
            # 효용
            V = intercept + np.dot(beta, X) + lambda_main * lv_main + moderation_term
            
            # Gradient 계산
            # ∂V/∂λ_main = LV_main
            # ∂V/∂λ_mod_i = LV_main * LV_mod_i
            
            grad_lambda_main = common_term * lv_main
            grad_lambda_mod = {}
            for mod_lv_name in self.moderator_lvs:
                lv_mod = lv[mod_lv_name]
                grad_lambda_mod[f'grad_lambda_mod_{mod_lv_name}'] = common_term * lv_main * lv_mod
            
            return {
                'grad_intercept': grad_intercept,
                'grad_beta': grad_beta,
                'grad_lambda_main': grad_lambda_main,
                **grad_lambda_mod
            }
    else:
        # 기본 모델 (기존 로직)
        lambda_lv = params['lambda']
        # ...
```

**참고 구현**: `gpu_gradient_batch.py`, 라인 1500-1650

---

## 검증 절차 (CPU Gradient 수정 후)

### 1. 단위 테스트
```bash
python tests/test_gradient_calculation.py
```

### 2. 간단한 검증
```bash
python scripts/validate_gradient_simple.py
```

**예상 출력**:
```
검증 중... (epsilon=1e-05)
  전체 Gradient Error (norm): 2.345e-05

  파라미터별 상세 비교:
    Param[ 0]: Analytic=  0.123456, Numerical=  0.123450, RelDiff=5.00e-05
    ...

  통과: 145/150 (96.7%)
  평균 상대 오차: 3.456e-05
  최대 상대 오차: 1.234e-03

  ✓ 검증 통과! (epsilon=1e-05)
```

### 3. 상세 검증
```bash
python scripts/validate_analytic_gradient.py
```

**예상 출력**:
```
7. scipy.optimize.check_grad 검증
  Epsilon = 1.00e-05: Gradient Error = 2.345e-05 ✓
  Epsilon = 1.00e-06: Gradient Error = 2.456e-06 ✓

8. 파라미터별 상세 검증
  ✓ 모든 파라미터 검증 통과!
```

---

## 결론

### 현재 상태
- ❌ CPU analytic gradient는 조절효과를 지원하지 않음
- ❌ 기존 검증 스크립트는 병렬 구조 + 기본 모델만 지원
- ✅ GPU analytic gradient는 조절효과를 지원함

### 즉시 조치
```python
# test_gpu_batch_iclv.py
use_analytic_gradient=False  # Numerical gradient 사용
```

### 향후 작업
1. CPU gradient에 조절효과 지원 추가
2. 검증 스크립트 수정 (계층적 + 조절효과)
3. Gradient 검증 실행 및 확인

---

**작성자**: Augment Agent  
**상세 보고서**: `docs/GRADIENT_VALIDATION_COMPATIBILITY_REPORT.md`

