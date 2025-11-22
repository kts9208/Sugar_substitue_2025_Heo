# 구조모델 우도 계산 수정

## 📌 문제점

**기존 구현**: 구조모델 우도를 전체 로그우도에 합산

```python
# ❌ 잘못된 구현
draw_ll = ll_measurement + ll_choice + ll_structural
```

**문제**: 구조모델은 **잠재변수 값을 생성하는 역할**만 하며, 별도의 우도 기여가 없음

---

## ✅ 수정 내용

### 1. 구조모델의 역할

구조모델은 다음 공식으로 잠재변수를 생성합니다:

```
LV = γ * X + η,  η ~ N(0, σ²)
```

여기서:
- `γ`: 경로계수 (path coefficients)
- `X`: 사회인구학적 변수
- `η`: 오차항 (draw에서 샘플링)

**역할**: 잠재변수 값 생성 → 측정모델과 선택모델에 전달

---

### 2. 우도 계산 수정

#### **수정 전**

```python
# 1. 구조모델: LV 생성
lv = structural_model.predict(ind_data, params, draw)

# 2. 측정모델 우도: P(Indicators|LV)
ll_measurement = measurement_model.log_likelihood(ind_data, lv, params)

# 3. 선택모델 우도: P(Choice|LV, X)
ll_choice = choice_model.log_likelihood(ind_data, lv, params)

# 4. 구조모델 우도: P(LV|X)
ll_structural = structural_model.log_likelihood(ind_data, lv, params, draw)

# ❌ 잘못된 결합 우도
draw_ll = ll_measurement + ll_choice + ll_structural
```

#### **수정 후**

```python
# 1. 구조모델: LV 생성
lv = structural_model.predict(ind_data, params, draw)

# 2. 측정모델 우도: P(Indicators|LV)
ll_measurement = measurement_model.log_likelihood(ind_data, lv, params)

# 3. 선택모델 우도: P(Choice|LV, X)
ll_choice = choice_model.log_likelihood(ind_data, lv, params)

# ✅ 올바른 결합 우도 (구조모델 우도 제외)
draw_ll = ll_measurement + ll_choice
```

---

### 3. 수정된 파일 목록

1. **`src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`**
   - 라인 86-98: `compute_individual_likelihood_gpu()` 함수
   - 라인 1169-1178: `compute_all_individuals_likelihood_full_batch_gpu()` 함수
   - 라인 1133-1135: 우도 성분 누적 변수
   - 라인 1215-1219: 로깅 출력

2. **`src/analysis/hybrid_choice_model/iclv_models/likelihood_calculator.py`**
   - 라인 185-187: `_compute_single_draw_likelihood()` 함수

3. **`src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator.py`**
   - 라인 205-212: `_joint_log_likelihood()` 함수

4. **`src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`**
   - 라인 101-105: 개인별 우도 계산

5. **`src/analysis/hybrid_choice_model/iclv_models/simultaneous_gpu_batch_estimator.py`**
   - 라인 492-499: 디버깅 로그

---

## 📊 이론적 근거

### ICLV 모델의 우도 구조

ICLV 모델의 결합 우도는 다음과 같이 정의됩니다:

```
L = P(Choice, Indicators | X)
  = ∫ P(Choice | LV, X) × P(Indicators | LV) × P(LV | X) dLV
```

**시뮬레이션 기반 추정**에서는 적분을 Monte Carlo 샘플링으로 근사합니다:

```
L ≈ (1/R) Σᵣ P(Choice | LVᵣ, X) × P(Indicators | LVᵣ)
```

여기서:
- `LVᵣ = γ * X + ηᵣ` (구조모델에서 생성)
- `P(LV | X)`는 **샘플링 과정에 이미 반영됨**

따라서 **구조모델 우도를 별도로 합산하면 이중 계산**이 됩니다!

---

## 🎯 결론

### 수정 전 vs 수정 후

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| **결합 우도** | `ll_measurement + ll_choice + ll_structural` | `ll_measurement + ll_choice` |
| **구조모델 역할** | 우도 기여 | LV 생성만 |
| **이론적 정확성** | ❌ 이중 계산 | ✅ 올바른 공식 |

### 영향

- ✅ **로그우도 값**: 구조모델 우도만큼 감소 (더 정확한 값)
- ✅ **파라미터 추정**: 구조모델 우도의 영향 제거 (더 정확한 추정)
- ✅ **모델 비교**: AIC/BIC가 올바른 우도 기반으로 계산됨

---

## 📝 참고 문헌

- Ben-Akiva, M., et al. (2002). "Hybrid Choice Models: Progress and Challenges"
- Train, K. (2009). "Discrete Choice Methods with Simulation"
- Bhat, C. R., & Dubey, S. K. (2014). "A new estimation approach to integrate latent psychological constructs in choice modeling"

---

**작성일**: 2025-11-22  
**작성자**: Sugar Substitute Research Team

