# Biogeme 라이브러리 BHHH 구현 및 전환 검토

## 📋 요약

**질문**: 자체 구현 대신 Biogeme 라이브러리를 사용한 BHHH 구현 및 전환 가능성 검토

**결론**: 
- ⚠️ **Biogeme는 ICLV 모델을 직접 지원하지 않음**
- ✅ **Biogeme는 BHHH 계산을 자동으로 수행**
- ❌ **현재 자체 구현이 더 적합함** (복잡한 ICLV 구조 때문)
- 💡 **부분 활용 가능**: 단순 선택모델 부분만 Biogeme 사용

---

## 1. Biogeme 라이브러리 개요

### 1.1. Biogeme란?

**Biogeme** (BIOgraphical GEneration of Models for Estimation)
- 개발: EPFL (École Polytechnique Fédérale de Lausanne)
- 개발자: Michel Bierlaire 교수
- 목적: 이산선택모델(Discrete Choice Models) 최대우도추정
- 언어: Python + C++ (성능 최적화)
- 버전: 3.3.1 (2025년 기준)

### 1.2. 주요 기능

| 기능 | 지원 여부 | 비고 |
|------|----------|------|
| **Logit 모델** | ✅ 완전 지원 | MNL, Nested, Cross-nested |
| **Probit 모델** | ✅ 지원 | Binary, Ordinal |
| **Mixed Logit** | ✅ 지원 | Random parameters |
| **Latent Class** | ✅ 지원 | Discrete mixture |
| **ICLV 모델** | ⚠️ 부분 지원 | 수동 구현 필요 |
| **BHHH 계산** | ✅ 자동 | `estimate()` 후 자동 계산 |
| **최적화 알고리즘** | ✅ 다양 | Newton, BFGS, scipy |

---

## 2. Biogeme의 BHHH 구현

### 2.1. 자동 BHHH 계산

Biogeme는 **추정 완료 후 자동으로 BHHH 행렬을 계산**합니다:

```python
from biogeme import biogeme as bio
from biogeme.expressions import Beta
from biogeme.models import loglogit

# 모델 정의
biogeme_model = bio.BIOGEME(database, log_probability)

# 추정 (BHHH 자동 계산)
results = biogeme_model.estimate()

# BHHH 행렬 접근
bhhh_matrix = results.data.bhhh  # BHHH 행렬
robust_se = results.getRobustStdErr()  # BHHH 기반 표준오차
```

**내부 동작**:
```python
# biogeme.biogeme.estimate() 내부
def estimate(self):
    # 1. 최적화 (Newton/BFGS)
    algorithm_results = model_estimation(...)
    
    # 2. BHHH 자동 계산
    logger.info('Calculate second derivatives and BHHH')
    f_g_h_b = self.function_evaluator.evaluate(
        the_betas=optimal_betas,
        gradient=True,
        hessian=True,
        bhhh=True  # ✅ BHHH 자동 계산
    )
    
    # 3. 결과 저장
    raw_results = RawEstimationResults(
        bhhh=f_g_h_b.bhhh,  # BHHH 행렬
        ...
    )
```

### 2.2. BHHH 공식

Biogeme는 표준 BHHH 공식을 사용합니다:

```
BHHH = Σ_i (∂LL_i/∂θ) × (∂LL_i/∂θ)^T
```

여기서:
- `LL_i`: 개인 i의 log-likelihood
- `∂LL_i/∂θ`: 개인 i의 gradient
- `Σ_i`: 모든 개인에 대한 합

### 2.3. 최적화 알고리즘

Biogeme가 지원하는 최적화 알고리즘:

```python
# biogeme.toml 설정 파일
[Estimation]
optimization_algorithm = "automatic"  # 또는 아래 중 선택

# 사용 가능한 알고리즘:
# - "automatic": 자동 선택 (Newton 또는 BFGS)
# - "scipy": scipy.optimize 사용
# - "TR-newton": Trust Region Newton
# - "TR-BFGS": Trust Region BFGS
# - "simple_bounds": Newton/BFGS with simple bounds (기본값)
# - "LS-newton": Line Search Newton
# - "LS-BFGS": Line Search BFGS
```

**기본 알고리즘**: Hybrid Newton/BFGS with Trust Region
- Newton 방법 사용 (Hessian 계산)
- BFGS로 fallback (Hessian 계산 실패 시)
- Trust Region으로 안정성 확보

---

## 3. Biogeme의 ICLV 지원 현황

### 3.1. 공식 문서 확인

Biogeme 공식 문서 (https://biogeme.epfl.ch/sphinx/auto_examples/latent/index.html):

**지원하는 Hybrid Choice Model 예제**:
1. ✅ **MIMIC 모델** (Multiple Indicators Multiple Causes)
2. ✅ **측정방정식** (Measurement equations)
3. ✅ **구조방정식** (Structural equations)
4. ✅ **선택모델** (Choice model)

**하지만**:
- ❌ **동시추정(Simultaneous Estimation) 자동화 없음**
- ❌ **다중 잠재변수 자동 처리 없음**
- ⚠️ **수동으로 likelihood 함수 구성 필요**

### 3.2. Biogeme ICLV 구현 방식

Biogeme에서 ICLV를 구현하려면 **수동으로 결합 likelihood를 정의**해야 합니다:

```python
from biogeme.expressions import Beta, bioDraws, MonteCarlo
from biogeme.models import loglogit
import biogeme.biogeme as bio

# 1. 잠재변수 정의 (구조방정식)
# LV = β_0 + β_1 * X_1 + β_2 * X_2 + ε
omega = bioDraws('omega', 'NORMAL')  # 오차항
LV = (beta_lv_const + 
      beta_lv_x1 * X1 + 
      beta_lv_x2 * X2 + 
      sigma_lv * omega)

# 2. 측정방정식 (Ordered Probit)
# 수동으로 각 indicator의 likelihood 정의
def ordered_probit_prob(indicator, lv, thresholds):
    # 각 카테고리별 확률 계산
    prob_cat1 = bioNormalCdf((thresholds[0] - lv) / sigma_ind)
    prob_cat2 = bioNormalCdf((thresholds[1] - lv) / sigma_ind) - prob_cat1
    # ... (모든 카테고리)
    return prob_cat1 * (indicator == 1) + prob_cat2 * (indicator == 2) + ...

# 3. 선택모델 (Logit)
V_alt1 = asc_alt1 + beta_price * price + beta_lv * LV
V_alt2 = asc_alt2 + beta_quality * quality
V = {1: V_alt1, 2: V_alt2}
prob_choice = loglogit(V, av, choice)

# 4. 결합 likelihood (수동 구성)
# LL = LL_measurement + LL_choice
prob_measurement = (ordered_probit_prob(ind1, LV, tau1) * 
                   ordered_probit_prob(ind2, LV, tau2) * 
                   ordered_probit_prob(ind3, LV, tau3))

# 5. Monte Carlo 적분 (잠재변수 적분)
joint_prob = prob_measurement * prob_choice
integrated_prob = MonteCarlo(joint_prob)  # E[prob | draws]

# 6. Log-likelihood
logprob = log(integrated_prob)

# 7. Biogeme 추정
biogeme_model = bio.BIOGEME(database, logprob)
results = biogeme_model.estimate()
```

### 3.3. 현재 자체 구현과 비교

| 측면 | 현재 자체 구현 | Biogeme 구현 |
|------|---------------|-------------|
| **다중 잠재변수** | ✅ 자동 처리 (5개 LV) | ❌ 수동 구성 필요 |
| **Ordered Probit** | ✅ 자동 계산 | ⚠️ 수동 확률 계산 |
| **GPU 가속** | ✅ CuPy 배치 처리 | ❌ 지원 안 함 |
| **Analytic Gradient** | ✅ 구현됨 | ⚠️ 수동 미분 필요 |
| **BHHH 계산** | ✅ 구현됨 | ✅ 자동 계산 |
| **코드 복잡도** | 중간 (자동화됨) | 높음 (수동 구성) |
| **유지보수** | 자체 관리 | 커뮤니티 지원 |

---

## 4. 전환 가능성 분석

### 4.1. 완전 전환 (❌ 권장하지 않음)

**이유**:
1. ❌ **ICLV 자동화 없음**: 모든 likelihood를 수동으로 구성해야 함
2. ❌ **다중 잠재변수 복잡**: 5개 LV × 각 3개 indicator = 15개 측정방정식 수동 작성
3. ❌ **GPU 가속 불가**: Biogeme는 CPU만 지원
4. ❌ **성능 저하**: 현재 GPU 배치 처리 (90초) → Biogeme CPU (예상 수 시간)
5. ❌ **코드 재작성**: 전체 시스템 재구현 필요

**예상 작업량**:
- 측정방정식 수동 구현: 15개 × 5 카테고리 = 75개 확률 계산
- 구조방정식 수동 구현: 5개 LV × 경로 수
- 선택모델 수동 구현: 3개 대안 × 속성
- 결합 likelihood 수동 구성
- **총 예상 시간: 2-3주**

### 4.2. 부분 전환 (⚠️ 제한적 활용)

**가능한 시나리오**:

#### 시나리오 1: 선택모델만 Biogeme 사용
```python
# 1단계: 자체 구현으로 잠재변수 추정
estimator = GPUBatchEstimator(config)
results = estimator.estimate(data)
lv_scores = results['latent_variable_scores']  # 잠재변수 점수

# 2단계: Biogeme로 선택모델 추정 (LV를 설명변수로)
import biogeme.database as db
import biogeme.biogeme as bio

# 데이터 준비
bio_data = db.Database('choice_data', choice_data_with_lv)

# 선택모델 정의
V_alt1 = asc_alt1 + beta_price * price + beta_lv * LV_score
V_alt2 = asc_alt2 + beta_quality * quality
V = {1: V_alt1, 2: V_alt2}
logprob = loglogit(V, av, choice)

# Biogeme 추정
biogeme_model = bio.BIOGEME(bio_data, logprob)
choice_results = biogeme_model.estimate()
```

**장점**:
- ✅ Biogeme의 검증된 선택모델 사용
- ✅ BHHH 자동 계산
- ✅ 다양한 최적화 알고리즘 선택 가능

**단점**:
- ❌ 2단계 추정 (동시추정 아님)
- ❌ 잠재변수 불확실성 무시
- ❌ 비효율적 (두 번 추정)

#### 시나리오 2: 검증 목적으로 Biogeme 사용
```python
# 자체 구현 결과
our_results = estimator.estimate(data)

# Biogeme로 동일 모델 추정 (검증)
biogeme_results = biogeme_model.estimate()

# 결과 비교
compare_results(our_results, biogeme_results)
```

**장점**:
- ✅ 결과 검증 가능
- ✅ 구현 정확성 확인

**단점**:
- ❌ 추가 작업 필요
- ❌ ICLV 전체 검증 불가 (Biogeme가 ICLV 자동화 미지원)

### 4.3. 하이브리드 접근 (💡 권장)

**현재 자체 구현 유지 + Biogeme 참고**:

1. ✅ **BHHH 계산 로직 참고**: Biogeme 소스코드에서 BHHH 구현 확인
2. ✅ **최적화 알고리즘 참고**: Trust Region 구현 방식 학습
3. ✅ **표준오차 계산 참고**: Robust SE 계산 방식 확인
4. ✅ **테스트 케이스 활용**: Biogeme 예제로 단순 모델 검증

---

## 5. 권장 사항

### 5.1. 현재 자체 구현 유지 (✅ 강력 권장)

**이유**:
1. ✅ **이미 완성도 높음**: GPU 가속, Analytic gradient, BHHH 모두 구현됨
2. ✅ **성능 우수**: 90초 vs Biogeme 예상 수 시간
3. ✅ **유연성**: 복잡한 ICLV 구조 자유롭게 구현 가능
4. ✅ **GPU 활용**: CuPy 배치 처리로 대규모 데이터 처리 가능
5. ✅ **유지보수 용이**: 전체 코드 제어 가능

### 5.2. Biogeme 참고 활용 (💡 권장)

**활용 방법**:

#### 1. BHHH 계산 검증
```python
# Biogeme 소스코드 참고
# https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/function_output.py

# 현재 구현 검증
def verify_bhhh_calculation():
    # 1. 개인별 gradient 계산
    individual_gradients = []
    for i in range(n_individuals):
        grad_i = compute_individual_gradient(...)
        individual_gradients.append(grad_i)
    
    # 2. BHHH 계산
    bhhh = np.zeros((n_params, n_params))
    for grad in individual_gradients:
        bhhh += np.outer(grad, grad)
    
    # 3. Biogeme 방식과 동일한지 확인
    assert np.allclose(bhhh, biogeme_bhhh)
```

#### 2. 최적화 알고리즘 개선
```python
# Biogeme의 Trust Region 구현 참고
# https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/optimization.py

# 현재 scipy.optimize.minimize 대신 Trust Region 구현 고려
from biogeme.optimization import algorithms

# Biogeme 알고리즘 사용 (선택모델만)
algorithm = algorithms.get('TR-newton')
```

#### 3. 표준오차 계산 검증
```python
# Biogeme의 Robust SE 계산 참고
# Sandwich estimator: (H^-1) @ BHHH @ (H^-1)

def compute_robust_se(hessian_inv, bhhh):
    # Biogeme 방식
    variance = hessian_inv @ bhhh @ hessian_inv
    robust_se = np.sqrt(np.diag(variance))
    return robust_se
```

### 5.3. 장기 계획

**단계별 개선**:

1. **현재 (2025)**: 자체 구현 유지 + Biogeme 참고
   - ✅ 현재 시스템 안정화
   - ✅ Biogeme 소스코드 학습
   - ✅ BHHH 계산 검증

2. **중기 (2026)**: 부분 통합 검토
   - 선택모델 부분만 Biogeme 사용 고려
   - 성능 비교 (GPU vs Biogeme)
   - 결과 검증

3. **장기 (2027+)**: 커뮤니티 기여
   - Biogeme에 ICLV 자동화 기능 제안
   - GPU 가속 기능 기여
   - 학술 논문 발표

---

## 6. 결론

### 6.1. 최종 권장사항

**✅ 현재 자체 구현 유지**

**이유**:
1. Biogeme는 ICLV 자동화를 지원하지 않음
2. 현재 구현이 성능과 기능 면에서 우수함
3. GPU 가속으로 대규모 데이터 처리 가능
4. 전환 시 2-3주 작업 + 성능 저하 예상

### 6.2. Biogeme 활용 방안

**💡 참고 및 검증 목적으로 활용**:
- BHHH 계산 로직 검증
- 최적화 알고리즘 학습
- 표준오차 계산 확인
- 단순 모델 테스트 케이스

### 6.3. 비교 요약

| 항목 | 현재 자체 구현 | Biogeme 전환 |
|------|---------------|-------------|
| **ICLV 지원** | ✅ 완전 자동화 | ❌ 수동 구성 |
| **성능** | ✅ 90초 (GPU) | ❌ 수 시간 (CPU) |
| **BHHH** | ✅ 구현됨 | ✅ 자동 계산 |
| **유지보수** | ✅ 자체 제어 | ⚠️ 외부 의존 |
| **작업량** | ✅ 0시간 | ❌ 2-3주 |
| **권장도** | ✅✅✅ 강력 권장 | ❌ 비권장 |

---

## 7. 실제 Biogeme ICLV 구현 예제

### 7.1. 실제 사용 사례 (GitHub Discussion)

**출처**: https://github.com/jax-ml/jax/discussions/32575

실제 연구자가 Biogeme로 ICLV 모델을 구현한 코드:

```python
# Hybrid Choice Model: Walk vs Others
import biogeme.database as db
from biogeme.expressions import Beta, Variable, Draws, LinearUtility, MonteCarlo, log
from biogeme.models import logit

# 1. 잠재변수 정의 (구조방정식)
b_Den = Beta("struct_accw_Den", 0.0, None, None, 0)
sigma_accw = Beta("struct_accw_sigma", 1.0, None, None, 0)
accw_linear = LinearUtility([LinearTermTuple(b_Den, Den_Recretional_Act_Origin)])
accw = accw_linear + sigma_accw * Draws("struct_accw_error", "NORMAL_MLHS_ANTI")

# 2. 측정방정식 (Ordered Probit - 수동 구현)
def ordered_probit(continuous_value, scale_parameter, values, thresholds):
    probs = {}
    probs[values[0]] = NormalCdf((thresholds[0]-continuous_value)/scale_parameter)
    for i in range(1,len(values)-1):
        probs[values[i]] = NormalCdf((thresholds[i]-continuous_value)/scale_parameter) - \
                           NormalCdf((thresholds[i-1]-continuous_value)/scale_parameter)
    probs[values[-1]] = 1 - NormalCdf((thresholds[-1]-continuous_value)/scale_parameter)
    return probs

# 3. 측정 likelihood (4개 indicators)
def measurement_likelihood(latent, indicators):
    factors = []
    for ind in indicators:
        intercept = Beta(f"meas_intercept_{ind}", 0.0, None, None, 0)
        loading = Numeric(1.0) if ind == indicators[0] else Beta(f"meas_coeff_{ind}",0.0,None,None,0)
        scale = Beta(f"meas_scale_{ind}",1.0,None,None,0)
        probs = ordered_probit(intercept + loading*latent, scale, DISCRETE_VALUES, thresholds)
        factors.append(Elem(probs, Variable(ind)))
    return MultipleProduct(factors)

meas_like = measurement_likelihood(accw, ["ACCW1","ACCW2","ACCW3","ACCW4"])

# 4. 선택모델 (Binary Logit)
v = {
    1: Numeric(0.0),  # Others (reference)
    2: ASC_walk + beta_accw_walk*accw + beta_Den_walk*Den_Recretional_Act_Origin
}
choice_like = logit(v, None, Choice)

# 5. 결합 likelihood + Monte Carlo 적분
conditional_like = choice_like * meas_like
loglike = log(MonteCarlo(conditional_like))

# 6. 추정
biogeme = BIOGEME(database, loglike, number_of_draws=1000)
results = biogeme.estimate()
```

**결과**:
- ✅ 성공적으로 추정 완료
- ✅ 18개 파라미터 추정
- ✅ BHHH 자동 계산됨
- ⚠️ **메모리 문제 발생**: 변수 추가 시 "Out of memory allocating 3466368000 bytes" 오류

### 7.2. Biogeme ICLV 구현의 한계

위 실제 사례에서 확인된 문제점:

1. **메모리 부족**:
   - 복잡한 모델 (변수 추가) 시 메모리 초과
   - JAX 기반 Biogeme 3.3.1에서 발생
   - 배치 처리 미지원

2. **수동 구현 필요**:
   - Ordered Probit 확률 수동 계산
   - 각 indicator별 likelihood 수동 구성
   - 결합 likelihood 수동 정의

3. **단일 잠재변수만**:
   - 위 예제는 1개 LV만 사용
   - 현재 프로젝트는 5개 LV 필요
   - 복잡도 5배 증가

### 7.3. 현재 자체 구현의 우위성

| 측면 | 현재 자체 구현 | Biogeme 실제 사례 |
|------|---------------|------------------|
| **잠재변수 수** | 5개 (자동 처리) | 1개 (수동 구성) |
| **메모리 관리** | ✅ GPU 배치 처리 | ❌ 메모리 초과 오류 |
| **Ordered Probit** | ✅ 자동 계산 | ⚠️ 수동 확률 계산 |
| **실행 시간** | 90초 | 49초 (단순 모델) |
| **확장성** | ✅ 변수 추가 용이 | ❌ 메모리 제약 |
| **코드 복잡도** | 낮음 (자동화) | 높음 (수동 구성) |

---

## 8. 참고 자료

### 8.1. Biogeme 공식 문서
- 공식 사이트: https://biogeme.epfl.ch/
- API 문서: https://biogeme.epfl.ch/sphinx/
- GitHub: https://github.com/michelbierlaire/biogeme
- 예제: https://biogeme.epfl.ch/sphinx/auto_examples/
- Hybrid Choice 예제: https://biogeme.epfl.ch/sphinx/auto_examples/latent/index.html

### 8.2. ICLV 관련 논문
- Ben-Akiva et al. (2002): "Hybrid Choice Models"
- Walker & Ben-Akiva (2002): "Generalized Random Utility Model"
- Daziano & Bolduc (2013): "Incorporating pro-environmental preferences"

### 8.3. 현재 구현 문서
- `docs/early_stopping_hessian_optimization.md`: BHHH 구현
- `docs/bhhh_iteration_count_analysis.md`: BHHH 성능 분석
- `analysis/bfgs_vs_bhhh_compatibility_analysis.md`: BFGS vs BHHH 비교

### 8.4. 실제 사용 사례
- GitHub JAX Discussion #32575: Biogeme ICLV 메모리 문제
- 연구자 보고: 복잡한 모델에서 메모리 초과 오류 발생

