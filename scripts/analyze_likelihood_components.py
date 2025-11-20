"""
우도 성분 분석: 측정모델, 선택모델, 구조모델의 우도 계산 방식 비교

목적: 왜 측정모델 우도가 압도적으로 크고, 선택모델/구조모델 우도가 작은지 분석
"""
import numpy as np
from scipy.stats import norm

print("="*80)
print("우도 성분 분석")
print("="*80)

# ============================================================================
# 1. 측정모델 우도 (Ordered Probit)
# ============================================================================
print(f"\n[1] 측정모델 우도 (Ordered Probit)")
print(f"{'='*80}")

# 예시: 1개 지표, 5점 척도
# P(Y=k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)

# 파라미터 (CFA 결과에서 가져온 값)
zeta = 1.0  # 요인적재량
tau = np.array([-2.0, -1.0, 0.0, 1.0])  # 4개 임계값 (5점 척도)
lv = 0.5  # 잠재변수 값
y_obs = 3  # 관측값 (3점)

# 선형 예측
V = zeta * lv
print(f"\n파라미터:")
print(f"  ζ (zeta): {zeta}")
print(f"  τ (tau): {tau}")
print(f"  LV: {lv}")
print(f"  V = ζ × LV = {V:.4f}")

# 확률 계산
k = y_obs - 1  # 0-based index
upper = tau[k] - V
lower = tau[k-1] - V
prob = norm.cdf(upper) - norm.cdf(lower)

print(f"\n확률 계산 (Y={y_obs}):")
print(f"  upper = τ_{k} - V = {tau[k]:.4f} - {V:.4f} = {upper:.4f}")
print(f"  lower = τ_{k-1} - V = {tau[k-1]:.4f} - {V:.4f} = {lower:.4f}")
print(f"  P(Y={y_obs}) = Φ({upper:.4f}) - Φ({lower:.4f}) = {prob:.6f}")

# 로그우도
ll_single = np.log(prob)
print(f"\n로그우도 (1개 지표):")
print(f"  ll = log({prob:.6f}) = {ll_single:.4f}")

# 38개 지표 (5개 잠재변수)
n_indicators = 38
ll_measurement_total = ll_single * n_indicators
print(f"\n로그우도 (38개 지표, 모두 동일 확률 가정):")
print(f"  ll_total = {ll_single:.4f} × {n_indicators} = {ll_measurement_total:.4f}")

# ============================================================================
# 2. 선택모델 우도 (Multinomial Logit)
# ============================================================================
print(f"\n{'='*80}")
print(f"[2] 선택모델 우도 (Multinomial Logit)")
print(f"{'='*80}")

# 예시: 1개 choice set, 3개 대안
# P_i = exp(V_i) / Σ_j exp(V_j)

# 효용 (ASC + β*X + θ*LV)
V_sugar = 0.5  # ASC_sugar + β*X + θ*LV
V_sugar_free = 0.3  # ASC_sugar_free + β*X + θ*LV
V_optout = 0.0  # reference

V = np.array([V_sugar, V_sugar_free, V_optout])
chosen_alt = 0  # sugar 선택

print(f"\n효용:")
print(f"  V_sugar = {V_sugar:.4f}")
print(f"  V_sugar_free = {V_sugar_free:.4f}")
print(f"  V_optout = {V_optout:.4f} (reference)")

# 확률 계산
exp_V = np.exp(V)
sum_exp_V = np.sum(exp_V)
prob_chosen = exp_V[chosen_alt] / sum_exp_V

print(f"\n확률 계산:")
print(f"  exp(V_sugar) = {exp_V[0]:.4f}")
print(f"  exp(V_sugar_free) = {exp_V[1]:.4f}")
print(f"  exp(V_optout) = {exp_V[2]:.4f}")
print(f"  Σ exp(V) = {sum_exp_V:.4f}")
print(f"  P(sugar) = {exp_V[0]:.4f} / {sum_exp_V:.4f} = {prob_chosen:.6f}")

# 로그우도
ll_choice_single = np.log(prob_chosen)
print(f"\n로그우도 (1개 choice set):")
print(f"  ll = log({prob_chosen:.6f}) = {ll_choice_single:.4f}")

# 6개 choice sets
n_choice_sets = 6
ll_choice_total = ll_choice_single * n_choice_sets
print(f"\n로그우도 (6개 choice sets, 모두 동일 확률 가정):")
print(f"  ll_total = {ll_choice_single:.4f} × {n_choice_sets} = {ll_choice_total:.4f}")

# ============================================================================
# 3. 구조모델 우도 (Normal Distribution)
# ============================================================================
print(f"\n{'='*80}")
print(f"[3] 구조모델 우도 (Normal Distribution)")
print(f"{'='*80}")

# 예시: 1개 경로 (HC → PB)
# P(PB | HC) = N(PB | γ*HC, σ²)

# 파라미터
gamma = 0.5  # 경로계수
sigma_sq = 1.0  # 오차분산
hc = 0.3  # 건강관심도 (예측변수)
pb_actual = 0.6  # 건강유익성 실제값 (목표변수)

# 예측값
pb_mean = gamma * hc
residual = pb_actual - pb_mean

print(f"\n파라미터:")
print(f"  γ (gamma): {gamma}")
print(f"  σ² (sigma_sq): {sigma_sq}")
print(f"  HC (예측변수): {hc}")
print(f"  PB_actual (실제값): {pb_actual}")

print(f"\n예측:")
print(f"  PB_mean = γ × HC = {gamma} × {hc} = {pb_mean:.4f}")
print(f"  residual = {pb_actual} - {pb_mean:.4f} = {residual:.4f}")

# 로그우도: log N(x | μ, σ²) = -0.5*log(2πσ²) - 0.5*(x-μ)²/σ²
ll_structural_single = -0.5 * np.log(2 * np.pi * sigma_sq) - 0.5 * (residual**2) / sigma_sq

print(f"\n로그우도 (1개 경로):")
print(f"  ll = -0.5×log(2π×{sigma_sq}) - 0.5×({residual:.4f})²/{sigma_sq}")
print(f"     = {ll_structural_single:.4f}")

# 2개 경로 (HC → PB, PB → PI)
n_paths = 2
ll_structural_total = ll_structural_single * n_paths
print(f"\n로그우도 (2개 경로, 모두 동일 가정):")
print(f"  ll_total = {ll_structural_single:.4f} × {n_paths} = {ll_structural_total:.4f}")

# ============================================================================
# 4. 비교 및 분석
# ============================================================================
print(f"\n{'='*80}")
print(f"[4] 비교 및 분석")
print(f"{'='*80}")

print(f"\n우도 성분 비교:")
print(f"  측정모델 (38개 지표): {ll_measurement_total:.4f}")
print(f"  선택모델 (6개 choice sets): {ll_choice_total:.4f}")
print(f"  구조모델 (2개 경로): {ll_structural_total:.4f}")
print(f"  합계: {ll_measurement_total + ll_choice_total + ll_structural_total:.4f}")

print(f"\n비율:")
total = abs(ll_measurement_total) + abs(ll_choice_total) + abs(ll_structural_total)
print(f"  측정모델: {abs(ll_measurement_total)/total*100:.1f}%")
print(f"  선택모델: {abs(ll_choice_total)/total*100:.1f}%")
print(f"  구조모델: {abs(ll_structural_total)/total*100:.1f}%")

print(f"\n{'='*80}")
print(f"[5] 핵심 발견")
print(f"{'='*80}")

print(f"""
1. **측정모델 우도가 큰 이유**:
   - 지표 수가 많음 (38개)
   - 각 지표마다 로그우도 누적 (합산)
   - Ordered Probit 확률이 일반적으로 낮음 (5점 척도 중 1개 선택)
   - 낮은 확률 → 큰 음수 로그우도

2. **선택모델 우도가 작은 이유**:
   - 선택 상황 수가 적음 (6개)
   - Multinomial Logit 확률이 상대적으로 높음 (3개 대안 중 1개)
   - 높은 확률 → 작은 음수 로그우도

3. **구조모델 우도가 작은 이유**:
   - 경로 수가 매우 적음 (2개)
   - Normal distribution의 로그우도는 잔차가 작으면 0에 가까움
   - 잔차가 크더라도 1개 경로당 1개 값만 기여

4. **근본 원인**:
   - **관측값 개수의 차이**: 측정모델 38개 vs 선택모델 6개 vs 구조모델 2개
   - 로그우도는 관측값마다 누적되므로, 관측값이 많을수록 절댓값이 커짐
   - 이는 **정상적인 현상**이며, 모델 구조의 차이를 반영함

5. **스케일링의 필요성**:
   - 측정모델 우도를 지표 수로 나누면 "평균 우도"가 됨
   - 이렇게 하면 각 모델 성분의 기여도를 공정하게 비교 가능
   - 최적화 시 gradient 균형 개선
""")

