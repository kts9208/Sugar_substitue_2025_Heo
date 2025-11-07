"""
-inf/NaN 발생 원인 분석 스크립트
"""
import pandas as pd
import numpy as np
from scipy.stats import norm

# 데이터 로드
data = pd.read_csv('data/processed/iclv/integrated_data.csv')
test_ids = data['respondent_id'].unique()[:5]
test_data = data[data['respondent_id'].isin(test_ids)]

print('='*70)
print('시뮬레이션: -inf 발생 시나리오')
print('='*70)

# 시나리오 1: Ordered Probit에서 확률이 0이 되는 경우
print('\n시나리오 1: Ordered Probit 확률 = 0')
print('-'*70)

# 예시 파라미터
zeta = 1.5
tau = np.array([-2, -1, 1, 2])
y = 5  # 관측값 (최대값)

# 다양한 LV 값에 대해 테스트
lv_values = [-5, -3, -1, 0, 1, 3, 5, 10, 20]
print(f'관측값 y={y}, zeta={zeta}, tau={tau}')
print(f'P(Y=5) = 1 - Φ(τ_4 - ζ*LV)')
print()

for lv in lv_values:
    V = zeta * lv
    # P(Y=5) = 1 - Φ(τ_4 - V)
    prob = 1 - norm.cdf(tau[-1] - V)
    log_prob = np.log(prob) if prob > 0 else -np.inf
    print(f'LV={lv:5.1f} → V={V:6.2f} → τ_4-V={tau[-1]-V:7.2f} → Φ={norm.cdf(tau[-1]-V):.6f} → P={prob:.6e} → log(P)={log_prob:10.2f}')

# 시나리오 2: Binary Probit에서 확률이 0 또는 1이 되는 경우
print('\n시나리오 2: Binary Probit 확률 ≈ 0 또는 1')
print('-'*70)

intercept = 0.0
beta_price = -0.0001
beta_health = 0.0005
lambda_lv = 0.02

price = 1000
health_label = 1
lv_values = [-10, -5, 0, 5, 10, 20, 50]

print(f'intercept={intercept}, β_price={beta_price}, β_health={beta_health}, λ={lambda_lv}')
print(f'price={price}, health_label={health_label}')
print(f'V = intercept + β_price*price + β_health*health + λ*LV')
print()

for lv in lv_values:
    V = intercept + beta_price * price + beta_health * health_label + lambda_lv * lv
    prob_yes = norm.cdf(V)
    
    # choice=1인 경우
    log_prob_1 = np.log(prob_yes) if prob_yes > 0 else -np.inf
    # choice=0인 경우
    log_prob_0 = np.log(1 - prob_yes) if (1 - prob_yes) > 0 else -np.inf
    
    print(f'LV={lv:5.1f} → V={V:7.4f} → P(Yes)={prob_yes:.6e} → log(P|choice=1)={log_prob_1:10.2f}, log(P|choice=0)={log_prob_0:10.2f}')

# 시나리오 3: Panel Product에서 -inf 누적
print('\n시나리오 3: Panel Product - 18개 선택 상황')
print('-'*70)

# 각 선택 상황의 로그우도가 -10이라고 가정
ll_per_choice = -10
n_choices = 18

total_ll = ll_per_choice * n_choices
print(f'각 선택 상황의 로그우도: {ll_per_choice}')
print(f'선택 상황 개수: {n_choices}')
print(f'Panel Product 로그우도: {ll_per_choice} × {n_choices} = {total_ll}')
print()

# 더 극단적인 경우
ll_values = [-5, -10, -20, -50, -100]
for ll in ll_values:
    total = ll * n_choices
    print(f'각 LL={ll:5.0f} → Panel Product LL = {total:7.0f}')

# 시나리오 4: 결합 로그우도
print('\n시나리오 4: 결합 로그우도 (측정 + 구조 + 선택)')
print('-'*70)

# 예시 값
ll_measurement = -30  # 6개 지표 × 평균 -5
ll_structural = -2    # 정규분포 로그우도
ll_choice = -180      # 18개 선택 × 평균 -10

total = ll_measurement + ll_structural + ll_choice
print(f'측정모델 LL: {ll_measurement:7.1f} (6개 지표)')
print(f'구조모델 LL: {ll_structural:7.1f}')
print(f'선택모델 LL: {ll_choice:7.1f} (18개 선택 상황)')
print(f'결합 LL:     {total:7.1f}')
print()
print(f'exp({total}) = {np.exp(total):.6e}')

