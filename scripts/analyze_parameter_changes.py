"""
파라미터 변화 추이 분석
"""

import numpy as np

# 주요 iteration의 파라미터 값 (로그에서 추출)
iterations = [1, 10, 20, 30, 40]

# health_concern zeta (처음 3개)
hc_zeta = {
    1: [1.0, 0.024, 0.024],
    10: [1.0, 0.0816, 0.0812],
    20: [1.0, 0.4003, 0.3970],
    30: [1.0, 0.5128, 0.5079],
    40: [1.0, 1.0677, 1.0571]
}

# health_concern sigma_sq (처음 3개)
hc_sigma = {
    1: [0.034, 0.034, 0.034],
    10: [0.171, 0.238, 0.237],
    20: [0.226, 0.221, 0.217],
    30: [0.212, 0.208, 0.204],
    40: [0.149, 0.152, 0.146]
}

# perceived_price zeta (처음 3개)
pp_zeta = {
    1: [1.0, 0.12, 0.12],
    10: [1.0, 0.1672, 0.1686],
    20: [1.0, 0.5030, 0.5193],
    30: [1.0, 0.5867, 0.5931],
    40: [1.0, 0.9149, 0.8574]
}

# nutrition_knowledge zeta (처음 3개)
nk_zeta = {
    1: [1.0, 0.022, 0.022],
    10: [1.0, 0.0677, 0.0694],
    20: [1.0, 0.3455, 0.3560],
    30: [1.0, 0.3702, 0.3878],
    40: [1.0, 0.3948, 0.4471]
}

# purchase_intention zeta (처음 3개)
pi_zeta = {
    1: [1.0, 0.083, 0.083],
    10: [1.0, 0.0987, 0.0991],
    20: [1.0, 0.2019, 0.2061],
    30: [1.0, 0.2418, 0.2466],
    40: [1.0, 0.4429, 0.4576]
}

# 선택모델 파라미터
intercept = {1: 0.290, 10: 0.288, 20: 0.271, 30: 0.267, 40: 0.251}
beta_sugar = {1: 0.23, 10: 0.230, 20: 0.229, 30: 0.228, 40: 0.227}
beta_health = {1: 0.22, 10: 0.220, 20: 0.219, 30: 0.218, 40: 0.217}
beta_price = {1: -0.056, 10: -0.059, 20: -0.063, 30: -0.065, 40: -0.069}
lambda_main = {1: 0.89, 10: 0.889, 20: 0.883, 30: 0.880, 40: 0.872}

print('=' * 80)
print('파라미터 변화 추이 분석')
print('=' * 80)
print()

print('[측정모델 - Factor Loadings (zeta)]')
print('-' * 80)
print('Iter   HC_q7    HC_q8    PP_q28   PP_q29   NK_q31   NK_q32   PI_q19   PI_q20')
print('-' * 80)
for i in iterations:
    print(f'{i:4d}   {hc_zeta[i][1]:.4f}  {hc_zeta[i][2]:.4f}  '
          f'{pp_zeta[i][1]:.4f}  {pp_zeta[i][2]:.4f}  '
          f'{nk_zeta[i][1]:.4f}  {nk_zeta[i][2]:.4f}  '
          f'{pi_zeta[i][1]:.4f}  {pi_zeta[i][2]:.4f}')

print()
print('변화량 (Iter 1 → 40):')
print(f'  HC_q7:  {hc_zeta[1][1]:.4f} → {hc_zeta[40][1]:.4f}  (Δ = {hc_zeta[40][1] - hc_zeta[1][1]:+.4f}, {((hc_zeta[40][1] - hc_zeta[1][1])/hc_zeta[1][1])*100:+.1f}%)')
print(f'  PP_q28: {pp_zeta[1][1]:.4f} → {pp_zeta[40][1]:.4f}  (Δ = {pp_zeta[40][1] - pp_zeta[1][1]:+.4f}, {((pp_zeta[40][1] - pp_zeta[1][1])/pp_zeta[1][1])*100:+.1f}%)')
print(f'  NK_q31: {nk_zeta[1][1]:.4f} → {nk_zeta[40][1]:.4f}  (Δ = {nk_zeta[40][1] - nk_zeta[1][1]:+.4f}, {((nk_zeta[40][1] - nk_zeta[1][1])/nk_zeta[1][1])*100:+.1f}%)')
print(f'  PI_q19: {pi_zeta[1][1]:.4f} → {pi_zeta[40][1]:.4f}  (Δ = {pi_zeta[40][1] - pi_zeta[1][1]:+.4f}, {((pi_zeta[40][1] - pi_zeta[1][1])/pi_zeta[1][1])*100:+.1f}%)')

print()
print('[측정모델 - Error Variances (sigma_sq)]')
print('-' * 80)
print('Iter   HC_q6    HC_q7    HC_q8')
print('-' * 80)
for i in iterations:
    print(f'{i:4d}   {hc_sigma[i][0]:.4f}  {hc_sigma[i][1]:.4f}  {hc_sigma[i][2]:.4f}')

print()
print('변화량 (Iter 1 → 40):')
print(f'  HC_q6: {hc_sigma[1][0]:.4f} → {hc_sigma[40][0]:.4f}  (Δ = {hc_sigma[40][0] - hc_sigma[1][0]:+.4f}, {((hc_sigma[40][0] - hc_sigma[1][0])/hc_sigma[1][0])*100:+.1f}%)')
print(f'  HC_q7: {hc_sigma[1][1]:.4f} → {hc_sigma[40][1]:.4f}  (Δ = {hc_sigma[40][1] - hc_sigma[1][1]:+.4f}, {((hc_sigma[40][1] - hc_sigma[1][1])/hc_sigma[1][1])*100:+.1f}%)')

print()
print('[선택모델 파라미터]')
print('-' * 80)
print('Iter   Intercept  β_sugar  β_health  β_price   λ_main')
print('-' * 80)
for i in iterations:
    print(f'{i:4d}   {intercept[i]:.6f}  {beta_sugar[i]:.6f}  {beta_health[i]:.6f}  {beta_price[i]:.6f}  {lambda_main[i]:.6f}')

print()
print('변화량 (Iter 1 → 40):')
print(f'  Intercept: {intercept[1]:.6f} → {intercept[40]:.6f}  (Δ = {intercept[40] - intercept[1]:+.6f}, {((intercept[40] - intercept[1])/intercept[1])*100:+.1f}%)')
print(f'  β_price:   {beta_price[1]:.6f} → {beta_price[40]:.6f}  (Δ = {beta_price[40] - beta_price[1]:+.6f}, {((beta_price[40] - beta_price[1])/beta_price[1])*100:+.1f}%)')
print(f'  λ_main:    {lambda_main[1]:.6f} → {lambda_main[40]:.6f}  (Δ = {lambda_main[40] - lambda_main[1]:+.6f}, {((lambda_main[40] - lambda_main[1])/lambda_main[1])*100:+.1f}%)')

print()
print('=' * 80)
print('파라미터 업데이트 패턴 분석')
print('=' * 80)
print()

# 각 구간별 변화율 계산
print('[구간별 변화율]')
print('-' * 80)
print('파라미터          Iter 1→10   Iter 10→20  Iter 20→30  Iter 30→40')
print('-' * 80)

def calc_change(d, i1, i2):
    return ((d[i2] - d[i1]) / d[i1]) * 100 if d[i1] != 0 else 0

# HC zeta[1]
hc_z1 = {i: hc_zeta[i][1] for i in iterations}
print(f'HC_zeta_q7        {calc_change(hc_z1, 1, 10):+7.1f}%   {calc_change(hc_z1, 10, 20):+7.1f}%   {calc_change(hc_z1, 20, 30):+7.1f}%   {calc_change(hc_z1, 30, 40):+7.1f}%')

# PP zeta[1]
pp_z1 = {i: pp_zeta[i][1] for i in iterations}
print(f'PP_zeta_q28       {calc_change(pp_z1, 1, 10):+7.1f}%   {calc_change(pp_z1, 10, 20):+7.1f}%   {calc_change(pp_z1, 20, 30):+7.1f}%   {calc_change(pp_z1, 30, 40):+7.1f}%')

# NK zeta[1]
nk_z1 = {i: nk_zeta[i][1] for i in iterations}
print(f'NK_zeta_q31       {calc_change(nk_z1, 1, 10):+7.1f}%   {calc_change(nk_z1, 10, 20):+7.1f}%   {calc_change(nk_z1, 20, 30):+7.1f}%   {calc_change(nk_z1, 30, 40):+7.1f}%')

# PI zeta[1]
pi_z1 = {i: pi_zeta[i][1] for i in iterations}
print(f'PI_zeta_q19       {calc_change(pi_z1, 1, 10):+7.1f}%   {calc_change(pi_z1, 10, 20):+7.1f}%   {calc_change(pi_z1, 20, 30):+7.1f}%   {calc_change(pi_z1, 30, 40):+7.1f}%')

# HC sigma[0]
hc_s0 = {i: hc_sigma[i][0] for i in iterations}
print(f'HC_sigma_q6       {calc_change(hc_s0, 1, 10):+7.1f}%   {calc_change(hc_s0, 10, 20):+7.1f}%   {calc_change(hc_s0, 20, 30):+7.1f}%   {calc_change(hc_s0, 30, 40):+7.1f}%')

print(f'β_price           {calc_change(beta_price, 1, 10):+7.1f}%   {calc_change(beta_price, 10, 20):+7.1f}%   {calc_change(beta_price, 20, 30):+7.1f}%   {calc_change(beta_price, 30, 40):+7.1f}%')
print(f'λ_main            {calc_change(lambda_main, 1, 10):+7.1f}%   {calc_change(lambda_main, 10, 20):+7.1f}%   {calc_change(lambda_main, 30, 40):+7.1f}%   {calc_change(lambda_main, 30, 40):+7.1f}%')

print()
print('관찰:')
print('  - Factor loadings (zeta): 초기에 급격히 증가, 후반부로 갈수록 변화율 감소')
print('  - Error variances (sigma_sq): Iter 1→10에서 급증 후 점진적 감소')
print('  - 선택모델 파라미터: 상대적으로 안정적, 작은 변화율')
print('  → 전형적인 BFGS 수렴 패턴 (초기 큰 변화 → 점진적 수렴)')

print()
print('=' * 80)

