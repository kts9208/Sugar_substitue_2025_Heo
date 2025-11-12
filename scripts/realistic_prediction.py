"""
Realistic parameter convergence prediction
"""

import numpy as np

print('=' * 100)
print('Parameter Convergence Prediction (Based on Iter 10, 20, 30, 40)')
print('=' * 100)
print()

# Iteration 10, 20, 30, 40 data
iters = [10, 20, 30, 40]

params = {
    'HC_zeta_q7': [0.0816, 0.4003, 0.5128, 1.0677],
    'HC_zeta_q8': [0.0812, 0.3970, 0.5079, 1.0571],
    'HC_sigma_q6': [0.1710, 0.2263, 0.2117, 0.1486],
    'HC_sigma_q7': [0.2380, 0.2205, 0.2076, 0.1515],
    'PB_zeta_q13': [0.0736, 0.1957, 0.2391, 0.4609],
    'PB_sigma_q12': [0.1910, 0.2104, 0.2161, 0.1802],
    'PP_zeta_q28': [0.3867, 0.5030, 0.5867, 0.9149],
    'PP_sigma_q27': [0.2800, 0.2872, 0.2350, 0.2895],
    'NK_zeta_q31': [0.1330, 0.3455, 0.3702, 0.3948],
    'NK_sigma_q30': [0.1050, 0.0973, 0.1065, 0.1805],
    'PI_zeta_q19': [0.1330, 0.2019, 0.2418, 0.4429],
    'PI_sigma_q18': [0.4820, 0.3752, 0.4632, 0.4770],
    'beta_intercept': [0.2880, 0.2711, 0.2674, 0.2513],
    'beta_sugar': [0.2300, 0.2263, 0.2257, 0.2233],
    'beta_price': [-0.0590, -0.0807, -0.0850, -0.1028],
    'lambda_main': [0.8899, 0.8821, 0.8804, 0.8726],
    'lambda_mod_pp': [-0.4700, -0.4818, -0.4835, -0.4909],
    'lambda_mod_nk': [1.2000, 1.1912, 1.1890, 1.1792],
}

names = {
    'HC_zeta_q7': 'health_concern zeta[1]',
    'HC_zeta_q8': 'health_concern zeta[2]',
    'HC_sigma_q6': 'health_concern sigma_sq[0]',
    'HC_sigma_q7': 'health_concern sigma_sq[1]',
    'PB_zeta_q13': 'perceived_benefit zeta[1]',
    'PB_sigma_q12': 'perceived_benefit sigma_sq[0]',
    'PP_zeta_q28': 'perceived_price zeta[1]',
    'PP_sigma_q27': 'perceived_price sigma_sq[0]',
    'NK_zeta_q31': 'nutrition_knowledge zeta[1]',
    'NK_sigma_q30': 'nutrition_knowledge sigma_sq[0]',
    'PI_zeta_q19': 'purchase_intention zeta[1]',
    'PI_sigma_q18': 'purchase_intention sigma_sq[0]',
    'beta_intercept': 'intercept',
    'beta_sugar': 'beta_sugar_free',
    'beta_price': 'beta_price',
    'lambda_main': 'lambda_main',
    'lambda_mod_pp': 'lambda_mod_PP',
    'lambda_mod_nk': 'lambda_mod_NK',
}

print(f'{"Parameter":<40s} {"Iter40":<10s} {"Change30-40":<12s} {"Predicted":<10s} {"Trend"}')
print('-' * 100)

predictions = {}

for key, values in params.items():
    v = np.array(values)
    current = v[-1]
    
    # Recent change (Iter 30 -> 40)
    recent_change = v[-1] - v[-2]
    
    # Conservative prediction: assume 50% decay, sum over 10 iterations
    # Geometric sum: change * (1 + 0.5 + 0.25 + ...) = change * 2
    predicted = current + recent_change * 2.0
    
    # Trend classification
    if abs(recent_change) < 0.01:
        trend = 'Almost converged'
    elif abs(recent_change) < 0.05:
        trend = 'Converging'
    elif abs(recent_change) < 0.2:
        trend = 'Changing'
    else:
        trend = 'Large change'
    
    predictions[key] = {
        'current': current,
        'predicted': predicted,
        'change': recent_change,
        'trend': trend
    }
    
    print(f'{names[key]:<40s} {current:>9.4f} {recent_change:>+11.4f} {predicted:>9.4f} {trend}')

print()
print('=' * 100)
print('Detailed Analysis by Parameter Type')
print('=' * 100)
print()

print('[1] Factor Loadings (zeta) - Measurement Model')
print('-' * 100)
print()

# Health Concern
hc_z = np.array(params['HC_zeta_q7'])
print(f'health_concern zeta[1]: {hc_z}')
print(f'  Changes: 10->20: +{hc_z[1]-hc_z[0]:.4f}, 20->30: +{hc_z[2]-hc_z[1]:.4f}, 30->40: +{hc_z[3]-hc_z[2]:.4f}')
print(f'  Change rate INCREASING! (0.32 -> 0.11 -> 0.55)')
print(f'  -> Iter 30->40 shows acceleration (+0.55)')
print(f'  -> Expected convergence: 1.5~2.0 (50+ iterations needed)')
print()

# Perceived Benefit
pb_z = np.array(params['PB_zeta_q13'])
print(f'perceived_benefit zeta[1]: {pb_z}')
print(f'  Changes: 10->20: +{pb_z[1]-pb_z[0]:.4f}, 20->30: +{pb_z[2]-pb_z[1]:.4f}, 30->40: +{pb_z[3]-pb_z[2]:.4f}')
print(f'  Change rate INCREASING! (0.12 -> 0.04 -> 0.22)')
print(f'  -> Expected convergence: 0.7~0.9')
print()

# Perceived Price
pp_z = np.array(params['PP_zeta_q28'])
print(f'perceived_price zeta[1]: {pp_z}')
print(f'  Changes: 10->20: +{pp_z[1]-pp_z[0]:.4f}, 20->30: +{pp_z[2]-pp_z[1]:.4f}, 30->40: +{pp_z[3]-pp_z[2]:.4f}')
print(f'  Change rate INCREASING! (0.12 -> 0.08 -> 0.33)')
print(f'  -> Expected convergence: 1.3~1.6')
print()

# Nutrition Knowledge
nk_z = np.array(params['NK_zeta_q31'])
print(f'nutrition_knowledge zeta[1]: {nk_z}')
print(f'  Changes: 10->20: +{nk_z[1]-nk_z[0]:.4f}, 20->30: +{nk_z[2]-nk_z[1]:.4f}, 30->40: +{nk_z[3]-nk_z[2]:.4f}')
print(f'  Change rate DECREASING (0.21 -> 0.02 -> 0.02)')
print(f'  -> Almost converged! Expected: 0.40~0.42')
print()

# Purchase Intention
pi_z = np.array(params['PI_zeta_q19'])
print(f'purchase_intention zeta[1]: {pi_z}')
print(f'  Changes: 10->20: +{pi_z[1]-pi_z[0]:.4f}, 20->30: +{pi_z[2]-pi_z[1]:.4f}, 30->40: +{pi_z[3]-pi_z[2]:.4f}')
print(f'  Change rate INCREASING! (0.07 -> 0.04 -> 0.20)')
print(f'  -> Expected convergence: 0.7~0.9')
print()

print('[2] Error Variances (sigma_sq) - Measurement Model')
print('-' * 100)
print()

# Health Concern sigma
hc_s = np.array(params['HC_sigma_q6'])
print(f'health_concern sigma_sq[0]: {hc_s}')
print(f'  Changes: 10->20: +{hc_s[1]-hc_s[0]:.4f}, 20->30: {hc_s[2]-hc_s[1]:.4f}, 30->40: {hc_s[3]-hc_s[2]:.4f}')
print(f'  DECREASING trend (0.23 -> 0.21 -> 0.15)')
print(f'  -> Expected convergence: 0.10~0.12')
print()

# Perceived Benefit sigma
pb_s = np.array(params['PB_sigma_q12'])
print(f'perceived_benefit sigma_sq[0]: {pb_s}')
print(f'  Changes: 10->20: +{pb_s[1]-pb_s[0]:.4f}, 20->30: +{pb_s[2]-pb_s[1]:.4f}, 30->40: {pb_s[3]-pb_s[2]:.4f}')
print(f'  Stable (0.19 -> 0.21 -> 0.22 -> 0.18)')
print(f'  -> Expected convergence: 0.16~0.18')
print()

# Purchase Intention sigma
pi_s = np.array(params['PI_sigma_q18'])
print(f'purchase_intention sigma_sq[0]: {pi_s}')
print(f'  Changes: 10->20: {pi_s[1]-pi_s[0]:.4f}, 20->30: +{pi_s[2]-pi_s[1]:.4f}, 30->40: +{pi_s[3]-pi_s[2]:.4f}')
print(f'  Stable (0.48 -> 0.38 -> 0.46 -> 0.48)')
print(f'  -> Expected convergence: 0.48~0.50')
print()

print('[3] Choice Model Parameters')
print('-' * 100)
print()

# Beta sugar
bs = np.array(params['beta_sugar'])
print(f'beta_sugar_free: {bs}')
print(f'  Changes: 10->20: {bs[1]-bs[0]:.4f}, 20->30: {bs[2]-bs[1]:.4f}, 30->40: {bs[3]-bs[2]:.4f}')
print(f'  Almost converged (change < 0.01)')
print(f'  -> Expected convergence: 0.22~0.23')
print()

# Beta price
bp = np.array(params['beta_price'])
print(f'beta_price: {bp}')
print(f'  Changes: 10->20: {bp[1]-bp[0]:.4f}, 20->30: {bp[2]-bp[1]:.4f}, 30->40: {bp[3]-bp[2]:.4f}')
print(f'  Continuous decrease (-0.06 -> -0.08 -> -0.09 -> -0.10)')
print(f'  -> Expected convergence: -0.12~-0.15 (negative)')
print()

# Lambda main
lm = np.array(params['lambda_main'])
print(f'lambda_main: {lm}')
print(f'  Changes: 10->20: {lm[1]-lm[0]:.4f}, 20->30: {lm[2]-lm[1]:.4f}, 30->40: {lm[3]-lm[2]:.4f}')
print(f'  Almost converged (change < 0.01)')
print(f'  -> Expected convergence: 0.86~0.87')
print()

print('=' * 100)
print('FINAL CONVERGENCE PREDICTION SUMMARY')
print('=' * 100)
print()

print('ALMOST CONVERGED (within 10 iterations):')
print('  - nutrition_knowledge zeta: 0.40~0.42')
print('  - beta_sugar_free: 0.22~0.23')
print('  - lambda_main: 0.86~0.87')
print('  - lambda_mod_PP: -0.50~-0.52')
print('  - lambda_mod_NK: 1.17~1.18')
print()

print('CONVERGING (30 iterations needed):')
print('  - perceived_benefit sigma_sq: 0.16~0.18')
print('  - perceived_price sigma_sq: 0.28~0.30')
print('  - purchase_intention sigma_sq: 0.48~0.50')
print('  - beta_price: -0.12~-0.15')
print('  - intercept: 0.20~0.23')
print()

print('LARGE CHANGES EXPECTED (50+ iterations needed):')
print('  - health_concern zeta: 1.5~2.0')
print('  - health_concern sigma_sq: 0.10~0.12')
print('  - perceived_benefit zeta: 0.7~0.9')
print('  - perceived_price zeta: 1.3~1.6')
print('  - purchase_intention zeta: 0.7~0.9')
print()

print('=' * 100)

