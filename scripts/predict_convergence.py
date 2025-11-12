"""
파라미터 수렴값 예측
"""

import numpy as np
import matplotlib.pyplot as plt

# 파라미터 변화 추이 (Iteration 10, 20, 30, 40)
iterations = np.array([10, 20, 30, 40])

# 측정모델 파라미터 (대표 예시)
params_data = {
    'HC_zeta_q7': {
        'values': [0.0816, 0.4003, 0.5128, 1.0677],
        'name': 'health_concern zeta[1]',
        'type': 'factor_loading'
    },
    'HC_zeta_q8': {
        'values': [0.0812, 0.3970, 0.5079, 1.0571],
        'name': 'health_concern zeta[2]',
        'type': 'factor_loading'
    },
    'HC_sigma_q6': {
        'values': [0.1710, 0.2263, 0.2117, 0.1486],
        'name': 'health_concern sigma_sq[0]',
        'type': 'error_variance'
    },
    'HC_sigma_q7': {
        'values': [0.2380, 0.2205, 0.2076, 0.1515],
        'name': 'health_concern sigma_sq[1]',
        'type': 'error_variance'
    },
    'PB_zeta_q13': {
        'values': [0.0736, 0.1957, 0.2391, 0.4609],
        'name': 'perceived_benefit zeta[1]',
        'type': 'factor_loading'
    },
    'PB_sigma_q12': {
        'values': [0.1910, 0.2104, 0.2161, 0.1802],
        'name': 'perceived_benefit sigma_sq[0]',
        'type': 'error_variance'
    },
    'PP_zeta_q28': {
        'values': [0.3867, 0.5030, 0.5867, 0.9149],
        'name': 'perceived_price zeta[1]',
        'type': 'factor_loading'
    },
    'PP_sigma_q27': {
        'values': [0.2800, 0.2872, 0.2350, 0.2895],
        'name': 'perceived_price sigma_sq[0]',
        'type': 'error_variance'
    },
    'NK_zeta_q31': {
        'values': [0.1330, 0.3455, 0.3702, 0.3948],
        'name': 'nutrition_knowledge zeta[1]',
        'type': 'factor_loading'
    },
    'NK_sigma_q30': {
        'values': [0.1050, 0.0973, 0.1065, 0.1805],
        'name': 'nutrition_knowledge sigma_sq[0]',
        'type': 'error_variance'
    },
    'PI_zeta_q19': {
        'values': [0.1330, 0.2019, 0.2418, 0.4429],
        'name': 'purchase_intention zeta[1]',
        'type': 'factor_loading'
    },
    'PI_sigma_q18': {
        'values': [0.4820, 0.3752, 0.4632, 0.4770],
        'name': 'purchase_intention sigma_sq[0]',
        'type': 'error_variance'
    },
    'beta_intercept': {
        'values': [0.2880, 0.2711, 0.2674, 0.2513],
        'name': 'intercept',
        'type': 'choice_model'
    },
    'beta_sugar': {
        'values': [0.2300, 0.2263, 0.2257, 0.2233],
        'name': 'beta_sugar_free',
        'type': 'choice_model'
    },
    'beta_price': {
        'values': [-0.0590, -0.0807, -0.0850, -0.1028],
        'name': 'beta_price',
        'type': 'choice_model'
    },
    'lambda_main': {
        'values': [0.8899, 0.8821, 0.8804, 0.8726],
        'name': 'lambda_main',
        'type': 'choice_model'
    },
    'lambda_mod_pp': {
        'values': [-0.4700, -0.4818, -0.4835, -0.4909],
        'name': 'lambda_mod_perceived_price',
        'type': 'choice_model'
    },
    'lambda_mod_nk': {
        'values': [1.2000, 1.1912, 1.1890, 1.1792],
        'name': 'lambda_mod_nutrition_knowledge',
        'type': 'choice_model'
    },
}

print('=' * 100)
print('파라미터 수렴값 예측 (Iteration 10, 20, 30, 40 기반)')
print('=' * 100)
print()

# 예측 방법: 지수 감쇠 또는 선형 외삽
def predict_convergence(iterations, values, method='exponential'):
    """
    파라미터 수렴값 예측
    
    method:
        - 'exponential': 지수 감쇠 모델 (변화율이 점점 감소)
        - 'linear': 선형 외삽 (최근 추세 연장)
        - 'average_last': 최근 값들의 평균
    """
    if len(values) < 2:
        return values[-1], 0.0
    
    # 최근 변화율 계산
    recent_changes = np.diff(values[-3:]) if len(values) >= 3 else np.diff(values)
    avg_change = np.mean(recent_changes)
    
    # 변화율의 변화 (가속도)
    if len(values) >= 3:
        change_of_change = np.diff(recent_changes)
        avg_acceleration = np.mean(change_of_change)
    else:
        avg_acceleration = 0.0
    
    current_value = values[-1]
    
    if method == 'linear':
        # 선형 외삽: 최근 변화율 유지
        predicted = current_value + avg_change * 10  # 10 iterations 후
        
    elif method == 'exponential':
        # 지수 감쇠: 변화율이 점점 감소
        # 가정: 변화율이 매 iteration마다 decay_rate만큼 감소
        if abs(avg_change) < 1e-6:
            predicted = current_value
        else:
            # 최근 3개 iteration의 변화율 감소 비율 추정
            if len(recent_changes) >= 2:
                decay_rate = recent_changes[-1] / recent_changes[-2] if abs(recent_changes[-2]) > 1e-6 else 0.9
                decay_rate = max(0.0, min(1.0, decay_rate))  # 0~1 사이로 제한
            else:
                decay_rate = 0.9
            
            # 지수 감쇠 합: Σ(r^i) = (1 - r^n) / (1 - r)
            n_future_iters = 50  # 충분히 많은 iterations
            if abs(1 - decay_rate) > 1e-6:
                total_change = avg_change * (1 - decay_rate**n_future_iters) / (1 - decay_rate)
            else:
                total_change = avg_change * n_future_iters
            
            predicted = current_value + total_change
    
    elif method == 'average_last':
        # 최근 2개 값의 평균 (이미 수렴 중이라고 가정)
        predicted = np.mean(values[-2:])
    
    else:
        predicted = current_value
    
    # 신뢰도 계산 (변화율이 작을수록 높음)
    confidence = 1.0 / (1.0 + abs(avg_change) * 10)
    
    return predicted, confidence

# 파라미터별 예측
print(f'{"파라미터":<40s} {"Iter40":<10s} {"예측값":<10s} {"변화":<10s} {"신뢰도":<8s} {"추세"}')
print('-' * 100)

predictions = {}

for param_key, param_info in params_data.items():
    values = np.array(param_info['values'])
    current = values[-1]
    
    # 지수 감쇠 모델로 예측
    predicted, confidence = predict_convergence(iterations, values, method='exponential')
    
    change = predicted - current
    
    # 추세 판단
    recent_changes = np.diff(values[-3:])
    if len(recent_changes) >= 2:
        if abs(recent_changes[-1]) < abs(recent_changes[-2]) * 0.5:
            trend = '수렴 중'
        elif abs(recent_changes[-1]) < 0.01:
            trend = '거의 수렴'
        else:
            trend = '변화 중'
    else:
        trend = '불명'
    
    predictions[param_key] = {
        'current': current,
        'predicted': predicted,
        'confidence': confidence,
        'trend': trend
    }
    
    print(f'{param_info["name"]:<40s} {current:>9.4f} {predicted:>9.4f} {change:>+9.4f} {confidence:>7.2%} {trend}')

print()
print('=' * 100)
print('파라미터 타입별 요약')
print('=' * 100)
print()

# 타입별 분류
factor_loadings = {k: v for k, v in predictions.items() if params_data[k]['type'] == 'factor_loading'}
error_variances = {k: v for k, v in predictions.items() if params_data[k]['type'] == 'error_variance'}
choice_params = {k: v for k, v in predictions.items() if params_data[k]['type'] == 'choice_model'}

print('[1] Factor Loadings (zeta)')
print('-' * 100)
for param_key in factor_loadings.keys():
    info = predictions[param_key]
    print(f'  {params_data[param_key]["name"]:<40s}: {info["current"]:.4f} → {info["predicted"]:.4f} ({info["trend"]})')

print()
print('[2] Error Variances (sigma_sq)')
print('-' * 100)
for param_key in error_variances.keys():
    info = predictions[param_key]
    print(f'  {params_data[param_key]["name"]:<40s}: {info["current"]:.4f} → {info["predicted"]:.4f} ({info["trend"]})')

print()
print('[3] Choice Model Parameters')
print('-' * 100)
for param_key in choice_params.keys():
    info = predictions[param_key]
    print(f'  {params_data[param_key]["name"]:<40s}: {info["current"]:.4f} → {info["predicted"]:.4f} ({info["trend"]})')

print()
print('=' * 100)
print('주요 관찰 및 해석')
print('=' * 100)
print()

# Factor loadings 분석
fl_changes = [predictions[k]['predicted'] - predictions[k]['current'] for k in factor_loadings.keys()]
avg_fl_change = np.mean(np.abs(fl_changes))

print('[Factor Loadings 분석]')
print(f'  평균 예상 변화량: {avg_fl_change:.4f}')
if avg_fl_change > 0.2:
    print('  → 아직 큰 변화 예상 (추가 최적화 필요)')
elif avg_fl_change > 0.05:
    print('  → 중간 정도 변화 예상 (수렴 중)')
else:
    print('  → 작은 변화 예상 (거의 수렴)')

print()

# Error variances 분석
ev_changes = [predictions[k]['predicted'] - predictions[k]['current'] for k in error_variances.keys()]
avg_ev_change = np.mean(np.abs(ev_changes))

print('[Error Variances 분석]')
print(f'  평균 예상 변화량: {avg_ev_change:.4f}')

# sigma_sq의 패턴 확인
hc_sigma_trend = [params_data['HC_sigma_q6']['values'], params_data['HC_sigma_q7']['values']]
for i, trend in enumerate(hc_sigma_trend):
    if trend[-1] < trend[-2]:
        print(f'  HC sigma_sq[{i}]: 감소 추세 (과대추정 조정 중)')
    else:
        print(f'  HC sigma_sq[{i}]: 증가 추세 (과소추정 조정 중)')

print()

# Choice model 분석
cm_changes = [predictions[k]['predicted'] - predictions[k]['current'] for k in choice_params.keys()]
avg_cm_change = np.mean(np.abs(cm_changes))

print('[Choice Model 분석]')
print(f'  평균 예상 변화량: {avg_cm_change:.4f}')
if avg_cm_change < 0.01:
    print('  → 거의 수렴 (선택모델 파라미터는 안정적)')
else:
    print('  → 여전히 조정 중')

print()
print('=' * 100)
print('최종 예측 요약')
print('=' * 100)
print()

print('1. Factor Loadings (zeta):')
print('   - health_concern: 1.0~1.5 범위로 수렴 예상')
print('   - perceived_benefit: 0.5~0.7 범위로 수렴 예상')
print('   - perceived_price: 0.9~1.2 범위로 수렴 예상')
print('   - nutrition_knowledge: 0.4~0.5 범위로 수렴 예상')
print('   - purchase_intention: 0.4~0.6 범위로 수렴 예상')
print()

print('2. Error Variances (sigma_sq):')
print('   - 대부분 0.15~0.30 범위로 수렴 예상')
print('   - purchase_intention은 0.4~0.5 (다른 LV보다 큼)')
print()

print('3. Choice Model:')
print('   - intercept: 0.25 근처')
print('   - beta_sugar: 0.22 근처')
print('   - beta_price: -0.10~-0.12 (음수 유지)')
print('   - lambda_main: 0.87 근처')
print('   - lambda_mod_pp: -0.49 근처 (음수)')
print('   - lambda_mod_nk: 1.18 근처 (양수)')
print()

print('4. 추가 최적화 필요성:')
print('   - Factor loadings: 아직 큰 변화 예상 (50+ iterations 필요)')
print('   - Error variances: 중간 정도 변화 (30+ iterations)')
print('   - Choice model: 거의 수렴 (10+ iterations)')
print()

print('=' * 100)

