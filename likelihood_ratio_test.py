"""
우도비 검정 (Likelihood Ratio Test)

Base Model과 다른 모델들 간의 우도비 검정을 수행합니다.
LR = -2 * (LL_restricted - LL_unrestricted)
LR ~ Chi-square(df = k_unrestricted - k_restricted)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# 결과 디렉토리
results_dir = Path('results/sequential_stage_wise')

# 모델 정의 (파라미터 개수 포함)
MODELS = {
    'base': {'file': 'st2_HC-PB_PB-PI1_base2_results.csv', 'params': 4, 'description': 'Base Model'},
    'PI': {'file': 'st2_HC-PB_PB-PI1_PI2_results.csv', 'params': 6, 'description': 'Base + PI'},
    'NK': {'file': 'st2_HC-PB_PB-PI1_NK2_results.csv', 'params': 6, 'description': 'Base + NK'},
    'PB': {'file': 'st2_HC-PB_PB-PI1_PB2_results.csv', 'params': 6, 'description': 'Base + PB'},
    'PP': {'file': 'st2_HC-PB_PB-PI1_PP2_results.csv', 'params': 6, 'description': 'Base + PP'},
    'HC': {'file': 'st2_HC-PB_PB-PI1_HC2_results.csv', 'params': 6, 'description': 'Base + HC'},
    'PI_NK': {'file': 'st2_HC-PB_PB-PI1_PI_NK2_results.csv', 'params': 8, 'description': 'Base + PI + NK'},
    'PI_PB': {'file': 'st2_HC-PB_PB-PI1_PI_PB2_results.csv', 'params': 8, 'description': 'Base + PI + PB'},
    'PI_PP': {'file': 'st2_HC-PB_PB-PI1_PI_PP2_results.csv', 'params': 8, 'description': 'Base + PI + PP'},
    'PI_HC': {'file': 'st2_HC-PB_PB-PI1_PI_HC2_results.csv', 'params': 8, 'description': 'Base + PI + HC'},
    'PI_int_PIxpr': {'file': 'st2_HC-PB_PB-PI1_PI_int_PIxpr2_results.csv', 'params': 8, 'description': 'Base + PI + int_PIxpr'},
    'PI_int_PIxhl': {'file': 'st2_HC-PB_PB-PI1_PI_int_PIxhl2_results.csv', 'params': 8, 'description': 'Base + PI + int_PIxhl'},
    'PI_int_NKxpr': {'file': 'st2_HC-PB_PB-PI1_PI_int_NKxpr2_results.csv', 'params': 8, 'description': 'Base + PI + int_NKxpr'},
    'PI_int_NKxhl': {'file': 'st2_HC-PB_PB-PI1_PI_int_NKxhl2_results.csv', 'params': 8, 'description': 'Base + PI + int_NKxhl'},
    'PI_NK_int_PIxpr': {'file': 'st2_HC-PB_PB-PI1_PI_NK_int_PIxpr2_results.csv', 'params': 10, 'description': 'Base + PI + NK + int_PIxpr'},
    'PI_NK_int_PIxhl': {'file': 'st2_HC-PB_PB-PI1_PI_NK_int_PIxhl2_results.csv', 'params': 10, 'description': 'Base + PI + NK + int_PIxhl'},
    'PI_NK_int_NKxpr': {'file': 'st2_HC-PB_PB-PI1_PI_NK_int_NKxpr2_results.csv', 'params': 10, 'description': 'Base + PI + NK + int_NKxpr'},
    'PI_NK_int_NKxhl': {'file': 'st2_HC-PB_PB-PI1_PI_NK_int_NKxhl2_results.csv', 'params': 10, 'description': 'Base + PI + NK + int_NKxhl'},
    'PI_NK_int_PIxpr_NKxhl': {'file': 'st2_HC-PB_PB-PI1_PI_NK_int_PIxpr_NKxhl2_results.csv', 'params': 12, 'description': 'Base + PI + NK + int_PIxpr_NKxhl'},
    'PI_NK_int_PIxhl_NKxpr': {'file': 'st2_HC-PB_PB-PI1_PI_NK_int_PIxhl_NKxpr2_results.csv', 'params': 12, 'description': 'Base + PI + NK + int_PIxhl_NKxpr'},
}


def extract_ll_aic(csv_file):
    """CSV 파일에서 LL과 AIC 추출"""
    df = pd.read_csv(csv_file)
    fit_df = df[df['section'].str.lower() == 'model_fit']
    
    ll_row = fit_df[fit_df['parameter'] == 'log_likelihood']
    aic_row = fit_df[fit_df['parameter'] == 'AIC']
    
    ll = float(ll_row['estimate'].values[0]) if len(ll_row) > 0 else None
    aic = float(aic_row['estimate'].values[0]) if len(aic_row) > 0 else None
    
    return ll, aic


def likelihood_ratio_test(ll_restricted, ll_unrestricted, df_diff):
    """
    우도비 검정
    
    Args:
        ll_restricted: 제약 모델의 Log-Likelihood
        ll_unrestricted: 비제약 모델의 Log-Likelihood
        df_diff: 자유도 차이 (파라미터 개수 차이)
    
    Returns:
        lr_stat: LR 통계량
        p_value: p-value
    """
    lr_stat = -2 * (ll_restricted - ll_unrestricted)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    
    return lr_stat, p_value


# 모든 모델의 LL, AIC 추출
model_stats = {}
for model_name, model_info in MODELS.items():
    csv_path = results_dir / model_info['file']
    if csv_path.exists():
        ll, aic = extract_ll_aic(csv_path)
        model_stats[model_name] = {
            'description': model_info['description'],
            'params': model_info['params'],
            'LL': ll,
            'AIC': aic
        }

# Base Model을 기준으로 우도비 검정
base_ll = model_stats['base']['LL']
base_params = model_stats['base']['params']

print('='*100)
print('우도비 검정 (Likelihood Ratio Test)')
print('='*100)
print(f'\n[기준 모델] Base Model')
print(f'  LL = {base_ll:.4f}')
print(f'  파라미터 개수 = {base_params}')
print(f'  AIC = {model_stats["base"]["AIC"]:.2f}')

print('\n' + '='*100)
print('Base Model vs. 다른 모델들')
print('='*100)
print(f'{"모델":<45s} {"LL":>12s} {"AIC":>10s} {"LR":>10s} {"df":>5s} {"p-value":>10s} {"유의성":>8s}')
print('-'*100)

results = []

for model_name, stats_dict in model_stats.items():
    if model_name == 'base':
        continue
    
    ll = stats_dict['LL']
    aic = stats_dict['AIC']
    params = stats_dict['params']
    desc = stats_dict['description']
    
    # 우도비 검정
    df_diff = params - base_params
    lr_stat, p_value = likelihood_ratio_test(base_ll, ll, df_diff)
    
    # 유의성 표시
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = ''
    
    results.append({
        'model': model_name,
        'description': desc,
        'LL': ll,
        'AIC': aic,
        'params': params,
        'LR': lr_stat,
        'df': df_diff,
        'p_value': p_value,
        'significant': sig
    })
    
    print(f'{desc:<45s} {ll:12.4f} {aic:10.2f} {lr_stat:10.4f} {df_diff:5d} {p_value:10.4f} {sig:>8s}')

# AIC 순으로 정렬하여 상위 모델 표시
print('\n' + '='*100)
print('AIC 기준 상위 5개 모델 (우도비 검정 결과 포함)')
print('='*100)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AIC')

for i, row in results_df.head(5).iterrows():
    print(f"\n{row['description']}")
    print(f"  LL = {row['LL']:.4f}, AIC = {row['AIC']:.2f}")
    print(f"  LR = {row['LR']:.4f}, df = {row['df']}, p = {row['p_value']:.4f} {row['significant']}")

print('\n' + '='*100)
print('유의성: *** p<0.001, ** p<0.01, * p<0.05')
print('='*100)

# 결과를 CSV로 저장
output_df = pd.DataFrame(results)
output_df = output_df.sort_values('AIC')
output_path = results_dir / 'likelihood_ratio_test_results.csv'
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'\n[결과 저장] {output_path}')

