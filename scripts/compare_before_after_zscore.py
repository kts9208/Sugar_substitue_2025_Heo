"""
Z-score 표준화 전후 비교
"""

import pandas as pd
import numpy as np

print('=' * 100)
print('Z-score 표준화 전후 비교')
print('=' * 100)
print()

# 이전 결과 (표준화 전)
df_before = pd.read_csv('../results/sequential_bootstrap_ci_1000.csv')
choice_before = df_before[df_before['Model'] == 'Choice'].copy()

# 현재 결과 (표준화 후)
df_after = pd.read_csv('../results/sequential_estimation_results.csv')
choice_after = df_after[df_after['Model'] == 'Choice'].copy()

print('선택모델 파라미터 비교:')
print()
print(f'{"파라미터":35s} {"표준화 전":>15s} {"표준화 후":>15s} {"변화":>15s}')
print('-' * 85)

# 파라미터 매핑
param_map = {
    'intercept': 'intercept',
    'β_sugar_free': 'β_sugar_free',
    'β_health_label': 'β_health_label',
    'β_price': 'β_price',
    'lambda_main': 'lambda_main',
    'lambda_mod_perceived_price': 'lambda_mod_perceived_price',
    'lambda_mod_nutrition_knowledge': 'lambda_mod_nutrition_knowledge'
}

for param_name, param_key in param_map.items():
    # 이전 값
    before_row = choice_before[choice_before['Parameter'] == param_name]
    if len(before_row) > 0:
        before_val = before_row['Mean'].values[0]
        before_p = before_row['p_value'].values[0]
    else:
        before_val = np.nan
        before_p = np.nan
    
    # 현재 값
    after_row = choice_after[choice_after['Parameter'] == param_key]
    if len(after_row) > 0:
        after_val = after_row['Estimate'].values[0]
        after_p = after_row['p_value'].values[0]
    else:
        after_val = np.nan
        after_p = np.nan
    
    # 변화
    if not np.isnan(before_val) and not np.isnan(after_val):
        change = after_val - before_val
        change_str = f'{change:+.4f}'
    else:
        change_str = 'N/A'
    
    # 유의성 표시
    before_sig = '***' if before_p < 0.001 else '**' if before_p < 0.01 else '*' if before_p < 0.05 else ''
    after_sig = '***' if after_p < 0.001 else '**' if after_p < 0.01 else '*' if after_p < 0.05 else ''
    
    print(f'{param_name:35s} {before_val:>12.4f}{before_sig:>3s} {after_val:>12.4f}{after_sig:>3s} {change_str:>15s}')

print()
print('유의성: *** p<0.001, ** p<0.01, * p<0.05')
print()

# p-value 비교
print('p-value 비교:')
print()
print(f'{"파라미터":35s} {"표준화 전":>15s} {"표준화 후":>15s} {"개선":>10s}')
print('-' * 75)

for param_name, param_key in param_map.items():
    # 이전 값
    before_row = choice_before[choice_before['Parameter'] == param_name]
    if len(before_row) > 0:
        before_p = before_row['p_value'].values[0]
    else:
        before_p = np.nan
    
    # 현재 값
    after_row = choice_after[choice_after['Parameter'] == param_key]
    if len(after_row) > 0:
        after_p = after_row['p_value'].values[0]
    else:
        after_p = np.nan
    
    # 개선 여부
    if not np.isnan(before_p) and not np.isnan(after_p):
        if after_p < before_p:
            improvement = '✓ 개선'
        elif after_p > before_p:
            improvement = '✗ 악화'
        else:
            improvement = '- 동일'
    else:
        improvement = 'N/A'
    
    print(f'{param_name:35s} {before_p:>15.4f} {after_p:>15.4f} {improvement:>10s}')

print()

# 주요 발견
print('주요 발견:')
print('-' * 100)
print()

# 잠재변수 파라미터만 추출
lv_params = ['lambda_main', 'lambda_mod_perceived_price', 'lambda_mod_nutrition_knowledge']

print('잠재변수 파라미터:')
for param_name in lv_params:
    param_key = param_name
    
    before_row = choice_before[choice_before['Parameter'] == param_name]
    after_row = choice_after[choice_after['Parameter'] == param_key]
    
    if len(before_row) > 0 and len(after_row) > 0:
        before_val = before_row['Mean'].values[0]
        before_p = before_row['p_value'].values[0]
        after_val = after_row['Estimate'].values[0]
        after_p = after_row['p_value'].values[0]
        
        print(f'\n{param_name}:')
        print(f'  표준화 전: {before_val:>8.4f} (p={before_p:.4f})')
        print(f'  표준화 후: {after_val:>8.4f} (p={after_p:.4f})')
        
        if after_p < 0.05:
            print(f'  ✓ 유의함!')
        else:
            print(f'  ✗ 여전히 비유의')

print()
print('=' * 100)

