import pandas as pd
from pathlib import Path

results_dir = Path('results/sequential_stage_wise')

# 모든 통합 결과 파일 목록 (신규 파일명 규칙)
result_files = {
    'Base': 'st2_HC-PB_PB-PI1_base2_results.csv',
    'Base+PI': 'st2_HC-PB_PB-PI1_PI2_results.csv',
    'Base+NK': 'st2_HC-PB_PB-PI1_NK2_results.csv',
    'Base+PP': 'st2_HC-PB_PB-PI1_PP2_results.csv',
    'Base+PI+NK': 'st2_HC-PB_PB-PI1_PI_NK2_results.csv',
    'Base+(PIxhl)': 'st2_HC-PB_PB-PI1_int_PIxhl2_results.csv',
    'Base+(PIxpr)': 'st2_HC-PB_PB-PI1_int_PIxpr2_results.csv',
    'Base+(NKxhl)': 'st2_HC-PB_PB-PI1_int_NKxhl2_results.csv',
    'Base+(NKxpr)': 'st2_HC-PB_PB-PI1_int_NKxpr2_results.csv',
    'Base+(PPxhl)': 'st2_HC-PB_PB-PI1_int_PPxhl2_results.csv',
    'Base+(PPxpr)': 'st2_HC-PB_PB-PI1_int_PPxpr2_results.csv',
    'Base+PI+(PIxhl)': 'st2_HC-PB_PB-PI1_PI_int_PIxhl2_results.csv',
    'Base+PI+(PIxpr)': 'st2_HC-PB_PB-PI1_PI_int_PIxpr2_results.csv',
    'Base+NK+(NKxpr)': 'st2_HC-PB_PB-PI1_NK_int_NKxpr2_results.csv',
    'Base+NK+(NKxhl)': 'st2_HC-PB_PB-PI1_NK_int_NKxhl2_results.csv',
    'Base+PP+(PPxhl)': 'st2_HC-PB_PB-PI1_PP_int_PPxhl2_results.csv',
    'Base+PP+(PPxpr)': 'st2_HC-PB_PB-PI1_PP_int_PPxpr2_results.csv',
    'Base+PI+NK+(PIxpr)': 'st2_HC-PB_PB-PI1_PI_NK_int_PIxpr2_results.csv',
    'Base+PI+NK+(NKxhl)': 'st2_HC-PB_PB-PI1_PI_NK_int_NKxhl2_results.csv',
    'Base+PI+NK+(NKxhl)+(PIxpr)': 'st2_HC-PB_PB-PI1_PI_NK_int_NKxhl_PIxpr2_results.csv',
}

print('='*120)
print('전체 모델 비교 요약 (20개 모델)')
print('='*120)

comparison = []

for model_name, filename in result_files.items():
    filepath = results_dir / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        
        # Model_Fit 섹션 추출
        fit_data = df[df['section'] == 'Model_Fit'].set_index('parameter')
        
        # 파라미터 개수 (Parameters 섹션)
        n_params = len(df[df['section'] == 'Parameters'])
        
        # 유의한 파라미터 개수 (p < 0.05)
        params_df = df[df['section'] == 'Parameters']
        sig_params = params_df[params_df['p_value'].notna() & (params_df['p_value'].astype(float) < 0.05)]
        n_sig = len(sig_params)
        
        comparison.append({
            'Model': model_name,
            'Log-Likelihood': float(fit_data.loc['log_likelihood', 'estimate']),
            'AIC': float(fit_data.loc['AIC', 'estimate']),
            'BIC': float(fit_data.loc['BIC', 'estimate']),
            'N_Params': n_params,
            'N_Sig': n_sig,
            'Sig_Ratio': f'{n_sig}/{n_params}'
        })
    else:
        print(f'경고: {filename} 파일이 존재하지 않습니다.')

comp_df = pd.DataFrame(comparison)
comp_df = comp_df.sort_values('AIC')
comp_df['Rank'] = range(1, len(comp_df) + 1)

# 컬럼 순서 조정
comp_df = comp_df[['Rank', 'Model', 'Log-Likelihood', 'AIC', 'BIC', 'N_Params', 'Sig_Ratio']]

print(comp_df.to_string(index=False))
print('\n' + '='*120)
print(f'\n최적 모델 (AIC 기준): {comp_df.iloc[0]["Model"]}')
print(f'   - Log-Likelihood: {comp_df.iloc[0]["Log-Likelihood"]:.2f}')
print(f'   - AIC: {comp_df.iloc[0]["AIC"]:.2f}')
print(f'   - BIC: {comp_df.iloc[0]["BIC"]:.2f}')
print(f'   - 유의한 파라미터: {comp_df.iloc[0]["Sig_Ratio"]}')
print('\n' + '='*120)

# CSV로 저장
output_path = results_dir / 'model_comparison_summary.csv'
comp_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'\n비교 결과 저장: {output_path}')

