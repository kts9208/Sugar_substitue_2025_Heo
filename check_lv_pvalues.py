"""
20개 모델 케이스에서 잠재변수/상호작용항의 p-value 확인
"""

import pandas as pd
from pathlib import Path

results_dir = Path('results/sequential_stage_wise')
csv_files = sorted(results_dir.glob('st2_HC-PB_PB-PI1_*2_results.csv'))

print('='*100)
print('잠재변수/상호작용항 p-value 분석')
print('='*100)

all_lv_params = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # 파라미터 섹션만 필터링
    param_df = df[df['section'].str.lower() == 'parameters'].copy()
    
    # theta (LV 주효과) 또는 gamma (LV-Attr 상호작용) 파라미터만 필터링
    lv_params = param_df[
        param_df['parameter'].str.startswith('theta_') | 
        param_df['parameter'].str.startswith('gamma_')
    ].copy()
    
    if len(lv_params) > 0:
        # 적합도 정보 추출
        fit_df = df[df['section'].str.lower() == 'model_fit']
        aic_row = fit_df[fit_df['parameter'] == 'AIC']
        aic = aic_row['estimate'].values[0] if len(aic_row) > 0 else 999999
        
        model_name = csv_file.stem.replace('st2_HC-PB_PB-PI1_', '').replace('2_results', '')
        
        for _, row in lv_params.iterrows():
            all_lv_params.append({
                'model': model_name,
                'aic': float(aic),
                'parameter': row['parameter'],
                'estimate': row['estimate'],
                'std_error': row['std_error'],
                't_statistic': row['t_statistic'],
                'p_value': row['p_value']
            })

# DataFrame으로 변환
df_all = pd.DataFrame(all_lv_params)
df_all['p_value'] = pd.to_numeric(df_all['p_value'], errors='coerce')

# p-value 순으로 정렬
df_sorted = df_all.sort_values('p_value')

print(f'\n총 {len(df_all)}개 잠재변수 파라미터 발견')
print(f'\n[p-value가 가장 작은 상위 20개]')
print('='*100)

for i, (_, row) in enumerate(df_sorted.head(20).iterrows(), 1):
    model = row['model']
    param = row['parameter']
    est = row['estimate']
    se = row['std_error']
    t = row['t_statistic']
    p = row['p_value']
    aic = row['aic']
    
    print(f"{i:2d}. [{model:40s}] {param:50s}")
    print(f"    계수={est:7.4f}, SE={se:6.4f}, t={t:7.3f}, p={p:.4f}, AIC={aic:.2f}")

# 통계 요약
print(f'\n{"="*100}')
print('p-value 분포')
print(f'{"="*100}')
print(f'p < 0.05: {len(df_all[df_all["p_value"] < 0.05])}개')
print(f'p < 0.10: {len(df_all[df_all["p_value"] < 0.10])}개')
print(f'p < 0.20: {len(df_all[df_all["p_value"] < 0.20])}개')
print(f'p >= 0.20: {len(df_all[df_all["p_value"] >= 0.20])}개')
print(f'\n최소 p-value: {df_all["p_value"].min():.4f}')
print(f'최대 p-value: {df_all["p_value"].max():.4f}')
print(f'평균 p-value: {df_all["p_value"].mean():.4f}')
print(f'중앙값 p-value: {df_all["p_value"].median():.4f}')

