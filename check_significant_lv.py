"""
20개 모델 케이스에서 유의한 잠재변수/상호작용항 검색
"""

import pandas as pd
from pathlib import Path

results_dir = Path('results/sequential_stage_wise')
csv_files = sorted(results_dir.glob('st2_HC-PB_PB-PI1_*2_results.csv'))

print('='*100)
print('유의한 잠재변수/상호작용항이 있는 모델 검색 (p < 0.10)')
print('='*100)

significant_models = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # 파라미터 섹션만 필터링 (대소문자 구분 없이)
    param_df = df[df['section'].str.lower() == 'parameters'].copy()

    # theta (LV 주효과) 또는 gamma (LV-Attr 상호작용) 파라미터만 필터링
    lv_params = param_df[
        param_df['parameter'].str.startswith('theta_') |
        param_df['parameter'].str.startswith('gamma_')
    ].copy()

    if len(lv_params) > 0:
        # p-value < 0.10인 파라미터 찾기
        lv_params['p_value'] = pd.to_numeric(lv_params['p_value'], errors='coerce')
        significant = lv_params[lv_params['p_value'] < 0.10]

        if len(significant) > 0:
            # 적합도 정보 추출
            fit_df = df[df['section'].str.lower() == 'model_fit']
            aic_row = fit_df[fit_df['parameter'] == 'AIC']
            aic = aic_row['estimate'].values[0] if len(aic_row) > 0 else 'N/A'
            
            model_name = csv_file.stem.replace('st2_HC-PB_PB-PI1_', '').replace('2_results', '')
            
            significant_models.append({
                'file': csv_file.name,
                'model': model_name,
                'aic': float(aic) if aic != 'N/A' else 999999,
                'significant_params': significant
            })
            
            print(f'\n[모델] {model_name}')
            print(f'  파일: {csv_file.name}')
            print(f'  AIC: {aic}')
            print(f'  유의한 LV 파라미터 ({len(significant)}개):')
            
            for _, row in significant.iterrows():
                param = row['parameter']
                coef = row['estimate']
                se = row['std_error']
                t_stat = row['t_statistic']
                p_val = row['p_value']
                sig = row['significance']
                
                print(f'    - {param:50s}: {coef:8.4f} (SE={se:6.4f}, t={t_stat:7.3f}, p={p_val:.4f}) {sig}')

print('\n' + '='*100)
print(f'총 {len(significant_models)}개 모델에서 유의한 LV 파라미터 발견')
print('='*100)

if len(significant_models) == 0:
    print('\n[경고] 유의한 잠재변수/상호작용항이 있는 모델이 없습니다.')
    print('       모든 모델에서 theta, gamma 파라미터가 p >= 0.10')
else:
    # AIC 순으로 정렬
    significant_models.sort(key=lambda x: x['aic'])
    
    print('\n' + '='*100)
    print('유의한 LV 파라미터가 있는 모델 - AIC 순위')
    print('='*100)
    
    for i, model in enumerate(significant_models, 1):
        print(f"{i}. {model['model']:40s} (AIC: {model['aic']:.2f})")

