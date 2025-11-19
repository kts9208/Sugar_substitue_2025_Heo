"""
통합 측정모델 파라미터 CSV 파일 확인
"""
import pandas as pd
from pathlib import Path

csv_path = Path('results/sequential_stage_wise/stage1_HC-PB_PB-PI_results_measurement_params.csv')

df = pd.read_csv(csv_path)

print('=== 통합 측정모델 파라미터 CSV ===')
print(f'총 행 수: {len(df)}')
print(f'컬럼: {list(df.columns)}')

# param_type별 개수
print(f'\nparam_type 분포:')
print(df['param_type'].value_counts())

# 요인적재량 샘플
print(f'\n=== 요인적재량 샘플 (처음 5개) ===')
loadings = df[df['param_type'] == 'loading']
print(loadings.head().to_string())

# 오차분산 샘플
print(f'\n=== 오차분산 샘플 (처음 5개) ===')
errors = df[df['param_type'] == 'error_variance']
print(errors.head().to_string())

# health_concern의 측정모델 파라미터
print(f'\n=== health_concern 측정모델 파라미터 ===')
hc_loadings = df[(df['rval'] == 'health_concern') & (df['param_type'] == 'loading')]
print(f'요인적재량: {len(hc_loadings)}개')

hc_errors = df[(df['lval'].isin(['q6', 'q7', 'q8', 'q9', 'q10', 'q11'])) & 
               (df['param_type'] == 'error_variance')]
print(f'오차분산: {len(hc_errors)}개')
print(f'\nhealth_concern 오차분산 값:')
for _, row in hc_errors.iterrows():
    print(f'  {row["lval"]}: {row["Estimate"]:.6f}')

