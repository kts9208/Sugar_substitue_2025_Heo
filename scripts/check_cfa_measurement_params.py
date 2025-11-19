"""
CFA 결과 통합 측정모델 파라미터 CSV 파일 확인
"""
import pandas as pd
from pathlib import Path

csv_path = Path('results/sequential_stage_wise/cfa_results_measurement_params.csv')

df = pd.read_csv(csv_path)

print('=== CFA 통합 측정모델 파라미터 CSV ===')
print(f'총 행 수: {len(df)}')
print(f'컬럼: {list(df.columns)}')

# param_type별 개수
print(f'\nparam_type 분포:')
print(df['param_type'].value_counts())

# 요인적재량 샘플
print(f'\n=== 요인적재량 샘플 (처음 10개) ===')
loadings = df[df['param_type'] == 'loading']
print(loadings.head(10).to_string())

# 오차분산 샘플
print(f'\n=== 오차분산 샘플 (처음 10개) ===')
errors = df[df['param_type'] == 'error_variance']
print(errors.head(10).to_string())

# 각 잠재변수별 측정모델 파라미터 개수
print(f'\n=== 잠재변수별 측정모델 파라미터 개수 ===')
lv_names = ['health_concern', 'perceived_benefit', 'perceived_price', 
            'nutrition_knowledge', 'purchase_intention']

for lv_name in lv_names:
    lv_loadings = df[(df['rval'] == lv_name) & (df['param_type'] == 'loading')]
    
    # 해당 잠재변수의 지표 추출
    indicators = lv_loadings['lval'].tolist()
    
    # 오차분산 개수
    lv_errors = df[(df['lval'].isin(indicators)) & 
                   (df['param_type'] == 'error_variance')]
    
    print(f'\n{lv_name}:')
    print(f'  요인적재량: {len(lv_loadings)}개')
    print(f'  오차분산: {len(lv_errors)}개')
    print(f'  지표: {indicators}')
    
    # 오차분산 값 출력
    if len(lv_errors) > 0:
        print(f'  오차분산 값:')
        for _, row in lv_errors.iterrows():
            print(f'    {row["lval"]}: {row["Estimate"]:.6f}')

