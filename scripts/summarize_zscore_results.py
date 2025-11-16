"""
Z-score 표준화 후 순차추정 결과 요약
"""

import pandas as pd
import numpy as np

# 현재 결과 (Z-score 표준화 후)
df_after = pd.read_csv('../results/sequential_estimation_results.csv')
choice_after = df_after[df_after['Model'] == 'Choice'].copy()

print('=' * 100)
print('Z-score 표준화 후 순차추정 결과')
print('=' * 100)
print()
print('선택모델 파라미터:')
print()
print(f'{"Parameter":35s} {"Estimate":>12s} {"Std_Err":>12s} {"p-value":>12s} {"Sig":>10s}')
print('-' * 85)

for _, row in choice_after.iterrows():
    param = row['Parameter']
    est = float(row['Estimate'])
    se = float(row['Std_Err'])
    p = float(row['p_value'])

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

    print(f'{param:35s} {est:>12.4f} {se:>12.4f} {p:>12.4f} {sig:>10s}')

print()
print('유의성: *** p<0.001, ** p<0.01, * p<0.05')
print()

# 주요 발견
print('=' * 100)
print('주요 발견:')
print('=' * 100)
print()

# 속성 변수
print('1. 선택 속성 변수 (Z-score 표준화됨):')
print()
attr_params = ['β_sugar_free', 'β_health_label', 'β_price']
for param in attr_params:
    row = choice_after[choice_after['Parameter'] == param]
    if len(row) > 0:
        est = float(row['Estimate'].values[0])
        p = float(row['p_value'].values[0])
        sig_text = '유의함' if p < 0.05 else '비유의'
        print(f'  {param:20s}: {est:>8.4f} (p={p:.4f}) - {sig_text}')

print()

# 잠재변수
print('2. 잠재변수 효과 (Z-score 표준화됨):')
print()
lv_params = ['lambda_main', 'lambda_mod_perceived_price', 'lambda_mod_nutrition_knowledge']
for param in lv_params:
    row = choice_after[choice_after['Parameter'] == param]
    if len(row) > 0:
        est = float(row['Estimate'].values[0])
        p = float(row['p_value'].values[0])
        sig_text = '유의함' if p < 0.05 else '비유의'
        print(f'  {param:35s}: {est:>8.4f} (p={p:.4f}) - {sig_text}')

print()
print('=' * 100)
print()

# 해석
print('해석:')
print()
print('Z-score 표준화 후:')
print('- 모든 변수가 평균=0, 표준편차=1로 표준화됨')
print('- 계수는 "1 표준편차 변화의 효과"를 나타냄')
print('- 스케일 불균형 문제 해결됨')
print()
print('결과:')
print('- β_price만 유의함 (p=0.0332)')
print('- β_sugar_free, β_health_label 비유의')
print('- 모든 잠재변수 효과 비유의')
print()
print('다음 단계:')
print('- 부트스트랩 1000회 실행하여 표준오차 재추정')
print('- 또는 모델 단순화 (조절효과 제거) 고려')
print()
print('=' * 100)

