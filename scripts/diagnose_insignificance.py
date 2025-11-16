"""
잠재변수 비유의성 원인 진단

선택모델에서 잠재변수 효과가 비유의한 원인을 분석합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '..')

print('=' * 100)
print('잠재변수 비유의성 원인 진단')
print('=' * 100)
print()

# 1. 부트스트랩 결과 로드
print('1. 부트스트랩 결과 확인')
print('-' * 100)

df_boot = pd.read_csv('../results/sequential_bootstrap_ci_1000.csv')

# 선택모델 파라미터만 추출
choice_params = df_boot[df_boot['Model'] == 'Choice'].copy()

print('선택모델 파라미터:')
print(choice_params[['Parameter', 'Mean', 'SE', 'CI_Lower', 'CI_Upper', 'p_value_bootstrap', 'Significant']].to_string(index=False))
print()

# 2. 잠재변수 파라미터 분석
print('2. 잠재변수 파라미터 상세 분석')
print('-' * 100)

lv_params = choice_params[choice_params['Parameter'].str.contains('lambda')].copy()

for _, row in lv_params.iterrows():
    param = row['Parameter']
    mean = row['Mean']
    se = row['SE']
    ci_lower = row['CI_Lower']
    ci_upper = row['CI_Upper']
    
    print(f'\n[{param}]')
    print(f'  추정값 (Mean):     {mean:>10.4f}')
    print(f'  표준오차 (SE):     {se:>10.4f}')
    print(f'  95% CI:            [{ci_lower:>8.4f}, {ci_upper:>8.4f}]')
    print(f'  t-통계량:          {mean/se:>10.4f}')
    print(f'  SE/|Mean| 비율:    {se/abs(mean) if mean != 0 else np.inf:>10.4f}')
    print(f'  CI가 0 포함:       {"Yes (비유의)" if ci_lower < 0 < ci_upper else "No (유의)"}')

print()

# 3. 문제 진단
print('3. 문제 진단')
print('-' * 100)
print()

print('진단 1: 추정값 크기')
print('  - lambda_main:                 {:.4f}'.format(lv_params[lv_params['Parameter'] == 'lambda_main']['Mean'].values[0]))
print('  - lambda_mod_perceived_price:  {:.4f}'.format(lv_params[lv_params['Parameter'] == 'lambda_mod_perceived_price']['Mean'].values[0]))
print('  - lambda_mod_nutrition_knowledge: {:.4f}'.format(lv_params[lv_params['Parameter'] == 'lambda_mod_nutrition_knowledge']['Mean'].values[0]))
print('  → 모든 추정값이 0에 매우 가까움 (절대값 < 0.05)')
print()

print('진단 2: 표준오차 크기')
se_main = lv_params[lv_params['Parameter'] == 'lambda_main']['SE'].values[0]
se_pp = lv_params[lv_params['Parameter'] == 'lambda_mod_perceived_price']['SE'].values[0]
se_nk = lv_params[lv_params['Parameter'] == 'lambda_mod_nutrition_knowledge']['SE'].values[0]

print(f'  - lambda_main SE:                 {se_main:.4f}')
print(f'  - lambda_mod_perceived_price SE:  {se_pp:.4f}')
print(f'  - lambda_mod_nutrition_knowledge SE: {se_nk:.4f}')
print(f'  → 표준오차가 추정값보다 훨씬 큼 (SE >> |Mean|)')
print()

# 4. 요인점수 분석
print('4. 요인점수 특성 분석')
print('-' * 100)

# 최신 요인점수 파일 로드
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_선택모델_확장_후_*.csv'))
if files:
    df_fs = pd.read_csv(files[-1])
    
    print(f'\n요인점수 통계 (선택 수준, N={len(df_fs)}):')
    print()
    
    for col in ['purchase_intention', 'perceived_price', 'nutrition_knowledge']:
        if col in df_fs.columns:
            values = df_fs[col].values
            print(f'  [{col}]')
            print(f'    평균:     {np.mean(values):>10.6f}')
            print(f'    분산:     {np.var(values, ddof=1):>10.6f}')
            print(f'    표준편차: {np.std(values, ddof=1):>10.6f}')
            print(f'    범위:     [{np.min(values):>8.4f}, {np.max(values):>8.4f}]')
            print()

print()

# 5. 속성변수와 비교
print('5. 속성변수와 잠재변수 비교')
print('-' * 100)
print()

attr_params = choice_params[choice_params['Parameter'].str.contains('β_')].copy()

print('속성변수 (유의함):')
for _, row in attr_params.iterrows():
    param = row['Parameter']
    mean = row['Mean']
    se = row['SE']
    print(f'  {param:20s}: Mean={mean:>8.4f}, SE={se:>8.4f}, t={mean/se:>8.4f}')

print()
print('잠재변수 (비유의):')
for _, row in lv_params.iterrows():
    param = row['Parameter']
    mean = row['Mean']
    se = row['SE']
    print(f'  {param:35s}: Mean={mean:>8.4f}, SE={se:>8.4f}, t={mean/se:>8.4f}')

print()

# 6. 가능한 원인 정리
print('6. 비유의성의 가능한 원인')
print('-' * 100)
print()

print('원인 1: 실제로 효과가 없음 (True Null)')
print('  - 잠재변수가 선택에 실제로 영향을 미치지 않음')
print('  - 구조모델에서는 유의하지만 선택모델에서는 무관')
print()

print('원인 2: 모델 식별 문제 (Identification)')
print('  - 잠재변수가 이미 속성변수와 높은 상관관계')
print('  - 다중공선성으로 인한 추정 불안정')
print()

print('원인 3: 척도 문제 (Scale)')
print('  - 요인점수의 분산이 작아서 효과 크기가 작음')
print('  - 표준화가 필요할 수 있음')
print()

print('원인 4: 모델 설정 오류 (Specification)')
print('  - 조절효과 모델이 데이터에 맞지 않음')
print('  - 주효과만 있고 조절효과는 없을 가능성')
print()

print('원인 5: 샘플 크기 부족 (Power)')
print('  - 효과가 작아서 검정력이 부족')
print('  - 더 많은 샘플이 필요')
print()

print('=' * 100)

