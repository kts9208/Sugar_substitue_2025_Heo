"""
스케일 문제 분석 (간소화 버전)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 100)
print('스케일 문제 분석')
print('=' * 100)
print()

# 1. 요인점수 로드
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_선택모델_확장_후_*.csv'))
if not files:
    print("요인점수 파일을 찾을 수 없습니다!")
    exit(1)

df_fs = pd.read_csv(files[-1])
print(f'요인점수 데이터: {len(df_fs)} 행')
print()

# 2. 원본 데이터 로드
df_data = pd.read_csv('../data/processed/iclv/integrated_data_cleaned.csv')
print(f'원본 데이터: {len(df_data)} 행')
print()

# 3. 속성변수 통계
print('1. 속성변수 스케일')
print('-' * 100)
print()

# 숫자형으로 변환
df_data['price_num'] = pd.to_numeric(df_data['price'], errors='coerce')
df_data['sugar_num'] = pd.to_numeric(df_data['sugar_content'], errors='coerce')
df_data['health_num'] = pd.to_numeric(df_data['health_label'], errors='coerce')

print(f'가격 (price):')
print(f'  평균:     {df_data["price_num"].mean():>10.2f}')
print(f'  표준편차: {df_data["price_num"].std():>10.2f}')
print(f'  범위:     [{df_data["price_num"].min():>8.0f}, {df_data["price_num"].max():>8.0f}]')
print()

print(f'무설탕 (sugar_content):')
print(f'  평균:     {df_data["sugar_num"].mean():>10.4f}')
print(f'  표준편차: {df_data["sugar_num"].std():>10.4f}')
print(f'  범위:     [{df_data["sugar_num"].min():>8.0f}, {df_data["sugar_num"].max():>8.0f}]')
print()

print(f'건강라벨 (health_label):')
print(f'  평균:     {df_data["health_num"].mean():>10.4f}')
print(f'  표준편차: {df_data["health_num"].std():>10.4f}')
print(f'  범위:     [{df_data["health_num"].min():>8.0f}, {df_data["health_num"].max():>8.0f}]')
print()

# 4. 요인점수 통계
print('2. 요인점수 스케일')
print('-' * 100)
print()

for col in df_fs.columns:
    values = df_fs[col].values
    print(f'{col}:')
    print(f'  평균:     {np.mean(values):>10.4f}')
    print(f'  표준편차: {np.std(values, ddof=1):>10.4f}')
    print(f'  범위:     [{np.min(values):>8.4f}, {np.max(values):>8.4f}]')
    print()

# 5. 스케일 비교
print('3. 스케일 비교 (표준편차 기준)')
print('-' * 100)
print()

price_std = df_data["price_num"].std()
sugar_std = df_data["sugar_num"].std()
health_std = df_data["health_num"].std()

pi_std = df_fs['purchase_intention'].std()
pp_std = df_fs['perceived_price'].std()
nk_std = df_fs['nutrition_knowledge'].std()

print(f'속성변수:')
print(f'  price:         {price_std:>10.2f}  (기준)')
print(f'  sugar_content: {sugar_std:>10.4f}  ({sugar_std/price_std:>6.4f}x)')
print(f'  health_label:  {health_std:>10.4f}  ({health_std/price_std:>6.4f}x)')
print()

print(f'요인점수:')
print(f'  purchase_intention:  {pi_std:>10.4f}  ({pi_std/price_std:>6.4f}x)')
print(f'  perceived_price:     {pp_std:>10.4f}  ({pp_std/price_std:>6.4f}x)')
print(f'  nutrition_knowledge: {nk_std:>10.4f}  ({nk_std/price_std:>6.4f}x)')
print()

# 6. 부트스트랩 결과
print('4. 추정된 계수')
print('-' * 100)
print()

df_boot = pd.read_csv('../results/sequential_bootstrap_ci_1000.csv')
choice_params = df_boot[df_boot['Model'] == 'Choice'].copy()

for _, row in choice_params.iterrows():
    param = row['Parameter']
    mean = row['Mean']
    se = row['SE']
    print(f'{param:35s}: {mean:>10.4f} (SE: {se:>8.4f})')

print()

# 7. 효과 크기 계산
print('5. 효과 크기 (1 표준편차 변화의 효용 변화)')
print('-' * 100)
print()

beta_price = choice_params[choice_params['Parameter'] == 'β_price']['Mean'].values[0]
beta_sugar = choice_params[choice_params['Parameter'] == 'β_sugar_free']['Mean'].values[0]
beta_health = choice_params[choice_params['Parameter'] == 'β_health_label']['Mean'].values[0]

lambda_main = choice_params[choice_params['Parameter'] == 'lambda_main']['Mean'].values[0]
lambda_pp = choice_params[choice_params['Parameter'] == 'lambda_mod_perceived_price']['Mean'].values[0]
lambda_nk = choice_params[choice_params['Parameter'] == 'lambda_mod_nutrition_knowledge']['Mean'].values[0]

print('속성변수 효과:')
print(f'  price:         β={beta_price:>8.4f} × σ={price_std:>8.2f} = {beta_price * price_std:>10.4f}')
print(f'  sugar_content: β={beta_sugar:>8.4f} × σ={sugar_std:>8.4f} = {beta_sugar * sugar_std:>10.4f}')
print(f'  health_label:  β={beta_health:>8.4f} × σ={health_std:>8.4f} = {beta_health * health_std:>10.4f}')
print()

print('잠재변수 효과:')
print(f'  purchase_intention (주효과):  λ={lambda_main:>8.4f} × σ={pi_std:>8.4f} = {lambda_main * pi_std:>10.4f}')
print(f'  perceived_price (조절):       λ={lambda_pp:>8.4f} × σ={pp_std:>8.4f} = {lambda_pp * pp_std:>10.4f}')
print(f'  nutrition_knowledge (조절):   λ={lambda_nk:>8.4f} × σ={nk_std:>8.4f} = {lambda_nk * nk_std:>10.4f}')
print()

# 8. 핵심 발견
print('6. 핵심 발견')
print('-' * 100)
print()

print(f'발견 1: 가격 변수의 스케일이 매우 큼')
print(f'  - price 표준편차: {price_std:.2f}')
print(f'  - 요인점수 표준편차: ~0.6-1.0')
print(f'  - 비율: {price_std/pi_std:.1f}배 차이')
print()

print(f'발견 2: 계수 크기의 불균형')
print(f'  - β_price = {beta_price:.4f}')
print(f'  - λ_main = {lambda_main:.4f}')
print(f'  - 비율: {abs(beta_price/lambda_main):.0f}배 차이')
print()

print(f'발견 3: 효과 크기는 비슷함')
print(f'  - price 효과: {beta_price * price_std:.4f}')
print(f'  - purchase_intention 효과: {lambda_main * pi_std:.4f}')
print(f'  → 스케일 차이를 계수가 보정하고 있음')
print()

print('=' * 100)

