"""
스케일 불일치 진단

효용함수의 속성변수와 잠재점수 간 스케일 차이를 분석합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 100)
print('스케일 불일치 진단')
print('=' * 100)
print()

# 1. 데이터 로드
data_path = Path('../data/processed/iclv/integrated_data_cleaned.csv')
df = pd.read_csv(data_path)

print(f'데이터 로드: {len(df)} 행')
print()

# 요인점수 로드
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_선택모델_확장_후_*.csv'))
if not files:
    print("요인점수 파일을 찾을 수 없습니다!")
    exit(1)

df_fs = pd.read_csv(files[-1])

# 데이터 병합
df_combined = df.copy()
for col in df_fs.columns:
    df_combined[f'fs_{col}'] = df_fs[col].values

print('1. 변수별 스케일 비교')
print('-' * 100)
print()

# 속성변수 (실제 컬럼명 사용)
attr_vars = {
    'sugar_content': '무설탕 여부 (0/1)',
    'health_label': '건강 라벨 여부 (0/1)',
    'price': '가격 (원)'
}

# 요인점수
fs_vars = {
    'fs_purchase_intention': '구매의도 (요인점수)',
    'fs_perceived_price': '지각된 가격 (요인점수)',
    'fs_nutrition_knowledge': '영양지식 (요인점수)',
    'fs_health_concern': '건강관심 (요인점수)',
    'fs_perceived_benefit': '지각된 혜택 (요인점수)'
}

print('속성변수 스케일:')
print()
for var, desc in attr_vars.items():
    values = df_combined[var].values
    print(f'{desc}')
    print(f'  변수명: {var}')
    print(f'  평균:     {np.mean(values):>10.4f}')
    print(f'  분산:     {np.var(values, ddof=1):>10.4f}')
    print(f'  표준편차: {np.std(values, ddof=1):>10.4f}')
    print(f'  범위:     [{np.min(values):>10.2f}, {np.max(values):>10.2f}]')
    print()

print()
print('요인점수 스케일:')
print()
for var, desc in fs_vars.items():
    values = df_combined[var].values
    print(f'{desc}')
    print(f'  변수명: {var}')
    print(f'  평균:     {np.mean(values):>10.4f}')
    print(f'  분산:     {np.var(values, ddof=1):>10.4f}')
    print(f'  표준편차: {np.std(values, ddof=1):>10.4f}')
    print(f'  범위:     [{np.min(values):>10.4f}, {np.max(values):>10.4f}]')
    print()

# 2. 스케일 비율 분석
print()
print('2. 스케일 비율 분석')
print('-' * 100)
print()

# 가격 변수를 기준으로 비교
price_std = np.std(df_combined['price'].values, ddof=1)

print(f'기준: attr_price 표준편차 = {price_std:.4f}')
print()

print('표준편차 비율 (attr_price 대비):')
print()

for var in attr_vars.keys():
    std = np.std(df_combined[var].values, ddof=1)
    ratio = std / price_std
    print(f'  {var:20s}: {std:>10.4f}  (비율: {ratio:>8.4f}x)')

print()

for var in fs_vars.keys():
    std = np.std(df_combined[var].values, ddof=1)
    ratio = std / price_std
    print(f'  {var:35s}: {std:>10.4f}  (비율: {ratio:>8.4f}x)')

print()

# 3. 부트스트랩 결과와 비교
print('3. 추정된 계수와 변수 스케일 비교')
print('-' * 100)
print()

df_boot = pd.read_csv('../results/sequential_bootstrap_ci_1000.csv')
choice_params = df_boot[df_boot['Model'] == 'Choice'].copy()

print('추정된 계수 (부트스트랩 평균):')
print()

for _, row in choice_params.iterrows():
    param = row['Parameter']
    mean = row['Mean']
    se = row['SE']
    print(f'  {param:35s}: {mean:>10.4f} (SE: {se:>8.4f})')

print()

# 4. 효과 크기 계산
print('4. 효과 크기 분석 (1 표준편차 변화의 효용 변화)')
print('-' * 100)
print()

# 속성변수 효과
print('속성변수 효과:')
print()

beta_sugar = choice_params[choice_params['Parameter'] == 'β_sugar_free']['Mean'].values[0]
beta_health = choice_params[choice_params['Parameter'] == 'β_health_label']['Mean'].values[0]
beta_price = choice_params[choice_params['Parameter'] == 'β_price']['Mean'].values[0]

std_sugar = np.std(df_combined['sugar_content'].values, ddof=1)
std_health = np.std(df_combined['health_label'].values, ddof=1)
std_price = np.std(df_combined['price'].values, ddof=1)

effect_sugar = beta_sugar * std_sugar
effect_health = beta_health * std_health
effect_price = beta_price * std_price

print(f'  무설탕:     β={beta_sugar:>8.4f} × σ={std_sugar:>8.4f} = {effect_sugar:>8.4f}')
print(f'  건강라벨:   β={beta_health:>8.4f} × σ={std_health:>8.4f} = {effect_health:>8.4f}')
print(f'  가격:       β={beta_price:>8.4f} × σ={std_price:>8.4f} = {effect_price:>8.4f}')

print()
print('잠재변수 효과:')
print()

lambda_main = choice_params[choice_params['Parameter'] == 'lambda_main']['Mean'].values[0]
lambda_pp = choice_params[choice_params['Parameter'] == 'lambda_mod_perceived_price']['Mean'].values[0]
lambda_nk = choice_params[choice_params['Parameter'] == 'lambda_mod_nutrition_knowledge']['Mean'].values[0]

std_pi = np.std(df_combined['fs_purchase_intention'].values, ddof=1)
std_pp = np.std(df_combined['fs_perceived_price'].values, ddof=1)
std_nk = np.std(df_combined['fs_nutrition_knowledge'].values, ddof=1)

effect_main = lambda_main * std_pi
effect_pp = lambda_pp * std_pp
effect_nk = lambda_nk * std_nk

print(f'  구매의도 (주효과):  λ={lambda_main:>8.4f} × σ={std_pi:>8.4f} = {effect_main:>8.4f}')
print(f'  지각가격 (조절):    λ={lambda_pp:>8.4f} × σ={std_pp:>8.4f} = {effect_pp:>8.4f}')
print(f'  영양지식 (조절):    λ={lambda_nk:>8.4f} × σ={std_nk:>8.4f} = {effect_nk:>8.4f}')

print()

# 5. 스케일 문제 진단
print('5. 스케일 문제 진단')
print('-' * 100)
print()

print('문제 1: 가격 변수의 스케일이 매우 큼')
print(f'  - price 범위: [{df_combined["price"].min():.0f}, {df_combined["price"].max():.0f}] 원')
print(f'  - price 표준편차: {std_price:.2f}')
print(f'  - 이진변수 (0/1) 표준편차: ~0.5')
print(f'  - 요인점수 표준편차: ~0.6-1.0')
print(f'  → 가격 변수가 다른 변수보다 {std_price/0.5:.1f}배 큰 스케일')
print()

print('문제 2: 계수 크기의 불균형')
print(f'  - β_price = {beta_price:.4f} (가격 계수)')
print(f'  - β_sugar_free = {beta_sugar:.4f} (무설탕 계수)')
print(f'  - λ_main = {lambda_main:.4f} (구매의도 계수)')
print(f'  → 가격 계수가 잠재변수 계수보다 {abs(beta_price/lambda_main):.0f}배 큼')
print()

print('문제 3: 조절효과 변수의 스케일')
df_combined['mod_pp'] = df_combined['price'] * df_combined['fs_perceived_price']
df_combined['mod_nk'] = df_combined['price'] * df_combined['fs_nutrition_knowledge']

std_mod_pp = np.std(df_combined['mod_pp'].values, ddof=1)
std_mod_nk = np.std(df_combined['mod_nk'].values, ddof=1)

print(f'  - attr_price × fs_perceived_price 표준편차: {std_mod_pp:.2f}')
print(f'  - attr_price × fs_nutrition_knowledge 표준편차: {std_mod_nk:.2f}')
print(f'  → 조절효과 변수가 주효과 변수보다 {std_mod_pp/std_pi:.1f}배 큰 스케일')
print()

# 6. 해결 방안
print('6. 해결 방안')
print('-' * 100)
print()

print('방안 1: 가격 변수 스케일 조정')
print('  - 가격을 1000원 단위로 변환: price_scaled = price / 1000')
print(f'  - 현재: [{df_combined["price"].min():.0f}, {df_combined["price"].max():.0f}]')
print(f'  - 변환 후: [{df_combined["price"].min()/1000:.1f}, {df_combined["price"].max()/1000:.1f}]')
print(f'  - 표준편차: {std_price:.2f} → {std_price/1000:.4f}')
print()

print('방안 2: 모든 변수 표준화')
print('  - 모든 연속형 변수를 평균 0, 분산 1로 표준화')
print('  - 계수 해석: 1 표준편차 변화의 효과')
print()

print('방안 3: 요인점수 스케일 조정')
print('  - 요인점수를 표준화: (x - mean) / std')
print('  - 현재 요인점수는 이미 평균 0이지만 분산이 1이 아님')
print()

print('=' * 100)

