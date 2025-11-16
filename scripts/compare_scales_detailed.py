"""
선택실험 속성값과 잠재점수의 스케일 상세 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 100)
print('선택실험 속성값 vs 잠재점수 스케일 비교')
print('=' * 100)
print()

# 1. 데이터 로드
df_data = pd.read_csv('../data/processed/iclv/integrated_data_cleaned.csv')
print(f'원본 데이터: {len(df_data)} 행')

# 요인점수 로드
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_선택모델_확장_후_*.csv'))
df_fs = pd.read_csv(files[-1])
print(f'요인점수 데이터: {len(df_fs)} 행')
print()

# 2. 선택실험 속성값 분석
print('1. 선택실험 속성값 (Choice Experiment Attributes)')
print('-' * 100)
print()

# 가격 (연속형)
price_values = pd.to_numeric(df_data['price'], errors='coerce').dropna()
print('가격 (price):')
print(f'  데이터 타입:  연속형 (원 단위)')
print(f'  범위:         [{price_values.min():.0f}, {price_values.max():.0f}] 원')
print(f'  평균:         {price_values.mean():.2f}')
print(f'  표준편차:     {price_values.std():.4f}')
print(f'  분산:         {price_values.var():.4f}')
print(f'  값 분포:      {price_values.value_counts().sort_index().to_dict()}')
print()

# 무설탕 (이진형)
sugar_values = pd.to_numeric(df_data['sugar_content'], errors='coerce').dropna()
print('무설탕 (sugar_content):')
print(f'  데이터 타입:  이진형 (0/1)')
print(f'  범위:         [{sugar_values.min():.0f}, {sugar_values.max():.0f}]')
print(f'  평균:         {sugar_values.mean():.4f}')
print(f'  표준편차:     {sugar_values.std():.4f}')
print(f'  분산:         {sugar_values.var():.4f}')
print(f'  값 분포:      {sugar_values.value_counts().sort_index().to_dict()}')
print()

# 건강라벨 (이진형)
health_values = pd.to_numeric(df_data['health_label'], errors='coerce').dropna()
print('건강라벨 (health_label):')
print(f'  데이터 타입:  이진형 (0/1)')
print(f'  범위:         [{health_values.min():.0f}, {health_values.max():.0f}]')
print(f'  평균:         {health_values.mean():.4f}')
print(f'  표준편차:     {health_values.std():.4f}')
print(f'  분산:         {health_values.var():.4f}')
print(f'  값 분포:      {health_values.value_counts().sort_index().to_dict()}')
print()

# 3. 잠재점수 분석
print('2. 잠재점수 (Latent Variable Factor Scores)')
print('-' * 100)
print()

for col in df_fs.columns:
    values = df_fs[col].values
    print(f'{col}:')
    print(f'  데이터 타입:  연속형 (표준화된 점수)')
    print(f'  범위:         [{values.min():.4f}, {values.max():.4f}]')
    print(f'  평균:         {values.mean():.6f}')
    print(f'  표준편차:     {values.std():.4f}')
    print(f'  분산:         {values.var():.4f}')
    print()

# 4. 스케일 비교 요약
print('3. 스케일 비교 요약')
print('-' * 100)
print()

print('표준편차 비교:')
print()
print(f'{"변수":40s} {"표준편차":>12s} {"상대 크기":>12s}')
print('-' * 70)

# 기준: 가격
price_std = price_values.std()
print(f'{"price (기준)":40s} {price_std:>12.4f} {"1.00x":>12s}')

# 이진 변수들
sugar_std = sugar_values.std()
health_std = health_values.std()
print(f'{"sugar_content":40s} {sugar_std:>12.4f} {f"{sugar_std/price_std:.2f}x":>12s}')
print(f'{"health_label":40s} {health_std:>12.4f} {f"{health_std/price_std:.2f}x":>12s}')

print()

# 잠재점수들
for col in df_fs.columns:
    lv_std = df_fs[col].std()
    print(f'{col:40s} {lv_std:>12.4f} {f"{lv_std/price_std:.2f}x":>12s}')

print()

# 5. 핵심 발견
print('4. 핵심 발견')
print('-' * 100)
print()

print('발견 1: 가격 변수의 스케일이 작음')
print(f'  - price 표준편차: {price_std:.4f}')
print(f'  - 이유: 가격이 2원 또는 3원으로만 구성됨 (매우 좁은 범위)')
print(f'  - 실제 가격 범위: {price_values.min():.0f}~{price_values.max():.0f}원')
print()

print('발견 2: 잠재점수의 스케일이 상대적으로 큼')
print(f'  - 잠재점수 표준편차: 0.6~1.0')
print(f'  - 가격 대비 비율: {df_fs["purchase_intention"].std()/price_std:.1f}배')
print()

print('발견 3: 이진 변수는 중간 스케일')
print(f'  - 이진 변수 표준편차: ~0.5')
print(f'  - 가격 대비 비율: {sugar_std/price_std:.1f}배')
print()

# 6. 효용함수에서의 영향
print('5. 효용함수에서의 영향')
print('-' * 100)
print()

df_boot = pd.read_csv('../results/sequential_bootstrap_ci_1000.csv')
choice_params = df_boot[df_boot['Model'] == 'Choice'].copy()

beta_price = choice_params[choice_params['Parameter'] == 'β_price']['Mean'].values[0]
beta_sugar = choice_params[choice_params['Parameter'] == 'β_sugar_free']['Mean'].values[0]
beta_health = choice_params[choice_params['Parameter'] == 'β_health_label']['Mean'].values[0]
lambda_main = choice_params[choice_params['Parameter'] == 'lambda_main']['Mean'].values[0]

pi_std = df_fs['purchase_intention'].std()

print('계수 크기:')
print(f'  β_price:   {beta_price:>10.4f}')
print(f'  β_sugar:   {beta_sugar:>10.4f}')
print(f'  β_health:  {beta_health:>10.4f}')
print(f'  λ_main:    {lambda_main:>10.4f}')
print()

print('효과 크기 (계수 × 표준편차):')
print(f'  price:              {beta_price:>8.4f} × {price_std:>6.4f} = {beta_price * price_std:>10.4f}')
print(f'  sugar_content:      {beta_sugar:>8.4f} × {sugar_std:>6.4f} = {beta_sugar * sugar_std:>10.4f}')
print(f'  health_label:       {beta_health:>8.4f} × {health_std:>6.4f} = {beta_health * health_std:>10.4f}')
print(f'  purchase_intention: {lambda_main:>8.4f} × {pi_std:>6.4f} = {lambda_main * pi_std:>10.4f}')
print()

print('계수 비율 (절대값):')
print(f'  |β_price| / |λ_main| = {abs(beta_price/lambda_main):.1f}배')
print()

print('스케일 비율:')
print(f'  σ_purchase_intention / σ_price = {pi_std/price_std:.1f}배')
print()

print('⚠️ 문제:')
print(f'  - 가격의 스케일이 매우 작음 (σ={price_std:.4f})')
print(f'  - 잠재점수의 스케일이 상대적으로 큼 (σ={pi_std:.4f})')
print(f'  - 스케일 차이: {pi_std/price_std:.1f}배')
print(f'  - 이로 인해 잠재변수 계수가 매우 작게 추정됨')
print()

print('=' * 100)

