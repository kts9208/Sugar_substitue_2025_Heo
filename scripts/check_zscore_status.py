"""
Z-score 표준화 상태 확인

1. 선택 속성 (price, sugar_free, health_label)
2. 요인점수 (SEM 추출)
3. 추정 결과
"""

import pandas as pd
import numpy as np
import os

print('=' * 100)
print('현재 데이터 Z-score 표준화 상태 확인')
print('=' * 100)
print()

# 1. 선택 데이터 확인
print('1. 선택 데이터 (integrated_data_cleaned.csv)')
print('-' * 100)
df = pd.read_csv('../data/processed/iclv/integrated_data_cleaned.csv')

# 선택 속성 확인
attrs = ['price', 'sugar_free', 'health_label']
print(f"{'변수':20s} {'평균':>12s} {'표준편차':>12s} {'최소':>12s} {'최대':>12s}")
print('-' * 70)
for attr in attrs:
    values = df[attr].dropna()
    print(f'{attr:20s} {values.mean():>12.6f} {values.std():>12.6f} {values.min():>12.4f} {values.max():>12.4f}')
print()

# 원본 값 확인
if 'price_original' in df.columns:
    print('원본 값 (표준화 전):')
    print(f"{'변수':20s} {'평균':>12s} {'표준편차':>12s} {'최소':>12s} {'최대':>12s}")
    print('-' * 70)
    for attr in attrs:
        col = f'{attr}_original'
        if col in df.columns:
            values = df[col].dropna()
            print(f'{attr:20s} {values.mean():>12.6f} {values.std():>12.6f} {values.min():>12.4f} {values.max():>12.4f}')
    print()

# 2. 요인점수 확인
print('2. 요인점수 (SEM 추출)')
print('-' * 100)

# 표준화된 요인점수 파일 확인
if os.path.exists('logs/factor_scores/factor_scores_standardized.csv'):
    df_fs = pd.read_csv('logs/factor_scores/factor_scores_standardized.csv')
    print('✅ 표준화된 요인점수 파일 존재')
    print()
    print(f"{'변수':30s} {'평균':>12s} {'표준편차':>12s}")
    print('-' * 60)
    for col in df_fs.columns:
        values = df_fs[col].values
        print(f'{col:30s} {values.mean():>12.6f} {values.std():>12.6f}')
else:
    print('❌ 표준화된 요인점수 파일 없음')
print()

# 3. 최근 추정 결과 확인
print('3. 최근 추정 결과')
print('-' * 100)
if os.path.exists('../results/sequential_estimation_results.csv'):
    df_res = pd.read_csv('../results/sequential_estimation_results.csv')
    choice_res = df_res[df_res['Model'] == 'Choice']
    print(choice_res.to_string(index=False))
else:
    print('❌ 추정 결과 파일 없음')
print()

print('=' * 100)
print('핵심 질문:')
print('=' * 100)
print()
print('Q1. 선택 속성이 Z-score 표준화되어 있는가?')
print('    → price, sugar_free, health_label의 mean≈0, std≈1 확인')
print()
print('Q2. 요인점수가 Z-score 표준화되어 있는가?')
print('    → SEM에서 추출한 요인점수의 mean≈0, std≈1 확인')
print()
print('Q3. 추정된 파라미터가 표준화된 스케일인가?')
print('    → β는 "1 표준편차 변화"에 대한 효과')
print('    → 원래 스케일로 해석하려면 언스케일링 필요')
print()
print('Q4. 결과 보고 시 언스케일링이 필요한가?')
print('    → 연구 목적에 따라 다름:')
print('      - 표준화 계수: 변수 간 상대적 중요도 비교')
print('      - 원래 계수: 실제 단위로 해석 (예: 가격 1000원 증가 시 효과)')
print()

