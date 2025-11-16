"""
Z-score 표준화 분석

1. sugar_content 더미 변수 변환 문제 확인
2. Z-score 표준화 효과 분석
3. 스케일 문제 해결 방안 제시
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 100)
print('Z-score 표준화 분석')
print('=' * 100)
print()

# 1. 데이터 로드
df = pd.read_csv('../data/processed/iclv/integrated_data_cleaned.csv')
print(f'데이터: {len(df)} 행')
print()

# 2. sugar_content 문제 확인
print('1. sugar_content 더미 변수 변환 문제')
print('-' * 100)
print()

print('현재 상태:')
print(f'  sugar_content 타입: {df["sugar_content"].dtype}')
print(f'  고유값: {df["sugar_content"].unique()}')
print(f'  결측값: {df["sugar_content"].isna().sum()} / {len(df)} ({df["sugar_content"].isna().sum()/len(df)*100:.1f}%)')
print()

print('⚠️ 문제:')
print('  - sugar_content가 문자열 ("무설탕", "알반당")로 저장됨')
print('  - 모델은 "sugar_free" 더미 변수 (0/1)를 기대')
print('  - 문자열을 숫자로 변환하지 않아서 NaN으로 처리됨')
print()

# 더미 변수 생성
df['sugar_free'] = (df['sugar_content'] == '무설탕').astype(float)
df.loc[df['sugar_content'].isna(), 'sugar_free'] = np.nan

print('✅ 해결책: 더미 변수 생성')
print(f'  sugar_free = 1 if sugar_content == "무설탕" else 0')
print(f'  값 분포: {df["sugar_free"].value_counts(dropna=False).to_dict()}')
print()

# 3. 현재 변수 스케일
print('2. 현재 변수 스케일 (더미 변수 포함)')
print('-' * 100)
print()

# 숫자형 변환
df['price_num'] = pd.to_numeric(df['price'], errors='coerce')
df['health_label_num'] = pd.to_numeric(df['health_label'], errors='coerce')

vars_dict = {
    'price': df['price_num'].dropna(),
    'sugar_free': df['sugar_free'].dropna(),
    'health_label': df['health_label_num'].dropna()
}

print(f'{"변수":20s} {"평균":>12s} {"표준편차":>12s} {"범위":>20s}')
print('-' * 70)
for name, values in vars_dict.items():
    print(f'{name:20s} {values.mean():>12.4f} {values.std():>12.4f} [{values.min():>8.2f}, {values.max():>8.2f}]')
print()

# 4. 요인점수 스케일
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_선택모델_확장_후_*.csv'))
if files:
    df_fs = pd.read_csv(files[-1])
    
    print('3. 요인점수 스케일')
    print('-' * 100)
    print()
    
    print(f'{"변수":30s} {"평균":>12s} {"표준편차":>12s} {"범위":>25s}')
    print('-' * 85)
    for col in df_fs.columns:
        values = df_fs[col].values
        print(f'{col:30s} {values.mean():>12.6f} {values.std():>12.4f} [{values.min():>10.4f}, {values.max():>10.4f}]')
    print()

# 5. Z-score 표준화 효과
print('4. Z-score 표준화 효과')
print('-' * 100)
print()

print('Z-score 표준화 공식: z = (x - mean(x)) / std(x)')
print()

# 표준화 전
print('표준화 전:')
print(f'{"변수":20s} {"평균":>12s} {"표준편차":>12s}')
print('-' * 50)
for name, values in vars_dict.items():
    print(f'{name:20s} {values.mean():>12.4f} {values.std():>12.4f}')
print()

# 표준화 후
print('표준화 후:')
print(f'{"변수":20s} {"평균":>12s} {"표준편차":>12s}')
print('-' * 50)
for name, values in vars_dict.items():
    z_values = (values - values.mean()) / values.std()
    print(f'{name:20s} {z_values.mean():>12.6f} {z_values.std():>12.4f}')
print()

print('✅ 효과:')
print('  - 모든 변수가 평균 0, 표준편차 1로 통일됨')
print('  - 스케일 차이 완전히 제거')
print('  - 가격을 1000으로 나눈 것은 표준화 후 의미 없음')
print()

# 6. 요인점수도 표준화 필요
print('5. 요인점수 표준화 필요성')
print('-' * 100)
print()

if files:
    print('현재 요인점수:')
    print(f'{"변수":30s} {"평균":>12s} {"표준편차":>12s}')
    print('-' * 60)
    for col in df_fs.columns:
        values = df_fs[col].values
        print(f'{col:30s} {values.mean():>12.6f} {values.std():>12.4f}')
    print()
    
    print('표준화 후 요인점수:')
    print(f'{"변수":30s} {"평균":>12s} {"표준편차":>12s}')
    print('-' * 60)
    for col in df_fs.columns:
        values = df_fs[col].values
        z_values = (values - values.mean()) / values.std()
        print(f'{col:30s} {z_values.mean():>12.6f} {z_values.std():>12.4f}')
    print()
    
    print('⚠️ 주의:')
    print('  - 요인점수는 이미 평균 0이지만 분산이 1이 아님')
    print('  - 표준화하면 모든 요인점수가 분산 1로 통일됨')
    print()

# 7. 계수 해석 변화
print('6. Z-score 표준화 후 계수 해석')
print('-' * 100)
print()

print('표준화 전:')
print('  β_price = -1.616')
print('  해석: 가격이 1 단위 증가하면 효용이 -1.616 감소')
print('  (현재 가격 단위가 애매함: 2, 2.5, 3)')
print()

print('표준화 후:')
print('  β_price_std = ?')
print('  해석: 가격이 1 표준편차 증가하면 효용이 β_price_std 감소')
print('  (모든 변수가 동일한 스케일)')
print()

print('✅ 장점:')
print('  1. 계수 크기로 상대적 중요도 직접 비교 가능')
print('  2. 수치적 안정성 향상')
print('  3. 최적화 알고리즘 수렴 개선')
print()

print('❌ 단점:')
print('  1. 계수 해석이 덜 직관적')
print('  2. 원래 단위로 역변환 필요')
print()

print('=' * 100)

