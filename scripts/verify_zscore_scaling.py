"""
Z-score 스케일링 검증

1. 데이터 로드 및 확인
2. 선택 속성 스케일 검증
3. 요인점수 스케일 검증
4. 원본 값과 표준화 값 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 100)
print('Z-score 스케일링 검증')
print('=' * 100)
print()

# 1. 데이터 로드
print('1. 수정된 데이터 로드')
print('-' * 100)
df = pd.read_csv('../data/processed/iclv/integrated_data_cleaned.csv')
print(f'데이터: {len(df)} 행')
print()

# 2. 선택 속성 검증
print('2. 선택 속성 Z-score 검증')
print('-' * 100)
print()

# 원본 값과 표준화 값 비교
attrs = ['price', 'sugar_free', 'health_label']

print('원본 값 (original):')
print(f'{"변수":20s} {"평균":>12s} {"표준편차":>12s} {"범위":>25s}')
print('-' * 85)
for attr in attrs:
    col_orig = f'{attr}_original'
    if col_orig in df.columns:
        values = df[col_orig].dropna()
        print(f'{attr:20s} {values.mean():>12.4f} {values.std():>12.4f} [{values.min():>10.2f}, {values.max():>10.2f}]')
print()

print('표준화 값 (standardized):')
print(f'{"변수":20s} {"평균":>12s} {"표준편차":>12s} {"범위":>25s}')
print('-' * 85)
for attr in attrs:
    values = df[attr].dropna()
    print(f'{attr:20s} {values.mean():>12.6f} {values.std():>12.6f} [{values.min():>10.4f}, {values.max():>10.4f}]')
print()

# 수동 검증: z = (x - mean) / std
print('수동 검증 (첫 10개 값):')
print(f'{"변수":15s} {"원본":>10s} {"표준화":>10s} {"수동계산":>10s} {"일치":>8s}')
print('-' * 60)

for attr in attrs:
    col_orig = f'{attr}_original'
    if col_orig in df.columns:
        # NaN이 아닌 첫 10개
        mask = df[col_orig].notna()
        orig_vals = df.loc[mask, col_orig].head(10).values
        std_vals = df.loc[mask, attr].head(10).values
        
        # 수동 계산
        mean_orig = df[col_orig].dropna().mean()
        std_orig = df[col_orig].dropna().std()
        manual_vals = (orig_vals - mean_orig) / std_orig
        
        # 비교
        for i in range(min(3, len(orig_vals))):  # 처음 3개만 출력
            match = '✓' if np.abs(std_vals[i] - manual_vals[i]) < 1e-6 else '✗'
            print(f'{attr:15s} {orig_vals[i]:>10.2f} {std_vals[i]:>10.4f} {manual_vals[i]:>10.4f} {match:>8s}')
print()

# 3. 요인점수 검증
print('3. 요인점수 Z-score 검증')
print('-' * 100)
print()

log_dir = Path('logs/factor_scores')

# 원본 요인점수
files_orig = sorted(log_dir.glob('factor_scores_선택모델_확장_후_*.csv'))
if files_orig:
    df_fs_orig = pd.read_csv(files_orig[-1])
    
    print('원본 요인점수:')
    print(f'{"변수":30s} {"평균":>12s} {"표준편차":>12s}')
    print('-' * 60)
    for col in df_fs_orig.columns:
        values = df_fs_orig[col].values
        print(f'{col:30s} {values.mean():>12.6f} {values.std():>12.6f}')
    print()

# 표준화된 요인점수
file_std = log_dir / 'factor_scores_standardized.csv'
if file_std.exists():
    df_fs_std = pd.read_csv(file_std)
    
    print('표준화된 요인점수:')
    print(f'{"변수":30s} {"평균":>12s} {"표준편차":>12s}')
    print('-' * 60)
    for col in df_fs_std.columns:
        values = df_fs_std[col].values
        print(f'{col:30s} {values.mean():>12.6f} {values.std():>12.6f}')
    print()
    
    # 수동 검증
    print('수동 검증 (첫 3개 값):')
    print(f'{"변수":25s} {"원본":>10s} {"표준화":>10s} {"수동계산":>10s} {"일치":>8s}')
    print('-' * 70)
    
    for col in df_fs_orig.columns:
        orig_vals = df_fs_orig[col].head(3).values
        std_vals = df_fs_std[col].head(3).values
        
        # 수동 계산
        mean_orig = df_fs_orig[col].mean()
        std_orig = df_fs_orig[col].std()
        manual_vals = (orig_vals - mean_orig) / std_orig
        
        # 비교
        for i in range(3):
            match = '✓' if np.abs(std_vals[i] - manual_vals[i]) < 1e-6 else '✗'
            print(f'{col:25s} {orig_vals[i]:>10.4f} {std_vals[i]:>10.4f} {manual_vals[i]:>10.4f} {match:>8s}')
    print()

# 4. 최종 검증
print('4. 최종 검증 결과')
print('-' * 100)
print()

all_pass = True

# 선택 속성 검증
print('선택 속성:')
for attr in attrs:
    values = df[attr].dropna()
    mean_ok = np.abs(values.mean()) < 1e-10
    std_ok = np.abs(values.std() - 1.0) < 0.01
    
    status = '✓' if (mean_ok and std_ok) else '✗'
    if not (mean_ok and std_ok):
        all_pass = False
    
    print(f'  {attr:20s} 평균≈0: {mean_ok} ({"✓" if mean_ok else "✗"}), 표준편차≈1: {std_ok} ({"✓" if std_ok else "✗"}) {status}')
print()

# 요인점수 검증
if file_std.exists():
    print('요인점수:')
    for col in df_fs_std.columns:
        values = df_fs_std[col].values
        mean_ok = np.abs(values.mean()) < 1e-10
        std_ok = np.abs(values.std() - 1.0) < 1e-10
        
        status = '✓' if (mean_ok and std_ok) else '✗'
        if not (mean_ok and std_ok):
            all_pass = False
        
        print(f'  {col:30s} 평균≈0: {mean_ok} ({"✓" if mean_ok else "✗"}), 표준편차≈1: {std_ok} ({"✓" if std_ok else "✗"}) {status}')
    print()

if all_pass:
    print('=' * 100)
    print('✅ Z-score 스케일링이 올바르게 구현되었습니다!')
    print('=' * 100)
else:
    print('=' * 100)
    print('⚠️ 일부 변수의 스케일링에 문제가 있습니다.')
    print('=' * 100)

print()

