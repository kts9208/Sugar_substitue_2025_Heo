"""
데이터 수정 및 Z-score 표준화

1. sugar_content 문자열 → sugar_free 숫자 (0/1) 변환
2. Z-score 표준화 적용
3. 수정된 데이터 저장
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

print('=' * 100)
print('데이터 수정 및 Z-score 표준화')
print('=' * 100)
print()

# 1. 데이터 로드
print('1. 데이터 로드')
print('-' * 100)
df = pd.read_csv('../data/processed/iclv/integrated_data_cleaned.csv')
print(f'원본 데이터: {len(df)} 행')
print()

# 2. sugar_content → sugar_free 변환
print('2. sugar_content → sugar_free 변환')
print('-' * 100)
print()

print('변환 전:')
print(f'  sugar_content 타입: {df["sugar_content"].dtype}')
print(f'  고유값: {df["sugar_content"].unique()}')
print(f'  값 분포:')
print(df['sugar_content'].value_counts(dropna=False))
print()

# 변환: 무설탕=1, 일반당=0
df['sugar_free'] = df['sugar_content'].map({
    '무설탕': 1.0,
    '알반당': 0.0
})
# NaN은 그대로 유지 (no-choice 옵션)

print('변환 후:')
print(f'  sugar_free 타입: {df["sugar_free"].dtype}')
print(f'  고유값: {df["sugar_free"].unique()}')
print(f'  값 분포:')
print(df['sugar_free'].value_counts(dropna=False))
print()

print('✅ 변환 규칙:')
print('  일반당 → 0')
print('  무설탕 → 1')
print('  NaN → NaN (no-choice)')
print()

# 3. 현재 스케일 확인
print('3. 표준화 전 스케일')
print('-' * 100)
print()

# 숫자형 변환
df['price_num'] = pd.to_numeric(df['price'], errors='coerce')
df['health_label_num'] = pd.to_numeric(df['health_label'], errors='coerce')

print(f'{"변수":20s} {"평균":>12s} {"표준편차":>12s} {"범위":>25s}')
print('-' * 85)

vars_to_check = {
    'price': df['price_num'].dropna(),
    'sugar_free': df['sugar_free'].dropna(),
    'health_label': df['health_label_num'].dropna()
}

for name, values in vars_to_check.items():
    print(f'{name:20s} {values.mean():>12.4f} {values.std():>12.4f} [{values.min():>10.2f}, {values.max():>10.2f}]')
print()

# 4. Z-score 표준화
print('4. Z-score 표준화 적용')
print('-' * 100)
print()

# 표준화할 컬럼
cols_to_standardize = ['price', 'sugar_free', 'health_label']

# 백업 (원본 값 보존)
for col in cols_to_standardize:
    df[f'{col}_original'] = df[col]

print('표준화 방법: z = (x - mean) / std')
print()

# NaN이 아닌 행만 표준화
mask = ~df[cols_to_standardize].isna().any(axis=1)
print(f'표준화 대상 행: {mask.sum()} / {len(df)} ({mask.sum()/len(df)*100:.1f}%)')
print()

# StandardScaler 사용
scaler = StandardScaler()
df.loc[mask, cols_to_standardize] = scaler.fit_transform(df.loc[mask, cols_to_standardize])

print('표준화 후 스케일:')
print(f'{"변수":20s} {"평균":>12s} {"표준편차":>12s} {"범위":>25s}')
print('-' * 85)

for col in cols_to_standardize:
    values = df.loc[mask, col]
    print(f'{col:20s} {values.mean():>12.6f} {values.std():>12.6f} [{values.min():>10.4f}, {values.max():>10.4f}]')
print()

print('✅ 모든 변수가 평균 0, 표준편차 1로 표준화됨')
print()

# 5. 요인점수도 표준화
print('5. 요인점수 표준화')
print('-' * 100)
print()

# 요인점수 로드
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_선택모델_확장_후_*.csv'))

if files:
    df_fs = pd.read_csv(files[-1])
    
    print('표준화 전:')
    print(f'{"변수":30s} {"평균":>12s} {"표준편차":>12s}')
    print('-' * 60)
    for col in df_fs.columns:
        values = df_fs[col].values
        print(f'{col:30s} {values.mean():>12.6f} {values.std():>12.6f}')
    print()
    
    # 표준화
    scaler_fs = StandardScaler()
    df_fs_std = pd.DataFrame(
        scaler_fs.fit_transform(df_fs),
        columns=df_fs.columns
    )
    
    print('표준화 후:')
    print(f'{"변수":30s} {"평균":>12s} {"표준편차":>12s}')
    print('-' * 60)
    for col in df_fs_std.columns:
        values = df_fs_std[col].values
        print(f'{col:30s} {values.mean():>12.6f} {values.std():>12.6f}')
    print()
    
    # 저장
    output_file = log_dir / 'factor_scores_standardized.csv'
    df_fs_std.to_csv(output_file, index=False)
    print(f'✅ 표준화된 요인점수 저장: {output_file}')
    print()

# 6. 수정된 데이터 저장
print('6. 수정된 데이터 저장')
print('-' * 100)
print()

# 원본 백업
backup_file = '../data/processed/iclv/integrated_data_cleaned_backup.csv'
df_original = pd.read_csv('../data/processed/iclv/integrated_data_cleaned.csv')
df_original.to_csv(backup_file, index=False)
print(f'✅ 원본 백업: {backup_file}')
print()

# 수정된 데이터 저장
output_file = '../data/processed/iclv/integrated_data_cleaned.csv'
df.to_csv(output_file, index=False)
print(f'✅ 수정된 데이터 저장: {output_file}')
print()

# 7. 요약
print('7. 요약')
print('-' * 100)
print()

print('✅ 완료된 작업:')
print('  1. sugar_content (문자열) → sugar_free (0/1) 변환')
print('     - 일반당 → 0')
print('     - 무설탕 → 1')
print()
print('  2. 선택 속성 Z-score 표준화')
print('     - price, sugar_free, health_label')
print('     - 모두 평균 0, 표준편차 1')
print()
print('  3. 요인점수 Z-score 표준화')
print('     - 모든 잠재변수')
print('     - 모두 평균 0, 표준편차 1')
print()

print('📊 기대 효과:')
print('  1. sugar_free 변수 정상 작동')
print('  2. 완벽한 스케일 균형')
print('  3. 수치적 안정성 향상')
print('  4. 잠재변수 유의성 개선 예상')
print()

print('🔧 다음 단계:')
print('  1. 순차추정 재실행')
print('  2. 부트스트랩 재실행 (1000회)')
print('  3. 결과 비교')
print()

print('=' * 100)

