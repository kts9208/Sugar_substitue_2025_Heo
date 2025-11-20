"""
원본 Excel 파일에서 257, 273번 응답자 확인
"""
import pandas as pd
from pathlib import Path

# 원본 데이터 로드
raw_path = Path('data/raw/Sugar_substitue_Raw data_251108.xlsx')
df = pd.read_excel(raw_path, sheet_name='DATA')

print("="*80)
print("원본 Excel 데이터 확인")
print("="*80)

print(f"\n전체 데이터 shape: {df.shape}")
print(f"전체 응답자 수: {len(df)}")
print(f"'no' 컬럼 범위: {df['no'].min()} ~ {df['no'].max()}")

# 'no' 컬럼의 중복 확인
no_counts = df['no'].value_counts()
duplicates = no_counts[no_counts > 1]

if len(duplicates) > 0:
    print(f"\n⚠️ 중복된 'no' 값 발견: {len(duplicates)}개")
    print(f"중복 목록:")
    for no_val, count in duplicates.items():
        print(f"  no={no_val}: {count}번 등장")
else:
    print(f"\n✓ 'no' 컬럼에 중복 없음 (모두 고유)")

# 257, 273 확인
print(f"\n{'='*80}")
print(f"응답자 257, 273 상세 확인")
print(f"{'='*80}")

for rid in [257, 273]:
    subset = df[df['no'] == rid]
    
    print(f"\n[respondent_id {rid}]")
    print(f"  등장 횟수: {len(subset)}번")
    
    if len(subset) == 0:
        print(f"  ⚠️ 데이터 없음!")
    elif len(subset) == 1:
        print(f"  ✓ 정상 (1번만 등장)")
        
        # DCE 응답 확인
        row = subset.iloc[0]
        print(f"\n  DCE 응답 (q21-q26):")
        for q in ['q21', 'q22', 'q23', 'q24', 'q25', 'q26']:
            if q in row.index:
                print(f"    {q}: {row[q]}")
    else:
        print(f"  ⚠️ 중복! ({len(subset)}번 등장)")
        
        # 각 행의 DCE 응답 비교
        for i, (idx, row) in enumerate(subset.iterrows(), 1):
            print(f"\n  [등장 #{i}] (행 인덱스: {idx})")
            print(f"    DCE 응답 (q21-q26):")
            for q in ['q21', 'q22', 'q23', 'q24', 'q25', 'q26']:
                if q in row.index:
                    print(f"      {q}: {row[q]}")

# 전체 'no' 값 분포 확인
print(f"\n{'='*80}")
print(f"전체 'no' 값 분포")
print(f"{'='*80}")

print(f"\n고유 'no' 값 개수: {df['no'].nunique()}")
print(f"전체 행 수: {len(df)}")

if df['no'].nunique() != len(df):
    print(f"\n⚠️ 중복 존재! (고유값 {df['no'].nunique()} < 전체 행 {len(df)})")
else:
    print(f"\n✓ 중복 없음 (고유값 = 전체 행)")

# 'no' 값의 연속성 확인
no_values = sorted(df['no'].unique())
print(f"\n'no' 값 범위: {min(no_values)} ~ {max(no_values)}")
print(f"예상 개수 (연속): {max(no_values) - min(no_values) + 1}")
print(f"실제 고유 개수: {len(no_values)}")

# 빠진 번호 확인
expected = set(range(min(no_values), max(no_values) + 1))
actual = set(no_values)
missing = expected - actual

if missing:
    print(f"\n빠진 번호: {sorted(list(missing))[:20]}")  # 처음 20개만
    if len(missing) > 20:
        print(f"  ... (총 {len(missing)}개)")
else:
    print(f"\n✓ 빠진 번호 없음 (연속)")

