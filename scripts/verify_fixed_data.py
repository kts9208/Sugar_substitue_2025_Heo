"""
수정된 데이터 최종 검증
"""
import pandas as pd
from pathlib import Path

print("="*80)
print("수정된 데이터 최종 검증")
print("="*80)

# 1. 원본 Excel 확인
print(f"\n[1] 원본 Excel 파일")
raw_path = Path('data/raw/Sugar_substitue_Raw data_251108.xlsx')
df_raw = pd.read_excel(raw_path, sheet_name='DATA')

print(f"  Shape: {df_raw.shape}")
print(f"  고유 'no' 개수: {df_raw['no'].nunique()}")
print(f"  전체 행 수: {len(df_raw)}")

no_counts = df_raw['no'].value_counts()
duplicates = no_counts[no_counts > 1]

if len(duplicates) > 0:
    print(f"  ✗ 중복 존재: {len(duplicates)}개")
    print(f"  중복 ID: {list(duplicates.index)}")
else:
    print(f"  ✓ 중복 없음 (모든 ID 고유)")

# 257, 273, 2, 4 확인
print(f"\n  특정 ID 확인:")
for rid in [2, 4, 257, 273]:
    count = (df_raw['no'] == rid).sum()
    print(f"    ID {rid}: {count}행")

# 2. DCE long format 확인
print(f"\n{'='*80}")
print(f"[2] DCE Long Format")
dce_path = Path('data/processed/dce/dce_long_format.csv')
dce = pd.read_csv(dce_path)

print(f"  Shape: {dce.shape}")
print(f"  고유 respondent_id 개수: {dce['respondent_id'].nunique()}")
print(f"  전체 행 수: {len(dce)}")

id_counts = dce.groupby('respondent_id').size()
print(f"\n  respondent_id별 행 수 분포:")
print(f"    {id_counts.value_counts().sort_index().to_dict()}")

# 중복 확인
dup_ids = id_counts[id_counts != 18].index.tolist()
if dup_ids:
    print(f"\n  ✗ 비정상 ID (행 수 ≠ 18): {len(dup_ids)}개")
    print(f"  비정상 ID: {dup_ids[:10]}")
else:
    print(f"\n  ✓ 모든 ID가 정확히 18행")

# 특정 ID 확인
print(f"\n  특정 ID 확인:")
for rid in [2, 4, 257, 273]:
    count = (dce['respondent_id'] == rid).sum()
    print(f"    ID {rid}: {count}행")

# 3. Integrated data 확인
print(f"\n{'='*80}")
print(f"[3] Integrated Data")
integrated_path = Path('data/processed/iclv/integrated_data.csv')
integrated = pd.read_csv(integrated_path)

print(f"  Shape: {integrated.shape}")
print(f"  고유 respondent_id 개수: {integrated['respondent_id'].nunique()}")
print(f"  전체 행 수: {len(integrated)}")

id_counts = integrated.groupby('respondent_id').size()
print(f"\n  respondent_id별 행 수 분포:")
print(f"    {id_counts.value_counts().sort_index().to_dict()}")

# 중복 확인
dup_ids = id_counts[id_counts != 18].index.tolist()
if dup_ids:
    print(f"\n  ✗ 비정상 ID (행 수 ≠ 18): {len(dup_ids)}개")
    print(f"  비정상 ID: {dup_ids[:10]}")
else:
    print(f"\n  ✓ 모든 ID가 정확히 18행")

# 특정 ID 확인
print(f"\n  특정 ID 확인:")
for rid in [2, 4, 257, 273]:
    count = (integrated['respondent_id'] == rid).sum()
    print(f"    ID {rid}: {count}행")

# 4. 최종 요약
print(f"\n{'='*80}")
print(f"최종 요약")
print(f"{'='*80}")

all_good = True

# 원본 Excel 검증
if df_raw['no'].nunique() == len(df_raw):
    print(f"✓ 원본 Excel: 모든 ID 고유 ({len(df_raw)}명)")
else:
    print(f"✗ 원본 Excel: 중복 존재")
    all_good = False

# DCE 검증
if (id_counts == 18).all() and len(dce) == dce['respondent_id'].nunique() * 18:
    print(f"✓ DCE Long Format: 모든 ID가 18행 ({dce['respondent_id'].nunique()}명 × 18 = {len(dce)}행)")
else:
    print(f"✗ DCE Long Format: 비정상")
    all_good = False

# Integrated 검증
id_counts_int = integrated.groupby('respondent_id').size()
if (id_counts_int == 18).all() and len(integrated) == integrated['respondent_id'].nunique() * 18:
    print(f"✓ Integrated Data: 모든 ID가 18행 ({integrated['respondent_id'].nunique()}명 × 18 = {len(integrated)}행)")
else:
    print(f"✗ Integrated Data: 비정상")
    all_good = False

print(f"\n{'='*80}")
if all_good:
    print(f"🎉 모든 검증 통과! 데이터가 정상적으로 수정되었습니다.")
else:
    print(f"⚠️ 일부 검증 실패. 위 내용을 확인하세요.")
print(f"{'='*80}")

