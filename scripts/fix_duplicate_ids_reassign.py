"""
중복 ID 재할당

respondent_id 257, 273이 각각 2번 등장하는데, 실제로는 다른 사람임
→ 두 번째 등장하는 사람에게 새로운 ID 부여

전략:
- 현재 사용되지 않는 ID 중에서 할당
- 빠진 번호 중 가장 작은 값 사용 또는 최대값+1 사용
"""
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# 원본 파일 경로
raw_path = Path('data/raw/Sugar_substitue_Raw data_251108.xlsx')

print("="*80)
print("중복 ID 재할당")
print("="*80)

# 백업
backup_path = raw_path.parent / f'Sugar_substitue_Raw data_251108_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
shutil.copy(raw_path, backup_path)
print(f"\n✓ 원본 백업 완료: {backup_path.name}")

# 모든 시트 로드
xl_file = pd.ExcelFile(raw_path)
print(f"\n시트 목록: {xl_file.sheet_names}")

# DATA 시트 로드
df_data = pd.read_excel(raw_path, sheet_name='DATA')
df_label = pd.read_excel(raw_path, sheet_name='LABEL')
df_code = pd.read_excel(raw_path, sheet_name='CODE')

print(f"\n[원본 데이터]")
print(f"  DATA shape: {df_data.shape}")
print(f"  고유 'no' 개수: {df_data['no'].nunique()}")
print(f"  전체 행 수: {len(df_data)}")

# 사용 가능한 ID 찾기
used_ids = set(df_data['no'].unique())
max_id = df_data['no'].max()

print(f"\n[사용 가능한 ID 찾기]")
print(f"  현재 최대 ID: {max_id}")

# 빠진 번호 찾기
all_possible = set(range(1, max_id + 1))
missing_ids = sorted(all_possible - used_ids)

print(f"  빠진 번호 개수: {len(missing_ids)}")
if len(missing_ids) > 0:
    print(f"  빠진 번호 (처음 10개): {missing_ids[:10]}")

# 새 ID 할당 전략: 빠진 번호 중 가장 작은 2개 사용
if len(missing_ids) >= 2:
    new_id_257 = missing_ids[0]
    new_id_273 = missing_ids[1]
    print(f"\n[새 ID 할당 (빠진 번호 사용)]")
else:
    # 빠진 번호가 부족하면 최대값+1, +2 사용
    new_id_257 = max_id + 1
    new_id_273 = max_id + 2
    print(f"\n[새 ID 할당 (최대값+1, +2 사용)]")

print(f"  257번의 두 번째 등장 → {new_id_257}")
print(f"  273번의 두 번째 등장 → {new_id_273}")

# ID 재할당
print(f"\n[ID 재할당 중...]")

# 중복된 ID의 인덱스 찾기
duplicate_info = []

for rid in [257, 273]:
    indices = df_data[df_data['no'] == rid].index.tolist()
    if len(indices) == 2:
        duplicate_info.append({
            'old_id': rid,
            'first_idx': indices[0],
            'second_idx': indices[1],
            'new_id': new_id_257 if rid == 257 else new_id_273
        })
        print(f"\n  ID {rid}:")
        print(f"    첫 번째 등장 (행 {indices[0]}): ID 유지 ({rid})")
        print(f"    두 번째 등장 (행 {indices[1]}): ID 변경 ({rid} → {new_id_257 if rid == 257 else new_id_273})")

# DATA 시트 수정
df_data_fixed = df_data.copy()
for info in duplicate_info:
    df_data_fixed.loc[info['second_idx'], 'no'] = info['new_id']

# LABEL 시트도 동일하게 수정
df_label_fixed = df_label.copy()
for info in duplicate_info:
    df_label_fixed.loc[info['second_idx'], 'no'] = info['new_id']

# 검증
print(f"\n[수정 후 검증]")
print(f"  DATA shape: {df_data_fixed.shape}")
print(f"  고유 'no' 개수: {df_data_fixed['no'].nunique()}")
print(f"  전체 행 수: {len(df_data_fixed)}")

if df_data_fixed['no'].nunique() == len(df_data_fixed):
    print(f"\n  ✓ 검증 성공: 모든 'no' 값이 고유함")
else:
    print(f"\n  ✗ 검증 실패: 여전히 중복 존재")
    duplicates = df_data_fixed['no'].value_counts()
    duplicates = duplicates[duplicates > 1]
    print(f"  중복 ID: {list(duplicates.index)}")

# 257, 273, 새 ID 확인
print(f"\n  ID 확인:")
for rid in [257, 273, new_id_257, new_id_273]:
    count = (df_data_fixed['no'] == rid).sum()
    print(f"    ID {rid}: {count}행")

# 저장
print(f"\n{'='*80}")
print(f"정리된 데이터 저장")
print(f"{'='*80}")

with pd.ExcelWriter(raw_path, engine='openpyxl') as writer:
    df_data_fixed.to_excel(writer, sheet_name='DATA', index=False)
    df_label_fixed.to_excel(writer, sheet_name='LABEL', index=False)
    df_code.to_excel(writer, sheet_name='CODE', index=False)
    print(f"  ✓ DATA: {df_data_fixed.shape}")
    print(f"  ✓ LABEL: {df_label_fixed.shape}")
    print(f"  ✓ CODE: {df_code.shape}")

print(f"\n✓ 저장 완료: {raw_path}")
print(f"✓ 백업 파일: {backup_path}")

print(f"\n{'='*80}")
print(f"다음 단계")
print(f"{'='*80}")
print(f"1. 전처리 파이프라인 재실행:")
print(f"   python scripts/preprocess_dce_data.py")
print(f"   python scripts/integrate_iclv_data.py")
print(f"2. 또는 전체 파이프라인:")
print(f"   python scripts/run_preprocessing_pipeline.py")

