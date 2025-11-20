"""
원본 Excel 파일의 중복 제거

Sugar_substitue_Raw data_251108.xlsx에서 중복된 응답자 제거
- respondent_id 257, 273의 첫 번째 응답만 유지
"""
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# 원본 파일 경로
raw_path = Path('data/raw/Sugar_substitue_Raw data_251108.xlsx')

print("="*80)
print("원본 Excel 파일 중복 제거")
print("="*80)

# 백업
backup_path = raw_path.parent / f'Sugar_substitue_Raw data_251108_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
shutil.copy(raw_path, backup_path)
print(f"\n✓ 원본 백업 완료: {backup_path.name}")

# 모든 시트 로드
xl_file = pd.ExcelFile(raw_path)
print(f"\n시트 목록: {xl_file.sheet_names}")

# 각 시트 처리
sheets_dict = {}

for sheet_name in xl_file.sheet_names:
    print(f"\n[시트: {sheet_name}]")
    df = pd.read_excel(raw_path, sheet_name=sheet_name)
    
    print(f"  원본 shape: {df.shape}")
    
    if 'no' in df.columns:
        # 중복 확인
        no_counts = df['no'].value_counts()
        duplicates = no_counts[no_counts > 1]
        
        if len(duplicates) > 0:
            print(f"  중복 발견: {len(duplicates)}개")
            print(f"  중복 ID: {list(duplicates.index)}")
            
            # 중복 제거: 각 'no' 값의 첫 번째 행만 유지
            df_cleaned = df.drop_duplicates(subset='no', keep='first')
            
            print(f"  정리 후 shape: {df_cleaned.shape}")
            print(f"  제거된 행 수: {len(df) - len(df_cleaned)}")
            
            # 257, 273 확인
            for rid in [257, 273]:
                original_count = (df['no'] == rid).sum()
                cleaned_count = (df_cleaned['no'] == rid).sum()
                print(f"    ID {rid}: {original_count}행 → {cleaned_count}행")
            
            sheets_dict[sheet_name] = df_cleaned
        else:
            print(f"  중복 없음")
            sheets_dict[sheet_name] = df
    else:
        print(f"  'no' 컬럼 없음 - 원본 유지")
        sheets_dict[sheet_name] = df

# 새 Excel 파일로 저장
print(f"\n{'='*80}")
print(f"정리된 데이터 저장")
print(f"{'='*80}")

with pd.ExcelWriter(raw_path, engine='openpyxl') as writer:
    for sheet_name, df in sheets_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  ✓ {sheet_name}: {df.shape}")

print(f"\n✓ 저장 완료: {raw_path}")
print(f"✓ 백업 파일: {backup_path}")

# 검증
print(f"\n{'='*80}")
print(f"검증")
print(f"{'='*80}")

df_verify = pd.read_excel(raw_path, sheet_name='DATA')
print(f"최종 DATA 시트 shape: {df_verify.shape}")
print(f"고유 'no' 개수: {df_verify['no'].nunique()}")
print(f"전체 행 수: {len(df_verify)}")

if df_verify['no'].nunique() == len(df_verify):
    print(f"\n✓ 검증 성공: 모든 'no' 값이 고유함")
else:
    print(f"\n✗ 검증 실패: 여전히 중복 존재")

# 257, 273 확인
for rid in [257, 273]:
    count = (df_verify['no'] == rid).sum()
    print(f"  ID {rid}: {count}행")

