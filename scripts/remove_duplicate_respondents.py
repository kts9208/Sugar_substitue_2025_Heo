"""
중복 데이터 제거 스크립트

Option 1: 첫 번째 응답만 유지 (추천)
- respondent_id 257, 273의 중복 데이터 제거
- 각 개인의 첫 18행만 유지
"""
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# 데이터 로드
data_path = Path('data/processed/iclv/integrated_data.csv')
data = pd.read_csv(data_path)

print("="*80)
print("중복 데이터 제거")
print("="*80)

# 원본 백업
backup_path = data_path.parent / f'integrated_data_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
shutil.copy(data_path, backup_path)
print(f"\n✓ 원본 백업 완료: {backup_path.name}")

# 중복 제거 전 통계
print(f"\n[제거 전]")
print(f"  전체 행 수: {len(data)}")
print(f"  전체 개인 수: {data['respondent_id'].nunique()}")

row_counts = data.groupby('respondent_id').size()
print(f"  개인별 행 수 분포:")
print(f"    {row_counts.value_counts().sort_index().to_dict()}")

# 중복된 개인 ID
dup_ids = [257, 273]

# 각 중복 개인에 대해 첫 18행만 유지
rows_to_drop = []

for rid in dup_ids:
    subset = data[data['respondent_id'] == rid]
    print(f"\n[respondent_id {rid}]")
    print(f"  현재 행 수: {len(subset)}")
    
    # 첫 18행의 인덱스
    first_18_indices = subset.index[:18]
    # 나머지 18행의 인덱스 (제거 대상)
    remaining_indices = subset.index[18:]
    
    print(f"  유지할 행: {len(first_18_indices)}개 (인덱스 {first_18_indices[0]} ~ {first_18_indices[-1]})")
    print(f"  제거할 행: {len(remaining_indices)}개 (인덱스 {remaining_indices[0]} ~ {remaining_indices[-1]})")
    
    rows_to_drop.extend(remaining_indices.tolist())

# 중복 행 제거
print(f"\n총 제거할 행 수: {len(rows_to_drop)}")
data_cleaned = data.drop(index=rows_to_drop)

# 중복 제거 후 통계
print(f"\n[제거 후]")
print(f"  전체 행 수: {len(data_cleaned)}")
print(f"  전체 개인 수: {data_cleaned['respondent_id'].nunique()}")

row_counts_cleaned = data_cleaned.groupby('respondent_id').size()
print(f"  개인별 행 수 분포:")
print(f"    {row_counts_cleaned.value_counts().sort_index().to_dict()}")

# 검증: 모든 개인이 18행인지 확인
if (row_counts_cleaned == 18).all():
    print(f"\n✓ 검증 성공: 모든 개인이 18행 (6 choice_sets × 3 alternatives)")
else:
    print(f"\n✗ 검증 실패: 일부 개인의 행 수가 18이 아닙니다!")
    print(row_counts_cleaned[row_counts_cleaned != 18])

# 저장
data_cleaned.to_csv(data_path, index=False)
print(f"\n✓ 정리된 데이터 저장 완료: {data_path}")

print("\n" + "="*80)
print("완료!")
print("="*80)
print(f"제거된 행 수: {len(data) - len(data_cleaned)}")
print(f"최종 데이터: {len(data_cleaned)}행 × {len(data_cleaned.columns)}열")
print(f"백업 파일: {backup_path}")

