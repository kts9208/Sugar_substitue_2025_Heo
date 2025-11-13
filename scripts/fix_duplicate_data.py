"""중복 데이터 제거"""
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'

print("데이터 로드 중...")
data = pd.read_csv(data_path)

print(f"원본 데이터: {data.shape}")
print(f"원본 개인 수: {data['respondent_id'].nunique()}")

# 개인별 행 수 확인
sizes_before = data.groupby('respondent_id').size()
print(f"\n중복 전 개인별 행 수 분포:")
print(sizes_before.value_counts().sort_index())

# 중복 제거: respondent_id, choice_set, alternative 조합으로 중복 제거
print("\n중복 제거 중...")
data_cleaned = data.drop_duplicates(subset=['respondent_id', 'choice_set', 'alternative'], keep='first')

print(f"\n정제 후 데이터: {data_cleaned.shape}")
print(f"정제 후 개인 수: {data_cleaned['respondent_id'].nunique()}")

# 개인별 행 수 확인
sizes_after = data_cleaned.groupby('respondent_id').size()
print(f"\n중복 제거 후 개인별 행 수 분포:")
print(sizes_after.value_counts().sort_index())

# 모든 개인이 18행인지 확인
if (sizes_after == 18).all():
    print("\n✅ 모든 개인이 18행입니다!")
else:
    print("\n⚠️ 여전히 18행이 아닌 개인이 있습니다:")
    print(sizes_after[sizes_after != 18])

# 저장
output_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data_cleaned.csv'
data_cleaned.to_csv(output_path, index=False)
print(f"\n정제된 데이터 저장: {output_path}")

