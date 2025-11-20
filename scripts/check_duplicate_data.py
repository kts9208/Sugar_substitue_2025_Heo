"""
중복 데이터 확인 스크립트
"""
import pandas as pd
from pathlib import Path

# 데이터 로드
data_path = Path('data/processed/iclv/integrated_data.csv')
data = pd.read_csv(data_path)

print("="*80)
print("중복 데이터 분석")
print("="*80)

# 개인별 행 수 확인
row_counts = data.groupby('respondent_id').size()
print(f"\n전체 데이터 shape: {data.shape}")
print(f"전체 개인 수: {len(row_counts)}")
print(f"\n개인별 행 수 분포:")
print(row_counts.value_counts().sort_index())

# 중복된 개인 찾기
dup_ids = row_counts[row_counts > 18].index.tolist()
print(f"\n중복된 respondent_id 개수: {len(dup_ids)}")
print(f"중복된 respondent_id: {dup_ids}")

# 각 중복 개인 상세 분석
for rid in dup_ids:
    subset = data[data['respondent_id'] == rid]
    print(f"\n{'='*80}")
    print(f"respondent_id: {rid}")
    print(f"{'='*80}")
    print(f"총 행 수: {len(subset)}")
    print(f"choice_set 개수: {subset['choice_set'].nunique()}")
    print(f"choice_set 값들: {sorted(subset['choice_set'].unique())}")
    
    # choice_set별 행 수
    cs_counts = subset.groupby('choice_set').size()
    print(f"\nchoice_set별 행 수:")
    print(cs_counts)
    
    # 첫 20행 출력
    print(f"\n첫 20행:")
    print(subset[['respondent_id', 'choice_set', 'alternative', 'alternative_name', 'choice']].head(20))
    
    # 중복 여부 확인 (choice_set, alternative 조합)
    duplicates = subset.duplicated(subset=['choice_set', 'alternative'], keep=False)
    if duplicates.any():
        print(f"\n⚠️ 중복된 (choice_set, alternative) 조합 발견!")
        print(f"중복 행 수: {duplicates.sum()}")
        print("\n중복된 행들:")
        print(subset[duplicates][['respondent_id', 'choice_set', 'alternative', 'alternative_name', 'choice']].head(20))

# 정상 개인 샘플 확인
normal_ids = row_counts[row_counts == 18].index.tolist()[:3]
print(f"\n{'='*80}")
print(f"정상 개인 샘플 (비교용)")
print(f"{'='*80}")
for rid in normal_ids:
    subset = data[data['respondent_id'] == rid]
    print(f"\nrespondent_id: {rid}")
    print(f"  총 행 수: {len(subset)}")
    print(f"  choice_set 개수: {subset['choice_set'].nunique()}")
    print(f"  choice_set 값들: {sorted(subset['choice_set'].unique())}")

