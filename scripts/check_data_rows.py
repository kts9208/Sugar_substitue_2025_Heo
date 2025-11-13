"""데이터 행 수 확인"""
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'

print("데이터 로드 중...")
data = pd.read_csv(data_path)

print(f"\n전체 데이터: {data.shape}")
print(f"전체 개인 수: {data['respondent_id'].nunique()}")

print("\n개인별 행 수 분포:")
sizes = data.groupby('respondent_id').size()
print(sizes.value_counts().sort_index())

print("\n36행인 개인 ID:")
problem_ids = sizes[sizes == 36].index.tolist()
print(problem_ids)

if len(problem_ids) > 0:
    print(f"\n첫 번째 문제 개인 (ID={problem_ids[0]}) 데이터:")
    problem_data = data[data['respondent_id'] == problem_ids[0]]
    print(problem_data[['respondent_id', 'choice_set', 'alternative']].to_string())
    
    print(f"\n정상 개인 (18행) 예시 (ID=1) 데이터:")
    normal_data = data[data['respondent_id'] == 1]
    print(normal_data[['respondent_id', 'choice_set', 'alternative']].to_string())

