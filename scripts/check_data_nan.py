"""데이터 NaN 확인"""
import pandas as pd
import numpy as np
from pathlib import Path

# 데이터 로드
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
data = pd.read_csv(data_path)

print("전체 데이터 shape:", data.shape)
print("\n선택 속성 NaN 개수:")
print(data[['sugar_free', 'health_label', 'price']].isna().sum())

print("\n첫 번째 개인 데이터:")
first_person = data[data['respondent_id'] == data['respondent_id'].iloc[0]]
print(f"개인 ID: {first_person['respondent_id'].iloc[0]}")
print(f"행 수: {len(first_person)}")
print("\n선택 속성:")
print(first_person[['choice_set', 'alternative', 'sugar_free', 'health_label', 'price', 'choice']])

