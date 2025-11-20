"""전처리 과정에서 개인 ID=2의 데이터가 손실되었는지 확인"""
import pandas as pd
import numpy as np

print("=" * 80)
print("1. 원본 데이터 확인 (st2_HC-PB_PB-PI1_PI2_results.csv)")
print("=" * 80)

raw = pd.read_csv('data/processed/st2_HC-PB_PB-PI1_PI2_results.csv')
print(f'원본 데이터 shape: {raw.shape}')
print(f'원본 데이터 개인 수: {len(raw["respondent_id"].unique())}')

# 개인 ID=2 확인
ind2_raw = raw[raw['respondent_id'] == 2]
print(f'\n개인 ID=2 행 수: {len(ind2_raw)}')

if len(ind2_raw) > 0:
    print(f'개인 ID=2 첫 번째 행:')
    indicators = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
    for ind in indicators:
        if ind in ind2_raw.columns:
            val = ind2_raw[ind].iloc[0]
            print(f'  {ind}: {val}')
        else:
            print(f'  {ind}: 컬럼 없음')
else:
    print('개인 ID=2가 원본 데이터에 없습니다!')

print("\n" + "=" * 80)
print("2. 전처리된 데이터 확인 (integrated_data.csv)")
print("=" * 80)

processed = pd.read_csv('data/processed/iclv/integrated_data.csv')
print(f'전처리 데이터 shape: {processed.shape}')
print(f'전처리 데이터 개인 수: {len(processed["respondent_id"].unique())}')

# 개인 ID=2 확인
ind2_processed = processed[processed['respondent_id'] == 2]
print(f'\n개인 ID=2 행 수: {len(ind2_processed)}')

if len(ind2_processed) > 0:
    print(f'개인 ID=2 첫 번째 행:')
    for ind in indicators:
        if ind in ind2_processed.columns:
            val = ind2_processed[ind].iloc[0]
            print(f'  {ind}: {val}')
        else:
            print(f'  {ind}: 컬럼 없음')
else:
    print('개인 ID=2가 전처리 데이터에 없습니다!')

print("\n" + "=" * 80)
print("3. 전처리 스크립트 확인")
print("=" * 80)
print("전처리 스크립트: scripts/preprocess_iclv_data.py")
print("확인 필요: 데이터 병합 과정에서 개인 ID=2의 지표 데이터가 손실되었는지")

