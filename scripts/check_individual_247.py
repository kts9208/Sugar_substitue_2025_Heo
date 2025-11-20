"""개인 247의 데이터 확인 (전처리 후)"""
import pandas as pd
import numpy as np

# 데이터 로드
data = pd.read_csv('data/processed/iclv/integrated_data.csv')

print("=" * 80)
print("전처리 후 개인 247 데이터 확인")
print("=" * 80)

# 개인 247 (인덱스 246)
unique_ids = data['respondent_id'].unique()
print(f'\n전체 개인 수: {len(unique_ids)}')
print(f'개인 ID 범위: {unique_ids.min()} ~ {unique_ids.max()}')

ind_247_id = unique_ids[246]
print(f'\n개인 247 (인덱스 246) ID: {ind_247_id}')

# 개인 247 데이터
ind_247 = data[data['respondent_id'] == ind_247_id].iloc[0]

# 측정 지표 (실제 사용되는 것만)
indicators = [
    'q6', 'q7', 'q8', 'q9', 'q10', 'q11',  # health_concern
    'q12', 'q13', 'q14', 'q15', 'q16', 'q17',  # perceived_benefit
    'q18', 'q19', 'q20',  # purchase_intention
    'q27', 'q28', 'q29',  # perceived_price
    'q30', 'q31', 'q32', 'q33', 'q34', 'q35', 'q36', 'q37', 'q38', 'q39',
    'q40', 'q41', 'q42', 'q43', 'q44', 'q45', 'q46', 'q47', 'q48', 'q49'  # nutrition_knowledge
]

print(f'\n개인 247 지표 값 (처음 10개):')
for ind in indicators[:10]:
    if ind in ind_247.index:
        val = ind_247[ind]
        if pd.isna(val):
            print(f'  {ind}: NaN ❌')
        else:
            print(f'  {ind}: {val} ✅')
    else:
        print(f'  {ind}: 컬럼 없음 ❌')

# NaN 개수 확인
existing_indicators = [ind for ind in indicators if ind in ind_247.index]
nan_count = ind_247[existing_indicators].isna().sum()

print(f'\nNaN 개수: {nan_count}/{len(existing_indicators)}')

if nan_count == 0:
    print('✅ 모든 지표 값이 정상입니다!')

    # 통계
    print(f'\n지표 통계:')
    print(f'  최소값: {ind_247[existing_indicators].min()}')
    print(f'  최대값: {ind_247[existing_indicators].max()}')
    print(f'  평균: {ind_247[existing_indicators].mean():.4f}')
    print(f'  표준편차: {ind_247[existing_indicators].std():.4f}')
else:
    print(f'❌ {nan_count}개 지표에 NaN이 있습니다!')
    print('NaN이 있는 지표:')
    for ind in existing_indicators:
        if pd.isna(ind_247[ind]):
            print(f'  - {ind}')

