"""
income=6 문제 확인 스크립트
"""
import pandas as pd
import numpy as np

print('='*70)
print('income=6 문제 확인')
print('='*70)

# Raw data 로드
excel_file = 'data/raw/Sugar_substitue_Raw data_250730.xlsx'
raw_data = pd.read_excel(excel_file, sheet_name='DATA')

# q52가 income 열
print(f'\nq52 (income) 값 분포:')
print(raw_data['q52'].value_counts().sort_index())

# income=6인 사람들 확인
income_6 = raw_data[raw_data['q52'] == 6]
print(f'\nincome=6인 사람 수: {len(income_6)}')
print(f'respondent_id: {income_6["no"].tolist()}')

# income mapping 확인
income_mapping = {1: 2000, 2: 4000, 3: 6000, 4: 8000, 5: 10000}
print(f'\nincome_mapping (현재):')
for k, v in income_mapping.items():
    print(f'  {k} → {v}')

print(f'\n문제: income=6은 mapping에 없음!')
print(f'결과: income_continuous = NaN → income_std = NaN')

# CODE 시트에서 income 코딩 확인
print(f'\n\nCODE 시트에서 income 코딩 확인:')
code_sheet = pd.read_excel(excel_file, sheet_name='CODE')
print(code_sheet.head(30))

