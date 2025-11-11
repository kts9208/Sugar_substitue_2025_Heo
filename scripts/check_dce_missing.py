import pandas as pd

df = pd.read_excel('data/raw/Sugar_substitue_Raw data_251108.xlsx', sheet_name='DATA')
dce_cols = ['no', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26']
df_dce = df[dce_cols]

print('결측치 확인:')
print(df_dce.isnull().sum())

print('\n값 범위:')
for col in ['q21', 'q22', 'q23', 'q24', 'q25', 'q26']:
    print(f'{col}: {sorted(df_dce[col].dropna().unique())}')

print('\n결측치가 있는 응답자:')
missing_mask = df_dce[['q21', 'q22', 'q23', 'q24', 'q25', 'q26']].isnull().any(axis=1)
print(df_dce[missing_mask])

