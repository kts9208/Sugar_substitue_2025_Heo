import pandas as pd

# DCE 데이터
df_dce = pd.read_csv('data/processed/dce/dce_long_format.csv')
print("DCE 데이터:")
print(f"  - 총 행 수: {len(df_dce)}")
print(f"  - 고유 respondent_id: {df_dce['respondent_id'].nunique()}")
print(f"  - respondent_id 범위: {df_dce['respondent_id'].min()} ~ {df_dce['respondent_id'].max()}")
print(f"  - respondent_id 샘플: {sorted(df_dce['respondent_id'].unique())[:10]}")

# 건강관심도 데이터
df_health = pd.read_csv('data/processed/survey/health_concern.csv')
df_health = df_health.rename(columns={'no': 'respondent_id'})
print("\n건강관심도 데이터:")
print(f"  - 총 행 수: {len(df_health)}")
print(f"  - 고유 respondent_id: {df_health['respondent_id'].nunique()}")
print(f"  - respondent_id 범위: {df_health['respondent_id'].min()} ~ {df_health['respondent_id'].max()}")
print(f"  - respondent_id 샘플: {sorted(df_health['respondent_id'].unique())[:10]}")

# 사회인구학적 데이터
df = pd.read_excel('data/raw/Sugar_substitue_Raw data_251108.xlsx', sheet_name='DATA')
df_sociodem = df[['no']].copy()
df_sociodem = df_sociodem.rename(columns={'no': 'respondent_id'})
print("\n사회인구학적 데이터:")
print(f"  - 총 행 수: {len(df_sociodem)}")
print(f"  - 고유 respondent_id: {df_sociodem['respondent_id'].nunique()}")
print(f"  - respondent_id 범위: {df_sociodem['respondent_id'].min()} ~ {df_sociodem['respondent_id'].max()}")
print(f"  - respondent_id 샘플: {sorted(df_sociodem['respondent_id'].unique())[:10]}")

# 중복 확인
print("\n중복 확인:")
print(f"  - 건강관심도 중복: {df_health['respondent_id'].duplicated().sum()}")
print(f"  - 사회인구학적 중복: {df_sociodem['respondent_id'].duplicated().sum()}")

# 매칭 확인
dce_ids = set(df_dce['respondent_id'].unique())
health_ids = set(df_health['respondent_id'].unique())
sociodem_ids = set(df_sociodem['respondent_id'].unique())

print("\n매칭 확인:")
print(f"  - DCE에만 있는 ID: {len(dce_ids - health_ids - sociodem_ids)}")
print(f"  - 건강관심도에만 있는 ID: {len(health_ids - dce_ids)}")
print(f"  - 사회인구학적에만 있는 ID: {len(sociodem_ids - dce_ids)}")
print(f"  - 공통 ID: {len(dce_ids & health_ids & sociodem_ids)}")

