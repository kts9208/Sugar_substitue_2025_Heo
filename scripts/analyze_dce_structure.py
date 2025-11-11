"""
DCE 데이터 구조 분석 스크립트
목적: q21-q26 변수의 의미와 구조를 파악
"""

import pandas as pd
import numpy as np

# 1. 원본 데이터 로드
print("=" * 80)
print("DCE 데이터 구조 분석")
print("=" * 80)

# DATA 시트
df_data = pd.read_excel('data/raw/Sugar_substitue_Raw data_251108.xlsx', sheet_name='DATA')
print("\n[1] DATA 시트 - DCE 변수 (q21-q26)")
print("-" * 80)
print(f"총 응답자 수: {len(df_data)}")
print(f"\nDCE 관련 컬럼: {[c for c in df_data.columns if 'q2' in c and c[1:].replace('_', '').isdigit()]}")

# q21-q26 데이터 확인
dce_cols = ['q21', 'q22', 'q23', 'q24', 'q25', 'q26']
print(f"\n첫 10명의 DCE 응답:")
print(df_data[['no'] + dce_cols].head(10))

print(f"\n각 변수의 고유값:")
for col in dce_cols:
    unique_vals = sorted(df_data[col].unique())
    print(f"  {col}: {unique_vals} (범위: {min(unique_vals)}-{max(unique_vals)})")

print(f"\n기술통계:")
print(df_data[dce_cols].describe())

# LABEL 시트
try:
    df_label = pd.read_excel('data/raw/Sugar_substitue_Raw data_251108.xlsx', sheet_name='LABEL')
    print("\n[2] LABEL 시트 - 변수 라벨")
    print("-" * 80)
    for col in dce_cols:
        if col in df_label.columns:
            label = df_label[col].iloc[0]
            print(f"  {col}: {label}")
except Exception as e:
    print(f"\nLABEL 시트 읽기 실패: {e}")

# CODE 시트
try:
    df_code = pd.read_excel('data/raw/Sugar_substitue_Raw data_251108.xlsx', sheet_name='CODE')
    print("\n[3] CODE 시트 - 코드북")
    print("-" * 80)
    print("전체 CODE 시트 내용:")
    print(df_code.head(100))
except Exception as e:
    print(f"\nCODE 시트 읽기 실패: {e}")

# 패턴 분석
print("\n[4] 패턴 분석")
print("-" * 80)

# 각 변수의 값 분포
print("\n값 분포:")
for col in dce_cols:
    value_counts = df_data[col].value_counts().sort_index()
    print(f"\n{col}:")
    print(value_counts)

# 상관관계
print("\n\n변수 간 상관관계:")
print(df_data[dce_cols].corr())

print("\n" + "=" * 80)
print("분석 완료")
print("=" * 80)

