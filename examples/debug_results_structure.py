"""
Results 구조 디버깅
"""

import sys
from pathlib import Path
import pandas as pd
import pickle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 기존 결과 파일 로드
stage2_result_path = project_root / 'results' / 'final' / 'sequential' / '2path' / 'stage2' / 'st2_2path1_base2_results.csv'

print("="*70)
print("기존 결과 파일 확인")
print("="*70)
print(f"파일: {stage2_result_path.name}")

df = pd.read_csv(stage2_result_path)
print(f"\n컬럼: {list(df.columns)}")
print(f"\n행 수: {len(df)}")
print("\n내용:")
print(df.to_string())

# theta와 gamma 파라미터 확인
theta_params = df[df['parameter'].str.contains('theta', na=False)]
gamma_params = df[df['parameter'].str.contains('gamma', na=False)]

print(f"\n\ntheta 파라미터 수: {len(theta_params)}")
print(f"gamma 파라미터 수: {len(gamma_params)}")

if len(theta_params) > 0:
    print("\ntheta 파라미터:")
    print(theta_params[['parameter', 'estimate', 'p_value', 'significance']].to_string())

if len(gamma_params) > 0:
    print("\ngamma 파라미터:")
    print(gamma_params[['parameter', 'estimate', 'p_value', 'significance']].to_string())

