"""
현재 데이터의 표준화 여부 확인

1. CFA 결과의 표준화 여부
2. 구조모델 추정시 사용되는 LV 값의 표준화 여부
3. 선택속성값의 표준화 여부
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("="*80)
print("데이터 표준화 여부 확인")
print("="*80)

# ============================================================================
# 1. 통합 데이터 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[1] 통합 데이터 (integrated_data.csv)")
print(f"{'='*80}")

data = pd.read_csv('data/processed/iclv/integrated_data.csv')
print(f"데이터 shape: {data.shape}")

# 선택 속성
print(f"\n[1-1] 선택 속성 (Choice Attributes)")
print(f"{'='*80}")

for col in ['price', 'health_label']:
    if col in data.columns:
        values = data[col].dropna()
        print(f"\n{col}:")
        print(f"  평균: {values.mean():.6f}")
        print(f"  표준편차: {values.std():.6f}")
        print(f"  범위: [{values.min():.2f}, {values.max():.2f}]")
        print(f"  고유값 개수: {values.nunique()}")
        if values.nunique() <= 10:
            print(f"  고유값: {sorted(values.unique())}")
        
        # 표준화 여부 판단
        if abs(values.mean()) < 0.01 and abs(values.std() - 1.0) < 0.1:
            print(f"  ✅ 표준화됨 (평균 ≈ 0, 표준편차 ≈ 1)")
        elif values.min() >= 0 and values.max() <= 1:
            print(f"  ℹ️  이진 변수 (0-1)")
        else:
            print(f"  ❌ 표준화 안됨 (원척도)")

# 사회인구학적 변수
print(f"\n[1-2] 사회인구학적 변수 (Sociodemographics)")
print(f"{'='*80}")

for col in ['age_std', 'income_std', 'gender', 'education_level']:
    if col in data.columns:
        values = data[col].dropna()
        print(f"\n{col}:")
        print(f"  평균: {values.mean():.6f}")
        print(f"  표준편차: {values.std():.6f}")
        print(f"  범위: [{values.min():.2f}, {values.max():.2f}]")
        
        # 표준화 여부 판단
        if abs(values.mean()) < 0.01 and abs(values.std() - 1.0) < 0.1:
            print(f"  ✅ 표준화됨 (평균 ≈ 0, 표준편차 ≈ 1)")
        else:
            print(f"  ❌ 표준화 안됨 (원척도)")

# 잠재변수 지표
print(f"\n[1-3] 잠재변수 지표 (Indicators)")
print(f"{'='*80}")

indicators = {
    'health_concern': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    'perceived_benefit': ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
    'perceived_price': ['q27', 'q28', 'q29'],
    'nutrition_knowledge': [f'q{i}' for i in range(30, 50)],
    'purchase_intention': ['q18', 'q19', 'q20']
}

for lv_name, inds in indicators.items():
    print(f"\n{lv_name}:")
    for ind in inds[:3]:  # 처음 3개만
        if ind in data.columns:
            values = data[ind].dropna()
            print(f"  {ind}: mean={values.mean():.4f}, std={values.std():.4f}, range=[{values.min():.0f}, {values.max():.0f}]")
    
    # 표준화 여부 판단
    first_ind = inds[0]
    if first_ind in data.columns:
        values = data[first_ind].dropna()
        if abs(values.mean()) < 0.01 and abs(values.std() - 1.0) < 0.1:
            print(f"  ✅ 표준화됨")
        else:
            print(f"  ❌ 표준화 안됨 (원척도, 1-5점 리커트)")

# ============================================================================
# 2. CFA 결과 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[2] CFA 결과 (요인점수)")
print(f"{'='*80}")

cfa_path = Path('results/sequential_stage_wise/cfa_results.pkl')
if cfa_path.exists():
    with open(cfa_path, 'rb') as f:
        cfa_results = pickle.load(f)
    
    if 'factor_scores' in cfa_results:
        factor_scores = cfa_results['factor_scores']
        
        print(f"\nCFA 요인점수 (원본, 표준화 전):")
        for lv_name, scores in factor_scores.items():
            print(f"\n{lv_name}:")
            print(f"  평균: {scores.mean():.6f}")
            print(f"  표준편차: {scores.std():.6f}")
            print(f"  범위: [{scores.min():.4f}, {scores.max():.4f}]")
            
            # 표준화 여부 판단
            if abs(scores.mean()) < 0.01 and abs(scores.std() - 1.0) < 0.1:
                print(f"  ✅ 표준화됨 (평균 ≈ 0, 표준편차 ≈ 1)")
            else:
                print(f"  ❌ 표준화 안됨 (원척도)")
else:
    print(f"\nCFA 결과 파일 없음: {cfa_path}")

# ============================================================================
# 3. 순차추정 2단계에서 사용되는 요인점수 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[3] 순차추정 2단계 요인점수 (표준화 후)")
print(f"{'='*80}")

# 가장 최근 stage1 결과 찾기
stage1_dir = Path('results/sequential_stage_wise')
stage1_files = list(stage1_dir.glob('stage1_*_results.pkl'))

if stage1_files:
    latest_file = max(stage1_files, key=lambda p: p.stat().st_mtime)
    print(f"\n파일: {latest_file.name}")
    
    with open(latest_file, 'rb') as f:
        stage1_results = pickle.load(f)
    
    if 'factor_scores' in stage1_results:
        factor_scores = stage1_results['factor_scores']
        
        print(f"\n순차추정 1단계 요인점수 (표준화 후):")
        for lv_name, scores in factor_scores.items():
            print(f"\n{lv_name}:")
            print(f"  평균: {scores.mean():.6f}")
            print(f"  표준편차: {scores.std():.6f}")
            print(f"  범위: [{scores.min():.4f}, {scores.max():.4f}]")
            
            # 표준화 여부 판단
            if abs(scores.mean()) < 0.01 and abs(scores.std() - 1.0) < 0.1:
                print(f"  ✅ 표준화됨 (평균 ≈ 0, 표준편차 ≈ 1)")
            else:
                print(f"  ❌ 표준화 안됨")
else:
    print(f"\nstage1 결과 파일 없음")

print(f"\n{'='*80}")
print(f"요약 완료")
print(f"{'='*80}")

