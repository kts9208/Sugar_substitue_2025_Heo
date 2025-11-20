"""
CFA 결과 파라미터의 표준화 여부 확인

측정모델 우도 계산시 사용되는 파라미터:
- ζ (zeta): 요인적재량 (factor loadings)
- σ² (sigma_sq): 측정오차 분산 (measurement error variance)
"""
import pickle
import numpy as np
from pathlib import Path

print("="*80)
print("CFA 파라미터 표준화 여부 확인")
print("="*80)

# CFA 결과 로드
cfa_path = Path('results/sequential_stage_wise/cfa_results.pkl')

if not cfa_path.exists():
    print(f"\nCFA 결과 파일 없음: {cfa_path}")
    exit(1)

with open(cfa_path, 'rb') as f:
    cfa_results = pickle.load(f)

print(f"\nCFA 결과 파일: {cfa_path}")
print(f"키: {list(cfa_results.keys())}")

# ============================================================================
# 1. 요인적재량 (ζ) 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[1] 요인적재량 (ζ, Factor Loadings)")
print(f"{'='*80}")

if 'measurement_params' in cfa_results:
    measurement_params = cfa_results['measurement_params']
    
    for lv_name, params in measurement_params.items():
        print(f"\n{lv_name}:")
        
        if 'zeta' in params:
            zeta = params['zeta']
            print(f"  ζ (요인적재량):")
            print(f"    값: {zeta}")
            print(f"    평균: {np.mean(zeta):.4f}")
            print(f"    표준편차: {np.std(zeta):.4f}")
            print(f"    범위: [{np.min(zeta):.4f}, {np.max(zeta):.4f}]")
            
            # 표준화 여부 판단
            if np.allclose(zeta, 1.0, atol=0.1):
                print(f"    ✅ 모두 1에 가까움 (표준화됨)")
            elif np.abs(np.mean(zeta)) < 0.01 and np.abs(np.std(zeta) - 1.0) < 0.1:
                print(f"    ✅ 표준화됨 (평균 ≈ 0, 표준편차 ≈ 1)")
            else:
                print(f"    ❌ 표준화 안됨 (원척도)")
        
        if 'sigma_sq' in params:
            sigma_sq = params['sigma_sq']
            print(f"  σ² (측정오차 분산):")
            print(f"    값: {sigma_sq}")
            print(f"    평균: {np.mean(sigma_sq):.4f}")
            print(f"    표준편차: {np.std(sigma_sq):.4f}")
            print(f"    범위: [{np.min(sigma_sq):.4f}, {np.max(sigma_sq):.4f}]")

# ============================================================================
# 2. semopy 원본 파라미터 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[2] semopy 원본 파라미터")
print(f"{'='*80}")

if 'model_params' in cfa_results:
    import pandas as pd
    params_df = cfa_results['model_params']
    
    # 요인적재량 (op == '~')
    loadings = params_df[params_df['op'] == '~'].copy()
    
    if len(loadings) > 0:
        print(f"\n요인적재량 (Factor Loadings):")
        print(f"  개수: {len(loadings)}")
        print(f"  평균: {loadings['Estimate'].mean():.4f}")
        print(f"  표준편차: {loadings['Estimate'].std():.4f}")
        print(f"  범위: [{loadings['Estimate'].min():.4f}, {loadings['Estimate'].max():.4f}]")
        
        print(f"\n처음 10개:")
        print(loadings[['lval', 'op', 'rval', 'Estimate']].head(10).to_string(index=False))
        
        # 잠재변수별 통계
        print(f"\n잠재변수별 요인적재량 통계:")
        for lv in loadings['rval'].unique():
            lv_loadings = loadings[loadings['rval'] == lv]['Estimate']
            print(f"  {lv}: 평균={lv_loadings.mean():.4f}, std={lv_loadings.std():.4f}, 범위=[{lv_loadings.min():.4f}, {lv_loadings.max():.4f}]")
    
    # 측정오차 분산 (op == '~~' and lval == rval)
    variances = params_df[(params_df['op'] == '~~') & (params_df['lval'] == params_df['rval'])].copy()
    
    # 관측변수의 분산만 (잠재변수 제외)
    latent_vars = loadings['rval'].unique()
    obs_variances = variances[~variances['lval'].isin(latent_vars)]
    
    if len(obs_variances) > 0:
        print(f"\n측정오차 분산 (Measurement Error Variance):")
        print(f"  개수: {len(obs_variances)}")
        print(f"  평균: {obs_variances['Estimate'].mean():.4f}")
        print(f"  표준편차: {obs_variances['Estimate'].std():.4f}")
        print(f"  범위: [{obs_variances['Estimate'].min():.4f}, {obs_variances['Estimate'].max():.4f}]")
        
        print(f"\n처음 10개:")
        print(obs_variances[['lval', 'op', 'rval', 'Estimate']].head(10).to_string(index=False))

# ============================================================================
# 3. 표준화 여부 판단
# ============================================================================
print(f"\n{'='*80}")
print(f"[3] 표준화 여부 판단")
print(f"{'='*80}")

if 'model_params' in cfa_results:
    loadings = params_df[params_df['op'] == '~']
    
    print(f"\n요인적재량 (ζ):")
    
    # 각 잠재변수의 첫 번째 지표 확인 (identification constraint)
    first_indicators = {}
    for lv in loadings['rval'].unique():
        lv_loadings = loadings[loadings['rval'] == lv]
        first_ind = lv_loadings.iloc[0]
        first_indicators[lv] = first_ind['Estimate']
        print(f"  {lv}의 첫 번째 지표 ({first_ind['lval']}): ζ = {first_ind['Estimate']:.4f}")
    
    # 판단
    all_ones = all(np.isclose(val, 1.0, atol=0.01) for val in first_indicators.values())
    
    if all_ones:
        print(f"\n  ✅ 모든 첫 번째 지표의 ζ = 1.0 (식별 제약)")
        print(f"  ✅ 이는 표준화된 해(standardized solution)를 의미합니다.")
        print(f"  ✅ 잠재변수 분산 Var(LV) = 1로 고정됨")
    else:
        print(f"\n  ❌ 첫 번째 지표의 ζ ≠ 1.0")
        print(f"  ❌ 비표준화 해(unstandardized solution)")
    
    print(f"\n측정오차 분산 (σ²):")
    if len(obs_variances) > 0:
        print(f"  평균: {obs_variances['Estimate'].mean():.4f}")
        print(f"  범위: [{obs_variances['Estimate'].min():.4f}, {obs_variances['Estimate'].max():.4f}]")
        
        # 원척도 지표의 분산과 비교
        print(f"\n  ℹ️  원척도 지표(1-5점)의 이론적 분산:")
        print(f"     - 균등분포 가정: Var = (5-1)²/12 ≈ 1.33")
        print(f"     - 실제 관측 분산: 약 0.6-1.0 (데이터 확인 필요)")
        print(f"  ℹ️  CFA 측정오차 분산: 평균 {obs_variances['Estimate'].mean():.4f}")
        
        if obs_variances['Estimate'].mean() < 1.0:
            print(f"  ✅ 측정오차 분산이 작음 → 높은 신뢰도")
        else:
            print(f"  ⚠️  측정오차 분산이 큼 → 낮은 신뢰도")

print(f"\n{'='*80}")
print(f"분석 완료")
print(f"{'='*80}")

