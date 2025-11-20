"""
CFA 결과에서 측정모델 파라미터 확인
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

print("="*80)
print("CFA 파라미터 확인")
print("="*80)

# CFA 결과 로드
cfa_path = Path('results/sequential_stage_wise/cfa_results.pkl')
with open(cfa_path, 'rb') as f:
    cfa_results = pickle.load(f)

print(f"\nCFA 결과 키: {list(cfa_results.keys())}")

# 1. Loadings (요인적재량)
if 'loadings' in cfa_results:
    loadings = cfa_results['loadings']
    print(f"\n[1] Loadings (요인적재량)")
    print(f"{'='*80}")
    if isinstance(loadings, pd.DataFrame):
        print(loadings.head(10))
    else:
        print(loadings)

# 2. Measurement Errors (측정오차)
if 'measurement_errors' in cfa_results:
    errors = cfa_results['measurement_errors']
    print(f"\n[2] Measurement Errors (측정오차)")
    print(f"{'='*80}")
    if isinstance(errors, pd.DataFrame):
        print(errors.head(10))
        print(f"\n통계 (Estimate 컬럼):")
        print(f"  평균: {errors['Estimate'].mean():.4f}")
        print(f"  범위: [{errors['Estimate'].min():.4f}, {errors['Estimate'].max():.4f}]")
    else:
        print(errors)

# 3. Fit Indices
if 'fit_indices' in cfa_results:
    fit = cfa_results['fit_indices']
    print(f"\n[3] Fit Indices")
    print(f"{'='*80}")
    if isinstance(fit, pd.DataFrame):
        print(fit)
    else:
        print(fit)

# 4. Log-likelihood
if 'log_likelihood' in cfa_results:
    ll = cfa_results['log_likelihood']
    print(f"\n[4] Log-likelihood")
    print(f"{'='*80}")
    print(f"  {ll}")

# 5. CSV 파일 확인
print(f"\n{'='*80}")
print(f"[5] CSV 파일 확인")
print(f"{'='*80}")

csv_path = Path('results/sequential_stage_wise/cfa_results_measurement_params.csv')
if csv_path.exists():
    params_df = pd.read_csv(csv_path)
    print(f"\n측정모델 파라미터 CSV:")
    print(params_df.head(20))
    
    # σ² 확인
    if 'sigma_sq' in params_df.columns:
        print(f"\nσ² 통계:")
        print(f"  평균: {params_df['sigma_sq'].mean():.4f}")
        print(f"  범위: [{params_df['sigma_sq'].min():.4f}, {params_df['sigma_sq'].max():.4f}]")
        print(f"\nσ² 분포 (처음 10개):")
        print(params_df[['latent_variable', 'indicator', 'sigma_sq']].head(10))
else:
    print(f"CSV 파일 없음: {csv_path}")

# 6. 실제 우도 계산 예시
print(f"\n{'='*80}")
print(f"[6] 실제 우도 계산 예시")
print(f"{'='*80}")

# measurement_errors에서 σ² 가져오기
if 'measurement_errors' in cfa_results:
    errors = cfa_results['measurement_errors']

    # health_concern의 첫 번째 지표 (q6)
    q6_error = errors[errors['lval'] == 'q6'].iloc[0]

    print(f"\nhealth_concern 첫 번째 지표 (q6):")
    print(f"  σ² (Estimate): {q6_error['Estimate']:.4f}")

    # loadings에서 ζ 가져오기
    loadings = cfa_results['loadings']
    q6_loading = loadings[loadings['lval'] == 'q6'].iloc[0]
    print(f"  ζ (Estimate): {q6_loading['Estimate']:.4f}")

    # 우도 계산
    lv = 0.5
    y_obs = 4.0
    zeta = q6_loading['Estimate']
    sigma_sq = q6_error['Estimate']

    y_pred = zeta * lv
    residual = y_obs - y_pred
    ll = -0.5 * np.log(2 * np.pi * sigma_sq) - 0.5 * (residual**2) / sigma_sq

    print(f"\n우도 계산 (LV={lv}, y_obs={y_obs}):")
    print(f"  y_pred = {zeta:.4f} × {lv} = {y_pred:.4f}")
    print(f"  residual = {y_obs} - {y_pred:.4f} = {residual:.4f}")
    print(f"  ll = -0.5×log(2π×{sigma_sq:.4f}) - 0.5×({residual:.4f})²/{sigma_sq:.4f}")
    print(f"     = {ll:.4f}")

    # 모든 지표의 평균 우도
    print(f"\n모든 지표의 평균 σ²로 계산:")
    avg_sigma_sq = errors['Estimate'].mean()
    ll_avg = -0.5 * np.log(2 * np.pi * avg_sigma_sq) - 0.5 * (residual**2) / avg_sigma_sq
    print(f"  평균 σ²: {avg_sigma_sq:.4f}")
    print(f"  ll = {ll_avg:.4f}")

    # 38개 지표 전체 우도 (평균 σ² 사용)
    print(f"\n38개 지표 전체 우도 (평균 σ² 사용):")
    total_ll = ll_avg * 38
    print(f"  ll × 38 = {ll_avg:.4f} × 38 = {total_ll:.4f}")
    print(f"  지표당 평균: {total_ll / 38:.4f}")

