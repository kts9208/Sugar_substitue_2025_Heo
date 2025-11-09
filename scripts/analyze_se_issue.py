"""
표준오차가 1.0 근처인 이유 분석
"""

import pandas as pd
import numpy as np

# 결과 파일 로드
df = pd.read_csv('results/iclv_full_data_results.csv')

# 파라미터만 추출 (Estimation statistics 제외)
df_params = df[df['Coefficient'].notna() & (df['Coefficient'] != '') & (df['Coefficient'] != 'Estimation statistics')].copy()

# 표준오차 분석
df_params['Std. Err.'] = pd.to_numeric(df_params['Std. Err.'], errors='coerce')
df_params['Estimate'] = pd.to_numeric(df_params['Estimate'], errors='coerce')
df_params['P. Value'] = pd.to_numeric(df_params['P. Value'], errors='coerce')

print('='*80)
print('표준오차 분포 분석')
print('='*80)
print(f'\n전체 파라미터 수: {len(df_params)}')
print(f'\n표준오차 통계:')
print(df_params['Std. Err.'].describe())

print(f'\n\n표준오차 범위별 파라미터 수:')
print(f'  SE < 0.1: {(df_params["Std. Err."] < 0.1).sum()}개')
print(f'  0.1 <= SE < 0.5: {((df_params["Std. Err."] >= 0.1) & (df_params["Std. Err."] < 0.5)).sum()}개')
print(f'  0.5 <= SE < 0.9: {((df_params["Std. Err."] >= 0.5) & (df_params["Std. Err."] < 0.9)).sum()}개')
print(f'  0.9 <= SE < 1.1: {((df_params["Std. Err."] >= 0.9) & (df_params["Std. Err."] < 1.1)).sum()}개')
print(f'  SE >= 1.1: {(df_params["Std. Err."] >= 1.1).sum()}개')

print(f'\n\n표준오차가 1.0 근처인 파라미터 (0.99 < SE < 1.02):')
se_near_one = df_params[(df_params['Std. Err.'] > 0.99) & (df_params['Std. Err.'] < 1.02)]
print(f'총 {len(se_near_one)}개')
for idx, row in se_near_one.iterrows():
    print(f'  {row["Coefficient"]:20s}: Estimate={row["Estimate"]:8.4f}, SE={row["Std. Err."]:.6f}, p={row["P. Value"]:.4f}')

print(f'\n\n표준오차가 작은 파라미터 (SE < 0.1):')
se_small = df_params[df_params['Std. Err.'] < 0.1]
print(f'총 {len(se_small)}개')
for idx, row in se_small.iterrows():
    print(f'  {row["Coefficient"]:20s}: Estimate={row["Estimate"]:8.4f}, SE={row["Std. Err."]:.6f}, p={row["P. Value"]:.4f}')

print(f'\n\n추정값 대비 표준오차 비율 (SE/|Estimate|):')
df_params['SE_ratio'] = df_params['Std. Err.'] / np.abs(df_params['Estimate'])
print(df_params['SE_ratio'].describe())

print(f'\n\nSE/|Estimate| > 0.5 인 파라미터 (불안정):')
unstable = df_params[df_params['SE_ratio'] > 0.5].sort_values('SE_ratio', ascending=False)
print(f'총 {len(unstable)}개')
for idx, row in unstable.head(15).iterrows():
    print(f'  {row["Coefficient"]:20s}: Estimate={row["Estimate"]:8.4f}, SE={row["Std. Err."]:.4f}, Ratio={row["SE_ratio"]:.2f}')

# 파라미터 그룹별 분석
print(f'\n\n{"="*80}')
print('파라미터 그룹별 표준오차 분석')
print('='*80)

# 측정모델 - zeta
zeta_params = df_params[df_params['Coefficient'].str.startswith('ζ_', na=False)]
print(f'\n[측정모델 - 요인적재량 ζ] (n={len(zeta_params)})')
print(f'  평균 SE: {zeta_params["Std. Err."].mean():.4f}')
print(f'  SE 범위: [{zeta_params["Std. Err."].min():.4f}, {zeta_params["Std. Err."].max():.4f}]')
print(f'  SE > 1.0인 개수: {(zeta_params["Std. Err."] > 1.0).sum()}개')

# 측정모델 - tau
tau_params = df_params[df_params['Coefficient'].str.startswith('τ_', na=False)]
print(f'\n[측정모델 - 임계값 τ] (n={len(tau_params)})')
print(f'  평균 SE: {tau_params["Std. Err."].mean():.4f}')
print(f'  SE 범위: [{tau_params["Std. Err."].min():.4f}, {tau_params["Std. Err."].max():.4f}]')
print(f'  SE > 1.0인 개수: {(tau_params["Std. Err."] > 1.0).sum()}개')

# 구조모델 - gamma
gamma_params = df_params[df_params['Coefficient'].str.startswith('γ_', na=False)]
print(f'\n[구조모델 - γ] (n={len(gamma_params)})')
print(f'  평균 SE: {gamma_params["Std. Err."].mean():.4f}')
print(f'  SE 범위: [{gamma_params["Std. Err."].min():.4f}, {gamma_params["Std. Err."].max():.4f}]')
print(f'  SE > 1.0인 개수: {(gamma_params["Std. Err."] > 1.0).sum()}개')
print('\n  상세:')
for idx, row in gamma_params.iterrows():
    print(f'    {row["Coefficient"]:20s}: Estimate={row["Estimate"]:8.4f}, SE={row["Std. Err."]:.6f}')

# 선택모델 - beta, lambda
choice_params = df_params[df_params['Coefficient'].str.startswith('β_', na=False) | (df_params['Coefficient'] == 'λ')]
print(f'\n[선택모델 - β, λ] (n={len(choice_params)})')
print(f'  평균 SE: {choice_params["Std. Err."].mean():.4f}')
print(f'  SE 범위: [{choice_params["Std. Err."].min():.4f}, {choice_params["Std. Err."].max():.4f}]')
print(f'  SE > 1.0인 개수: {(choice_params["Std. Err."] > 1.0).sum()}개')
print('\n  상세:')
for idx, row in choice_params.iterrows():
    print(f'    {row["Coefficient"]:20s}: Estimate={row["Estimate"]:8.4f}, SE={row["Std. Err."]:.6f}')

# 의심되는 문제 진단
print(f'\n\n{"="*80}')
print('진단 결과')
print('='*80)

# 1. SE가 정확히 1.0인 경우
exactly_one = df_params[np.abs(df_params['Std. Err.'] - 1.0) < 0.001]
if len(exactly_one) > 0:
    print(f'\n⚠️  표준오차가 정확히 1.0인 파라미터: {len(exactly_one)}개')
    print('   → Hessian 계산 실패 또는 초기값 문제 가능성')

# 2. SE가 0.99-1.02 범위인 경우
near_one = df_params[(df_params['Std. Err.'] > 0.99) & (df_params['Std. Err.'] < 1.02)]
if len(near_one) > len(df_params) * 0.5:
    print(f'\n⚠️  표준오차가 1.0 근처인 파라미터: {len(near_one)}개 ({len(near_one)/len(df_params)*100:.1f}%)')
    print('   → 모델 식별 문제(identification problem) 가능성')
    print('   → 파라미터 간 다중공선성 가능성')

# 3. 구조모델 SE가 모두 1.0 근처
if len(gamma_params) > 0 and (gamma_params['Std. Err.'] > 0.99).all():
    print(f'\n⚠️  구조모델 파라미터의 SE가 모두 1.0 근처')
    print('   → 사회인구학적 변수가 잠재변수를 설명하지 못함')
    print('   → 구조모델 제거 고려 필요')

# 4. 추정값이 매우 작은데 SE가 큰 경우
small_estimate_large_se = df_params[(np.abs(df_params['Estimate']) < 0.1) & (df_params['Std. Err.'] > 0.5)]
if len(small_estimate_large_se) > 0:
    print(f'\n⚠️  추정값이 작은데 SE가 큰 파라미터: {len(small_estimate_large_se)}개')
    print('   → 파라미터가 0에 가까움 (효과 없음)')
    for idx, row in small_estimate_large_se.iterrows():
        print(f'      {row["Coefficient"]:20s}: Estimate={row["Estimate"]:8.4f}, SE={row["Std. Err."]:.4f}')

print('\n' + '='*80)

