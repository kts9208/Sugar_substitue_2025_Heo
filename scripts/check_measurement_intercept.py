"""
측정모델에 절편(intercept)이 있는지 확인
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("측정모델 절편(Intercept) 확인")
print("="*80)

# ============================================================================
# 1. CFA 결과에서 절편 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[1] CFA 결과에서 절편 확인")
print(f"{'='*80}")

cfa_path = Path('results/sequential_stage_wise/cfa_results.pkl')
with open(cfa_path, 'rb') as f:
    cfa_results = pickle.load(f)

loadings_df = cfa_results['loadings']

print(f"\n전체 파라미터 타입:")
print(loadings_df['op'].value_counts())

# 절편 확인 (op == '1' 또는 op == '~1')
intercepts = loadings_df[loadings_df['op'] == '1']

print(f"\n절편 (op == '1'):")
print(f"  개수: {len(intercepts)}")

if len(intercepts) > 0:
    print(f"\n처음 10개:")
    print(intercepts[['lval', 'op', 'rval', 'Estimate']].head(10))
    
    print(f"\n통계:")
    print(f"  평균: {intercepts['Estimate'].mean():.4f}")
    print(f"  표준편차: {intercepts['Estimate'].std():.4f}")
    print(f"  범위: [{intercepts['Estimate'].min():.4f}, {intercepts['Estimate'].max():.4f}]")
else:
    print(f"  ❌ 절편 없음!")

# ============================================================================
# 2. 측정모델 공식 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[2] 현재 측정모델 공식")
print(f"{'='*80}")

print(f"\n코드 (gpu_measurement_equations.py, line 577-578):")
print(f"  y_pred = zeta[i] * lv_gpu")
print(f"  residual = y_obs - y_pred")

print(f"\n공식:")
print(f"  Y_i = ζ_i * LV + ε_i")
print(f"  ε_i ~ N(0, σ²_i)")

print(f"\n❌ 절편 없음!")

# ============================================================================
# 3. 올바른 측정모델 공식
# ============================================================================
print(f"\n{'='*80}")
print(f"[3] 올바른 측정모델 공식")
print(f"{'='*80}")

print(f"\n일반적인 CFA 측정모델:")
print(f"  Y_i = α_i + ζ_i * LV + ε_i")
print(f"  여기서:")
print(f"    α_i: 절편 (intercept)")
print(f"    ζ_i: 요인적재량 (factor loading)")
print(f"    LV: 잠재변수 (latent variable)")
print(f"    ε_i ~ N(0, σ²_i): 측정오차")

print(f"\n✅ 절편 필요!")

# ============================================================================
# 4. 절편이 없을 때의 문제
# ============================================================================
print(f"\n{'='*80}")
print(f"[4] 절편이 없을 때의 문제")
print(f"{'='*80}")

print(f"\n현재 모델 (절편 없음):")
print(f"  Y_i = ζ_i * LV + ε_i")
print(f"  LV ~ N(0, 1) (표준화)")
print(f"  E[Y_i] = ζ_i * E[LV] = ζ_i * 0 = 0")
print(f"  ❌ 예측 평균이 0!")

print(f"\n실제 데이터:")
print(f"  지표(q6-q49): 1-5점 리커트")
print(f"  관측 평균: 3-4")
print(f"  ❌ 예측 평균(0) vs 관측 평균(3-4) → 큰 불일치!")

print(f"\n올바른 모델 (절편 있음):")
print(f"  Y_i = α_i + ζ_i * LV + ε_i")
print(f"  LV ~ N(0, 1) (표준화)")
print(f"  E[Y_i] = α_i + ζ_i * E[LV] = α_i + ζ_i * 0 = α_i")
print(f"  ✅ 예측 평균 = α_i (지표의 평균)")

# ============================================================================
# 5. 실제 계산 예시
# ============================================================================
print(f"\n{'='*80}")
print(f"[5] 실제 계산 예시 (지표 q6)")
print(f"{'='*80}")

# 데이터 로드
data = pd.read_csv('data/processed/iclv/integrated_data.csv')

# q6 통계
q6_mean = data['q6'].mean()
q6_std = data['q6'].std()

print(f"\nq6 통계:")
print(f"  평균: {q6_mean:.4f}")
print(f"  표준편차: {q6_std:.4f}")

# CFA 파라미터
loadings = loadings_df[loadings_df['op'] == '~']
q6_loading = loadings[loadings['lval'] == 'q6']['Estimate'].values[0]

errors = cfa_results['measurement_errors']
q6_error = errors[(errors['lval'] == 'q6') & (errors['rval'] == 'q6')]['Estimate'].values[0]

print(f"\nCFA 파라미터:")
print(f"  ζ (요인적재량): {q6_loading:.4f}")
print(f"  σ² (측정오차 분산): {q6_error:.4f}")

# 절편 확인
if len(intercepts) > 0:
    q6_intercept_rows = intercepts[intercepts['lval'] == 'q6']
    if len(q6_intercept_rows) > 0:
        q6_intercept = q6_intercept_rows['Estimate'].values[0]
        print(f"  α (절편): {q6_intercept:.4f}")
    else:
        print(f"  α (절편): 없음")
        q6_intercept = None
else:
    print(f"  α (절편): 없음")
    q6_intercept = None

# 예측 계산
print(f"\n예측 계산 (LV = 0.5):")

lv = 0.5

if q6_intercept is not None:
    y_pred_with_intercept = q6_intercept + q6_loading * lv
    print(f"  절편 있음: Y_pred = {q6_intercept:.4f} + {q6_loading:.4f} * {lv} = {y_pred_with_intercept:.4f}")
else:
    print(f"  절편 없음: Y_pred = {q6_loading:.4f} * {lv} = {q6_loading * lv:.4f}")

y_pred_no_intercept = q6_loading * lv

print(f"\n비교:")
print(f"  관측 평균: {q6_mean:.4f}")
if q6_intercept is not None:
    print(f"  예측 (절편 있음): {y_pred_with_intercept:.4f}")
    print(f"  잔차 (절편 있음): {q6_mean - y_pred_with_intercept:.4f}")
print(f"  예측 (절편 없음): {y_pred_no_intercept:.4f}")
print(f"  잔차 (절편 없음): {q6_mean - y_pred_no_intercept:.4f}")

# 우도 계산
print(f"\n우도 계산 (Y_obs = {q6_mean:.4f}):")

if q6_intercept is not None:
    residual_with = q6_mean - y_pred_with_intercept
    ll_with = -0.5 * np.log(2 * np.pi * q6_error) - 0.5 * (residual_with ** 2) / q6_error
    print(f"  절편 있음: ll = {ll_with:.4f}")

residual_no = q6_mean - y_pred_no_intercept
ll_no = -0.5 * np.log(2 * np.pi * q6_error) - 0.5 * (residual_no ** 2) / q6_error
print(f"  절편 없음: ll = {ll_no:.4f}")

if q6_intercept is not None:
    print(f"\n차이: {ll_with - ll_no:.4f}")

# ============================================================================
# 6. 결론
# ============================================================================
print(f"\n{'='*80}")
print(f"[6] 결론")
print(f"{'='*80}")

if len(intercepts) == 0:
    print(f"\n❌ **현재 측정모델에 절편이 없습니다!**")
    print(f"\n문제점:")
    print(f"  1. LV ~ N(0, 1) → E[Y] = 0 (예측 평균이 0)")
    print(f"  2. 실제 지표 평균: 3-4 (1-5점 리커트)")
    print(f"  3. 큰 잔차 → 매우 낮은 우도")
    
    print(f"\n해결 방안:")
    print(f"  1. CFA에서 절편 추정")
    print(f"  2. 측정모델 우도 계산시 절편 포함")
    print(f"     Y_i = α_i + ζ_i * LV + ε_i")
else:
    print(f"\n✅ CFA에 절편이 있습니다.")
    print(f"  개수: {len(intercepts)}")
    print(f"\n하지만 측정모델 우도 계산시 절편을 사용하지 않습니다!")
    print(f"  현재 코드: y_pred = zeta[i] * lv_gpu")
    print(f"  올바른 코드: y_pred = alpha[i] + zeta[i] * lv_gpu")

print(f"\n{'='*80}")

