"""
측정모델 우도 분석

1. 실제 사용되는 측정 방법 확인 (Ordered Probit vs Continuous Linear)
2. 지표당 평균 우도가 큰 이유 분석
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path

print("="*80)
print("측정모델 우도 분석")
print("="*80)

# 데이터 로드
data_path = Path('data/processed/iclv/integrated_data.csv')
data = pd.read_csv(data_path)

# 첫 번째 개인 데이터
person_1 = data[data['respondent_id'] == 1].iloc[0]

# 측정 지표 (5개 잠재변수, 38개 지표)
indicators = {
    'health_concern': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    'perceived_benefit': ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
    'perceived_price': ['q27', 'q28', 'q29'],
    'nutrition_knowledge': [f'q{i}' for i in range(30, 50)],
    'purchase_intention': ['q18', 'q19', 'q20']
}

print(f"\n[1] 데이터 확인")
print(f"{'='*80}")
print(f"개인 1의 지표 값 (처음 10개):")
all_indicators = []
for lv_name, inds in indicators.items():
    all_indicators.extend(inds)

for i, ind in enumerate(all_indicators[:10], 1):
    value = person_1[ind]
    print(f"  {i:2d}. {ind}: {value}")

# ============================================================================
# 2. Continuous Linear 우도 계산
# ============================================================================
print(f"\n{'='*80}")
print(f"[2] Continuous Linear 우도 계산")
print(f"{'='*80}")

# 가정: LV = 0.5, ζ = 1.0, σ² = 1.0
lv = 0.5
zeta = 1.0
sigma_sq = 1.0

# 예시 지표 값
y_obs = 4.0  # 관측값

# 예측값
y_pred = zeta * lv
residual = y_obs - y_pred

# 로그우도: log N(y | ζ*LV, σ²)
ll_continuous = -0.5 * np.log(2 * np.pi * sigma_sq) - 0.5 * (residual**2) / sigma_sq

print(f"\n파라미터:")
print(f"  LV: {lv}")
print(f"  ζ (zeta): {zeta}")
print(f"  σ² (sigma_sq): {sigma_sq}")
print(f"  y_obs: {y_obs}")

print(f"\n계산:")
print(f"  y_pred = ζ × LV = {zeta} × {lv} = {y_pred:.4f}")
print(f"  residual = y_obs - y_pred = {y_obs} - {y_pred:.4f} = {residual:.4f}")
print(f"  ll = -0.5×log(2π×{sigma_sq}) - 0.5×({residual:.4f})²/{sigma_sq}")
print(f"     = {ll_continuous:.4f}")

# ============================================================================
# 3. Ordered Probit 우도 계산 (비교)
# ============================================================================
print(f"\n{'='*80}")
print(f"[3] Ordered Probit 우도 계산 (비교)")
print(f"{'='*80}")

# 가정: 5점 척도
tau = np.array([-2.0, -1.0, 0.0, 1.0])  # 4개 임계값
y_obs_int = int(y_obs)  # 4

# 선형 예측
V = zeta * lv

# 확률 계산
k = y_obs_int - 1  # 0-based index
upper = tau[k] - V
lower = tau[k-1] - V
prob = norm.cdf(upper) - norm.cdf(lower)

ll_ordered = np.log(prob)

print(f"\n파라미터:")
print(f"  LV: {lv}")
print(f"  ζ (zeta): {zeta}")
print(f"  τ (tau): {tau}")
print(f"  y_obs: {y_obs_int}")

print(f"\n계산:")
print(f"  V = ζ × LV = {zeta} × {lv} = {V:.4f}")
print(f"  upper = τ_{k} - V = {tau[k]:.4f} - {V:.4f} = {upper:.4f}")
print(f"  lower = τ_{k-1} - V = {tau[k-1]:.4f} - {V:.4f} = {lower:.4f}")
print(f"  P(Y={y_obs_int}) = Φ({upper:.4f}) - Φ({lower:.4f}) = {prob:.6f}")
print(f"  ll = log({prob:.6f}) = {ll_ordered:.4f}")

# ============================================================================
# 4. 비교
# ============================================================================
print(f"\n{'='*80}")
print(f"[4] 비교")
print(f"{'='*80}")

print(f"\n지표당 로그우도:")
print(f"  Continuous Linear: {ll_continuous:.4f}")
print(f"  Ordered Probit:    {ll_ordered:.4f}")
print(f"  차이:              {abs(ll_continuous - ll_ordered):.4f}")

print(f"\n38개 지표 전체 (모두 동일 가정):")
print(f"  Continuous Linear: {ll_continuous * 38:.4f}")
print(f"  Ordered Probit:    {ll_ordered * 38:.4f}")

# ============================================================================
# 5. 왜 Continuous Linear 우도가 더 큰가?
# ============================================================================
print(f"\n{'='*80}")
print(f"[5] 왜 Continuous Linear 우도가 더 큰가?")
print(f"{'='*80}")

print(f"""
**핵심 차이**:

1. **Continuous Linear**:
   - 모델: Y = ζ*LV + ε, ε ~ N(0, σ²)
   - 우도: log N(y | ζ*LV, σ²)
   - 잔차가 작으면 우도가 높음 (0에 가까움)
   - 잔차가 크면 우도가 낮음 (큰 음수)
   - **σ²가 작을수록 우도가 낮아짐** (분산이 작으면 데이터에 더 엄격)

2. **Ordered Probit**:
   - 모델: P(Y=k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
   - 우도: log P(Y=k)
   - 5점 척도 중 1개 선택 → 확률 ~0.2-0.3
   - log(0.2) ≈ -1.6 (항상 큰 음수)

3. **실제 값 비교** (위 예시):
   - Continuous: {ll_continuous:.4f}
   - Ordered:    {ll_ordered:.4f}
   - Continuous가 더 큼 (덜 negative)

4. **왜 Continuous가 더 큰가?**
   - Continuous는 **잔차 크기**에 따라 우도가 변함
   - 잔차가 작으면 (모델이 데이터를 잘 설명) → 우도 높음
   - Ordered는 **확률**에 따라 우도가 변함
   - 5점 척도 → 확률 최대 ~0.4 → log(0.4) ≈ -0.9 (항상 음수)

5. **σ²의 영향**:
   - σ² = 1.0: ll ≈ {ll_continuous:.4f}
   - σ² = 0.1: ll ≈ {-0.5 * np.log(2 * np.pi * 0.1) - 0.5 * (residual**2) / 0.1:.4f}
   - σ² = 10.0: ll ≈ {-0.5 * np.log(2 * np.pi * 10.0) - 0.5 * (residual**2) / 10.0:.4f}
   - **σ²가 작을수록 우도가 낮아짐!**
""")

# ============================================================================
# 6. CFA 결과에서 σ² 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[6] CFA 결과에서 σ² 확인")
print(f"{'='*80}")

import pickle

cfa_path = Path('results/sequential_stage_wise/cfa_results.pkl')
if cfa_path.exists():
    with open(cfa_path, 'rb') as f:
        cfa_results = pickle.load(f)
    
    print(f"\nCFA 결과 로드 성공")
    print(f"CFA 결과 키: {list(cfa_results.keys())}")

    # 측정모델 파라미터 확인
    if 'measurement' in cfa_results:
        meas_params = cfa_results['measurement']
        print(f"측정모델 LV: {list(meas_params.keys())}")

        for lv_name in ['health_concern', 'perceived_benefit', 'perceived_price',
                        'nutrition_knowledge', 'purchase_intention']:
            if lv_name in meas_params:
                params = meas_params[lv_name]
                print(f"\n{lv_name}:")
                print(f"  파라미터 키: {list(params.keys())}")

                if 'sigma_sq' in params:
                    sigma_sq_vals = params['sigma_sq']
                    print(f"  σ² (처음 3개): {sigma_sq_vals[:3]}")
                    print(f"  σ² 평균: {np.mean(sigma_sq_vals):.4f}")
                    print(f"  σ² 범위: [{np.min(sigma_sq_vals):.4f}, {np.max(sigma_sq_vals):.4f}]")

                    # 실제 우도 계산 예시
                    lv_val = 0.5
                    zeta_val = params.get('zeta', np.ones(len(sigma_sq_vals)))[0]
                    y_obs = 4.0
                    y_pred = zeta_val * lv_val
                    residual = y_obs - y_pred
                    ll = -0.5 * np.log(2 * np.pi * sigma_sq_vals[0]) - 0.5 * (residual**2) / sigma_sq_vals[0]
                    print(f"  예시 우도 (첫 지표, LV={lv_val}, y={y_obs}): {ll:.4f}")
else:
    print(f"\nCFA 결과 파일 없음: {cfa_path}")

