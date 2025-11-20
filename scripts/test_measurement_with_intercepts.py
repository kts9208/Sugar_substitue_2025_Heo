"""
측정모델 우도 계산 테스트 (절편 포함)
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.gpu_measurement_equations import GPUContinuousLinearMeasurement
from analysis.hybrid_choice_model.iclv_models.multi_latent_config import MeasurementConfig

print("="*80)
print("측정모델 우도 계산 테스트 (절편 포함)")
print("="*80)

# 1. CFA 결과 로드
cfa_path = project_root / 'results' / 'sequential_stage_wise' / 'cfa_results.pkl'
with open(cfa_path, 'rb') as f:
    cfa_results = pickle.load(f)

print(f"\nCFA 결과 로드 완료")
print(f"Keys: {list(cfa_results.keys())}")

# 2. health_concern 파라미터 추출
lv_name = 'health_concern'
indicators = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']

loadings_df = cfa_results['loadings']
errors_df = cfa_results['measurement_errors']
intercepts_df = cfa_results.get('intercepts')

# zeta
zeta_values = []
for indicator in indicators:
    row = loadings_df[(loadings_df['lval'] == indicator) &
                     (loadings_df['op'] == '~') &
                     (loadings_df['rval'] == lv_name)]
    zeta_values.append(float(row['Estimate'].iloc[0]))

# sigma_sq
sigma_sq_values = []
for indicator in indicators:
    row = errors_df[(errors_df['lval'] == indicator) &
                   (errors_df['op'] == '~~') &
                   (errors_df['rval'] == indicator)]
    sigma_sq_values.append(float(row['Estimate'].iloc[0]))

# alpha (절편)
alpha_values = []
if intercepts_df is not None:
    for indicator in indicators:
        row = intercepts_df[(intercepts_df['lval'] == indicator) &
                           (intercepts_df['op'] == '~') &
                           (intercepts_df['rval'] == '1')]
        alpha_values.append(float(row['Estimate'].iloc[0]))
else:
    alpha_values = [0.0] * len(indicators)

print(f"\n{lv_name} 파라미터:")
print(f"  zeta: {zeta_values}")
print(f"  sigma_sq: {sigma_sq_values}")
print(f"  alpha: {alpha_values}")

# 3. 측정모델 생성
config = MeasurementConfig(
    latent_variable=lv_name,
    indicators=indicators,
    measurement_method='continuous_linear'
)

model = GPUContinuousLinearMeasurement(config, use_gpu=False)  # CPU 모드로 테스트

# 4. 테스트 데이터 (q6 = 4.0)
test_data = pd.DataFrame({
    'q6': [4.0],
    'q7': [3.5],
    'q8': [3.5],
    'q9': [4.0],
    'q10': [4.0],
    'q11': [3.5]
})

# 5. 잠재변수 값
lv_value = 0.5

# 6. 우도 계산 (절편 없음)
params_no_intercept = {
    'zeta': np.array(zeta_values),
    'sigma_sq': np.array(sigma_sq_values)
}

ll_no_intercept = model.log_likelihood(test_data, lv_value, params_no_intercept)

print(f"\n{'='*80}")
print(f"우도 계산 결과 (LV = {lv_value})")
print(f"{'='*80}")

print(f"\n[1] 절편 없음:")
print(f"  로그우도: {ll_no_intercept:.4f}")
print(f"  지표당 평균: {ll_no_intercept / len(indicators):.4f}")

# 7. 우도 계산 (절편 포함)
params_with_intercept = {
    'zeta': np.array(zeta_values),
    'sigma_sq': np.array(sigma_sq_values),
    'alpha': np.array(alpha_values)
}

ll_with_intercept = model.log_likelihood(test_data, lv_value, params_with_intercept)

print(f"\n[2] 절편 포함:")
print(f"  로그우도: {ll_with_intercept:.4f}")
print(f"  지표당 평균: {ll_with_intercept / len(indicators):.4f}")

# 8. 비교
print(f"\n{'='*80}")
print(f"비교")
print(f"{'='*80}")

improvement = ll_with_intercept - ll_no_intercept
improvement_ratio = np.exp(improvement)

print(f"\n우도 개선:")
print(f"  차이: {improvement:.4f}")
print(f"  비율: {improvement_ratio:.2f}배")

# 9. 상세 분석 (q6)
print(f"\n{'='*80}")
print(f"상세 분석 (q6)")
print(f"{'='*80}")

i = 0  # q6
y_obs = test_data['q6'].iloc[0]
zeta = zeta_values[i]
sigma_sq = sigma_sq_values[i]
alpha = alpha_values[i]

print(f"\n파라미터:")
print(f"  y_obs: {y_obs:.4f}")
print(f"  LV: {lv_value:.4f}")
print(f"  ζ: {zeta:.4f}")
print(f"  σ²: {sigma_sq:.4f}")
print(f"  α: {alpha:.4f}")

# 절편 없음
y_pred_no = zeta * lv_value
residual_no = y_obs - y_pred_no
ll_no = -0.5 * np.log(2 * np.pi * sigma_sq) - 0.5 * (residual_no ** 2) / sigma_sq

print(f"\n절편 없음:")
print(f"  Y_pred = ζ * LV = {zeta:.4f} * {lv_value:.4f} = {y_pred_no:.4f}")
print(f"  residual = {y_obs:.4f} - {y_pred_no:.4f} = {residual_no:.4f}")
print(f"  ll = {ll_no:.4f}")

# 절편 포함
y_pred_with = alpha + zeta * lv_value
residual_with = y_obs - y_pred_with
ll_with = -0.5 * np.log(2 * np.pi * sigma_sq) - 0.5 * (residual_with ** 2) / sigma_sq

print(f"\n절편 포함:")
print(f"  Y_pred = α + ζ * LV = {alpha:.4f} + {zeta:.4f} * {lv_value:.4f} = {y_pred_with:.4f}")
print(f"  residual = {y_obs:.4f} - {y_pred_with:.4f} = {residual_with:.4f}")
print(f"  ll = {ll_with:.4f}")

print(f"\n개선: {ll_with - ll_no:.4f}")

print(f"\n{'='*80}")
print(f"테스트 완료!")
print(f"{'='*80}")

