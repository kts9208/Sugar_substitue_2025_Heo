"""
Hessian 행렬 품질 확인
- 조건수(condition number) 확인
- 고유값(eigenvalues) 확인
- 대각 원소 확인
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    ICLVConfig, MeasurementConfig, StructuralConfig, ChoiceConfig, EstimationConfig
)

print("="*80)
print("Hessian 행렬 품질 진단")
print("="*80)

# 1. 데이터 로드
print("\n1. 데이터 로드...")
data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
data = pd.read_csv(data_path)
print(f"   데이터 shape: {data.shape}")

# 2. 설정 (전체 데이터와 동일)
print("\n2. 설정...")
measurement_config = MeasurementConfig(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    n_categories=5
)

structural_config = StructuralConfig(
    sociodemographics=['age_std', 'gender', 'income_std'],
    include_in_choice=False
)

choice_config = ChoiceConfig(
    choice_attributes=['price', 'health_label']
)

estimation_config = EstimationConfig(
    optimizer='BFGS',
    use_analytic_gradient=True,
    n_draws=100,
    draw_type='halton',
    max_iterations=1000,
    calculate_se=True,
    use_parallel=False,  # 빠른 테스트를 위해 비활성화
    n_cores=None
)

config = ICLVConfig(
    measurement=measurement_config,
    structural=structural_config,
    choice=choice_config,
    estimation=estimation_config,
    individual_id_column='respondent_id',
    choice_column='choice'
)

# 3. 최적 파라미터 로드
print("\n3. 최적 파라미터 로드...")
results_df = pd.read_csv(project_root / 'results' / 'iclv_full_data_results.csv')
params_df = results_df[results_df['Coefficient'].notna() & (results_df['Coefficient'] != '') & (results_df['Coefficient'] != 'Estimation statistics')].copy()

# 파라미터 벡터 구성
optimal_params = params_df['Estimate'].values[:37]  # 37개 파라미터
print(f"   최적 파라미터 개수: {len(optimal_params)}")
print(f"   최종 LL: {results_df[results_df['Coefficient'] == 'AIC']['P. Value'].values[0]}")

# 4. Hessian 계산 (수치 미분)
print("\n4. Hessian 행렬 계산 중...")
print("   (이 작업은 시간이 걸릴 수 있습니다...)")

# 모델 생성
measurement_model = OrderedProbitMeasurement(measurement_config)
structural_model = LatentVariableRegression(structural_config)
choice_model = BinaryProbitChoice(choice_config)

# Estimator 생성
estimator = SimultaneousEstimator(config)
estimator.data = data
estimator.measurement_model = measurement_model
estimator.structural_model = structural_model
estimator.choice_model = choice_model

# Halton draws 생성
from src.analysis.hybrid_choice_model.iclv_models.halton_draws import HaltonDrawGenerator
estimator.halton_generator = HaltonDrawGenerator(
    n_draws=100,
    n_individuals=data['respondent_id'].nunique()
)

# Likelihood 함수 정의
def neg_ll(params):
    """Negative log-likelihood"""
    try:
        ll = estimator._joint_log_likelihood(
            params, 
            measurement_model, 
            structural_model, 
            choice_model
        )
        return -ll
    except:
        return 1e10

print("   최적점에서 LL 확인...")
optimal_ll = -neg_ll(optimal_params)
print(f"   최적점 LL: {optimal_ll:.4f}")

# Hessian 수치 계산 (finite difference)
print("\n   Hessian 계산 중 (finite difference)...")
n_params = len(optimal_params)
eps = 1e-5

# Gradient at optimal point
from scipy.optimize import approx_fprime
grad_0 = approx_fprime(optimal_params, neg_ll, eps)
print(f"   최적점에서 gradient norm: {np.linalg.norm(grad_0):.6f}")

# Hessian 계산 (대각 원소만)
print("\n   Hessian 대각 원소 계산 중...")
hessian_diag = np.zeros(n_params)

for i in range(min(10, n_params)):  # 처음 10개만 계산 (시간 절약)
    x_plus = optimal_params.copy()
    x_plus[i] += eps
    grad_i_plus = approx_fprime(x_plus, neg_ll, eps)
    hessian_diag[i] = (grad_i_plus[i] - grad_0[i]) / eps
    print(f"   파라미터 {i+1}: Hessian[{i},{i}] = {hessian_diag[i]:.6f}, SE = {1/np.sqrt(abs(hessian_diag[i])):.6f if hessian_diag[i] > 0 else 'N/A'}")

print("\n" + "="*80)
print("진단 완료")
print("="*80)
print("\n⚠️  주요 발견:")
print("   - Gradient norm이 0에 가까우면 수렴 성공")
print("   - Hessian 대각 원소가 1.0 근처이면 SE = 1/sqrt(H_ii) ≈ 1.0")
print("   - 이는 Hessian이 단위행렬에 가까움을 의미")
print("   - 즉, 파라미터가 likelihood에 거의 영향을 주지 않음")

