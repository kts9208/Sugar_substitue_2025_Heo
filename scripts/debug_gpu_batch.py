"""GPU 배치 처리 디버깅"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    ChoiceConfig,
    EstimationConfig
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    MultiLatentStructuralConfig,
    MultiLatentConfig
)
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_gpu_batch_estimator import SimultaneousGPUBatchEstimator
from dataclasses import dataclass

@dataclass
class DataConfig:
    individual_id: str = 'respondent_id'
    choice_id: str = 'choice_set'

# 데이터 로드
data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
data = pd.read_csv(data_path)

# 간단한 설정 (1개 잠재변수만)
measurement_configs = {
    'health_concern': MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8'],  # 3개만
        n_categories=5
    )
}

structural_config = MultiLatentStructuralConfig(
    endogenous_lv='health_concern',
    exogenous_lvs=[],
    covariates=['age_std'],
    error_variance=1.0
)

choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price']
)

estimation_config = EstimationConfig(
    optimizer='BFGS',
    n_draws=10,  # 적은 draws
    draw_type='halton',
    max_iterations=5
)

config = MultiLatentConfig(
    measurement_configs=measurement_configs,
    structural=structural_config,
    choice=choice_config,
    estimation=estimation_config,
    individual_id_column='respondent_id',
    choice_column='choice'
)

config.data = DataConfig()

# Estimator 생성
estimator = SimultaneousGPUBatchEstimator(config, data, use_gpu=False)

# 초기 파라미터
init_params = estimator._initialize_parameters()
print(f"초기 파라미터 수: {len(init_params)}")
print(f"초기 파라미터: {init_params}")

# 언팩
param_dict = estimator._unpack_parameters(init_params)
print(f"\n언팩 후:")
print(f"  측정모델 zeta: {param_dict['measurement']['health_concern']['zeta']}")
print(f"  구조모델 gamma_lv: {param_dict['structural']['gamma_lv']}")
print(f"  구조모델 gamma_x: {param_dict['structural']['gamma_x']}")
print(f"  선택모델 intercept: {param_dict['choice']['intercept']}")
print(f"  선택모델 beta: {param_dict['choice']['beta']}")
print(f"  선택모델 lambda: {param_dict['choice']['lambda']}")

# 우도 계산
print(f"\n우도 계산 중...")
ll = estimator._compute_batch_likelihood(init_params)
print(f"초기 로그우도: {ll:.4f}")

# 파라미터 약간 변경
modified_params = init_params.copy()
modified_params[-5] = 0.1  # beta[0] 변경
print(f"\n파라미터 변경: beta[0] = 0 -> 0.1")

param_dict2 = estimator._unpack_parameters(modified_params)
print(f"  변경 후 beta: {param_dict2['choice']['beta']}")

ll2 = estimator._compute_batch_likelihood(modified_params)
print(f"변경 후 로그우도: {ll2:.4f}")
print(f"우도 변화: {ll2 - ll:.4f}")

