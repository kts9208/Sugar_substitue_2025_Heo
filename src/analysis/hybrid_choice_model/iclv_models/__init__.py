"""
ICLV (Integrated Choice and Latent Variable) Models Module

이 모듈은 King (2022) 논문의 ICLV 방법론을 Python으로 구현합니다.

핵심 기능:
1. Ordered Probit 측정모델
2. 잠재변수 구조방정식
3. Binary Probit 선택모델
4. 동시 추정 (Simultaneous Estimation)
5. Conditional/Unconditional WTP 계산
6. Halton Draws 시뮬레이션

참조:
- King, P. M. (2022). Willingness-to-pay for precautionary control of microplastics.
  Journal of Environmental Economics and Policy.
  https://doi.org/10.1080/21606544.2022.2146757
- Apollo R package: http://www.apollochoicemodelling.com/

Author: Sugar Substitute Research Team
Date: 2025-11-03
Version: 1.0.0
"""

from .iclv_config import (
    ICLVConfig,
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig,
    create_iclv_config
)

# 아직 구현되지 않은 모듈 (주석 처리)
# from .iclv_analyzer import (
#     ICLVAnalyzer,
#     ICLVResults,
#     run_iclv_analysis
# )

from .measurement_equations import (
    OrderedProbitMeasurement,
    estimate_measurement_model
)

from .structural_equations import (
    LatentVariableRegression,
    estimate_structural_model
)

from .choice_equations import (
    BinaryProbitChoice,
    estimate_choice_model
)

from .simultaneous_estimator import (
    SimultaneousEstimator,
    HaltonDrawGenerator,
    estimate_iclv_simultaneous
)

# 다중 잠재변수 ICLV
from .multi_latent_config import (
    MultiLatentConfig,
    MultiLatentStructuralConfig,
    create_default_multi_lv_config
)

from .multi_latent_measurement import (
    MultiLatentMeasurement
)

from .multi_latent_structural import (
    MultiLatentStructural
)

from .multi_latent_estimator import (
    MultiLatentSimultaneousEstimator
)

# 아직 구현되지 않은 모듈 (주석 처리)
# from .wtp_calculator import (
#     WTPCalculator,
#     calculate_conditional_wtp,
#     calculate_unconditional_wtp
# )

__all__ = [
    # 설정
    'ICLVConfig',
    'MeasurementConfig',
    'StructuralConfig',
    'ChoiceConfig',
    'create_iclv_config',

    # 메인 분석기 (아직 구현 안됨)
    # 'ICLVAnalyzer',
    # 'ICLVResults',
    # 'run_iclv_analysis',

    # 측정모델
    'OrderedProbitMeasurement',
    'estimate_measurement_model',

    # 구조모델
    'LatentVariableRegression',
    'estimate_structural_model',

    # 선택모델
    'BinaryProbitChoice',
    'estimate_choice_model',

    # 동시 추정
    'SimultaneousEstimator',
    'HaltonDrawGenerator',
    'estimate_iclv_simultaneous',

    # 다중 잠재변수 ICLV
    'MultiLatentConfig',
    'MultiLatentStructuralConfig',
    'create_default_multi_lv_config',
    'MultiLatentMeasurement',
    'MultiLatentStructural',
    'MultiLatentSimultaneousEstimator',

    # WTP 계산 (아직 구현 안됨)
    # 'WTPCalculator',
    # 'calculate_conditional_wtp',
    # 'calculate_unconditional_wtp',
]

__version__ = '1.0.0'
__author__ = 'Sugar Substitute Research Team'

