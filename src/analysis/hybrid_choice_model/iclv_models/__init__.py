"""
ICLV (Integrated Choice and Latent Variable) Models Module

이 모듈은 King (2022) 논문의 ICLV 방법론을 Python으로 구현합니다.

핵심 기능:
1. Ordered Probit 측정모델
2. 잠재변수 구조방정식
3. 동시 추정 (Simultaneous Estimation)
4. Conditional/Unconditional WTP 계산
5. Halton Draws 시뮬레이션

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
    create_iclv_config
)

from .iclv_analyzer import (
    ICLVAnalyzer,
    ICLVResults,
    run_iclv_analysis
)

from .measurement_equations import (
    OrderedProbitMeasurement,
    estimate_measurement_model
)

from .structural_equations import (
    LatentVariableRegression,
    estimate_structural_model
)

from .simultaneous_estimator import (
    SimultaneousEstimator,
    HaltonDrawGenerator,
    estimate_iclv_simultaneous
)

from .wtp_calculator import (
    WTPCalculator,
    calculate_conditional_wtp,
    calculate_unconditional_wtp
)

__all__ = [
    # 설정
    'ICLVConfig',
    'MeasurementConfig',
    'StructuralConfig',
    'create_iclv_config',
    
    # 메인 분석기
    'ICLVAnalyzer',
    'ICLVResults',
    'run_iclv_analysis',
    
    # 측정모델
    'OrderedProbitMeasurement',
    'estimate_measurement_model',
    
    # 구조모델
    'LatentVariableRegression',
    'estimate_structural_model',
    
    # 동시 추정
    'SimultaneousEstimator',
    'HaltonDrawGenerator',
    'estimate_iclv_simultaneous',
    
    # WTP 계산
    'WTPCalculator',
    'calculate_conditional_wtp',
    'calculate_unconditional_wtp',
]

__version__ = '1.0.0'
__author__ = 'Sugar Substitute Research Team'

