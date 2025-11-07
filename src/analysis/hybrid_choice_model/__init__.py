"""
Hybrid Choice Model Module

이 모듈은 DCE(Discrete Choice Experiment)와 SEM(Structural Equation Modeling)을 
결합한 하이브리드 선택 모델 분석 기능을 제공합니다.

주요 특징:
1. 다양한 선택모델 지원 (MNL, RPL, Mixed Logit, Nested Logit, Probit)
2. 잠재변수와 관측변수의 통합 모델링
3. 동시 추정을 통한 일관된 모수 추정
4. 확장 가능한 모듈화 설계
5. 기존 DCE, SEM, 효용함수 모듈과의 완벽한 통합

핵심 구성요소:
- ChoiceModelFactory: 다양한 선택모델 생성
- HybridChoiceAnalyzer: 통합 분석 엔진
- DataIntegration: DCE-SEM 데이터 통합
- LatentVariableModeling: 잠재변수 모델링
- UtilityFunctionModeling: 하이브리드 효용함수 구축
- SimultaneousEstimation: 동시 추정 엔진
- ResultsAnalysis: 결과 분석 및 비교

Author: Sugar Substitute Research Team
Date: 2025-09-19
Version: 1.0.0
"""

from .config.hybrid_config import (
    HybridConfig,
    ChoiceModelConfig,
    EstimationConfig,
    create_default_config,
    create_custom_config
)

from .choice_models.choice_model_factory import (
    ChoiceModelFactory,
    create_choice_model,
    get_available_models,
    get_model_info
)

from .choice_models.base_choice_model import (
    BaseChoiceModel,
    ChoiceModelType
)

from .data_integration.hybrid_data_integrator import (
    HybridDataIntegrator,
    integrate_dce_sem_data,
    validate_hybrid_data
)

# 아직 구현되지 않은 모듈들 (주석 처리)
# from .latent_variable_modeling.measurement_model import (
#     MeasurementModel,
#     estimate_measurement_model
# )

# from .latent_variable_modeling.factor_score_calculator import (
#     FactorScoreCalculator,
#     calculate_factor_scores
# )

# from .utility_function_modeling.hybrid_utility_builder import (
#     HybridUtilityBuilder,
#     build_hybrid_utility
# )

# from .simultaneous_estimation.hybrid_estimator import (
#     HybridEstimator,
#     estimate_hybrid_model
# )

# from .results.hybrid_results_analyzer import (
#     HybridResultsAnalyzer,
#     analyze_hybrid_results
# )

from .main_analyzer import (
    HybridChoiceAnalyzer,
    run_hybrid_analysis,
    run_model_comparison
)

# 버전 정보
__version__ = "1.0.0"
__author__ = "Sugar Substitute Research Team"

# 지원되는 선택모델 타입
SUPPORTED_CHOICE_MODELS = [
    "multinomial_logit",      # MNL
    "random_parameters_logit", # RPL
    "mixed_logit",            # Mixed Logit
    "nested_logit",           # Nested Logit
    "multinomial_probit"      # Probit
]

# 기본 설정
DEFAULT_CONFIG = {
    "choice_model_type": "multinomial_logit",
    "estimation_method": "maximum_likelihood",
    "convergence_tolerance": 1e-6,
    "max_iterations": 1000,
    "random_parameters": [],
    "nesting_structure": None,
    "simulation_draws": 1000
}

# 편의 함수들
def quick_hybrid_analysis(dce_data, sem_data, choice_model_type="multinomial_logit", **kwargs):
    """
    빠른 하이브리드 분석 실행
    
    Args:
        dce_data: DCE 데이터
        sem_data: SEM 데이터  
        choice_model_type: 선택모델 타입
        **kwargs: 추가 설정
        
    Returns:
        분석 결과 딕셔너리
    """
    config = create_default_config()
    config.choice_model_type = choice_model_type
    
    # 추가 설정 적용
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    analyzer = HybridChoiceAnalyzer(config)
    return analyzer.run_full_analysis(dce_data, sem_data)

def compare_choice_models(dce_data, sem_data, model_types=None, **kwargs):
    """
    여러 선택모델 비교 분석
    
    Args:
        dce_data: DCE 데이터
        sem_data: SEM 데이터
        model_types: 비교할 모델 타입 리스트
        **kwargs: 추가 설정
        
    Returns:
        모델 비교 결과
    """
    if model_types is None:
        model_types = ["multinomial_logit", "random_parameters_logit", "mixed_logit"]
    
    results = {}
    for model_type in model_types:
        try:
            result = quick_hybrid_analysis(dce_data, sem_data, model_type, **kwargs)
            results[model_type] = result
        except Exception as e:
            results[model_type] = {"error": str(e)}
    
    return results

# 모듈 정보 출력
def get_module_info():
    """모듈 정보 반환"""
    return {
        "name": "Hybrid Choice Model",
        "version": __version__,
        "author": __author__,
        "supported_models": SUPPORTED_CHOICE_MODELS,
        "description": "DCE와 SEM을 결합한 하이브리드 선택 모델 분석 모듈"
    }

# 모든 공개 API
__all__ = [
    # 설정
    "HybridConfig", "ChoiceModelConfig", "EstimationConfig",
    "create_default_config", "create_custom_config",
    
    # 선택모델
    "ChoiceModelFactory", "BaseChoiceModel", "ChoiceModelType",
    "create_choice_model", "get_available_models", "get_model_info",
    
    # 데이터 통합
    "HybridDataIntegrator", "integrate_dce_sem_data", "validate_hybrid_data",
    
    # 잠재변수 모델링 (아직 구현 안됨)
    # "MeasurementModel", "FactorScoreCalculator",
    # "estimate_measurement_model", "calculate_factor_scores",

    # 효용함수 모델링 (아직 구현 안됨)
    # "HybridUtilityBuilder", "build_hybrid_utility",

    # 추정 (아직 구현 안됨)
    # "HybridEstimator", "estimate_hybrid_model",

    # 결과 분석 (아직 구현 안됨)
    # "HybridResultsAnalyzer", "analyze_hybrid_results",
    
    # 메인 분석기
    "HybridChoiceAnalyzer", "run_hybrid_analysis", "run_model_comparison",
    
    # 편의 함수
    "quick_hybrid_analysis", "compare_choice_models", "get_module_info",
    
    # 상수
    "SUPPORTED_CHOICE_MODELS", "DEFAULT_CONFIG"
]
