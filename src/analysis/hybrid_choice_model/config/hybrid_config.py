"""
Hybrid Choice Model Configuration

하이브리드 선택 모델의 설정을 관리하는 모듈입니다.
다양한 선택모델 타입과 추정 방법에 대한 설정을 제공합니다.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ChoiceModelType(Enum):
    """지원되는 선택모델 타입"""
    MULTINOMIAL_LOGIT = "multinomial_logit"
    RANDOM_PARAMETERS_LOGIT = "random_parameters_logit"
    MIXED_LOGIT = "mixed_logit"
    NESTED_LOGIT = "nested_logit"
    MULTINOMIAL_PROBIT = "multinomial_probit"


class EstimationMethod(Enum):
    """추정 방법"""
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    SIMULATED_MAXIMUM_LIKELIHOOD = "simulated_maximum_likelihood"
    BAYESIAN = "bayesian"
    METHOD_OF_MOMENTS = "method_of_moments"


class RandomParameterDistribution(Enum):
    """확률모수 분포"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    BETA = "beta"


@dataclass
class ChoiceModelConfig:
    """선택모델 설정"""
    model_type: ChoiceModelType = ChoiceModelType.MULTINOMIAL_LOGIT
    
    # 확률모수 로짓 설정
    random_parameters: List[str] = field(default_factory=list)
    random_parameter_distributions: Dict[str, RandomParameterDistribution] = field(default_factory=dict)
    
    # 중첩로짓 설정
    nesting_structure: Optional[Dict[str, List[str]]] = None
    nest_parameters: Dict[str, float] = field(default_factory=dict)
    
    # 혼합로짓 설정
    mixing_variables: List[str] = field(default_factory=list)
    class_probability_variables: List[str] = field(default_factory=list)
    
    # 프로빗 설정
    correlation_structure: str = "unstructured"  # "unstructured", "diagonal", "factor"
    
    # 일반 설정
    include_constants: bool = True
    reference_alternative: Optional[str] = None
    scale_parameter: float = 1.0


@dataclass
class EstimationConfig:
    """추정 설정"""
    method: EstimationMethod = EstimationMethod.MAXIMUM_LIKELIHOOD
    
    # 최적화 설정
    optimizer: str = "BFGS"  # "BFGS", "L-BFGS-B", "Newton-CG", "trust-ncg"
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-5
    
    # 시뮬레이션 설정 (RPL, Mixed Logit용)
    simulation_draws: int = 1000
    halton_draws: bool = True
    seed: Optional[int] = 42
    
    # 수치적 안정성
    hessian_approximation: str = "BFGS"  # "BFGS", "exact", "finite_difference"
    step_size: float = 1e-8
    
    # 병렬 처리
    n_cores: int = 1
    parallel_processing: bool = False
    
    # 로버스트 표준오차
    robust_standard_errors: bool = True
    bootstrap_iterations: int = 0


@dataclass
class DataConfig:
    """데이터 설정"""
    # DCE 데이터 설정
    choice_column: str = "choice"
    alternative_column: str = "alternative"
    individual_column: str = "individual_id"
    choice_set_column: str = "choice_set"
    
    # SEM 데이터 설정
    latent_variables: List[str] = field(default_factory=list)
    observed_variables: Dict[str, List[str]] = field(default_factory=dict)
    
    # 데이터 전처리
    standardize_variables: bool = True
    handle_missing_data: str = "listwise"  # "listwise", "pairwise", "imputation"
    outlier_detection: bool = True
    outlier_threshold: float = 3.0


@dataclass
class HybridConfig:
    """하이브리드 선택 모델 통합 설정"""
    
    # 하위 설정들
    choice_model: ChoiceModelConfig = field(default_factory=ChoiceModelConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 하이브리드 모델 특화 설정
    simultaneous_estimation: bool = True
    measurement_model_first: bool = True
    
    # 결과 설정
    save_results: bool = True
    results_directory: str = "results/current/hybrid_choice_model"
    detailed_output: bool = True
    
    # 시각화 설정
    create_visualizations: bool = True
    save_plots: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"
    
    # 로깅 설정
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """설정 검증"""
        self._validate_config()
    
    def _validate_config(self):
        """설정 유효성 검사"""
        # 선택모델 타입별 설정 검증
        if self.choice_model.model_type == ChoiceModelType.RANDOM_PARAMETERS_LOGIT:
            if not self.choice_model.random_parameters:
                logger.warning("RPL 모델이지만 확률모수가 지정되지 않았습니다.")
        
        if self.choice_model.model_type == ChoiceModelType.NESTED_LOGIT:
            if not self.choice_model.nesting_structure:
                raise ValueError("중첩로짓 모델에는 nesting_structure가 필요합니다.")
        
        # 추정 방법과 모델 타입 호환성 검사
        if (self.choice_model.model_type in [ChoiceModelType.RANDOM_PARAMETERS_LOGIT, 
                                           ChoiceModelType.MIXED_LOGIT] and
            self.estimation.method == EstimationMethod.MAXIMUM_LIKELIHOOD):
            logger.warning("RPL/Mixed Logit 모델에는 시뮬레이션 기반 추정이 권장됩니다.")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridConfig':
        """딕셔너리에서 설정 생성"""
        # Enum 변환
        if 'choice_model' in config_dict:
            cm = config_dict['choice_model']
            if 'model_type' in cm and isinstance(cm['model_type'], str):
                cm['model_type'] = ChoiceModelType(cm['model_type'])
            if 'random_parameter_distributions' in cm:
                cm['random_parameter_distributions'] = {
                    k: RandomParameterDistribution(v) if isinstance(v, str) else v
                    for k, v in cm['random_parameter_distributions'].items()
                }
        
        if 'estimation' in config_dict:
            est = config_dict['estimation']
            if 'method' in est and isinstance(est['method'], str):
                est['method'] = EstimationMethod(est['method'])
        
        return cls(
            choice_model=ChoiceModelConfig(**config_dict.get('choice_model', {})),
            estimation=EstimationConfig(**config_dict.get('estimation', {})),
            data=DataConfig(**config_dict.get('data', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['choice_model', 'estimation', 'data']}
        )


def create_default_config() -> HybridConfig:
    """기본 설정 생성"""
    return HybridConfig()


def create_custom_config(
    choice_model_type: Union[str, ChoiceModelType] = "multinomial_logit",
    estimation_method: Union[str, EstimationMethod] = "maximum_likelihood",
    random_parameters: Optional[List[str]] = None,
    **kwargs
) -> HybridConfig:
    """커스텀 설정 생성"""
    
    # 문자열을 Enum으로 변환
    if isinstance(choice_model_type, str):
        choice_model_type = ChoiceModelType(choice_model_type)
    if isinstance(estimation_method, str):
        estimation_method = EstimationMethod(estimation_method)
    
    config = create_default_config()
    config.choice_model.model_type = choice_model_type
    config.estimation.method = estimation_method
    
    if random_parameters:
        config.choice_model.random_parameters = random_parameters
    
    # 추가 설정 적용
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.choice_model, key):
            setattr(config.choice_model, key, value)
        elif hasattr(config.estimation, key):
            setattr(config.estimation, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
    
    return config


def load_config_from_file(file_path: Union[str, Path]) -> HybridConfig:
    """파일에서 설정 로드"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return HybridConfig.from_dict(config_dict)


def save_config_to_file(config: HybridConfig, file_path: Union[str, Path]):
    """설정을 파일에 저장"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Enum을 문자열로 변환하여 저장
    config_dict = config.to_dict()
    
    def convert_enums(obj):
        if isinstance(obj, dict):
            return {k: convert_enums(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_enums(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj
    
    config_dict = convert_enums(config_dict)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"설정이 저장되었습니다: {file_path}")


# 편의 함수들
def create_mnl_config(**kwargs) -> HybridConfig:
    """MNL 모델 설정 생성"""
    return create_custom_config(
        choice_model_type="multinomial_logit",
        estimation_method="maximum_likelihood",
        **kwargs
    )


def create_rpl_config(random_parameters: List[str], **kwargs) -> HybridConfig:
    """RPL 모델 설정 생성"""
    return create_custom_config(
        choice_model_type="random_parameters_logit",
        estimation_method="simulated_maximum_likelihood",
        random_parameters=random_parameters,
        **kwargs
    )


def create_mixed_logit_config(mixing_variables: List[str], **kwargs) -> HybridConfig:
    """Mixed Logit 모델 설정 생성"""
    config = create_custom_config(
        choice_model_type="mixed_logit",
        estimation_method="simulated_maximum_likelihood",
        **kwargs
    )
    config.choice_model.mixing_variables = mixing_variables
    return config


def create_nested_logit_config(nesting_structure: Dict[str, List[str]], **kwargs) -> HybridConfig:
    """Nested Logit 모델 설정 생성"""
    config = create_custom_config(
        choice_model_type="nested_logit",
        estimation_method="maximum_likelihood",
        **kwargs
    )
    config.choice_model.nesting_structure = nesting_structure
    return config
