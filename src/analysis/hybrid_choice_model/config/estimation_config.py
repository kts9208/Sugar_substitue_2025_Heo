"""
Estimation Configuration

하이브리드 선택 모델의 추정 설정을 관리합니다.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class EstimationMethod(Enum):
    """추정 방법"""
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    SIMULATED_MAXIMUM_LIKELIHOOD = "simulated_maximum_likelihood"
    BAYESIAN = "bayesian"
    METHOD_OF_MOMENTS = "method_of_moments"


class OptimizationAlgorithm(Enum):
    """최적화 알고리즘"""
    BFGS = "BFGS"
    L_BFGS_B = "L-BFGS-B"
    NEWTON_CG = "Newton-CG"
    TRUST_NCG = "trust-ncg"
    SLSQP = "SLSQP"
    NELDER_MEAD = "Nelder-Mead"


@dataclass
class EstimationConfig:
    """추정 설정 클래스"""
    
    # 추정 방법
    method: EstimationMethod = EstimationMethod.MAXIMUM_LIKELIHOOD
    optimizer: OptimizationAlgorithm = OptimizationAlgorithm.BFGS
    
    # 수렴 기준
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-5
    
    # 시뮬레이션 설정 (SML용)
    simulation_draws: int = 1000
    halton_draws: bool = True
    antithetic_draws: bool = False
    
    # 초기값 설정
    use_random_starts: bool = False
    n_random_starts: int = 5
    initial_parameter_bounds: Dict[str, tuple] = None
    
    # 수치적 안정성
    numerical_precision: float = 1e-10
    step_size: float = 1e-5
    finite_difference_step: float = 1e-8
    
    # 병렬 처리
    parallel_processing: bool = False
    n_cores: Optional[int] = None
    
    # 로버스트 추정
    robust_standard_errors: bool = True
    bootstrap_samples: int = 0
    
    def __post_init__(self):
        """설정 후처리"""
        if self.initial_parameter_bounds is None:
            self.initial_parameter_bounds = {}
    
    def validate(self) -> bool:
        """설정 검증"""
        if self.max_iterations <= 0:
            return False
        if self.convergence_tolerance <= 0:
            return False
        if self.simulation_draws <= 0:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "method": self.method.value,
            "optimizer": self.optimizer.value,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
            "simulation_draws": self.simulation_draws,
            "halton_draws": self.halton_draws,
            "antithetic_draws": self.antithetic_draws,
            "use_random_starts": self.use_random_starts,
            "n_random_starts": self.n_random_starts,
            "initial_parameter_bounds": self.initial_parameter_bounds,
            "numerical_precision": self.numerical_precision,
            "step_size": self.step_size,
            "finite_difference_step": self.finite_difference_step,
            "parallel_processing": self.parallel_processing,
            "n_cores": self.n_cores,
            "robust_standard_errors": self.robust_standard_errors,
            "bootstrap_samples": self.bootstrap_samples
        }


def create_default_estimation_config() -> EstimationConfig:
    """기본 추정 설정 생성"""
    return EstimationConfig()


def create_sml_config(simulation_draws: int = 1000, **kwargs) -> EstimationConfig:
    """시뮬레이션 최대우도 설정 생성"""
    config = EstimationConfig(
        method=EstimationMethod.SIMULATED_MAXIMUM_LIKELIHOOD,
        simulation_draws=simulation_draws,
        **kwargs
    )
    return config


def create_robust_config(**kwargs) -> EstimationConfig:
    """로버스트 추정 설정 생성"""
    config = EstimationConfig(
        robust_standard_errors=True,
        bootstrap_samples=500,
        use_random_starts=True,
        n_random_starts=10,
        **kwargs
    )
    return config
