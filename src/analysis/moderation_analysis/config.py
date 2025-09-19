"""
Moderation Analysis Configuration Module

조절효과 분석을 위한 설정 클래스와 기본 설정을 제공합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModerationAnalysisConfig:
    """조절효과 분석 설정 클래스"""
    
    # 데이터 설정
    data_dir: str = "processed_data/survey_data"
    results_dir: str = "moderation_analysis_results"
    
    # 분석 설정
    estimator: str = "ML"  # Maximum Likelihood
    standardized: bool = True
    bootstrap_samples: int = 5000
    confidence_level: float = 0.95
    
    # 조절효과 분석 설정
    center_variables: bool = True  # 변수 중심화
    interaction_method: str = "product"  # product, orthogonal
    simple_slopes_values: List[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0])  # 표준편차 단위
    
    # 모델 적합 설정
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    optimizer: str = "SLSQP"
    
    # 결과 설정
    save_csv: bool = True
    save_json: bool = True
    save_plots: bool = True
    save_report: bool = True
    
    # 시각화 설정
    plot_style: str = "seaborn-v0_8"
    figure_size: tuple = (10, 8)
    dpi: int = 300
    color_palette: str = "viridis"
    
    # 로깅 설정
    log_level: str = "INFO"
    verbose: bool = True
    
    def __post_init__(self):
        """설정 검증 및 초기화"""
        # 디렉토리 생성
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # 설정 검증
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("confidence_level은 0과 1 사이의 값이어야 합니다.")
        
        if self.bootstrap_samples < 100:
            logger.warning("bootstrap_samples가 100보다 작습니다. 최소 1000을 권장합니다.")
        
        if self.estimator not in ["ML", "GLS", "WLS", "ULS"]:
            logger.warning(f"알 수 없는 estimator: {self.estimator}. ML을 사용합니다.")
            self.estimator = "ML"


def create_default_moderation_config() -> ModerationAnalysisConfig:
    """기본 조절효과 분석 설정 생성"""
    return ModerationAnalysisConfig()


def create_custom_moderation_config(
    data_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    estimator: str = "ML",
    standardized: bool = True,
    bootstrap_samples: int = 5000,
    confidence_level: float = 0.95,
    center_variables: bool = True,
    simple_slopes_values: Optional[List[float]] = None,
    **kwargs
) -> ModerationAnalysisConfig:
    """사용자 정의 조절효과 분석 설정 생성"""
    
    config_dict = {
        'estimator': estimator,
        'standardized': standardized,
        'bootstrap_samples': bootstrap_samples,
        'confidence_level': confidence_level,
        'center_variables': center_variables,
    }
    
    if data_dir is not None:
        config_dict['data_dir'] = data_dir
    
    if results_dir is not None:
        config_dict['results_dir'] = results_dir
    
    if simple_slopes_values is not None:
        config_dict['simple_slopes_values'] = simple_slopes_values
    
    # 추가 키워드 인수 병합
    config_dict.update(kwargs)
    
    return ModerationAnalysisConfig(**config_dict)


def get_factor_items_mapping() -> Dict[str, List[str]]:
    """5개 요인별 문항 매핑 정보 (실제 데이터 파일 기준)"""
    return {
        'health_concern': [f'q{i}' for i in range(6, 12)],  # q6~q11
        'perceived_benefit': [f'q{i}' for i in range(16, 18)],  # q16~q17 (실제 데이터)
        'purchase_intention': [f'q{i}' for i in range(18, 20)],  # q18~q19 (실제 데이터)
        'perceived_price': [f'q{i}' for i in range(27, 30)],  # q27~q29
        'nutrition_knowledge': [f'q{i}' for i in range(30, 50)]  # q30~q49
    }


def get_factor_descriptions() -> Dict[str, str]:
    """요인별 설명"""
    return {
        'health_concern': '건강관심도',
        'perceived_benefit': '지각된 혜택',
        'purchase_intention': '구매의도',
        'perceived_price': '지각된 가격',
        'nutrition_knowledge': '영양지식'
    }


def validate_factor_combination(independent_var: str, dependent_var: str, 
                              moderator_var: str) -> bool:
    """요인 조합 유효성 검증"""
    available_factors = list(get_factor_items_mapping().keys())
    
    factors = [independent_var, dependent_var, moderator_var]
    
    # 모든 요인이 사용 가능한 요인인지 확인
    for factor in factors:
        if factor not in available_factors:
            logger.error(f"알 수 없는 요인: {factor}")
            return False
    
    # 독립변수와 종속변수가 다른지 확인
    if independent_var == dependent_var:
        logger.error("독립변수와 종속변수는 달라야 합니다.")
        return False
    
    # 조절변수가 독립변수나 종속변수와 다른지 확인
    if moderator_var in [independent_var, dependent_var]:
        logger.warning(f"조절변수 {moderator_var}가 독립변수 또는 종속변수와 동일합니다.")
    
    return True


# 기본 설정 인스턴스
DEFAULT_CONFIG = create_default_moderation_config()
