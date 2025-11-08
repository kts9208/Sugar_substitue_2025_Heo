"""
Factor Analysis Configuration Module

이 모듈은 Factor Loading 분석을 위한 설정과 모델 스펙을 관리합니다.
기존 FactorConfig를 활용하여 semopy 모델 스펙을 생성합니다.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# 기존 모듈 임포트를 위한 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "processed_data" / "modules"))

try:
    from survey_data_preprocessor import FactorConfig
except ImportError as e:
    # 무시 가능한 오류 (ICLV 추정과 무관)
    FactorConfig = None

logger = logging.getLogger(__name__)


@dataclass
class FactorAnalysisConfig:
    """Factor Analysis 설정을 저장하는 데이터클래스"""
    
    # 분석 설정
    estimator: str = 'MLW'  # Maximum Likelihood with Wishart
    optimizer: str = 'SLSQP'  # Sequential Least Squares Programming
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # 모델 적합도 설정
    calculate_fit_indices: bool = True
    bootstrap_samples: int = 0  # 0이면 부트스트랩 안함
    confidence_level: float = 0.95
    
    # 출력 설정
    standardized: bool = True
    include_modification_indices: bool = True
    verbose: bool = True
    
    # 데이터 설정
    missing_data_method: str = 'listwise'  # 'listwise', 'fiml'
    
    def __post_init__(self):
        """초기화 후 검증"""
        valid_estimators = ['MLW', 'ML', 'GLS', 'WLS', 'ULS']
        if self.estimator not in valid_estimators:
            raise ValueError(f"지원되지 않는 추정방법: {self.estimator}")
        
        valid_optimizers = ['SLSQP', 'L-BFGS-B', 'trust-constr']
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"지원되지 않는 최적화 방법: {self.optimizer}")


class FactorModelSpecBuilder:
    """semopy 모델 스펙을 생성하는 클래스"""
    
    def __init__(self):
        """모델 스펙 빌더 초기화"""
        self.factor_config = FactorConfig()
        
    def create_single_factor_spec(self, factor_name: str) -> str:
        """
        단일 요인에 대한 semopy 모델 스펙 생성

        Args:
            factor_name (str): 요인 이름

        Returns:
            str: semopy 모델 스펙 문자열
        """
        if factor_name not in self.factor_config.get_all_factors():
            raise ValueError(f"알 수 없는 요인: {factor_name}")

        questions = self.factor_config.get_factor_questions(factor_name)

        # DCE 변수와 인구통계학적 변수는 제외 (요인분석 부적합)
        if factor_name in ['dce_variables', 'demographics_1', 'demographics_2']:
            raise ValueError(f"{factor_name}은 요인분석에 적합하지 않습니다")

        # 단일 요인 모델 스펙 생성 (모든 loading 자유 추정)
        spec_lines = []
        spec_lines.append(f"# {factor_name} Factor Model")
        spec_lines.append(f"{factor_name} =~ " + " + ".join(questions))

        # 잠재변수 분산을 1로 고정하여 모델 식별
        spec_lines.append(f"{factor_name} ~~ 1*{factor_name}")

        return "\n".join(spec_lines)
    
    def create_multi_factor_spec(self, factor_names: List[str],
                                allow_correlations: bool = True) -> str:
        """
        다중 요인에 대한 semopy 모델 스펙 생성 (동적 생성)

        Args:
            factor_names (List[str]): 요인 이름 리스트
            allow_correlations (bool): 요인간 상관관계 허용 여부

        Returns:
            str: semopy 모델 스펙 문자열
        """
        from pathlib import Path
        import pandas as pd

        # 부적합한 요인들 제외
        excluded_factors = ['dce_variables', 'demographics_1', 'demographics_2']
        valid_factors = [f for f in factor_names if f not in excluded_factors]

        if not valid_factors:
            raise ValueError("요인분석에 적합한 요인이 없습니다")

        # 실제 데이터에서 사용 가능한 문항들을 확인
        survey_data_dir = Path("processed_data/survey_data")
        factor_items = {}

        factor_files = {
            'health_concern': 'health_concern.csv',
            'perceived_benefit': 'perceived_benefit.csv',
            'purchase_intention': 'purchase_intention.csv',
            'perceived_price': 'perceived_price.csv',
            'nutrition_knowledge': 'nutrition_knowledge.csv'
        }

        for factor_name in valid_factors:
            if factor_name in factor_files:
                file_path = survey_data_dir / factor_files[factor_name]
                if file_path.exists():
                    data = pd.read_csv(file_path)
                    items = [col for col in data.columns if col.startswith('q')]
                    factor_items[factor_name] = items
                else:
                    logger.warning(f"데이터 파일을 찾을 수 없음: {file_path}")

        spec_lines = []
        spec_lines.append("# Multi-Factor Model (Dynamic)")

        # 각 요인의 측정 모델 (실제 데이터 기반)
        for factor_name in valid_factors:
            if factor_name in factor_items and factor_items[factor_name]:
                questions = factor_items[factor_name]
                spec_lines.append(f"{factor_name} =~ " + " + ".join(questions))
            else:
                logger.warning(f"요인 {factor_name}의 문항을 찾을 수 없음")

        # 잠재변수 분산을 1로 고정하여 모델 식별
        spec_lines.append("")
        spec_lines.append("# Factor variances fixed to 1 for identification")
        for factor_name in valid_factors:
            if factor_name in factor_items and factor_items[factor_name]:
                spec_lines.append(f"{factor_name} ~~ 1*{factor_name}")

        # 요인간 상관관계 (기본적으로 허용)
        if allow_correlations and len(valid_factors) > 1:
            spec_lines.append("")
            spec_lines.append("# Factor correlations (default: estimated)")
            for i, factor1 in enumerate(valid_factors):
                for factor2 in valid_factors[i+1:]:
                    spec_lines.append(f"# {factor1} ~~ {factor2}")

        return "\n".join(spec_lines)
    
    def get_analyzable_factors(self) -> List[str]:
        """
        요인분석에 적합한 요인들의 리스트 반환
        
        Returns:
            List[str]: 분석 가능한 요인 이름 리스트
        """
        all_factors = self.factor_config.get_all_factors()
        excluded = ['dce_variables', 'demographics_1', 'demographics_2']
        return [f for f in all_factors if f not in excluded]


def create_factor_model_spec(factor_names: Optional[List[str]] = None, 
                           single_factor: Optional[str] = None,
                           allow_correlations: bool = True) -> str:
    """
    Factor 모델 스펙을 생성하는 편의 함수
    
    Args:
        factor_names (Optional[List[str]]): 다중 요인 분석할 요인들
        single_factor (Optional[str]): 단일 요인 분석할 요인
        allow_correlations (bool): 요인간 상관관계 허용 여부
        
    Returns:
        str: semopy 모델 스펙 문자열
    """
    builder = FactorModelSpecBuilder()
    
    if single_factor:
        return builder.create_single_factor_spec(single_factor)
    elif factor_names:
        return builder.create_multi_factor_spec(factor_names, allow_correlations)
    else:
        # 기본: 모든 분석 가능한 요인들로 다중 요인 모델
        analyzable_factors = builder.get_analyzable_factors()
        return builder.create_multi_factor_spec(analyzable_factors, allow_correlations)


def get_default_config() -> FactorAnalysisConfig:
    """기본 설정을 반환하는 편의 함수"""
    return FactorAnalysisConfig()


def create_custom_config(**kwargs) -> FactorAnalysisConfig:
    """사용자 정의 설정을 생성하는 편의 함수"""
    return FactorAnalysisConfig(**kwargs)
