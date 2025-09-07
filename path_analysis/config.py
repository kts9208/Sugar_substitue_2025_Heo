"""
Path Analysis Configuration Module

경로분석을 위한 설정 클래스와 유틸리티 함수들을 제공합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathAnalysisConfig:
    """경로분석 설정 클래스"""
    
    # 추정 방법
    estimator: str = 'MLW'  # MLW, ML, GLS, WLS, DWLS, ULS
    optimizer: str = 'SLSQP'  # SLSQP, L-BFGS-B, trust-constr
    
    # 모델 설정
    standardized: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # 수렴 기준
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # 결과 저장 설정
    save_results: bool = True
    results_dir: str = "path_analysis_results"
    
    # 가시화 설정
    create_diagrams: bool = True
    diagram_format: str = 'png'  # png, pdf, svg
    
    # 효과 분석 설정
    calculate_effects: bool = True
    include_bootstrap_ci: bool = True
    
    # 데이터 설정
    data_dir: str = "processed_data/survey_data"
    missing_data_method: str = 'listwise'  # listwise, fiml
    
    # 로깅 설정
    verbose: bool = True
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """설정 검증"""
        self._validate_estimator()
        self._validate_optimizer()
        self._validate_paths()
    
    def _validate_estimator(self):
        """추정방법 검증"""
        valid_estimators = ['MLW', 'ML', 'GLS', 'WLS', 'DWLS', 'ULS']
        if self.estimator not in valid_estimators:
            raise ValueError(f"estimator는 {valid_estimators} 중 하나여야 합니다.")
    
    def _validate_optimizer(self):
        """최적화 방법 검증"""
        valid_optimizers = ['SLSQP', 'L-BFGS-B', 'trust-constr']
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer는 {valid_optimizers} 중 하나여야 합니다.")
    
    def _validate_paths(self):
        """경로 검증 및 생성"""
        # 결과 디렉토리 생성
        results_path = Path(self.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터 디렉토리 확인
        data_path = Path(self.data_dir)
        if not data_path.exists():
            logger.warning(f"데이터 디렉토리가 존재하지 않습니다: {self.data_dir}")


def create_default_path_config(**kwargs) -> PathAnalysisConfig:
    """
    기본 경로분석 설정 생성
    
    Args:
        **kwargs: 설정 오버라이드
        
    Returns:
        PathAnalysisConfig: 설정 객체
    """
    return PathAnalysisConfig(**kwargs)


def create_mediation_config(**kwargs) -> PathAnalysisConfig:
    """
    매개효과 분석용 설정 생성
    
    Args:
        **kwargs: 설정 오버라이드
        
    Returns:
        PathAnalysisConfig: 매개효과 분석용 설정
    """
    mediation_defaults = {
        'bootstrap_samples': 5000,  # 매개효과는 더 많은 부트스트랩 필요
        'include_bootstrap_ci': True,
        'calculate_effects': True,
        'confidence_level': 0.95
    }
    
    # 기본값과 사용자 입력 병합
    merged_kwargs = {**mediation_defaults, **kwargs}
    return PathAnalysisConfig(**merged_kwargs)


def create_exploratory_config(**kwargs) -> PathAnalysisConfig:
    """
    탐색적 경로분석용 설정 생성
    
    Args:
        **kwargs: 설정 오버라이드
        
    Returns:
        PathAnalysisConfig: 탐색적 분석용 설정
    """
    exploratory_defaults = {
        'bootstrap_samples': 1000,
        'create_diagrams': True,
        'verbose': True,
        'standardized': True
    }
    
    merged_kwargs = {**exploratory_defaults, **kwargs}
    return PathAnalysisConfig(**merged_kwargs)


# 사전 정의된 모델 템플릿
PREDEFINED_MODELS = {
    'simple_mediation': {
        'description': '단순 매개모델 (X -> M -> Y)',
        'variables': ['X', 'M', 'Y'],
        'paths': [
            ('X', 'M'),  # a path
            ('M', 'Y'),  # b path  
            ('X', 'Y')   # c' path (direct effect)
        ]
    },
    
    'multiple_mediation': {
        'description': '다중 매개모델 (X -> M1,M2 -> Y)',
        'variables': ['X', 'M1', 'M2', 'Y'],
        'paths': [
            ('X', 'M1'),
            ('X', 'M2'),
            ('M1', 'Y'),
            ('M2', 'Y'),
            ('X', 'Y')
        ]
    },
    
    'serial_mediation': {
        'description': '순차 매개모델 (X -> M1 -> M2 -> Y)',
        'variables': ['X', 'M1', 'M2', 'Y'],
        'paths': [
            ('X', 'M1'),
            ('M1', 'M2'),
            ('M2', 'Y'),
            ('X', 'Y')
        ]
    }
}


def get_predefined_model(model_name: str) -> Dict[str, Any]:
    """
    사전 정의된 모델 템플릿 반환
    
    Args:
        model_name (str): 모델 이름
        
    Returns:
        Dict[str, Any]: 모델 템플릿
    """
    if model_name not in PREDEFINED_MODELS:
        available_models = list(PREDEFINED_MODELS.keys())
        raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다. "
                        f"사용 가능한 모델: {available_models}")
    
    return PREDEFINED_MODELS[model_name].copy()


def list_predefined_models() -> List[str]:
    """사용 가능한 사전 정의된 모델 목록 반환"""
    return list(PREDEFINED_MODELS.keys())
