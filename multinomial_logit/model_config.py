"""
Multinomial Logit Model 구성 모듈

이 모듈은 Multinomial Logit Model의 설정과 구성을 관리하는
재사용 가능한 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Multinomial Logit Model 설정을 저장하는 데이터클래스"""
    
    # 모델 파라미터
    max_iterations: int = 1000
    tolerance: float = 1e-6
    method: str = 'bfgs'  # 최적화 방법
    
    # 데이터 설정
    feature_names: List[str] = None
    feature_descriptions: Dict[str, str] = None
    
    # 결과 설정
    confidence_level: float = 0.95
    include_marginal_effects: bool = True
    include_elasticities: bool = True
    
    # 검증 설정
    validate_data: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.feature_names is None:
            self.feature_names = []
        if self.feature_descriptions is None:
            self.feature_descriptions = {}


class ModelConfigManager:
    """모델 설정을 관리하는 클래스"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        모델 설정 관리자 초기화
        
        Args:
            config (Optional[ModelConfig]): 모델 설정 객체
        """
        self.config = config if config is not None else ModelConfig()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """설정의 유효성을 검증"""
        if self.config.max_iterations <= 0:
            raise ValueError("max_iterations는 양수여야 합니다")
        
        if self.config.tolerance <= 0:
            raise ValueError("tolerance는 양수여야 합니다")
        
        if not 0 < self.config.confidence_level < 1:
            raise ValueError("confidence_level은 0과 1 사이의 값이어야 합니다")
        
        valid_methods = ['bfgs', 'newton', 'lbfgs', 'powell', 'cg']
        if self.config.method.lower() not in valid_methods:
            logger.warning(f"권장되지 않는 최적화 방법: {self.config.method}")
    
    def update_feature_info(self, feature_names: List[str], 
                          feature_descriptions: Optional[Dict[str, str]] = None) -> None:
        """
        특성 정보를 업데이트
        
        Args:
            feature_names (List[str]): 특성 이름 리스트
            feature_descriptions (Optional[Dict[str, str]]): 특성 설명 딕셔너리
        """
        self.config.feature_names = feature_names.copy()
        
        if feature_descriptions is not None:
            self.config.feature_descriptions = feature_descriptions.copy()
        else:
            # 기본 설명 생성
            self.config.feature_descriptions = {
                name: f"Feature: {name}" for name in feature_names
            }
        
        logger.info(f"특성 정보 업데이트 완료: {len(feature_names)}개 특성")
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """
        최적화 파라미터를 반환
        
        Returns:
            Dict[str, Any]: 최적화 파라미터
        """
        return {
            'method': self.config.method,
            'maxiter': self.config.max_iterations,
            'ftol': self.config.tolerance,
            'disp': self.config.verbose
        }
    
    def get_confidence_interval_alpha(self) -> float:
        """
        신뢰구간을 위한 알파 값을 반환
        
        Returns:
            float: 알파 값 (1 - confidence_level)
        """
        return 1 - self.config.confidence_level
    
    def create_results_template(self) -> Dict[str, Any]:
        """
        결과를 저장할 템플릿을 생성
        
        Returns:
            Dict[str, Any]: 결과 템플릿
        """
        template = {
            'model_info': {
                'feature_names': self.config.feature_names.copy(),
                'feature_descriptions': self.config.feature_descriptions.copy(),
                'n_features': len(self.config.feature_names)
            },
            'estimation_results': {
                'coefficients': None,
                'standard_errors': None,
                'z_scores': None,
                'p_values': None,
                'confidence_intervals': None
            },
            'model_statistics': {
                'log_likelihood': None,
                'aic': None,
                'bic': None,
                'pseudo_r_squared': None,
                'n_observations': None,
                'n_choice_sets': None
            },
            'convergence_info': {
                'converged': None,
                'iterations': None,
                'optimization_method': self.config.method
            }
        }
        
        # 선택적 결과 추가
        if self.config.include_marginal_effects:
            template['marginal_effects'] = None
        
        if self.config.include_elasticities:
            template['elasticities'] = None
        
        return template
    
    def validate_input_data(self, X: np.ndarray, y: np.ndarray, 
                          choice_sets: np.ndarray) -> bool:
        """
        입력 데이터의 유효성을 검증
        
        Args:
            X (np.ndarray): 설명변수 행렬
            y (np.ndarray): 선택 결과 벡터
            choice_sets (np.ndarray): 선택 세트 인덱스
            
        Returns:
            bool: 유효성 검증 결과
        """
        if not self.config.validate_data:
            return True
        
        try:
            # 기본 차원 검증
            if len(X.shape) != 2:
                logger.error("X는 2차원 배열이어야 합니다")
                return False
            
            if len(y.shape) != 1:
                logger.error("y는 1차원 배열이어야 합니다")
                return False
            
            if X.shape[0] != len(y):
                logger.error("X와 y의 행 수가 일치하지 않습니다")
                return False
            
            # 특성 수 검증
            if len(self.config.feature_names) > 0:
                if X.shape[1] != len(self.config.feature_names):
                    logger.error("X의 열 수와 feature_names의 길이가 일치하지 않습니다")
                    return False
            
            # 선택 결과 검증
            if not np.all(np.isin(y, [0, 1])):
                logger.error("y는 0 또는 1의 값만 가져야 합니다")
                return False
            
            # 선택 세트 검증
            if len(choice_sets) == 0:
                logger.error("choice_sets가 비어있습니다")
                return False
            
            # 각 선택 세트에서 정확히 하나의 선택이 있는지 확인
            for i, start_idx in enumerate(choice_sets):
                if i < len(choice_sets) - 1:
                    end_idx = choice_sets[i + 1]
                else:
                    end_idx = len(y)
                
                choice_sum = y[start_idx:end_idx].sum()
                if choice_sum != 1:
                    logger.warning(f"선택 세트 {i}에서 선택 개수가 1이 아닙니다: {choice_sum}")
            
            logger.info("입력 데이터 유효성 검증 완료")
            return True
            
        except Exception as e:
            logger.error(f"데이터 유효성 검증 중 오류: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        현재 설정의 요약을 반환
        
        Returns:
            Dict[str, Any]: 설정 요약
        """
        return {
            'optimization': {
                'method': self.config.method,
                'max_iterations': self.config.max_iterations,
                'tolerance': self.config.tolerance
            },
            'features': {
                'count': len(self.config.feature_names),
                'names': self.config.feature_names
            },
            'analysis': {
                'confidence_level': self.config.confidence_level,
                'include_marginal_effects': self.config.include_marginal_effects,
                'include_elasticities': self.config.include_elasticities
            },
            'validation': {
                'validate_data': self.config.validate_data,
                'verbose': self.config.verbose
            }
        }


def create_default_config(feature_names: List[str], 
                         feature_descriptions: Optional[Dict[str, str]] = None) -> ModelConfig:
    """
    기본 모델 설정을 생성하는 편의 함수
    
    Args:
        feature_names (List[str]): 특성 이름 리스트
        feature_descriptions (Optional[Dict[str, str]]): 특성 설명 딕셔너리
        
    Returns:
        ModelConfig: 기본 설정 객체
    """
    config = ModelConfig()
    config.feature_names = feature_names.copy()
    
    if feature_descriptions is not None:
        config.feature_descriptions = feature_descriptions.copy()
    else:
        config.feature_descriptions = {
            name: f"Feature: {name}" for name in feature_names
        }
    
    return config


def create_custom_config(max_iterations: int = 1000,
                        tolerance: float = 1e-6,
                        method: str = 'bfgs',
                        confidence_level: float = 0.95,
                        **kwargs) -> ModelConfig:
    """
    사용자 정의 모델 설정을 생성하는 편의 함수
    
    Args:
        max_iterations (int): 최대 반복 횟수
        tolerance (float): 수렴 허용 오차
        method (str): 최적화 방법
        confidence_level (float): 신뢰수준
        **kwargs: 추가 설정 파라미터
        
    Returns:
        ModelConfig: 사용자 정의 설정 객체
    """
    config = ModelConfig(
        max_iterations=max_iterations,
        tolerance=tolerance,
        method=method,
        confidence_level=confidence_level
    )
    
    # 추가 파라미터 설정
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
