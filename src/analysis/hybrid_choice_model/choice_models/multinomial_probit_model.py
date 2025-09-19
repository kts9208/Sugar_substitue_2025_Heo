"""
Multinomial Probit Model

다항프로빗 모델 구현입니다.
정규분포 기반의 선택모델입니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .base_choice_model import BaseChoiceModel, ChoiceModelType, ChoiceModelResults

logger = logging.getLogger(__name__)


@dataclass
class ProbitResults(ChoiceModelResults):
    """Probit 모델 결과 클래스"""
    
    # Probit 특화 결과
    covariance_matrix: Optional[np.ndarray] = None
    correlation_matrix: Optional[np.ndarray] = None
    correlation_structure: str = "unstructured"
    
    def get_correlation_summary(self) -> pd.DataFrame:
        """상관관계 요약"""
        if self.correlation_matrix is None:
            return pd.DataFrame()
        
        n_alt = self.correlation_matrix.shape[0]
        correlations = []
        
        for i in range(n_alt):
            for j in range(i+1, n_alt):
                correlations.append({
                    "alternative_1": f"alt_{i}",
                    "alternative_2": f"alt_{j}",
                    "correlation": self.correlation_matrix[i, j]
                })
        
        return pd.DataFrame(correlations)


class MultinomialProbitModel(BaseChoiceModel):
    """다항프로빗 모델 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Probit 특화 설정
        self.correlation_structure = config.get('correlation_structure', 'unstructured')
        self.simulation_draws = config.get('simulation_draws', 1000)
        
        # 추정 설정
        self.max_iterations = config.get('max_iterations', 1000)
        self.convergence_tolerance = config.get('convergence_tolerance', 1e-6)
    
    @property
    def model_type(self) -> ChoiceModelType:
        return ChoiceModelType.MULTINOMIAL_PROBIT
    
    def fit(self, data: pd.DataFrame, **kwargs) -> ProbitResults:
        """Probit 모델 추정 (기본 구현)"""
        logger.info("Multinomial Probit 모델 추정을 시작합니다...")
        
        # 데이터 검증
        self.validate_data(data)
        self.data = self.prepare_data(data)
        
        # 임시 결과 반환 (실제 구현 필요)
        n_alternatives = len(self.data[self.alternative_column].unique())
        
        # 임시 상관행렬
        correlation_matrix = np.eye(n_alternatives)
        correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.3
        
        results = ProbitResults(
            model_type=self.model_type,
            log_likelihood=-1000.0,
            aic=2000.0,
            bic=2100.0,
            parameters={"temp_param": 1.0},
            standard_errors={"temp_param": 0.1},
            t_statistics={"temp_param": 10.0},
            p_values={"temp_param": 0.001},
            rho_squared=0.3,
            adjusted_rho_squared=0.25,
            sample_size=len(data),
            correlation_matrix=correlation_matrix,
            correlation_structure=self.correlation_structure
        )
        
        self.results = results
        self.is_fitted = True
        
        logger.info("Multinomial Probit 모델 추정 완료 (임시 구현)")
        return results
    
    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """선택 확률 예측 (기본 구현)"""
        if not self.is_fitted:
            raise ValueError("모델이 추정되지 않았습니다.")
        
        n_obs = len(data)
        n_alternatives = 3  # 임시값
        uniform_probs = np.ones((n_obs, n_alternatives)) / n_alternatives
        return pd.DataFrame(uniform_probs, columns=[f"alt_{i}" for i in range(n_alternatives)])
    
    def predict_choices(self, data: pd.DataFrame) -> pd.Series:
        """선택 예측"""
        probabilities = self.predict_probabilities(data)
        return probabilities.idxmax(axis=1)
