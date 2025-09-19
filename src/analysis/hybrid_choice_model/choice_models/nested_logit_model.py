"""
Nested Logit Model

중첩로짓 모델 구현입니다.
계층적 선택구조를 고려한 선택모델입니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .base_choice_model import BaseChoiceModel, ChoiceModelType, ChoiceModelResults

logger = logging.getLogger(__name__)


@dataclass
class NestedLogitResults(ChoiceModelResults):
    """Nested Logit 모델 결과 클래스"""
    
    # Nested Logit 특화 결과
    nest_parameters: Optional[Dict[str, float]] = None
    nesting_structure: Optional[Dict[str, List[str]]] = None
    inclusive_values: Optional[pd.DataFrame] = None
    
    def get_nest_summary(self) -> pd.DataFrame:
        """네스트별 요약 정보"""
        if not self.nest_parameters or not self.nesting_structure:
            return pd.DataFrame()
        
        summary_data = []
        for nest_name, alternatives in self.nesting_structure.items():
            summary_data.append({
                "nest": nest_name,
                "alternatives": alternatives,
                "nest_parameter": self.nest_parameters.get(nest_name, np.nan),
                "n_alternatives": len(alternatives)
            })
        
        return pd.DataFrame(summary_data)


class NestedLogitModel(BaseChoiceModel):
    """중첩로짓 모델 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Nested Logit 특화 설정
        self.nesting_structure = config.get('nesting_structure', {})
        self.nest_parameters = config.get('nest_parameters', {})
        
        # 추정 설정
        self.max_iterations = config.get('max_iterations', 1000)
        self.convergence_tolerance = config.get('convergence_tolerance', 1e-6)
        
        # 네스팅 구조 검증
        if not self.nesting_structure:
            raise ValueError("중첩로짓 모델에는 nesting_structure가 필요합니다.")
    
    @property
    def model_type(self) -> ChoiceModelType:
        return ChoiceModelType.NESTED_LOGIT
    
    def fit(self, data: pd.DataFrame, **kwargs) -> NestedLogitResults:
        """Nested Logit 모델 추정 (기본 구현)"""
        logger.info("Nested Logit 모델 추정을 시작합니다...")
        
        # 데이터 검증
        self.validate_data(data)
        self.data = self.prepare_data(data)
        
        # 임시 결과 반환 (실제 구현 필요)
        results = NestedLogitResults(
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
            nest_parameters=self.nest_parameters,
            nesting_structure=self.nesting_structure
        )
        
        self.results = results
        self.is_fitted = True
        
        logger.info("Nested Logit 모델 추정 완료 (임시 구현)")
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
