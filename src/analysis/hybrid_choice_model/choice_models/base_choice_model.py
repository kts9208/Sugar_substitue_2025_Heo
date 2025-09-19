"""
Base Choice Model

모든 선택모델의 기본 클래스와 인터페이스를 정의합니다.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChoiceModelType(Enum):
    """선택모델 타입"""
    MULTINOMIAL_LOGIT = "multinomial_logit"
    RANDOM_PARAMETERS_LOGIT = "random_parameters_logit"
    MIXED_LOGIT = "mixed_logit"
    NESTED_LOGIT = "nested_logit"
    MULTINOMIAL_PROBIT = "multinomial_probit"


@dataclass
class ChoiceModelResults:
    """선택모델 결과 기본 클래스"""
    
    # 기본 결과
    model_type: ChoiceModelType
    log_likelihood: float
    aic: float
    bic: float
    
    # 모수 추정 결과
    parameters: Dict[str, float]
    standard_errors: Dict[str, float]
    t_statistics: Dict[str, float]
    p_values: Dict[str, float]
    
    # 적합도 지수
    rho_squared: float
    adjusted_rho_squared: float
    
    # 예측 결과
    predicted_probabilities: Optional[pd.DataFrame] = None
    predicted_choices: Optional[pd.Series] = None
    
    # 추가 정보
    convergence_status: bool = True
    iterations: int = 0
    estimation_time: float = 0.0
    sample_size: int = 0
    
    # 모델별 특화 결과 (하위 클래스에서 확장)
    additional_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_results is None:
            self.additional_results = {}
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """요약 통계 반환"""
        return {
            "model_type": self.model_type.value,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "rho_squared": self.rho_squared,
            "adjusted_rho_squared": self.adjusted_rho_squared,
            "sample_size": self.sample_size,
            "convergence_status": self.convergence_status,
            "iterations": self.iterations,
            "estimation_time": self.estimation_time
        }
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """모수 요약 테이블 반환"""
        summary_data = []
        for param_name in self.parameters.keys():
            summary_data.append({
                "parameter": param_name,
                "estimate": self.parameters[param_name],
                "std_error": self.standard_errors.get(param_name, np.nan),
                "t_statistic": self.t_statistics.get(param_name, np.nan),
                "p_value": self.p_values.get(param_name, np.nan)
            })
        
        return pd.DataFrame(summary_data)


class BaseChoiceModel(ABC):
    """선택모델 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 모델 설정 딕셔너리
        """
        self.config = config
        self.is_fitted = False
        self.results = None
        self.data = None
        self.choice_column = config.get('choice_column', 'choice')
        self.alternative_column = config.get('alternative_column', 'alternative')
        self.individual_column = config.get('individual_column', 'individual_id')
        
        # 로깅 설정
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def model_type(self) -> ChoiceModelType:
        """모델 타입 반환"""
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> ChoiceModelResults:
        """
        모델 추정
        
        Args:
            data: 선택 데이터
            **kwargs: 추가 매개변수
            
        Returns:
            추정 결과
        """
        pass
    
    @abstractmethod
    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        선택 확률 예측
        
        Args:
            data: 예측할 데이터
            
        Returns:
            선택 확률 데이터프레임
        """
        pass
    
    @abstractmethod
    def predict_choices(self, data: pd.DataFrame) -> pd.Series:
        """
        선택 예측
        
        Args:
            data: 예측할 데이터
            
        Returns:
            예측된 선택
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        데이터 유효성 검사
        
        Args:
            data: 검사할 데이터
            
        Returns:
            유효성 여부
        """
        required_columns = [self.choice_column, self.alternative_column, self.individual_column]
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"필수 컬럼이 없습니다: {col}")
        
        # 선택 데이터 검사
        if data[self.choice_column].isnull().any():
            raise ValueError("선택 데이터에 결측값이 있습니다.")
        
        # 개체별 선택 일관성 검사
        choice_counts = data.groupby([self.individual_column, self.choice_column]).size()
        if (choice_counts > 1).any():
            self.logger.warning("일부 개체에서 중복 선택이 발견되었습니다.")
        
        return True
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 전처리
        
        Args:
            data: 원본 데이터
            
        Returns:
            전처리된 데이터
        """
        # 데이터 복사
        processed_data = data.copy()
        
        # 결측값 처리
        if processed_data.isnull().any().any():
            self.logger.warning("결측값이 발견되어 제거합니다.")
            processed_data = processed_data.dropna()
        
        # 데이터 타입 확인 및 변환
        if processed_data[self.choice_column].dtype not in ['int64', 'bool']:
            processed_data[self.choice_column] = processed_data[self.choice_column].astype(int)
        
        return processed_data
    
    def calculate_fit_statistics(self, log_likelihood: float, n_params: int, n_obs: int) -> Dict[str, float]:
        """
        적합도 통계 계산
        
        Args:
            log_likelihood: 로그우도
            n_params: 모수 개수
            n_obs: 관측치 개수
            
        Returns:
            적합도 통계 딕셔너리
        """
        # AIC, BIC 계산
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_obs)
        
        # Null 로그우도 (모든 대안이 동일한 확률을 가질 때)
        n_alternatives = len(self.data[self.alternative_column].unique())
        null_log_likelihood = n_obs * np.log(1 / n_alternatives)
        
        # Rho-squared 계산
        rho_squared = 1 - (log_likelihood / null_log_likelihood)
        adjusted_rho_squared = 1 - ((log_likelihood - n_params) / null_log_likelihood)
        
        return {
            "aic": aic,
            "bic": bic,
            "rho_squared": rho_squared,
            "adjusted_rho_squared": adjusted_rho_squared,
            "null_log_likelihood": null_log_likelihood
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_type": self.model_type.value,
            "is_fitted": self.is_fitted,
            "config": self.config,
            "data_shape": self.data.shape if self.data is not None else None
        }
    
    def save_results(self, file_path: str):
        """결과 저장"""
        if not self.is_fitted:
            raise ValueError("모델이 추정되지 않았습니다.")
        
        # 결과를 딕셔너리로 변환하여 저장
        results_dict = {
            "model_info": self.get_model_info(),
            "summary_statistics": self.results.get_summary_statistics(),
            "parameters": self.results.get_parameter_summary().to_dict('records')
        }
        
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"결과가 저장되었습니다: {file_path}")
    
    def __str__(self) -> str:
        """문자열 표현"""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_type.value} ({status})"
    
    def __repr__(self) -> str:
        """객체 표현"""
        return f"{self.__class__.__name__}(model_type={self.model_type.value}, is_fitted={self.is_fitted})"
