"""
Choice Equations for ICLV Models

ICLV 선택모델: 속성 + 잠재변수 → 선택

Based on King (2022) Apollo R code implementation.

Author: Sugar Substitute Research Team
Date: 2025-11-05
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
import logging
from dataclasses import dataclass

# ChoiceConfig 정의 (import 오류 방지)
try:
    from .iclv_config import ChoiceConfig
except ImportError:
    from typing import Literal
    
    @dataclass
    class ChoiceConfig:
        """선택모델 설정"""
        choice_attributes: List[str]
        choice_type: Literal['binary', 'multinomial', 'ordered'] = 'binary'
        price_variable: str = 'price'
        initial_betas: Optional[Dict[str, float]] = None
        initial_lambda: float = 1.0
        thresholds: Optional[List[float]] = None

logger = logging.getLogger(__name__)


class BinaryProbitChoice:
    """
    Binary Probit 선택모델 (ICLV용)
    
    Model:
        V = intercept + β*X + λ*LV
        P(Yes) = Φ(V)
        P(No) = 1 - Φ(V)
    
    여기서:
        - V: 효용 (Utility)
        - X: 선택 속성 (Choice attributes, e.g., price, quality)
        - β: 속성 계수 (Attribute coefficients)
        - λ: 잠재변수 계수 (Latent variable coefficient)
        - LV: 잠재변수 (Latent Variable)
        - Φ: 표준정규 누적분포함수
    
    King (2022) Apollo R 코드 기반:
        op_settings = list(
            outcomeOrdered = Q6ResearchResponse,
            V = intercept + b_bid*Q6Bid + lambda*LV,
            tau = list(-100, 0),
            componentName = "choice",
            coding = c(-1, 0, 1)
        )
        P[['choice']] = apollo_op(op_settings, functionality)
    
    Usage:
        >>> config = ChoiceConfig(
        ...     choice_attributes=['price', 'quality'],
        ...     choice_type='binary',
        ...     price_variable='price'
        ... )
        >>> model = BinaryProbitChoice(config)
        >>> 
        >>> # Simultaneous 추정용
        >>> ll = model.log_likelihood(data, lv, params)
        >>> 
        >>> # 예측용
        >>> probs = model.predict_probabilities(data, lv, params)
    """
    
    def __init__(self, config: ChoiceConfig):
        """
        초기화
        
        Args:
            config: 선택모델 설정
        """
        self.config = config
        self.choice_attributes = config.choice_attributes
        self.price_variable = config.price_variable
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"BinaryProbitChoice 초기화")
        self.logger.info(f"  선택 속성: {self.choice_attributes}")
        self.logger.info(f"  가격 변수: {self.price_variable}")
    
    def log_likelihood(self, data: pd.DataFrame, lv: np.ndarray,
                      params: Dict) -> float:
        """
        선택모델 로그우도
        
        P(Choice|X, LV) = Φ(V) if choice=1, 1-Φ(V) if choice=0
        
        V = intercept + β*X + λ*LV
        
        Args:
            data: 선택 데이터 (n_obs, n_vars)
                  'choice' 열 필수 (0 or 1)
            lv: 잠재변수 값 (n_obs,) 또는 스칼라
            params: {
                'intercept': float,
                'beta': np.ndarray,  # 속성 계수 (n_attributes,)
                'lambda': float      # 잠재변수 계수
            }
        
        Returns:
            로그우도 값 (스칼라)
        
        Example:
            >>> params = {
            ...     'intercept': 0.5,
            ...     'beta': np.array([-2.0, 0.3]),
            ...     'lambda': 1.5
            ... }
            >>> ll = model.log_likelihood(data, lv, params)
        """
        intercept = params['intercept']
        beta = params['beta']
        lambda_lv = params['lambda']
        
        # 선택 속성 추출
        X = data[self.choice_attributes].values

        # 선택 결과 (0 or 1)
        if 'choice' in data.columns:
            choice = data['choice'].values
        elif 'Choice' in data.columns:
            choice = data['Choice'].values
        else:
            raise ValueError("데이터에 'choice' 또는 'Choice' 열이 없습니다.")

        # 잠재변수 처리 (스칼라 또는 배열)
        if np.isscalar(lv):
            lv_array = np.full(len(data), lv)
        else:
            lv_array = lv

        # 🔴 수정: NaN 처리 (opt-out 대안)
        # NaN이 있는 행은 효용을 0으로 설정 (opt-out 정규화)
        has_nan = np.isnan(X).any(axis=1)

        # 효용 계산
        # V = intercept + β*X + λ*LV
        V = np.zeros(len(data))
        for i in range(len(data)):
            if has_nan[i]:
                V[i] = 0.0  # opt-out: 효용 = 0
            else:
                V[i] = intercept + X[i] @ beta + lambda_lv * lv_array[i]

        # 확률 계산
        # P(Yes) = Φ(V), P(No) = 1 - Φ(V)
        prob_yes = norm.cdf(V)

        # 수치 안정성을 위해 클리핑
        prob_yes = np.clip(prob_yes, 1e-10, 1 - 1e-10)

        # 로그우도
        # log L = Σ [choice * log(Φ(V)) + (1-choice) * log(1-Φ(V))]
        ll = np.sum(
            choice * np.log(prob_yes) +
            (1 - choice) * np.log(1 - prob_yes)
        )

        return ll
    
    def predict_probabilities(self, data: pd.DataFrame, lv: np.ndarray,
                             params: Dict) -> np.ndarray:
        """
        선택 확률 예측
        
        P(Yes) = Φ(intercept + β*X + λ*LV)
        
        Args:
            data: 선택 데이터
            lv: 잠재변수 값
            params: 파라미터 딕셔너리
        
        Returns:
            선택 확률 (n_obs,)
        
        Example:
            >>> probs = model.predict_probabilities(data, lv, params)
        """
        intercept = params['intercept']
        beta = params['beta']
        lambda_lv = params['lambda']
        
        # 선택 속성 추출
        X = data[self.choice_attributes].values
        
        # 잠재변수 처리
        if np.isscalar(lv):
            lv_array = np.full(len(data), lv)
        else:
            lv_array = lv
        
        # 효용 계산
        V = intercept + X @ beta + lambda_lv * lv_array
        
        # 확률 계산
        prob_yes = norm.cdf(V)
        
        return prob_yes
    
    def predict(self, data: pd.DataFrame, lv: np.ndarray,
               params: Dict, threshold: float = 0.5) -> np.ndarray:
        """
        선택 예측
        
        Args:
            data: 선택 데이터
            lv: 잠재변수 값
            params: 파라미터 딕셔너리
            threshold: 선택 임계값 (기본: 0.5)
        
        Returns:
            예측된 선택 (n_obs,) - 0 or 1
        
        Example:
            >>> predictions = model.predict(data, lv, params)
        """
        probs = self.predict_probabilities(data, lv, params)
        predictions = (probs >= threshold).astype(int)
        
        return predictions
    
    def get_initial_params(self, data: pd.DataFrame) -> Dict:
        """
        초기 파라미터 생성
        
        Args:
            data: 선택 데이터
        
        Returns:
            {'intercept': float, 'beta': np.ndarray, 'lambda': float}
        
        Example:
            >>> params = model.get_initial_params(data)
        """
        n_attributes = len(self.choice_attributes)
        
        # 기본 초기값
        params = {
            'intercept': 0.0,
            'beta': np.zeros(n_attributes),
            'lambda': 1.0
        }
        
        # 가격 변수가 있으면 음수로 초기화
        if self.price_variable in self.choice_attributes:
            price_idx = self.choice_attributes.index(self.price_variable)
            params['beta'][price_idx] = -1.0
        
        self.logger.info(f"초기 파라미터: {params}")
        
        return params
    
    def calculate_wtp(self, params: Dict, attribute: str) -> float:
        """
        WTP (Willingness-to-Pay) 계산
        
        WTP = -β_attribute / β_price
        
        Args:
            params: 파라미터 딕셔너리
            attribute: WTP를 계산할 속성
        
        Returns:
            WTP 값
        
        Example:
            >>> wtp = model.calculate_wtp(params, 'quality')
        """
        beta = params['beta']
        
        # 가격 계수
        price_idx = self.choice_attributes.index(self.price_variable)
        beta_price = beta[price_idx]
        
        # 속성 계수
        attr_idx = self.choice_attributes.index(attribute)
        beta_attr = beta[attr_idx]
        
        # WTP = -β_attr / β_price
        wtp = -beta_attr / beta_price
        
        return wtp


def estimate_choice_model(data: pd.DataFrame, latent_var: np.ndarray,
                         choice_attributes: List[str],
                         price_variable: str = 'price',
                         **kwargs) -> Dict:
    """
    선택모델 추정 헬퍼 함수
    
    Args:
        data: 선택 데이터
        latent_var: 잠재변수 값
        choice_attributes: 선택 속성 리스트
        price_variable: 가격 변수명
        **kwargs: 추가 설정
    
    Returns:
        추정 결과
    
    Example:
        >>> results = estimate_choice_model(
        ...     data,
        ...     latent_var,
        ...     choice_attributes=['price', 'quality'],
        ...     price_variable='price'
        ... )
    """
    config = ChoiceConfig(
        choice_attributes=choice_attributes,
        choice_type='binary',
        price_variable=price_variable,
        **kwargs
    )
    
    model = BinaryProbitChoice(config)
    
    # 간단한 추정 (로그우도 최대화)
    initial_params = model.get_initial_params(data)
    
    def negative_log_likelihood(params_array):
        params = {
            'intercept': params_array[0],
            'beta': params_array[1:1+len(choice_attributes)],
            'lambda': params_array[-1]
        }
        return -model.log_likelihood(data, latent_var, params)
    
    # 초기값 배열
    x0 = np.concatenate([
        [initial_params['intercept']],
        initial_params['beta'],
        [initial_params['lambda']]
    ])
    
    # 최적화
    result = minimize(negative_log_likelihood, x0, method='BFGS')
    
    # 결과 정리
    estimated_params = {
        'intercept': result.x[0],
        'beta': result.x[1:1+len(choice_attributes)],
        'lambda': result.x[-1],
        'log_likelihood': -result.fun,
        'success': result.success
    }
    
    return estimated_params

