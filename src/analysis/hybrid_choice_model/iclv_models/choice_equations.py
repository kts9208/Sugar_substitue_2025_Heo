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

    ✅ 디폴트: 조절효과 활성화

    기본 모델:
        V = intercept + β*X + λ*LV
        P(Yes) = Φ(V)

    조절효과 모델 (디폴트):
        V = intercept + β*X + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)
        P(Yes) = Φ(V)

    여기서:
        - V: 효용 (Utility)
        - X: 선택 속성 (Choice attributes, e.g., price, quality)
        - β: 속성 계수 (Attribute coefficients)
        - λ_main: 주효과 계수 (Main effect coefficient)
        - λ_mod_i: 조절효과 계수 (Moderation effect coefficients)
        - LV_main: 주 잠재변수 (Main latent variable, e.g., purchase_intention)
        - LV_mod_i: 조절 잠재변수 (Moderator latent variables, e.g., perceived_price, nutrition_knowledge)
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
        ...     price_variable='price',
        ...     moderation_enabled=True,
        ...     moderator_lvs=['perceived_price', 'nutrition_knowledge'],
        ...     main_lv='purchase_intention'
        ... )
        >>> model = BinaryProbitChoice(config)
        >>>
        >>> # Simultaneous 추정용
        >>> ll = model.log_likelihood(data, lv_dict, params)
        >>>
        >>> # 예측용
        >>> probs = model.predict_probabilities(data, lv_dict, params)
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

        # ✅ 조절효과 설정
        self.moderation_enabled = getattr(config, 'moderation_enabled', False)
        self.moderator_lvs = getattr(config, 'moderator_lvs', None)
        self.main_lv = getattr(config, 'main_lv', 'purchase_intention')

        self.logger = logging.getLogger(__name__)

        self.logger.info(f"BinaryProbitChoice 초기화")
        self.logger.info(f"  선택 속성: {self.choice_attributes}")
        self.logger.info(f"  가격 변수: {self.price_variable}")
        self.logger.info(f"  조절효과: {self.moderation_enabled}")
        if self.moderation_enabled:
            self.logger.info(f"  주 LV: {self.main_lv}")
            self.logger.info(f"  조절 LV: {self.moderator_lvs}")
    
    def log_likelihood(self, data: pd.DataFrame, lv,
                      params: Dict) -> float:
        """
        선택모델 로그우도

        ✅ 조절효과 지원

        기본 모델:
            V = intercept + β*X + λ*LV

        조절효과 모델 (디폴트):
            V = intercept + β*X + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)

        Args:
            data: 선택 데이터 (n_obs, n_vars)
                  'choice' 열 필수 (0 or 1)
            lv: 잠재변수 값
                - 기본 모델: (n_obs,) 또는 스칼라
                - 조절효과 모델: Dict[str, np.ndarray] 또는 Dict[str, float]
                  예: {
                      'purchase_intention': np.array([0.5, 0.3, ...]),
                      'perceived_price': np.array([-0.2, 0.1, ...]),
                      'nutrition_knowledge': np.array([0.8, 0.6, ...])
                  }
            params: 파라미터
                - 기본 모델: {
                    'intercept': float,
                    'beta': np.ndarray,
                    'lambda': float
                  }
                - 조절효과 모델: {
                    'intercept': float,
                    'beta': np.ndarray,
                    'lambda_main': float,
                    'lambda_mod_perceived_price': float,
                    'lambda_mod_nutrition_knowledge': float
                  }

        Returns:
            로그우도 값 (스칼라)

        Example (조절효과):
            >>> params = {
            ...     'intercept': 0.5,
            ...     'beta': np.array([-2.0, 0.3, 0.5]),
            ...     'lambda_main': 1.0,
            ...     'lambda_mod_perceived_price': -0.3,
            ...     'lambda_mod_nutrition_knowledge': 0.2
            ... }
            >>> lv_dict = {
            ...     'purchase_intention': np.array([0.5, 0.3]),
            ...     'perceived_price': np.array([-0.2, 0.1]),
            ...     'nutrition_knowledge': np.array([0.8, 0.6])
            ... }
            >>> ll = model.log_likelihood(data, lv_dict, params)
        """
        intercept = params['intercept']
        beta = params['beta']

        # 선택 속성 추출
        X = data[self.choice_attributes].values

        # 선택 결과 (0 or 1)
        if 'choice' in data.columns:
            choice = data['choice'].values
        elif 'Choice' in data.columns:
            choice = data['Choice'].values
        else:
            raise ValueError("데이터에 'choice' 또는 'Choice' 열이 없습니다.")

        # NaN 처리 (opt-out 대안)
        has_nan = np.isnan(X).any(axis=1)

        # 효용 계산
        V = np.zeros(len(data))

        if self.moderation_enabled and isinstance(lv, dict):
            # ✅ 조절효과 모델
            lambda_main = params.get('lambda_main', params.get('lambda', 1.0))

            # 주 LV 추출
            lv_main = lv[self.main_lv]
            if np.isscalar(lv_main):
                lv_main_array = np.full(len(data), lv_main)
            else:
                lv_main_array = lv_main

            # 조절 LV 추출
            moderator_arrays = {}
            for mod_lv in self.moderator_lvs:
                lv_mod = lv[mod_lv]
                if np.isscalar(lv_mod):
                    moderator_arrays[mod_lv] = np.full(len(data), lv_mod)
                else:
                    moderator_arrays[mod_lv] = lv_mod

            # 효용 계산: V = intercept + β*X + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)
            for i in range(len(data)):
                if has_nan[i]:
                    V[i] = 0.0  # opt-out: 효용 = 0
                else:
                    # 기본 효용
                    V[i] = intercept + X[i] @ beta + lambda_main * lv_main_array[i]

                    # 조절효과 추가
                    for mod_lv in self.moderator_lvs:
                        param_name = f'lambda_mod_{mod_lv}'
                        if param_name in params:
                            lambda_mod = params[param_name]
                            interaction = lv_main_array[i] * moderator_arrays[mod_lv][i]
                            V[i] += lambda_mod * interaction

        else:
            # 기본 모델 (하위 호환)
            lambda_lv = params.get('lambda', params.get('lambda_main', 1.0))

            # 잠재변수 처리 (스칼라 또는 배열)
            if isinstance(lv, dict):
                # dict인 경우 main_lv 사용
                lv_value = lv[self.main_lv]
            else:
                lv_value = lv

            if np.isscalar(lv_value):
                lv_array = np.full(len(data), lv_value)
            else:
                lv_array = lv_value

            # 효용 계산: V = intercept + β*X + λ*LV
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
    
    def predict_probabilities(self, data: pd.DataFrame, lv,
                             params: Dict) -> np.ndarray:
        """
        선택 확률 예측

        ✅ 조절효과 지원

        기본 모델:
            P(Yes) = Φ(intercept + β*X + λ*LV)

        조절효과 모델:
            P(Yes) = Φ(intercept + β*X + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i))

        Args:
            data: 선택 데이터
            lv: 잠재변수 값 (스칼라/배열 또는 딕셔너리)
            params: 파라미터 딕셔너리

        Returns:
            선택 확률 (n_obs,)

        Example:
            >>> probs = model.predict_probabilities(data, lv_dict, params)
        """
        intercept = params['intercept']
        beta = params['beta']

        # 선택 속성 추출
        X = data[self.choice_attributes].values

        # 효용 계산
        if self.moderation_enabled and isinstance(lv, dict):
            # ✅ 조절효과 모델
            lambda_main = params.get('lambda_main', params.get('lambda', 1.0))

            # 주 LV 추출
            lv_main = lv[self.main_lv]
            if np.isscalar(lv_main):
                lv_main_array = np.full(len(data), lv_main)
            else:
                lv_main_array = lv_main

            # 조절 LV 추출
            moderator_arrays = {}
            for mod_lv in self.moderator_lvs:
                lv_mod = lv[mod_lv]
                if np.isscalar(lv_mod):
                    moderator_arrays[mod_lv] = np.full(len(data), lv_mod)
                else:
                    moderator_arrays[mod_lv] = lv_mod

            # 기본 효용
            V = intercept + X @ beta + lambda_main * lv_main_array

            # 조절효과 추가
            for mod_lv in self.moderator_lvs:
                param_name = f'lambda_mod_{mod_lv}'
                if param_name in params:
                    lambda_mod = params[param_name]
                    interaction = lv_main_array * moderator_arrays[mod_lv]
                    V += lambda_mod * interaction

        else:
            # 기본 모델 (하위 호환)
            lambda_lv = params.get('lambda', params.get('lambda_main', 1.0))

            # 잠재변수 처리
            if isinstance(lv, dict):
                lv_value = lv[self.main_lv]
            else:
                lv_value = lv

            if np.isscalar(lv_value):
                lv_array = np.full(len(data), lv_value)
            else:
                lv_array = lv_value

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

        ✅ 조절효과 지원

        Args:
            data: 선택 데이터

        Returns:
            기본 모델:
                {'intercept': float, 'beta': np.ndarray, 'lambda': float}

            조절효과 모델:
                {
                    'intercept': float,
                    'beta': np.ndarray,
                    'lambda_main': float,
                    'lambda_mod_perceived_price': float,
                    'lambda_mod_nutrition_knowledge': float
                }

        Example:
            >>> params = model.get_initial_params(data)
        """
        n_attributes = len(self.choice_attributes)

        # 기본 초기값
        params = {
            'intercept': 0.0,
            'beta': np.zeros(n_attributes)
        }

        # 가격 변수가 있으면 음수로 초기화
        if self.price_variable in self.choice_attributes:
            price_idx = self.choice_attributes.index(self.price_variable)
            params['beta'][price_idx] = -1.0

        if self.moderation_enabled:
            # ✅ 조절효과 모델
            params['lambda_main'] = 1.0

            # 조절효과 초기값
            for mod_lv in self.moderator_lvs:
                param_name = f'lambda_mod_{mod_lv}'
                # 가격은 부적 조절, 지식은 정적 조절 가정
                if 'price' in mod_lv.lower():
                    params[param_name] = -0.3
                elif 'knowledge' in mod_lv.lower():
                    params[param_name] = 0.2
                else:
                    params[param_name] = 0.0
        else:
            # 기본 모델 (하위 호환)
            params['lambda'] = 1.0

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

