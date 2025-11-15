"""
Choice Equations for ICLV Models

ICLV 선택모델: 속성 + 잠재변수 → 선택

Based on King (2022) Apollo R code implementation.

Author: Sugar Substitute Research Team
Date: 2025-11-05
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from abc import ABC, abstractmethod
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


class BaseICLVChoice(ABC):
    """
    ICLV 선택모델 베이스 클래스

    공통 기능:
    - 효용 계산 (조절효과 지원)
    - 잠재변수 처리
    - opt-out 대안 처리

    하위 클래스가 구현해야 할 메서드:
    - log_likelihood(): 모델별 확률 계산 및 로그우도
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

        # ✅ 조절효과 설정 (디폴트: 활성화)
        self.moderation_enabled = getattr(config, 'moderation_enabled', False)
        self.moderator_lvs = getattr(config, 'moderator_lvs', None)
        self.main_lv = getattr(config, 'main_lv', 'purchase_intention')

        self.logger = logging.getLogger(__name__)

        self.logger.info(f"{self.__class__.__name__} 초기화")
        self.logger.info(f"  선택 속성: {self.choice_attributes}")
        self.logger.info(f"  가격 변수: {self.price_variable}")
        self.logger.info(f"  조절효과: {self.moderation_enabled}")
        if self.moderation_enabled:
            self.logger.info(f"  주 LV: {self.main_lv}")
            self.logger.info(f"  조절 LV: {self.moderator_lvs}")

    def _compute_utilities(self, data: pd.DataFrame, lv, params: Dict) -> np.ndarray:
        """
        효용 계산 (공통 로직)

        ✅ 조절효과 지원

        기본 모델:
            V = intercept + β*X + λ*LV

        조절효과 모델 (디폴트):
            V = intercept + β*X + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)

        Args:
            data: 선택 데이터
            lv: 잠재변수 값
                - Dict[str, np.ndarray] (조절효과 모델)
                - np.ndarray 또는 float (기본 모델)
            params: 파라미터 딕셔너리

        Returns:
            효용 벡터 (n_obs,)
        """
        intercept = params['intercept']
        beta = params['beta']

        # 선택 속성 추출
        X = data[self.choice_attributes].values

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

        return V

    @abstractmethod
    def log_likelihood(self, data: pd.DataFrame, lv, params: Dict) -> float:
        """
        선택모델 로그우도 (하위 클래스에서 구현)

        Args:
            data: 선택 데이터
            lv: 잠재변수 값
            params: 파라미터 딕셔너리

        Returns:
            로그우도 값
        """
        pass


class BinaryProbitChoice(BaseICLVChoice):
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
        # 베이스 클래스 초기화
        super().__init__(config)
    
    def log_likelihood(self, data: pd.DataFrame, lv,
                      params: Dict) -> float:
        """
        Binary Probit 로그우도

        ✅ 조절효과 지원

        기본 모델:
            V = intercept + β*X + λ*LV
            P(Yes) = Φ(V)

        조절효과 모델 (디폴트):
            V = intercept + β*X + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)
            P(Yes) = Φ(V)

        Args:
            data: 선택 데이터 (n_obs, n_vars)
                  'choice' 열 필수 (0 or 1)
            lv: 잠재변수 값
                - 기본 모델: (n_obs,) 또는 스칼라
                - 조절효과 모델: Dict[str, np.ndarray] 또는 Dict[str, float]
            params: 파라미터 딕셔너리

        Returns:
            로그우도 값 (스칼라)
        """
        # 선택 결과 (0 or 1)
        if 'choice' in data.columns:
            choice = data['choice'].values
        elif 'Choice' in data.columns:
            choice = data['Choice'].values
        else:
            raise ValueError("데이터에 'choice' 또는 'Choice' 열이 없습니다.")

        # ✅ 베이스 클래스의 효용 계산 사용
        V = self._compute_utilities(data, lv, params)

        # 확률 계산: P(Yes) = Φ(V)
        prob_yes = norm.cdf(V)

        # 수치 안정성을 위해 클리핑
        prob_yes = np.clip(prob_yes, 1e-10, 1 - 1e-10)

        # 로그우도: log L = Σ [choice * log(Φ(V)) + (1-choice) * log(1-Φ(V))]
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


class MultinomialLogitChoice(BaseICLVChoice):
    """
    Multinomial Logit 선택모델 (ICLV용)

    ✅ BinaryProbitChoice와 동일한 인터페이스
    ✅ 조절효과 지원
    ✅ GPU 배치 처리 호환

    모델:
        V_j = intercept + β*X_j + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)
        P(j) = exp(V_j) / Σ_k exp(V_k)

    여기서:
        - j: 대안 인덱스 (제품A, 제품B, 구매안함)
        - V_j: 대안 j의 효용
        - P(j): 대안 j의 선택 확률

    데이터 구조:
        - 각 선택 상황은 3개 행 (제품A, 제품B, 구매안함)
        - choice 컬럼: 선택된 대안은 1, 나머지는 0
        - opt-out 대안: 속성이 NaN → 효용 = 0 (기준 대안)

    Usage:
        >>> config = ChoiceConfig(
        ...     choice_attributes=['sugar_free', 'health_label', 'price'],
        ...     choice_type='multinomial',
        ...     moderation_enabled=True,
        ...     moderator_lvs=['perceived_price', 'nutrition_knowledge'],
        ...     main_lv='purchase_intention'
        ... )
        >>> model = MultinomialLogitChoice(config)
        >>> ll = model.log_likelihood(data, lv_dict, params)
    """

    def __init__(self, config: ChoiceConfig):
        """
        초기화

        Args:
            config: 선택모델 설정
        """
        # 베이스 클래스 초기화
        super().__init__(config)

        # MNL 특화 설정
        self.n_alternatives = 3  # 제품A, 제품B, 구매안함

        self.logger.info(f"  대안 수: {self.n_alternatives}")

    def log_likelihood(self, data: pd.DataFrame, lv, params: Dict) -> float:
        """
        Multinomial Logit 로그우도

        ✅ 조절효과 지원

        모델:
            V_j = intercept + β*X_j + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)
            P(j) = exp(V_j) / Σ_k exp(V_k)

        Args:
            data: 선택 데이터
                  각 선택 상황은 n_alternatives개 행
                  'choice' 열: 선택된 대안은 1, 나머지는 0
            lv: 잠재변수 값
                - Dict[str, np.ndarray] (조절효과 모델)
                - np.ndarray 또는 float (기본 모델)
            params: 파라미터 딕셔너리

        Returns:
            로그우도 값 (스칼라)
        """
        # 선택 결과 (0 or 1)
        if 'choice' in data.columns:
            choice = data['choice'].values
        elif 'Choice' in data.columns:
            choice = data['Choice'].values
        else:
            raise ValueError("데이터에 'choice' 또는 'Choice' 열이 없습니다.")

        # ✅ 베이스 클래스의 효용 계산 사용
        V = self._compute_utilities(data, lv, params)

        # 선택 상황별로 그룹화하여 Softmax 계산
        n_rows = len(data)
        n_choice_situations = n_rows // self.n_alternatives

        total_ll = 0.0

        for i in range(n_choice_situations):
            start_idx = i * self.n_alternatives
            end_idx = start_idx + self.n_alternatives

            # 이 선택 상황의 효용들
            V_situation = V[start_idx:end_idx]  # (n_alternatives,)

            # Softmax 확률 계산 (수치 안정성)
            V_max = np.max(V_situation)
            exp_V = np.exp(V_situation - V_max)
            prob = exp_V / np.sum(exp_V)

            # 수치 안정성을 위해 클리핑
            prob = np.clip(prob, 1e-10, 1 - 1e-10)

            # 선택된 대안 찾기
            choices = choice[start_idx:end_idx]
            chosen_idx = np.argmax(choices)

            # 로그우도 누적
            total_ll += np.log(prob[chosen_idx])

        return total_ll

    def predict_probabilities(self, data: pd.DataFrame, lv, params: Dict) -> np.ndarray:
        """
        선택 확률 예측

        Args:
            data: 선택 데이터
            lv: 잠재변수 값
            params: 파라미터 딕셔너리

        Returns:
            선택 확률 배열 (n_obs,)
        """
        # 효용 계산
        V = self._compute_utilities(data, lv, params)

        # 선택 상황별로 Softmax 계산
        n_rows = len(data)
        n_choice_situations = n_rows // self.n_alternatives

        probabilities = np.zeros(n_rows)

        for i in range(n_choice_situations):
            start_idx = i * self.n_alternatives
            end_idx = start_idx + self.n_alternatives

            # 이 선택 상황의 효용들
            V_situation = V[start_idx:end_idx]

            # Softmax 확률
            V_max = np.max(V_situation)
            exp_V = np.exp(V_situation - V_max)
            prob = exp_V / np.sum(exp_V)

            # 확률 저장
            probabilities[start_idx:end_idx] = prob

        return probabilities

    def fit(self, data: pd.DataFrame, factor_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        선택모델 추정 (순차추정 Step 2)

        ✅ 요인점수를 독립변수로 사용
        ✅ 조절효과 지원

        효용함수:
            V = intercept + β*X + λ_main*PI + λ_mod_PP*(PI×PP) + λ_mod_NK*(PI×NK)

        여기서:
            - X: 선택 속성 (sugar_free, health_label, price)
            - PI: 구매의도 요인점수 (주효과)
            - PP: 지각된 가격 요인점수 (조절효과)
            - NK: 영양지식 요인점수 (조절효과)

        Args:
            data: 선택 데이터
                  각 선택 상황은 3개 행 (제품A, 제품B, 구매안함)
                  'choice' 열: 선택된 대안은 1, 나머지는 0
            factor_scores: SEM에서 추출한 요인점수
                {
                    'purchase_intention': np.ndarray (n_individuals,),
                    'perceived_price': np.ndarray (n_individuals,),
                    'nutrition_knowledge': np.ndarray (n_individuals,)
                }

        Returns:
            {
                'params': 추정된 파라미터,
                'log_likelihood': 로그우도,
                'aic': AIC,
                'bic': BIC,
                'success': 성공 여부,
                'message': 최적화 메시지,
                'n_iterations': 반복 횟수
            }

        Example:
            >>> config = ChoiceConfig(
            ...     choice_attributes=['sugar_free', 'health_label', 'price'],
            ...     moderation_enabled=True,
            ...     moderator_lvs=['perceived_price', 'nutrition_knowledge'],
            ...     main_lv='purchase_intention'
            ... )
            >>> model = MultinomialLogitChoice(config)
            >>> results = model.fit(choice_data, factor_scores)
        """
        self.logger.info("=" * 70)
        self.logger.info("선택모델 추정 시작 (MultinomialLogitChoice)")
        self.logger.info("=" * 70)

        # 1. 데이터 검증
        if 'choice' not in data.columns and 'Choice' not in data.columns:
            raise ValueError("데이터에 'choice' 또는 'Choice' 열이 없습니다.")

        # 2. 요인점수를 개인별로 복제 (각 선택 상황마다 n_alternatives개 행)
        # 선택 데이터는 (n_individuals * n_choice_sets * n_alternatives) 행
        # 요인점수는 (n_individuals,) 배열
        # 각 개인의 모든 선택 상황에 동일한 요인점수 사용

        n_rows = len(data)
        n_individuals = len(next(iter(factor_scores.values())))

        self.logger.info(f"요인점수 확장:")
        self.logger.info(f"  전체 데이터 행 수: {n_rows}")
        self.logger.info(f"  개인 수: {n_individuals}")

        # respondent_id 기준으로 요인점수 매핑 (부트스트랩 안전)
        if 'respondent_id' in data.columns:
            # 개인 ID 추출
            unique_ids = data['respondent_id'].unique()

            # 요인점수를 ID 순서대로 매핑
            lv_expanded = {}
            for lv_name, scores in factor_scores.items():
                # 각 행의 respondent_id에 해당하는 요인점수 할당
                id_to_score = {unique_ids[i]: scores[i] for i in range(len(unique_ids))}
                expanded = np.array([id_to_score[rid] for rid in data['respondent_id']])
                lv_expanded[lv_name] = expanded
                self.logger.info(f"  {lv_name}: {scores.shape} → {expanded.shape}")
        else:
            # respondent_id가 없는 경우 (하위 호환)
            rows_per_individual = n_rows // n_individuals
            self.logger.info(f"  개인당 행 수: {rows_per_individual}")

            lv_expanded = {}
            for lv_name, scores in factor_scores.items():
                expanded = np.repeat(scores, rows_per_individual)
                lv_expanded[lv_name] = expanded
                self.logger.info(f"  {lv_name}: {scores.shape} → {expanded.shape}")

        # 3. 초기 파라미터 생성
        initial_params = self.get_initial_params(data)
        self.logger.info(f"초기 파라미터: {initial_params}")

        # 4. 파라미터를 배열로 변환 (최적화용)
        param_names, x0 = self._params_to_array(initial_params)
        self.logger.info(f"최적화 파라미터 개수: {len(x0)}")

        # 5. 목적함수 정의 (음의 로그우도) + 반복 로깅
        iteration_count = [0]  # 리스트로 감싸서 클로저에서 수정 가능하게

        def negative_log_likelihood(params_array):
            params = self._array_to_params(param_names, params_array)
            ll = self.log_likelihood(data, lv_expanded, params)
            nll = -ll

            # 반복 로깅 (10회마다)
            iteration_count[0] += 1
            if iteration_count[0] % 10 == 0 or iteration_count[0] == 1:
                print(f"  반복 {iteration_count[0]:3d}: NLL = {nll:12.4f}, LL = {ll:12.4f}")
                self.logger.info(f"  반복 {iteration_count[0]:3d}: NLL = {nll:12.4f}, LL = {ll:12.4f}")

            return nll

        # 6. 최적화 실행
        print(f"\n[선택모델 최적화 시작] method=L-BFGS-B")
        self.logger.info("최적화 시작 (method=L-BFGS-B)...")

        initial_nll = negative_log_likelihood(x0)
        print(f"  초기 NLL: {initial_nll:.4f}")
        self.logger.info(f"  초기 NLL: {initial_nll:.4f}")

        # 반복 카운터 초기화
        iteration_count[0] = 0

        result = minimize(
            negative_log_likelihood,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False}  # disp=False로 변경 (우리가 직접 로깅)
        )

        # 최적화 완료 로깅
        print(f"\n[선택모델 최적화 완료]")
        print(f"  총 반복 횟수: {result.nit}")
        print(f"  함수 평가 횟수: {result.nfev}")
        print(f"  최종 LL: {-result.fun:.4f}")
        print(f"  수렴 여부: {result.success}")
        print(f"  메시지: {result.message}")

        self.logger.info(f"최적화 완료:")
        self.logger.info(f"  총 반복 횟수: {result.nit}")
        self.logger.info(f"  함수 평가 횟수: {result.nfev}")
        self.logger.info(f"  최종 LL: {-result.fun:.4f}")
        self.logger.info(f"  수렴 여부: {result.success}")
        self.logger.info(f"  메시지: {result.message}")

        # 7. 결과 정리
        estimated_params = self._array_to_params(param_names, result.x)
        log_likelihood = -result.fun
        n_params = len(result.x)
        n_choice_situations = n_rows // self.n_alternatives
        n_obs = n_choice_situations

        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_obs) - 2 * log_likelihood

        results = {
            'params': estimated_params,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'n_params': n_params,
            'n_obs': n_obs,
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit
        }

        # 8. 표준오차 및 p-value 계산 (동시추정 모듈과 동일한 방식)
        try:
            if hasattr(result, 'hess_inv'):
                hess_inv = result.hess_inv
                if hasattr(hess_inv, 'todense'):
                    hess_inv = hess_inv.todense()

                # Hessian 역행렬 저장
                results['hessian_inv'] = np.array(hess_inv)

                # 대각 원소 추출 (분산)
                variances = np.diag(hess_inv)

                # 음수 분산 처리 (수치 오류)
                variances = np.maximum(variances, 1e-10)

                # 표준오차
                se = np.sqrt(variances)
                results['standard_errors'] = se

                # t-통계량
                results['t_statistics'] = result.x / se

                # p-값 (양측 검정, 대표본이므로 정규분포 사용)
                from scipy.stats import norm
                results['p_values'] = 2 * (1 - norm.cdf(np.abs(results['t_statistics'])))

                # 파라미터별로 구조화
                results['parameter_statistics'] = self._structure_statistics(
                    param_names, result.x, se, results['t_statistics'], results['p_values']
                )

                self.logger.info("표준오차 및 p-value 계산 완료")

            else:
                self.logger.warning("Hessian 정보가 없어 표준오차를 계산할 수 없습니다.")
                results['hessian_inv'] = None
                results['standard_errors'] = None
                results['t_statistics'] = None
                results['p_values'] = None
                results['parameter_statistics'] = None

        except Exception as e:
            self.logger.warning(f"표준오차 계산 실패: {e}")
            results['hessian_inv'] = None
            results['standard_errors'] = None
            results['t_statistics'] = None
            results['p_values'] = None
            results['parameter_statistics'] = None

        self.logger.info("=" * 70)
        self.logger.info("선택모델 추정 완료")
        self.logger.info(f"  로그우도: {log_likelihood:.2f}")
        self.logger.info(f"  AIC: {aic:.2f}")
        self.logger.info(f"  BIC: {bic:.2f}")
        self.logger.info(f"  성공: {result.success}")
        self.logger.info("=" * 70)

        return results

    def get_initial_params(self, data: pd.DataFrame) -> Dict:
        """
        초기 파라미터 생성

        ✅ 조절효과 지원

        Args:
            data: 선택 데이터

        Returns:
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

    def _params_to_array(self, params: Dict) -> Tuple[List[str], np.ndarray]:
        """
        파라미터 딕셔너리를 배열로 변환 (최적화용)

        Args:
            params: 파라미터 딕셔너리

        Returns:
            (param_names, param_array)
        """
        param_names = []
        param_values = []

        # intercept
        param_names.append('intercept')
        param_values.append(params['intercept'])

        # beta
        for i, attr in enumerate(self.choice_attributes):
            param_names.append(f'beta_{attr}')
            param_values.append(params['beta'][i])

        # lambda_main 또는 lambda
        if 'lambda_main' in params:
            param_names.append('lambda_main')
            param_values.append(params['lambda_main'])
        elif 'lambda' in params:
            param_names.append('lambda')
            param_values.append(params['lambda'])

        # lambda_mod_*
        if self.moderation_enabled:
            for mod_lv in self.moderator_lvs:
                param_name = f'lambda_mod_{mod_lv}'
                if param_name in params:
                    param_names.append(param_name)
                    param_values.append(params[param_name])

        return param_names, np.array(param_values)

    def _array_to_params(self, param_names: List[str], param_array: np.ndarray) -> Dict:
        """
        배열을 파라미터 딕셔너리로 변환

        Args:
            param_names: 파라미터 이름 리스트
            param_array: 파라미터 값 배열

        Returns:
            파라미터 딕셔너리
        """
        params = {}

        # beta 수집용
        beta_values = []

        for name, value in zip(param_names, param_array):
            if name == 'intercept':
                params['intercept'] = value
            elif name.startswith('beta_'):
                beta_values.append(value)
            elif name == 'lambda_main':
                params['lambda_main'] = value
            elif name == 'lambda':
                params['lambda'] = value
            elif name.startswith('lambda_mod_'):
                params[name] = value

        # beta 배열로 변환
        if beta_values:
            params['beta'] = np.array(beta_values)

        return params

    def _structure_statistics(self, param_names: List[str],
                             estimates: np.ndarray,
                             std_errors: np.ndarray,
                             t_stats: np.ndarray,
                             p_values: np.ndarray) -> Dict:
        """
        파라미터별 통계량을 구조화된 딕셔너리로 변환

        Args:
            param_names: 파라미터 이름 리스트
            estimates: 추정값 배열
            std_errors: 표준오차 배열
            t_stats: t-통계량 배열
            p_values: p-value 배열

        Returns:
            구조화된 통계량 딕셔너리
            {
                'intercept': {'estimate': ..., 'se': ..., 't': ..., 'p': ...},
                'beta': {
                    'sugar_free': {'estimate': ..., 'se': ..., 't': ..., 'p': ...},
                    'health_label': {...},
                    'price': {...}
                },
                'lambda_main': {...},
                'lambda_mod_perceived_price': {...},
                'lambda_mod_nutrition_knowledge': {...}
            }
        """
        stats = {}

        for i, name in enumerate(param_names):
            stat_dict = {
                'estimate': estimates[i],
                'se': std_errors[i],
                't': t_stats[i],
                'p': p_values[i]
            }

            if name == 'intercept':
                stats['intercept'] = stat_dict
            elif name.startswith('beta_'):
                # beta 파라미터는 속성별로 그룹화
                if 'beta' not in stats:
                    stats['beta'] = {}
                attr_name = name.replace('beta_', '')
                stats['beta'][attr_name] = stat_dict
            elif name == 'lambda_main':
                stats['lambda_main'] = stat_dict
            elif name == 'lambda':
                stats['lambda'] = stat_dict
            elif name.startswith('lambda_mod_'):
                stats[name] = stat_dict

        return stats

