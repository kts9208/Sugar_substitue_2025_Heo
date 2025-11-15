"""
Likelihood Calculator for ICLV Models

ICLV 모델의 우도 계산 전략 패턴입니다.
동시추정과 순차추정에 따라 다른 우도 계산 방식을 사용합니다.

단일책임 원칙:
- SimultaneousLikelihoodCalculator: 동시추정용 결합 우도
- SequentialLikelihoodCalculator: 순차추정용 단계별 우도
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from scipy.special import logsumexp
import logging

logger = logging.getLogger(__name__)


class BaseLikelihoodCalculator(ABC):
    """
    우도 계산 베이스 클래스
    """
    
    @abstractmethod
    def compute_log_likelihood(self, data: pd.DataFrame, params: np.ndarray,
                               measurement_model, structural_model, 
                               choice_model, **kwargs) -> float:
        """
        로그우도 계산
        
        Args:
            data: 데이터
            params: 파라미터
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
            **kwargs: 추가 인자
        
        Returns:
            로그우도 값
        """
        pass


class SimultaneousLikelihoodCalculator(BaseLikelihoodCalculator):
    """
    동시추정용 우도 계산
    
    결합 우도함수:
    L = ∏ᵢ ∫ P(Choice|LV) × P(Indicators|LV) × P(LV|X) dLV
    
    시뮬레이션 기반 추정:
    L ≈ ∏ᵢ (1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)
    """
    
    def __init__(self, config, individual_id_column: str):
        """
        Args:
            config: ICLVConfig 또는 MultiLatentConfig
            individual_id_column: 개인 ID 컬럼명
        """
        self.config = config
        self.individual_id_column = individual_id_column
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compute_log_likelihood(self, data: pd.DataFrame, params: np.ndarray,
                               measurement_model, structural_model, 
                               choice_model, **kwargs) -> float:
        """
        동시추정 로그우도 계산
        
        Args:
            data: 전체 데이터
            params: 파라미터 벡터
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
            **kwargs: 
                - draws: Halton draws (n_individuals, n_draws) or (n_individuals, n_draws, n_dims)
                - param_dict: 언팩된 파라미터 딕셔너리
        
        Returns:
            전체 로그우도
        """
        draws = kwargs.get('draws')
        param_dict = kwargs.get('param_dict')
        
        if draws is None or param_dict is None:
            raise ValueError("draws와 param_dict가 필요합니다.")
        
        total_ll = 0.0
        individual_ids = data[self.individual_id_column].unique()
        
        # 개인별 우도 계산
        for i, ind_id in enumerate(individual_ids):
            ind_data = data[data[self.individual_id_column] == ind_id]
            ind_draws = draws[i]
            
            person_ll = self._compute_individual_likelihood(
                ind_data, ind_draws, param_dict,
                measurement_model, structural_model, choice_model
            )
            total_ll += person_ll
        
        return total_ll
    
    def _compute_individual_likelihood(self, ind_data: pd.DataFrame,
                                      ind_draws: np.ndarray,
                                      param_dict: Dict,
                                      measurement_model,
                                      structural_model,
                                      choice_model) -> float:
        """
        개인별 우도 계산 (시뮬레이션 기반)
        
        Args:
            ind_data: 개인 데이터
            ind_draws: 개인의 draws
            param_dict: 파라미터 딕셔너리
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
        
        Returns:
            개인의 로그우도
        """
        draw_lls = []
        
        # 각 draw에 대해 우도 계산
        for draw in ind_draws:
            draw_ll = self._compute_single_draw_likelihood(
                ind_data, draw, param_dict,
                measurement_model, structural_model, choice_model
            )
            draw_lls.append(draw_ll)
        
        # logsumexp를 사용하여 평균 계산
        person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))
        
        return person_ll
    
    def _compute_single_draw_likelihood(self, ind_data: pd.DataFrame,
                                       draw: np.ndarray,
                                       param_dict: Dict,
                                       measurement_model,
                                       structural_model,
                                       choice_model) -> float:
        """
        단일 draw에 대한 우도 계산
        
        Args:
            ind_data: 개인 데이터
            draw: 단일 draw
            param_dict: 파라미터 딕셔너리
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
        
        Returns:
            draw의 로그우도
        """
        # 1. 구조모델: LV 예측
        lv = structural_model.predict(ind_data, param_dict['structural'], draw)

        # 2. 측정모델 우도: P(Indicators|LV)
        ll_measurement = measurement_model.log_likelihood(
            ind_data, lv, param_dict['measurement']
        )

        # 3. 선택모델 우도: P(Choice|LV) - Panel Product
        choice_set_lls = []
        for idx in range(len(ind_data)):
            ll_choice_t = choice_model.log_likelihood(
                ind_data.iloc[idx:idx+1],
                lv,
                param_dict['choice']
            )
            choice_set_lls.append(ll_choice_t)

        ll_choice = sum(choice_set_lls)

        # 4. 구조모델 우도: P(LV|X)
        ll_structural = structural_model.log_likelihood(
            ind_data, lv, param_dict['structural'], draw
        )

        # 5. 결합 로그우도
        draw_ll = ll_measurement + ll_choice + ll_structural

        # 비유한 값 처리
        if not np.isfinite(draw_ll):
            draw_ll = -1e10

        return draw_ll


class SequentialLikelihoodCalculator(BaseLikelihoodCalculator):
    """
    순차추정용 우도 계산

    각 단계별로 독립적인 우도를 계산합니다:
    1. 측정모델: P(Indicators|LV)
    2. 구조모델: P(LV|X)
    3. 선택모델: P(Choice|LV)
    """

    def __init__(self, config, individual_id_column: str):
        """
        Args:
            config: ICLVConfig 또는 MultiLatentConfig
            individual_id_column: 개인 ID 컬럼명
        """
        self.config = config
        self.individual_id_column = individual_id_column
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def compute_log_likelihood(self, data: pd.DataFrame, params: np.ndarray,
                               measurement_model, structural_model,
                               choice_model, **kwargs) -> float:
        """
        순차추정 로그우도 계산

        순차추정에서는 이 메서드를 직접 사용하지 않고,
        각 단계별 메서드를 개별적으로 호출합니다.

        Args:
            data: 전체 데이터
            params: 파라미터 벡터 (사용 안 함)
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
            **kwargs:
                - stage: 'measurement', 'structural', 'choice'
                - stage_params: 해당 단계의 파라미터
                - latent_vars: 잠재변수 값 (구조/선택 단계용)

        Returns:
            해당 단계의 로그우도
        """
        stage = kwargs.get('stage')
        stage_params = kwargs.get('stage_params')

        if stage == 'measurement':
            return self.compute_measurement_likelihood(
                data, stage_params, measurement_model, **kwargs
            )
        elif stage == 'structural':
            return self.compute_structural_likelihood(
                data, stage_params, structural_model, **kwargs
            )
        elif stage == 'choice':
            return self.compute_choice_likelihood(
                data, stage_params, choice_model, **kwargs
            )
        else:
            raise ValueError(f"알 수 없는 단계: {stage}")

    def compute_measurement_likelihood(self, data: pd.DataFrame,
                                      params: np.ndarray,
                                      measurement_model,
                                      **kwargs) -> float:
        """
        측정모델 우도 계산

        Args:
            data: 전체 데이터
            params: 측정모델 파라미터
            measurement_model: 측정모델
            **kwargs:
                - latent_vars: 잠재변수 값 (개인별)

        Returns:
            측정모델 로그우도
        """
        latent_vars = kwargs.get('latent_vars')

        if latent_vars is None:
            raise ValueError("latent_vars가 필요합니다.")

        total_ll = 0.0
        individual_ids = data[self.individual_id_column].unique()

        # 개인별 우도 계산
        for i, ind_id in enumerate(individual_ids):
            ind_data = data[data[self.individual_id_column] == ind_id]
            ind_lv = latent_vars[i] if isinstance(latent_vars, np.ndarray) else latent_vars[ind_id]

            ll = measurement_model.log_likelihood(ind_data, ind_lv, params)
            total_ll += ll

        return total_ll

    def compute_structural_likelihood(self, data: pd.DataFrame,
                                     params: np.ndarray,
                                     structural_model,
                                     **kwargs) -> float:
        """
        구조모델 우도 계산

        Args:
            data: 전체 데이터
            params: 구조모델 파라미터
            structural_model: 구조모델
            **kwargs:
                - latent_vars: 잠재변수 값 (개인별)

        Returns:
            구조모델 로그우도
        """
        latent_vars = kwargs.get('latent_vars')

        if latent_vars is None:
            raise ValueError("latent_vars가 필요합니다.")

        total_ll = 0.0
        individual_ids = data[self.individual_id_column].unique()

        # 개인별 우도 계산
        for i, ind_id in enumerate(individual_ids):
            ind_data = data[data[self.individual_id_column] == ind_id]
            ind_lv = latent_vars[i] if isinstance(latent_vars, np.ndarray) else latent_vars[ind_id]

            # 구조모델 우도 (OLS 잔차 기반)
            ll = structural_model.log_likelihood(ind_data, ind_lv, params)
            total_ll += ll

        return total_ll

    def compute_choice_likelihood(self, data: pd.DataFrame,
                                 params: np.ndarray,
                                 choice_model,
                                 **kwargs) -> float:
        """
        선택모델 우도 계산

        Args:
            data: 전체 데이터
            params: 선택모델 파라미터
            choice_model: 선택모델
            **kwargs:
                - latent_vars: 잠재변수 값 (개인별)

        Returns:
            선택모델 로그우도
        """
        latent_vars = kwargs.get('latent_vars')

        if latent_vars is None:
            raise ValueError("latent_vars가 필요합니다.")

        total_ll = 0.0
        individual_ids = data[self.individual_id_column].unique()

        # 개인별 우도 계산
        for i, ind_id in enumerate(individual_ids):
            ind_data = data[data[self.individual_id_column] == ind_id]
            ind_lv = latent_vars[i] if isinstance(latent_vars, np.ndarray) else latent_vars[ind_id]

            # Panel Product: 개인의 여러 선택 상황
            for idx in range(len(ind_data)):
                ll = choice_model.log_likelihood(
                    ind_data.iloc[idx:idx+1],
                    ind_lv,
                    params
                )
                total_ll += ll

        return total_ll

