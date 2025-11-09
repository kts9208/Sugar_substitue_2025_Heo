"""
Multi-Latent Variable Structural Model

다중 잠재변수 구조모델입니다.
외생 잠재변수와 내생 잠재변수의 관계를 모델링합니다.

구조:
- 외생 LV: LV_i = η_i ~ N(0, 1)
- 내생 LV: LV_endo = Σ(γ_i * LV_i) + Σ(γ_j * X_j) + η

Author: Sugar Substitute Research Team
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import norm
import logging

from .multi_latent_config import MultiLatentStructuralConfig

logger = logging.getLogger(__name__)


class MultiLatentStructural:
    """
    다중 잠재변수 구조모델
    
    구조방정식:
    - 외생 LV (4개): LV_i = η_i ~ N(0, 1)
    - 내생 LV (1개): LV_endo = Σ(γ_i * LV_i) + Σ(γ_j * X_j) + η
    
    Example:
        외생 LV:
            건강관심도 = η1 ~ N(0, 1)
            건강유익성 = η2 ~ N(0, 1)
            가격수준 = η3 ~ N(0, 1)
            영양지식 = η4 ~ N(0, 1)
        
        내생 LV:
            구매의도 = γ1*건강관심도 + γ2*건강유익성 + γ3*가격수준 + γ4*영양지식
                     + γ5*age + γ6*gender + γ7*income + γ8*education + η
    """
    
    def __init__(self, config: MultiLatentStructuralConfig):
        """
        초기화
        
        Args:
            config: 다중 잠재변수 구조모델 설정
        """
        self.config = config
        self.endogenous_lv = config.endogenous_lv
        self.exogenous_lvs = config.exogenous_lvs
        self.covariates = config.covariates
        self.error_variance = config.error_variance
        
        self.n_exo = len(self.exogenous_lvs)
        self.n_cov = len(self.covariates)
        self.n_params = self.n_exo + self.n_cov
        
        logger.info(f"MultiLatentStructural 초기화")
        logger.info(f"  외생 LV ({self.n_exo}개): {self.exogenous_lvs}")
        logger.info(f"  내생 LV: {self.endogenous_lv}")
        logger.info(f"  공변량 ({self.n_cov}개): {self.covariates}")
        logger.info(f"  총 파라미터: {self.n_params}개")
    
    def predict(self, data: pd.DataFrame,
                exo_draws: np.ndarray,
                params: Dict[str, np.ndarray],
                endo_draw: float) -> Dict[str, float]:
        """
        모든 잠재변수 예측
        
        Args:
            data: 개인 데이터 (첫 번째 행의 공변량 사용)
            exo_draws: 외생 LV draws (n_exo,) - 표준정규분포
            params: 구조모델 파라미터
                {
                    'gamma_lv': np.ndarray (n_exo,),  # 외생 LV 계수
                    'gamma_x': np.ndarray (n_cov,)    # 공변량 계수
                }
            endo_draw: 내생 LV 오차항 draw - 표준정규분포
        
        Returns:
            모든 잠재변수 값
            {
                'health_concern': 0.5,
                'perceived_benefit': 0.3,
                'perceived_price': -0.2,
                'nutrition_knowledge': 0.8,
                'purchase_intention': 0.6
            }
        """
        latent_vars = {}
        
        # 1. 외생 LV (표준정규분포)
        for i, lv_name in enumerate(self.exogenous_lvs):
            latent_vars[lv_name] = exo_draws[i]
        
        # 2. 내생 LV
        gamma_lv = params['gamma_lv']
        gamma_x = params['gamma_x']
        
        # 외생 LV 효과
        lv_effect = np.sum(gamma_lv * exo_draws)
        
        # 공변량 효과 (첫 번째 행 사용 - 개인 특성)
        first_row = data.iloc[0]
        x_effect = 0.0
        for i, var in enumerate(self.covariates):
            if var in first_row.index:
                value = first_row[var]
                if pd.isna(value):
                    value = 0.0
                x_effect += gamma_x[i] * value
        
        # 내생 LV = 외생 LV 효과 + 공변량 효과 + 오차항
        latent_vars[self.endogenous_lv] = (
            lv_effect + x_effect + np.sqrt(self.error_variance) * endo_draw
        )
        
        return latent_vars
    
    def log_likelihood(self, data: pd.DataFrame,
                      latent_vars: Dict[str, float],
                      exo_draws: np.ndarray,
                      params: Dict[str, np.ndarray],
                      endo_draw: float) -> float:
        """
        구조모델 로그우도
        
        LL = Σ log P(LV_exo) + log P(LV_endo | LV_exo, X)
        
        외생 LV: P(LV_i) = N(0, 1)
        내생 LV: P(LV_endo | LV_exo, X) = N(γ_lv*LV_exo + γ_x*X, σ²)
        
        Args:
            data: 개인 데이터
            latent_vars: 모든 잠재변수 값
            exo_draws: 외생 LV draws
            params: 구조모델 파라미터
            endo_draw: 내생 LV 오차항 draw
        
        Returns:
            구조모델 로그우도
        """
        ll = 0.0
        
        # 1. 외생 LV 로그우도: N(0, 1)
        for lv_name in self.exogenous_lvs:
            lv = latent_vars[lv_name]
            ll += norm.logpdf(lv, loc=0, scale=1)
        
        # 2. 내생 LV 로그우도: N(γ_lv*LV_exo + γ_x*X, σ²)
        gamma_lv = params['gamma_lv']
        gamma_x = params['gamma_x']
        
        # 평균 계산
        lv_effect = np.sum(gamma_lv * exo_draws)
        
        first_row = data.iloc[0]
        x_effect = 0.0
        for i, var in enumerate(self.covariates):
            if var in first_row.index:
                value = first_row[var]
                if pd.isna(value):
                    value = 0.0
                x_effect += gamma_x[i] * value
        
        lv_endo_mean = lv_effect + x_effect
        lv_endo = latent_vars[self.endogenous_lv]
        
        ll += norm.logpdf(lv_endo, loc=lv_endo_mean, scale=np.sqrt(self.error_variance))
        
        return ll
    
    def get_n_parameters(self) -> int:
        """
        파라미터 수 반환
        
        Returns:
            n_exo + n_cov
        """
        return self.n_params
    
    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        """
        파라미터 초기화
        
        Returns:
            {
                'gamma_lv': np.ndarray (n_exo,),
                'gamma_x': np.ndarray (n_cov,)
            }
        """
        params = {
            'gamma_lv': np.zeros(self.n_exo),  # 외생 LV 계수
            'gamma_x': np.zeros(self.n_cov)    # 공변량 계수
        }
        return params
    
    def validate_parameters(self, params: Dict[str, np.ndarray]) -> bool:
        """
        파라미터 유효성 검증
        
        Args:
            params: 구조모델 파라미터
        
        Returns:
            유효하면 True
        """
        # gamma_lv 검증
        if 'gamma_lv' not in params:
            logger.error("gamma_lv가 없습니다.")
            return False
        
        gamma_lv = params['gamma_lv']
        if len(gamma_lv) != self.n_exo:
            logger.error(
                f"gamma_lv 크기 불일치: expected {self.n_exo}, got {len(gamma_lv)}"
            )
            return False
        
        # gamma_x 검증
        if 'gamma_x' not in params:
            logger.error("gamma_x가 없습니다.")
            return False
        
        gamma_x = params['gamma_x']
        if len(gamma_x) != self.n_cov:
            logger.error(
                f"gamma_x 크기 불일치: expected {self.n_cov}, got {len(gamma_x)}"
            )
            return False
        
        return True
    
    def get_parameter_names(self) -> Dict[str, List[str]]:
        """
        파라미터 이름 반환
        
        Returns:
            {
                'gamma_lv': ['gamma_health_concern', 'gamma_perceived_benefit', ...],
                'gamma_x': ['gamma_age_std', 'gamma_gender', ...]
            }
        """
        names = {
            'gamma_lv': [f'gamma_{lv}' for lv in self.exogenous_lvs],
            'gamma_x': [f'gamma_{var}' for var in self.covariates]
        }
        return names

