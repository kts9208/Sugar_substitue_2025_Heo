"""
Structural Equations for ICLV Models

ICLV 구조모델: 사회인구학적 변수 → 잠재변수

Based on King (2022) Apollo R code implementation.

Author: Sugar Substitute Research Team
Date: 2025-11-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
import logging
from dataclasses import dataclass

# StructuralConfig 정의 (import 오류 방지)
try:
    from .iclv_config import StructuralConfig
except ImportError:
    @dataclass
    class StructuralConfig:
        """구조모델 설정"""
        sociodemographics: List[str]
        include_in_choice: bool = True
        initial_gammas: Optional[Dict[str, float]] = None
        error_variance: float = 1.0
        fix_error_variance: bool = True

logger = logging.getLogger(__name__)


class LatentVariableRegression:
    """
    ICLV 구조모델 (Structural Equations)
    
    Model:
        LV = γ*X + η
        η ~ N(0, σ²)
    
    여기서:
        - LV: 잠재변수 (Latent Variable)
        - X: 사회인구학적 변수 (Sociodemographics)
        - γ: 회귀계수 (Regression coefficients)
        - η: 오차항 (Error term)
        - σ²: 오차 분산 (Error variance)
    
    King (2022) Apollo R 코드 기반:
        apollo_randCoeff = function(apollo_beta, apollo_inputs) {
            randcoeff = list()
            randcoeff[["LV"]] = gamma_age * age + 
                                gamma_gender * gender + 
                                gamma_income * income + 
                                eta
            return(randcoeff)
        }
    
    Usage:
        >>> config = StructuralConfig(
        ...     sociodemographics=['age', 'gender', 'income']
        ... )
        >>> model = LatentVariableRegression(config)
        >>> 
        >>> # Simultaneous 추정용
        >>> lv = model.predict(data, params, draw)
        >>> ll = model.log_likelihood(data, lv, params, draw)
        >>> 
        >>> # Sequential 추정용
        >>> params = model.fit(data, latent_var)
    """
    
    def __init__(self, config: StructuralConfig):
        """
        초기화
        
        Args:
            config: 구조모델 설정
        """
        self.config = config
        self.sociodemographics = config.sociodemographics
        self.error_variance = config.error_variance
        self.fix_error_variance = config.fix_error_variance
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"LatentVariableRegression 초기화")
        self.logger.info(f"  사회인구학적 변수: {self.sociodemographics}")
        self.logger.info(f"  오차 분산: {self.error_variance} (고정: {self.fix_error_variance})")
    
    def predict(self, data: pd.DataFrame, params: Dict,
                draw: float) -> float:
        """
        잠재변수 예측 (시뮬레이션 기반)

        🔴 수정: 개인당 1개의 LV 값 반환 (스칼라)

        LV = γ*X + σ*draw

        King (2022) Apollo R 코드:
            LV = gamma_age * age + gamma_gender * gender + ... + eta

        Args:
            data: 사회인구학적 변수 데이터 (개인의 여러 선택 상황)
            params: {'gamma': np.ndarray}  # 회귀계수 (n_vars,)
            draw: 표준정규분포 draw (Halton sequence) - 스칼라

        Returns:
            잠재변수 값 (스칼라 - 개인당 1개)

        Example:
            >>> params = {'gamma': np.array([0.5, -0.3, 0.2])}
            >>> draw = 0.5  # 표준정규분포 draw
            >>> lv = model.predict(data, params, draw)
        """
        gamma = params['gamma']

        # 🔴 수정: 첫 번째 행만 사용 (사회인구학적 변수는 개인 특성)
        first_row = data.iloc[0]

        # 선형 예측 (평균)
        lv_mean = 0.0
        for i, var in enumerate(self.sociodemographics):
            if var in first_row.index:
                value = first_row[var]
                # 🔴 수정: NaN 처리 (0으로 대체)
                if pd.isna(value):
                    value = 0.0
                lv_mean += gamma[i] * value

        # 오차항 추가 (시뮬레이션)
        lv = lv_mean + np.sqrt(self.error_variance) * draw

        return lv
    
    def log_likelihood(self, data: pd.DataFrame, lv: float,
                      params: Dict, draw: float) -> float:
        """
        구조모델 로그우도

        🔴 수정: 개인당 1개의 LV에 대한 로그우도

        P(LV|X) ~ N(γ*X, σ²)

        정규분포 확률밀도함수:
            f(LV|X) = (1/√(2πσ²)) * exp(-(LV - γ*X)²/(2σ²))

        로그우도:
            log L = -0.5 * log(2πσ²) - 0.5 * (LV - γ*X)²/σ²

        Args:
            data: 사회인구학적 변수 데이터 (개인의 여러 선택 상황)
            lv: 잠재변수 값 (스칼라 - 개인당 1개)
            params: {'gamma': np.ndarray}  # 회귀계수
            draw: 표준정규분포 draw (사용하지 않음, 인터페이스 일관성용)

        Returns:
            로그우도 값 (스칼라)

        Example:
            >>> ll = model.log_likelihood(data, lv, params, draw)
        """
        gamma = params['gamma']

        # 🔴 수정: 첫 번째 행만 사용
        first_row = data.iloc[0]

        # 평균
        lv_mean = 0.0
        for i, var in enumerate(self.sociodemographics):
            if var in first_row.index:
                value = first_row[var]
                # 🔴 수정: NaN 처리 (0으로 대체)
                if pd.isna(value):
                    value = 0.0
                lv_mean += gamma[i] * value

        # 로그우도 (정규분포)
        # log f(LV|X) = -0.5 * log(2πσ²) - 0.5 * (LV - μ)²/σ²
        ll = -0.5 * np.log(2 * np.pi * self.error_variance)
        ll -= 0.5 * ((lv - lv_mean) ** 2) / self.error_variance

        return ll
    
    def fit(self, data: pd.DataFrame, latent_var: np.ndarray) -> Dict:
        """
        구조모델 단독 추정 (Sequential 방식용)
        
        OLS 회귀분석:
            LV = γ*X + ε
            γ = (X'X)⁻¹X'LV
        
        Args:
            data: 사회인구학적 변수 데이터 (n_obs, n_vars)
            latent_var: 잠재변수 값 (n_obs,)
                       측정모델에서 추정된 요인점수
        
        Returns:
            {
                'gamma': np.ndarray,  # 회귀계수 (n_vars,)
                'sigma': float,       # 잔차 표준편차
                'r_squared': float,   # 결정계수
                'fitted_values': np.ndarray,  # 적합값
                'residuals': np.ndarray       # 잔차
            }
        
        Example:
            >>> # 측정모델에서 요인점수 추출
            >>> factor_scores = measurement_model.predict_factors(data)
            >>> 
            >>> # 구조모델 추정
            >>> results = structural_model.fit(data, factor_scores)
            >>> print(f"R²: {results['r_squared']:.3f}")
        """
        self.logger.info("구조모델 Sequential 추정 시작 (OLS)")
        
        # 사회인구학적 변수 추출
        X = data[self.sociodemographics].values
        y = latent_var
        
        # OLS 추정
        # γ = (X'X)⁻¹X'y
        gamma, residuals_sum, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        # 적합값
        fitted_values = X @ gamma
        
        # 잔차
        residuals = y - fitted_values
        
        # 잔차 분산
        sigma = np.std(residuals, ddof=len(gamma))
        
        # 결정계수
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum(residuals ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        self.logger.info(f"  회귀계수: {gamma}")
        self.logger.info(f"  잔차 표준편차: {sigma:.4f}")
        self.logger.info(f"  R²: {r_squared:.4f}")
        
        return {
            'gamma': gamma,
            'sigma': sigma,
            'r_squared': r_squared,
            'fitted_values': fitted_values,
            'residuals': residuals
        }
    
    def get_initial_params(self, data: pd.DataFrame, 
                          latent_var: Optional[np.ndarray] = None) -> Dict:
        """
        초기 파라미터 생성
        
        Args:
            data: 사회인구학적 변수 데이터
            latent_var: 잠재변수 값 (있으면 OLS로 추정, 없으면 0으로 초기화)
        
        Returns:
            {'gamma': np.ndarray}
        
        Example:
            >>> # 잠재변수 없이 초기화
            >>> params = model.get_initial_params(data)
            >>> 
            >>> # 잠재변수로 OLS 추정
            >>> params = model.get_initial_params(data, factor_scores)
        """
        n_vars = len(self.sociodemographics)
        
        if latent_var is not None:
            # OLS로 추정
            results = self.fit(data, latent_var)
            gamma = results['gamma']
            self.logger.info(f"초기 파라미터 (OLS): {gamma}")
        else:
            # 0으로 초기화
            gamma = np.zeros(n_vars)
            self.logger.info(f"초기 파라미터 (0): {gamma}")
        
        return {'gamma': gamma}
    
    def get_initial_params_from_semopy(self, data: pd.DataFrame,
                                      latent_var: np.ndarray) -> Dict:
        """
        semopy로 초기 파라미터 생성
        
        기존 semopy 경로분석 결과를 활용하여 좋은 초기값을 생성합니다.
        
        Args:
            data: 사회인구학적 변수 데이터
            latent_var: 잠재변수 값
        
        Returns:
            {'gamma': np.ndarray}
        
        Example:
            >>> params = model.get_initial_params_from_semopy(data, factor_scores)
        """
        try:
            from semopy import Model
            
            # 모델 스펙 생성
            sociodem_vars = " + ".join(self.sociodemographics)
            model_spec = f"LV ~ {sociodem_vars}"
            
            # 데이터 준비
            data_with_lv = data[self.sociodemographics].copy()
            data_with_lv['LV'] = latent_var
            
            # semopy 적합
            model = Model(model_spec)
            model.fit(data_with_lv)
            
            # 파라미터 추출
            params_df = model.inspect()
            gamma = params_df[params_df['op'] == '~']['Estimate'].values
            
            self.logger.info(f"초기 파라미터 (semopy): {gamma}")
            
            return {'gamma': gamma}
        
        except ImportError:
            self.logger.warning("semopy를 사용할 수 없습니다. OLS로 대체합니다.")
            return self.get_initial_params(data, latent_var)
        except Exception as e:
            self.logger.warning(f"semopy 추정 실패: {e}. OLS로 대체합니다.")
            return self.get_initial_params(data, latent_var)


def estimate_structural_model(data: pd.DataFrame, latent_var: np.ndarray,
                              sociodemographics: List[str],
                              **kwargs) -> Dict:
    """
    구조모델 추정 헬퍼 함수
    
    Args:
        data: 사회인구학적 변수 데이터
        latent_var: 잠재변수 값
        sociodemographics: 사회인구학적 변수 리스트
        **kwargs: 추가 설정
    
    Returns:
        추정 결과
    
    Example:
        >>> results = estimate_structural_model(
        ...     data, 
        ...     factor_scores,
        ...     sociodemographics=['age', 'gender', 'income']
        ... )
    """
    config = StructuralConfig(
        sociodemographics=sociodemographics,
        **kwargs
    )
    
    model = LatentVariableRegression(config)
    results = model.fit(data, latent_var)
    
    return results

