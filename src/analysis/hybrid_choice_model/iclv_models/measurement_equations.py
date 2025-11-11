"""
Ordered Probit Measurement Model for ICLV

이 모듈은 King (2022)의 Apollo R 코드를 기반으로 
Ordered Probit 측정모델을 Python으로 구현합니다.

Reference:
- King, A. M. (2022). Microplastics in seafood: Consumer risk perceptions and 
  willingness to pay. Food Quality and Preference, 102, 104650.
- Apollo R package: apollo_op() function for ordered probit
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class OrderedProbitMeasurement:
    """
    Ordered Probit 측정모델
    
    리커트 척도 데이터를 올바르게 모델링하는 측정방정식입니다.
    
    Model:
        P(Y_i = k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
    
    여기서:
        - Y_i: 관측지표 (1, 2, 3, 4, 5 for 5-point Likert scale)
        - τ: 임계값 (thresholds) - 범주 경계
        - ζ: 요인적재량 (factor loadings)
        - LV: 잠재변수 (latent variable)
        - Φ: 표준정규 누적분포함수
    
    Apollo R 코드 대응:
        op_settings = list(
            outcomeOrdered = Q13,
            V = zeta_Q13 * LV,
            tau = c(tau_Q13_1, tau_Q13_2, tau_Q13_3, tau_Q13_4),
            componentName = "indic_Q13"
        )
        P[["indic_Q13"]] = apollo_op(op_settings, functionality)
    """
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config: MeasurementConfig 객체
                - indicators: 관측지표 리스트
                - n_categories: 범주 수 (예: 5점 척도 = 5)
                - indicator_types: 지표 유형 ('ordered', 'continuous', 'binary')
        """
        self.config = config
        self.n_indicators = len(config.indicators)
        self.n_categories = config.n_categories
        self.n_thresholds = config.n_categories - 1  # 5점 척도 → 4개 임계값
        
        # 파라미터 (추정 후 저장)
        self.zeta = None  # 요인적재량 (n_indicators,)
        self.tau = None   # 임계값 (n_indicators, n_thresholds)
        
        self.fitted = False
        
        logger.info(f"OrderedProbitMeasurement 초기화: {self.n_indicators}개 지표, {self.n_categories}점 척도")
    
    def log_likelihood(self, data: pd.DataFrame, latent_var: float,
                      params: Dict[str, np.ndarray]) -> float:
        """
        로그우도 계산 (King 2022 Apollo 코드 기반)

        🔴 수정: 측정모델은 개인당 1번만 계산 (지표는 개인 특성)

        Args:
            data: 관측지표 데이터 (개인의 여러 선택 상황)
            latent_var: 잠재변수 값 (스칼라 - 개인당 1개)
            params: 파라미터 딕셔너리
                - 'zeta': 요인적재량 (n_indicators,)
                - 'tau': 임계값 (n_indicators, n_thresholds)

        Returns:
            로그우도 값
        """
        zeta = params['zeta']
        tau = params['tau']

        total_ll = 0.0

        # 🔴 수정: 첫 번째 행만 사용 (지표는 개인 특성이므로 모든 행이 동일)
        first_row = data.iloc[0]

        # 각 지표에 대해
        for i, indicator in enumerate(self.config.indicators):
            if indicator not in first_row.index:
                continue

            y = first_row[indicator]
            if np.isnan(y):
                continue

            zeta_i = zeta[i]
            tau_i = tau[i]  # (n_thresholds,)

            # Ordered Probit 확률 계산
            prob = self._ordered_probit_probability(y, latent_var, zeta_i, tau_i)

            # 로그우도 누적
            if prob > 0:
                total_ll += np.log(prob)
            else:
                total_ll += -1e10  # 매우 작은 값

        return total_ll
    
    def _ordered_probit_probability(self, y: float, lv: float, 
                                   zeta: float, tau: np.ndarray) -> float:
        """
        Ordered Probit 확률 계산
        
        P(Y=k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
        
        Apollo R 코드:
            V = zeta * LV
            P(Y=k) = pnorm(tau[k] - V) - pnorm(tau[k-1] - V)
        
        Args:
            y: 관측값 (1, 2, 3, 4, 5)
            lv: 잠재변수 값
            zeta: 요인적재량
            tau: 임계값 배열 (n_thresholds,)
        
        Returns:
            확률 P(Y=k)
        """
        k = int(y) - 1  # 1-5 → 0-4
        
        # V = zeta * LV (Apollo 코드와 동일)
        V = zeta * lv
        
        # 경계 조건
        if k == 0:
            # P(Y=1) = Φ(τ_1 - V)
            prob = norm.cdf(tau[0] - V)
        elif k == self.n_categories - 1:
            # P(Y=5) = 1 - Φ(τ_4 - V)
            prob = 1 - norm.cdf(tau[-1] - V)
        else:
            # P(Y=k) = Φ(τ_k - V) - Φ(τ_{k-1} - V)
            prob = norm.cdf(tau[k] - V) - norm.cdf(tau[k-1] - V)
        
        # 수치 안정성을 위해 클리핑
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        
        return prob
    
    def predict(self, latent_var: np.ndarray, params: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        잠재변수로부터 관측지표 예측
        
        Args:
            latent_var: 잠재변수 값 (n_obs,)
            params: 파라미터 딕셔너리
        
        Returns:
            예측된 지표 값 (n_obs, n_indicators)
        """
        zeta = params['zeta']
        tau = params['tau']
        
        n_obs = len(latent_var)
        predictions = np.zeros((n_obs, self.n_indicators))
        
        for i in range(self.n_indicators):
            zeta_i = zeta[i]
            tau_i = tau[i]
            
            for j in range(n_obs):
                lv = latent_var[j]
                
                # 각 범주의 확률 계산
                probs = []
                for k in range(1, self.n_categories + 1):
                    prob = self._ordered_probit_probability(k, lv, zeta_i, tau_i)
                    probs.append(prob)
                
                # 가장 높은 확률의 범주 선택
                predictions[j, i] = np.argmax(probs) + 1
        
        return pd.DataFrame(predictions, columns=self.config.indicators)
    
    def predict_probabilities(self, latent_var: np.ndarray, 
                             params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        각 범주의 확률 예측
        
        Args:
            latent_var: 잠재변수 값 (n_obs,)
            params: 파라미터 딕셔너리
        
        Returns:
            지표별 범주 확률 딕셔너리
        """
        zeta = params['zeta']
        tau = params['tau']
        
        n_obs = len(latent_var)
        probabilities = {}
        
        for i, indicator in enumerate(self.config.indicators):
            zeta_i = zeta[i]
            tau_i = tau[i]
            
            # (n_obs, n_categories) 확률 행렬
            probs = np.zeros((n_obs, self.n_categories))
            
            for j in range(n_obs):
                lv = latent_var[j]
                
                for k in range(1, self.n_categories + 1):
                    probs[j, k-1] = self._ordered_probit_probability(k, lv, zeta_i, tau_i)
            
            probabilities[indicator] = probs
        
        return probabilities
    
    def fit(self, data: pd.DataFrame, initial_params: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        측정모델 단독 추정 (Sequential 방식용)
        
        Note: ICLV 동시 추정에서는 이 메서드를 사용하지 않고,
              SimultaneousEstimator가 log_likelihood()를 직접 호출합니다.
        
        Args:
            data: 관측지표 데이터
            initial_params: 초기 파라미터 (선택)
        
        Returns:
            추정 결과 딕셔너리
        """
        logger.info("측정모델 단독 추정 시작 (Sequential 방식)")
        
        # 초기 파라미터 설정
        if initial_params is None:
            initial_params = self._get_initial_parameters()
        
        # 간단한 요인점수 계산 (평균)
        latent_var = data[self.config.indicators].mean(axis=1).values
        
        # 파라미터 벡터화
        param_vector = self._pack_parameters(initial_params)
        
        # 우도함수 정의
        def negative_log_likelihood(params_vec):
            params = self._unpack_parameters(params_vec)
            ll = self.log_likelihood(data, latent_var, params)
            return -ll
        
        # 최적화
        result = minimize(
            negative_log_likelihood,
            param_vector,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        # 결과 저장
        self.zeta = self._unpack_parameters(result.x)['zeta']
        self.tau = self._unpack_parameters(result.x)['tau']
        self.fitted = True
        
        logger.info("측정모델 추정 완료")
        
        return {
            'zeta': self.zeta,
            'tau': self.tau,
            'log_likelihood': -result.fun,
            'success': result.success
        }
    
    def _get_initial_parameters(self) -> Dict[str, np.ndarray]:
        """초기 파라미터 설정 (King 2022 스타일)"""
        return {
            'zeta': np.ones(self.n_indicators),  # 요인적재량 = 1.0
            'tau': np.tile([-2, -1, 1, 2], (self.n_indicators, 1))  # 5점 척도 기본값
        }
    
    def _pack_parameters(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """파라미터를 벡터로 변환"""
        param_vector = []
        param_vector.extend(params['zeta'])
        param_vector.extend(params['tau'].flatten())
        return np.array(param_vector)
    
    def _unpack_parameters(self, param_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """벡터를 파라미터로 변환"""
        idx = 0

        # zeta
        zeta = param_vector[idx:idx + self.n_indicators]
        idx += self.n_indicators

        # tau
        tau = param_vector[idx:].reshape(self.n_indicators, self.n_thresholds)

        return {'zeta': zeta, 'tau': tau}

    def get_n_parameters(self) -> int:
        """
        총 파라미터 수 반환

        Returns:
            파라미터 수 (zeta + tau)
        """
        # zeta: n_indicators
        # tau: n_indicators * n_thresholds
        return self.n_indicators + (self.n_indicators * self.n_thresholds)


class ContinuousLinearMeasurement:
    """
    연속형 선형 측정모델 (Continuous Linear Measurement)

    리커트 척도를 연속형 변수로 간주하여 측정하는 SEM 방식의 측정모델입니다.

    Model:
        Y_i = ζ_i * LV + ε_i
        ε_i ~ N(0, σ²_i)

    여기서:
        - Y_i: 관측지표 (리커트 척도 값)
        - ζ_i: 요인적재량 (factor loading)
        - LV: 잠재변수 (latent variable)
        - ε_i: 측정오차
        - σ²_i: 오차분산

    Parameters:
        - zeta: 요인적재량 (n_indicators,)
        - sigma_sq: 오차분산 (n_indicators,)

    Total parameters: 2 * n_indicators (첫 번째 적재량 고정 시 2*n - 1)

    Reference:
        - Bollen, K. A. (1989). Structural Equations with Latent Variables.
        - Jöreskog, K. G. (1970). A general method for analysis of covariance structures.
    """

    def __init__(self, config):
        """
        초기화

        Args:
            config: MeasurementConfig 객체
                - indicators: 관측지표 리스트
                - fix_first_loading: 첫 번째 적재량 고정 여부
                - fix_error_variance: 오차분산 고정 여부
        """
        self.config = config
        self.n_indicators = len(config.indicators)

        # 파라미터 (추정 후 저장)
        self.zeta = None      # 요인적재량 (n_indicators,)
        self.sigma_sq = None  # 오차분산 (n_indicators,)

        self.fitted = False

        logger.info(f"ContinuousLinearMeasurement 초기화: {self.n_indicators}개 지표")

    def log_likelihood(self, data: pd.DataFrame, latent_var: float,
                      params: Dict[str, np.ndarray]) -> float:
        """
        로그우도 계산 (정규분포 기반)

        Args:
            data: 관측지표 데이터
            latent_var: 잠재변수 값 (스칼라)
            params: 파라미터 딕셔너리
                - 'zeta': 요인적재량 (n_indicators,)
                - 'sigma_sq': 오차분산 (n_indicators,)

        Returns:
            로그우도 값
        """
        zeta = params['zeta']
        sigma_sq = params['sigma_sq']

        total_ll = 0.0

        # 첫 번째 행만 사용 (지표는 개인 특성)
        first_row = data.iloc[0]

        # 각 지표에 대해
        for i, indicator in enumerate(self.config.indicators):
            if indicator not in first_row.index:
                continue

            y_obs = first_row[indicator]

            if pd.isna(y_obs):
                continue

            # 예측값: Y_pred = ζ_i * LV
            y_pred = zeta[i] * latent_var

            # 잔차
            residual = y_obs - y_pred

            # 정규분포 로그우도
            # log N(y | μ, σ²) = -0.5 * log(2π * σ²) - 0.5 * (y - μ)² / σ²
            ll_i = -0.5 * np.log(2 * np.pi * sigma_sq[i])
            ll_i += -0.5 * (residual ** 2) / sigma_sq[i]

            total_ll += ll_i

        return total_ll

    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        """
        파라미터 초기화

        Returns:
            {
                'zeta': np.ndarray (n_indicators,),
                'sigma_sq': np.ndarray (n_indicators,)
            }
        """
        params = {}

        # 요인적재량 초기화
        if self.config.initial_loadings is not None:
            zeta = np.array([
                self.config.initial_loadings.get(ind, 1.0)
                for ind in self.config.indicators
            ])
        else:
            # 기본값: 모두 1.0
            zeta = np.ones(self.n_indicators)

        # 첫 번째 적재량 고정 (식별)
        if self.config.fix_first_loading:
            zeta[0] = 1.0

        params['zeta'] = zeta

        # 오차분산 초기화
        sigma_sq = np.ones(self.n_indicators) * self.config.initial_error_variance
        params['sigma_sq'] = sigma_sq

        return params

    def get_n_parameters(self) -> int:
        """
        추정할 파라미터 수 반환

        Returns:
            파라미터 수
        """
        n_params = 0

        # 요인적재량
        if self.config.fix_first_loading:
            n_params += self.n_indicators - 1  # 첫 번째 고정
        else:
            n_params += self.n_indicators

        # 오차분산
        if not self.config.fix_error_variance:
            n_params += self.n_indicators

        return n_params

    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """
        파라미터 제약조건 (bounds)

        Returns:
            [(lower, upper), ...] 리스트
        """
        bounds = []

        # 요인적재량: [-10, 10]
        start_idx = 1 if self.config.fix_first_loading else 0
        for i in range(start_idx, self.n_indicators):
            bounds.append((-10.0, 10.0))

        # 오차분산: [0.01, 100] (양수)
        if not self.config.fix_error_variance:
            for i in range(self.n_indicators):
                bounds.append((0.01, 100.0))

        return bounds


def estimate_measurement_model(data: pd.DataFrame, config,
                               initial_params: Optional[Dict[str, np.ndarray]] = None) -> Tuple[OrderedProbitMeasurement, Dict[str, Any]]:
    """
    측정모델 추정 헬퍼 함수

    Args:
        data: 관측지표 데이터
        config: MeasurementConfig 객체
        initial_params: 초기 파라미터 (선택)

    Returns:
        (모델 객체, 추정 결과)
    """
    model = OrderedProbitMeasurement(config)
    results = model.fit(data, initial_params)
    return model, results

