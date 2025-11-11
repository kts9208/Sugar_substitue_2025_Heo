"""
Multi-Latent Variable Measurement Model

다중 잠재변수 측정모델 컨테이너입니다.
기존 OrderedProbitMeasurement를 재사용하여 여러 잠재변수의 측정모델을 관리합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from .measurement_equations import OrderedProbitMeasurement, ContinuousLinearMeasurement
from .multi_latent_config import MultiLatentConfig

logger = logging.getLogger(__name__)


class MultiLatentMeasurement:
    """
    다중 잠재변수 측정모델 컨테이너

    여러 측정 방법을 지원하여 5개 잠재변수의 측정모델을 관리합니다.

    지원 측정 방법:
        - ordered_probit: 순서형 프로빗 (리커트 척도를 순서형으로 처리)
        - continuous_linear: 연속형 선형 (리커트 척도를 연속형으로 처리, SEM 방식)

    측정방정식:
        LV_j =~ ζ_j1*Ind_j1 + ζ_j2*Ind_j2 + ... + ζ_jk*Ind_jk

    로그우도:
        LL = Σ_j LL_j(Indicators_j | LV_j)

    Example:
        >>> configs = {
        ...     'health_concern': MeasurementConfig(..., measurement_method='continuous_linear'),
        ...     'perceived_benefit': MeasurementConfig(..., measurement_method='ordered_probit'),
        ...     ...
        ... }
        >>> model = MultiLatentMeasurement(configs)
        >>> latent_vars = {
        ...     'health_concern': 0.5,
        ...     'perceived_benefit': 0.3,
        ...     ...
        ... }
        >>> params = {
        ...     'health_concern': {'zeta': ..., 'sigma_sq': ...},  # continuous_linear
        ...     'perceived_benefit': {'zeta': ..., 'tau': ...},    # ordered_probit
        ...     ...
        ... }
        >>> ll = model.log_likelihood(data, latent_vars, params)
    """

    def __init__(self, measurement_configs: Dict[str, 'MeasurementConfig']):
        """
        초기화

        Args:
            measurement_configs: 잠재변수별 측정모델 설정
                {
                    'health_concern': MeasurementConfig(...),
                    'perceived_benefit': MeasurementConfig(...),
                    'perceived_price': MeasurementConfig(...),
                    'nutrition_knowledge': MeasurementConfig(...),
                    'purchase_intention': MeasurementConfig(...)
                }
        """
        self.configs = measurement_configs
        self.models = {}

        # 각 잠재변수에 대해 측정모델 생성 (measurement_method에 따라 선택)
        for lv_name, config in measurement_configs.items():
            # measurement_method 확인 (기본값: continuous_linear)
            method = getattr(config, 'measurement_method', 'continuous_linear')

            if method == 'continuous_linear':
                self.models[lv_name] = ContinuousLinearMeasurement(config)
                logger.info(f"측정모델 생성 (연속형 선형): {lv_name} ({len(config.indicators)}개 지표)")
            elif method == 'ordered_probit':
                self.models[lv_name] = OrderedProbitMeasurement(config)
                logger.info(f"측정모델 생성 (순서형 프로빗): {lv_name} ({len(config.indicators)}개 지표)")
            else:
                raise ValueError(f"지원하지 않는 측정 방법: {method}")

        self.n_latent_vars = len(self.models)
        logger.info(f"MultiLatentMeasurement 초기화 완료: {self.n_latent_vars}개 잠재변수")
    
    def log_likelihood(self, data: pd.DataFrame,
                      latent_vars: Dict[str, float],
                      params: Dict[str, Dict[str, np.ndarray]]) -> float:
        """
        전체 측정모델 로그우도 계산
        
        LL = Σ_j LL_j(Indicators_j | LV_j)
        
        Args:
            data: 관측지표 데이터
            latent_vars: 잠재변수 값
                {
                    'health_concern': 0.5,
                    'perceived_benefit': 0.3,
                    'perceived_price': -0.2,
                    'nutrition_knowledge': 0.8,
                    'purchase_intention': 0.6
                }
            params: 측정모델 파라미터
                {
                    'health_concern': {'zeta': np.ndarray, 'tau': np.ndarray},
                    'perceived_benefit': {'zeta': np.ndarray, 'tau': np.ndarray},
                    ...
                }
        
        Returns:
            전체 측정모델 로그우도
        """
        total_ll = 0.0
        
        # 각 잠재변수의 측정모델 우도를 합산
        for lv_name, model in self.models.items():
            lv = latent_vars[lv_name]
            lv_params = params[lv_name]
            
            # 기존 OrderedProbitMeasurement의 log_likelihood 재사용
            ll = model.log_likelihood(data, lv, lv_params)
            total_ll += ll
        
        return total_ll
    
    def get_n_parameters(self) -> int:
        """
        총 파라미터 수 계산
        
        각 잠재변수의 파라미터 수를 합산:
        - 요인적재량 (ζ): n_indicators
        - 임계값 (τ): n_indicators × (n_categories - 1)
        
        Returns:
            총 파라미터 수
        """
        total = 0
        for model in self.models.values():
            # zeta + tau
            n_indicators = model.n_indicators
            n_thresholds = model.n_thresholds
            total += n_indicators + (n_indicators * n_thresholds)
        return total
    
    def get_parameter_info(self) -> Dict[str, Dict]:
        """
        각 잠재변수의 파라미터 정보 반환
        
        Returns:
            {
                'health_concern': {
                    'n_indicators': 6,
                    'n_zeta': 6,
                    'n_tau': 24,
                    'total': 30
                },
                ...
            }
        """
        info = {}
        for lv_name, model in self.models.items():
            n_indicators = model.n_indicators

            # SimpleMeanMeasurement인 경우 파라미터 없음
            if isinstance(model, SimpleMeanMeasurement):
                info[lv_name] = {
                    'n_indicators': n_indicators,
                    'n_zeta': 0,
                    'n_tau': 0,
                    'total': 0
                }
            else:
                # OrderedProbitMeasurement인 경우
                n_thresholds = model.n_thresholds
                n_zeta = n_indicators
                n_tau = n_indicators * n_thresholds

                info[lv_name] = {
                    'n_indicators': n_indicators,
                    'n_zeta': n_zeta,
                    'n_tau': n_tau,
                    'total': n_zeta + n_tau
                }

        return info
    
    def initialize_parameters(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        측정모델 파라미터 초기화

        Returns:
            {
                'health_concern': {
                    'zeta': np.ndarray (n_indicators,),
                    'tau': np.ndarray (n_indicators, n_thresholds)
                },
                ...
            }

            Note: SimpleMeanMeasurement는 빈 딕셔너리 반환
        """
        params = {}

        for lv_name, model in self.models.items():
            # SimpleMeanMeasurement인 경우 빈 딕셔너리
            if isinstance(model, SimpleMeanMeasurement):
                params[lv_name] = {}
                continue

            # OrderedProbitMeasurement인 경우
            n_indicators = model.n_indicators
            n_thresholds = model.n_thresholds

            # 요인적재량 초기값: 모두 1.0 (첫 번째는 고정)
            zeta = np.ones(n_indicators)

            # 임계값 초기값: 균등 간격
            tau = np.zeros((n_indicators, n_thresholds))
            for i in range(n_indicators):
                tau[i, :] = np.linspace(-2, 2, n_thresholds)

            params[lv_name] = {
                'zeta': zeta,
                'tau': tau
            }

        return params
    
    def validate_parameters(self, params: Dict[str, Dict[str, np.ndarray]]) -> bool:
        """
        파라미터 유효성 검증
        
        Args:
            params: 측정모델 파라미터
        
        Returns:
            유효하면 True
        """
        for lv_name, model in self.models.items():
            if lv_name not in params:
                logger.error(f"잠재변수 '{lv_name}'의 파라미터가 없습니다.")
                return False
            
            lv_params = params[lv_name]
            
            # zeta 검증
            if 'zeta' not in lv_params:
                logger.error(f"잠재변수 '{lv_name}'의 zeta가 없습니다.")
                return False
            
            zeta = lv_params['zeta']
            if len(zeta) != model.n_indicators:
                logger.error(
                    f"잠재변수 '{lv_name}'의 zeta 크기 불일치: "
                    f"expected {model.n_indicators}, got {len(zeta)}"
                )
                return False
            
            # tau 검증
            if 'tau' not in lv_params:
                logger.error(f"잠재변수 '{lv_name}'의 tau가 없습니다.")
                return False
            
            tau = lv_params['tau']
            expected_shape = (model.n_indicators, model.n_thresholds)
            if tau.shape != expected_shape:
                logger.error(
                    f"잠재변수 '{lv_name}'의 tau 크기 불일치: "
                    f"expected {expected_shape}, got {tau.shape}"
                )
                return False
            
            # tau 순서 검증 (각 지표의 임계값이 증가하는지)
            for i in range(model.n_indicators):
                if not np.all(np.diff(tau[i, :]) > 0):
                    logger.warning(
                        f"잠재변수 '{lv_name}' 지표 {i}의 임계값이 증가하지 않습니다: {tau[i, :]}"
                    )
        
        return True
    
    def get_latent_variable_names(self) -> List[str]:
        """잠재변수 이름 목록 반환"""
        return list(self.models.keys())

