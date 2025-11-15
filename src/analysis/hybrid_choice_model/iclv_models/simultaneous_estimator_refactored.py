"""
Simultaneous Estimation for ICLV Models (Refactored)

ICLV 모델의 동시 추정 엔진입니다 (리팩토링 버전).
BaseEstimator를 상속하여 코드 중복을 제거하고 단일책임 원칙을 준수합니다.

단일책임 원칙:
- 동시추정 전용 로직만 포함
- Halton draws 생성은 draw_generator 모듈 사용
- 초기값 설정은 initializer 모듈 사용
- 우도 계산은 likelihood_calculator 모듈 사용

참조:
- King (2022) - Apollo 패키지 사용
- Train (2009) - Discrete Choice Methods with Simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import optimize
import logging

from .base_estimator import BaseEstimator
from .draw_generator import HaltonDrawGenerator, RandomDrawGenerator
from .initializer import SimultaneousInitializer
from .likelihood_calculator import SimultaneousLikelihoodCalculator
from .gradient_calculator import (
    MeasurementGradient,
    StructuralGradient,
    ChoiceGradient,
    JointGradient
)
from .parameter_scaler import ParameterScaler
from .bhhh_calculator import BHHHCalculator

logger = logging.getLogger(__name__)


class SimultaneousEstimator(BaseEstimator):
    """
    ICLV 모델 동시 추정기 (리팩토링 버전)
    
    측정모델, 구조모델, 선택모델을 동시에 추정합니다.
    
    결합 우도함수:
    L = ∏ᵢ ∫ P(Choice|LV) × P(Indicators|LV) × P(LV|X) dLV
    
    시뮬레이션 기반 추정:
    L ≈ ∏ᵢ (1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)
    """
    
    def __init__(self, config):
        """
        Args:
            config: ICLVConfig 또는 MultiLatentConfig
        """
        super().__init__(config)
        
        # 동시추정 전용 컴포넌트
        self.initializer = SimultaneousInitializer(config)
        self.likelihood_calculator = None  # estimate()에서 초기화
        self.draw_generator = None  # estimate()에서 초기화
        
        # Gradient calculators (Apollo 방식)
        self.measurement_grad = None
        self.structural_grad = None
        self.choice_grad = None
        self.joint_grad = None
        self.use_analytic_gradient = False
        
        # Parameter scaler
        self.param_scaler = None
        
        # BHHH calculator
        self.bhhh_calculator = None
    
    def estimate(self, data: pd.DataFrame,
                measurement_model,
                structural_model,
                choice_model,
                log_file: Optional[str] = None,
                initial_params: Optional[np.ndarray] = None,
                **kwargs) -> Dict:
        """
        ICLV 모델 동시 추정
        
        Args:
            data: 통합 데이터
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            log_file: 로그 파일 경로
            initial_params: 사용자 정의 초기값
            **kwargs: 추가 인자
        
        Returns:
            추정 결과 딕셔너리
        """
        self.logger.info("="*70)
        self.logger.info("ICLV 모델 동시 추정 시작 (리팩토링 버전)")
        self.logger.info("="*70)
        
        # 데이터 검증
        self._validate_data(data)
        self.data = data
        
        # 로그 설정
        if log_file is None:
            from pathlib import Path
            from datetime import datetime
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = results_dir / f'iclv_estimation_log_{timestamp}.txt'
        
        self._setup_iteration_logger(str(log_file))
        
        try:
            # 1. Halton draws 생성
            n_individuals = data[self.config.individual_id_column].nunique()
            self._setup_draw_generator(n_individuals, structural_model)
            
            # 2. Likelihood calculator 초기화
            self.likelihood_calculator = SimultaneousLikelihoodCalculator(
                self.config,
                self.config.individual_id_column
            )
            
            # 3. Gradient calculators 초기화 (필요시)
            self._setup_gradient_calculators(measurement_model, structural_model, choice_model)
            
            # 4. 초기 파라미터 설정
            if initial_params is None:
                initial_params = self._initialize_parameters(
                    measurement_model, structural_model, choice_model
                )
            
            self.iteration_logger.info(f"초기 파라미터 설정 완료: {len(initial_params)} 파라미터")
            
            # 5. 파라미터 이름 생성
            self.param_names = self.initializer.get_parameter_names(
                measurement_model, structural_model, choice_model
            )
            
            # 6. 파라미터 스케일링 설정
            self._setup_parameter_scaler(initial_params)
            
            # 7. 최적화 실행
            result = self._run_optimization(
                initial_params,
                measurement_model,
                structural_model,
                choice_model
            )

            self.logger.info("\n" + "="*70)
            self.logger.info("동시 추정 완료!")
            self.logger.info("="*70)

            return result

        finally:
            self._close_iteration_logger()

    # ========================================================================
    # 초기화 메서드
    # ========================================================================

    def _initialize_parameters(self, measurement_model, structural_model,
                              choice_model) -> np.ndarray:
        """
        초기 파라미터 설정 (동시추정용)

        Returns:
            초기 파라미터 벡터
        """
        return self.initializer.initialize(
            measurement_model, structural_model, choice_model
        )

    def _setup_draw_generator(self, n_individuals: int, structural_model):
        """
        Halton draws 생성기 설정

        Args:
            n_individuals: 개인 수
            structural_model: 구조모델 (잠재변수 차원 확인용)
        """
        # 잠재변수 차원 확인
        n_dimensions = 1
        if hasattr(structural_model, 'n_latent_vars'):
            n_dimensions = structural_model.n_latent_vars
        elif hasattr(structural_model, 'models'):
            n_dimensions = len(structural_model.models)

        self.iteration_logger.info(
            f"Halton draws 생성 시작: "
            f"n_draws={self.config.estimation.n_draws}, "
            f"n_individuals={n_individuals}, "
            f"n_dimensions={n_dimensions}"
        )

        # Draw generator 선택
        draw_type = getattr(self.config.estimation, 'draw_type', 'halton')

        if draw_type == 'halton':
            self.draw_generator = HaltonDrawGenerator(
                n_draws=self.config.estimation.n_draws,
                n_individuals=n_individuals,
                n_dimensions=n_dimensions,
                scramble=getattr(self.config.estimation, 'scramble_halton', True),
                seed=getattr(self.config.estimation, 'seed', None)
            )
        else:
            self.draw_generator = RandomDrawGenerator(
                n_draws=self.config.estimation.n_draws,
                n_individuals=n_individuals,
                n_dimensions=n_dimensions,
                seed=getattr(self.config.estimation, 'seed', None)
            )

        self.iteration_logger.info("Draws 생성 완료")

    def _setup_gradient_calculators(self, measurement_model, structural_model,
                                   choice_model):
        """
        Gradient calculators 초기화 (Apollo 방식)

        Args:
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
        """
        # Gradient 사용 여부 확인
        use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B', 'BHHH']

        if not use_gradient:
            self.iteration_logger.info("Gradient 미사용 (optimizer: {})".format(
                self.config.estimation.optimizer
            ))
            return

        # Analytic gradient 사용 여부
        self.use_analytic_gradient = getattr(
            self.config.estimation, 'use_analytic_gradient', False
        )

        if not self.use_analytic_gradient:
            self.iteration_logger.info("Numeric gradient 사용")
            return

        self.iteration_logger.info("Analytic gradient calculators 초기화...")

        # 다중 잠재변수 지원 확인
        from .multi_latent_config import MultiLatentConfig
        is_multi_latent = isinstance(self.config, MultiLatentConfig)

        if is_multi_latent:
            # 다중 잠재변수
            from .multi_latent_gradient import (
                MultiLatentMeasurementGradient,
                MultiLatentStructuralGradient,
                MultiLatentJointGradient
            )

            self.measurement_grad = MultiLatentMeasurementGradient(
                self.config.measurement_configs
            )
            self.structural_grad = MultiLatentStructuralGradient(
                n_exo=self.config.structural.n_exo,
                n_cov=self.config.structural.n_cov,
                error_variance=self.config.structural.error_variance
            )
            self.choice_grad = ChoiceGradient(
                n_attributes=len(self.config.choice.choice_attributes)
            )

            self.joint_grad = MultiLatentJointGradient(
                self.measurement_grad,
                self.structural_grad,
                self.choice_grad
            )
            self.joint_grad.iteration_logger = self.iteration_logger
            self.joint_grad.config = self.config
        else:
            # 단일 잠재변수
            self.measurement_grad = MeasurementGradient(
                n_indicators=len(self.config.measurement.indicators),
                n_categories=self.config.measurement.n_categories
            )
            self.structural_grad = StructuralGradient(
                n_sociodem=len(self.config.structural.sociodemographics),
                error_variance=1.0
            )
            self.choice_grad = ChoiceGradient(
                n_attributes=len(self.config.choice.choice_attributes)
            )

            self.joint_grad = JointGradient(
                self.measurement_grad,
                self.structural_grad,
                self.choice_grad
            )

        self.iteration_logger.info("Gradient calculators 초기화 완료")

    def _setup_parameter_scaler(self, initial_params: np.ndarray):
        """
        파라미터 스케일링 설정

        Args:
            initial_params: 초기 파라미터
        """
        use_parameter_scaling = getattr(
            self.config.estimation, 'use_parameter_scaling', True
        )

        if use_parameter_scaling:
            # Custom scales 생성
            custom_scales = self._get_custom_scales(self.param_names)

            self.iteration_logger.info("파라미터 스케일링 활성화")
            self.param_scaler = ParameterScaler(
                initial_params=initial_params,
                param_names=self.param_names,
                custom_scales=custom_scales,
                logger=self.iteration_logger
            )
        else:
            # 항등 스케일러
            self.iteration_logger.info("파라미터 스케일링 비활성화")
            self.param_scaler = ParameterScaler(
                initial_params=initial_params,
                param_names=self.param_names,
                custom_scales={name: 1.0 for name in self.param_names},
                logger=self.iteration_logger
            )

    def _get_custom_scales(self, param_names: list) -> Dict[str, float]:
        """
        파라미터별 커스텀 스케일 생성

        Args:
            param_names: 파라미터 이름 리스트

        Returns:
            파라미터별 스케일 딕셔너리
        """
        custom_scales = {}

        for name in param_names:
            if 'zeta' in name:
                custom_scales[name] = 1.0
            elif 'sigma_sq' in name or 'tau' in name:
                custom_scales[name] = 0.1
            elif 'gamma' in name:
                custom_scales[name] = 1.0
            elif 'beta' in name:
                custom_scales[name] = 1.0
            elif 'lambda' in name:
                custom_scales[name] = 1.0
            else:
                custom_scales[name] = 1.0

        return custom_scales

    # ========================================================================
    # 최적화 메서드
    # ========================================================================

    def _run_optimization(self, initial_params: np.ndarray,
                         measurement_model,
                         structural_model,
                         choice_model) -> Dict:
        """
        최적화 실행

        Args:
            initial_params: 초기 파라미터
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            추정 결과 딕셔너리
        """
        # 파라미터 스케일링
        initial_params_scaled = self.param_scaler.scale_parameters(initial_params)

        # 우도함수 정의
        def negative_ll(params_scaled):
            # 스케일 복원
            params = self.param_scaler.unscale_parameters(params_scaled)

            # 파라미터 언팩
            param_dict = self._unpack_parameters(
                params, measurement_model, structural_model, choice_model
            )

            # Draws 가져오기
            draws = self.draw_generator.get_draws()

            # 로그우도 계산
            ll = self.likelihood_calculator.compute_log_likelihood(
                self.data,
                params,
                measurement_model,
                structural_model,
                choice_model,
                draws=draws,
                param_dict=param_dict
            )

            return -ll

        # Gradient 함수 정의 (필요시)
        gradient_func = None
        if self.use_analytic_gradient and self.joint_grad is not None:
            def gradient_wrapper(params_scaled):
                params = self.param_scaler.unscale_parameters(params_scaled)
                param_dict = self._unpack_parameters(
                    params, measurement_model, structural_model, choice_model
                )
                draws = self.draw_generator.get_draws()

                grad = self.joint_grad.compute_gradient(
                    self.data,
                    param_dict,
                    draws,
                    measurement_model,
                    structural_model,
                    choice_model
                )

                # 스케일링 적용
                grad_scaled = self.param_scaler.scale_gradient(grad)

                return -grad_scaled

            gradient_func = gradient_wrapper

        # 최적화 실행
        self.iteration_logger.info(f"최적화 시작: {self.config.estimation.optimizer}")

        result = optimize.minimize(
            negative_ll,
            initial_params_scaled,
            method=self.config.estimation.optimizer,
            jac=gradient_func,
            options={
                'maxiter': self.config.estimation.max_iterations,
                'disp': True
            }
        )

        # 결과 처리
        final_params_scaled = result.x
        final_params = self.param_scaler.unscale_parameters(final_params_scaled)
        final_ll = -result.fun

        self.iteration_logger.info(f"최적화 완료: LL = {final_ll:.2f}")

        # 결과 딕셔너리 생성
        return self._create_result_dict(
            params=final_params,
            log_likelihood=final_ll,
            n_iterations=result.nit,
            success=result.success,
            optimizer_result=result
        )

    def _compute_log_likelihood(self, params: np.ndarray, *args) -> float:
        """
        로그우도 계산

        Args:
            params: 파라미터 벡터
            *args: (measurement_model, structural_model, choice_model)

        Returns:
            로그우도 값
        """
        measurement_model, structural_model, choice_model = args

        # 파라미터 언팩
        param_dict = self._unpack_parameters(
            params, measurement_model, structural_model, choice_model
        )

        # Draws 가져오기
        draws = self.draw_generator.get_draws()

        # 로그우도 계산
        ll = self.likelihood_calculator.compute_log_likelihood(
            self.data,
            params,
            measurement_model,
            structural_model,
            choice_model,
            draws=draws,
            param_dict=param_dict
        )

        return ll

    # ========================================================================
    # 파라미터 언팩 메서드
    # ========================================================================

    def _unpack_parameters(self, params: np.ndarray,
                          measurement_model,
                          structural_model,
                          choice_model) -> Dict:
        """
        파라미터 벡터를 각 모델별 딕셔너리로 언팩

        Args:
            params: 파라미터 벡터
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            파라미터 딕셔너리
            {
                'measurement': {...},
                'structural': {...},
                'choice': {...}
            }
        """
        idx = 0
        param_dict = {}

        # 1. 측정모델 파라미터
        measurement_params = {}

        if hasattr(measurement_model, 'models'):
            # 다중 잠재변수
            for lv_name, model in measurement_model.models.items():
                # zeta
                n_indicators = len(model.config.indicators)
                measurement_params[f'zeta_{lv_name}'] = params[idx:idx+n_indicators]
                idx += n_indicators

                # sigma_sq or tau
                if hasattr(model.config, 'measurement_method'):
                    if model.config.measurement_method == 'continuous_linear':
                        measurement_params[f'sigma_sq_{lv_name}'] = params[idx:idx+n_indicators]
                        idx += n_indicators
                    else:
                        n_thresholds = model.n_thresholds
                        for i in range(n_indicators):
                            measurement_params[f'tau_{lv_name}_{i}'] = params[idx:idx+n_thresholds]
                            idx += n_thresholds
        else:
            # 단일 잠재변수
            n_indicators = len(measurement_model.config.indicators)
            measurement_params['zeta'] = params[idx:idx+n_indicators]
            idx += n_indicators

            n_thresholds = measurement_model.n_thresholds
            for i in range(n_indicators):
                measurement_params[f'tau_{i}'] = params[idx:idx+n_thresholds]
                idx += n_thresholds

        param_dict['measurement'] = measurement_params

        # 2. 구조모델 파라미터
        structural_params = {}

        if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
            # 계층적 구조
            for path in structural_model.hierarchical_paths:
                n_predictors = len(path['predictors'])
                structural_params[f"gamma_{path['target']}"] = params[idx:idx+n_predictors]
                idx += n_predictors
        elif hasattr(structural_model, 'endogenous_lv'):
            # 병렬 구조
            structural_params['gamma_lv'] = params[idx:idx+structural_model.n_exo]
            idx += structural_model.n_exo

            if hasattr(structural_model, 'covariates'):
                n_cov = len(structural_model.covariates)
                structural_params['gamma_x'] = params[idx:idx+n_cov]
                idx += n_cov
        else:
            # 단일 잠재변수
            if hasattr(structural_model, 'sociodemographics'):
                n_sociodem = len(structural_model.sociodemographics)
                structural_params['gamma'] = params[idx:idx+n_sociodem]
                idx += n_sociodem

        param_dict['structural'] = structural_params

        # 3. 선택모델 파라미터
        choice_params = {}

        # intercept
        choice_params['intercept'] = params[idx]
        idx += 1

        # beta
        n_attrs = len(choice_model.choice_attributes)
        choice_params['beta'] = params[idx:idx+n_attrs]
        idx += n_attrs

        # lambda
        if hasattr(choice_model, 'moderation_enabled') and choice_model.moderation_enabled:
            choice_params['lambda_main'] = params[idx]
            idx += 1

            if hasattr(choice_model, 'moderator_lvs'):
                for mod_lv in choice_model.moderator_lvs:
                    choice_params[f'lambda_mod_{mod_lv}'] = params[idx]
                    idx += 1
        else:
            choice_params['lambda'] = params[idx]
            idx += 1

        param_dict['choice'] = choice_params

        return param_dict

