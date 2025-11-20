"""
Simultaneous Estimation for ICLV Models

ICLV 모델의 동시 추정 엔진입니다.
Apollo 패키지의 동시 추정 방법론을 Python으로 구현합니다.

참조:
- King (2022) - Apollo 패키지 사용
- Train (2009) - Discrete Choice Methods with Simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from scipy import optimize
from scipy.stats import norm, qmc
from scipy.special import logsumexp
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

from .gradient_calculator import (
    MeasurementGradient,
    StructuralGradient,
    ChoiceGradient,
    JointGradient
)
from .parameter_scaler import ParameterScaler
from .parameter_context import ParameterContext
from .bhhh_calculator import BHHHCalculator
from .parameter_manager import ParameterManager
from .gpu_compute_state import GPUComputeState

logger = logging.getLogger(__name__)


# ============================================================================
# 병렬처리를 위한 전역 함수 (pickle 가능)
# ============================================================================

def _compute_individual_likelihood_parallel(args):
    """
    개인별 우도 계산 (병렬처리용 전역 함수)

    Args:
        args: (ind_data_dict, ind_draws, param_dict, config_dict)
            - ind_data_dict: 개인 데이터 (dict 형태)
            - ind_draws: Halton draws
            - param_dict: 파라미터 딕셔너리
            - config_dict: 설정 정보

    Returns:
        개인의 로그우도
    """
    # 병렬 프로세스에서 불필요한 로그 억제
    import logging
    logging.getLogger('root').setLevel(logging.CRITICAL)

    from .measurement_equations import OrderedProbitMeasurement
    from .structural_equations import LatentVariableRegression
    from .choice_equations import BinaryProbitChoice
    from .iclv_config import MeasurementConfig, StructuralConfig, ChoiceConfig

    ind_data_dict, ind_draws, param_dict, config_dict = args

    # DataFrame 복원
    ind_data = pd.DataFrame(ind_data_dict)

    # 모델 재생성 (각 프로세스에서)
    measurement_config = MeasurementConfig(**config_dict['measurement'])
    structural_config = StructuralConfig(**config_dict['structural'])
    choice_config = ChoiceConfig(**config_dict['choice'])

    measurement_model = OrderedProbitMeasurement(measurement_config)
    structural_model = LatentVariableRegression(structural_config)
    choice_model = BinaryProbitChoice(choice_config)

    # 우도 계산
    draw_lls = []

    for j, draw in enumerate(ind_draws):
        # 구조모델: LV = γ*X + η
        lv = structural_model.predict(ind_data, param_dict['structural'], draw)

        # 측정모델 우도: P(Indicators|LV)
        ll_measurement = measurement_model.log_likelihood(
            ind_data, lv, param_dict['measurement']
        )

        # Panel Product: 개인의 여러 선택 상황에 대한 확률을 곱함
        choice_set_lls = []
        for idx in range(len(ind_data)):
            ll_choice_t = choice_model.log_likelihood(
                ind_data.iloc[idx:idx+1],
                lv,
                param_dict['choice']
            )
            choice_set_lls.append(ll_choice_t)

        ll_choice = sum(choice_set_lls)

        # 구조모델 우도: P(LV|X)
        ll_structural = structural_model.log_likelihood(
            ind_data, lv, param_dict['structural'], draw
        )

        # 결합 로그우도
        draw_ll = ll_measurement + ll_choice + ll_structural

        if not np.isfinite(draw_ll):
            draw_ll = -1e10

        draw_lls.append(draw_ll)

    # logsumexp를 사용하여 평균 계산
    person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))

    return person_ll


class HaltonDrawGenerator:
    """
    Halton 시퀀스 생성기
    
    준난수(Quasi-random) 시퀀스를 생성하여 시뮬레이션 정확도를 향상시킵니다.
    일반 난수보다 공간을 더 균등하게 커버합니다.
    
    참조: Apollo 패키지의 Halton draws
    """
    
    def __init__(self, n_draws: int, n_individuals: int, n_dimensions: int = 1,
                 scramble: bool = True, seed: Optional[int] = None):
        """
        Args:
            n_draws: 개인당 draw 수
            n_individuals: 개인 수
            n_dimensions: 차원 수 (1차 LV 개수 + 2차+ LV 개수)
            scramble: 스크램블 여부 (권장)
            seed: 난수 시드
        """
        self.n_draws = n_draws
        self.n_individuals = n_individuals
        self.n_dimensions = n_dimensions
        self.scramble = scramble
        self.seed = seed

        self.draws = None
        self._generate_draws()

    def _generate_draws(self):
        """Halton 시퀀스 생성"""
        logger.info(f"Halton draws 생성: {self.n_individuals} 개인 × {self.n_draws} draws × {self.n_dimensions} 차원")

        # scipy의 Halton 시퀀스 생성기 사용
        sampler = qmc.Halton(d=self.n_dimensions, scramble=self.scramble, seed=self.seed)

        # 균등분포 [0,1] 샘플 생성
        uniform_draws = sampler.random(n=self.n_individuals * self.n_draws)

        # 표준정규분포로 변환 (역누적분포함수)
        normal_draws = norm.ppf(uniform_draws)

        # 형태 재구성
        if self.n_dimensions == 1:
            # 단일 차원: (n_individuals, n_draws)
            self.draws = normal_draws.reshape(self.n_individuals, self.n_draws)
        else:
            # 다차원: (n_individuals, n_draws, n_dimensions)
            self.draws = normal_draws.reshape(self.n_individuals, self.n_draws, self.n_dimensions)

        logger.info(f"Halton draws 생성 완료: shape={self.draws.shape}")
    
    def get_draws(self) -> np.ndarray:
        """생성된 draws 반환"""
        return self.draws
    
    def get_draw_for_individual(self, individual_idx: int) -> np.ndarray:
        """특정 개인의 draws 반환"""
        return self.draws[individual_idx, :]


class SimultaneousEstimator:
    """
    ICLV 모델 동시 추정기
    
    측정모델, 구조모델, 선택모델을 동시에 추정합니다.
    
    결합 우도함수:
    L = ∏ᵢ ∫ P(Choice|LV) × P(Indicators|LV) × P(LV|X) dLV
    
    시뮬레이션 기반 추정:
    L ≈ ∏ᵢ (1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)
    """
    
    def __init__(self, config):
        """
        Args:
            config: ICLVConfig 객체
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.halton_generator = None
        self.data = None
        self.results = None

        # 로그 파일 핸들러 (추정 시작 시 설정)
        self.log_file_handler = None
        self.iteration_logger = None

        # Gradient calculators (Apollo 방식)
        self.measurement_grad = None
        self.structural_grad = None
        self.choice_grad = None
        self.joint_grad = None
        self.use_analytic_gradient = False  # 기본값: 수치적 그래디언트

        # ✅ ParameterManager 초기화
        self.param_manager = ParameterManager(config)
        self.param_names = None  # estimate() 시작 시 생성

    def _setup_iteration_logger(self, log_file_path: str):
        """
        반복 과정 로깅을 위한 파일 핸들러 설정

        Args:
            log_file_path: 로그 파일 경로
        """
        # 반복 과정 전용 로거 생성
        self.iteration_logger = logging.getLogger('iclv_iteration')
        self.iteration_logger.setLevel(logging.INFO)

        # 기존 핸들러 제거 (중복 방지)
        self.iteration_logger.handlers.clear()

        # 파일 핸들러 추가
        self.log_file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        self.log_file_handler.setLevel(logging.INFO)

        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.log_file_handler.setFormatter(formatter)

        self.iteration_logger.addHandler(self.log_file_handler)

        # CSV 로그 파일 설정 (파라미터 및 그래디언트 값 저장용)
        import csv
        csv_log_path = log_file_path.replace('.txt', '_params_grads.csv')
        self.csv_log_file = open(csv_log_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = None  # 첫 번째 기록 시 헤더와 함께 초기화
        self.csv_log_path = csv_log_path

    def _log_params_grads_to_csv(self, iteration, params, grads):
        """
        파라미터와 그래디언트 값을 CSV 파일에 기록

        Args:
            iteration: Major iteration 번호
            params: 파라미터 값 배열 (external scale)
            grads: 그래디언트 값 배열
        """
        import csv

        # 첫 번째 기록 시 헤더 작성
        if self.csv_writer is None:
            fieldnames = ['iteration']

            # 파라미터 이름 추가
            for idx in range(len(params)):
                param_name = self.param_names[idx] if hasattr(self, 'param_names') and idx < len(self.param_names) else f"param_{idx}"
                fieldnames.append(f'{param_name}_value')
                fieldnames.append(f'{param_name}_grad')

            self.csv_writer = csv.DictWriter(self.csv_log_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()

        # 데이터 행 작성
        row = {'iteration': iteration}
        for idx in range(len(params)):
            param_name = self.param_names[idx] if hasattr(self, 'param_names') and idx < len(self.param_names) else f"param_{idx}"
            row[f'{param_name}_value'] = params[idx]
            row[f'{param_name}_grad'] = grads[idx]

        self.csv_writer.writerow(row)
        self.csv_log_file.flush()  # 즉시 디스크에 기록

        # 콘솔 핸들러 제거 (중복 방지 - 파일만 사용)
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(formatter)
        # self.iteration_logger.addHandler(console_handler)

        self.iteration_logger.info("="*70)
        self.iteration_logger.info("ICLV 모델 추정 시작")
        self.iteration_logger.info("="*70)

    def _close_iteration_logger(self):
        """반복 과정 로거 종료"""
        if self.log_file_handler:
            self.iteration_logger.removeHandler(self.log_file_handler)
            self.log_file_handler.close()
            self.log_file_handler = None
    
    def estimate(self, data: pd.DataFrame,
                measurement_model,
                structural_model,
                choice_model,
                log_file: Optional[str] = None,
                initial_params: Optional[Dict] = None) -> Dict:
        """
        ICLV 모델 동시 추정 (측정모델 고정)

        ✅ 측정모델 파라미터는 항상 고정 (CFA 결과 사용)
        ✅ 구조모델 + 선택모델 파라미터만 추정

        Args:
            data: 통합 데이터
            measurement_model: 측정모델 객체 (CFA 결과 포함)
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            log_file: 로그 파일 경로 (None이면 자동 생성)
            initial_params: 초기 파라미터 딕셔너리 (선택, 구조모델 + 선택모델만)

        Returns:
            추정 결과 딕셔너리
        """
        # 로그 파일 설정
        if log_file is None:
            from pathlib import Path
            from datetime import datetime

            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)

            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = results_dir / f'iclv_estimation_log_{timestamp}.txt'

        self._setup_iteration_logger(str(log_file))

        self.iteration_logger.info("SimultaneousEstimator.estimate() 시작")
        self.iteration_logger.info("ICLV 모델 동시 추정 시작")

        # 메모리 모니터의 logger를 iteration_logger로 업데이트
        if hasattr(self, 'memory_monitor') and self.memory_monitor is not None:
            self.memory_monitor.logger = self.iteration_logger

        self.data = data
        n_individuals = data[self.config.individual_id_column].nunique()

        self.iteration_logger.info(f"데이터 shape: {data.shape}")
        self.iteration_logger.info(f"개인 수: {n_individuals}")

        # Halton draws 생성 (이미 설정되어 있으면 건너뛰기)
        if not hasattr(self, 'halton_generator') or self.halton_generator is None:
            # ✅ 다차원 Halton draws: 1차 LV + 2차+ LV
            n_exo = len(structural_model.exogenous_lvs)
            higher_order_lvs = structural_model.get_higher_order_lvs()
            n_higher_order = len(higher_order_lvs)
            n_dimensions = n_exo + n_higher_order

            self.iteration_logger.info(
                f"\n{'='*70}\n"
                f"Halton draws 생성 시작\n"
                f"{'='*70}\n"
                f"  n_draws: {self.config.estimation.n_draws}\n"
                f"  n_individuals: {n_individuals}\n"
                f"  n_dimensions: {n_dimensions}\n"
                f"    - 1차 LV ({n_exo}개): {structural_model.exogenous_lvs}\n"
                f"    - 고차 LV ({n_higher_order}개): {higher_order_lvs}\n"
                f"{'='*70}"
            )
            self.halton_generator = HaltonDrawGenerator(
                n_draws=self.config.estimation.n_draws,
                n_individuals=n_individuals,
                n_dimensions=n_dimensions,
                scramble=self.config.estimation.scramble_halton
            )

            # 🔍 디버깅: 첫 번째 개인의 첫 번째 draw 출력
            draws = self.halton_generator.get_draws()
            self.iteration_logger.info(
                f"\nHalton draws 생성 완료\n"
                f"  Shape: {draws.shape}\n"
                f"  첫 번째 개인의 첫 번째 draw: {draws[0, 0] if draws.ndim > 1 else draws[0]}\n"
                f"{'='*70}\n"
            )
        else:
            self.iteration_logger.info("Halton draws 이미 설정됨 (건너뛰기)")

        # Gradient calculators 초기화 (Apollo 방식)
        use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B', 'BHHH']
        if use_gradient and hasattr(self.config.estimation, 'use_analytic_gradient'):
            self.use_analytic_gradient = self.config.estimation.use_analytic_gradient
        else:
            self.use_analytic_gradient = False

        if self.use_analytic_gradient:
            self.iteration_logger.info("Analytic gradient calculators 초기화 (Apollo 방식)...")

            # 다중 잠재변수 지원 확인
            from .multi_latent_config import MultiLatentConfig
            is_multi_latent = isinstance(self.config, MultiLatentConfig)

            if is_multi_latent:
                # 다중 잠재변수: MultiLatentMeasurementGradient 사용
                from .multi_latent_gradient import MultiLatentMeasurementGradient
                self.measurement_grad = MultiLatentMeasurementGradient(
                    self.config.measurement_configs
                )
                self.iteration_logger.info(f"다중 잠재변수 측정모델 gradient 초기화: {len(self.config.measurement_configs)}개 LV")
            else:
                # 단일 잠재변수
                self.measurement_grad = MeasurementGradient(
                    n_indicators=len(self.config.measurement.indicators),
                    n_categories=self.config.measurement.n_categories
                )
                self.iteration_logger.info("단일 잠재변수 측정모델 gradient 초기화")

            # 구조모델 gradient
            if is_multi_latent:
                # 다중 잠재변수: MultiLatentStructuralGradient 사용
                from .multi_latent_gradient import MultiLatentStructuralGradient
                self.structural_grad = MultiLatentStructuralGradient(
                    n_exo=self.config.structural.n_exo,
                    n_cov=self.config.structural.n_cov,
                    error_variance=self.config.structural.error_variance
                )
                self.iteration_logger.info("다중 잠재변수 구조모델 gradient 초기화")
            else:
                # 단일 잠재변수
                self.structural_grad = StructuralGradient(
                    n_sociodem=len(self.config.structural.sociodemographics),
                    error_variance=1.0
                )
                self.iteration_logger.info("단일 잠재변수 구조모델 gradient 초기화")

            # 선택모델 gradient (다중 잠재변수도 동일)
            self.choice_grad = ChoiceGradient(
                n_attributes=len(self.config.choice.choice_attributes)
            )

            # JointGradient
            if is_multi_latent:
                # 다중 잠재변수: MultiLatentJointGradient 사용
                from .multi_latent_gradient import MultiLatentJointGradient

                # GPU 사용 여부 확인
                use_gpu_gradient = False
                gpu_measurement_model = None

                if hasattr(self, 'use_gpu') and self.use_gpu:
                    if hasattr(self, 'gpu_measurement_model') and self.gpu_measurement_model is not None:
                        use_gpu_gradient = True
                        gpu_measurement_model = self.gpu_measurement_model
                        self.iteration_logger.info("GPU 배치 그래디언트 활성화")

                # 완전 병렬 처리 옵션 확인
                use_full_parallel = getattr(self, 'use_full_parallel', True)

                self.joint_grad = MultiLatentJointGradient(
                    self.measurement_grad,
                    self.structural_grad,
                    self.choice_grad,
                    use_gpu=use_gpu_gradient,
                    gpu_measurement_model=gpu_measurement_model,
                    use_full_parallel=use_full_parallel
                )
                # ✅ iteration_logger와 config 전달
                self.joint_grad.iteration_logger = self.iteration_logger
                self.joint_grad.config = self.config

                self.iteration_logger.info("다중 잠재변수 JointGradient 초기화 완료")
                self.iteration_logger.info("✅ 동시추정: 측정모델 그래디언트 계산 제외 (고정 파라미터)")
            else:
                # 단일 잠재변수
                self.joint_grad = JointGradient(
                    self.measurement_grad,
                    self.structural_grad,
                    self.choice_grad
                )
                self.iteration_logger.info("단일 잠재변수 JointGradient 초기화 완료")

        # ✅ 동시추정 전용 초기 파라미터 설정 (측정모델 제외)
        self.iteration_logger.info("초기 파라미터 설정 시작...")
        initial_params_opt = self._get_initial_parameters_simultaneous(
            measurement_model, structural_model, choice_model,
            user_initial_params=initial_params
        )
        self.iteration_logger.info(f"✅ 초기 파라미터 설정 완료: {len(initial_params_opt)}개 (측정모델 제외)")

        # 최적화할 파라미터 이름 (측정모델 제외)
        param_names_opt = self.param_manager.get_optimized_parameter_names(
            structural_model, choice_model
        )

        # ✅ self.param_names를 최적화 파라미터로 업데이트
        self.param_names = param_names_opt

        # 최적화에 사용할 파라미터
        initial_params = initial_params_opt
        param_names = param_names_opt

        # 파라미터 스케일링 설정 확인
        use_parameter_scaling = getattr(self.config.estimation, 'use_parameter_scaling', True)

        if use_parameter_scaling:
            # Custom scales 생성 (gradient 균형 최적화)
            custom_scales = self._get_custom_scales(param_names)

            # Apollo-style 파라미터 스케일링 초기화
            self.iteration_logger.info("=" * 80)
            self.iteration_logger.info("파라미터 스케일링 초기화 (Gradient-Balanced)")
            self.iteration_logger.info("=" * 80)
            self.param_scaler = ParameterScaler(
                initial_params=initial_params,
                param_names=param_names,
                custom_scales=custom_scales,
                logger=self.iteration_logger
            )

            # 초기 파라미터를 스케일링 (External → Internal)
            initial_params_scaled = self.param_scaler.scale_parameters(initial_params)

            # 스케일링 비교 로깅
            self.param_scaler.log_parameter_comparison(initial_params, initial_params_scaled)
        else:
            # 스케일링 비활성화: 항등 스케일러 사용
            self.iteration_logger.info("=" * 80)
            self.iteration_logger.info("파라미터 스케일링 비활성화")
            self.iteration_logger.info("=" * 80)
            self.param_scaler = ParameterScaler(
                initial_params=initial_params,
                param_names=param_names,
                custom_scales={name: 1.0 for name in param_names},  # 모든 스케일을 1.0으로 설정
                logger=self.iteration_logger
            )
            initial_params_scaled = initial_params  # 스케일링 없음

        # ✅ ParameterContext 생성 (파라미터 변환 로직 단일화)
        self.iteration_logger.info("=" * 80)
        self.iteration_logger.info("ParameterContext 초기화 (동시추정 전용)")
        self.iteration_logger.info("=" * 80)
        param_context = ParameterContext(
            param_manager=self.param_manager,
            param_scaler=self.param_scaler,
            measurement_model=measurement_model,
            logger=self.iteration_logger
        )
        self.iteration_logger.info("=" * 80)

        # 결합 우도함수 정의 (단계별 로깅 추가)
        iteration_count = [0]  # Mutable counter
        best_ll = [-np.inf]  # Track best log-likelihood
        func_call_count = [0]  # 함수 호출 횟수 (우도 계산)
        major_iter_count = [0]  # Major iteration 카운터
        line_search_call_count = [0]  # Line search 내 함수 호출 카운터
        last_major_iter_func_value = [None]  # 마지막 major iteration의 함수값
        last_major_iter_ftol = [None]  # 마지막 major iteration의 ftol 값
        last_major_iter_gtol = [None]  # 마지막 major iteration의 gtol 값
        current_major_iter_start_call = [0]  # 현재 major iteration 시작 시 함수 호출 번호
        line_search_func_values = []  # Line search 중 함수값 기록
        line_search_start_func_value = [None]  # Line search 시작 시 함수값
        line_search_start_params = [None]  # Line search 시작 시 파라미터
        line_search_gradient = [None]  # Line search 시작 시 gradient
        line_search_directional_derivative = [None]  # ∇f(x)^T·d (시작 시)

        def negative_log_likelihood(params_scaled):
            """
            Negative log-likelihood function (스케일된 파라미터 사용)

            Args:
                params_scaled: 스케일된 (internal) 파라미터
                              (측정모델 고정 시 구조모델+선택모델만 포함)

            Returns:
                Negative log-likelihood
            """
            func_call_count[0] += 1

            # ✅ ParameterContext를 사용한 파라미터 변환
            # ✅ 동시추정: params_scaled는 이미 최적화 파라미터만 포함 (8개)
            params_opt = param_context.to_full_external(params_scaled)

            # Line search 중인지 판단
            # Major iteration 시작 직후 첫 호출이 아니면 line search 중
            calls_since_major_start = func_call_count[0] - current_major_iter_start_call[0]

            if calls_since_major_start == 1:
                # Major iteration 시작 시 첫 함수 호출
                context = f"Major Iteration #{major_iter_count[0] + 1} 시작"
                line_search_call_count[0] = 0
                line_search_func_values.clear()
                line_search_start_params[0] = params_scaled.copy()  # ✅ 최적화 파라미터만 (8개)
            elif calls_since_major_start > 1:
                # Line search 중
                line_search_call_count[0] += 1
                context = f"Line Search 함수 호출 #iter{major_iter_count[0] + 1}-{line_search_call_count[0]}"
            else:
                # 초기 호출
                context = "초기 함수값 계산"
                line_search_start_params[0] = params_scaled.copy()  # ✅ 초기 호출 시에도 저장

            # 단계 로그: 우도 계산 시작
            self.iteration_logger.info(f"[{context}] [단계 1/2] 전체 우도 계산")

            ll = self._joint_log_likelihood(
                params_opt, measurement_model, structural_model, choice_model
            )

            # Track best value
            if ll > best_ll[0]:
                best_ll[0] = ll
                improvement = "[NEW BEST]"
            else:
                improvement = ""

            # 함수값 출력
            neg_ll = -ll  # scipy가 최소화하는 값
            log_msg = f"  LL = {ll:12.4f} (Best: {best_ll[0]:12.4f}) {improvement}"
            self.iteration_logger.info(log_msg)

            # Line search 중이면 함수값 변화 로깅
            if calls_since_major_start == 1:
                line_search_start_func_value[0] = neg_ll
                # ✅ params_scaled 저장 (최적화 파라미터만, 8개)
                # (이미 위에서 line_search_start_params[0] = params_scaled.copy() 실행됨)
            elif calls_since_major_start > 1:
                line_search_func_values.append(neg_ll)

                # 파라미터 변화량과 함수값 변화 로깅
                if line_search_start_params[0] is not None:
                    # ✅ 같은 타입끼리 비교 (params_scaled vs params_scaled)
                    param_diff = params_scaled - line_search_start_params[0]
                    param_change_norm = np.linalg.norm(param_diff)

                    f_start = line_search_start_func_value[0]
                    f_current = neg_ll
                    f_decrease = f_start - f_current

                    self.iteration_logger.info(
                        f"  파라미터 변화량 (L2 norm): {param_change_norm:.6e}\n"
                        f"  함수값 변화: {f_decrease:+.4f} ({'감소' if f_decrease > 0 else '증가'})"
                    )

                # Line search가 maxls에 도달했는지 체크
                if line_search_call_count[0] >= 10:  # maxls = 10
                    self.iteration_logger.info(
                        f"\n⚠️  [Line Search 경고] maxls={10}에 도달했습니다.\n"
                        f"  시작 함수값: {line_search_start_func_value[0]:.4f}\n"
                        f"  현재 함수값: {neg_ll:.4f}\n"
                        f"  변화량: {neg_ll - line_search_start_func_value[0]:.4f}\n"
                        f"  Line search가 Wolfe 조건을 만족하는 step size를 찾지 못했을 수 있습니다."
                    )

            return neg_ll

        # Get parameter bounds (측정모델 제외)
        self.iteration_logger.info("파라미터 bounds 계산 시작...")
        bounds = self.param_manager.get_optimized_parameter_bounds(
            structural_model, choice_model
        )
        self.iteration_logger.info(f"파라미터 bounds 계산 완료 (총 {len(bounds)}개)")

        # 최적화 방법 선택
        use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B', 'BHHH']

        # Gradient 함수 정의 (Apollo 방식)
        grad_call_count = [0]  # 그래디언트 호출 횟수

        def gradient_function(params_scaled):
            """
            Analytic gradient 계산 (Apollo 방식, 스케일된 파라미터 사용)

            Args:
                params_scaled: 스케일된 (internal) 파라미터
                              (측정모델 고정 시 구조모델+선택모델만 포함)

            Returns:
                Gradient w.r.t. scaled parameters
            """
            if not self.use_analytic_gradient:
                return None  # 수치적 그래디언트 사용

            grad_call_count[0] += 1

            # ✅ ParameterContext를 사용한 파라미터 변환
            # ✅ 동시추정: params_scaled는 이미 최적화 파라미터만 포함 (8개)
            params_opt = param_context.to_full_external(params_scaled)

            # Line search 중인지 판단 (gradient_function에서도 계산 필요)
            calls_since_major_start = func_call_count[0] - current_major_iter_start_call[0]

            # 단계 로그: 그래디언트 계산 시작 (모든 호출에서 출력)
            # Major iteration 번호 포함
            context_str = f"iter{major_iter_count[0]}-{calls_since_major_start}" if major_iter_count[0] > 0 else "init"
            self.iteration_logger.info(f"[단계 2/2] Analytic Gradient 계산 #{context_str}")

            # 메모리 체크 (그래디언트 계산 전) - 비활성화
            # if hasattr(self, 'memory_monitor'):
            #     # 5회마다 메모리 상태 로깅
            #     if grad_call_count[0] % 5 == 1:
            #         self.memory_monitor.log_memory_stats(f"Gradient 계산 #{grad_call_count[0]}")
            #
            #     # 항상 임계값 체크 및 필요시 정리
            #     mem_info = self.memory_monitor.check_and_cleanup(f"Gradient 계산 #{grad_call_count[0]}")

            # ✅ 리팩토링: 순수한 gradient 계산은 _compute_gradient 메서드로 위임
            # ✅ 동시추정: _compute_gradient는 이미 최적화 파라미터 그래디언트만 반환 (측정모델 제외)
            neg_grad_opt = self._compute_gradient(
                params_opt, measurement_model, structural_model, choice_model
            )

            # ✅ ParameterContext를 사용한 그래디언트 스케일링
            neg_grad_scaled = param_context.scale_gradient(neg_grad_opt)

            # Line search 중인지 판단
            calls_since_major_start = func_call_count[0] - current_major_iter_start_call[0]

            # Gradient 방향 검증 (첫 번째 호출 시)
            if grad_call_count[0] == 1:
                grad_norm_opt = np.linalg.norm(neg_grad_opt)
                grad_norm_scaled = np.linalg.norm(neg_grad_scaled)
                self.iteration_logger.info(
                    f"\n[Gradient 방향 검증 - External (원본)]\n"
                    f"  Gradient norm: {grad_norm_opt:.6e}\n"
                    f"  Gradient max: {np.max(np.abs(neg_grad_opt)):.6e}\n"
                    f"  Gradient (처음 5개): {neg_grad_opt[:5]}\n"
                    f"  Gradient (마지막 5개): {neg_grad_opt[-5:]}\n"
                )
                self.iteration_logger.info(
                    f"\n[Gradient 방향 검증 - Internal (스케일됨)]\n"
                    f"  Gradient norm: {grad_norm_scaled:.6e}\n"
                    f"  Gradient max: {np.max(np.abs(neg_grad_scaled)):.6e}\n"
                    f"  Gradient (처음 5개): {neg_grad_scaled[:5]}\n"
                    f"  Gradient (마지막 5개): {neg_grad_scaled[-5:]}\n"
                    f"  주의: scipy는 이 gradient를 사용하여 descent direction을 계산합니다.\n"
                    f"       d = -H^(-1) · gradient이므로, gradient가 양수면 d는 음수 방향입니다."
                )

                # 스케일링 비교 로깅
                self.param_scaler.log_gradient_comparison(neg_grad_opt, neg_grad_scaled)

            # Line search 시작 시 방향 미분 저장
            if calls_since_major_start == 1:
                # Major iteration 시작 시 gradient 저장 (스케일된 gradient 사용)
                line_search_gradient[0] = neg_grad_scaled.copy()
                # 다음 함수 호출에서 탐색 방향을 알 수 있으므로, 방향 미분은 나중에 계산

                # ✅ Major iteration 시작 시 파라미터 저장 (탐색 방향 계산용)
                if not hasattr(gradient_function, 'major_iter_start_params'):
                    gradient_function.major_iter_start_params = {}
                gradient_function.major_iter_start_params[major_iter_count[0] + 1] = params_scaled.copy()

            # Line search 중이면 Wolfe 조건 계산
            elif calls_since_major_start > 1 and line_search_start_params[0] is not None:
                # 탐색 방향 계산: d = params_scaled - line_search_start_params
                search_direction = params_scaled - line_search_start_params[0]

                # ✅ 첫 line search 호출 시 탐색 방향 로깅
                if line_search_call_count[0] == 1:
                    iter_num = major_iter_count[0] + 1

                    # 탐색 방향 통계
                    d_norm = np.linalg.norm(search_direction)
                    d_max = np.max(np.abs(search_direction))

                    # Gradient와 탐색 방향 비교
                    grad_norm = np.linalg.norm(neg_grad_scaled)

                    # d = -H^(-1) · grad이므로, 만약 H = I이면 d ≈ -grad
                    # 상관계수 계산 (방향 유사도)
                    if grad_norm > 0 and d_norm > 0:
                        # 정규화된 벡터 간 내적 = 코사인 유사도
                        cosine_similarity = -np.dot(search_direction, neg_grad_scaled) / (d_norm * grad_norm)
                    else:
                        cosine_similarity = 0.0

                    # ✅ 이전 iteration 파라미터와 비교
                    if hasattr(gradient_function, 'major_iter_start_params') and (iter_num - 1) in gradient_function.major_iter_start_params:
                        prev_params = gradient_function.major_iter_start_params[iter_num - 1]
                        param_change = params_scaled - prev_params
                        param_change_norm = np.linalg.norm(param_change)

                        # 실제 파라미터 변화 = α × d (이전 iteration에서)
                        # 현재는 새 iteration의 첫 line search이므로, 이전 iteration의 최종 결과
                        param_change_info = (
                            f"\n  이전 iteration 대비 파라미터 변화:\n"
                            f"    - 변화량 norm: {param_change_norm:.6e}\n"
                            f"    - 변화량 max: {np.max(np.abs(param_change)):.6e}\n"
                            f"    - 변화 상위 5개 인덱스: {np.argsort(np.abs(param_change))[-5:][::-1]}\n"
                            f"    - 변화 상위 5개 값: {param_change[np.argsort(np.abs(param_change))[-5:][::-1]]}\n"
                        )
                    else:
                        param_change_info = ""

                    self.iteration_logger.info(
                        f"\n[탐색 방향 분석 - Iteration #{iter_num}]\n"
                        f"  탐색 방향 d norm: {d_norm:.6e}\n"
                        f"  탐색 방향 d max: {d_max:.6e}\n"
                        f"  Gradient norm: {grad_norm:.6e}\n"
                        f"  d와 -grad의 코사인 유사도: {cosine_similarity:.6f}\n"
                        f"    (1.0 = 완전 동일 방향 [H=I], 0.0 = 직교, -1.0 = 반대 방향)\n"
                        f"  d 상위 5개: {search_direction[:5]}\n"
                        f"  -grad 상위 5개: {-neg_grad_scaled[:5]}\n"
                        f"  → Hessian이 방향을 {'거의 조정 안 함' if cosine_similarity > 0.99 else '조정함'}\n"
                        f"{param_change_info}"
                    )

                    # ✅ 실제 계산에 사용된 파라미터 값 로깅 비활성화 (요청사항 4)
                    # params_external = self.param_scaler.unscale_parameters(params_scaled)
                    # top_10_indices = np.argsort(np.abs(params_external))[-10:][::-1]
                    # self.iteration_logger.info(
                    #     f"\n[실제 계산에 사용된 파라미터 값 - Iteration #{iter_num}]\n"
                    #     f"  (External scale, 상위 10개)\n"
                    # )
                    # for idx in top_10_indices:
                    #     param_name = self.param_names[idx] if hasattr(self, 'param_names') and idx < len(self.param_names) else f"param_{idx}"
                    #     self.iteration_logger.info(
                    #         f"    [{idx:2d}] {param_name:40s}: {params_external[idx]:+.6e} (internal: {params_scaled[idx]:+.6e})"
                    #     )
                    pass

                # 현재 위치에서 방향 미분: ∇f(x + α·d)^T·d (스케일된 gradient 사용)
                directional_derivative_new = np.dot(neg_grad_scaled, search_direction)

                # Line search 시작 시 방향 미분 계산 (첫 line search 호출 시)
                if line_search_directional_derivative[0] is None and line_search_gradient[0] is not None:
                    # 시작 위치에서 방향 미분: ∇f(x)^T·d
                    line_search_directional_derivative[0] = np.dot(line_search_gradient[0], search_direction)

                # Wolfe 조건 체크
                if line_search_directional_derivative[0] is not None:
                    dd_start = line_search_directional_derivative[0]
                    dd_new = directional_derivative_new

                    # Armijo 조건: f(x+αd) ≤ f(x) + c₁·α·∇f(x)ᵀd
                    # 여기서 α = ||search_direction|| / ||d||인데, d를 모르므로
                    # 대신 이전 함수 호출의 함수값을 사용
                    c1 = 1e-3  # 조정된 값 (기본값: 1e-4)

                    # 이전 line search 호출의 함수값 가져오기
                    if len(line_search_func_values) > 0:
                        f_start = line_search_start_func_value[0]
                        f_current = line_search_func_values[-1]  # 가장 최근 함수값

                        # Armijo 조건 근사 체크
                        # f(x+αd) - f(x) ≤ c₁·α·∇f(x)ᵀd
                        # α·∇f(x)ᵀd를 정확히 모르므로, 단순히 함수값 감소 체크
                        armijo_satisfied = (f_current <= f_start)
                    else:
                        armijo_satisfied = None

                    # Curvature 조건: |∇f(x + α·d)^T·d| ≤ c2·|∇f(x)^T·d|
                    c2 = 0.5  # 조정된 값 (기본값: 0.9)
                    curvature_lhs = abs(dd_new)
                    curvature_rhs = c2 * abs(dd_start)
                    curvature_satisfied = curvature_lhs <= curvature_rhs

                    # Strong Wolfe 조건 = Armijo + Curvature
                    strong_wolfe_satisfied = (armijo_satisfied and curvature_satisfied) if armijo_satisfied is not None else curvature_satisfied

                    # Wolfe 조건 체크 로깅 비활성화 (요청사항 1)
                    # armijo_msg = ""
                    # if armijo_satisfied is not None:
                    #     armijo_msg = f"  Armijo 조건 (c1={c1}): {'✓ 만족' if armijo_satisfied else '❌ 불만족'}\n"
                    # self.iteration_logger.info(
                    #     f"\n[Wolfe 조건 체크]\n"
                    #     f"{armijo_msg}"
                    #     f"  Curvature 조건 (c2={c2}): {'✓ 만족' if curvature_satisfied else '❌ 불만족'}\n"
                    #     f"  → Strong Wolfe: {'✓ 만족' if strong_wolfe_satisfied else '❌ 불만족'}\n"
                    #     f"  → Gradient가 {'충분히 평평해짐' if curvature_satisfied else '아직 가파름'}"
                    # )

            # 스케일된 gradient 반환 (optimizer는 internal parameters에 대해 작동)
            return neg_grad_scaled

        print("=" * 70, flush=True)
        if use_gradient:
            print(f"최적화 시작: {self.config.estimation.optimizer} (gradient-based)", flush=True)
            if self.use_analytic_gradient:
                print("Analytic gradient 사용 (Apollo 방식 + Parameter Scaling)", flush=True)
            else:
                print("수치적 그래디언트 사용 (2-point finite difference)", flush=True)
        else:
            print("최적화 시작: Nelder-Mead (gradient-free)", flush=True)
        print(f"초기 파라미터 개수: {len(initial_params_scaled)}", flush=True)
        self.iteration_logger.info(f"최대 반복 횟수: {self.config.estimation.max_iterations}")
        self.iteration_logger.info("=" * 70)

        # 병렬처리 설정 로깅
        use_parallel = getattr(self.config.estimation, 'use_parallel', False)
        if use_parallel:
            n_cores = getattr(self.config.estimation, 'n_cores', None)
            if n_cores is None:
                n_cores = max(1, multiprocessing.cpu_count() - 1)
            self.iteration_logger.info(f"병렬처리 활성화: {n_cores} 코어 사용")
        else:
            self.iteration_logger.info("순차처리 사용")

        # 조기 종료를 위한 Wrapper 클래스 (BFGS 정상 종료 활용)
        class EarlyStoppingWrapper:
            """
            목적 함수와 gradient 함수를 감싸서 조기 종료 구현
            StopIteration 예외 대신 매우 큰 값을 반환하여 BFGS가 정상 종료하도록 유도
            → BFGS가 정상 종료하면 result.hess_inv 자동 제공 (추가 계산 0회!)
            """

            def __init__(self, func, grad_func, patience=5, tol=1e-6, logger=None, iteration_logger=None, param_scaler=None, param_names=None, parent_estimator=None):
                """
                Args:
                    func: 목적 함수 (negative log-likelihood)
                    grad_func: Gradient 함수
                    patience: 함수 호출 기준 개선 없는 횟수 (기본값: 5)
                    tol: LL 변화 허용 오차 (절대값)
                    logger: 메인 로거
                    iteration_logger: 반복 로거
                    param_scaler: 파라미터 스케일러 (외부 클래스에서 전달)
                    param_names: 파라미터 이름 리스트 (외부 클래스에서 전달)
                    parent_estimator: 부모 estimator 인스턴스 (외부 클래스에서 전달)
                """
                self.func = func
                self.grad_func = grad_func
                self.patience = patience
                self.tol = tol
                self.logger = logger
                self.iteration_logger = iteration_logger
                self.param_scaler = param_scaler  # ✅ 외부에서 전달받은 param_scaler 저장
                self.param_names = param_names    # ✅ 외부에서 전달받은 param_names 저장
                self.parent_estimator = parent_estimator  # ✅ 부모 estimator 저장

                self.best_ll = np.inf
                self.best_x = None  # 최적 파라미터 저장
                self.no_improvement_count = 0
                self.func_call_count = 0
                self.grad_call_count = 0
                self.early_stopped = False
                self.bfgs_iteration_count = 0  # BFGS iteration 카운터

            def objective(self, x):
                """
                목적 함수 wrapper - 조기 종료 시 매우 큰 값 반환
                """
                # 이미 조기 종료된 경우: 매우 큰 값 반환하여 BFGS가 종료하도록 유도
                if self.early_stopped:
                    return 1e10

                self.func_call_count += 1
                current_ll = self.func(x)

                # LL 개선 체크
                if current_ll < self.best_ll - self.tol:
                    # 명확한 개선
                    self.best_ll = current_ll
                    self.best_x = x.copy()  # 최적 파라미터 저장
                    self.no_improvement_count = 0
                else:
                    # 개선 없음
                    self.no_improvement_count += 1

                # 조기 종료 조건 체크
                if self.no_improvement_count >= self.patience:
                    self.early_stopped = True
                    msg = f"조기 종료: {self.patience}회 연속 함수 호출에서 LL 개선 없음 (Best LL={self.best_ll:.4f})"
                    if self.iteration_logger:
                        self.iteration_logger.info(msg)
                    # StopIteration 대신 매우 큰 값 반환
                    return 1e10

                return current_ll

            def gradient(self, x):
                """
                Gradient 함수 wrapper - 조기 종료 시 0 벡터 반환
                """
                # 이미 조기 종료된 경우: 0 벡터 반환하여 BFGS가 종료하도록 유도
                if self.early_stopped:
                    return np.zeros_like(x)

                self.grad_call_count += 1
                return self.grad_func(x)

            def callback(self, xk):
                """
                BFGS callback - 매 Major iteration마다 호출됨
                조기 종료 시 최적 파라미터로 복원
                ftol AND gtol 조건을 모두 체크하여 조기 종료
                """
                self.bfgs_iteration_count += 1
                major_iter_count[0] = self.bfgs_iteration_count

                # ✅ Hessian 추적을 위한 변수 저장
                if not hasattr(self, 'prev_xk'):
                    self.prev_xk = None
                if not hasattr(self, 'prev_grad'):
                    self.prev_grad = None

                # ✅ ftol AND gtol 조건 체크를 위한 변수
                if not hasattr(self, 'ftol_threshold'):
                    self.ftol_threshold = 1e-6  # ftol 기준
                if not hasattr(self, 'gtol_threshold'):
                    self.gtol_threshold = 1e-5  # gtol 기준

                # Major iteration 완료 로깅
                if self.iteration_logger:
                    # 현재 함수값 계산
                    current_f = self.func(xk)
                    current_ll = -current_f

                    # Line search 통계
                    line_search_calls = line_search_call_count[0]

                    # Line search 성공 여부 판단
                    if line_search_start_func_value[0] is not None:
                        f_start = line_search_start_func_value[0]
                        f_final = current_f
                        f_decrease = f_start - f_final

                        if f_decrease > 0:
                            ls_status = f"[OK] 성공 (함수값 감소: {f_decrease:.4f})"
                        elif f_decrease == 0:
                            ls_status = f"[WARN] 정체 (함수값 변화 없음)"
                        else:
                            ls_status = f"[FAIL] 실패 (함수값 증가: {-f_decrease:.4f})"
                    else:
                        ls_status = "N/A (첫 iteration)"

                    # ftol 계산 (이전 major iteration과 비교)
                    if last_major_iter_func_value[0] is not None:
                        f_prev = last_major_iter_func_value[0]
                        f_curr = current_f
                        rel_change = abs(f_prev - f_curr) / max(abs(f_prev), abs(f_curr), 1.0)

                        # 이전 ftol 대비 변화량 계산
                        if last_major_iter_ftol[0] is not None:
                            ftol_change = rel_change - last_major_iter_ftol[0]
                            ftol_change_pct = (ftol_change / last_major_iter_ftol[0]) * 100 if last_major_iter_ftol[0] != 0 else 0
                            ftol_status = f"ftol = {rel_change:.6e} (기준: 1e-3, 변화: {ftol_change:+.2e} [{ftol_change_pct:+.1f}%])"
                        else:
                            ftol_status = f"ftol = {rel_change:.6e} (기준: 1e-3)"

                        if rel_change <= 1e-3:
                            ftol_status += " [OK] 수렴 조건 만족"

                        last_major_iter_ftol[0] = rel_change
                    else:
                        ftol_status = "ftol = N/A (첫 iteration)"

                    # Gradient norm 계산
                    if self.grad_func:
                        grad = self.grad_func(xk)
                        grad_norm = np.linalg.norm(grad, ord=np.inf)

                        # ✅ 고정되지 않은 파라미터의 그래디언트만 계산 (L-BFGS-B의 projected gradient와 유사)
                        non_zero_grad = grad[np.abs(grad) > 1e-10]
                        if len(non_zero_grad) > 0:
                            grad_norm_active = np.linalg.norm(non_zero_grad, ord=np.inf)
                            n_active = len(non_zero_grad)
                        else:
                            grad_norm_active = 0.0
                            n_active = 0

                        # 이전 gtol 대비 변화량 계산
                        if last_major_iter_gtol[0] is not None:
                            gtol_change = grad_norm - last_major_iter_gtol[0]
                            gtol_change_pct = (gtol_change / last_major_iter_gtol[0]) * 100 if last_major_iter_gtol[0] != 0 else 0
                            gtol_status = f"gtol = {grad_norm:.6e} (전체), {grad_norm_active:.6e} (활성 {n_active}개) (기준: 1e-3, 변화: {gtol_change:+.2e} [{gtol_change_pct:+.1f}%])"
                        else:
                            gtol_status = f"gtol = {grad_norm:.6e} (전체), {grad_norm_active:.6e} (활성 {n_active}개) (기준: 1e-3)"

                        if grad_norm_active <= 1e-3:
                            gtol_status += " [OK] 활성 파라미터 수렴"

                        last_major_iter_gtol[0] = grad_norm

                        # ✅ 전체 파라미터 값과 그래디언트 값 로깅 (요청사항 3)
                        # 상위 10개 대신 전체 파라미터 출력
                        params_external = self.param_scaler.unscale_parameters(xk)

                        gradient_details = "\n  전체 파라미터 값 및 그래디언트:\n"
                        for idx in range(len(params_external)):
                            param_name = self.param_names[idx] if hasattr(self, 'param_names') and idx < len(self.param_names) else f"param_{idx}"
                            gradient_details += f"    [{idx:2d}] {param_name:50s}: param={params_external[idx]:+12.6e}, grad={grad[idx]:+12.6e}\n"

                        # CSV 파일에 기록 (요청사항 5)
                        if self.parent_estimator is not None and hasattr(self.parent_estimator, '_log_params_grads_to_csv'):
                            self.parent_estimator._log_params_grads_to_csv(major_iter_count[0], params_external, grad)
                    else:
                        gtol_status = "gtol = N/A"
                        gradient_details = ""

                    # ✅ Hessian 업데이트 정보 로깅
                    hessian_update_info = ""
                    if self.prev_xk is not None and self.prev_grad is not None:
                        # s_k = x_k - x_{k-1}
                        s_k = xk - self.prev_xk
                        # y_k = grad_k - grad_{k-1}
                        current_grad = self.grad_func(xk)
                        y_k = current_grad - self.prev_grad

                        # s_k, y_k 통계
                        s_norm = np.linalg.norm(s_k)
                        y_norm = np.linalg.norm(y_k)
                        s_y_dot = np.dot(s_k, y_k)

                        # BFGS 업데이트 조건 체크
                        if s_y_dot > 0:
                            rho = 1.0 / s_y_dot
                            hessian_update_info = (
                                f"\n  Hessian 업데이트 정보:\n"
                                f"    - s_k (파라미터 변화) norm: {s_norm:.6e}\n"
                                f"    - y_k (gradient 변화) norm: {y_norm:.6e}\n"
                                f"    - s_k^T · y_k: {s_y_dot:.6e} (양수 OK)\n"
                                f"    - ρ = 1/(s_k^T · y_k): {rho:.6e}\n"
                                f"    - s_k 상위 5개: {s_k[:5]}\n"
                                f"    - y_k 상위 5개: {y_k[:5]}\n"
                            )
                        else:
                            hessian_update_info = (
                                f"\n  ⚠️  Hessian 업데이트 경고:\n"
                                f"    - s_k^T · y_k: {s_y_dot:.6e} (음수 또는 0 ❌)\n"
                                f"    - BFGS 업데이트가 건너뛰어질 수 있음!\n"
                            )
                    else:
                        hessian_update_info = "\n  Hessian 업데이트: 첫 iteration (초기 H = I)\n"

                    # 현재 상태 저장 (다음 iteration을 위해)
                    self.prev_xk = xk.copy()
                    self.prev_grad = self.grad_func(xk).copy()

                    self.iteration_logger.info(
                        f"\n{'='*80}\n"
                        f"[Major Iteration #{self.bfgs_iteration_count} 완료]\n"
                        f"  최종 LL: {current_ll:.4f}\n"
                        f"  Line Search: {line_search_calls}회 함수 호출 - {ls_status}\n"
                        f"  함수 호출: {self.func_call_count}회 (누적)\n"
                        f"  그래디언트 호출: {self.grad_call_count}회 (누적)\n"
                        f"  수렴 조건:\n"
                        f"    - {ftol_status}\n"
                        f"    - {gtol_status}\n"
                        f"{gradient_details}"
                        f"{hessian_update_info}"
                        f"  Hessian 근사: BFGS 공식으로 업데이트 완료\n"
                        f"{'='*80}"
                    )

                    # ✅ ftol AND gtol 조건 체크 (둘 다 만족해야 조기 종료)
                    ftol_satisfied = False
                    gtol_satisfied = False

                    if last_major_iter_func_value[0] is not None:
                        f_prev = last_major_iter_func_value[0]
                        f_curr = current_f
                        rel_change = abs(f_prev - f_curr) / max(abs(f_prev), abs(f_curr), 1.0)
                        ftol_satisfied = (rel_change <= self.ftol_threshold)

                    if self.grad_func:
                        grad = self.grad_func(xk)
                        grad_norm_active = np.linalg.norm(grad[np.abs(grad) > 1e-10], ord=np.inf) if np.any(np.abs(grad) > 1e-10) else 0.0
                        gtol_satisfied = (grad_norm_active <= self.gtol_threshold)

                    # ftol AND gtol 모두 만족하면 조기 종료
                    if ftol_satisfied and gtol_satisfied:
                        self.early_stopped = True
                        self.best_x = xk.copy()
                        msg = (
                            f"\n{'='*80}\n"
                            f"✅ 수렴 완료: ftol AND gtol 조건 모두 만족\n"
                            f"  - ftol: {rel_change:.6e} <= {self.ftol_threshold:.6e} ✓\n"
                            f"  - gtol: {grad_norm_active:.6e} <= {self.gtol_threshold:.6e} ✓\n"
                            f"  - Major iteration: {self.bfgs_iteration_count}\n"
                            f"  - 최종 LL: {current_ll:.4f}\n"
                            f"{'='*80}"
                        )
                        if self.iteration_logger:
                            self.iteration_logger.info(msg)
                        # StopIteration 대신 early_stopped 플래그 설정
                        # 다음 objective/gradient 호출 시 큰 값/0 벡터 반환하여 종료 유도

                    # 다음 major iteration을 위한 준비
                    last_major_iter_func_value[0] = current_f
                    current_major_iter_start_call[0] = func_call_count[0]
                    line_search_call_count[0] = 0  # Line search 카운터 리셋
                    line_search_func_values.clear()
                    line_search_directional_derivative[0] = None  # 방향 미분 리셋

                if self.early_stopped and self.best_x is not None:
                    # 조기 종료 후에는 최적 파라미터를 유지
                    xk[:] = self.best_x

        if use_gradient:
            self.iteration_logger.info(f"최적화 시작: {self.config.estimation.optimizer} (gradient-based)")
            if self.use_analytic_gradient:
                self.iteration_logger.info("Analytic gradient 사용 (Apollo 방식)")
            else:
                self.iteration_logger.info("수치적 그래디언트 사용 (2-point finite difference)")

            # 조기 종료 설정 확인
            use_early_stopping = getattr(self.config.estimation, 'early_stopping', False)
            early_stopping_patience = getattr(self.config.estimation, 'early_stopping_patience', 5)
            early_stopping_tol = getattr(self.config.estimation, 'early_stopping_tol', 1e-6)

            # 조기 종료 Wrapper 생성
            early_stopping_wrapper = EarlyStoppingWrapper(
                func=negative_log_likelihood,
                grad_func=gradient_function if self.use_analytic_gradient else None,
                patience=early_stopping_patience if use_early_stopping else 999999,  # 비활성화 시 매우 큰 값
                tol=early_stopping_tol,
                logger=self.logger,
                iteration_logger=self.iteration_logger,
                param_scaler=self.param_scaler,  # ✅ param_scaler 전달
                param_names=self.param_names,    # ✅ param_names 전달
                parent_estimator=self            # ✅ parent_estimator 전달
            )

            # 초기 함수 호출 시작 위치 설정
            current_major_iter_start_call[0] = func_call_count[0]

            if use_early_stopping:
                self.iteration_logger.info(f"조기 종료 활성화: {early_stopping_patience}회 연속 함수 호출에서 LL 개선 없으면 종료 (tol={early_stopping_tol})")
            else:
                self.iteration_logger.info("조기 종료 비활성화 (정상 종료만 사용)")

            # BFGS 또는 L-BFGS-B (정상 종료로 처리)
            # 수치적 그래디언트 함수 (epsilon 제어)
            if not self.use_analytic_gradient:
                from scipy.optimize import approx_fprime

                # 그래디언트 호출 카운터
                grad_call_count = [0]

                def numerical_gradient(x):
                    grad_call_count[0] += 1
                    grad = approx_fprime(x, early_stopping_wrapper.objective, epsilon=1e-4)

                    # 처음 5번만 로깅
                    if grad_call_count[0] <= 5:
                        self.iteration_logger.info(f"[그래디언트 계산 #{grad_call_count[0]}]")
                        self.iteration_logger.info(f"  파라미터 (처음 10개): {x[:10]}")
                        self.iteration_logger.info(f"  그래디언트 (처음 10개): {grad[:10]}")
                        self.iteration_logger.info(f"  그래디언트 norm: {np.linalg.norm(grad):.6f}")
                        self.iteration_logger.info(f"  그래디언트 max: {np.max(np.abs(grad)):.6f}")

                    return grad

                jac_function = numerical_gradient
            else:
                jac_function = early_stopping_wrapper.gradient

            # Optimizer별 옵션 설정
            if self.config.estimation.optimizer == 'BHHH':
                # BHHH: Newton-CG with custom Hessian (OPG)
                optimizer_options = {
                    'maxiter': 200,  # Major iteration 최대 횟수
                    'xtol': 1e-5,    # 파라미터 변화 허용 오차
                    'disp': True
                }

                # BHHH Hessian 함수 생성
                self.iteration_logger.info("BHHH 최적화 알고리즘 초기화...")
                self.iteration_logger.info("  - 방법: Newton-CG with OPG (Outer Product of Gradients)")
                self.iteration_logger.info("  - Hessian 계산: 각 iteration마다 개인별 gradient로 OPG 계산")

                bhhh_hess_func = self._create_bhhh_hessian_function(
                    measurement_model,
                    structural_model,
                    choice_model,
                    negative_log_likelihood,
                    gradient_function
                )

                self.iteration_logger.info(f"BHHH 옵션: xtol={optimizer_options['xtol']}")

                result = optimize.minimize(
                    early_stopping_wrapper.objective,
                    initial_params_scaled,
                    method='Newton-CG',  # Newton-CG는 custom hess 지원
                    jac=jac_function,
                    hess=bhhh_hess_func,  # ← BHHH Hessian 제공!
                    callback=early_stopping_wrapper.callback,
                    options=optimizer_options
                )

            elif self.config.estimation.optimizer == 'BFGS':
                optimizer_options = {
                    'maxiter': 200,  # Major iteration 최대 횟수
                    'ftol': 1e-3,    # 함수값 상대적 변화 0.1% 이하면 종료
                    'gtol': 1e-3,    # 그래디언트 norm 허용 오차
                    'c1': 1e-4,      # Armijo 조건 파라미터 (scipy 기본값)
                    'c2': 0.9,       # Curvature 조건 파라미터 (scipy 기본값)
                    'disp': True
                }
                self.iteration_logger.info(f"BFGS 옵션: c1={optimizer_options['c1']}, c2={optimizer_options['c2']} (scipy 기본값)")

                result = optimize.minimize(
                    early_stopping_wrapper.objective,  # Wrapper의 objective 사용
                    initial_params_scaled,  # 스케일된 초기 파라미터 사용
                    method='BFGS',
                    jac=jac_function,
                    callback=early_stopping_wrapper.callback,  # Callback 추가
                    options=optimizer_options
                )

            elif self.config.estimation.optimizer == 'L-BFGS-B':
                optimizer_options = {
                    'maxiter': 200,  # Major iteration 최대 횟수
                    'maxls': 20,     # Line search 최대 횟수 (기본값: 20)
                    'disp': True
                    # ftol, gtol을 설정하지 않음 → scipy 기본값 사용
                    # 기본값: ftol=2.220446049250313e-09, pgtol=1e-05
                }
                self.iteration_logger.info(
                    f"L-BFGS-B 옵션:\n"
                    f"  - maxiter: {optimizer_options['maxiter']}\n"
                    f"  - ftol: 기본값 (2.22e-09 * factr, factr=1e7)\n"
                    f"  - pgtol: 기본값 (1e-05)\n"
                    f"  - maxls: {optimizer_options['maxls']} (line search 최대 횟수)\n"
                    f"\n"
                    f"  ✅ 커스텀 수렴 조건 (callback에서 ftol AND gtol 모두 체크):\n"
                    f"    1. ftol 조건: (f^k - f^{{k+1}})/max{{|f^k|,|f^{{k+1}}|,1}} <= 1e-6\n"
                    f"    2. gtol 조건: max{{|proj g_i|}} <= 1e-5\n"
                    f"    → 두 조건을 모두 만족해야 조기 종료\n"
                    f"\n"
                    f"  💡 scipy의 기본 수렴 조건과 병행하여 사용합니다."
                )

                result = optimize.minimize(
                    early_stopping_wrapper.objective,  # Wrapper의 objective 사용
                    initial_params_scaled,  # 스케일된 초기 파라미터 사용
                    method='L-BFGS-B',
                    jac=jac_function,
                    bounds=bounds,
                    callback=early_stopping_wrapper.callback,  # Callback 추가
                    options=optimizer_options
                )

            else:
                optimizer_options = {
                    'maxiter': 200,
                    'disp': True
                }

                result = optimize.minimize(
                    early_stopping_wrapper.objective,  # Wrapper의 objective 사용
                    initial_params_scaled,  # 스케일된 초기 파라미터 사용
                    method=self.config.estimation.optimizer,
                    jac=jac_function,
                    callback=early_stopping_wrapper.callback,  # Callback 추가
                    options=optimizer_options
                )

            # 최적화 결과 로깅
            self.iteration_logger.info(f"최적화 종료: {result.message}")
            self.iteration_logger.info(f"  성공 여부: {result.success}")
            self.iteration_logger.info(f"  Major iterations: {major_iter_count[0]}")
            self.iteration_logger.info(f"  함수 호출: {result.nfev}회")

            # Line search 실패 경고
            if not result.success and 'ABNORMAL_TERMINATION_IN_LNSRCH' in result.message:
                self.iteration_logger.warning(
                    "\n⚠️  Line Search 실패로 종료되었습니다.\n"
                    "  가능한 원인:\n"
                    "    1. Gradient 계산 오류\n"
                    "    2. 함수가 너무 평평함 (flat region)\n"
                    "    3. 수치적 불안정성\n"
                    "  권장 조치:\n"
                    "    - maxls 값을 증가 (현재: 10)\n"
                    "    - ftol, gtol 값을 완화\n"
                    "    - 초기값 변경"
                )
                self.iteration_logger.warning(
                    "\n⚠️  Line Search 실패로 종료되었습니다.\n"
                    "  가능한 원인:\n"
                    "    1. Gradient 계산 오류\n"
                    "    2. 함수가 너무 평평함 (flat region)\n"
                    "    3. 수치적 불안정성\n"
                    "  권장 조치:\n"
                    "    - maxls 값을 증가 (현재: 10)\n"
                    "    - ftol, gtol 값을 완화\n"
                    "    - 초기값 변경"
                )

            # 조기 종료된 경우 최적 파라미터로 복원
            if early_stopping_wrapper.early_stopped:
                from scipy.optimize import OptimizeResult

                # Wrapper에 저장된 최적 파라미터로 result 객체 재생성
                result = OptimizeResult(
                    x=early_stopping_wrapper.best_x,
                    success=True,
                    message=f"Early stopping: {early_stopping_wrapper.patience}회 연속 개선 없음",
                    fun=early_stopping_wrapper.best_ll,
                    nit=early_stopping_wrapper.func_call_count,
                    nfev=early_stopping_wrapper.func_call_count,
                    njev=early_stopping_wrapper.grad_call_count,
                    hess_inv=None  # 나중에 설정
                )

            # Hessian 역행렬 처리
            if self.config.estimation.calculate_se:
                # BFGS의 hess_inv가 있으면 사용 (추가 계산 0회!)
                if hasattr(result, 'hess_inv') and result.hess_inv is not None:
                    self.iteration_logger.info("Hessian 역행렬: BFGS에서 자동 제공 (추가 계산 0회)")
                    self.iteration_logger.info("Hessian 역행렬: BFGS에서 자동 제공 (추가 계산 0회)")

                    # ✅ Hessian 역행렬 통계 로깅
                    hess_inv = result.hess_inv
                    if hasattr(hess_inv, 'todense'):
                        hess_inv_array = hess_inv.todense()
                    else:
                        hess_inv_array = hess_inv

                    # ✅ Hessian 역행렬을 result에 저장 (나중에 CSV로 저장)
                    self.hessian_inv_matrix = np.array(hess_inv_array)

                    # 대각 원소 (각 파라미터의 분산 근사)
                    diag_elements = np.diag(hess_inv_array)

                    # 비대각 원소 (파라미터 간 공분산)
                    off_diag_mask = ~np.eye(hess_inv_array.shape[0], dtype=bool)
                    off_diag_elements = hess_inv_array[off_diag_mask]

                    self.iteration_logger.info(
                        f"\n{'='*80}\n"
                        f"최종 Hessian 역행렬 (H^(-1)) 통계\n"
                        f"{'='*80}\n"
                        f"  Shape: {hess_inv_array.shape}\n"
                        f"  대각 원소 (분산 근사):\n"
                        f"    - 범위: [{np.min(diag_elements):.6e}, {np.max(diag_elements):.6e}]\n"
                        f"    - 평균: {np.mean(diag_elements):.6e}\n"
                        f"    - 중앙값: {np.median(diag_elements):.6e}\n"
                        f"    - 음수 개수: {np.sum(diag_elements < 0)}/{len(diag_elements)}\n"
                        f"  비대각 원소 (공분산):\n"
                        f"    - 범위: [{np.min(off_diag_elements):.6e}, {np.max(off_diag_elements):.6e}]\n"
                        f"    - 평균: {np.mean(off_diag_elements):.6e}\n"
                        f"    - 절대값 평균: {np.mean(np.abs(off_diag_elements)):.6e}\n"
                        f"\n  상위 10개 대각 원소 (파라미터 인덱스):\n"
                    )

                    # 상위 10개 대각 원소
                    top_10_indices = np.argsort(np.abs(diag_elements))[-10:][::-1]
                    for idx in top_10_indices:
                        # ✅ 측정모델 파라미터 고정 시 param_names는 최적화된 파라미터만 포함
                        param_name = param_names[idx] if idx < len(param_names) else f"param_{idx}"
                        self.iteration_logger.info(
                            f"    [{idx:2d}] {param_name:40s}: {diag_elements[idx]:+.6e}"
                        )

                    self.iteration_logger.info(f"{'='*80}\n")

                    # ✅ Hessian 역행렬은 별도 파일로 저장 (로그에는 출력하지 않음)
                    # (HESSIAN_ROW 로그 삭제 - 로그 파일 크기 절약)

                else:
                    # BFGS hess_inv가 없으면 BHHH 방법으로 계산 (L-BFGS-B의 경우)
                    self.iteration_logger.warning("Hessian 역행렬 없음 (L-BFGS-B는 hess_inv 제공 안 함)")
                    self.iteration_logger.warning("Hessian 역행렬 없음 (L-BFGS-B는 hess_inv 제공 안 함)")
                    self.iteration_logger.info("BHHH 방법으로 Hessian 계산 시작...")
                    self.iteration_logger.info("BHHH 방법으로 Hessian 계산 시작...")

                    try:
                        # BHHH 방법으로 Hessian 계산
                        hess_inv_bhhh = self._compute_bhhh_hessian_inverse(
                            result.x,
                            measurement_model,
                            structural_model,
                            choice_model
                        )

                        if hess_inv_bhhh is not None:
                            self.hessian_inv_matrix = hess_inv_bhhh
                            self.iteration_logger.info("BHHH Hessian 계산 성공")
                            self.iteration_logger.info("BHHH Hessian 계산 성공")

                            # BHHH Hessian 통계 로깅 (BFGS와 동일한 형식)
                            diag_elements = np.diag(hess_inv_bhhh)
                            off_diag_mask = ~np.eye(hess_inv_bhhh.shape[0], dtype=bool)
                            off_diag_elements = hess_inv_bhhh[off_diag_mask]

                            self.iteration_logger.info(
                                f"\n{'='*80}\n"
                                f"최종 Hessian 역행렬 (H^(-1)) - BHHH 방법\n"
                                f"{'='*80}\n"
                                f"  Shape: {hess_inv_bhhh.shape}\n"
                                f"  대각 원소 (분산 근사):\n"
                                f"    - 범위: [{np.min(diag_elements):.6e}, {np.max(diag_elements):.6e}]\n"
                                f"    - 평균: {np.mean(diag_elements):.6e}\n"
                                f"    - 중앙값: {np.median(diag_elements):.6e}\n"
                                f"    - 음수 개수: {np.sum(diag_elements < 0)}/{len(diag_elements)}\n"
                                f"  비대각 원소 (공분산):\n"
                                f"    - 범위: [{np.min(off_diag_elements):.6e}, {np.max(off_diag_elements):.6e}]\n"
                                f"    - 평균: {np.mean(off_diag_elements):.6e}\n"
                                f"    - 절대값 평균: {np.mean(np.abs(off_diag_elements)):.6e}\n"
                                f"\n  상위 10개 대각 원소 (파라미터 인덱스):\n"
                            )

                            # 상위 10개 대각 원소
                            top_10_indices = np.argsort(np.abs(diag_elements))[-10:][::-1]
                            for idx in top_10_indices:
                                param_name = self.param_names[idx] if hasattr(self, 'param_names') and idx < len(self.param_names) else f"param_{idx}"
                                self.iteration_logger.info(
                                    f"    [{idx:2d}] {param_name:40s}: {diag_elements[idx]:+.6e}"
                                )

                            self.iteration_logger.info(f"{'='*80}\n")
                        else:
                            self.iteration_logger.warning("BHHH Hessian 계산 실패")
                            self.iteration_logger.warning("BHHH Hessian 계산 실패")
                            self.hessian_inv_matrix = None

                    except Exception as e:
                        self.iteration_logger.error(f"BHHH Hessian 계산 중 오류: {e}")
                        self.iteration_logger.error(f"BHHH Hessian 계산 중 오류: {e}")
                        import traceback
                        self.iteration_logger.debug(traceback.format_exc())
                        self.hessian_inv_matrix = None
            else:
                self.hessian_inv_matrix = None

            # 최종 로그
            if early_stopping_wrapper.early_stopped:
                self.iteration_logger.info(f"조기 종료 완료: 함수 호출 {early_stopping_wrapper.func_call_count}회, LL={-early_stopping_wrapper.best_ll:.4f}")
                self.iteration_logger.info(f"조기 종료 완료: 함수 호출 {early_stopping_wrapper.func_call_count}회, LL={-early_stopping_wrapper.best_ll:.4f}")
            else:
                self.iteration_logger.info(f"정상 종료: 함수 호출 {early_stopping_wrapper.func_call_count}회")
                self.iteration_logger.info(f"정상 종료: 함수 호출 {early_stopping_wrapper.func_call_count}회")
        else:
            self.iteration_logger.info(f"최적화 시작: Nelder-Mead (gradient-free)")
            self.iteration_logger.info(f"최적화 시작: Nelder-Mead (gradient-free)")

            result = optimize.minimize(
                negative_log_likelihood,
                initial_params_scaled,  # 스케일된 초기 파라미터 사용
                method='Nelder-Mead',
                options={
                    'maxiter': self.config.estimation.max_iterations,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'disp': True
                }
            )

        if result.success:
            self.iteration_logger.info("최적화 성공")
            self.iteration_logger.info("최적화 성공")
        else:
            self.iteration_logger.warning(f"최적화 실패: {result.message}")
            self.iteration_logger.warning(f"최적화 실패: {result.message}")

        self.iteration_logger.info("=" * 70)
        self.iteration_logger.info(f"최종 로그우도: {-result.fun:.4f}")
        self.iteration_logger.info(f"반복 횟수: {iteration_count[0]}")
        self.iteration_logger.info("=" * 70)

        # 최적 파라미터 언스케일링 (Internal → External)
        # self.iteration_logger.info("")  # ✅ 빈 로그 비활성화
        self.iteration_logger.info("=" * 80)
        self.iteration_logger.info("최적 파라미터 변환 (Scaled → Full External)")
        self.iteration_logger.info("=" * 80)

        optimal_params_scaled = result.x

        # ✅ ParameterContext를 사용한 파라미터 변환 (한 줄로 간소화)
        optimal_params_full = param_context.to_full_external(optimal_params_scaled)
        optimal_params_opt = param_context.to_optimized_external(optimal_params_scaled)

        self.iteration_logger.info(
            f"✅ 파라미터 변환 완료:\n"
            f"  - 최적화 파라미터: {len(optimal_params_opt)}개\n"
            f"  - 전체 파라미터: {len(optimal_params_full)}개"
        )

        # 스케일링 비교 로깅 (최적화된 파라미터만)
        self.param_scaler.log_parameter_comparison(optimal_params_opt, optimal_params_scaled)

        # result.x를 전체 external parameters로 교체
        result.x = optimal_params_full

        # 결과 처리 (로거 종료 전에 수행)
        self.results = self._process_results(
            result, measurement_model, structural_model, choice_model
        )

        # 로거 종료 (결과 처리 후)
        self._close_iteration_logger()

        return self.results
    
    def _compute_individual_likelihood(self, ind_id, ind_data, ind_draws,
                                       param_dict, measurement_model,
                                       structural_model, choice_model) -> float:
        """
        개인별 우도 계산 (병렬화 가능)

        Args:
            ind_id: 개인 ID
            ind_data: 개인 데이터
            ind_draws: 개인의 Halton draws (n_draws, n_dimensions)
            param_dict: 파라미터 딕셔너리
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            개인의 로그우도
        """
        draw_lls = []

        # ✅ 차원 정보 추출
        n_exo = len(structural_model.exogenous_lvs)
        higher_order_lvs = structural_model.get_higher_order_lvs()
        n_higher_order = len(higher_order_lvs)

        # 🔍 디버깅: 첫 번째 개인의 첫 번째 draw만 로깅
        log_debug = (ind_id == ind_data[self.config.individual_id_column].iloc[0])

        for j, draw in enumerate(ind_draws):
            # ✅ draws 분리: 1차 LV + 2차+ LV
            if ind_draws.ndim == 1:
                # 1차원 (하위 호환)
                exo_draws = np.array([draw])
                higher_order_draws = {}

                if log_debug and j == 0:
                    self.iteration_logger.info(
                        f"[개인 {ind_id}, Draw #0] 1차원 draws (하위 호환 모드)\n"
                        f"  exo_draws: {exo_draws}\n"
                        f"  higher_order_draws: {higher_order_draws}"
                    )
            else:
                # 다차원
                exo_draws = draw[:n_exo]  # 처음 n_exo개: 1차 LV
                higher_order_draws_array = draw[n_exo:]  # 나머지: 2차+ LV

                # 딕셔너리로 변환
                higher_order_draws = {}
                for i, lv_name in enumerate(higher_order_lvs):
                    higher_order_draws[lv_name] = higher_order_draws_array[i]

                if log_debug and j == 0:
                    self.iteration_logger.info(
                        f"[개인 {ind_id}, Draw #0] 다차원 draws 분리\n"
                        f"  ind_draws.shape: {ind_draws.shape}\n"
                        f"  draw.shape: {draw.shape}\n"
                        f"  n_exo: {n_exo}, n_higher_order: {n_higher_order}\n"
                        f"  exo_draws ({len(exo_draws)}개): {exo_draws}\n"
                        f"  higher_order_draws ({len(higher_order_draws)}개): {higher_order_draws}"
                    )

            # ✅ 구조모델: LV = γ*X + η (올바른 인자 전달)
            # 🔍 디버깅: 첫 번째 draw에 로거 전달
            if log_debug and j == 0:
                structural_model._iteration_logger = self.iteration_logger

            lv = structural_model.predict(
                data=ind_data,
                exo_draws=exo_draws,
                params=param_dict['structural'],
                higher_order_draws=higher_order_draws
            )

            if log_debug and j == 0:
                self.iteration_logger.info(
                    f"[개인 {ind_id}, Draw #0] 예측된 잠재변수\n"
                    f"  lv: {lv}"
                )

                # 🔍 첫 번째 draw 이후 디버깅 플래그 비활성화
                if hasattr(structural_model, '_debug_predict'):
                    structural_model._debug_predict = False
                if hasattr(structural_model, '_debug_ll'):
                    structural_model._debug_ll = False

            # 측정모델 우도: P(Indicators|LV)
            ll_measurement = measurement_model.log_likelihood(
                ind_data, lv, param_dict['measurement']
            )

            # 🔍 디버깅: 첫 번째 draw에서 측정모델 파라미터 확인
            if log_debug and j == 0:
                first_lv = list(param_dict['measurement'].keys())[0]
                first_params = param_dict['measurement'][first_lv]
                self.iteration_logger.info(
                    f"[개인 {ind_id}, Draw #0] 측정모델 파라미터 (첫 번째 LV: {first_lv})\n"
                    f"  zeta (처음 3개): {first_params['zeta'][:3] if len(first_params['zeta']) >= 3 else first_params['zeta']}\n"
                    f"  sigma_sq (처음 3개): {first_params['sigma_sq'][:3] if len(first_params['sigma_sq']) >= 3 else first_params['sigma_sq']}"
                )

            # Panel Product: 개인의 여러 선택 상황에 대한 확률을 곱함
            choice_set_lls = []
            for idx in range(len(ind_data)):
                ll_choice_t = choice_model.log_likelihood(
                    ind_data.iloc[idx:idx+1],  # 각 선택 상황
                    lv,
                    param_dict['choice']
                )
                choice_set_lls.append(ll_choice_t)

            # Panel product: log(P1 * P2 * ... * PT) = log(P1) + log(P2) + ... + log(PT)
            ll_choice = sum(choice_set_lls)

            # ✅ 구조모델 우도: P(LV|X) - 정규분포 가정 (올바른 인자 전달)
            # 🔍 디버깅: 첫 번째 draw에 로거 전달
            if log_debug and j == 0:
                structural_model._iteration_logger = self.iteration_logger

            ll_structural = structural_model.log_likelihood(
                data=ind_data,
                latent_vars=lv,
                exo_draws=exo_draws,
                params=param_dict['structural'],
                higher_order_draws=higher_order_draws
            )

            # 결합 로그우도
            draw_ll = ll_measurement + ll_choice + ll_structural

            if log_debug and j == 0:
                self.iteration_logger.info(
                    f"[개인 {ind_id}, Draw #0] 우도 성분\n"
                    f"  ll_measurement: {ll_measurement:.4f}\n"
                    f"  ll_choice: {ll_choice:.4f}\n"
                    f"  ll_structural: {ll_structural:.4f}\n"
                    f"  draw_ll (합계): {draw_ll:.4f}\n"
                    f"\n"
                    f"  ⚠️ 우도 성분 비율:\n"
                    f"    측정모델: {ll_measurement:.1f} ({100*ll_measurement/draw_ll:.1f}%)\n"
                    f"    선택모델: {ll_choice:.1f} ({100*ll_choice/draw_ll:.1f}%)\n"
                    f"    구조모델: {ll_structural:.1f} ({100*ll_structural/draw_ll:.1f}%)"
                )

            # 🔴 수정: -inf를 매우 작은 값으로 대체 (연속성 확보 for gradient)
            if not np.isfinite(draw_ll):
                draw_ll = -1e10  # -inf 대신 매우 작은 값

            draw_lls.append(draw_ll)

        # 🔴 수정: logsumexp를 사용하여 평균 계산
        # log[(1/R) Σᵣ exp(ll_r)] = logsumexp(ll_r) - log(R)
        person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))

        return person_ll

    def _joint_log_likelihood(self, params: np.ndarray,
                             measurement_model,
                             structural_model,
                             choice_model) -> float:
        """
        결합 로그우도 계산

        시뮬레이션 기반:
        log L ≈ Σᵢ log[(1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)]
        """
        # 파라미터 분해
        param_dict = self._unpack_parameters(
            params, measurement_model, structural_model, choice_model
        )

        # 🔍 디버깅: 첫 번째 호출 시 파라미터 로깅
        if not hasattr(self, '_first_ll_logged'):
            self._first_ll_logged = True
            self.iteration_logger.info(
                f"\n{'='*70}\n"
                f"첫 번째 로그우도 계산\n"
                f"{'='*70}\n"
                f"  구조모델 파라미터: {param_dict['structural']}\n"
                f"  선택모델 파라미터 (일부): {list(param_dict['choice'].keys())[:5]}...\n"
                f"{'='*70}\n"
            )

        # 🔍 구조모델 디버깅 플래그 활성화 (매 iteration마다, 첫 번째 개인의 첫 번째 draw만)
        structural_model._debug_predict = True
        structural_model._debug_ll = True

        # 메모리 체크 (Halton draws 가져오기 전)
        if hasattr(self, 'memory_monitor') and hasattr(self, '_likelihood_call_count'):
            self.memory_monitor.log_memory_stats(f"Halton draws 가져오기 전 (우도 #{self._likelihood_call_count})")

        draws = self.halton_generator.get_draws()

        # 메모리 체크 (Halton draws 가져온 후)
        if hasattr(self, 'memory_monitor') and hasattr(self, '_likelihood_call_count'):
            self.memory_monitor.log_memory_stats(f"Halton draws 가져온 후 (우도 #{self._likelihood_call_count})")

        individual_ids = self.data[self.config.individual_id_column].unique()

        # 병렬처리 여부 확인
        use_parallel = getattr(self.config.estimation, 'use_parallel', False)

        if use_parallel:
            # 병렬처리 사용 (전역 함수 사용)
            n_cores = getattr(self.config.estimation, 'n_cores', None)
            if n_cores is None:
                n_cores = max(1, multiprocessing.cpu_count() - 1)

            # 설정 정보를 dict로 변환 (pickle 가능)
            config_dict = {
                'measurement': {
                    'latent_variable': self.config.measurement.latent_variable,
                    'indicators': self.config.measurement.indicators,
                    'n_categories': self.config.measurement.n_categories
                },
                'structural': {
                    'sociodemographics': self.config.structural.sociodemographics,
                    'error_variance': self.config.structural.error_variance
                },
                'choice': {
                    'choice_attributes': self.config.choice.choice_attributes
                }
            }

            # 개인별 데이터 준비 (dict 형태로 변환)
            args_list = []
            for i, ind_id in enumerate(individual_ids):
                ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                ind_data_dict = ind_data.to_dict('list')  # pickle 가능한 dict로 변환
                ind_draws = draws[i, :]
                args_list.append((ind_data_dict, ind_draws, param_dict, config_dict))

            # 병렬 계산
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                person_lls = list(executor.map(_compute_individual_likelihood_parallel, args_list))

            total_ll = sum(person_lls)
        else:
            # 순차처리
            total_ll = 0.0
            for i, ind_id in enumerate(individual_ids):
                ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                ind_draws = draws[i, :]

                person_ll = self._compute_individual_likelihood(
                    ind_id, ind_data, ind_draws, param_dict,
                    measurement_model, structural_model, choice_model
                )
                total_ll += person_ll

        return total_ll

    def _get_parameter_bounds(self, measurement_model,
                              structural_model, choice_model,
                              exclude_measurement: bool = False) -> list:
        """
        Parameter bounds for L-BFGS-B

        ✅ ParameterManager에 위임 (단일 진실 공급원)
        ✅ Optimizer 종류와 무관하게 동일한 로직 사용
        ✅ 파라미터 구조 변경 시 ParameterManager만 수정
        ✅ exclude_measurement=True이면 측정모델 파라미터 제외

        Args:
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            exclude_measurement: True이면 측정모델 파라미터 제외

        Returns:
            bounds: [(lower, upper), ...] list
        """
        # ✅ ParameterManager에 완전히 위임
        return self.param_manager.get_parameter_bounds(
            measurement_model, structural_model, choice_model,
            exclude_measurement=exclude_measurement
        )

    # ❌ 제거됨: _get_parameter_names
    # ✅ ParameterManager.get_parameter_names()를 사용하도록 변경
    # (중복 로직 제거, 단일 진실 공급원)

    def _get_custom_scales(self, param_names: List[str]) -> Dict[str, float]:
        """
        Custom scale 값 생성 (gradient 균형 최적화)

        목표: 모든 internal gradient를 50~1,000 범위로

        Args:
            param_names: 파라미터 이름 리스트

        Returns:
            custom_scales: 파라미터 이름 → scale 값 매핑
        """
        custom_scales = {}

        for name in param_names:
            # zeta (factor loading) 스케일
            if name.startswith('zeta_'):
                if 'health_concern' in name:
                    custom_scales[name] = 0.024
                elif 'perceived_benefit' in name:
                    custom_scales[name] = 0.050
                elif 'perceived_price' in name:
                    custom_scales[name] = 0.120
                elif 'nutrition_knowledge' in name:
                    custom_scales[name] = 0.022
                elif 'purchase_intention' in name:
                    custom_scales[name] = 0.083
                else:
                    custom_scales[name] = 0.05  # 기본값

            # sigma_sq (error variance) 스케일
            elif name.startswith('sigma_sq_'):
                if 'health_concern' in name:
                    custom_scales[name] = 0.034
                elif 'perceived_benefit' in name:
                    custom_scales[name] = 0.036
                elif 'perceived_price' in name:
                    custom_scales[name] = 0.023
                elif 'nutrition_knowledge' in name:
                    custom_scales[name] = 0.046
                elif 'purchase_intention' in name:
                    custom_scales[name] = 0.026
                else:
                    custom_scales[name] = 0.03  # 기본값

            # beta (choice model coefficients) 스케일
            elif name.startswith('beta_'):
                if name == 'beta_intercept':
                    custom_scales[name] = 0.290
                elif name == 'beta_sugar_free':
                    custom_scales[name] = 0.230
                elif name == 'beta_health_label':
                    custom_scales[name] = 0.220
                elif name == 'beta_price':
                    custom_scales[name] = 0.056
                else:
                    custom_scales[name] = 0.2  # 기본값

            # lambda (latent variable coefficients) 스케일
            elif name.startswith('lambda_'):
                # ✅ 모든 LV 주효과 지원
                if name == 'lambda_main':
                    custom_scales[name] = 0.890
                elif name == 'lambda_mod_perceived_price':
                    custom_scales[name] = 0.470
                elif name == 'lambda_mod_nutrition_knowledge':
                    custom_scales[name] = 1.200
                # 개별 LV lambda 스케일
                elif name == 'lambda_health_concern':
                    custom_scales[name] = 0.8
                elif name == 'lambda_perceived_benefit':
                    custom_scales[name] = 0.8
                elif name == 'lambda_perceived_price':
                    custom_scales[name] = 0.8
                elif name == 'lambda_nutrition_knowledge':
                    custom_scales[name] = 0.8
                elif name == 'lambda_purchase_intention':
                    custom_scales[name] = 0.8
                else:
                    custom_scales[name] = 0.5  # 기본값

            # gamma (structural model coefficients) 스케일
            elif name.startswith('gamma_'):
                custom_scales[name] = 0.5  # 기본값

            # tau (thresholds) 스케일
            elif name.startswith('tau_'):
                custom_scales[name] = 1.0  # 스케일링 안함

            # 기타
            else:
                custom_scales[name] = 1.0  # 스케일링 안함

        return custom_scales

    def _get_initial_parameters_simultaneous(self, measurement_model, structural_model,
                                            choice_model, user_initial_params: Dict = None) -> np.ndarray:
        """
        동시추정 전용 초기값 설정

        ✅ 측정모델 파라미터는 완전히 제외
        ✅ 구조모델 + 선택모델만 처리 (8개 파라미터)
        ✅ 간소화된 로직으로 파라미터 이름-값 매칭 보장

        Args:
            measurement_model: 측정모델 (파라미터 이름 생성에는 사용 안 함)
            structural_model: 구조모델
            choice_model: 선택모델
            user_initial_params: 사용자 정의 초기값 딕셔너리
                {'measurement': {...}, 'structural': {...}, 'choice': {...}}

        Returns:
            초기 파라미터 배열 (8개, 측정모델 제외)
        """
        # ✅ 최적화 파라미터 이름만 생성 (구조모델 + 선택모델)
        param_names_opt = self.param_manager.get_optimized_parameter_names(
            structural_model, choice_model
        )

        self.iteration_logger.info(f"✅ 동시추정 초기값 설정: {len(param_names_opt)}개 파라미터 (측정모델 제외)")
        self.iteration_logger.info(f"   파라미터 이름: {param_names_opt}")

        # ✅ 사용자 정의 초기값 필수 검증
        if user_initial_params is None:
            raise ValueError(
                "동시추정은 초기값이 필수입니다!\n"
                "initial_params 딕셔너리를 제공해야 합니다.\n"
                "예: initial_params = {'structural': {...}, 'choice': {...}}\n"
                "측정모델 파라미터는 measurement_model 객체에 이미 로드되어 있어야 합니다."
            )

        if not isinstance(user_initial_params, dict):
            raise TypeError(
                f"initial_params는 딕셔너리여야 합니다. 현재 타입: {type(user_initial_params)}"
            )

        self.iteration_logger.info("사용자 정의 초기값 사용")
        self.iteration_logger.info(f"   제공된 키: {list(user_initial_params.keys())}")

        # 구조모델 + 선택모델만 추출
        opt_dict = {
            'structural': user_initial_params.get('structural', {}),
            'choice': user_initial_params.get('choice', {})
        }

        self.iteration_logger.info(f"   구조모델 파라미터: {list(opt_dict['structural'].keys())}")
        self.iteration_logger.info(f"   선택모델 파라미터: {list(opt_dict['choice'].keys())}")

        # ✅ 동시추정 전용 변환 함수 사용
        initial_values = self.param_manager.dict_to_array_optimized(
            opt_dict, param_names_opt, structural_model, choice_model
        )

        self.iteration_logger.info(f"✅ 초기값 변환 완료: {len(initial_values)}개")

        # ✅ 초기값 검증: 이름-값 매칭 확인
        self.iteration_logger.info("=" * 80)
        self.iteration_logger.info("초기값 검증: 파라미터 이름-값 매칭 확인")
        self.iteration_logger.info("=" * 80)

        mismatch_found = False
        for i, (name, value) in enumerate(zip(param_names_opt, initial_values)):
            # 딕셔너리에서 직접 값 추출
            if name.startswith('gamma_') and '_to_' in name:
                expected_value = opt_dict['structural'].get(name, None)
                source = 'structural'
            else:
                expected_value = opt_dict['choice'].get(name, None)
                source = 'choice'

            # 매칭 확인
            if expected_value is not None:
                if abs(value - expected_value) > 1e-6:
                    self.iteration_logger.error(
                        f"[{i:2d}] {name:50s} = {value:10.6f} (MISMATCH! Expected: {expected_value:10.6f}, Source: {source})"
                    )
                    mismatch_found = True
                else:
                    self.iteration_logger.info(f"[{i:2d}] {name:50s} = {value:10.6f} ✓")
            else:
                self.iteration_logger.warning(f"[{i:2d}] {name:50s} = {value:10.6f} (NOT FOUND in {source}, using default)")

        if mismatch_found:
            self.iteration_logger.error("=" * 80)
            self.iteration_logger.error("파라미터 이름-값 매칭 오류 발견!")
            self.iteration_logger.error("=" * 80)
            raise ValueError(
                "파라미터 이름-값 매칭 오류!\n"
                "위 로그에서 MISMATCH 표시된 파라미터를 확인하세요."
            )

        self.iteration_logger.info("=" * 80)
        self.iteration_logger.info("✅ 모든 파라미터 이름-값 매칭 검증 완료")
        self.iteration_logger.info("=" * 80)

        return initial_values

    def _get_initial_parameters(self, measurement_model,
                                structural_model, choice_model) -> np.ndarray:
        """
        초기 파라미터 생성 (ParameterManager 사용)

        사용자 정의 초기값(self.user_initial_params)이 있으면 사용하고,
        없으면 자동 생성합니다.

        Returns:
            초기 파라미터 벡터
        """
        # ✅ 전체 파라미터 이름 리스트 생성 (초기값 생성용)
        # 🔍 디버깅: choice_model 설정 확인
        self.iteration_logger.info(f"[DEBUG _get_initial_parameters] choice_model.all_lvs_as_main = {getattr(choice_model, 'all_lvs_as_main', None)}")
        self.iteration_logger.info(f"[DEBUG _get_initial_parameters] choice_model.main_lvs = {getattr(choice_model, 'main_lvs', None)}")

        param_names_full = self.param_manager.get_parameter_names(
            measurement_model, structural_model, choice_model
        )

        self.iteration_logger.info(f"전체 파라미터 이름 리스트 생성 완료: {len(param_names_full)}개")
        # 🔍 디버깅: 선택모델 파라미터 이름 확인
        choice_param_names = [name for name in param_names_full if name.startswith(('beta_', 'lambda_', 'gamma_')) and '_to_' not in name]
        self.iteration_logger.info(f"[DEBUG _get_initial_parameters] 선택모델 파라미터 이름: {choice_param_names}")

        # ✅ 사용자 정의 초기값 확인
        if hasattr(self, 'user_initial_params') and self.user_initial_params is not None:
            user_params = self.user_initial_params

            # 딕셔너리 형태인 경우 (순차추정 결과)
            if isinstance(user_params, dict):
                self.iteration_logger.info("사용자 정의 초기값 (딕셔너리) 사용")

                # ✅ 선택모델 파라미터 확인
                user_choice_params = user_params.get('choice', {})

                # 측정모델 + 구조모델 파라미터는 사용자 값 사용
                partial_dict = {
                    'measurement': user_params.get('measurement', {}),
                    'structural': user_params.get('structural', {}),
                    'choice': {}  # 일단 빈 딕셔너리로 초기화
                }

                # ✅ 선택모델 초기값: 사용자 제공 값이 있으면 사용, 없으면 자동 생성
                if user_choice_params and len(user_choice_params) > 0:
                    # 사용자가 선택모델 파라미터를 제공한 경우
                    self.iteration_logger.info(f"사용자 정의 선택모델 초기값 사용 ({len(user_choice_params)}개 파라미터)")
                    partial_dict['choice'] = user_choice_params
                else:
                    # 선택모델 초기값 자동 생성
                    self.iteration_logger.info("선택모델 초기값 자동 생성")
                    # MultinomialLogitChoice는 data 인자 필요
                    if hasattr(choice_model, 'get_initial_params'):
                        import inspect
                        sig = inspect.signature(choice_model.get_initial_params)
                        if 'data' in sig.parameters:
                            choice_initial = choice_model.get_initial_params(data=self.data)
                        else:
                            choice_initial = choice_model.get_initial_params()
                    else:
                        choice_initial = {}

                    partial_dict['choice'] = choice_initial

                # 딕셔너리 → 배열 변환
                initial_values = self.param_manager.dict_to_array(
                    partial_dict, param_names_full, measurement_model
                )

                self.iteration_logger.info(f"사용자 정의 초기값 변환 완료: {len(initial_values)}개")

            # 배열 형태인 경우 (이전 동시추정 결과)
            elif isinstance(user_params, np.ndarray):
                self.iteration_logger.info(f"사용자 정의 초기값 (배열) 사용: {len(user_params)}개")
                initial_values = user_params
            else:
                self.iteration_logger.warning(f"사용자 정의 초기값 형식 오류: {type(user_params)}")
                self.iteration_logger.warning("자동 초기화를 사용합니다.")
                initial_values = self.param_manager.get_initial_values(
                    param_names_full, measurement_model, structural_model, choice_model
                )
        else:
            # ✅ 자동 초기값 생성 (이름 기반)
            initial_values = self.param_manager.get_initial_values(
                param_names_full, measurement_model, structural_model, choice_model
            )

            self.iteration_logger.info(f"자동 초기 파라미터 생성 완료: {len(initial_values)}개")

        return initial_values



    # ❌ 중복 함수 제거됨 - 위의 _get_parameter_bounds() 사용
    def _unpack_parameters(self, params: np.ndarray,
                          measurement_model,
                          structural_model,
                          choice_model) -> Dict[str, Dict]:
        """
        파라미터 벡터를 딕셔너리로 변환 (동시추정 전용)

        ✅ params는 최적화 파라미터만 포함 (8개, 측정모델 제외)
        ✅ 측정모델 파라미터는 CFA 결과에서 자동으로 추가됨

        Args:
            params: 최적화 파라미터 배열 (8개, 측정모델 제외)
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체

        Returns:
            파라미터 딕셔너리 (측정모델 포함)
        """
        # ✅ 동시추정 전용 변환 함수 사용
        param_dict = self.param_manager.array_to_dict_optimized(
            params, self.param_names,
            measurement_model, structural_model, choice_model
        )

        return param_dict

    def _compute_gradient(self, params: np.ndarray,
                         measurement_model,
                         structural_model,
                         choice_model) -> np.ndarray:
        """
        순수한 analytic gradient 계산 (상태 의존성 제거)

        이 메서드는 단위테스트 및 gradient 검증을 위해 추출되었습니다.
        estimate() 내부의 gradient_function()과 동일한 로직을 사용합니다.

        Args:
            params: 파라미터 벡터 (unscaled, external)
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            gradient 벡터 (negative gradient for minimization)
        """
        # 파라미터 딕셔너리로 변환
        param_dict = self._unpack_parameters(
            params, measurement_model, structural_model, choice_model
        )

        # 🔍 디버깅: param_dict['choice'] 키 확인
        if 'choice' in param_dict:
            self.iteration_logger.info(f"[DEBUG _compute_gradient] param_dict['choice'] 키: {list(param_dict['choice'].keys())}")

        # 병렬처리 설정 가져오기
        use_parallel = getattr(self.config.estimation, 'use_parallel', False)
        n_cores = getattr(self.config.estimation, 'n_cores', None)

        # 다중 잠재변수 여부 확인
        from .multi_latent_config import MultiLatentConfig
        is_multi_latent = isinstance(self.config, MultiLatentConfig)

        if is_multi_latent:
            # ✅ GPU 상태 객체 생성
            gpu_state = GPUComputeState.from_joint_gradient(
                self.joint_grad,
                getattr(self, 'gpu_measurement_model', None)
            )

            # ✅ 통합 로깅
            self._log_gpu_status(gpu_state)

            # 개인 데이터 준비
            individual_ids = self.data[self.config.individual_id_column].unique()
            self.iteration_logger.info(f"처리할 개인 수: {len(individual_ids)}")

            all_ind_data = []
            all_ind_draws = []

            for ind_id in individual_ids:
                ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                ind_idx = np.where(individual_ids == ind_id)[0][0]
                ind_draws = self.halton_generator.get_draws()[ind_idx]

                all_ind_data.append(ind_data)
                all_ind_draws.append(ind_draws)

            # NumPy 배열로 변환
            all_ind_draws = np.array(all_ind_draws)  # (N, n_draws, n_dims)

            # 🎯 단일 진입점으로 gradient 계산
            all_grad_dicts = self.joint_grad.compute_gradients(
                all_ind_data=all_ind_data,
                all_ind_draws=all_ind_draws,
                params_dict=param_dict,
                measurement_model=measurement_model,
                structural_model=structural_model,
                choice_model=choice_model,
                iteration_logger=self.iteration_logger,
                log_level='MINIMAL'
            )

            # 모든 개인의 gradient 합산
            total_grad_dict = None
            for ind_grad in all_grad_dicts:
                if total_grad_dict is None:
                    import copy
                    total_grad_dict = copy.deepcopy(ind_grad)
                else:
                    # 재귀적으로 합산
                    def add_gradients(total, ind):
                        for key in total:
                            if isinstance(total[key], dict):
                                add_gradients(total[key], ind[key])
                            elif isinstance(total[key], np.ndarray):
                                total[key] += ind[key]
                            else:
                                total[key] += ind[key]

                    add_gradients(total_grad_dict, ind_grad)

            grad_dict = total_grad_dict
        else:
            # 단일 잠재변수: compute_gradient 사용
            grad_dict = self.joint_grad.compute_gradient(
                data=self.data,
                params_dict=param_dict,
                draws=self.halton_generator.get_draws(),
                individual_id_column=self.config.individual_id_column,
                measurement_model=measurement_model,
                structural_model=structural_model,
                choice_model=choice_model,
                indicators=self.config.measurement.indicators,
                sociodemographics=self.config.structural.sociodemographics,
                choice_attributes=self.config.choice.choice_attributes,
                use_parallel=use_parallel,
                n_cores=n_cores
            )

        # 그래디언트 벡터로 변환 (파라미터 순서와 동일)
        grad_vector = self._pack_gradient(grad_dict, measurement_model, structural_model, choice_model)

        # Negative gradient (minimize -LL)
        return -grad_vector

    def _pack_gradient(self, grad_dict: Dict, measurement_model,
                      structural_model, choice_model) -> np.ndarray:
        """
        그래디언트 딕셔너리를 벡터로 변환 (ParameterManager 사용)

        Args:
            grad_dict: 그래디언트 딕셔너리 (GPU에서 직접 반환, grad_ 접두사 없음)
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            gradient_vector: 그래디언트 벡터 (최적화 파라미터만, 측정모델 제외)
        """
        # ✅ 동시추정: 측정모델 그래디언트 제거 (고정 파라미터)
        grad_dict_opt = {
            'structural': grad_dict.get('structural', {}),
            'choice': grad_dict.get('choice', {})
        }

        # ✅ Gradient 딕셔너리 검증 (최적화 파라미터만)
        self._validate_gradient_dict(grad_dict_opt, self.param_names, measurement_model)

        # ✅ ParameterManager를 사용하여 배열로 변환 (최적화 파라미터만)
        gradient_vector = self.param_manager.dict_to_array(
            grad_dict_opt, self.param_names, measurement_model
        )

        return gradient_vector



    def _validate_gradient_dict(self, grad_dict: Dict, param_names: list, measurement_model) -> None:
        """
        Gradient 딕셔너리가 모든 필요한 파라미터를 포함하는지 검증

        Args:
            grad_dict: Gradient 딕셔너리 (parameter 스타일로 변환된 후)
            param_names: 필요한 파라미터 이름 리스트

        Raises:
            ValueError: 필요한 파라미터가 누락된 경우
        """
        missing_params = []

        for name in param_names:
            # 측정모델 파라미터
            if name.startswith('zeta_'):
                # ✅ indicator 이름 파싱 (예: zeta_health_concern_q7)
                parts = name.split('_')
                lv_name = '_'.join(parts[1:-1])  # 'health_concern'

                # measurement[lv_name]['zeta'] 배열이 있는지 확인
                found = False
                if 'measurement' in grad_dict and lv_name in grad_dict['measurement']:
                    if isinstance(grad_dict['measurement'][lv_name], dict):
                        if 'zeta' in grad_dict['measurement'][lv_name]:
                            found = True
                if not found:
                    missing_params.append(name)

            elif name.startswith('sigma_sq_'):
                # ✅ indicator 이름 파싱 (예: sigma_sq_health_concern_q7)
                parts = name.split('_')
                lv_name = '_'.join(parts[2:-1])  # 'sigma_sq' 제외

                # measurement[lv_name]['sigma_sq'] 배열이 있는지 확인
                found = False
                if 'measurement' in grad_dict and lv_name in grad_dict['measurement']:
                    if isinstance(grad_dict['measurement'][lv_name], dict):
                        if 'sigma_sq' in grad_dict['measurement'][lv_name]:
                            found = True
                if not found:
                    missing_params.append(name)

            elif name.startswith('tau_'):
                # ✅ indicator 이름 파싱 (예: tau_health_concern_q7_1)
                parts = name.split('_')
                lv_name = '_'.join(parts[1:-2])  # 'tau' 제외, indicator와 tau_idx 제외

                # measurement[lv_name]['tau'] 배열이 있는지 확인
                found = False
                if 'measurement' in grad_dict and lv_name in grad_dict['measurement']:
                    if isinstance(grad_dict['measurement'][lv_name], dict):
                        if 'tau' in grad_dict['measurement'][lv_name]:
                            found = True
                if not found:
                    missing_params.append(name)

            # 구조모델 파라미터
            elif name.startswith('gamma_') and '_to_' in name:
                if 'structural' not in grad_dict or name not in grad_dict['structural']:
                    missing_params.append(name)

            # 선택모델 파라미터
            elif name == 'beta_intercept':
                # beta_intercept → 'intercept'
                if 'choice' not in grad_dict or 'intercept' not in grad_dict['choice']:
                    missing_params.append(name)

            elif name.startswith('beta_'):
                # beta_sugar_free, beta_health_label, beta_price → 'beta' 배열
                if 'choice' not in grad_dict or 'beta' not in grad_dict['choice']:
                    missing_params.append(name)

            elif name.startswith('lambda_'):
                if 'choice' not in grad_dict or name not in grad_dict['choice']:
                    missing_params.append(name)

            elif name.startswith('gamma_') and not '_to_' in name:
                # LV-Attribute 상호작용 파라미터
                if 'choice' not in grad_dict or name not in grad_dict['choice']:
                    missing_params.append(name)

        if missing_params:
            # 상세한 에러 메시지 생성
            error_msg = [
                "=" * 80,
                "Gradient 딕셔너리 검증 실패",
                "=" * 80,
                f"누락된 파라미터 ({len(missing_params)}개):",
            ]
            for param in missing_params[:10]:  # 최대 10개만 표시
                error_msg.append(f"  - {param}")
            if len(missing_params) > 10:
                error_msg.append(f"  ... 외 {len(missing_params) - 10}개")

            error_msg.append("")
            error_msg.append("사용 가능한 Gradient 키:")
            error_msg.append(f"  measurement: {list(grad_dict.get('measurement', {}).keys())[:5]}...")
            error_msg.append(f"  structural: {list(grad_dict.get('structural', {}).keys())}")
            error_msg.append(f"  choice: {list(grad_dict.get('choice', {}).keys())}")
            error_msg.append("=" * 80)

            raise ValueError("\n".join(error_msg))

    def _log_gpu_status(self, gpu_state: GPUComputeState, prefix: str = ""):
        """
        GPU 상태를 일관되게 로깅

        Args:
            gpu_state: GPU 계산 상태 객체
            prefix: 로그 메시지 접두사
        """
        separator = "=" * 80
        mode_msg = f"{prefix}Gradient 계산 모드: {gpu_state.get_mode_name()}"

        # 콘솔과 파일 모두에 기록
        self.iteration_logger.info(separator)
        self.iteration_logger.info(mode_msg)
        self.iteration_logger.info(separator)

        self.iteration_logger.info(separator)
        self.iteration_logger.info(mode_msg)
        self.iteration_logger.info(separator)

        # 상세 정보는 파일에만 기록
        status = gpu_state.get_status_dict()
        self.iteration_logger.info(f"{prefix}  enabled: {status['enabled']}")
        self.iteration_logger.info(f"{prefix}  measurement_model: {status['measurement_model_available']}")
        self.iteration_logger.info(f"{prefix}  full_parallel: {status['full_parallel']}")
        self.iteration_logger.info(f"{prefix}  is_ready: {status['is_ready']}")
        self.iteration_logger.info(separator)

    def _process_results(self, optimization_result,
                        measurement_model,
                        structural_model,
                        choice_model) -> Dict:
        """최적화 결과 처리"""
        
        param_dict = self._unpack_parameters(
            optimization_result.x,
            measurement_model,
            structural_model,
            choice_model
        )
        
        results = {
            'success': optimization_result.success,
            'message': optimization_result.message,
            'log_likelihood': -optimization_result.fun,
            'n_iterations': optimization_result.nit,
            'parameters': param_dict,
            'raw_params': optimization_result.x,
            
            # 모델 적합도
            'n_observations': len(self.data),
            'n_parameters': len(optimization_result.x),
        }
        
        # AIC, BIC 계산
        ll = results['log_likelihood']
        k = results['n_parameters']
        n = results['n_observations']
        
        results['aic'] = -2 * ll + 2 * k
        results['bic'] = -2 * ll + k * np.log(n)
        
        # 표준오차 계산 (Hessian 기반)
        if self.config.estimation.calculate_se:
            try:
                # BFGS는 hess_inv를 반환 (역 Hessian)
                # 표준오차 = sqrt(diag(H^-1))
                if hasattr(optimization_result, 'hess_inv'):
                    hess_inv = optimization_result.hess_inv
                    if hasattr(hess_inv, 'todense'):
                        hess_inv = hess_inv.todense()

                    # ✅ Hessian 역행렬을 결과에 저장
                    results['hessian_inv'] = np.array(hess_inv)

                    # 대각 원소 추출 (분산)
                    variances = np.diag(hess_inv)

                    # 음수 분산 처리 (수치 오류)
                    variances = np.maximum(variances, 1e-10)

                    se = np.sqrt(variances)
                    results['standard_errors'] = se

                    # t-통계량
                    results['t_statistics'] = optimization_result.x / se

                    # p-값 (양측 검정, 대표본이므로 정규분포 사용)
                    from scipy.stats import norm
                    results['p_values'] = 2 * (1 - norm.cdf(np.abs(results['t_statistics'])))

                    # 파라미터별로 구조화
                    self.iteration_logger.info("파라미터별 통계량 구조화 중...")
                    results['parameter_statistics'] = self._structure_statistics(
                        optimization_result.x, se,
                        results['t_statistics'], results['p_values'],
                        measurement_model, structural_model, choice_model
                    )
                    self.iteration_logger.info("파라미터별 통계량 구조화 완료")

                else:
                    self.iteration_logger.warning("Hessian 정보가 없어 표준오차를 계산할 수 없습니다.")
                    results['hessian_inv'] = None

            except Exception as e:
                self.iteration_logger.warning(f"표준오차 계산 실패: {e}")
                import traceback
                self.iteration_logger.debug(traceback.format_exc())
                results['hessian_inv'] = None
        else:
            results['hessian_inv'] = None

        # CSV 로그 파일 닫기
        if hasattr(self, 'csv_log_file') and self.csv_log_file:
            self.csv_log_file.close()
            self.iteration_logger.info(f"CSV 로그 파일 저장 완료: {self.csv_log_path}")

        return results

    def _structure_statistics(self, estimates, std_errors, t_stats, p_values,
                              measurement_model, structural_model, choice_model):
        """
        파라미터별 통계량을 구조화된 딕셔너리로 변환

        Args:
            estimates: 추정값 벡터
            std_errors: 표준오차 벡터
            t_stats: t-통계량 벡터
            p_values: p-value 벡터
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            구조화된 통계량 딕셔너리
            {
                'measurement': {'zeta': {...}, 'tau': {...}},
                'structural': {'gamma': {...}},
                'choice': {'intercept': {...}, 'beta': {...}, 'lambda': {...}}
            }
        """
        # 파라미터 언팩
        param_dict = self._unpack_parameters(
            estimates, measurement_model, structural_model, choice_model
        )

        # 동일한 방식으로 표준오차, t-통계량, p-value 언팩
        se_dict = self._unpack_parameters(
            std_errors, measurement_model, structural_model, choice_model
        )
        t_dict = self._unpack_parameters(
            t_stats, measurement_model, structural_model, choice_model
        )
        p_dict = self._unpack_parameters(
            p_values, measurement_model, structural_model, choice_model
        )

        # 구조화된 결과 생성
        structured = {
            'measurement': {},
            'structural': {},
            'choice': {}
        }

        # 측정모델
        if 'measurement' in param_dict:
            for key in param_dict['measurement']:
                structured['measurement'][key] = {
                    'estimate': param_dict['measurement'][key],
                    'std_error': se_dict['measurement'][key],
                    't_statistic': t_dict['measurement'][key],
                    'p_value': p_dict['measurement'][key]
                }

        # 구조모델
        if 'structural' in param_dict:
            for key in param_dict['structural']:
                structured['structural'][key] = {
                    'estimate': param_dict['structural'][key],
                    'std_error': se_dict['structural'][key],
                    't_statistic': t_dict['structural'][key],
                    'p_value': p_dict['structural'][key]
                }

        # 선택모델
        if 'choice' in param_dict:
            for key in param_dict['choice']:
                structured['choice'][key] = {
                    'estimate': param_dict['choice'][key],
                    'std_error': se_dict['choice'][key],
                    't_statistic': t_dict['choice'][key],
                    'p_value': p_dict['choice'][key]
                }

        return structured

    def _create_bhhh_hessian_function(
        self,
        measurement_model,
        structural_model,
        choice_model,
        negative_log_likelihood_func,
        gradient_func
    ):
        """
        BHHH Hessian 계산 함수 생성 (scipy.optimize.minimize의 hess 파라미터용)

        BHHH 방법:
        - Hessian을 직접 계산하지 않고 OPG (Outer Product of Gradients)로 대체
        - OPG = Σ_i (grad_i × grad_i^T)
        - 각 iteration마다 모든 개인의 gradient를 계산하여 OPG 생성

        Args:
            negative_log_likelihood_func: negative log-likelihood 함수
            gradient_func: gradient 함수

        Returns:
            callable: hess(x) -> np.ndarray (n_params, n_params)
        """
        from src.analysis.hybrid_choice_model.iclv_models.bhhh_calculator import BHHHCalculator

        bhhh_calc = BHHHCalculator(logger=self.iteration_logger)
        hess_call_count = [0]  # Hessian 호출 횟수 추적

        # ✅ Major iteration 추적을 위한 변수
        prev_x = [None]  # 이전 파라미터
        prev_ll = [None]  # 이전 LL
        major_iter_count = [0]  # Major iteration 카운터

        def bhhh_hessian(x):
            """
            현재 파라미터에서 BHHH Hessian 계산

            Args:
                x: 현재 파라미터 벡터 (scaled)

            Returns:
                BHHH Hessian (n_params, n_params)
            """
            import time
            hess_start_time = time.time()

            hess_call_count[0] += 1

            # ✅ Major iteration 판단: 파라미터가 변경되었으면 새로운 major iteration
            is_new_major_iter = False
            if prev_x[0] is None or not np.allclose(x, prev_x[0], rtol=1e-10):
                major_iter_count[0] += 1
                is_new_major_iter = True

                # ✅ Major Iteration 시작 로깅
                self.iteration_logger.info(
                    f"\n{'='*80}\n"
                    f"[Major Iteration #{major_iter_count[0]} 시작]\n"
                    f"{'='*80}"
                )

            self.iteration_logger.info(
                f"\n{'='*80}\n"
                f"BHHH Hessian 계산 #{hess_call_count[0]}\n"
                f"{'='*80}"
            )

            # 파라미터 언스케일링
            if self.param_scaler is not None:
                x_unscaled = self.param_scaler.unscale_parameters(x)
            else:
                x_unscaled = x

            # 파라미터 언팩
            param_dict = self._unpack_parameters(
                x_unscaled, measurement_model, structural_model, choice_model
            )

            # 개인별 gradient 계산
            self.iteration_logger.info("개인별 gradient 계산 시작...")
            individual_ids = self.data[self.config.individual_id_column].unique()
            n_individuals = len(individual_ids)

            # ✅ GPU batch 활용 여부 확인
            use_gpu = hasattr(self.joint_grad, 'use_gpu') and self.joint_grad.use_gpu

            # ✅ 완전 GPU Batch: 모든 개인을 동시에 처리
            if use_gpu and hasattr(self.joint_grad, 'compute_all_individuals_gradients_batch'):
                import time

                self.iteration_logger.info(
                    f"  ✅ 완전 GPU Batch 모드: {n_individuals}명 동시 처리"
                )

                # 모든 개인의 데이터와 draws 준비
                prep_start = time.time()
                all_ind_data = []
                all_ind_draws = []

                for ind_id in individual_ids:
                    ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                    ind_idx = np.where(individual_ids == ind_id)[0][0]
                    ind_draws = self.halton_generator.get_draws()[ind_idx]

                    all_ind_data.append(ind_data)
                    all_ind_draws.append(ind_draws)

                # NumPy 배열로 변환
                all_ind_draws = np.array(all_ind_draws)  # (N, n_draws, n_dims)
                prep_time = time.time() - prep_start

                self.iteration_logger.info(
                    f"  데이터 준비 완료 ({prep_time:.3f}초): "
                    f"all_ind_draws shape = {all_ind_draws.shape}"
                )

                # ✅ 완전 GPU Batch로 모든 개인의 gradient 동시 계산
                # 🚀 326명 × 100 draws × 80 params = 2,608,000개 동시 계산
                gpu_start = time.time()

                # 완전 GPU Batch 사용 (hasattr로 확인)
                if hasattr(self.joint_grad, 'compute_all_individuals_gradients_full_batch'):
                    all_grad_dicts = self.joint_grad.compute_all_individuals_gradients_full_batch(
                        all_ind_data=all_ind_data,
                        all_ind_draws=all_ind_draws,
                        params_dict=param_dict,
                        measurement_model=measurement_model,
                        structural_model=structural_model,
                        choice_model=choice_model,
                        iteration_logger=self.iteration_logger,
                        log_level='MODERATE' if hess_call_count[0] <= 2 else 'MINIMAL'
                    )
                else:
                    # 폴백: 일반 batch
                    all_grad_dicts = self.joint_grad.compute_all_individuals_gradients_batch(
                        all_ind_data=all_ind_data,
                        all_ind_draws=all_ind_draws,
                        params_dict=param_dict,
                        measurement_model=measurement_model,
                        structural_model=structural_model,
                        choice_model=choice_model,
                        iteration_logger=self.iteration_logger,
                        log_level='MODERATE' if hess_call_count[0] <= 2 else 'MINIMAL'
                    )

                gpu_time = time.time() - gpu_start

                self.iteration_logger.info(
                    f"  GPU Batch gradient 계산 완료 ({gpu_time:.3f}초)"
                )

                # Gradient 벡터로 변환 및 스케일링
                self.iteration_logger.info(f"  개인별 gradient 벡터 변환 시작 ({len(all_grad_dicts)}명)...")
                individual_gradients = []
                for i, ind_grad_dict in enumerate(all_grad_dicts):
                    grad_vector = self._pack_gradient(
                        ind_grad_dict,
                        measurement_model,
                        structural_model,
                        choice_model
                    )

                    if self.param_scaler is not None:
                        grad_vector = self.param_scaler.scale_gradient(grad_vector)

                    individual_gradients.append(grad_vector)

                    # 처음 3명만 상세 로깅
                    if i < 3:
                        self.iteration_logger.info(
                            f"  개인 {i} (ID={individual_ids[i]}): gradient norm = {np.linalg.norm(grad_vector):.6e}"
                        )

                self.iteration_logger.info(
                    f"✅ 완전 GPU Batch gradient 계산 완료: {n_individuals}명"
                )

                # Gradient 통계 로깅
                grad_norms = [np.linalg.norm(g) for g in individual_gradients]
                self.iteration_logger.info(
                    f"  Gradient norm 통계: min={min(grad_norms):.6e}, "
                    f"max={max(grad_norms):.6e}, mean={np.mean(grad_norms):.6e}"
                )

            else:
                # 기존 방식: 개인별 순차 처리 (각 개인 내부는 GPU batch)
                if use_gpu:
                    self.iteration_logger.info("  GPU batch 모드로 개인별 gradient 계산 (순차)")
                else:
                    self.iteration_logger.info("  CPU 모드로 개인별 gradient 계산")

                individual_gradients = []
                for i, ind_id in enumerate(individual_ids):
                    # 개인 데이터 및 draws 가져오기
                    ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                    ind_idx = np.where(individual_ids == ind_id)[0][0]
                    ind_draws = self.halton_generator.get_draws()[ind_idx]

                    # 개인별 gradient 계산
                    ind_grad_dict = self.joint_grad.compute_individual_gradient(
                        ind_data=ind_data,
                        ind_draws=ind_draws,
                        params_dict=param_dict,
                        measurement_model=measurement_model,
                        structural_model=structural_model,
                        choice_model=choice_model,
                        ind_id=ind_id
                    )

                    # Gradient 벡터로 변환
                    grad_vector = self._pack_gradient(
                        ind_grad_dict,
                        measurement_model,
                        structural_model,
                        choice_model
                    )

                    # 스케일링 적용
                    if self.param_scaler is not None:
                        grad_vector = self.param_scaler.scale_gradient(grad_vector)

                    individual_gradients.append(grad_vector)

                    # 처음 3명만 상세 로깅
                    if i < 3:
                        self.iteration_logger.info(
                            f"  개인 {i} (ID={ind_id}): gradient norm = {np.linalg.norm(grad_vector):.6e}"
                        )

                self.iteration_logger.info(
                    f"개인별 gradient 계산 완료: {n_individuals}명"
                )

            # BHHH Hessian 계산 (OPG)
            import time
            self.iteration_logger.info("OPG 행렬 계산 중...")
            opg_start = time.time()
            hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
                individual_gradients,
                for_minimization=True  # scipy는 최소화 문제
            )
            opg_time = time.time() - opg_start
            self.iteration_logger.info(f"OPG 계산 완료 ({opg_time:.3f}초)")

            hess_total_time = time.time() - hess_start_time

            self.iteration_logger.info(
                f"\n{'='*80}\n"
                f"BHHH Hessian 계산 완료 (총 {hess_total_time:.3f}초)\n"
                f"{'='*80}\n"
                f"  Shape: {hessian_bhhh.shape}\n"
                f"  시간 분석:\n"
                f"    - 데이터 준비: {prep_time if 'prep_time' in locals() else 0:.3f}초\n"
                f"    - GPU Batch gradient: {gpu_time if 'gpu_time' in locals() else 0:.3f}초\n"
                f"    - OPG 계산: {opg_time:.3f}초\n"
                f"  성능:\n"
                f"    - 개인당 시간: {hess_total_time / n_individuals * 1000:.2f}ms\n"
                f"    - 처리량: {n_individuals / hess_total_time:.1f} 개인/초\n"
                f"{'='*80}"
            )

            # ✅ Major Iteration 완료 로깅 및 CSV 저장
            if is_new_major_iter:
                # 현재 LL 계산
                x_unscaled = self.param_scaler.unscale_parameters(x) if self.param_scaler is not None else x
                current_ll = -negative_log_likelihood_func(x)  # objective는 -LL이므로 부호 반전

                # 파라미터 변화량 계산
                if prev_x[0] is not None:
                    param_change = np.linalg.norm(x - prev_x[0])
                    ll_change = current_ll - prev_ll[0] if prev_ll[0] is not None else 0.0
                else:
                    param_change = 0.0
                    ll_change = 0.0

                # Gradient 계산 (전체 gradient)
                grad = gradient_func(x)
                grad_norm = np.linalg.norm(grad)

                # 파라미터 및 gradient 상세 로깅
                params_external = self.param_scaler.unscale_parameters(x) if self.param_scaler is not None else x

                gradient_details = "\n  전체 파라미터 값 및 그래디언트:\n"
                for idx in range(len(params_external)):
                    param_name = self.param_names[idx] if hasattr(self, 'param_names') and idx < len(self.param_names) else f"param_{idx}"
                    gradient_details += f"    [{idx:2d}] {param_name:50s}: param={params_external[idx]:+12.6e}, grad={grad[idx]:+12.6e}\n"

                # CSV 파일에 기록
                if hasattr(self, '_log_params_grads_to_csv'):
                    self._log_params_grads_to_csv(major_iter_count[0], params_external, grad)

                # Major Iteration 완료 로깅
                self.iteration_logger.info(
                    f"\n{'='*80}\n"
                    f"[Major Iteration #{major_iter_count[0]} 완료]\n"
                    f"  최종 LL: {current_ll:.4f}\n"
                    f"  파라미터 변화량 (L2 norm): {param_change:.6e}\n"
                    f"  LL 변화: {ll_change:+.4f}\n"
                    f"  Gradient norm: {grad_norm:.6e}\n"
                    f"{gradient_details}"
                    f"  Hessian 근사: BHHH (OPG) 방법으로 계산 완료\n"
                    f"{'='*80}"
                )

                # 현재 상태 저장
                prev_x[0] = x.copy()
                prev_ll[0] = current_ll

            return hessian_bhhh

        return bhhh_hessian

    def _compute_bhhh_hessian_inverse(
        self,
        optimal_params: np.ndarray,
        measurement_model,
        structural_model,
        choice_model,
        max_individuals: int = 100,
        use_all_individuals: bool = False
    ) -> Optional[np.ndarray]:
        """
        BHHH 방법으로 Hessian 역행렬 계산

        Args:
            optimal_params: 최적 파라미터 벡터
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
            max_individuals: 최대 개인 수 (샘플링)
            use_all_individuals: True면 모든 개인 사용

        Returns:
            Hessian 역행렬 (n_params, n_params) 또는 None (실패 시)
        """
        try:
            # BHHH 계산기 초기화
            bhhh_calc = BHHHCalculator(logger=self.iteration_logger)

            # 파라미터 언팩
            param_dict = self._unpack_parameters(
                optimal_params, measurement_model, structural_model, choice_model
            )

            # 개인별 gradient 계산
            self.iteration_logger.info("개인별 gradient 계산 시작...")
            individual_gradients = []

            # 개인 ID 목록
            individual_ids = self.data[self.config.individual_id_column].unique()
            n_total_individuals = len(individual_ids)

            # 샘플링 여부 결정
            if use_all_individuals:
                n_individuals = n_total_individuals
                sampled_ids = individual_ids
            else:
                n_individuals = min(max_individuals, n_total_individuals)
                # 균등 샘플링
                step = max(1, n_total_individuals // n_individuals)
                sampled_ids = individual_ids[::step][:n_individuals]

            self.iteration_logger.info(
                f"BHHH 계산: {n_individuals}명 사용 "
                f"(전체 {n_total_individuals}명 중)"
            )

            # 다중 잠재변수 여부 확인
            from .multi_latent_config import MultiLatentConfig
            is_multi_latent = isinstance(self.config, MultiLatentConfig)

            if is_multi_latent:
                # 다중 잠재변수: compute_individual_gradient 사용
                from .multi_latent_gradient import MultiLatentJointGradient

                for i, ind_id in enumerate(sampled_ids):
                    if i % 10 == 0:
                        self.iteration_logger.info(f"  진행: {i}/{n_individuals}")

                    # 개인 데이터
                    ind_data = self.data[
                        self.data[self.config.individual_id_column] == ind_id
                    ]

                    # 개인 draws
                    ind_idx = np.where(individual_ids == ind_id)[0][0]
                    ind_draws = self.halton_generator.get_draws()[ind_idx]

                    # 개인별 gradient 계산
                    ind_grad_dict = self.joint_grad.compute_individual_gradient(
                        ind_data=ind_data,
                        ind_draws=ind_draws,
                        params_dict=param_dict,
                        measurement_model=measurement_model,
                        structural_model=structural_model,
                        choice_model=choice_model,
                        ind_id=ind_id
                    )

                    # Gradient를 벡터로 변환
                    grad_vector = self._pack_gradient(
                        ind_grad_dict,
                        measurement_model,
                        structural_model,
                        choice_model
                    )

                    # 처음 3명의 gradient 상세 로깅
                    if i < 3:
                        self.iteration_logger.info(
                            f"\n개인 {i} (ID={ind_id}) Gradient 벡터:\n"
                            f"  Shape: {grad_vector.shape}\n"
                            f"  Norm: {np.linalg.norm(grad_vector):.6e}\n"
                            f"  범위: [{np.min(grad_vector):.6e}, {np.max(grad_vector):.6e}]\n"
                            f"  처음 5개 값: {grad_vector[:5]}"
                        )

                    individual_gradients.append(grad_vector)

            else:
                # 단일 잠재변수 (기존 방식)
                self.iteration_logger.warning(
                    "단일 잠재변수 모델의 BHHH는 아직 구현되지 않았습니다."
                )
                return None

            self.iteration_logger.info(
                f"\n{'='*80}\n"
                f"개인별 gradient 계산 완료\n"
                f"{'='*80}\n"
                f"  총 개인 수: {len(individual_gradients)}명\n"
                f"  Gradient 벡터 길이: {len(individual_gradients[0])}개 파라미터\n"
                f"{'='*80}"
            )

            # BHHH Hessian 계산
            self.iteration_logger.info("\n" + "="*80)
            self.iteration_logger.info("BHHH Hessian 계산 시작 (OPG 방식)")
            self.iteration_logger.info("="*80)
            hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
                individual_gradients,
                for_minimization=True  # scipy.optimize.minimize는 최소화 문제
            )

            # Hessian 역행렬 계산
            self.iteration_logger.info("Hessian 역행렬 계산 중...")
            hess_inv = bhhh_calc.compute_hessian_inverse(
                hessian_bhhh,
                regularization=1e-8
            )

            # 표준오차 계산 (검증용)
            se = bhhh_calc.compute_standard_errors(hess_inv)
            self.iteration_logger.info(
                f"BHHH 표준오차 범위: "
                f"[{np.min(se):.6e}, {np.max(se):.6e}]"
            )

            return hess_inv

        except Exception as e:
            self.iteration_logger.error(f"BHHH Hessian 계산 실패: {e}")
            import traceback
            self.iteration_logger.debug(traceback.format_exc())
            return None


def estimate_iclv_simultaneous(data: pd.DataFrame, config,
                               measurement_model,
                               structural_model,
                               choice_model) -> Dict:
    """
    ICLV 모델 동시 추정 헬퍼 함수
    
    Args:
        data: 통합 데이터
        config: ICLVConfig
        measurement_model: 측정모델
        structural_model: 구조모델
        choice_model: 선택모델
    
    Returns:
        추정 결과
    """
    estimator = SimultaneousEstimator(config)
    return estimator.estimate(data, measurement_model, structural_model, choice_model)

