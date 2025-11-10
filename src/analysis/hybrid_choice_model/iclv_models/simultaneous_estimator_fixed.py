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
    
    def __init__(self, n_draws: int, n_individuals: int, 
                 scramble: bool = True, seed: Optional[int] = None):
        """
        Args:
            n_draws: 개인당 draw 수
            n_individuals: 개인 수
            scramble: 스크램블 여부 (권장)
            seed: 난수 시드
        """
        self.n_draws = n_draws
        self.n_individuals = n_individuals
        self.scramble = scramble
        self.seed = seed
        
        self.draws = None
        self._generate_draws()
    
    def _generate_draws(self):
        """Halton 시퀀스 생성"""
        logger.info(f"Halton draws 생성: {self.n_individuals} 개인 × {self.n_draws} draws")
        
        # scipy의 Halton 시퀀스 생성기 사용
        sampler = qmc.Halton(d=1, scramble=self.scramble, seed=self.seed)
        
        # 균등분포 [0,1] 샘플 생성
        uniform_draws = sampler.random(n=self.n_individuals * self.n_draws)
        
        # 표준정규분포로 변환 (역누적분포함수)
        normal_draws = norm.ppf(uniform_draws)
        
        # (n_individuals, n_draws) 형태로 재구성
        self.draws = normal_draws.reshape(self.n_individuals, self.n_draws)
        
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
                log_file: Optional[str] = None) -> Dict:
        """
        ICLV 모델 동시 추정

        Args:
            data: 통합 데이터
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            log_file: 로그 파일 경로 (None이면 자동 생성)

        Returns:
            추정 결과 딕셔너리
        """
        # 로그 파일 설정
        if log_file is None:
            from pathlib import Path
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            log_file = results_dir / 'iclv_estimation_log.txt'

        self._setup_iteration_logger(str(log_file))

        self.iteration_logger.info("SimultaneousEstimator.estimate() 시작")
        self.logger.info("ICLV 모델 동시 추정 시작")

        # 메모리 모니터의 logger를 iteration_logger로 업데이트
        if hasattr(self, 'memory_monitor') and self.memory_monitor is not None:
            self.memory_monitor.logger = self.iteration_logger

        self.data = data
        n_individuals = data[self.config.individual_id_column].nunique()

        self.iteration_logger.info(f"데이터 shape: {data.shape}")
        self.iteration_logger.info(f"개인 수: {n_individuals}")

        # Halton draws 생성 (이미 설정되어 있으면 건너뛰기)
        if not hasattr(self, 'halton_generator') or self.halton_generator is None:
            self.iteration_logger.info(f"Halton draws 생성 시작... (n_draws={self.config.estimation.n_draws}, n_individuals={n_individuals})")
            self.halton_generator = HaltonDrawGenerator(
                n_draws=self.config.estimation.n_draws,
                n_individuals=n_individuals,
                scramble=self.config.estimation.scramble_halton
            )
            self.iteration_logger.info("Halton draws 생성 완료")
        else:
            self.iteration_logger.info("Halton draws 이미 설정됨 (건너뛰기)")

        # Gradient calculators 초기화 (Apollo 방식)
        use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B']
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

                self.joint_grad = MultiLatentJointGradient(
                    self.measurement_grad,
                    self.structural_grad,
                    self.choice_grad,
                    use_gpu=use_gpu_gradient,
                    gpu_measurement_model=gpu_measurement_model
                )
                self.iteration_logger.info("다중 잠재변수 JointGradient 초기화 완료")
            else:
                # 단일 잠재변수
                self.joint_grad = JointGradient(
                    self.measurement_grad,
                    self.structural_grad,
                    self.choice_grad
                )
                self.iteration_logger.info("단일 잠재변수 JointGradient 초기화 완료")

        # 초기 파라미터 설정
        self.iteration_logger.info("초기 파라미터 설정 시작...")
        initial_params = self._get_initial_parameters(
            measurement_model, structural_model, choice_model
        )
        self.iteration_logger.info(f"초기 파라미터 설정 완료 (총 {len(initial_params)}개)")

        # 결합 우도함수 정의 (단계별 로깅 추가)
        iteration_count = [0]  # Mutable counter
        best_ll = [-np.inf]  # Track best log-likelihood
        func_call_count = [0]  # 함수 호출 횟수 (우도 계산)
        major_iter_count = [0]  # Major iteration 카운터
        line_search_call_count = [0]  # Line search 내 함수 호출 카운터
        last_major_iter_func_value = [None]  # 마지막 major iteration의 함수값
        current_major_iter_start_call = [0]  # 현재 major iteration 시작 시 함수 호출 번호
        line_search_func_values = []  # Line search 중 함수값 기록
        line_search_start_func_value = [None]  # Line search 시작 시 함수값
        line_search_start_params = [None]  # Line search 시작 시 파라미터
        line_search_gradient = [None]  # Line search 시작 시 gradient
        line_search_directional_derivative = [None]  # ∇f(x)^T·d (시작 시)

        def negative_log_likelihood(params):
            func_call_count[0] += 1

            # Line search 중인지 판단
            # Major iteration 시작 직후 첫 호출이 아니면 line search 중
            calls_since_major_start = func_call_count[0] - current_major_iter_start_call[0]

            if calls_since_major_start == 1:
                # Major iteration 시작 시 첫 함수 호출
                context = f"Major Iteration #{major_iter_count[0] + 1} 시작"
                line_search_call_count[0] = 0
                line_search_func_values.clear()
                line_search_start_params[0] = params.copy()
            elif calls_since_major_start > 1:
                # Line search 중
                line_search_call_count[0] += 1
                context = f"Line Search 함수 호출 #{line_search_call_count[0]}"
            else:
                # 초기 호출
                context = "초기 함수값 계산"

            # 단계 로그: 우도 계산 시작
            self.iteration_logger.info(f"\n[{context}] [단계 1/2] 전체 우도 계산")

            ll = self._joint_log_likelihood(
                params, measurement_model, structural_model, choice_model
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
                line_search_start_params[0] = params.copy()
            elif calls_since_major_start > 1:
                line_search_func_values.append(neg_ll)

                # 파라미터 변화량과 함수값 변화 로깅
                if line_search_start_params[0] is not None:
                    param_diff = params - line_search_start_params[0]
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

        # Get parameter bounds
        self.iteration_logger.info("파라미터 bounds 계산 시작...")
        bounds = self._get_parameter_bounds(
            measurement_model, structural_model, choice_model
        )
        self.iteration_logger.info(f"파라미터 bounds 계산 완료 (총 {len(bounds)}개)")

        # 최적화 방법 선택
        use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B']

        # Gradient 함수 정의 (Apollo 방식)
        grad_call_count = [0]  # 그래디언트 호출 횟수

        def gradient_function(params):
            """Analytic gradient 계산 (Apollo 방식)"""
            if not self.use_analytic_gradient:
                return None  # 수치적 그래디언트 사용

            grad_call_count[0] += 1

            # 단계 로그: 그래디언트 계산 시작 (모든 호출에서 출력)
            self.iteration_logger.info(f"\n[단계 2/2] Analytic Gradient 계산 #{grad_call_count[0]}")

            # 메모리 체크 (그래디언트 계산 전) - 5회마다 로깅
            if hasattr(self, 'memory_monitor'):
                # 5회마다 메모리 상태 로깅
                if grad_call_count[0] % 5 == 1:
                    self.memory_monitor.log_memory_stats(f"Gradient 계산 #{grad_call_count[0]}")

                # 항상 임계값 체크 및 필요시 정리
                mem_info = self.memory_monitor.check_and_cleanup(f"Gradient 계산 #{grad_call_count[0]}")

            # 파라미터 딕셔너리로 변환
            param_dict = self._unpack_parameters(
                params, measurement_model, structural_model, choice_model
            )

            # 병렬처리 설정 가져오기
            use_parallel = getattr(self.config.estimation, 'use_parallel', False)
            n_cores = getattr(self.config.estimation, 'n_cores', None)

            # 다중 잠재변수 여부 확인
            from .multi_latent_config import MultiLatentConfig
            is_multi_latent = isinstance(self.config, MultiLatentConfig)

            if is_multi_latent:
                # 다중 잠재변수: compute_individual_gradient 사용
                from .multi_latent_gradient import MultiLatentJointGradient

                # 개인별 그래디언트 계산 및 합산
                individual_ids = self.data[self.config.individual_id_column].unique()
                total_grad_dict = None

                for ind_id in individual_ids:
                    ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                    ind_idx = np.where(individual_ids == ind_id)[0][0]
                    ind_draws = self.halton_generator.get_draws()[ind_idx]

                    ind_grad = self.joint_grad.compute_individual_gradient(
                        ind_data=ind_data,
                        ind_draws=ind_draws,
                        params_dict=param_dict,
                        measurement_model=measurement_model,
                        structural_model=structural_model,
                        choice_model=choice_model
                    )

                    # 그래디언트 합산 (재귀적으로 처리)
                    if total_grad_dict is None:
                        # 첫 번째 개인: deep copy
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
            neg_grad = -grad_vector

            # Line search 중인지 판단
            calls_since_major_start = func_call_count[0] - current_major_iter_start_call[0]

            # Gradient 방향 검증 (첫 번째 호출 시)
            if grad_call_count[0] == 1:
                grad_norm = np.linalg.norm(neg_grad)
                self.iteration_logger.info(
                    f"\n[Gradient 방향 검증]\n"
                    f"  Gradient norm: {grad_norm:.6e}\n"
                    f"  Gradient (처음 5개): {neg_grad[:5]}\n"
                    f"  Gradient (마지막 5개): {neg_grad[-5:]}\n"
                    f"  주의: scipy는 이 gradient를 사용하여 descent direction을 계산합니다.\n"
                    f"       d = -H^(-1) · gradient이므로, gradient가 양수면 d는 음수 방향입니다."
                )

            # Line search 시작 시 방향 미분 저장
            if calls_since_major_start == 1:
                # Major iteration 시작 시 gradient 저장
                line_search_gradient[0] = neg_grad.copy()
                # 다음 함수 호출에서 탐색 방향을 알 수 있으므로, 방향 미분은 나중에 계산

            # Line search 중이면 Curvature 조건 계산
            elif calls_since_major_start > 1 and line_search_start_params[0] is not None:
                # 탐색 방향 계산: d = params - line_search_start_params
                search_direction = params - line_search_start_params[0]

                # 현재 위치에서 방향 미분: ∇f(x + α·d)^T·d
                directional_derivative_new = np.dot(neg_grad, search_direction)

                # Line search 시작 시 방향 미분 계산 (첫 line search 호출 시)
                if line_search_directional_derivative[0] is None and line_search_gradient[0] is not None:
                    # 시작 위치에서 방향 미분: ∇f(x)^T·d
                    line_search_directional_derivative[0] = np.dot(line_search_gradient[0], search_direction)

                # Curvature 조건 체크
                if line_search_directional_derivative[0] is not None:
                    dd_start = line_search_directional_derivative[0]
                    dd_new = directional_derivative_new

                    # Curvature 조건: |∇f(x + α·d)^T·d| ≤ c2·|∇f(x)^T·d|
                    c2 = 0.9  # scipy 기본값
                    curvature_lhs = abs(dd_new)
                    curvature_rhs = c2 * abs(dd_start)
                    curvature_satisfied = curvature_lhs <= curvature_rhs

                    self.iteration_logger.info(
                        f"\n[Curvature 조건 체크]\n"
                        f"  ∇f(x)^T·d (시작): {dd_start:.6e}\n"
                        f"  ∇f(x+α·d)^T·d (현재): {dd_new:.6e}\n"
                        f"  |∇f(x+α·d)^T·d|: {curvature_lhs:.6e}\n"
                        f"  c2·|∇f(x)^T·d|: {curvature_rhs:.6e}\n"
                        f"  Curvature 조건: {'✓ 만족' if curvature_satisfied else '❌ 불만족'}\n"
                        f"  → Gradient가 {'충분히 평평해짐' if curvature_satisfied else '아직 가파름'}"
                    )

            return neg_grad

        print("=" * 70, flush=True)
        if use_gradient:
            print(f"최적화 시작: {self.config.estimation.optimizer} (gradient-based)", flush=True)
            if self.use_analytic_gradient:
                print("Analytic gradient 사용 (Apollo 방식)", flush=True)
            else:
                print("수치적 그래디언트 사용 (2-point finite difference)", flush=True)
        else:
            print("최적화 시작: Nelder-Mead (gradient-free)", flush=True)
        print(f"초기 파라미터 개수: {len(initial_params)}", flush=True)
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

            def __init__(self, func, grad_func, patience=5, tol=1e-6, logger=None, iteration_logger=None):
                """
                Args:
                    func: 목적 함수 (negative log-likelihood)
                    grad_func: Gradient 함수
                    patience: 함수 호출 기준 개선 없는 횟수 (기본값: 5)
                    tol: LL 변화 허용 오차 (절대값)
                    logger: 메인 로거
                    iteration_logger: 반복 로거
                """
                self.func = func
                self.grad_func = grad_func
                self.patience = patience
                self.tol = tol
                self.logger = logger
                self.iteration_logger = iteration_logger

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
                    if self.logger:
                        self.logger.info(msg)
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
                """
                self.bfgs_iteration_count += 1
                major_iter_count[0] = self.bfgs_iteration_count

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
                            ls_status = f"✓ 성공 (함수값 감소: {f_decrease:.4f})"
                        elif f_decrease == 0:
                            ls_status = f"⚠️  정체 (함수값 변화 없음)"
                        else:
                            ls_status = f"❌ 실패 (함수값 증가: {-f_decrease:.4f})"
                    else:
                        ls_status = "N/A (첫 iteration)"

                    # ftol 계산 (이전 major iteration과 비교)
                    if last_major_iter_func_value[0] is not None:
                        f_prev = last_major_iter_func_value[0]
                        f_curr = current_f
                        rel_change = abs(f_prev - f_curr) / max(abs(f_prev), abs(f_curr), 1.0)
                        ftol_status = f"ftol = {rel_change:.6e} (기준: 1e-3)"
                        if rel_change <= 1e-3:
                            ftol_status += " ✓ 수렴 조건 만족"
                    else:
                        ftol_status = "ftol = N/A (첫 iteration)"

                    # Gradient norm 계산
                    if self.grad_func:
                        grad = self.grad_func(xk)
                        grad_norm = np.linalg.norm(grad, ord=np.inf)
                        gtol_status = f"gtol = {grad_norm:.6e} (기준: 1e-3)"
                        if grad_norm <= 1e-3:
                            gtol_status += " ✓ 수렴 조건 만족"
                    else:
                        gtol_status = "gtol = N/A"

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
                        f"  Hessian 근사: BFGS 공식으로 업데이트 완료\n"
                        f"{'='*80}"
                    )

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
            self.logger.info(f"최적화 시작: {self.config.estimation.optimizer} (gradient-based)")
            self.iteration_logger.info(f"최적화 시작: {self.config.estimation.optimizer} (gradient-based)")
            if self.use_analytic_gradient:
                self.logger.info("Analytic gradient 사용 (Apollo 방식)")
                self.iteration_logger.info("Analytic gradient 사용 (Apollo 방식)")
            else:
                self.logger.info("수치적 그래디언트 사용 (2-point finite difference)")
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
                iteration_logger=self.iteration_logger
            )

            # 초기 함수 호출 시작 위치 설정
            current_major_iter_start_call[0] = func_call_count[0]

            if use_early_stopping:
                self.logger.info(f"조기 종료 활성화: {early_stopping_patience}회 연속 함수 호출에서 LL 개선 없으면 종료 (tol={early_stopping_tol})")
                self.iteration_logger.info(f"조기 종료 활성화: {early_stopping_patience}회 연속 함수 호출에서 LL 개선 없으면 종료 (tol={early_stopping_tol})")
            else:
                self.logger.info("조기 종료 비활성화 (정상 종료만 사용)")
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

            result = optimize.minimize(
                early_stopping_wrapper.objective,  # Wrapper의 objective 사용
                initial_params,
                method=self.config.estimation.optimizer,
                jac=jac_function,
                bounds=bounds if self.config.estimation.optimizer == 'L-BFGS-B' else None,
                callback=early_stopping_wrapper.callback,  # Callback 추가
                options={
                    'maxiter': 200,  # Major iteration 최대 횟수
                    'ftol': 1e-3,    # 함수값 상대적 변화 0.1% 이하면 종료
                    'gtol': 1e-3,    # 그래디언트 norm 허용 오차
                    'maxls': 10,     # Line search 최대 횟수 (기본값: 20)
                    'disp': True
                }
            )

            # 최적화 결과 로깅
            self.logger.info(f"\n최적화 종료: {result.message}")
            self.iteration_logger.info(f"\n최적화 종료: {result.message}")
            self.logger.info(f"  성공 여부: {result.success}")
            self.iteration_logger.info(f"  성공 여부: {result.success}")
            self.logger.info(f"  Major iterations: {major_iter_count[0]}")
            self.iteration_logger.info(f"  Major iterations: {major_iter_count[0]}")
            self.logger.info(f"  함수 호출: {result.nfev}회")
            self.iteration_logger.info(f"  함수 호출: {result.nfev}회")

            # Line search 실패 경고
            if not result.success and 'ABNORMAL_TERMINATION_IN_LNSRCH' in result.message:
                self.logger.warning(
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
                    self.logger.info("Hessian 역행렬: BFGS에서 자동 제공 (추가 계산 0회)")
                    self.iteration_logger.info("Hessian 역행렬: BFGS에서 자동 제공 (추가 계산 0회)")
                else:
                    # BFGS hess_inv가 없으면 경고만 출력 (L-BFGS-B의 경우)
                    self.logger.warning("Hessian 역행렬 없음 (L-BFGS-B는 hess_inv 제공 안 함)")
                    self.iteration_logger.warning("Hessian 역행렬 없음 (L-BFGS-B는 hess_inv 제공 안 함)")
                    self.logger.info("표준오차 계산을 위해서는 BFGS 방법 사용 권장")
                    self.iteration_logger.info("표준오차 계산을 위해서는 BFGS 방법 사용 권장")

            # 최종 로그
            if early_stopping_wrapper.early_stopped:
                self.logger.info(f"조기 종료 완료: 함수 호출 {early_stopping_wrapper.func_call_count}회, LL={-early_stopping_wrapper.best_ll:.4f}")
                self.iteration_logger.info(f"조기 종료 완료: 함수 호출 {early_stopping_wrapper.func_call_count}회, LL={-early_stopping_wrapper.best_ll:.4f}")
            else:
                self.logger.info(f"정상 종료: 함수 호출 {early_stopping_wrapper.func_call_count}회")
                self.iteration_logger.info(f"정상 종료: 함수 호출 {early_stopping_wrapper.func_call_count}회")
        else:
            self.logger.info(f"최적화 시작: Nelder-Mead (gradient-free)")
            self.iteration_logger.info(f"최적화 시작: Nelder-Mead (gradient-free)")

            result = optimize.minimize(
                negative_log_likelihood,
                initial_params,
                method='Nelder-Mead',
                options={
                    'maxiter': self.config.estimation.max_iterations,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'disp': True
                }
            )

        if result.success:
            self.logger.info("최적화 성공")
            self.iteration_logger.info("최적화 성공")
        else:
            self.logger.warning(f"최적화 실패: {result.message}")
            self.iteration_logger.warning(f"최적화 실패: {result.message}")

        self.iteration_logger.info("=" * 70)
        self.iteration_logger.info(f"최종 로그우도: {-result.fun:.4f}")
        self.iteration_logger.info(f"반복 횟수: {iteration_count[0]}")
        self.iteration_logger.info("=" * 70)

        # 결과 처리
        self.results = self._process_results(
            result, measurement_model, structural_model, choice_model
        )

        # 로거 종료
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
            ind_draws: 개인의 Halton draws
            param_dict: 파라미터 딕셔너리
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            개인의 로그우도
        """
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
                    ind_data.iloc[idx:idx+1],  # 각 선택 상황
                    lv,
                    param_dict['choice']
                )
                choice_set_lls.append(ll_choice_t)

            # Panel product: log(P1 * P2 * ... * PT) = log(P1) + log(P2) + ... + log(PT)
            ll_choice = sum(choice_set_lls)

            # 구조모델 우도: P(LV|X) - 정규분포 가정
            ll_structural = structural_model.log_likelihood(
                ind_data, lv, param_dict['structural'], draw
            )

            # 결합 로그우도
            draw_ll = ll_measurement + ll_choice + ll_structural

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
                              structural_model, choice_model) -> list:
        """
        Parameter bounds for L-BFGS-B

        Returns:
            bounds: [(lower, upper), ...] list
        """
        bounds = []

        # Measurement model parameters
        # - Factor loadings (zeta): [0.1, 10]
        n_indicators = len(self.config.measurement.indicators)
        bounds.extend([(0.1, 10.0)] * n_indicators)

        # - Thresholds (tau): [-10, 10]
        n_thresholds = self.config.measurement.n_categories - 1
        for _ in range(n_indicators):
            bounds.extend([(-10.0, 10.0)] * n_thresholds)

        # Structural model parameters (gamma): unbounded
        n_sociodem = len(self.config.structural.sociodemographics)
        bounds.extend([(None, None)] * n_sociodem)

        # Choice model parameters
        # - Intercept: unbounded
        bounds.append((None, None))

        # - Attribute coefficients (beta): unbounded
        n_attributes = len(self.config.choice.choice_attributes)
        bounds.extend([(None, None)] * n_attributes)

        # - Latent variable coefficient (lambda): unbounded
        bounds.append((None, None))

        # - Sociodemographic coefficients: unbounded
        if self.config.structural.include_in_choice:
            bounds.extend([(None, None)] * n_sociodem)

        return bounds

    def _get_initial_parameters(self, measurement_model,
                                structural_model, choice_model) -> np.ndarray:
        """초기 파라미터 설정"""
        
        params = []
        
        # 측정모델 파라미터
        # - 요인적재량 (zeta)
        n_indicators = len(self.config.measurement.indicators)
        params.extend([1.0] * n_indicators)  # zeta
        
        # - 임계값 (tau)
        n_thresholds = self.config.measurement.n_categories - 1
        for _ in range(n_indicators):
            params.extend([-2, -1, 1, 2])  # 5점 척도 기본값
        
        # 구조모델 파라미터 (gamma)
        n_sociodem = len(self.config.structural.sociodemographics)
        params.extend([0.0] * n_sociodem)
        
        # 선택모델 파라미터
        # - 절편
        params.append(0.0)
        
        # - 속성 계수 (beta)
        n_attributes = len(self.config.choice.choice_attributes)
        params.extend([0.0] * n_attributes)
        
        # - 잠재변수 계수 (lambda)
        params.append(1.0)
        
        # - 사회인구학적 변수 계수 (선택모델에 포함되는 경우)
        if self.config.structural.include_in_choice:
            params.extend([0.0] * n_sociodem)
        
        return np.array(params)
    

    
    def _get_parameter_bounds(self, measurement_model,
                              structural_model, choice_model) -> list:
        """
        Parameter bounds for L-BFGS-B
        
        Returns:
            bounds: [(lower, upper), ...] list
        """
        bounds = []
        
        # Measurement model parameters
        # - Factor loadings (zeta): [0.1, 10]
        n_indicators = len(self.config.measurement.indicators)
        bounds.extend([(0.1, 10.0)] * n_indicators)
        
        # - Thresholds (tau): [-10, 10]
        n_thresholds = self.config.measurement.n_categories - 1
        for _ in range(n_indicators):
            bounds.extend([(-10.0, 10.0)] * n_thresholds)
        
        # Structural model parameters (gamma): unbounded
        n_sociodem = len(self.config.structural.sociodemographics)
        bounds.extend([(None, None)] * n_sociodem)
        
        # Choice model parameters
        # - Intercept: unbounded
        bounds.append((None, None))
        
        # - Attribute coefficients (beta): unbounded
        n_attributes = len(self.config.choice.choice_attributes)
        bounds.extend([(None, None)] * n_attributes)
        
        # - Latent variable coefficient (lambda): unbounded
        bounds.append((None, None))
        
        # - Sociodemographic coefficients: unbounded
        if self.config.structural.include_in_choice:
            bounds.extend([(None, None)] * n_sociodem)
        
        return bounds
    def _unpack_parameters(self, params: np.ndarray,
                          measurement_model,
                          structural_model,
                          choice_model) -> Dict[str, Dict]:
        """파라미터 벡터를 딕셔너리로 변환"""
        
        idx = 0
        param_dict = {
            'measurement': {},
            'structural': {},
            'choice': {}
        }
        
        # 측정모델 파라미터
        n_indicators = len(self.config.measurement.indicators)
        param_dict['measurement']['zeta'] = params[idx:idx+n_indicators]
        idx += n_indicators

        n_thresholds = self.config.measurement.n_categories - 1
        # tau를 2D 배열로 저장 (n_indicators, n_thresholds)
        tau_list = []
        for i in range(n_indicators):
            tau_list.append(params[idx:idx+n_thresholds])
            idx += n_thresholds
        param_dict['measurement']['tau'] = np.array(tau_list)
        
        # 구조모델 파라미터
        n_sociodem = len(self.config.structural.sociodemographics)
        param_dict['structural']['gamma'] = params[idx:idx+n_sociodem]
        idx += n_sociodem
        
        # 선택모델 파라미터
        param_dict['choice']['intercept'] = params[idx]
        idx += 1
        
        n_attributes = len(self.config.choice.choice_attributes)
        param_dict['choice']['beta'] = params[idx:idx+n_attributes]
        idx += n_attributes
        
        param_dict['choice']['lambda'] = params[idx]
        idx += 1
        
        if self.config.structural.include_in_choice:
            param_dict['choice']['beta_sociodem'] = params[idx:idx+n_sociodem]
            idx += n_sociodem
        
        return param_dict

    def _pack_gradient(self, grad_dict: Dict, measurement_model,
                      structural_model, choice_model) -> np.ndarray:
        """
        그래디언트 딕셔너리를 벡터로 변환 (파라미터 순서와 동일)

        Args:
            grad_dict: 그래디언트 딕셔너리
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            gradient_vector: 그래디언트 벡터
        """
        gradient_list = []

        # 다중 잠재변수 여부 확인
        from .multi_latent_config import MultiLatentConfig
        is_multi_latent = isinstance(self.config, MultiLatentConfig)

        if is_multi_latent:
            # 다중 잠재변수: 각 LV별로 그래디언트 추출
            for lv_name in measurement_model.models.keys():
                lv_grad = grad_dict['measurement'][lv_name]
                gradient_list.append(lv_grad['grad_zeta'])
                gradient_list.append(lv_grad['grad_tau'].flatten())

            # 구조모델 그래디언트
            gradient_list.append(grad_dict['structural']['grad_gamma_lv'])
            gradient_list.append(grad_dict['structural']['grad_gamma_x'])
        else:
            # 단일 잠재변수
            gradient_list.append(grad_dict['grad_zeta'])
            gradient_list.append(grad_dict['grad_tau'].flatten())
            gradient_list.append(grad_dict['grad_gamma'])

        # 선택모델 그래디언트 (공통)
        gradient_list.append(np.array([grad_dict['choice']['grad_intercept']]))
        gradient_list.append(grad_dict['choice']['grad_beta'])
        gradient_list.append(np.array([grad_dict['choice']['grad_lambda']]))

        # 사회인구학적 변수가 선택모델에 포함되는 경우
        if hasattr(self.config.structural, 'include_in_choice') and self.config.structural.include_in_choice:
            # 현재는 구현되지 않음
            n_sociodem = len(self.config.structural.sociodemographics)
            gradient_list.append(np.zeros(n_sociodem))

        # 벡터로 결합
        gradient_vector = np.concatenate(gradient_list)

        return gradient_vector

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
                    self.logger.info("파라미터별 통계량 구조화 중...")
                    results['parameter_statistics'] = self._structure_statistics(
                        optimization_result.x, se,
                        results['t_statistics'], results['p_values'],
                        measurement_model, structural_model, choice_model
                    )
                    self.logger.info("파라미터별 통계량 구조화 완료")

                else:
                    self.logger.warning("Hessian 정보가 없어 표준오차를 계산할 수 없습니다.")

            except Exception as e:
                self.logger.warning(f"표준오차 계산 실패: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

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

