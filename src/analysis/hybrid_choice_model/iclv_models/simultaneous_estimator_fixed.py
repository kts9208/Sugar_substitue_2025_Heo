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

        # 콘솔 핸들러도 추가 (선택적)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.iteration_logger.addHandler(console_handler)

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

        self.data = data
        n_individuals = data[self.config.individual_id_column].nunique()

        self.iteration_logger.info(f"데이터 shape: {data.shape}")
        self.iteration_logger.info(f"개인 수: {n_individuals}")
        self.logger.info(f"개인 수: {n_individuals}")

        # Halton draws 생성
        self.iteration_logger.info(f"Halton draws 생성 시작... (n_draws={self.config.estimation.n_draws}, n_individuals={n_individuals})")
        self.logger.info(f"Halton draws 생성 중... (n_draws={self.config.estimation.n_draws})")
        self.halton_generator = HaltonDrawGenerator(
            n_draws=self.config.estimation.n_draws,
            n_individuals=n_individuals,
            scramble=self.config.estimation.scramble_halton
        )
        self.iteration_logger.info("Halton draws 생성 완료")
        self.logger.info("Halton draws 생성 완료")

        # Gradient calculators 초기화 (Apollo 방식)
        use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B']
        if use_gradient and hasattr(self.config.estimation, 'use_analytic_gradient'):
            self.use_analytic_gradient = self.config.estimation.use_analytic_gradient
        else:
            self.use_analytic_gradient = False

        if self.use_analytic_gradient:
            self.iteration_logger.info("Analytic gradient calculators 초기화 (Apollo 방식)...")
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
            self.iteration_logger.info("Analytic gradient calculators 초기화 완료")

        # 초기 파라미터 설정
        self.iteration_logger.info("초기 파라미터 설정 시작...")
        self.logger.info("초기 파라미터 설정 중...")
        initial_params = self._get_initial_parameters(
            measurement_model, structural_model, choice_model
        )
        self.iteration_logger.info(f"초기 파라미터 설정 완료 (총 {len(initial_params)}개)")
        self.logger.info(f"초기 파라미터 설정 완료 (총 {len(initial_params)}개)")

        # 결합 우도함수 정의 (gradient check 로깅 추가)
        iteration_count = [0]  # Mutable counter
        best_ll = [-np.inf]  # Track best log-likelihood

        def negative_log_likelihood(params):
            iteration_count[0] += 1
            ll = self._joint_log_likelihood(
                params, measurement_model, structural_model, choice_model
            )

            # Track best value
            if ll > best_ll[0]:
                best_ll[0] = ll
                improvement = "[NEW BEST]"
            else:
                improvement = ""

            # Log every iteration with more detail
            # 개인 수에 따라 로깅 빈도 조정
            n_individuals = self.data[self.config.individual_id_column].nunique()
            log_interval = 10 if n_individuals > 100 else 5

            if iteration_count[0] % log_interval == 0 or improvement:
                log_msg = (
                    f"Iter {iteration_count[0]:4d}: LL = {ll:12.4f} "
                    f"(Best: {best_ll[0]:12.4f}) {improvement}"
                )
                self.iteration_logger.info(log_msg)

            return -ll

        # Get parameter bounds
        self.iteration_logger.info("파라미터 bounds 계산 시작...")
        bounds = self._get_parameter_bounds(
            measurement_model, structural_model, choice_model
        )
        self.iteration_logger.info(f"파라미터 bounds 계산 완료 (총 {len(bounds)}개)")

        # 최적화 방법 선택
        use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B']

        # Gradient 함수 정의 (Apollo 방식)
        def gradient_function(params):
            """Analytic gradient 계산 (Apollo 방식)"""
            if not self.use_analytic_gradient:
                return None  # 수치적 그래디언트 사용

            # 파라미터 딕셔너리로 변환
            param_dict = self._unpack_parameters(
                params, measurement_model, structural_model, choice_model
            )

            # 병렬처리 설정 가져오기
            use_parallel = getattr(self.config.estimation, 'use_parallel', False)
            n_cores = getattr(self.config.estimation, 'n_cores', None)

            # Analytic gradient 계산
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

        # 조기 종료를 위한 callback 클래스
        class EarlyStoppingCallback:
            def __init__(self, patience=10, tol=1e-6, logger=None, iteration_logger=None):
                """
                조기 종료 callback

                Args:
                    patience: LL 개선이 없는 연속 반복 횟수
                    tol: LL 변화 허용 오차 (절대값)
                """
                self.patience = patience
                self.tol = tol
                self.best_ll = np.inf
                self.no_improvement_count = 0
                self.iteration_count = 0
                self.logger = logger
                self.iteration_logger = iteration_logger

            def __call__(self, xk):
                """매 반복마다 호출되는 함수"""
                self.iteration_count += 1
                current_ll = negative_log_likelihood(xk)

                # LL 개선 체크
                if current_ll < self.best_ll - self.tol:
                    # 개선됨
                    self.best_ll = current_ll
                    self.no_improvement_count = 0
                else:
                    # 개선 없음
                    self.no_improvement_count += 1

                # 조기 종료 조건
                if self.no_improvement_count >= self.patience:
                    msg = f"조기 종료: {self.patience}회 연속 LL 개선 없음 (LL={-self.best_ll:.4f})"
                    if self.logger:
                        self.logger.info(msg)
                    if self.iteration_logger:
                        self.iteration_logger.info(msg)
                    raise StopIteration(msg)

        if use_gradient:
            self.logger.info(f"최적화 시작: {self.config.estimation.optimizer} (gradient-based)")
            self.iteration_logger.info(f"최적화 시작: {self.config.estimation.optimizer} (gradient-based)")
            if self.use_analytic_gradient:
                self.logger.info("Analytic gradient 사용 (Apollo 방식)")
                self.iteration_logger.info("Analytic gradient 사용 (Apollo 방식)")
            else:
                self.logger.info("수치적 그래디언트 사용 (2-point finite difference)")
                self.iteration_logger.info("수치적 그래디언트 사용 (2-point finite difference)")

            # 조기 종료 callback 생성
            early_stopping = EarlyStoppingCallback(
                patience=10,
                tol=1e-6,
                logger=self.logger,
                iteration_logger=self.iteration_logger
            )

            self.logger.info("조기 종료 활성화: 10회 연속 LL 개선 없으면 종료 (tol=1e-6)")
            self.iteration_logger.info("조기 종료 활성화: 10회 연속 LL 개선 없으면 종료 (tol=1e-6)")

            # BFGS 또는 L-BFGS-B
            try:
                result = optimize.minimize(
                    negative_log_likelihood,
                    initial_params,
                    method=self.config.estimation.optimizer,
                    jac=gradient_function if self.use_analytic_gradient else '2-point',
                    bounds=bounds if self.config.estimation.optimizer == 'L-BFGS-B' else None,
                    callback=early_stopping,
                    options={
                        'maxiter': self.config.estimation.max_iterations,
                        'ftol': 1e-6,
                        'gtol': 1e-5,
                        'disp': True
                    }
                )
            except StopIteration as e:
                # 조기 종료된 경우 - 최적 파라미터로 result 객체 생성
                from scipy.optimize import OptimizeResult

                # best_params를 사용하여 최종 파라미터 벡터 생성
                final_params_vector = params_to_vector(best_params)

                result = OptimizeResult(
                    x=final_params_vector,
                    success=True,
                    message=f"Early stopping: {str(e)}",
                    fun=best_ll,
                    nit=early_stopping.iteration_count,
                    nfev=early_stopping.iteration_count
                )

                self.logger.info(f"조기 종료 완료: 반복 {early_stopping.iteration_count}회, LL={-best_ll:.4f}")
                self.iteration_logger.info(f"조기 종료 완료: 반복 {early_stopping.iteration_count}회, LL={-best_ll:.4f}")
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

        draws = self.halton_generator.get_draws()
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

        # 측정모델 그래디언트
        gradient_list.append(grad_dict['grad_zeta'])
        gradient_list.append(grad_dict['grad_tau'].flatten())

        # 구조모델 그래디언트
        gradient_list.append(grad_dict['grad_gamma'])

        # 선택모델 그래디언트
        gradient_list.append(np.array([grad_dict['grad_intercept']]))
        gradient_list.append(grad_dict['grad_beta'])
        gradient_list.append(np.array([grad_dict['grad_lambda']]))

        # 사회인구학적 변수가 선택모델에 포함되는 경우
        if self.config.structural.include_in_choice:
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
                hessian = optimization_result.hess_inv
                if hasattr(hessian, 'todense'):
                    hessian = hessian.todense()
                
                se = np.sqrt(np.diag(hessian))
                results['standard_errors'] = se
                
                # t-통계량
                results['t_statistics'] = optimization_result.x / se
                
                # p-값
                from scipy.stats import t
                results['p_values'] = 2 * (1 - t.cdf(np.abs(results['t_statistics']), n - k))
                
            except Exception as e:
                self.logger.warning(f"표준오차 계산 실패: {e}")
        
        return results


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

