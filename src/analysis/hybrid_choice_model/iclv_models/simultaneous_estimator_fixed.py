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

from .gradient_calculator import (
    MeasurementGradient,
    StructuralGradient,
    ChoiceGradient,
    JointGradient
)

logger = logging.getLogger(__name__)


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

        # Gradient calculators (Apollo 방식)
        self.measurement_grad = None
        self.structural_grad = None
        self.choice_grad = None
        self.joint_grad = None
        self.use_analytic_gradient = False  # 기본값: 수치적 그래디언트
    
    def estimate(self, data: pd.DataFrame, 
                measurement_model,
                structural_model,
                choice_model) -> Dict:
        """
        ICLV 모델 동시 추정
        
        Args:
            data: 통합 데이터
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
        
        Returns:
            추정 결과 딕셔너리
        """
        print("=" * 70, flush=True)
        print("SimultaneousEstimator.estimate() 시작", flush=True)
        print("=" * 70, flush=True)
        self.logger.info("ICLV 모델 동시 추정 시작")

        self.data = data
        print(f"데이터 shape: {data.shape}", flush=True)
        n_individuals = data[self.config.individual_id_column].nunique()
        print(f"개인 수: {n_individuals}", flush=True)
        self.logger.info(f"개인 수: {n_individuals}")

        # Halton draws 생성
        print(f"Halton draws 생성 시작... (n_draws={self.config.estimation.n_draws}, n_individuals={n_individuals})", flush=True)
        self.logger.info(f"Halton draws 생성 중... (n_draws={self.config.estimation.n_draws})")
        self.halton_generator = HaltonDrawGenerator(
            n_draws=self.config.estimation.n_draws,
            n_individuals=n_individuals,
            scramble=self.config.estimation.scramble_halton
        )
        print("Halton draws 생성 완료", flush=True)
        self.logger.info("Halton draws 생성 완료")

        # Gradient calculators 초기화 (Apollo 방식)
        use_gradient = self.config.estimation.optimizer in ['BFGS', 'L-BFGS-B']
        if use_gradient and hasattr(self.config.estimation, 'use_analytic_gradient'):
            self.use_analytic_gradient = self.config.estimation.use_analytic_gradient
        else:
            self.use_analytic_gradient = False

        if self.use_analytic_gradient:
            print("Analytic gradient calculators 초기화 (Apollo 방식)...", flush=True)
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
            print("Analytic gradient calculators 초기화 완료", flush=True)

        # 초기 파라미터 설정
        print("초기 파라미터 설정 시작...", flush=True)
        self.logger.info("초기 파라미터 설정 중...")
        initial_params = self._get_initial_parameters(
            measurement_model, structural_model, choice_model
        )
        print(f"초기 파라미터 설정 완료 (총 {len(initial_params)}개)", flush=True)
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
                improvement = "[NEW BEST]"  # 🔴 ✓ 대신 ASCII 문자 사용
            else:
                improvement = ""

            # Log every iteration with more detail
            if iteration_count[0] % 5 == 0 or improvement:
                print(
                    f"Iter {iteration_count[0]:4d}: LL = {ll:12.4f} "
                    f"(Best: {best_ll[0]:12.4f}) {improvement}",
                    flush=True
                )

            return -ll

        # Get parameter bounds
        print("파라미터 bounds 계산 시작...", flush=True)
        bounds = self._get_parameter_bounds(
            measurement_model, structural_model, choice_model
        )
        print(f"파라미터 bounds 계산 완료 (총 {len(bounds)}개)", flush=True)

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
                choice_attributes=self.config.choice.choice_attributes
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
        print(f"최대 반복 횟수: {self.config.estimation.max_iterations}", flush=True)
        print("=" * 70, flush=True)

        if use_gradient:
            self.logger.info(f"최적화 시작: {self.config.estimation.optimizer} (gradient-based)")
            if self.use_analytic_gradient:
                self.logger.info("Analytic gradient 사용 (Apollo 방식)")
            else:
                self.logger.info("수치적 그래디언트 사용 (2-point finite difference)")

            # BFGS 또는 L-BFGS-B
            result = optimize.minimize(
                negative_log_likelihood,
                initial_params,
                method=self.config.estimation.optimizer,
                jac=gradient_function if self.use_analytic_gradient else '2-point',
                bounds=bounds if self.config.estimation.optimizer == 'L-BFGS-B' else None,
                options={
                    'maxiter': self.config.estimation.max_iterations,
                    'ftol': 1e-6,
                    'gtol': 1e-5,
                    'disp': True
                }
            )
        else:
            self.logger.info(f"최적화 시작: Nelder-Mead (gradient-free)")

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
        else:
            self.logger.warning(f"최적화 실패: {result.message}")
        
        # 결과 처리
        self.results = self._process_results(
            result, measurement_model, structural_model, choice_model
        )
        
        return self.results
    
    def _joint_log_likelihood(self, params: np.ndarray,
                             measurement_model,
                             structural_model,
                             choice_model) -> float:
        """
        결합 로그우도 계산

        시뮬레이션 기반:
        log L ≈ Σᵢ log[(1/R) Σᵣ P(Choice|LVᵣ) × P(Indicators|LVᵣ) × P(LVᵣ|X)]
        """
        # print("_joint_log_likelihood 시작", flush=True)

        # 파라미터 분해
        param_dict = self._unpack_parameters(
            params, measurement_model, structural_model, choice_model
        )

        total_ll = 0.0
        draws = self.halton_generator.get_draws()

        # 개인별 우도 계산
        individual_ids = self.data[self.config.individual_id_column].unique()
        # print(f"개인 수: {len(individual_ids)}", flush=True)

        for i, ind_id in enumerate(individual_ids):
            ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
            ind_draws = draws[i, :]  # 이 개인의 draws

            # 🔴 수정: logsumexp를 사용하여 수치 안정성 확보
            # King (2022) Apollo 방식
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

