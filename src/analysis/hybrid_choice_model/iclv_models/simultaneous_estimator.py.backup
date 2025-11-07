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
import logging

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
        self.logger.info("ICLV 모델 동시 추정 시작")
        
        self.data = data
        n_individuals = data[self.config.individual_id_column].nunique()
        
        # Halton draws 생성
        self.halton_generator = HaltonDrawGenerator(
            n_draws=self.config.estimation.n_draws,
            n_individuals=n_individuals,
            scramble=self.config.estimation.scramble_halton
        )
        
        # 초기 파라미터 설정
        initial_params = self._get_initial_parameters(
            measurement_model, structural_model, choice_model
        )
        
        # 결합 우도함수 정의
        def negative_log_likelihood(params):
            return -self._joint_log_likelihood(
                params, measurement_model, structural_model, choice_model
            )
        
        # 최적화
        self.logger.info(f"최적화 시작: {self.config.estimation.optimizer}")
        result = optimize.minimize(
            negative_log_likelihood,
            initial_params,
            method=self.config.estimation.optimizer,
            options={
                'maxiter': self.config.estimation.max_iterations,
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

        # 파라미터 분해
        param_dict = self._unpack_parameters(
            params, measurement_model, structural_model, choice_model
        )

        total_ll = 0.0
        draws = self.halton_generator.get_draws()

        # 개인별 우도 계산
        individual_ids = self.data[self.config.individual_id_column].unique()

        for i, ind_id in enumerate(individual_ids):
            if i % 10 == 0:  # 디버깅: 10명마다 출력
                print(f"   Processing individual {i+1}/{len(individual_ids)}...", flush=True)

            ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
            ind_draws = draws[i, :]  # 이 개인의 draws

            # 시뮬레이션 우도 (각 draw에 대해)
            sim_likelihood = 0.0

            for j, draw in enumerate(ind_draws):
                # 구조모델: LV = γ*X + η
                lv = structural_model.predict(ind_data, param_dict['structural'], draw)

                # 측정모델 우도: P(Indicators|LV)
                ll_measurement = measurement_model.log_likelihood(
                    ind_data, lv, param_dict['measurement']
                )

                # 선택모델 우도: P(Choice|LV, Attributes)
                ll_choice = choice_model.log_likelihood(
                    ind_data, lv, param_dict['choice']
                )

                # 구조모델 우도: P(LV|X) - 정규분포 가정
                ll_structural = structural_model.log_likelihood(
                    ind_data, lv, param_dict['structural'], draw
                )

                # 결합 우도
                sim_likelihood += np.exp(ll_measurement + ll_choice + ll_structural)

            # 평균 (시뮬레이션)
            sim_likelihood /= len(ind_draws)

            # 로그 변환
            if sim_likelihood > 0:
                total_ll += np.log(sim_likelihood)
            else:
                total_ll += -1e10  # 매우 작은 값

        print(f"   Log-likelihood: {total_ll:.4f}", flush=True)
        return total_ll
    
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

