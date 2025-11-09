"""
Multi-Latent Variable Simultaneous Estimator

다중 잠재변수 ICLV 모델의 동시추정 엔진입니다.
기존 SimultaneousEstimator의 로직을 확장하여 5개 잠재변수를 지원합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import optimize
from scipy.stats import norm, qmc
from scipy.special import logsumexp
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

from .multi_latent_measurement import MultiLatentMeasurement
from .multi_latent_structural import MultiLatentStructural
from .choice_equations import BinaryProbitChoice
from .multi_latent_config import MultiLatentConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 병렬처리를 위한 전역 함수 (pickle 가능)
# ============================================================================

def _compute_multi_lv_individual_likelihood_parallel(args):
    """
    다중 잠재변수 개인별 우도 계산 (병렬처리용 전역 함수)

    Args:
        args: (ind_data_dict, ind_draws, param_dict, config_dict)

    Returns:
        개인의 로그우도
    """
    # 병렬 프로세스에서 불필요한 로그 억제
    import logging
    logging.getLogger('root').setLevel(logging.CRITICAL)

    from .multi_latent_measurement import MultiLatentMeasurement
    from .multi_latent_structural import MultiLatentStructural
    from .choice_equations import BinaryProbitChoice
    from .multi_latent_config import MultiLatentConfig, MultiLatentStructuralConfig
    from .iclv_config import MeasurementConfig, ChoiceConfig, EstimationConfig

    ind_id, ind_data_dict, ind_draws, param_dict, config_dict = args

    # DataFrame 복원
    ind_data = pd.DataFrame(ind_data_dict)

    # 설정 복원
    measurement_configs = {}
    for lv_name, lv_config_dict in config_dict['measurement'].items():
        measurement_configs[lv_name] = MeasurementConfig(**lv_config_dict)

    structural_config = MultiLatentStructuralConfig(**config_dict['structural'])
    choice_config = ChoiceConfig(**config_dict['choice'])

    # 모델 재생성
    measurement_model = MultiLatentMeasurement(measurement_configs)
    structural_model = MultiLatentStructural(structural_config)
    choice_model = BinaryProbitChoice(choice_config)

    # 우도 계산
    draw_lls = []
    n_exo = structural_config.n_exo
    endogenous_lv = structural_config.endogenous_lv

    for draw_idx in range(len(ind_draws)):
        # Draws 분리
        exo_draws = ind_draws[draw_idx, :n_exo]
        endo_draw = ind_draws[draw_idx, n_exo]

        # 구조모델: 모든 LV 예측
        latent_vars = structural_model.predict(
            ind_data, exo_draws, param_dict['structural'], endo_draw
        )

        # 측정모델 우도
        ll_measurement = measurement_model.log_likelihood(
            ind_data, latent_vars, param_dict['measurement']
        )

        # 선택모델 우도 (내생 LV만 사용)
        lv_endo = latent_vars[endogenous_lv]
        ll_choice = 0.0
        for idx in range(len(ind_data)):
            ll_choice += choice_model.log_likelihood(
                ind_data.iloc[idx:idx+1],
                lv_endo,
                param_dict['choice']
            )

        # 구조모델 우도
        ll_structural = structural_model.log_likelihood(
            ind_data, latent_vars, exo_draws, param_dict['structural'], endo_draw
        )

        # 결합 로그우도
        draw_ll = ll_measurement + ll_choice + ll_structural

        if not np.isfinite(draw_ll):
            draw_ll = -1e10

        draw_lls.append(draw_ll)

    # logsumexp
    person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))
    return (ind_id, person_ll)


class HaltonDrawGenerator:
    """Halton 시퀀스 생성기 (기존 코드 재사용)"""
    
    def __init__(self, n_individuals: int, n_draws: int, n_dimensions: int, seed: int = 42):
        """
        Args:
            n_individuals: 개인 수
            n_draws: 개인당 draws 수
            n_dimensions: 차원 수 (외생 LV 수 + 1)
        """
        self.n_individuals = n_individuals
        self.n_draws = n_draws
        self.n_dimensions = n_dimensions
        self.seed = seed
        
        # Halton 시퀀스 생성
        sampler = qmc.Halton(d=n_dimensions, scramble=True, seed=seed)
        uniform_draws = sampler.random(n=n_individuals * n_draws)
        
        # 표준정규분포로 변환
        self.draws = norm.ppf(uniform_draws)
        
        # (n_individuals, n_draws, n_dimensions) 형태로 reshape
        self.draws = self.draws.reshape(n_individuals, n_draws, n_dimensions)
        
        logger.info(f"Halton draws 생성: {n_individuals}명 × {n_draws}draws × {n_dimensions}차원")
    
    def get_draws(self) -> np.ndarray:
        """Draws 반환"""
        return self.draws


class MultiLatentSimultaneousEstimator:
    """
    다중 잠재변수 동시추정 엔진
    
    결합 로그우도:
    LL = Σ_i log[(1/R) Σ_r P(Choice_i|LV5_r) 
                        × P(Indicators1_i|LV1_r) 
                        × P(Indicators2_i|LV2_r)
                        × P(Indicators3_i|LV3_r)
                        × P(Indicators4_i|LV4_r)
                        × P(Indicators5_i|LV5_r)
                        × P(LV5_r|LV1_r,LV2_r,LV3_r,LV4_r,X_i)
                        × P(LV1_r)
                        × P(LV2_r)
                        × P(LV3_r)
                        × P(LV4_r)]
    """
    
    def __init__(self, config: MultiLatentConfig, data: pd.DataFrame):
        """
        초기화
        
        Args:
            config: 다중 LV ICLV 설정
            data: 통합 데이터
        """
        self.config = config
        self.data = data
        
        # 모델 생성
        self.measurement_model = MultiLatentMeasurement(config.measurement_configs)
        self.structural_model = MultiLatentStructural(config.structural)
        self.choice_model = BinaryProbitChoice(config.choice)
        
        # Halton draws 생성
        individual_ids = data[config.individual_id_column].unique()
        n_individuals = len(individual_ids)
        n_draws = config.estimation.n_draws
        n_dimensions = config.structural.n_exo + 1  # 외생 LV + 내생 LV 오차항
        
        self.halton_generator = HaltonDrawGenerator(
            n_individuals=n_individuals,
            n_draws=n_draws,
            n_dimensions=n_dimensions,
            seed=42  # 고정 seed
        )
        
        # 파라미터 개수 계산
        n_measurement_params = self.measurement_model.get_n_parameters()
        n_structural_params = config.structural.n_exo + config.structural.n_cov  # gamma_lv + gamma_x
        n_choice_params = 1 + len(config.choice.choice_attributes) + 1  # intercept + beta + lambda
        total_params = n_measurement_params + n_structural_params + n_choice_params

        # 로깅
        logger.info("=" * 70)
        logger.info("MultiLatentSimultaneousEstimator 초기화")
        logger.info(f"  개인 수: {n_individuals:,}")
        logger.info(f"  관측치 수: {len(data):,}")
        logger.info(f"  개인당 선택 상황: {len(data) / n_individuals:.1f}")
        logger.info(f"  Halton draws: {n_draws}")
        logger.info(f"  차원: {n_dimensions} (외생 LV {config.structural.n_exo} + 내생 LV 오차 1)")
        logger.info(f"  측정모델 파라미터: {n_measurement_params}")
        logger.info(f"  구조모델 파라미터: {n_structural_params}")
        logger.info(f"  선택모델 파라미터: {n_choice_params}")
        logger.info(f"  총 파라미터: {total_params}")
        logger.info(f"  총 시뮬레이션: {n_individuals:,} × {n_draws} = {n_individuals * n_draws:,}")
        logger.info("=" * 70)
        
        # 추정 결과 저장
        self.results = None
    
    def _compute_individual_likelihood(self, ind_id, ind_data, ind_draws, param_dict) -> float:
        """
        개인별 우도 계산
        
        Args:
            ind_id: 개인 ID
            ind_data: 개인 데이터 (여러 선택 상황)
            ind_draws: 개인의 Halton draws (n_draws, n_dimensions)
            param_dict: 파라미터 딕셔너리
        
        Returns:
            개인의 로그우도
        """
        draw_lls = []
        
        n_exo = self.config.structural.n_exo
        
        for draw_idx in range(len(ind_draws)):
            # 1. Draws 분리
            exo_draws = ind_draws[draw_idx, :n_exo]  # 외생 LV (4개)
            endo_draw = ind_draws[draw_idx, n_exo]   # 내생 LV 오차항 (1개)
            
            # 2. 구조모델: 모든 LV 예측
            latent_vars = self.structural_model.predict(
                ind_data, exo_draws, param_dict['structural'], endo_draw
            )
            
            # 3. 측정모델 우도 (5개 LV)
            ll_measurement = self.measurement_model.log_likelihood(
                ind_data, latent_vars, param_dict['measurement']
            )
            
            # 4. 선택모델 우도 (내생 LV만 사용)
            lv_endo = latent_vars[self.config.structural.endogenous_lv]
            ll_choice = 0.0
            for idx in range(len(ind_data)):
                ll_choice += self.choice_model.log_likelihood(
                    ind_data.iloc[idx:idx+1],
                    lv_endo,
                    param_dict['choice']
                )
            
            # 5. 구조모델 우도
            ll_structural = self.structural_model.log_likelihood(
                ind_data, latent_vars, exo_draws, param_dict['structural'], endo_draw
            )
            
            # 6. 결합 로그우도
            draw_ll = ll_measurement + ll_choice + ll_structural
            
            # -inf 처리
            if not np.isfinite(draw_ll):
                draw_ll = -1e10
            
            draw_lls.append(draw_ll)
        
        # logsumexp
        person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))
        return person_ll
    
    def _joint_log_likelihood(self, params: np.ndarray) -> float:
        """
        결합 로그우도 계산 (병렬 처리 지원)

        Args:
            params: 파라미터 벡터 (1D array)

        Returns:
            전체 로그우도
        """
        # 파라미터 분해
        param_dict = self._unpack_parameters(params)

        # Halton draws
        draws = self.halton_generator.get_draws()
        individual_ids = self.data[self.config.individual_id_column].unique()

        # 병렬처리 여부 확인
        use_parallel = getattr(self.config.estimation, 'use_parallel', False)

        if use_parallel:
            # 병렬처리 사용
            n_cores = getattr(self.config.estimation, 'n_cores', None)
            if n_cores is None:
                n_cores = max(1, multiprocessing.cpu_count() - 1)

            # 설정 정보를 dict로 변환 (pickle 가능)
            config_dict = {
                'measurement': {},
                'structural': {
                    'endogenous_lv': self.config.structural.endogenous_lv,
                    'exogenous_lvs': self.config.structural.exogenous_lvs,
                    'covariates': self.config.structural.covariates,
                    'error_variance': self.config.structural.error_variance
                },
                'choice': {
                    'choice_attributes': self.config.choice.choice_attributes
                }
            }

            # 측정모델 설정 변환
            for lv_name, lv_config in self.config.measurement_configs.items():
                config_dict['measurement'][lv_name] = {
                    'latent_variable': lv_config.latent_variable,
                    'indicators': lv_config.indicators,
                    'n_categories': lv_config.n_categories
                }

            # 개인별 데이터 준비
            args_list = []
            for i, ind_id in enumerate(individual_ids):
                ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                ind_data_dict = ind_data.to_dict('list')
                ind_draws = draws[i, :, :]
                args_list.append((ind_id, ind_data_dict, ind_draws, param_dict, config_dict))

            # 병렬 계산
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                results = list(executor.map(_compute_multi_lv_individual_likelihood_parallel, args_list))

            # 결과 정리 (ind_id, person_ll)
            person_lls = [ll for _, ll in results]
            total_ll = sum(person_lls)
        else:
            # 순차처리
            total_ll = 0.0
            for i, ind_id in enumerate(individual_ids):
                ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                ind_draws = draws[i, :, :]

                person_ll = self._compute_individual_likelihood(
                    ind_id, ind_data, ind_draws, param_dict
                )
                total_ll += person_ll

        return total_ll
    
    def _unpack_parameters(self, params: np.ndarray) -> Dict:
        """
        파라미터 벡터를 딕셔너리로 분해
        
        파라미터 순서:
        1. 측정모델 (5개 LV)
        2. 구조모델 (gamma_lv, gamma_x)
        3. 선택모델 (beta, lambda)
        
        Args:
            params: 파라미터 벡터
        
        Returns:
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
        for lv_name, model in self.measurement_model.models.items():
            n_indicators = model.n_indicators
            n_thresholds = model.n_thresholds
            
            # zeta
            zeta = params[idx:idx+n_indicators]
            idx += n_indicators
            
            # tau
            tau = params[idx:idx+n_indicators*n_thresholds].reshape(n_indicators, n_thresholds)
            idx += n_indicators * n_thresholds
            
            measurement_params[lv_name] = {'zeta': zeta, 'tau': tau}
        
        param_dict['measurement'] = measurement_params
        
        # 2. 구조모델 파라미터
        n_exo = self.structural_model.n_exo
        n_cov = self.structural_model.n_cov
        
        gamma_lv = params[idx:idx+n_exo]
        idx += n_exo
        
        gamma_x = params[idx:idx+n_cov]
        idx += n_cov
        
        param_dict['structural'] = {
            'gamma_lv': gamma_lv,
            'gamma_x': gamma_x
        }
        
        # 3. 선택모델 파라미터
        n_choice_attrs = len(self.config.choice.choice_attributes)
        
        beta_intercept = params[idx]
        idx += 1
        
        beta = params[idx:idx+n_choice_attrs]
        idx += n_choice_attrs
        
        lambda_lv = params[idx]
        idx += 1
        
        param_dict['choice'] = {
            'intercept': beta_intercept,  # BinaryProbitChoice expects 'intercept'
            'beta': beta,
            'lambda': lambda_lv
        }
        
        return param_dict

    def _pack_parameters(self, param_dict: Dict) -> np.ndarray:
        """
        파라미터 딕셔너리를 벡터로 변환

        Args:
            param_dict: 파라미터 딕셔너리

        Returns:
            파라미터 벡터
        """
        params_list = []

        # 1. 측정모델
        for lv_name in self.measurement_model.get_latent_variable_names():
            lv_params = param_dict['measurement'][lv_name]
            params_list.append(lv_params['zeta'])
            params_list.append(lv_params['tau'].flatten())

        # 2. 구조모델
        params_list.append(param_dict['structural']['gamma_lv'])
        params_list.append(param_dict['structural']['gamma_x'])

        # 3. 선택모델
        params_list.append(np.array([param_dict['choice']['intercept']]))
        params_list.append(param_dict['choice']['beta'])
        params_list.append(np.array([param_dict['choice']['lambda']]))

        return np.concatenate(params_list)

    def _initialize_parameters(self) -> np.ndarray:
        """
        파라미터 초기화

        Returns:
            초기 파라미터 벡터
        """
        param_dict = {}

        # 1. 측정모델
        param_dict['measurement'] = self.measurement_model.initialize_parameters()

        # 2. 구조모델
        param_dict['structural'] = self.structural_model.initialize_parameters()

        # 3. 선택모델
        n_choice_attrs = len(self.config.choice.choice_attributes)
        param_dict['choice'] = {
            'intercept': 0.0,
            'beta': np.zeros(n_choice_attrs),
            'lambda': 1.0
        }

        return self._pack_parameters(param_dict)

    def estimate(self) -> Dict:
        """
        모델 추정

        Returns:
            추정 결과 딕셔너리
        """
        logger.info("=" * 70)
        logger.info("다중 잠재변수 ICLV 모델 추정 시작")
        logger.info("=" * 70)

        # 병렬처리 설정 로깅
        use_parallel = getattr(self.config.estimation, 'use_parallel', False)
        n_individuals = len(self.data[self.config.individual_id_column].unique())

        if use_parallel:
            n_cores = getattr(self.config.estimation, 'n_cores', None)
            if n_cores is None:
                n_cores = max(1, multiprocessing.cpu_count() - 1)
            logger.info(f"🚀 병렬처리 활성화")
            logger.info(f"   - 사용 코어: {n_cores}/{multiprocessing.cpu_count()}개 ({n_cores/multiprocessing.cpu_count()*100:.1f}%)")
            logger.info(f"   - 개인당 코어: {n_individuals/n_cores:.1f}명/코어")
            logger.info(f"   - 예상 속도 향상: ~{n_cores}배")
        else:
            logger.info("⚠️  순차처리 사용 (병렬처리 비활성화)")
            logger.info(f"   - 사용 가능한 코어: {multiprocessing.cpu_count()}개")
            logger.info(f"   - 병렬처리를 활성화하려면 config.estimation.use_parallel=True 설정")
            logger.info(f"   - 예상 소요 시간: 순차 처리로 오래 걸릴 수 있습니다")

        start_time = time.time()

        # 초기 파라미터
        initial_params = self._initialize_parameters()
        logger.info(f"초기 파라미터 수: {len(initial_params)}")

        # 초기 로그우도
        logger.info("초기 로그우도 계산 중...")
        ll_start_time = time.time()
        initial_ll = self._joint_log_likelihood(initial_params)
        ll_elapsed = time.time() - ll_start_time
        logger.info(f"초기 로그우도: {initial_ll:.4f} (소요: {ll_elapsed:.1f}초)")
        logger.info(f"  → 1회 우도 계산 시간: {ll_elapsed:.1f}초")
        logger.info(f"  → 예상 총 소요 시간: {ll_elapsed * self.config.estimation.max_iterations / 60:.1f}분 (최대 반복 기준)")

        # 목적 함수 (negative log-likelihood)
        def objective(params):
            ll = self._joint_log_likelihood(params)
            return -ll

        # 최적화
        logger.info(f"\n최적화 시작: {self.config.estimation.optimizer}")
        logger.info(f"  최대 반복: {self.config.estimation.max_iterations}")
        logger.info(f"  초기 LL: {initial_ll:.4f}")
        logger.info("  진행 상황은 아래에 표시됩니다...\n")

        # 반복 카운터
        iteration_count = [0]
        last_log_time = [time.time()]
        best_ll = [-np.inf]
        ll_history = []

        def callback(xk):
            """최적화 진행 상황 로깅"""
            iteration_count[0] += 1
            current_time = time.time()

            # 매 반복마다 LL 계산 (로깅은 조건부)
            ll = -objective(xk)
            ll_history.append(ll)

            # 개선 여부 확인
            is_improvement = ll > best_ll[0]
            if is_improvement:
                best_ll[0] = ll

            # 5초마다 또는 5 반복마다 또는 개선 시 로깅
            should_log = (current_time - last_log_time[0] > 5 or
                         iteration_count[0] % 5 == 0 or
                         is_improvement)

            if should_log:
                elapsed = current_time - start_time
                iter_per_sec = iteration_count[0] / elapsed if elapsed > 0 else 0
                remaining_iters = self.config.estimation.max_iterations - iteration_count[0]
                eta_sec = remaining_iters / iter_per_sec if iter_per_sec > 0 else 0

                improvement_str = " [✨ NEW BEST]" if is_improvement else ""
                logger.info(f"  반복 {iteration_count[0]:3d}: LL = {ll:12.4f} (Best: {best_ll[0]:12.4f}){improvement_str}")
                logger.info(f"         경과: {elapsed:.1f}초 | 속도: {iter_per_sec:.2f} iter/s | 예상 남은 시간: {eta_sec/60:.1f}분")
                last_log_time[0] = current_time

        result = optimize.minimize(
            objective,
            initial_params,
            method=self.config.estimation.optimizer,
            callback=callback,
            options={
                'maxiter': self.config.estimation.max_iterations,
                'disp': False  # callback으로 직접 로깅
            }
        )

        # 결과 처리
        elapsed_time = time.time() - start_time

        final_params = result.x
        final_ll = -result.fun

        logger.info("\n" + "=" * 70)
        logger.info("✅ 추정 완료")
        logger.info("=" * 70)
        logger.info(f"  최종 로그우도: {final_ll:.4f}")
        logger.info(f"  초기 로그우도: {initial_ll:.4f}")
        logger.info(f"  LL 개선: {final_ll - initial_ll:.4f} ({(final_ll - initial_ll)/abs(initial_ll)*100:.2f}%)")
        logger.info(f"  소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
        logger.info(f"  반복 횟수: {result.nit}")
        logger.info(f"  수렴 여부: {'✅ 성공' if result.success else '❌ 실패'}")
        if use_parallel:
            logger.info(f"  병렬 처리: {n_cores}개 코어 사용")
            logger.info(f"  예상 순차 시간: ~{elapsed_time * n_cores / 60:.0f}분")
        logger.info("=" * 70)

        # 파라미터 분해
        param_dict = self._unpack_parameters(final_params)

        # 결과 저장
        self.results = {
            'params': param_dict,
            'log_likelihood': final_ll,
            'n_parameters': len(final_params),
            'n_observations': len(self.data),
            'n_individuals': len(self.data[self.config.individual_id_column].unique()),
            'convergence': result.success,
            'iterations': result.nit,
            'elapsed_time': elapsed_time,
            'optimizer_result': result
        }

        return self.results

    def print_results(self):
        """추정 결과 출력"""
        if self.results is None:
            logger.error("추정 결과가 없습니다. estimate()를 먼저 실행하세요.")
            return

        print("\n" + "=" * 70)
        print("다중 잠재변수 ICLV 모델 추정 결과")
        print("=" * 70)

        # 모델 적합도
        print("\n[모델 적합도]")
        print(f"  로그우도: {self.results['log_likelihood']:.4f}")
        print(f"  파라미터 수: {self.results['n_parameters']}")
        print(f"  관측치 수: {self.results['n_observations']}")
        print(f"  개인 수: {self.results['n_individuals']}")

        # 측정모델 파라미터
        print("\n[측정모델 파라미터]")
        for lv_name, lv_params in self.results['params']['measurement'].items():
            print(f"\n  {lv_name}:")
            print(f"    zeta: {lv_params['zeta']}")

        # 구조모델 파라미터
        print("\n[구조모델 파라미터]")
        structural_params = self.results['params']['structural']

        print("\n  외생 LV → 내생 LV:")
        for i, lv_name in enumerate(self.structural_model.exogenous_lvs):
            print(f"    gamma_{lv_name}: {structural_params['gamma_lv'][i]:.4f}")

        print("\n  공변량 → 내생 LV:")
        for i, var in enumerate(self.structural_model.covariates):
            print(f"    gamma_{var}: {structural_params['gamma_x'][i]:.4f}")

        # 선택모델 파라미터
        print("\n[선택모델 파라미터]")
        choice_params = self.results['params']['choice']
        print(f"  beta_intercept: {choice_params['beta_intercept']:.4f}")
        for i, attr in enumerate(self.config.choice.choice_attributes):
            print(f"  beta_{attr}: {choice_params['beta'][i]:.4f}")
        print(f"  lambda (LV → Choice): {choice_params['lambda']:.4f}")

        print("\n" + "=" * 70)

