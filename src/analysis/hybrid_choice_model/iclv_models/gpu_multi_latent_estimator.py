"""
GPU-Accelerated Multi-Latent Variable ICLV Estimator

CuPy를 사용하여 GPU에서 다중 잠재변수 ICLV 모델을 추정합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy import optimize
from scipy.special import logsumexp

from .gpu_measurement_equations import GPUMultiLatentMeasurement, GPU_AVAILABLE
from .multi_latent_structural import MultiLatentStructural
from .choice_equations import BinaryProbitChoice

logger = logging.getLogger(__name__)

# GPU 사용 가능 여부
if GPU_AVAILABLE:
    import cupy as cp
    logger.info("✅ GPU 가속 사용 가능")
else:
    logger.warning("⚠️ CuPy 미설치 - CPU 모드로 작동")


class GPUMultiLatentSimultaneousEstimator:
    """
    GPU 가속 다중 잠재변수 ICLV 동시추정
    
    측정모델의 정규분포 CDF 계산을 GPU에서 수행하여 속도를 향상시킵니다.
    """
    
    def __init__(self, config, data: pd.DataFrame, use_gpu: bool = True):
        """
        초기화
        
        Args:
            config: MultiLatentConfig 객체
            data: 통합 데이터
            use_gpu: GPU 사용 여부 (기본값: True)
        """
        self.config = config
        self.data = data
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # 모델 생성
        self.measurement_model = GPUMultiLatentMeasurement(
            config.measurement_configs, 
            use_gpu=self.use_gpu
        )
        self.structural_model = MultiLatentStructural(config.structural)
        self.choice_model = BinaryProbitChoice(config.choice)
        
        # Halton draws 생성
        n_individuals = data[config.individual_id_column].nunique()
        n_draws = config.estimation.n_draws
        n_dimensions = config.structural.n_exo + 1  # 외생 LV + 내생 LV 오차
        
        self.halton_generator = HaltonDrawGenerator(
            n_individuals, n_draws, n_dimensions
        )
        
        # 로깅
        n_measurement_params = self.measurement_model.get_n_parameters()
        n_structural_params = config.structural.n_exo + config.structural.n_cov
        n_choice_params = 1 + len(config.choice.choice_attributes) + 1
        total_params = n_measurement_params + n_structural_params + n_choice_params
        
        gpu_status = "🚀 GPU" if self.use_gpu else "💻 CPU"
        logger.info("=" * 70)
        logger.info(f"{gpu_status} MultiLatentSimultaneousEstimator 초기화")
        logger.info(f"  개인 수: {n_individuals:,}")
        logger.info(f"  관측치 수: {len(data):,}")
        logger.info(f"  Halton draws: {n_draws}")
        logger.info(f"  측정모델 파라미터: {n_measurement_params}")
        logger.info(f"  구조모델 파라미터: {n_structural_params}")
        logger.info(f"  선택모델 파라미터: {n_choice_params}")
        logger.info(f"  총 파라미터: {total_params}")
        logger.info("=" * 70)
    
    def _compute_individual_likelihood(self, ind_id, ind_data, ind_draws, param_dict):
        """개인별 우도 계산 (GPU 사용)"""
        draw_lls = []
        n_exo = self.config.structural.n_exo
        endogenous_lv = self.config.structural.endogenous_lv
        
        for draw_idx in range(len(ind_draws)):
            exo_draws = ind_draws[draw_idx, :n_exo]
            endo_draw = ind_draws[draw_idx, n_exo]
            
            # 구조모델: 모든 LV 예측
            latent_vars = self.structural_model.predict(
                ind_data, exo_draws, param_dict['structural'], endo_draw
            )
            
            # 측정모델 우도 (GPU 가속)
            ll_measurement = self.measurement_model.log_likelihood(
                ind_data, latent_vars, param_dict['measurement']
            )
            
            # 선택모델 우도
            lv_endo = latent_vars[endogenous_lv]
            ll_choice = 0.0
            for idx in range(len(ind_data)):
                ll_choice += self.choice_model.log_likelihood(
                    ind_data.iloc[idx:idx+1],
                    lv_endo,
                    param_dict['choice']
                )
            
            # 구조모델 우도
            ll_structural = self.structural_model.log_likelihood(
                ind_data, latent_vars, exo_draws, param_dict['structural'], endo_draw
            )
            
            # 결합 로그우도
            draw_ll = ll_measurement + ll_choice + ll_structural
            
            if not np.isfinite(draw_ll):
                draw_ll = -1e10
            
            draw_lls.append(draw_ll)
        
        # logsumexp
        person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))
        return person_ll
    
    def _joint_log_likelihood(self, params: np.ndarray) -> float:
        """결합 로그우도 계산"""
        param_dict = self._unpack_parameters(params)
        
        draws = self.halton_generator.get_draws()
        individual_ids = self.data[self.config.individual_id_column].unique()
        
        # CPU 병렬처리 사용 (GPU는 측정모델 내부에서만 사용)
        use_parallel = getattr(self.config.estimation, 'use_parallel', False)
        
        if use_parallel:
            n_cores = getattr(self.config.estimation, 'n_cores', None)
            if n_cores is None:
                n_cores = max(1, multiprocessing.cpu_count() - 1)
            
            # 설정 정보를 dict로 변환
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
                args_list.append((ind_id, ind_data_dict, ind_draws, param_dict, config_dict, self.use_gpu))
            
            # 병렬 계산
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                results = list(executor.map(_compute_gpu_individual_likelihood_parallel, args_list))
            
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
        """파라미터 벡터를 딕셔너리로 분해"""
        idx = 0
        param_dict = {}
        
        # 1. 측정모델 파라미터
        param_dict['measurement'] = {}
        for lv_name, model in self.measurement_model.models.items():
            n_indicators = model.n_indicators
            n_thresholds = model.n_thresholds
            
            zeta = params[idx:idx + n_indicators]
            idx += n_indicators
            
            tau = params[idx:idx + n_indicators * n_thresholds]
            tau = tau.reshape(n_indicators, n_thresholds)
            idx += n_indicators * n_thresholds
            
            param_dict['measurement'][lv_name] = {'zeta': zeta, 'tau': tau}
        
        # 2. 구조모델 파라미터
        n_exo = self.config.structural.n_exo
        n_cov = self.config.structural.n_cov
        
        gamma_lv = params[idx:idx + n_exo]
        idx += n_exo
        
        gamma_x = params[idx:idx + n_cov]
        idx += n_cov
        
        param_dict['structural'] = {'gamma_lv': gamma_lv, 'gamma_x': gamma_x}
        
        # 3. 선택모델 파라미터
        beta_intercept = params[idx]
        idx += 1
        
        n_choice_attrs = len(self.config.choice.choice_attributes)
        beta = params[idx:idx + n_choice_attrs]
        idx += n_choice_attrs
        
        lambda_lv = params[idx]
        idx += 1
        
        param_dict['choice'] = {
            'intercept': beta_intercept,
            'beta': beta,
            'lambda': lambda_lv
        }
        
        return param_dict
    
    def _pack_parameters(self, param_dict: Dict) -> np.ndarray:
        """딕셔너리를 파라미터 벡터로 변환"""
        params_list = []
        
        # 1. 측정모델
        for lv_name in sorted(param_dict['measurement'].keys()):
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
        """초기 파라미터 생성"""
        param_dict = {}
        
        param_dict['measurement'] = self.measurement_model.initialize_parameters()
        param_dict['structural'] = self.structural_model.initialize_parameters()
        
        n_choice_attrs = len(self.config.choice.choice_attributes)
        param_dict['choice'] = {
            'intercept': 0.0,
            'beta': np.zeros(n_choice_attrs),
            'lambda': 1.0
        }
        
        return self._pack_parameters(param_dict)
    
    def estimate(self) -> Dict:
        """모델 추정"""
        gpu_status = "🚀 GPU" if self.use_gpu else "💻 CPU"
        logger.info("=" * 70)
        logger.info(f"{gpu_status} 다중 잠재변수 ICLV 모델 추정 시작")
        logger.info("=" * 70)
        
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
        
        # 목적 함수
        def objective(params):
            ll = self._joint_log_likelihood(params)
            return -ll
        
        # 최적화
        logger.info(f"\n최적화 시작: {self.config.estimation.optimizer}")
        
        iteration_count = [0]
        last_log_time = [time.time()]
        best_ll = [-np.inf]
        
        def callback(xk):
            iteration_count[0] += 1
            current_time = time.time()
            
            ll = -objective(xk)
            
            is_improvement = ll > best_ll[0]
            if is_improvement:
                best_ll[0] = ll
            
            should_log = (current_time - last_log_time[0] > 5 or 
                         iteration_count[0] % 5 == 0 or 
                         is_improvement)
            
            if should_log:
                elapsed = current_time - start_time
                iter_per_sec = iteration_count[0] / elapsed if elapsed > 0 else 0
                
                improvement_str = " [✨ NEW BEST]" if is_improvement else ""
                logger.info(f"  반복 {iteration_count[0]:3d}: LL = {ll:12.4f} (Best: {best_ll[0]:12.4f}){improvement_str}")
                logger.info(f"         경과: {elapsed:.1f}초 | 속도: {iter_per_sec:.2f} iter/s")
                last_log_time[0] = current_time
        
        result = optimize.minimize(
            objective,
            initial_params,
            method=self.config.estimation.optimizer,
            callback=callback,
            options={'maxiter': self.config.estimation.max_iterations}
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("=" * 70)
        logger.info(f"{gpu_status} 추정 완료!")
        logger.info(f"  최종 LL: {-result.fun:.4f}")
        logger.info(f"  반복 횟수: {result.nit}")
        logger.info(f"  총 소요 시간: {total_time/60:.1f}분")
        logger.info(f"  수렴 여부: {result.success}")
        logger.info("=" * 70)
        
        return {
            'params': self._unpack_parameters(result.x),
            'log_likelihood': -result.fun,
            'success': result.success,
            'n_iterations': result.nit,
            'time_elapsed': total_time
        }


# 병렬처리용 전역 함수
def _compute_gpu_individual_likelihood_parallel(args):
    """개인별 우도 계산 (병렬처리용, GPU 지원)"""
    ind_id, ind_data_dict, ind_draws, param_dict, config_dict, use_gpu = args
    
    import logging
    logging.getLogger('root').setLevel(logging.CRITICAL)
    
    from .gpu_measurement_equations import GPUMultiLatentMeasurement
    from .multi_latent_structural import MultiLatentStructural
    from .choice_equations import BinaryProbitChoice
    from .multi_latent_config import MultiLatentStructuralConfig
    from .iclv_config import MeasurementConfig, ChoiceConfig
    
    ind_data = pd.DataFrame(ind_data_dict)
    
    # 설정 복원
    measurement_configs = {}
    for lv_name, lv_config_dict in config_dict['measurement'].items():
        measurement_configs[lv_name] = MeasurementConfig(**lv_config_dict)
    
    structural_config = MultiLatentStructuralConfig(**config_dict['structural'])
    choice_config = ChoiceConfig(**config_dict['choice'])
    
    # 모델 재생성 (GPU 사용)
    measurement_model = GPUMultiLatentMeasurement(measurement_configs, use_gpu=use_gpu)
    structural_model = MultiLatentStructural(structural_config)
    choice_model = BinaryProbitChoice(choice_config)
    
    # 우도 계산
    draw_lls = []
    n_exo = structural_config.n_exo
    endogenous_lv = structural_config.endogenous_lv
    
    for draw_idx in range(len(ind_draws)):
        exo_draws = ind_draws[draw_idx, :n_exo]
        endo_draw = ind_draws[draw_idx, n_exo]
        
        latent_vars = structural_model.predict(
            ind_data, exo_draws, param_dict['structural'], endo_draw
        )
        
        ll_measurement = measurement_model.log_likelihood(
            ind_data, latent_vars, param_dict['measurement']
        )
        
        lv_endo = latent_vars[endogenous_lv]
        ll_choice = 0.0
        for idx in range(len(ind_data)):
            ll_choice += choice_model.log_likelihood(
                ind_data.iloc[idx:idx+1],
                lv_endo,
                param_dict['choice']
            )
        
        ll_structural = structural_model.log_likelihood(
            ind_data, latent_vars, exo_draws, param_dict['structural'], endo_draw
        )
        
        draw_ll = ll_measurement + ll_choice + ll_structural
        
        if not np.isfinite(draw_ll):
            draw_ll = -1e10
        
        draw_lls.append(draw_ll)
    
    person_ll = logsumexp(draw_lls) - np.log(len(draw_lls))
    return (ind_id, person_ll)


class HaltonDrawGenerator:
    """Halton 시퀀스 생성기"""
    
    def __init__(self, n_individuals: int, n_draws: int, n_dimensions: int, seed: int = 42):
        self.n_individuals = n_individuals
        self.n_draws = n_draws
        self.n_dimensions = n_dimensions
        self.seed = seed
        self._draws = None
    
    def get_draws(self) -> np.ndarray:
        """Halton draws 생성 또는 반환"""
        if self._draws is None:
            self._draws = self._generate_halton_draws()
        return self._draws
    
    def _generate_halton_draws(self) -> np.ndarray:
        """Halton 시퀀스 생성"""
        from scipy.stats import qmc
        
        sampler = qmc.Halton(d=self.n_dimensions, scramble=True, seed=self.seed)
        
        draws = np.zeros((self.n_individuals, self.n_draws, self.n_dimensions))
        
        for i in range(self.n_individuals):
            uniform_draws = sampler.random(n=self.n_draws)
            from scipy.stats import norm
            draws[i] = norm.ppf(uniform_draws)
        
        return draws

