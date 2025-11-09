"""
GPU Batch Processing ICLV Estimator

완전한 GPU 배치 처리로 다중 잠재변수 ICLV 모델을 추정합니다.
모든 개인 × draws를 한 번에 GPU에서 처리하여 성능을 극대화합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import time
from scipy import optimize
from scipy.special import logsumexp

from .gpu_measurement_equations import GPUMultiLatentMeasurement, GPU_AVAILABLE
from .multi_latent_structural import MultiLatentStructural
from .choice_equations import BinaryProbitChoice

logger = logging.getLogger(__name__)

# GPU 사용 가능 여부
if GPU_AVAILABLE:
    import cupy as cp
    from cupyx.scipy.special import ndtr
    logger.info("✅ GPU 배치 처리 모드")
else:
    logger.warning("⚠️ CuPy 미설치 - CPU 모드로 작동")


class GPUBatchEstimator:
    """
    GPU 배치 처리 ICLV 동시추정
    
    모든 개인 × draws를 하나의 배치로 GPU에서 처리하여
    최대 성능을 달성합니다.
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
        
        # 개인 ID 목록
        self.individual_ids = data[config.individual_id_column].unique()
        self.n_individuals = len(self.individual_ids)
        
        # Halton draws 생성
        n_draws = config.estimation.n_draws
        n_dimensions = config.structural.n_exo + 1  # 외생 LV + 내생 LV 오차
        
        self.halton_generator = HaltonDrawGenerator(
            self.n_individuals, n_draws, n_dimensions
        )
        
        # 배치 크기
        self.batch_size = self.n_individuals * n_draws
        
        # 로깅
        n_measurement_params = self.measurement_model.get_n_parameters()
        n_structural_params = config.structural.n_exo + config.structural.n_cov
        n_choice_params = 1 + len(config.choice.choice_attributes) + 1
        total_params = n_measurement_params + n_structural_params + n_choice_params
        
        gpu_status = "🚀 GPU 배치" if self.use_gpu else "💻 CPU"
        logger.info("=" * 70)
        logger.info(f"{gpu_status} Estimator 초기화")
        logger.info(f"  개인 수: {self.n_individuals:,}")
        logger.info(f"  관측치 수: {len(data):,}")
        logger.info(f"  Halton draws: {n_draws}")
        logger.info(f"  배치 크기: {self.batch_size:,} (개인 × draws)")
        logger.info(f"  측정모델 파라미터: {n_measurement_params}")
        logger.info(f"  구조모델 파라미터: {n_structural_params}")
        logger.info(f"  선택모델 파라미터: {n_choice_params}")
        logger.info(f"  총 파라미터: {total_params}")
        logger.info("=" * 70)
    
    def _prepare_batch_data(self) -> Tuple[Dict, Dict]:
        """
        배치 처리를 위한 데이터 준비
        
        Returns:
            indicator_data: {lv_name: (n_individuals, n_indicators)} 지표 데이터
            choice_data: (n_individuals, n_choice_situations, n_attributes) 선택 데이터
        """
        # 각 개인의 첫 번째 행에서 지표 데이터 추출
        indicator_data = {}
        
        for lv_name, config in self.config.measurement_configs.items():
            n_indicators = len(config.indicators)
            data_array = np.zeros((self.n_individuals, n_indicators))
            
            for i, ind_id in enumerate(self.individual_ids):
                ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                first_row = ind_data.iloc[0]
                
                for j, indicator in enumerate(config.indicators):
                    if indicator in first_row.index and not pd.isna(first_row[indicator]):
                        data_array[i, j] = first_row[indicator]
                    else:
                        data_array[i, j] = 0  # NaN은 0으로 (나중에 마스킹)
            
            indicator_data[lv_name] = data_array
        
        # 선택 데이터 준비
        choice_data = self._prepare_choice_data()
        
        return indicator_data, choice_data
    
    def _prepare_choice_data(self) -> Dict:
        """선택 데이터 준비"""
        choice_data = {
            'individual_ids': [],
            'choices': [],
            'attributes': []
        }

        for ind_id in self.individual_ids:
            ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]

            # NaN이 있는 행 제거 (alternative=3인 "선택하지 않음" 옵션)
            # 선택 속성 중 하나라도 NaN이면 제외
            valid_mask = ~ind_data[self.config.choice.choice_attributes].isna().any(axis=1)
            ind_data_valid = ind_data[valid_mask]

            choice_data['individual_ids'].append(ind_id)
            choice_data['choices'].append(ind_data_valid[self.config.choice_column].values)

            # 속성 데이터
            attr_values = []
            for attr in self.config.choice.choice_attributes:
                attr_values.append(ind_data_valid[attr].values)
            choice_data['attributes'].append(np.column_stack(attr_values))

        return choice_data
    
    def _compute_batch_likelihood(self, params: np.ndarray) -> float:
        """
        배치 우도 계산 (GPU 가속)

        모든 개인 × draws를 한 번에 처리
        """
        t_start = time.time()

        # 파라미터 분해
        param_dict = self._unpack_parameters(params)
        t1 = time.time()

        # Halton draws 가져오기
        draws = self.halton_generator.get_draws()  # (n_individuals, n_draws, n_dimensions)
        t2 = time.time()

        # 데이터 준비
        indicator_data, choice_data = self._prepare_batch_data()
        t3 = time.time()

        # 배치 확장: (n_individuals, n_draws, ...)
        n_draws = draws.shape[1]
        n_exo = self.config.structural.n_exo

        # 모든 개인 × draws에 대한 잠재변수 계산
        all_latent_vars = {}  # {lv_name: (batch_size,)}

        # 배치 인덱스 생성
        batch_indices = []
        for i in range(self.n_individuals):
            for d in range(n_draws):
                batch_indices.append((i, d))

        # 구조모델: 모든 배치에 대한 잠재변수 예측
        all_latent_vars = self._compute_batch_latent_vars(
            draws, param_dict['structural'], batch_indices
        )
        t4 = time.time()

        # 측정모델 배치 데이터 준비
        measurement_batch_data = self._prepare_measurement_batch(
            indicator_data, batch_indices
        )
        t5 = time.time()

        # 측정모델 우도 (GPU 배치)
        ll_measurement_batch = self.measurement_model.log_likelihood_batch(
            measurement_batch_data,
            all_latent_vars,
            param_dict['measurement']
        )  # (batch_size,)
        t6 = time.time()

        # 선택모델 우도 (배치)
        ll_choice_batch = self._compute_choice_batch_likelihood(
            choice_data, all_latent_vars, param_dict['choice'], batch_indices
        )  # (batch_size,)
        t7 = time.time()

        # 구조모델 우도 (배치)
        ll_structural_batch = self._compute_structural_batch_likelihood(
            draws, all_latent_vars, param_dict['structural'], batch_indices
        )  # (batch_size,)
        t8 = time.time()
        
        # 전체 우도
        ll_total_batch = ll_measurement_batch + ll_choice_batch + ll_structural_batch

        # 개인별로 재구성 및 시뮬레이션 평균
        ll_total_batch = ll_total_batch.reshape(self.n_individuals, n_draws)

        # 각 개인별 로그 시뮬레이션 평균
        person_lls = logsumexp(ll_total_batch, axis=1) - np.log(n_draws)

        # 전체 로그우도
        total_ll = np.sum(person_lls)

        t_total = time.time() - t_start

        # 타이밍 로그 출력
        print(f"  [시간] 파라미터:{t1-t_start:.2f}s | Draws:{t2-t1:.2f}s | 데이터:{t3-t2:.2f}s | "
              f"잠재변수:{t4-t3:.2f}s | 측정준비:{t5-t4:.2f}s")
        print(f"  [시간] 측정우도:{t6-t5:.2f}s (GPU) | 선택우도:{t7-t6:.2f}s | 구조우도:{t8-t7:.2f}s | 총:{t_total:.2f}s")
        print(f"  [우도] LL = {total_ll:.2f} (측정:{np.sum(ll_measurement_batch):.2f}, "
              f"선택:{np.sum(ll_choice_batch):.2f}, 구조:{np.sum(ll_structural_batch):.2f})")

        return total_ll
    
    def _compute_batch_latent_vars(self, draws, structural_params, batch_indices):
        """배치 잠재변수 계산"""
        batch_size = len(batch_indices)
        n_exo = self.config.structural.n_exo
        
        # 외생 잠재변수
        latent_vars = {}
        exo_lvs = self.config.structural.exogenous_lvs
        
        for lv_idx, lv_name in enumerate(exo_lvs):
            lv_values = np.zeros(batch_size)
            for batch_idx, (ind_idx, draw_idx) in enumerate(batch_indices):
                lv_values[batch_idx] = draws[ind_idx, draw_idx, lv_idx]
            latent_vars[lv_name] = lv_values
        
        # 내생 잠재변수
        endo_lv = self.config.structural.endogenous_lv
        endo_values = np.zeros(batch_size)
        
        gamma_lv = structural_params['gamma_lv']
        gamma_x = structural_params['gamma_x']
        covariates = self.config.structural.covariates
        
        for batch_idx, (ind_idx, draw_idx) in enumerate(batch_indices):
            ind_id = self.individual_ids[ind_idx]
            ind_data = self.data[self.data[self.config.individual_id_column] == ind_id].iloc[0]
            
            # 외생 LV 효과
            endo_mean = 0.0
            for lv_idx, lv_name in enumerate(exo_lvs):
                endo_mean += gamma_lv[lv_idx] * draws[ind_idx, draw_idx, lv_idx]
            
            # 공변량 효과
            for cov_idx, cov_name in enumerate(covariates):
                if cov_name in ind_data.index:
                    endo_mean += gamma_x[cov_idx] * ind_data[cov_name]
            
            # 오차항 추가
            endo_error = draws[ind_idx, draw_idx, n_exo]
            endo_values[batch_idx] = endo_mean + endo_error
        
        latent_vars[endo_lv] = endo_values
        
        return latent_vars
    
    def _prepare_measurement_batch(self, indicator_data, batch_indices):
        """측정모델 배치 데이터 준비"""
        batch_size = len(batch_indices)
        measurement_batch = {}
        
        for lv_name, data_array in indicator_data.items():
            n_indicators = data_array.shape[1]
            batch_array = np.zeros((batch_size, n_indicators))
            
            for batch_idx, (ind_idx, draw_idx) in enumerate(batch_indices):
                batch_array[batch_idx] = data_array[ind_idx]
            
            measurement_batch[lv_name] = batch_array
        
        return measurement_batch

    def _compute_choice_batch_likelihood(self, choice_data, latent_vars, choice_params, batch_indices):
        """
        선택모델 배치 우도 계산 (GPU 가속)

        모든 배치 × 선택 상황을 한 번에 처리
        """
        batch_size = len(batch_indices)
        beta_intercept = choice_params['intercept']
        beta = choice_params['beta']
        lambda_lv = choice_params['lambda']
        endo_lv = self.config.structural.endogenous_lv

        # GPU 모드: 완전 벡터화
        if self.use_gpu:
            return self._compute_choice_batch_likelihood_gpu(
                choice_data, latent_vars, choice_params, batch_indices
            )

        # CPU 모드: 기존 루프 방식
        ll_choice = np.zeros(batch_size)
        for batch_idx, (ind_idx, draw_idx) in enumerate(batch_indices):
            choices = choice_data['choices'][ind_idx]
            attributes = choice_data['attributes'][ind_idx]
            lv_value = latent_vars[endo_lv][batch_idx]

            # 각 선택 상황에 대해
            for t in range(len(choices)):
                # 효용: V = β0 + β*X + λ*LV
                utility = beta_intercept
                utility += np.dot(beta, attributes[t])
                utility += lambda_lv * lv_value

                # Probit 확률 (안전한 계산)
                from scipy.stats import norm
                cdf_val = norm.cdf(utility)
                # 수치 안정성을 위해 클리핑
                cdf_val = np.clip(cdf_val, 1e-10, 1 - 1e-10)

                if choices[t] == 1:
                    prob = cdf_val
                else:
                    prob = 1 - cdf_val

                ll_choice[batch_idx] += np.log(prob)

        return ll_choice

    def _compute_choice_batch_likelihood_gpu(self, choice_data, latent_vars, choice_params, batch_indices):
        """
        선택모델 GPU 배치 우도 계산

        모든 개인 × draws × 선택 상황을 하나의 큰 배열로 처리
        """
        beta_intercept = choice_params['intercept']
        beta = choice_params['beta']
        lambda_lv = choice_params['lambda']
        endo_lv = self.config.structural.endogenous_lv

        # 1. 모든 선택 데이터를 하나의 배열로 수집
        all_choices = []
        all_attributes = []
        all_lv_values = []
        batch_choice_counts = []  # 각 배치의 선택 개수

        for batch_idx, (ind_idx, draw_idx) in enumerate(batch_indices):
            choices = choice_data['choices'][ind_idx]
            attributes = choice_data['attributes'][ind_idx]
            lv_value = latent_vars[endo_lv][batch_idx]

            n_choices = len(choices)
            batch_choice_counts.append(n_choices)

            all_choices.extend(choices)
            all_attributes.append(attributes)
            all_lv_values.extend([lv_value] * n_choices)

        # NumPy 배열로 변환
        all_choices = np.array(all_choices)  # (total_choices,)
        all_attributes = np.vstack(all_attributes)  # (total_choices, n_attrs)
        all_lv_values = np.array(all_lv_values)  # (total_choices,)

        # 2. GPU로 전송
        all_choices_gpu = cp.asarray(all_choices)
        all_attributes_gpu = cp.asarray(all_attributes)
        all_lv_values_gpu = cp.asarray(all_lv_values)
        beta_gpu = cp.asarray(beta)

        # 3. 효용 계산 (벡터화)
        # V = β0 + β*X + λ*LV
        utilities = beta_intercept + cp.dot(all_attributes_gpu, beta_gpu) + lambda_lv * all_lv_values_gpu

        # 4. Probit 확률 계산 (GPU)
        cdf_vals = ndtr(utilities)  # GPU에서 한 번에!
        cdf_vals = cp.clip(cdf_vals, 1e-10, 1 - 1e-10)

        # 5. 선택에 따른 확률
        probs = cp.where(all_choices_gpu == 1, cdf_vals, 1 - cdf_vals)

        # 6. 로그 확률
        log_probs = cp.log(probs)

        # 7. 각 배치별로 합산
        ll_choice = np.zeros(len(batch_indices))
        start_idx = 0
        for batch_idx, n_choices in enumerate(batch_choice_counts):
            end_idx = start_idx + n_choices
            ll_choice[batch_idx] = float(cp.sum(log_probs[start_idx:end_idx]))
            start_idx = end_idx

        return ll_choice

    def _compute_structural_batch_likelihood(self, draws, latent_vars, structural_params, batch_indices):
        """
        구조모델 배치 우도 계산 (GPU 가속)

        모든 배치를 한 번에 처리
        """
        # GPU 모드: 완전 벡터화
        if self.use_gpu:
            return self._compute_structural_batch_likelihood_gpu(
                draws, latent_vars, structural_params, batch_indices
            )

        # CPU 모드: 기존 루프 방식
        batch_size = len(batch_indices)
        n_exo = self.config.structural.n_exo
        endo_lv = self.config.structural.endogenous_lv
        exo_lvs = self.config.structural.exogenous_lvs

        gamma_lv = structural_params['gamma_lv']
        gamma_x = structural_params['gamma_x']
        covariates = self.config.structural.covariates
        error_variance = self.config.structural.error_variance

        ll_structural = np.zeros(batch_size)

        for batch_idx, (ind_idx, draw_idx) in enumerate(batch_indices):
            ind_id = self.individual_ids[ind_idx]
            ind_data = self.data[self.data[self.config.individual_id_column] == ind_id].iloc[0]

            # 내생 LV 예측값
            endo_mean = 0.0
            for lv_idx, lv_name in enumerate(exo_lvs):
                endo_mean += gamma_lv[lv_idx] * draws[ind_idx, draw_idx, lv_idx]

            for cov_idx, cov_name in enumerate(covariates):
                if cov_name in ind_data.index:
                    endo_mean += gamma_x[cov_idx] * ind_data[cov_name]

            # 오차항 우도
            endo_error = draws[ind_idx, draw_idx, n_exo]
            endo_actual = latent_vars[endo_lv][batch_idx]
            residual = endo_actual - endo_mean

            # 정규분포 로그우도
            ll_structural[batch_idx] = -0.5 * np.log(2 * np.pi * error_variance)
            ll_structural[batch_idx] -= 0.5 * (residual ** 2) / error_variance

        return ll_structural

    def _compute_structural_batch_likelihood_gpu(self, draws, latent_vars, structural_params, batch_indices):
        """
        구조모델 GPU 배치 우도 계산

        모든 배치의 구조모델 우도를 벡터 연산으로 한 번에 계산
        """
        batch_size = len(batch_indices)
        n_exo = self.config.structural.n_exo
        endo_lv = self.config.structural.endogenous_lv
        exo_lvs = self.config.structural.exogenous_lvs

        gamma_lv = structural_params['gamma_lv']
        gamma_x = structural_params['gamma_x']
        covariates = self.config.structural.covariates
        error_variance = self.config.structural.error_variance

        # 1. 외생 LV 기여도 계산 (배치 전체)
        # draws: (n_individuals, n_draws, n_dimensions)
        # 각 배치에 대한 외생 LV 값 추출
        exo_lv_values = np.zeros((batch_size, n_exo))
        for batch_idx, (ind_idx, draw_idx) in enumerate(batch_indices):
            exo_lv_values[batch_idx, :] = draws[ind_idx, draw_idx, :n_exo]

        # 2. 공변량 기여도 계산 (배치 전체)
        cov_values = np.zeros((batch_size, len(covariates)))
        for batch_idx, (ind_idx, draw_idx) in enumerate(batch_indices):
            ind_id = self.individual_ids[ind_idx]
            ind_data = self.data[self.data[self.config.individual_id_column] == ind_id].iloc[0]
            for cov_idx, cov_name in enumerate(covariates):
                if cov_name in ind_data.index:
                    cov_values[batch_idx, cov_idx] = ind_data[cov_name]

        # 3. GPU로 전송
        exo_lv_values_gpu = cp.asarray(exo_lv_values)  # (batch_size, n_exo)
        cov_values_gpu = cp.asarray(cov_values)  # (batch_size, n_cov)
        gamma_lv_gpu = cp.asarray(gamma_lv)  # (n_exo,)
        gamma_x_gpu = cp.asarray(gamma_x)  # (n_cov,)

        # 4. 내생 LV 예측값 계산 (벡터화)
        # endo_mean = gamma_lv @ exo_lv + gamma_x @ covariates
        endo_means = cp.dot(exo_lv_values_gpu, gamma_lv_gpu) + cp.dot(cov_values_gpu, gamma_x_gpu)

        # 5. 실제 내생 LV 값
        endo_actual = np.array([latent_vars[endo_lv][i] for i in range(batch_size)])
        endo_actual_gpu = cp.asarray(endo_actual)

        # 6. 잔차 계산
        residuals = endo_actual_gpu - endo_means

        # 7. 정규분포 로그우도 (벡터화)
        # ll = -0.5 * log(2π*σ²) - 0.5 * (residual² / σ²)
        log_const = -0.5 * cp.log(2 * cp.pi * error_variance)
        ll_structural_gpu = log_const - 0.5 * (residuals ** 2) / error_variance

        # 8. CPU로 반환
        ll_structural = cp.asnumpy(ll_structural_gpu)

        return ll_structural

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

    def estimate(self, initial_params: np.ndarray = None,
                method: str = 'BFGS', maxiter: int = 100) -> Dict:
        """
        모델 추정

        Args:
            initial_params: 초기 파라미터 (None이면 자동 생성)
            method: 최적화 방법
            maxiter: 최대 반복 횟수

        Returns:
            추정 결과 딕셔너리
        """
        if initial_params is None:
            initial_params = self._initialize_parameters()

        logger.info("=" * 70)
        logger.info("GPU 배치 추정 시작")
        logger.info(f"  초기 파라미터 수: {len(initial_params)}")
        logger.info(f"  최적화 방법: {method}")
        logger.info(f"  최대 반복: {maxiter}")
        logger.info("=" * 70)

        start_time = time.time()

        # 콜백 함수
        self.iteration = 0
        self.best_ll = -np.inf

        def callback(params):
            self.iteration += 1
            ll = self._compute_batch_likelihood(params)

            if ll > self.best_ll:
                self.best_ll = ll

            if self.iteration % 5 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  반복 {self.iteration:3d} | LL = {ll:12.2f} | "
                          f"Best = {self.best_ll:12.2f} | 시간 = {elapsed:.1f}s")

        # 목적함수 (음의 로그우도)
        def objective(params):
            print(f"\n=== 반복 {self.iteration + 1} ===")
            ll = self._compute_batch_likelihood(params)
            print(f"=== 반복 {self.iteration + 1} 완료 ===\n")
            return -ll

        # 최적화
        result = optimize.minimize(
            objective,
            initial_params,
            method=method,
            callback=callback,
            options={'maxiter': maxiter, 'disp': True}
        )

        elapsed_time = time.time() - start_time

        # 결과 정리
        final_params = result.x
        final_ll = -result.fun

        logger.info("=" * 70)
        logger.info("추정 완료!")
        logger.info(f"  최종 LL: {final_ll:.2f}")
        logger.info(f"  반복 횟수: {self.iteration}")
        logger.info(f"  소요 시간: {elapsed_time:.1f}초")
        logger.info(f"  수렴 여부: {result.success}")
        logger.info("=" * 70)

        return {
            'params': final_params,
            'log_likelihood': final_ll,
            'iterations': self.iteration,
            'time': elapsed_time,
            'success': result.success,
            'message': result.message
        }

    def _initialize_parameters(self) -> np.ndarray:
        """파라미터 초기화"""
        params_list = []

        # 측정모델
        for lv_name, model in self.measurement_model.models.items():
            init_params = model.initialize_parameters()
            params_list.append(init_params['zeta'])
            params_list.append(init_params['tau'].flatten())

        # 구조모델
        n_exo = self.config.structural.n_exo
        n_cov = self.config.structural.n_cov
        params_list.append(np.ones(n_exo) * 0.5)
        params_list.append(np.zeros(n_cov))

        # 선택모델
        params_list.append(np.array([0.0]))  # intercept
        n_choice_attrs = len(self.config.choice.choice_attributes)
        params_list.append(np.zeros(n_choice_attrs))
        params_list.append(np.array([1.0]))  # lambda

        return np.concatenate(params_list)


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

        # Halton 시퀀스 생성기
        sampler = qmc.Halton(d=self.n_dimensions, scramble=True, seed=self.seed)

        # 균등분포 샘플
        n_total = self.n_individuals * self.n_draws
        uniform_samples = sampler.random(n=n_total)

        # 표준정규분포로 변환
        from scipy.stats import norm
        normal_samples = norm.ppf(uniform_samples)

        # (n_individuals, n_draws, n_dimensions)로 재구성
        draws = normal_samples.reshape(self.n_individuals, self.n_draws, self.n_dimensions)

        logger.info(f"Halton draws 생성: {draws.shape}")

        return draws

