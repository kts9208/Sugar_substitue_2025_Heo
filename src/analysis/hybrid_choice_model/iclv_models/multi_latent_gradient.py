"""
Multi-Latent Variable Analytic Gradient Calculator

다중 잠재변수 ICLV 모델을 위한 해석적 그래디언트 계산기입니다.

구조:
- 외생 LV (4개): health_concern, perceived_benefit, perceived_price, nutrition_knowledge
- 내생 LV (1개): purchase_intention = f(외생 LV, 공변량)

Author: Sugar Substitute Research Team
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import norm
import logging

# ✅ 공통 gradient 계산 함수 import
from .gradient_core import (
    compute_score_gradient,
    compute_ordered_probit_gradient_terms,
    compute_variance_gradient
)

logger = logging.getLogger(__name__)


class MultiLatentMeasurementGradient:
    """
    다중 잠재변수 측정모델 그래디언트 계산
    
    각 잠재변수마다 독립적인 측정모델을 가지므로,
    각 LV에 대한 그래디언트를 개별적으로 계산합니다.
    """
    
    def __init__(self, measurement_configs: Dict):
        """
        Args:
            measurement_configs: {lv_name: MeasurementConfig}
        """
        self.measurement_configs = measurement_configs
        self.lv_names = list(measurement_configs.keys())
        
        # 각 LV별 지표 수와 카테고리 수
        self.n_indicators = {}
        self.n_categories = {}
        self.n_thresholds = {}
        
        for lv_name, config in measurement_configs.items():
            self.n_indicators[lv_name] = len(config.indicators)
            self.n_categories[lv_name] = config.n_categories
            self.n_thresholds[lv_name] = config.n_categories - 1
    
    def compute_gradient(self, data: pd.DataFrame, 
                        latent_vars: Dict[str, float],
                        params: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        다중 잠재변수 측정모델 그래디언트 계산
        
        각 LV에 대해 독립적으로 계산합니다.
        
        Args:
            data: 관측 데이터
            latent_vars: {lv_name: lv_value}
            params: {lv_name: {'zeta': ..., 'tau': ...}}
        
        Returns:
            {lv_name: {'grad_zeta': ..., 'grad_tau': ...}}
        """
        gradients = {}
        
        for lv_name in self.lv_names:
            lv = latent_vars[lv_name]
            lv_params = params[lv_name]
            config = self.measurement_configs[lv_name]
            
            # 단일 LV 그래디언트 계산
            grad = self._compute_single_lv_gradient(
                data, lv, lv_params, config.indicators, lv_name
            )
            
            gradients[lv_name] = grad
        
        return gradients
    
    def _compute_single_lv_gradient(self, data: pd.DataFrame, lv: float,
                                   params: Dict[str, np.ndarray],
                                   indicators: List[str],
                                   lv_name: str) -> Dict[str, np.ndarray]:
        """
        단일 잠재변수에 대한 측정모델 그래디언트

        Continuous Linear:
        - Y = ζ * LV + ε, ε ~ N(0, σ²)
        - ∂ log L / ∂ζ_i = (y_i - ζ_i*LV) / σ²_i * LV
        - ∂ log L / ∂σ²_i = -1/(2σ²_i) + (y_i - ζ_i*LV)² / (2σ⁴_i)

        Ordered Probit:
        - ∂ log L / ∂ζ_i = (φ(τ_k - ζ*LV) - φ(τ_{k-1} - ζ*LV)) / P(Y=k) * (-LV)
        - ∂ log L / ∂τ_k = φ(τ_k - ζ*LV) / P(Y=k)
        """
        zeta = params['zeta']

        # 측정 방법 확인
        config = self.measurement_configs[lv_name]
        measurement_method = getattr(config, 'measurement_method', 'ordered_probit')

        if measurement_method == 'continuous_linear':
            # Continuous Linear 방식
            sigma_sq = params['sigma_sq']
            return self._compute_continuous_linear_gradient(
                data, lv, zeta, sigma_sq, indicators
            )
        else:
            # Ordered Probit 방식 (기존)
            tau = params['tau']
            return self._compute_ordered_probit_gradient(
                data, lv, zeta, tau, indicators, lv_name
            )

    def _compute_continuous_linear_gradient(self, data: pd.DataFrame, lv: float,
                                           zeta: np.ndarray, sigma_sq: np.ndarray,
                                           indicators: List[str]) -> Dict[str, np.ndarray]:
        """
        Continuous Linear 측정모델 그래디언트

        Y = ζ * LV + ε, ε ~ N(0, σ²)

        ✅ gradient_core.compute_score_gradient() 사용
        """
        n_ind = len(indicators)
        grad_zeta = np.zeros(n_ind)
        grad_sigma_sq = np.zeros(n_ind)

        first_row = data.iloc[0]

        for i, indicator in enumerate(indicators):
            y = first_row[indicator]
            if pd.isna(y):
                continue

            zeta_i = zeta[i]
            sigma_sq_i = sigma_sq[i]

            # 예측값
            y_pred = zeta_i * lv

            # ✅ 공통 함수 사용: ∂ log L / ∂ζ_i
            grad_zeta[i] = compute_score_gradient(
                observed=y,
                predicted=y_pred,
                variance=sigma_sq_i,
                derivative_term=lv
            )

            # ✅ 공통 함수 사용: ∂ log L / ∂σ²_i
            grad_sigma_sq[i] = compute_variance_gradient(
                observed=y,
                predicted=y_pred,
                variance=sigma_sq_i
            )

        return {
            'grad_zeta': grad_zeta,
            'grad_sigma_sq': grad_sigma_sq
        }

    def _compute_ordered_probit_gradient(self, data: pd.DataFrame, lv: float,
                                        zeta: np.ndarray, tau: np.ndarray,
                                        indicators: List[str],
                                        lv_name: str) -> Dict[str, np.ndarray]:
        """
        Ordered Probit 측정모델 그래디언트

        ✅ gradient_core.compute_ordered_probit_gradient_terms() 사용
        """

        n_ind = self.n_indicators[lv_name]
        n_thresh = self.n_thresholds[lv_name]
        n_cat = self.n_categories[lv_name]

        grad_zeta = np.zeros(n_ind)
        grad_tau = np.zeros((n_ind, n_thresh))

        first_row = data.iloc[0]

        for i, indicator in enumerate(indicators):
            y = first_row[indicator]
            if pd.isna(y):
                continue

            k = int(y) - 1  # 1-5 → 0-4
            zeta_i = zeta[i]
            tau_i = tau[i]

            V = zeta_i * lv

            # ✅ 공통 함수 사용: P(Y=k), φ(lower), φ(upper) 계산
            prob, phi_lower, phi_upper = compute_ordered_probit_gradient_terms(
                observed_category=k,
                latent_value=V,
                thresholds=tau_i,
                n_categories=n_cat
            )

            # ∂ log L / ∂ζ_i = (φ_lower - φ_upper) / P(Y=k) × LV
            grad_zeta[i] = (phi_lower - phi_upper) / prob * lv

            # ∂ log L / ∂τ
            if k == 0:
                grad_tau[i, 0] = phi_upper / prob
            elif k == n_cat - 1:
                grad_tau[i, -1] = -phi_lower / prob
            else:
                grad_tau[i, k-1] = -phi_lower / prob
                grad_tau[i, k] = phi_upper / prob

        return {
            'grad_zeta': grad_zeta,
            'grad_tau': grad_tau
        }


class MultiLatentStructuralGradient:
    """
    다중 잠재변수 구조모델 그래디언트 계산
    
    구조방정식:
    - 외생 LV: LV_i ~ N(0, 1)
    - 내생 LV: LV_endo = Σ(γ_lv_i * LV_i) + Σ(γ_x_j * X_j) + η
    
    그래디언트:
    - ∂ log L / ∂γ_lv_i = (LV_endo - μ_endo) / σ² * LV_i
    - ∂ log L / ∂γ_x_j = (LV_endo - μ_endo) / σ² * X_j
    """
    
    def __init__(self, n_exo: int, n_cov: int, error_variance: float = 1.0):
        """
        Args:
            n_exo: 외생 LV 개수
            n_cov: 공변량 개수
            error_variance: 오차 분산
        """
        self.n_exo = n_exo
        self.n_cov = n_cov
        self.error_variance = error_variance
    
    def compute_gradient(self, data: pd.DataFrame,
                        latent_vars: Dict[str, float],
                        exo_draws: np.ndarray,
                        params: Dict[str, np.ndarray],
                        covariates: List[str],
                        endogenous_lv: str,
                        exogenous_lvs: List[str],
                        hierarchical_paths: List[Dict] = None) -> Dict[str, np.ndarray]:
        """
        다중 잠재변수 구조모델 그래디언트 계산

        ✅ 계층적 구조와 병렬 구조 모두 지원

        Args:
            data: 개인 데이터
            latent_vars: 모든 잠재변수 값 {lv_name: value}
            exo_draws: 외생 LV draws (n_exo,)
            params:
                - 병렬 구조: {'gamma_lv': ..., 'gamma_x': ...}
                - 계층적 구조: {'gamma_{pred}_to_{target}': ...}
            covariates: 공변량 변수명 리스트
            endogenous_lv: 내생 LV 이름
            exogenous_lvs: 외생 LV 이름 리스트
            hierarchical_paths: 계층적 경로 (None이면 병렬 구조)

        Returns:
            - 병렬 구조: {'grad_gamma_lv': ..., 'grad_gamma_x': ...}
            - 계층적 구조: {'grad_gamma_{pred}_to_{target}': ...}
        """
        # ✅ 계층적 구조 지원
        if hierarchical_paths is not None and len(hierarchical_paths) > 0:
            # 계층적 구조: 각 경로별로 gradient 계산
            gradients = {}

            for path in hierarchical_paths:
                target = path['target']
                predictors = path['predictors']

                # 현재는 단일 predictor만 지원
                if len(predictors) != 1:
                    raise ValueError(f"현재 단일 predictor만 지원합니다: {predictors}")

                predictor = predictors[0]
                param_key = f"gamma_{predictor}_to_{target}"

                # 파라미터 추출
                gamma = params[param_key]

                # LV 값 추출
                target_value = latent_vars[target]
                pred_value = latent_vars[predictor]

                # 예측값 계산: target = gamma * predictor + error
                mu = gamma * pred_value

                # ✅ 공통 함수 사용: ∂ log L / ∂γ
                grad_gamma = compute_score_gradient(
                    observed=target_value,
                    predicted=mu,
                    variance=self.error_variance,
                    derivative_term=pred_value
                )

                gradients[f'grad_{param_key}'] = grad_gamma

            return gradients

        else:
            # 병렬 구조 (기존 방식)
            gamma_lv = params['gamma_lv']
            gamma_x = params['gamma_x']

            # 내생 LV 실제값
            lv_endo = latent_vars[endogenous_lv]

            # 외생 LV 효과
            lv_effect = np.sum(gamma_lv * exo_draws)

            # 공변량 효과
            first_row = data.iloc[0]
            X = np.zeros(self.n_cov)
            for j, var in enumerate(covariates):
                if var in first_row.index:
                    value = first_row[var]
                    if not pd.isna(value):
                        X[j] = value

            x_effect = np.sum(gamma_x * X)

            # 예측 평균
            lv_endo_mean = lv_effect + x_effect

            # ✅ 공통 함수 사용: ∂ log L / ∂γ_lv_i
            grad_gamma_lv = compute_score_gradient(
                observed=lv_endo,
                predicted=lv_endo_mean,
                variance=self.error_variance,
                derivative_term=exo_draws
            )

            # ✅ 공통 함수 사용: ∂ log L / ∂γ_x_j
            grad_gamma_x = compute_score_gradient(
                observed=lv_endo,
                predicted=lv_endo_mean,
                variance=self.error_variance,
                derivative_term=X
            )

            return {
                'grad_gamma_lv': grad_gamma_lv,
                'grad_gamma_x': grad_gamma_x
            }


class MultiLatentJointGradient:
    """
    다중 잠재변수 결합 그래디언트 계산
    
    Joint LL = Σ_i log[(1/R) Σ_r P(Choice|LV_endo_r) * P(Indicators|LV_all_r) * P(LV_all_r|X)]
    
    Apollo 방식의 analytic gradient 계산:
    1. 각 모델의 gradient를 개별적으로 계산
    2. Chain rule을 사용하여 결합
    3. 시뮬레이션 draws에 대해 가중평균
    """
    
    def __init__(self, measurement_grad: MultiLatentMeasurementGradient,
                 structural_grad: MultiLatentStructuralGradient,
                 choice_grad,
                 use_gpu: bool = False,
                 gpu_measurement_model = None,
                 use_full_parallel: bool = True,
                 measurement_params_fixed: bool = False):
        """
        Args:
            measurement_grad: 다중 LV 측정모델 그래디언트 계산기
            structural_grad: 다중 LV 구조모델 그래디언트 계산기
            choice_grad: 선택모델 그래디언트 계산기
            use_gpu: GPU 배치 그래디언트 사용 여부
            gpu_measurement_model: GPU 측정모델 (use_gpu=True일 때 필요)
            use_full_parallel: 완전 병렬 처리 사용 여부 (Advanced Indexing)
            measurement_params_fixed: 측정모델 파라미터 고정 여부 (동시추정용)
        """
        self.measurement_grad = measurement_grad
        self.structural_grad = structural_grad
        self.choice_grad = choice_grad
        self.use_gpu = use_gpu
        self.gpu_measurement_model = gpu_measurement_model
        self.use_full_parallel = use_full_parallel

        # ✅ 측정모델 파라미터 고정 여부
        self.measurement_params_fixed = measurement_params_fixed

        if self.use_gpu:
            try:
                from . import gpu_gradient_batch
                self.gpu_grad = gpu_gradient_batch

                # 완전 병렬 처리 모듈 로드
                if self.use_full_parallel:
                    from . import gpu_gradient_full_parallel
                    self.gpu_grad_full = gpu_gradient_full_parallel
                    logger.info("✨ GPU 완전 병렬 그래디언트 활성화 (Advanced Indexing)")
                else:
                    logger.info("GPU 배치 그래디언트 활성화")
            except ImportError as e:
                logger.warning(f"GPU 그래디언트 모듈을 불러올 수 없습니다: {e}. CPU 모드로 전환.")
                self.use_gpu = False
                self.use_full_parallel = False
    
    def compute_gradients(self,
                         all_ind_data: list,
                         all_ind_draws: np.ndarray,
                         params_dict: Dict,
                         measurement_model,
                         structural_model,
                         choice_model,
                         iteration_logger=None,
                         log_level: str = 'MINIMAL',
                         structural_weight: float = 1.0) -> list:
        """
        🎯 단일 진입점: 모든 개인의 gradient 계산

        🔴 SIGN PROTOCOL (Level 2 - Pass-through):
        ==========================================
        This function is a pass-through dispatcher that routes to GPU or CPU implementations.

        CRITICAL RULES:
        1. This function receives POSITIVE gradients (∇LL) from lower levels
        2. This function MUST NOT change signs - it only routes
        3. The output is still POSITIVE gradients (∇LL)

        GPU/CPU 분기를 내부에서 처리하여 호출자는 모드를 신경 쓰지 않음

        Args:
            all_ind_data: 모든 개인의 데이터 리스트
            all_ind_draws: 모든 개인의 draws (N, n_draws, n_dims)
            params_dict: 파라미터 딕셔너리
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
            iteration_logger: 로거
            log_level: 로깅 레벨
            structural_weight: 구조모델 우도 스케일링 가중치 (기본값: 1.0)

        Returns:
            List[Dict]: 개인별 gradient 딕셔너리 리스트
                        Each gradient is POSITIVE (∂LL/∂param) - Ascent direction
        """
        # GPU 상태 확인
        gpu_ready = self.use_gpu and self.gpu_measurement_model is not None

        if gpu_ready:
            # GPU 모드: 완전 병렬 처리
            # 🔴 SIGN: Returns POSITIVE gradients (∇LL)
            total_loglike_gradient_per_individual = self.compute_all_individuals_gradients_full_batch(
                all_ind_data, all_ind_draws, params_dict,
                measurement_model, structural_model, choice_model,
                iteration_logger, log_level,
                structural_weight=structural_weight  # ✅ 구조모델 스케일링 전달
            )
        else:
            # CPU 모드: 순차 처리
            # 🔴 SIGN: Returns POSITIVE gradients (∇LL)
            total_loglike_gradient_per_individual = self.compute_all_individuals_gradients_batch(
                all_ind_data, all_ind_draws, params_dict,
                measurement_model, structural_model, choice_model,
                iteration_logger, log_level
            )

        # 🔴 SIGN PROTOCOL: Return POSITIVE gradients (∇LL) - Ascent direction
        return total_loglike_gradient_per_individual

    def compute_individual_gradient(self, ind_data: pd.DataFrame,
                                   ind_draws: np.ndarray,
                                   params_dict: Dict,
                                   measurement_model,
                                   structural_model,
                                   choice_model,
                                   ind_id: int = None) -> Dict:
        """
        개인별 그래디언트 계산 (다중 잠재변수)

        Args:
            ind_data: 개인 데이터
            ind_draws: 개인의 draws (n_draws, n_dimensions)
                      [외생LV1, 외생LV2, ..., 내생LV오차]
            params_dict: 파라미터 딕셔너리
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            ind_id: 개인 ID (디버깅용)

        Returns:
            개인의 가중평균 그래디언트
        """
        if self.use_gpu and self.gpu_measurement_model is not None:
            return self._compute_individual_gradient_gpu(
                ind_data, ind_draws, params_dict,
                measurement_model, structural_model, choice_model, ind_id
            )
        else:
            return self._compute_individual_gradient_cpu(
                ind_data, ind_draws, params_dict,
                measurement_model, structural_model, choice_model
            )

    def compute_all_individuals_gradients_batch(
        self,
        all_ind_data: List[pd.DataFrame],
        all_ind_draws: np.ndarray,
        params_dict: Dict,
        measurement_model,
        structural_model,
        choice_model,
        iteration_logger=None,
        log_level: str = 'MINIMAL'
    ) -> List[Dict]:
        """
        모든 개인의 gradient를 GPU batch로 동시 계산

        ✅ 완전 GPU Batch: N명의 개인을 동시에 처리

        Args:
            all_ind_data: 모든 개인의 데이터 리스트 [DataFrame_1, ..., DataFrame_N]
            all_ind_draws: 모든 개인의 draws (N, n_draws, n_dims)
            params_dict: 파라미터 딕셔너리
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
            iteration_logger: 로거
            log_level: 로깅 레벨

        Returns:
            개인별 gradient 딕셔너리 리스트 [grad_dict_1, ..., grad_dict_N]
        """
        if self.use_gpu and self.gpu_measurement_model is not None:
            # GPU batch 모드
            return self.gpu_grad.compute_all_individuals_gradients_batch_gpu(
                self.gpu_measurement_model,
                all_ind_data,
                all_ind_draws,
                params_dict,
                measurement_model,
                structural_model,
                choice_model,
                iteration_logger=iteration_logger,
                log_level=log_level
            )
        else:
            # CPU 모드 (순차 처리)
            if iteration_logger:
                iteration_logger.info("CPU 모드로 개인별 gradient 순차 계산")

            all_gradients = []
            for ind_idx, (ind_data, ind_draws) in enumerate(zip(all_ind_data, all_ind_draws)):
                ind_grad = self._compute_individual_gradient_cpu(
                    ind_data, ind_draws, params_dict,
                    measurement_model, structural_model, choice_model
                )
                all_gradients.append(ind_grad)

                # 진행 상황 로깅
                if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
                    if (ind_idx + 1) % max(1, len(all_ind_data) // 10) == 0:
                        progress = (ind_idx + 1) / len(all_ind_data) * 100
                        iteration_logger.info(f"  진행: {ind_idx + 1}/{len(all_ind_data)} ({progress:.0f}%)")

            return all_gradients

    def compute_all_individuals_gradients_full_batch(
        self,
        all_ind_data: List[pd.DataFrame],
        all_ind_draws: np.ndarray,
        params_dict: Dict,
        measurement_model,
        structural_model,
        choice_model,
        iteration_logger=None,
        log_level: str = 'MINIMAL',
        use_scaling: bool = False,  # ✅ 측정모델 우도 스케일링 사용 여부
        structural_weight: float = 1.0  # ✅ 구조모델 우도 스케일링 가중치
    ) -> List[Dict]:
        """
        모든 개인의 gradient를 완전 GPU batch로 동시 계산

        🚀 완전 병렬 처리 (Advanced Indexing):
        - use_full_parallel=True: 측정모델 38개 지표를 1번 GPU 호출로 계산 (38배 빠름)
        - use_full_parallel=False: LV별 순차, 지표별 병렬 (5번 GPU 호출)

        성능:
        - 측정모델: 1번 GPU 커널 호출 (기존 38번 → 38배 개선)
        - 메모리: 9.45 MB (Zero-padding 24.87 MB 대비 62% 절약)

        Args:
            all_ind_data: 모든 개인의 데이터 리스트 [DataFrame_1, ..., DataFrame_N]
            all_ind_draws: 모든 개인의 draws (N, n_draws, n_dims)
            params_dict: 파라미터 딕셔너리
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
            iteration_logger: 로거
            log_level: 로깅 레벨
            use_scaling: 측정모델 우도 스케일링 사용 여부 (기본값: False)
            structural_weight: 구조모델 우도 스케일링 가중치 (기본값: 1.0)

        Returns:
            개인별 gradient 딕셔너리 리스트 [grad_dict_1, ..., grad_dict_N]
        """
        if self.use_gpu and self.gpu_measurement_model is not None:
            # ✨ 완전 병렬 처리 (Advanced Indexing)
            if self.use_full_parallel and hasattr(self, 'gpu_grad_full'):
                return self.gpu_grad_full.compute_all_individuals_gradients_full_parallel_gpu(
                    self.gpu_measurement_model,
                    all_ind_data,
                    all_ind_draws,
                    params_dict,
                    measurement_model,
                    structural_model,
                    choice_model,
                    iteration_logger=iteration_logger,
                    log_level=log_level,
                    use_scaling=use_scaling,  # ✅ 측정모델 스케일링 전달
                    structural_weight=structural_weight  # ✅ 구조모델 스케일링 전달
                )
            else:
                # 기존 완전 GPU batch 모드 (LV별 순차)
                return self.gpu_grad.compute_all_individuals_gradients_full_batch_gpu(
                    self.gpu_measurement_model,
                    all_ind_data,
                    all_ind_draws,
                    params_dict,
                    measurement_model,
                    structural_model,
                    choice_model,
                    iteration_logger=iteration_logger,
                    log_level=log_level,
                    use_scaling=use_scaling,  # ✅ 측정모델 스케일링 전달
                    structural_weight=structural_weight  # ✅ 구조모델 스케일링 전달
                )
        else:
            # CPU 모드는 일반 batch로 폴백
            return self.compute_all_individuals_gradients_batch(
                all_ind_data,
                all_ind_draws,
                params_dict,
                measurement_model,
                structural_model,
                choice_model,
                iteration_logger,
                log_level
            )

    def _compute_individual_gradient_cpu(self, ind_data: pd.DataFrame,
                                        ind_draws: np.ndarray,
                                        params_dict: Dict,
                                        measurement_model,
                                        structural_model,
                                        choice_model) -> Dict:
        """
        개인별 그래디언트 계산 - CPU 버전
        """
        n_draws = len(ind_draws)
        # 외생 LV 개수 계산 (계층적 구조와 병렬 구조 모두 지원)
        if hasattr(structural_model, 'n_exo'):
            n_exo = structural_model.n_exo
        else:
            n_exo = len(structural_model.exogenous_lvs)

        # 각 draw의 likelihood와 gradient 저장
        draw_likelihoods = []
        draw_gradients = []

        for draw_idx in range(n_draws):
            # Draws 분리
            exo_draws = ind_draws[draw_idx, :n_exo]
            endo_draw = ind_draws[draw_idx, n_exo]
            
            # 모든 LV 예측
            latent_vars = structural_model.predict(
                ind_data, exo_draws, params_dict['structural'], endo_draw
            )
            
            # 각 모델의 log-likelihood 계산
            ll_measurement = measurement_model.log_likelihood(
                ind_data, latent_vars, params_dict['measurement']
            )
            
            # 선택모델 (내생 LV만 사용)
            lv_endo = latent_vars[structural_model.endogenous_lv]
            ll_choice = 0.0
            for idx in range(len(ind_data)):
                ll_choice += choice_model.log_likelihood(
                    ind_data.iloc[idx:idx+1], lv_endo, params_dict['choice']
                )
            
            ll_structural = structural_model.log_likelihood(
                ind_data, latent_vars, exo_draws, params_dict['structural'], endo_draw
            )
            
            # 결합 log-likelihood
            joint_ll = ll_measurement + ll_choice + ll_structural
            
            # Likelihood (not log)
            likelihood = np.exp(joint_ll) if np.isfinite(joint_ll) else 1e-100
            draw_likelihoods.append(likelihood)
            
            # 각 모델의 gradient 계산
            # ✅ 측정모델 파라미터 고정 시 그래디언트 계산 스킵
            if self.measurement_params_fixed:
                # 측정모델 그래디언트를 0으로 설정 (파라미터 고정)
                grad_meas = {}
                for lv_name in self.measurement_grad.lv_names:
                    config = self.measurement_grad.measurement_configs[lv_name]
                    measurement_method = getattr(config, 'measurement_method', 'ordered_probit')

                    n_ind = len(config.indicators)
                    grad_meas[lv_name] = {'grad_zeta': np.zeros(n_ind)}

                    if measurement_method == 'continuous_linear':
                        grad_meas[lv_name]['grad_sigma_sq'] = np.zeros(n_ind)
                    else:
                        n_thresh = config.n_categories - 1
                        grad_meas[lv_name]['grad_tau'] = np.zeros((n_ind, n_thresh))
            else:
                # 파라미터가 변하므로 그래디언트 계산
                grad_meas = self.measurement_grad.compute_gradient(
                    ind_data, latent_vars, params_dict['measurement']
                )
            
            # ✅ 계층적 경로 전달
            hierarchical_paths = getattr(structural_model, 'hierarchical_paths', None)

            grad_struct = self.structural_grad.compute_gradient(
                ind_data, latent_vars, exo_draws, params_dict['structural'],
                structural_model.covariates, structural_model.endogenous_lv,
                structural_model.exogenous_lvs,
                hierarchical_paths=hierarchical_paths
            )
            
            # ✅ 모든 LV 주효과 또는 조절효과 모델은 latent_vars 전체를 전달
            lambda_lv_keys = [key for key in params_dict['choice'].keys() if key.startswith('lambda_') and key not in ['lambda_main']]

            if len(lambda_lv_keys) > 1 or 'lambda_main' in params_dict['choice']:
                # 모든 LV 주효과 또는 조절효과 모델: 모든 LV 전달
                grad_choice = self.choice_grad.compute_gradient(
                    ind_data, latent_vars, params_dict['choice'],
                    choice_model.config.choice_attributes
                )
            else:
                # 기본 모델: 내생 LV만 전달
                grad_choice = self.choice_grad.compute_gradient(
                    ind_data, lv_endo, params_dict['choice'],
                    choice_model.config.choice_attributes
                )
            
            # 그래디언트 저장
            draw_gradients.append({
                'measurement': grad_meas,
                'structural': grad_struct,
                'choice': grad_choice
            })
        
        # Importance weights 계산
        total_likelihood = sum(draw_likelihoods)
        if total_likelihood == 0:
            weights = np.ones(n_draws) / n_draws
        else:
            weights = np.array(draw_likelihoods) / total_likelihood
        
        # 가중평균 그래디언트 계산
        weighted_grad = self._compute_weighted_gradient(weights, draw_gradients)
        
        return weighted_grad
    
    def _compute_weighted_gradient(self, weights: np.ndarray,
                                   draw_gradients: List[Dict]) -> Dict:
        """
        가중평균 그래디언트 계산

        ✅ continuous_linear과 ordered_probit 둘 다 지원
        """
        # 초기화 (첫 번째 draw의 구조를 사용)
        first_grad = draw_gradients[0]

        # 측정모델 그래디언트 초기화
        weighted_meas = {}
        for lv_name in first_grad['measurement'].keys():
            lv_grad = first_grad['measurement'][lv_name]
            weighted_meas[lv_name] = {
                'grad_zeta': np.zeros_like(lv_grad['grad_zeta'])
            }

            # ✅ continuous_linear: grad_sigma_sq, ordered_probit: grad_tau
            if 'grad_sigma_sq' in lv_grad:
                weighted_meas[lv_name]['grad_sigma_sq'] = np.zeros_like(lv_grad['grad_sigma_sq'])
            elif 'grad_tau' in lv_grad:
                weighted_meas[lv_name]['grad_tau'] = np.zeros_like(lv_grad['grad_tau'])

        # 구조모델 그래디언트 초기화 (✅ 계층적 vs 병렬 구조)
        if 'grad_gamma_lv' in first_grad['structural']:
            # 병렬 구조
            weighted_struct = {
                'grad_gamma_lv': np.zeros_like(first_grad['structural']['grad_gamma_lv']),
                'grad_gamma_x': np.zeros_like(first_grad['structural']['grad_gamma_x'])
            }
        else:
            # 계층적 구조
            weighted_struct = {}
            for key in first_grad['structural'].keys():
                weighted_struct[key] = 0.0

        # 선택모델 그래디언트 초기화 (✅ 조절효과 vs 기본 모델)
        weighted_choice = {
            'grad_intercept': 0.0,
            'grad_beta': np.zeros_like(first_grad['choice']['grad_beta'])
        }

        # ✅ 모든 LV 주효과 vs 조절효과 vs 기본 모델
        lambda_grad_keys = [key for key in first_grad['choice'].keys() if key.startswith('grad_lambda_')]

        if len(lambda_grad_keys) > 1 and 'grad_lambda_main' not in first_grad['choice']:
            # 모든 LV 주효과 모델: grad_lambda_{lv_name}
            for key in lambda_grad_keys:
                weighted_choice[key] = 0.0
        elif 'grad_lambda_main' in first_grad['choice']:
            # 조절효과 모델
            weighted_choice['grad_lambda_main'] = 0.0
            for key in first_grad['choice'].keys():
                if key.startswith('grad_lambda_mod_'):
                    weighted_choice[key] = 0.0
        else:
            # 기본 모델
            weighted_choice['grad_lambda'] = 0.0

        # 가중합 계산
        for w, grad in zip(weights, draw_gradients):
            # 측정모델
            for lv_name in grad['measurement'].keys():
                weighted_meas[lv_name]['grad_zeta'] += w * grad['measurement'][lv_name]['grad_zeta']

                # ✅ continuous_linear vs ordered_probit
                if 'grad_sigma_sq' in grad['measurement'][lv_name]:
                    weighted_meas[lv_name]['grad_sigma_sq'] += w * grad['measurement'][lv_name]['grad_sigma_sq']
                elif 'grad_tau' in grad['measurement'][lv_name]:
                    weighted_meas[lv_name]['grad_tau'] += w * grad['measurement'][lv_name]['grad_tau']

            # 구조모델 (✅ 계층적 vs 병렬)
            for key in grad['structural'].keys():
                weighted_struct[key] += w * grad['structural'][key]

            # 선택모델 (✅ 조절효과 vs 기본)
            weighted_choice['grad_intercept'] += w * grad['choice']['grad_intercept']
            weighted_choice['grad_beta'] += w * grad['choice']['grad_beta']

            for key in grad['choice'].keys():
                if key.startswith('grad_lambda'):
                    weighted_choice[key] += w * grad['choice'][key]

        return {
            'measurement': weighted_meas,
            'structural': weighted_struct,
            'choice': weighted_choice
        }

    def _compute_individual_gradient_gpu(self, ind_data: pd.DataFrame,
                                        ind_draws: np.ndarray,
                                        params_dict: Dict,
                                        measurement_model,
                                        structural_model,
                                        choice_model,
                                        ind_id: int = None) -> Dict:
        """
        개인별 그래디언트 계산 - GPU 배치 버전 (Importance Weighting 적용)

        CPU 구현과 동일한 로직:
        1. 각 draw의 likelihood 계산
        2. Importance weights 계산
        3. 가중평균 그래디언트 계산
        4. GPU 배치 처리로 성능 향상
        """
        n_draws = len(ind_draws)

        # 로깅 설정 가져오기
        iteration_logger = getattr(self, 'iteration_logger', None)
        log_level = 'MINIMAL'  # 기본값
        if hasattr(self, 'config') and hasattr(self.config, 'estimation'):
            log_level = getattr(self.config.estimation, 'gradient_log_level', 'MINIMAL')

        # 첫 번째 개인에 대해서만 상세 로깅
        should_log = (ind_id is not None and not hasattr(self, '_first_gradient_logged'))

        # ✅ 계층적 구조 지원
        is_hierarchical = hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical

        if is_hierarchical:
            # 계층적 구조: 1차 LV 개수
            n_first_order = len(structural_model.exogenous_lvs)
            n_higher_order = len(structural_model.get_higher_order_lvs())
        else:
            # 병렬 구조 (하위 호환)
            n_exo = structural_model.n_exo

        # 모든 draws의 LV 값 미리 계산
        lvs_list = []
        exo_draws_list = []

        for draw_idx in range(n_draws):
            if is_hierarchical:
                # 계층적 구조: 1차 LV draws + 고차 LV 오차항
                first_order_draws = ind_draws[draw_idx, :n_first_order]
                higher_order_errors = ind_draws[draw_idx, n_first_order:]

                # 고차 LV 오차항을 딕셔너리로 변환
                higher_order_lvs = structural_model.get_higher_order_lvs()
                error_dict = {lv_name: higher_order_errors[i] for i, lv_name in enumerate(higher_order_lvs)}

                # ✅ 디버깅: error_dict 확인 (첫 번째 draw만)
                if should_log and draw_idx == 0 and iteration_logger:
                    iteration_logger.info(f"[그래디언트 계산] Draw {draw_idx}:")
                    iteration_logger.info(f"  higher_order_lvs: {higher_order_lvs}")
                    iteration_logger.info(f"  higher_order_errors: {higher_order_errors}")
                    iteration_logger.info(f"  error_dict: {error_dict}")
                    # predict() 함수 내부 디버깅 활성화
                    structural_model._debug_predict = True

                # ✅ 수정: higher_order_draws를 키워드 인자로 명시적으로 전달
                latent_vars = structural_model.predict(
                    ind_data, first_order_draws, params_dict['structural'],
                    endo_draw=None, higher_order_draws=error_dict
                )

                # ✅ 디버깅: 예측된 LV 값 확인 (첫 번째 draw만)
                if should_log and draw_idx == 0 and iteration_logger:
                    iteration_logger.info(f"  예측된 LV: {latent_vars}")
                    # predict() 함수 내부 디버깅 비활성화
                    structural_model._debug_predict = False
                exo_draws_list.append(first_order_draws)
            else:
                # 병렬 구조 (하위 호환)
                exo_draws = ind_draws[draw_idx, :n_exo]
                endo_draw = ind_draws[draw_idx, n_exo]

                latent_vars = structural_model.predict(
                    ind_data, exo_draws, params_dict['structural'], endo_draw
                )
                exo_draws_list.append(exo_draws)

            lvs_list.append(latent_vars)

        # ✅ 1. 각 draw의 결합 likelihood 계산 (importance weighting용)
        ll_batch = self.gpu_grad.compute_joint_likelihood_batch_gpu(
            self.gpu_measurement_model,
            ind_data,
            lvs_list,
            ind_draws,
            params_dict,
            structural_model,
            choice_model
        )

        # ✅ 2. Importance weights 계산 (Apollo 방식)
        weights = self.gpu_grad.compute_importance_weights_gpu(ll_batch, ind_id)

        # ✅ 3. 가중평균 그래디언트 계산
        grad_meas = self.gpu_grad.compute_measurement_gradient_batch_gpu(
            self.gpu_measurement_model,
            ind_data,
            lvs_list,
            params_dict['measurement'],
            weights,  # ✅ weights 전달
            iteration_logger=iteration_logger if should_log else None,
            log_level=log_level if should_log else 'MINIMAL'
        )

        # ✅ 구조모델 gradient: 계층적 구조 지원
        if is_hierarchical:
            grad_struct = self.gpu_grad.compute_structural_gradient_batch_gpu(
                ind_data,
                lvs_list,
                exo_draws_list,
                params_dict,  # ✅ 전체 파라미터 딕셔너리 전달
                structural_model.covariates,
                structural_model.endogenous_lv,
                structural_model.exogenous_lvs,
                weights,
                is_hierarchical=True,
                hierarchical_paths=structural_model.hierarchical_paths,
                gpu_measurement_model=self.gpu_measurement_model,  # ✅ GPU 측정모델 전달
                choice_model=choice_model,  # ✅ 선택모델 전달
                iteration_logger=iteration_logger if should_log else None,
                log_level=log_level if should_log else 'MINIMAL'
            )
        else:
            grad_struct = self.gpu_grad.compute_structural_gradient_batch_gpu(
                ind_data,
                lvs_list,
                exo_draws_list,
                params_dict,  # ✅ 전체 파라미터 딕셔너리 전달
                structural_model.covariates,
                structural_model.endogenous_lv,
                structural_model.exogenous_lvs,
                weights,
                gpu_measurement_model=self.gpu_measurement_model,  # ✅ GPU 측정모델 전달
                choice_model=choice_model,  # ✅ 선택모델 전달
                iteration_logger=iteration_logger if should_log else None,
                log_level=log_level if should_log else 'MINIMAL'
            )

        # ✅ 선택모델 gradient: 조절효과 지원
        moderation_enabled = hasattr(choice_model.config, 'moderators') and choice_model.config.moderators
        if moderation_enabled:
            grad_choice = self.gpu_grad.compute_choice_gradient_batch_gpu(
                ind_data,
                lvs_list,
                params_dict['choice'],
                structural_model.endogenous_lv,
                choice_model.config.choice_attributes,
                weights,
                moderators=choice_model.config.moderators,
                iteration_logger=iteration_logger if should_log else None,
                log_level=log_level if should_log else 'MINIMAL'
            )
        else:
            grad_choice = self.gpu_grad.compute_choice_gradient_batch_gpu(
                ind_data,
                lvs_list,
                params_dict['choice'],
                structural_model.endogenous_lv,
                choice_model.config.choice_attributes,
                weights,
                iteration_logger=iteration_logger if should_log else None,
                log_level=log_level if should_log else 'MINIMAL'
            )

        # 첫 번째 그래디언트 로깅 완료 표시
        if should_log:
            self._first_gradient_logged = True

        # 결합 그래디언트
        return {
            'measurement': grad_meas,
            'structural': grad_struct,
            'choice': grad_choice
        }

