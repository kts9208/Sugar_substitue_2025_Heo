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
        
        ∂ log L / ∂ζ_i = (φ(τ_k - ζ*LV) - φ(τ_{k-1} - ζ*LV)) / P(Y=k) * (-LV)
        ∂ log L / ∂τ_k = φ(τ_k - ζ*LV) / P(Y=k)
        """
        zeta = params['zeta']
        tau = params['tau']
        
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
            
            # P(Y=k) 계산
            if k == 0:
                prob = norm.cdf(tau_i[0] - V)
                phi_upper = norm.pdf(tau_i[0] - V)
                phi_lower = 0.0
            elif k == n_cat - 1:
                prob = 1 - norm.cdf(tau_i[-1] - V)
                phi_upper = 0.0
                phi_lower = norm.pdf(tau_i[-1] - V)
            else:
                prob = norm.cdf(tau_i[k] - V) - norm.cdf(tau_i[k-1] - V)
                phi_upper = norm.pdf(tau_i[k] - V)
                phi_lower = norm.pdf(tau_i[k-1] - V)
            
            # 수치 안정성
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            
            # ∂ log L / ∂ζ_i
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
                        exogenous_lvs: List[str]) -> Dict[str, np.ndarray]:
        """
        다중 잠재변수 구조모델 그래디언트 계산
        
        Args:
            data: 개인 데이터
            latent_vars: 모든 잠재변수 값 {lv_name: value}
            exo_draws: 외생 LV draws (n_exo,)
            params: {'gamma_lv': ..., 'gamma_x': ...}
            covariates: 공변량 변수명 리스트
            endogenous_lv: 내생 LV 이름
            exogenous_lvs: 외생 LV 이름 리스트
        
        Returns:
            {'grad_gamma_lv': ..., 'grad_gamma_x': ...}
        """
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
        
        # 잔차
        residual = lv_endo - lv_endo_mean
        
        # 그래디언트
        # ∂ log L / ∂γ_lv_i = (LV_endo - μ_endo) / σ² * LV_i
        grad_gamma_lv = residual / self.error_variance * exo_draws
        
        # ∂ log L / ∂γ_x_j = (LV_endo - μ_endo) / σ² * X_j
        grad_gamma_x = residual / self.error_variance * X
        
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
                 gpu_measurement_model = None):
        """
        Args:
            measurement_grad: 다중 LV 측정모델 그래디언트 계산기
            structural_grad: 다중 LV 구조모델 그래디언트 계산기
            choice_grad: 선택모델 그래디언트 계산기
            use_gpu: GPU 배치 그래디언트 사용 여부
            gpu_measurement_model: GPU 측정모델 (use_gpu=True일 때 필요)
        """
        self.measurement_grad = measurement_grad
        self.structural_grad = structural_grad
        self.choice_grad = choice_grad
        self.use_gpu = use_gpu
        self.gpu_measurement_model = gpu_measurement_model

        if self.use_gpu:
            try:
                from . import gpu_gradient_batch
                self.gpu_grad = gpu_gradient_batch
                logger.info("GPU 배치 그래디언트 활성화")
            except ImportError:
                logger.warning("GPU 그래디언트 모듈을 불러올 수 없습니다. CPU 모드로 전환.")
                self.use_gpu = False
    
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
        n_exo = structural_model.n_exo

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
            grad_meas = self.measurement_grad.compute_gradient(
                ind_data, latent_vars, params_dict['measurement']
            )
            
            grad_struct = self.structural_grad.compute_gradient(
                ind_data, latent_vars, exo_draws, params_dict['structural'],
                structural_model.covariates, structural_model.endogenous_lv,
                structural_model.exogenous_lvs
            )
            
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
        """
        # 초기화 (첫 번째 draw의 구조를 사용)
        first_grad = draw_gradients[0]
        
        # 측정모델 그래디언트 초기화
        weighted_meas = {}
        for lv_name in first_grad['measurement'].keys():
            weighted_meas[lv_name] = {
                'grad_zeta': np.zeros_like(first_grad['measurement'][lv_name]['grad_zeta']),
                'grad_tau': np.zeros_like(first_grad['measurement'][lv_name]['grad_tau'])
            }
        
        # 구조모델 그래디언트 초기화
        weighted_struct = {
            'grad_gamma_lv': np.zeros_like(first_grad['structural']['grad_gamma_lv']),
            'grad_gamma_x': np.zeros_like(first_grad['structural']['grad_gamma_x'])
        }
        
        # 선택모델 그래디언트 초기화
        weighted_choice = {
            'grad_intercept': 0.0,
            'grad_beta': np.zeros_like(first_grad['choice']['grad_beta']),
            'grad_lambda': 0.0
        }
        
        # 가중합 계산
        for w, grad in zip(weights, draw_gradients):
            # 측정모델
            for lv_name in grad['measurement'].keys():
                weighted_meas[lv_name]['grad_zeta'] += w * grad['measurement'][lv_name]['grad_zeta']
                weighted_meas[lv_name]['grad_tau'] += w * grad['measurement'][lv_name]['grad_tau']
            
            # 구조모델
            weighted_struct['grad_gamma_lv'] += w * grad['structural']['grad_gamma_lv']
            weighted_struct['grad_gamma_x'] += w * grad['structural']['grad_gamma_x']
            
            # 선택모델
            weighted_choice['grad_intercept'] += w * grad['choice']['grad_intercept']
            weighted_choice['grad_beta'] += w * grad['choice']['grad_beta']
            weighted_choice['grad_lambda'] += w * grad['choice']['grad_lambda']
        
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

                latent_vars = structural_model.predict(
                    ind_data, first_order_draws, params_dict['structural'], error_dict
                )
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
                params_dict['structural'],
                structural_model.covariates,
                structural_model.endogenous_lv,
                structural_model.exogenous_lvs,
                weights,
                is_hierarchical=True,
                hierarchical_paths=structural_model.hierarchical_paths,
                iteration_logger=iteration_logger if should_log else None,
                log_level=log_level if should_log else 'MINIMAL'
            )
        else:
            grad_struct = self.gpu_grad.compute_structural_gradient_batch_gpu(
                ind_data,
                lvs_list,
                exo_draws_list,
                params_dict['structural'],
                structural_model.covariates,
                structural_model.endogenous_lv,
                structural_model.exogenous_lvs,
                weights,
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

