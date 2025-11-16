"""
Analytic Gradient Calculator for ICLV Models

Apollo 수준의 성능을 위한 해석적 그래디언트 계산

References:
- Train (2009) - Discrete Choice Methods with Simulation
- Ben-Akiva & Lerman (1985) - Discrete Choice Analysis
- Apollo R package - gradient calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.stats import norm
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ✅ 공통 gradient 계산 함수 import
from .gradient_core import compute_probit_gradient_common_term

logger = logging.getLogger(__name__)


def _compute_individual_gradient_parallel(args):
    """
    개인별 그래디언트 계산 (병렬처리용 전역 함수)

    Args:
        args: (ind_data_dict, ind_draws, params_dict, config_dict)

    Returns:
        개인의 weighted gradient dict
    """
    # 병렬 프로세스에서 불필요한 로그 억제
    import logging
    logging.getLogger('root').setLevel(logging.CRITICAL)

    from .measurement_equations import OrderedProbitMeasurement
    from .structural_equations import LatentVariableRegression
    from .choice_equations import BinaryProbitChoice
    from .iclv_config import MeasurementConfig, StructuralConfig, ChoiceConfig

    ind_data_dict, ind_draws, params_dict, config_dict = args

    # DataFrame 복원
    ind_data = pd.DataFrame(ind_data_dict)

    # 모델 재생성
    measurement_config = MeasurementConfig(**config_dict['measurement'])
    structural_config = StructuralConfig(**config_dict['structural'])
    choice_config = ChoiceConfig(**config_dict['choice'])

    measurement_model = OrderedProbitMeasurement(measurement_config)
    structural_model = LatentVariableRegression(structural_config)
    choice_model = BinaryProbitChoice(choice_config)

    # Gradient 계산기 재생성
    from .gradient_calculator import MeasurementGradient, StructuralGradient, ChoiceGradient

    measurement_grad = MeasurementGradient(
        n_indicators=len(config_dict['measurement']['indicators']),
        n_categories=config_dict['measurement']['n_categories']
    )
    structural_grad = StructuralGradient(
        n_sociodem=len(config_dict['structural']['sociodemographics']),
        error_variance=config_dict['structural']['error_variance']
    )
    choice_grad = ChoiceGradient(
        n_attributes=len(config_dict['choice']['choice_attributes'])
    )

    # 각 draw에 대한 likelihood와 gradient 저장
    draw_likelihoods = []
    draw_gradients = []

    for j, draw in enumerate(ind_draws):
        # LV 예측
        lv = structural_model.predict(ind_data, params_dict['structural'], draw)

        # 각 모델의 log-likelihood 계산
        ll_measurement = measurement_model.log_likelihood(
            ind_data, lv, params_dict['measurement']
        )

        # Panel Product
        choice_set_lls = []
        for idx in range(len(ind_data)):
            ll_choice_t = choice_model.log_likelihood(
                ind_data.iloc[idx:idx+1], lv, params_dict['choice']
            )
            choice_set_lls.append(ll_choice_t)
        ll_choice = sum(choice_set_lls)

        ll_structural = structural_model.log_likelihood(
            ind_data, lv, params_dict['structural'], draw
        )

        # 결합 log-likelihood
        joint_ll = ll_measurement + ll_choice + ll_structural

        # Likelihood (not log)
        likelihood = np.exp(joint_ll) if np.isfinite(joint_ll) else 1e-100
        draw_likelihoods.append(likelihood)

        # 각 모델의 gradient 계산
        grad_meas = measurement_grad.compute_gradient(
            ind_data, lv, params_dict['measurement'],
            config_dict['measurement']['indicators']
        )
        grad_struct = structural_grad.compute_gradient(
            ind_data, lv, params_dict['structural'],
            config_dict['structural']['sociodemographics']
        )
        grad_choice = choice_grad.compute_gradient(
            ind_data, lv, params_dict['choice'],
            config_dict['choice']['choice_attributes']
        )

        # Gradient 저장
        draw_gradients.append({
            'zeta': grad_meas['grad_zeta'],
            'tau': grad_meas['grad_tau'],
            'gamma': grad_struct['grad_gamma'],
            'intercept': grad_choice['grad_intercept'],
            'beta': grad_choice['grad_beta'],
            'lambda': grad_choice['grad_lambda']
        })

    # Importance weights 계산 (Apollo 방식)
    total_likelihood = sum(draw_likelihoods)
    if total_likelihood > 0:
        weights = np.array(draw_likelihoods) / total_likelihood
    else:
        weights = np.ones(len(draw_likelihoods)) / len(draw_likelihoods)

    # Weighted average of gradients
    n_indicators = len(config_dict['measurement']['indicators'])
    n_thresholds = config_dict['measurement']['n_categories'] - 1
    n_sociodem = len(config_dict['structural']['sociodemographics'])
    n_attributes = len(config_dict['choice']['choice_attributes'])

    weighted_grad = {
        'zeta': np.zeros(n_indicators),
        'tau': np.zeros((n_indicators, n_thresholds)),
        'gamma': np.zeros(n_sociodem),
        'intercept': 0.0,
        'beta': np.zeros(n_attributes),
        'lambda': 0.0
    }

    for w, grad in zip(weights, draw_gradients):
        weighted_grad['zeta'] += w * grad['zeta']
        weighted_grad['tau'] += w * grad['tau']
        weighted_grad['gamma'] += w * grad['gamma']
        weighted_grad['intercept'] += w * grad['intercept']
        weighted_grad['beta'] += w * grad['beta']
        weighted_grad['lambda'] += w * grad['lambda']

    return weighted_grad


class MeasurementGradient:
    """
    측정모델 (Ordered Probit) 그래디언트 계산
    
    Model: P(Y=k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
    
    Parameters:
    - ζ (zeta): 요인적재량 (factor loadings)
    - τ (tau): 임계값 (thresholds)
    """
    
    def __init__(self, n_indicators: int, n_categories: int = 5):
        """
        Args:
            n_indicators: 지표 개수
            n_categories: 범주 개수 (5점 척도 = 5)
        """
        self.n_indicators = n_indicators
        self.n_categories = n_categories
        self.n_thresholds = n_categories - 1  # 5점 척도 → 4개 임계값
    
    def compute_gradient(self, data: pd.DataFrame, lv: float, 
                        params: Dict[str, np.ndarray],
                        indicators: list) -> Dict[str, np.ndarray]:
        """
        측정모델 그래디언트 계산
        
        ∂ log L / ∂ζ_i = (φ(τ_k - ζ*LV) - φ(τ_{k-1} - ζ*LV)) / P(Y=k) * (-LV)
        ∂ log L / ∂τ_k = φ(τ_k - ζ*LV) / P(Y=k)
        
        Args:
            data: 관측 데이터
            lv: 잠재변수 값
            params: {'zeta': np.ndarray, 'tau': np.ndarray}
            indicators: 지표 변수명 리스트
            
        Returns:
            {'grad_zeta': np.ndarray, 'grad_tau': np.ndarray}
        """
        zeta = params['zeta']
        tau = params['tau']
        
        grad_zeta = np.zeros(self.n_indicators)
        grad_tau = np.zeros((self.n_indicators, self.n_thresholds))
        
        # 첫 번째 행만 사용 (측정모델은 개인 특성)
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
            elif k == self.n_categories - 1:
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
            elif k == self.n_categories - 1:
                grad_tau[i, -1] = -phi_lower / prob
            else:
                grad_tau[i, k-1] = -phi_lower / prob
                grad_tau[i, k] = phi_upper / prob
        
        return {
            'grad_zeta': grad_zeta,
            'grad_tau': grad_tau
        }


class StructuralGradient:
    """
    구조모델 (Linear Regression) 그래디언트 계산
    
    Model: LV ~ N(γ*X, σ²)
    
    Parameters:
    - γ (gamma): 회귀계수
    """
    
    def __init__(self, n_sociodem: int, error_variance: float = 1.0):
        """
        Args:
            n_sociodem: 사회인구학적 변수 개수
            error_variance: 오차 분산
        """
        self.n_sociodem = n_sociodem
        self.error_variance = error_variance
    
    def compute_gradient(self, data: pd.DataFrame, lv: float,
                        params: Dict[str, np.ndarray],
                        sociodemographics: list) -> Dict[str, np.ndarray]:
        """
        구조모델 그래디언트 계산
        
        log L = -0.5 * log(2πσ²) - 0.5 * (LV - γ*X)² / σ²
        
        ∂ log L / ∂γ_j = (LV - γ*X) / σ² * X_j
        
        Args:
            data: 사회인구학적 데이터
            lv: 잠재변수 값
            params: {'gamma': np.ndarray}
            sociodemographics: 사회인구학적 변수명 리스트
            
        Returns:
            {'grad_gamma': np.ndarray}
        """
        gamma = params['gamma']
        
        # 첫 번째 행만 사용
        first_row = data.iloc[0]
        
        # X 벡터 구성
        X = np.zeros(self.n_sociodem)
        for j, var in enumerate(sociodemographics):
            if var in first_row.index:
                value = first_row[var]
                if not pd.isna(value):
                    X[j] = value
        
        # 평균 계산
        lv_mean = np.dot(gamma, X)
        
        # 그래디언트
        residual = lv - lv_mean
        grad_gamma = residual / self.error_variance * X
        
        return {'grad_gamma': grad_gamma}


class ChoiceGradient:
    """
    선택모델 (Binary Probit) 그래디언트 계산
    
    Model: P(choice=1) = Φ(V), V = β*X + λ*LV
    
    Parameters:
    - β (beta): 선택 속성 계수
    - λ (lambda): 잠재변수 계수
    - intercept: 절편
    """
    
    def __init__(self, n_attributes: int):
        """
        Args:
            n_attributes: 선택 속성 개수
        """
        self.n_attributes = n_attributes
    
    def compute_gradient(self, data: pd.DataFrame, lv,
                        params: Dict[str, np.ndarray],
                        choice_attributes: list) -> Dict[str, np.ndarray]:
        """
        선택모델 그래디언트 계산 (Panel Product 포함)

        ✅ 모든 LV 주효과 지원 (디폴트)

        모든 LV 주효과 모델:
            V = β*X + Σ(λ_i * LV_i)

        조절효과 모델:
            V = β*X + λ_main*LV_main + Σ(λ_mod_i * LV_main * LV_mod_i)

        기본 모델:
            V = β*X + λ*LV

        Args:
            data: 선택 데이터 (여러 선택 상황)
            lv: 잠재변수 값
                - 모든 LV 주효과: Dict[str, float]
                - 조절효과 모델: Dict[str, float]
                - 기본 모델: float
            params:
                - 모든 LV 주효과: {'intercept': float, 'beta': np.ndarray, 'lambda_{lv_name}': float, ...}
                - 조절효과 모델: {'intercept': float, 'beta': np.ndarray, 'lambda_main': float, 'lambda_mod_{lv}': float, ...}
                - 기본 모델: {'intercept': float, 'beta': np.ndarray, 'lambda': float}
            choice_attributes: 선택 속성 변수명 리스트

        Returns:
            - 모든 LV 주효과: {'grad_intercept': float, 'grad_beta': np.ndarray, 'grad_lambda_{lv_name}': float, ...}
            - 조절효과 모델: {'grad_intercept': float, 'grad_beta': np.ndarray, 'grad_lambda_main': float, 'grad_lambda_mod_{lv}': float, ...}
            - 기본 모델: {'grad_intercept': float, 'grad_beta': np.ndarray, 'grad_lambda': float}
        """
        intercept = params['intercept']
        beta = params['beta']

        # ✅ 모든 LV 주효과 여부 확인 (디폴트)
        lambda_lv_keys = [key for key in params.keys() if key.startswith('lambda_') and key not in ['lambda_main']]

        if len(lambda_lv_keys) > 1:
            # 모든 LV 주효과 모델: lambda_{lv_name}
            if not isinstance(lv, dict):
                raise ValueError("모든 LV 주효과 모델에서는 lv가 딕셔너리여야 합니다")

            grad_intercept = 0.0
            grad_beta = np.zeros(self.n_attributes)
            grad_lambda = {lv_name: 0.0 for lv_name in lv.keys()}

            # Panel Product: 모든 선택 상황에 대해 합산
            for idx in range(len(data)):
                row = data.iloc[idx]

                # 선택 속성
                X = np.array([row[attr] for attr in choice_attributes])

                # NaN 처리 (opt-out)
                if np.isnan(X).any():
                    continue

                # 선택 결과
                choice = row['choice']

                # 효용 계산: V = intercept + β*X + Σ(λ_i * LV_i)
                V = intercept + np.dot(beta, X)
                for lv_name in lv.keys():
                    lambda_lv = params[f'lambda_{lv_name}']
                    V += lambda_lv * lv[lv_name]

                # ✅ 공통 함수 사용: Binary Probit gradient 공통항 계산
                common_term = compute_probit_gradient_common_term(
                    choice=choice,
                    utility=V
                )

                # 각 파라미터의 gradient = common_term × 미분항
                grad_intercept += common_term  # ∂V/∂intercept = 1
                grad_beta += common_term * X  # ∂V/∂β = X

                # 각 LV의 gradient: ∂V/∂λ_i = LV_i
                for lv_name in lv.keys():
                    grad_lambda[lv_name] += common_term * lv[lv_name]

            # 결과 반환
            result = {
                'grad_intercept': grad_intercept,
                'grad_beta': grad_beta
            }
            for lv_name in lv.keys():
                result[f'grad_lambda_{lv_name}'] = grad_lambda[lv_name]

            return result

        # ✅ 조절효과 여부 확인
        elif 'lambda_main' in params:
            # 조절효과 모델
            lambda_main = params['lambda_main']

            # LV가 딕셔너리인지 확인 (다중 LV)
            if not isinstance(lv, dict):
                raise ValueError("조절효과 모델에서는 lv가 딕셔너리여야 합니다")

            # 조절 LV 추출
            moderator_lvs = [key.replace('lambda_mod_', '') for key in params.keys() if key.startswith('lambda_mod_')]

            # Main LV 찾기 (나머지 LV 중 하나)
            main_lv_name = None
            for lv_name in lv.keys():
                if lv_name not in moderator_lvs:
                    main_lv_name = lv_name
                    break

            if main_lv_name is None:
                raise ValueError(f"Main LV를 찾을 수 없습니다. lv keys: {lv.keys()}, moderators: {moderator_lvs}")

            lv_main = lv[main_lv_name]

            grad_intercept = 0.0
            grad_beta = np.zeros(self.n_attributes)
            grad_lambda_main = 0.0
            grad_lambda_mod = {mod_lv: 0.0 for mod_lv in moderator_lvs}

            # Panel Product: 모든 선택 상황에 대해 합산
            for idx in range(len(data)):
                row = data.iloc[idx]

                # 선택 속성
                X = np.array([row[attr] for attr in choice_attributes])

                # NaN 처리 (opt-out)
                if np.isnan(X).any():
                    continue

                # 선택 결과
                choice = row['choice']

                # 조절효과 계산
                moderation_term = 0.0
                for mod_lv_name in moderator_lvs:
                    lambda_mod = params[f'lambda_mod_{mod_lv_name}']
                    lv_mod = lv[mod_lv_name]
                    moderation_term += lambda_mod * lv_main * lv_mod

                # 효용
                V = intercept + np.dot(beta, X) + lambda_main * lv_main + moderation_term

                # ✅ 공통 함수 사용: Binary Probit gradient 공통항 계산
                common_term = compute_probit_gradient_common_term(
                    choice=choice,
                    utility=V
                )

                # 각 파라미터의 gradient = common_term × 미분항
                grad_intercept += common_term  # ∂V/∂intercept = 1
                grad_beta += common_term * X  # ∂V/∂β = X
                grad_lambda_main += common_term * lv_main  # ∂V/∂λ_main = LV_main

                # 조절효과 gradient: ∂V/∂λ_mod = LV_main × LV_mod
                for mod_lv_name in moderator_lvs:
                    lv_mod = lv[mod_lv_name]
                    grad_lambda_mod[mod_lv_name] += common_term * lv_main * lv_mod

            # 결과 반환
            result = {
                'grad_intercept': grad_intercept,
                'grad_beta': grad_beta,
                'grad_lambda_main': grad_lambda_main
            }
            for mod_lv_name in moderator_lvs:
                result[f'grad_lambda_mod_{mod_lv_name}'] = grad_lambda_mod[mod_lv_name]

            return result

        else:
            # 기본 모델
            lambda_lv = params['lambda']

            # LV가 float인지 확인
            if isinstance(lv, dict):
                # 딕셔너리인 경우 첫 번째 값 사용
                lv = list(lv.values())[0]

            grad_intercept = 0.0
            grad_beta = np.zeros(self.n_attributes)
            grad_lambda = 0.0

            # Panel Product: 모든 선택 상황에 대해 합산
            for idx in range(len(data)):
                row = data.iloc[idx]

                # 선택 속성
                X = np.array([row[attr] for attr in choice_attributes])

                # NaN 처리 (opt-out)
                if np.isnan(X).any():
                    continue

                # 선택 결과
                choice = row['choice']

                # 효용
                V = intercept + np.dot(beta, X) + lambda_lv * lv

                # ✅ 공통 함수 사용: Binary Probit gradient 공통항 계산
                common_term = compute_probit_gradient_common_term(
                    choice=choice,
                    utility=V
                )

                # 각 파라미터의 gradient = common_term × 미분항
                grad_intercept += common_term  # ∂V/∂intercept = 1
                grad_beta += common_term * X  # ∂V/∂β = X
                grad_lambda += common_term * lv  # ∂V/∂λ = LV

            return {
                'grad_intercept': grad_intercept,
                'grad_beta': grad_beta,
                'grad_lambda': grad_lambda
            }


class JointGradient:
    """
    결합 로그우도 그래디언트 계산 (Apollo 방식)

    Joint LL = Σ_i log[(1/R) Σ_r P(Choice|LV_r) * P(Indicators|LV_r) * P(LV_r|X)]

    Apollo의 analytic gradient 계산 방식:
    1. 각 모델의 gradient를 개별적으로 계산
    2. Chain rule을 사용하여 결합
    3. 시뮬레이션 draws에 대해 평균

    Reference: Apollo R package - apollo_estimate.R, apollo_gradient.R
    """

    def __init__(self, measurement_grad,
                 structural_grad: StructuralGradient,
                 choice_grad: ChoiceGradient):
        """
        Args:
            measurement_grad: 측정모델 그래디언트 계산기
                - 단일 LV: MeasurementGradient 객체
                - 다중 LV: Dict[str, MeasurementGradient]
            structural_grad: 구조모델 그래디언트 계산기
            choice_grad: 선택모델 그래디언트 계산기
        """
        self.measurement_grad = measurement_grad
        self.structural_grad = structural_grad
        self.choice_grad = choice_grad

        # 다중 잠재변수 여부 확인
        self.is_multi_latent = isinstance(measurement_grad, dict)

    def compute_gradient(self, data: pd.DataFrame, params_dict: Dict,
                        draws: np.ndarray, individual_id_column: str,
                        measurement_model, structural_model, choice_model,
                        indicators: list, sociodemographics: list,
                        choice_attributes: list,
                        use_parallel: bool = False,
                        n_cores: int = None) -> Dict[str, np.ndarray]:
        """
        결합 그래디언트 계산 (Apollo 방식)

        ∂ log L / ∂θ = Σ_i [Σ_r w_ir * (∂ log P_measurement / ∂θ +
                                          ∂ log P_choice / ∂θ +
                                          ∂ log P_structural / ∂θ)]

        where w_ir = P_ir / Σ_r P_ir (importance weight)

        Args:
            data: 전체 데이터
            params_dict: 파라미터 딕셔너리
            draws: Halton draws (n_individuals, n_draws)
            individual_id_column: 개인 ID 열 이름
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            indicators: 지표 변수명 리스트
            sociodemographics: 사회인구학적 변수명 리스트
            choice_attributes: 선택 속성 변수명 리스트
            use_parallel: 병렬처리 사용 여부
            n_cores: 사용할 코어 수 (None이면 자동 감지)

        Returns:
            gradient_dict: 각 파라미터 그룹별 그래디언트
        """
        individual_ids = data[individual_id_column].unique()
        n_individuals = len(individual_ids)

        # 그래디언트 초기화
        n_zeta = len(indicators)
        n_tau = len(indicators) * (measurement_model.n_categories - 1)
        n_gamma = len(sociodemographics)
        n_beta = len(choice_attributes)

        grad_zeta_total = np.zeros(n_zeta)
        grad_tau_total = np.zeros((len(indicators), measurement_model.n_categories - 1))
        grad_gamma_total = np.zeros(n_gamma)
        grad_intercept_total = 0.0
        grad_beta_total = np.zeros(n_beta)
        grad_lambda_total = 0.0

        # 병렬처리 여부 확인
        if use_parallel:
            # 병렬처리 사용
            if n_cores is None:
                n_cores = max(1, multiprocessing.cpu_count() - 1)

            # 설정 정보를 dict로 변환 (pickle 가능)
            config_dict = {
                'measurement': {
                    'latent_variable': 'health_concern',  # 하드코딩 (config에서 가져올 수 없음)
                    'indicators': indicators,
                    'n_categories': measurement_model.n_categories
                },
                'structural': {
                    'sociodemographics': sociodemographics,
                    'error_variance': structural_model.error_variance
                },
                'choice': {
                    'choice_attributes': choice_attributes
                }
            }

            # 개인별 데이터 준비
            args_list = []
            for i, ind_id in enumerate(individual_ids):
                ind_data = data[data[individual_id_column] == ind_id]
                ind_data_dict = ind_data.to_dict('list')
                ind_draws = draws[i, :]
                args_list.append((ind_data_dict, ind_draws, params_dict, config_dict))

            # 병렬 계산
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                individual_grads = list(executor.map(_compute_individual_gradient_parallel, args_list))

            # 개인별 gradient 합산
            for weighted_grad in individual_grads:
                grad_zeta_total += weighted_grad['zeta']
                grad_tau_total += weighted_grad['tau']
                grad_gamma_total += weighted_grad['gamma']
                grad_intercept_total += weighted_grad['intercept']
                grad_beta_total += weighted_grad['beta']
                grad_lambda_total += weighted_grad['lambda']

        else:
            # 순차처리 (기존 코드)
            # 개인별 그래디언트 계산
            for i, ind_id in enumerate(individual_ids):
                ind_data = data[data[individual_id_column] == ind_id]
                ind_draws = draws[i, :]

                # 각 draw에 대한 likelihood와 gradient 저장
                draw_likelihoods = []
                draw_gradients = []

                for j, draw in enumerate(ind_draws):
                    # LV 예측
                    lv = structural_model.predict(ind_data, params_dict['structural'], draw)

                    # 각 모델의 log-likelihood 계산
                    ll_measurement = measurement_model.log_likelihood(
                        ind_data, lv, params_dict['measurement']
                    )

                    # Panel Product
                    choice_set_lls = []
                    for idx in range(len(ind_data)):
                        ll_choice_t = choice_model.log_likelihood(
                            ind_data.iloc[idx:idx+1], lv, params_dict['choice']
                        )
                        choice_set_lls.append(ll_choice_t)
                    ll_choice = sum(choice_set_lls)

                    ll_structural = structural_model.log_likelihood(
                        ind_data, lv, params_dict['structural'], draw
                    )

                    # 결합 log-likelihood
                    joint_ll = ll_measurement + ll_choice + ll_structural

                    # Likelihood (not log)
                    likelihood = np.exp(joint_ll) if np.isfinite(joint_ll) else 1e-100
                    draw_likelihoods.append(likelihood)

                    # 각 모델의 gradient 계산
                    grad_meas = self.measurement_grad.compute_gradient(
                        ind_data, lv, params_dict['measurement'], indicators
                    )
                    grad_struct = self.structural_grad.compute_gradient(
                        ind_data, lv, params_dict['structural'], sociodemographics
                    )
                    grad_choice = self.choice_grad.compute_gradient(
                        ind_data, lv, params_dict['choice'], choice_attributes
                    )

                    # Gradient 저장
                    draw_gradients.append({
                        'zeta': grad_meas['grad_zeta'],
                        'tau': grad_meas['grad_tau'],
                        'gamma': grad_struct['grad_gamma'],
                        'intercept': grad_choice['grad_intercept'],
                        'beta': grad_choice['grad_beta'],
                        'lambda': grad_choice['grad_lambda']
                    })

                # Importance weights 계산 (Apollo 방식)
                total_likelihood = sum(draw_likelihoods)
                if total_likelihood > 0:
                    weights = np.array(draw_likelihoods) / total_likelihood
                else:
                    weights = np.ones(len(draw_likelihoods)) / len(draw_likelihoods)

                # Weighted average of gradients
                for j, grad in enumerate(draw_gradients):
                    w = weights[j]
                    grad_zeta_total += w * grad['zeta']
                    grad_tau_total += w * grad['tau']
                    grad_gamma_total += w * grad['gamma']
                    grad_intercept_total += w * grad['intercept']
                    grad_beta_total += w * grad['beta']
                    grad_lambda_total += w * grad['lambda']

        return {
            'grad_zeta': grad_zeta_total,
            'grad_tau': grad_tau_total,
            'grad_gamma': grad_gamma_total,
            'grad_intercept': grad_intercept_total,
            'grad_beta': grad_beta_total,
            'grad_lambda': grad_lambda_total
        }

