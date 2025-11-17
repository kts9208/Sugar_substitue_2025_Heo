"""
GPU 배치 그래디언트 계산

Analytic gradient를 GPU 배치로 계산하여 속도를 향상시킵니다.

CPU 구현 (multi_latent_gradient.py)의 로직을 따르면서:
1. Importance weighting 구현 (Apollo 방식)
2. 모든 선택 상황 처리
3. Likelihood 계산 후 가중평균
4. 수치 안정성 강화
5. GPU 배치 처리로 성능 향상
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupyx.scipy.special import ndtr as cp_ndtr
    CUPY_AVAILABLE = True

    # CuPy에는 norm.pdf가 없으므로 직접 구현
    def cp_norm_pdf(x):
        """표준정규분포 PDF: φ(x) = (1/√(2π)) * exp(-x²/2)"""
        return cp.exp(-0.5 * x**2) / cp.sqrt(2 * cp.pi)

    def log_sum_exp_gpu(log_values):
        """
        Log-sum-exp trick for numerical stability
        log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
        """
        max_val = cp.max(log_values)
        return max_val + cp.log(cp.sum(cp.exp(log_values - max_val)))

except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available. GPU gradient acceleration disabled.")
    cp_norm_pdf = None
    log_sum_exp_gpu = None


def compute_joint_likelihood_batch_gpu(
    gpu_measurement_model,
    ind_data: pd.DataFrame,
    lvs_list: List[Dict[str, float]],
    draws: np.ndarray,
    params_dict: Dict,
    structural_model,
    choice_model
) -> np.ndarray:
    """
    각 draw의 결합 likelihood 계산 (importance weighting용)

    기존 gpu_batch_utils의 함수들을 활용합니다.

    Args:
        gpu_measurement_model: GPU 측정모델
        ind_data: 개인 데이터
        lvs_list: 각 draw의 잠재변수 값 리스트
        draws: 개인의 draws (n_draws, n_dimensions)
        params_dict: 모든 파라미터 {'measurement': ..., 'structural': ..., 'choice': ...}
        structural_model: 구조모델 인스턴스
        choice_model: 선택모델 인스턴스

    Returns:
        각 draw의 log-likelihood 배열 (n_draws,)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    # 기존 GPU 우도 계산 함수들 import
    from . import gpu_batch_utils

    # 1. 측정모델 우도 (배치)
    ll_measurement = gpu_batch_utils.compute_measurement_batch_gpu(
        gpu_measurement_model,
        ind_data,
        lvs_list,
        params_dict['measurement']
    )

    # 2. 구조모델 우도 (배치)
    ll_structural = gpu_batch_utils.compute_structural_batch_gpu(
        ind_data,
        lvs_list,
        params_dict['structural'],
        draws,
        structural_model
    )

    # 3. 선택모델 우도 (배치)
    ll_choice = gpu_batch_utils.compute_choice_batch_gpu(
        ind_data,
        lvs_list,
        params_dict['choice'],
        choice_model
    )

    # 4. 결합 우도
    ll_joint = ll_measurement + ll_structural + ll_choice

    return ll_joint


def compute_importance_weights_gpu(ll_batch: np.ndarray, individual_id: int = None) -> np.ndarray:
    """
    Importance weights 계산 (Apollo 방식)

    w_r = L_r / Σ_s L_s = exp(ll_r) / Σ_s exp(ll_s)

    Log-sum-exp trick 사용하여 수치 안정성 확보

    Args:
        ll_batch: 각 draw의 log-likelihood (n_draws,)
        individual_id: 개인 ID (디버깅용)

    Returns:
        importance weights (n_draws,)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    ll_gpu = cp.asarray(ll_batch)

    # ✅ 디버깅: likelihood 값 확인
    ll_min = float(cp.min(ll_gpu))
    ll_max = float(cp.max(ll_gpu))
    ll_mean = float(cp.mean(ll_gpu))

    # NaN/Inf 체크 (입력 단계)
    if cp.any(cp.isnan(ll_gpu)):
        n_nan = int(cp.sum(cp.isnan(ll_gpu)))
        logger.error(f"[Individual {individual_id}] NaN detected in input LL batch ({n_nan}/{len(ll_batch)} draws)")
        logger.error(f"  LL range: [{ll_min:.2f}, {ll_max:.2f}], mean: {ll_mean:.2f}")
        logger.error(f"  First 10 LL values: {ll_batch[:10]}")
        raise ValueError(f"NaN detected in likelihood for individual {individual_id}. Cannot compute importance weights.")

    if cp.any(cp.isinf(ll_gpu)):
        n_inf = int(cp.sum(cp.isinf(ll_gpu)))
        logger.error(f"[Individual {individual_id}] Inf detected in input LL batch ({n_inf}/{len(ll_batch)} draws)")
        logger.error(f"  LL range: [{ll_min:.2f}, {ll_max:.2f}], mean: {ll_mean:.2f}")
        raise ValueError(f"Inf detected in likelihood for individual {individual_id}. Cannot compute importance weights.")

    # Log-sum-exp trick
    log_sum = log_sum_exp_gpu(ll_gpu)
    log_weights = ll_gpu - log_sum
    weights = cp.exp(log_weights)

    # 수치 안정성 체크 (출력 단계)
    if cp.any(cp.isnan(weights)):
        logger.error(f"[Individual {individual_id}] NaN detected in importance weights after exp()")
        logger.error(f"  Input LL range: [{ll_min:.2f}, {ll_max:.2f}], mean: {ll_mean:.2f}")
        logger.error(f"  log_sum: {float(log_sum)}")
        logger.error(f"  log_weights range: [{float(cp.min(log_weights)):.2f}, {float(cp.max(log_weights)):.2f}]")
        raise ValueError(f"NaN in importance weights for individual {individual_id} after normalization.")

    if cp.any(cp.isinf(weights)):
        logger.error(f"[Individual {individual_id}] Inf detected in importance weights after exp()")
        logger.error(f"  Input LL range: [{ll_min:.2f}, {ll_max:.2f}], mean: {ll_mean:.2f}")
        raise ValueError(f"Inf in importance weights for individual {individual_id} after normalization.")

    # 정규화 확인
    weight_sum = cp.sum(weights)
    if not cp.isclose(weight_sum, 1.0):
        logger.warning(f"[Individual {individual_id}] Weights sum to {float(weight_sum)}, renormalizing")
        weights = weights / weight_sum

    return cp.asnumpy(weights)


def compute_measurement_gradient_batch_gpu(
    gpu_measurement_model,
    ind_data: pd.DataFrame,
    lvs_list: List[Dict[str, float]],
    params: Dict[str, Dict],
    weights: np.ndarray,
    iteration_logger=None,
    log_level: str = 'DETAILED'
) -> Dict[str, Dict]:
    """
    측정모델 그래디언트를 GPU 배치로 계산 (가중평균 적용)

    CPU 구현 (gradient_calculator.py의 MeasurementGradient)을 따르면서:
    1. 모든 선택 상황 처리 (첫 번째 행만이 아님)
    2. Importance weighting 적용
    3. GPU 배치 처리로 성능 향상

    Args:
        gpu_measurement_model: GPU 측정모델
        ind_data: 개인 데이터 (모든 선택 상황)
        lvs_list: 각 draw의 잠재변수 값 [{lv_name: value}, ...]
        params: 측정모델 파라미터
        weights: Importance weights (n_draws,)
        iteration_logger: 로거 (optional)
        log_level: 로깅 레벨 ('MINIMAL', 'MODERATE', 'DETAILED')

    Returns:
        각 LV의 그래디언트 {lv_name: {'grad_zeta': ..., 'grad_tau': ...}} or
                          {lv_name: {'grad_zeta': ..., 'grad_sigma_sq': ...}}
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    n_draws = len(lvs_list)
    weights_gpu = cp.asarray(weights)  # (n_draws,)
    gradients = {}

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info("\n[측정모델 그래디언트 계산]")

    # 각 잠재변수별로 처리
    for lv_idx, lv_name in enumerate(params.keys()):
        zeta = params[lv_name]['zeta']
        n_indicators = len(zeta)

        # ✅ measurement_method 확인 (모델 객체에서 직접 가져오기)
        measurement_method = getattr(gpu_measurement_model.models[lv_name], 'measurement_method', 'ordered_probit')
        config = gpu_measurement_model.models[lv_name].config

        if iteration_logger and log_level == 'DETAILED' and lv_idx == 0:
            iteration_logger.info(f"  잠재변수: {lv_name}")
            iteration_logger.info(f"    - 측정 방법: {measurement_method}")
            iteration_logger.info(f"    - 지표 수: {n_indicators}")
            iteration_logger.info(f"    - zeta (처음 3개): {zeta[:min(3, len(zeta))]}")

        # LV 값들을 배열로 변환
        lv_values = np.array([lvs[lv_name] for lvs in lvs_list])
        lv_values_gpu = cp.asarray(lv_values)  # (n_draws,)
        zeta_gpu = cp.asarray(zeta)  # (n_indicators,)

        if iteration_logger and log_level == 'DETAILED' and lv_idx == 0:
            iteration_logger.info(f"    - LV 값 범위: [{float(cp.min(lv_values_gpu)):.4f}, {float(cp.max(lv_values_gpu)):.4f}]")
            iteration_logger.info(f"    - LV 값 평균: {float(cp.mean(lv_values_gpu)):.4f}")

        # 그래디언트 초기화 (각 draw별)
        grad_zeta_batch = cp.zeros((n_draws, n_indicators))

        if measurement_method == 'continuous_linear':
            # ✅ Continuous Linear 방식
            sigma_sq = params[lv_name]['sigma_sq']
            sigma_sq_gpu = cp.asarray(sigma_sq)  # (n_indicators,)
            grad_sigma_sq_batch = cp.zeros((n_draws, n_indicators))
        else:
            # Ordered Probit 방식 (기존)
            tau = params[lv_name]['tau']
            n_thresholds = tau.shape[1]
            tau_gpu = cp.asarray(tau)  # (n_indicators, n_thresholds)
            grad_tau_batch = cp.zeros((n_draws, n_indicators, n_thresholds))

        # ✅ 측정모델은 개인 수준 데이터이므로 첫 번째 행만 사용
        # (DCE long format으로 인해 동일한 값이 여러 행에 복제되어 있음)
        row = ind_data.iloc[0]

        # 지표별로 처리
        for i, indicator in enumerate(config.indicators):
            if indicator not in row.index:
                continue

            y = row[indicator]
            if pd.isna(y):
                continue

            if measurement_method == 'continuous_linear':
                # ✅ Continuous Linear: Y = ζ * LV + ε, ε ~ N(0, σ²)
                # log L = -0.5 * log(2π * σ²) - 0.5 * (y - ζ*LV)² / σ²

                # 예측값: y_pred = ζ_i * LV
                y_pred = zeta_gpu[i] * lv_values_gpu  # (n_draws,)

                # 잔차: residual = y - y_pred
                residual = y - y_pred  # (n_draws,)

                # 상세 로깅 (첫 번째 LV, 첫 번째 지표만)
                if iteration_logger and log_level == 'DETAILED' and lv_idx == 0 and i == 0:
                    iteration_logger.info(f"\n    [지표 {indicator} 계산 예시]")
                    iteration_logger.info(f"      - 관측값 y: {y:.4f}")
                    iteration_logger.info(f"      - zeta[{i}]: {float(zeta_gpu[i]):.6f}")
                    iteration_logger.info(f"      - sigma_sq[{i}]: {float(sigma_sq_gpu[i]):.6f}")
                    iteration_logger.info(f"      - LV (draw 0): {float(lv_values_gpu[0]):.4f}")
                    iteration_logger.info(f"      - 예측값 (draw 0): {float(y_pred[0]):.4f}")
                    iteration_logger.info(f"      - 잔차 (draw 0): {float(residual[0]):.4f}")

                # ∂ log L / ∂ζ_i = (y - ζ*LV) * LV / σ²
                grad_zeta_batch[:, i] = residual * lv_values_gpu / sigma_sq_gpu[i]

                # ∂ log L / ∂σ²_i = -0.5 / σ² + 0.5 * (y - ζ*LV)² / σ⁴
                grad_sigma_sq_batch[:, i] = -0.5 / sigma_sq_gpu[i] + 0.5 * (residual ** 2) / (sigma_sq_gpu[i] ** 2)

            else:
                # Ordered Probit (기존 방식)
                k = int(y) - 1  # 1-5 → 0-4

                # V = zeta_i * LV (broadcasting)
                V = zeta_gpu[i] * lv_values_gpu  # (n_draws,)

                # tau_i for this indicator
                tau_i = tau_gpu[i]  # (n_thresholds,)

                # P(Y=k) 계산
                if k == 0:
                    # P(Y=1) = Φ(τ_1 - V)
                    prob = cp_ndtr(tau_i[0] - V)
                    phi_upper = cp_norm_pdf(tau_i[0] - V)
                    phi_lower = cp.zeros_like(V)
                elif k == config.n_categories - 1:
                    # P(Y=5) = 1 - Φ(τ_4 - V)
                    prob = 1 - cp_ndtr(tau_i[-1] - V)
                    phi_upper = cp.zeros_like(V)
                    phi_lower = cp_norm_pdf(tau_i[-1] - V)
                else:
                    # P(Y=k) = Φ(τ_k - V) - Φ(τ_{k-1} - V)
                    prob = cp_ndtr(tau_i[k] - V) - cp_ndtr(tau_i[k-1] - V)
                    phi_upper = cp_norm_pdf(tau_i[k] - V)
                    phi_lower = cp_norm_pdf(tau_i[k-1] - V)

                # 수치 안정성
                prob = cp.clip(prob, 1e-10, 1 - 1e-10)

                # ∂ log L / ∂ζ_i = (φ_upper - φ_lower) / P * (-LV)
                grad_zeta_batch[:, i] = (phi_upper - phi_lower) / prob * (-lv_values_gpu)

                # ∂ log L / ∂τ_k
                if k == 0:
                    grad_tau_batch[:, i, 0] = phi_upper / prob
                elif k == config.n_categories - 1:
                    grad_tau_batch[:, i, -1] = -phi_lower / prob
                else:
                    grad_tau_batch[:, i, k] = phi_upper / prob
                    grad_tau_batch[:, i, k-1] = -phi_lower / prob

        # ✅ 수정: 가중평균 적용 (단순 합산이 아님)
        # grad_weighted = Σ_r w_r * grad_r
        grad_zeta_weighted = cp.sum(weights_gpu[:, None] * grad_zeta_batch, axis=0)

        if iteration_logger and log_level == 'DETAILED' and lv_idx == 0:
            iteration_logger.info(f"\n    [가중평균 적용 전후]")
            iteration_logger.info(f"      - grad_zeta_batch (draw 0, 처음 3개): {[float(x) for x in grad_zeta_batch[0, :min(3, n_indicators)]]}")
            iteration_logger.info(f"      - weights (처음 5개): {[float(x) for x in weights_gpu[:5]]}")
            iteration_logger.info(f"      - grad_zeta_weighted (처음 3개): {[float(x) for x in grad_zeta_weighted[:min(3, n_indicators)]]}")

        # NaN 체크
        if cp.any(cp.isnan(grad_zeta_weighted)):
            logger.warning(f"NaN detected in grad_zeta for {lv_name}")
            grad_zeta_weighted = cp.nan_to_num(grad_zeta_weighted, nan=0.0)

        if measurement_method == 'continuous_linear':
            grad_sigma_sq_weighted = cp.sum(weights_gpu[:, None] * grad_sigma_sq_batch, axis=0)

            if iteration_logger and log_level == 'DETAILED' and lv_idx == 0:
                iteration_logger.info(f"      - grad_sigma_sq_weighted (처음 3개): {[float(x) for x in grad_sigma_sq_weighted[:min(3, n_indicators)]]}")

            if cp.any(cp.isnan(grad_sigma_sq_weighted)):
                logger.warning(f"NaN detected in grad_sigma_sq for {lv_name}")
                grad_sigma_sq_weighted = cp.nan_to_num(grad_sigma_sq_weighted, nan=0.0)
        else:
            grad_tau_weighted = cp.sum(weights_gpu[:, None, None] * grad_tau_batch, axis=0)

            if cp.any(cp.isnan(grad_tau_weighted)):
                logger.warning(f"NaN detected in grad_tau for {lv_name}")
                grad_tau_weighted = cp.nan_to_num(grad_tau_weighted, nan=0.0)

        # Gradient clipping
        grad_zeta_weighted = cp.clip(grad_zeta_weighted, -1e6, 1e6)

        # ✅ fix_first_loading 고려: 첫 번째 loading이 고정되면 gradient 제외
        fix_first_loading = getattr(config, 'fix_first_loading', True)
        if fix_first_loading:
            # 첫 번째 zeta는 1.0으로 고정 (gradient 제외)
            grad_zeta_final = cp.asnumpy(grad_zeta_weighted[1:])
        else:
            grad_zeta_final = cp.asnumpy(grad_zeta_weighted)

        # GPU에서 CPU로 전송
        if measurement_method == 'continuous_linear':
            grad_sigma_sq_weighted = cp.clip(grad_sigma_sq_weighted, -1e6, 1e6)
            gradients[lv_name] = {
                'grad_zeta': grad_zeta_final,
                'grad_sigma_sq': cp.asnumpy(grad_sigma_sq_weighted)
            }

            if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
                iteration_logger.info(f"\n  [{lv_name}] 최종 그래디언트:")
                iteration_logger.info(f"    - grad_zeta: 범위=[{float(cp.min(grad_zeta_weighted)):.4f}, {float(cp.max(grad_zeta_weighted)):.4f}], norm={float(cp.linalg.norm(grad_zeta_weighted)):.4f}")
                iteration_logger.info(f"    - grad_sigma_sq: 범위=[{float(cp.min(grad_sigma_sq_weighted)):.4f}, {float(cp.max(grad_sigma_sq_weighted)):.4f}], norm={float(cp.linalg.norm(grad_sigma_sq_weighted)):.4f}")
        else:
            grad_tau_weighted = cp.clip(grad_tau_weighted, -1e6, 1e6)
            gradients[lv_name] = {
                'grad_zeta': grad_zeta_final,
                'grad_tau': cp.asnumpy(grad_tau_weighted)
            }

            if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
                iteration_logger.info(f"\n  [{lv_name}] 최종 그래디언트:")
                iteration_logger.info(f"    - grad_zeta: 범위=[{float(cp.min(grad_zeta_weighted)):.4f}, {float(cp.max(grad_zeta_weighted)):.4f}], norm={float(cp.linalg.norm(grad_zeta_weighted)):.4f}")
                iteration_logger.info(f"    - grad_tau: 범위=[{float(cp.min(grad_tau_weighted)):.4f}, {float(cp.max(grad_tau_weighted)):.4f}], norm={float(cp.linalg.norm(grad_tau_weighted)):.4f}")

    return gradients


def compute_structural_gradient_batch_gpu(
    ind_data: pd.DataFrame,
    lvs_list: List[Dict[str, float]],
    exo_draws_list: List[np.ndarray],
    params: Dict,
    covariates: List[str],
    endogenous_lv: str,
    exogenous_lvs: List[str],
    weights: np.ndarray,
    error_variance: float = 1.0,
    is_hierarchical: bool = False,
    hierarchical_paths: List[Dict] = None,
    iteration_logger=None,
    log_level: str = 'DETAILED'
) -> Dict[str, np.ndarray]:
    """
    구조모델 그래디언트를 GPU 배치로 계산 (가중평균 적용)

    ✅ 계층적 구조와 병렬 구조 모두 지원

    CPU 구현 (gradient_calculator.py의 StructuralGradient)을 따르면서:
    1. Importance weighting 적용
    2. GPU 배치 처리로 성능 향상

    Args:
        ind_data: 개인 데이터
        lvs_list: 각 draw의 잠재변수 값
        exo_draws_list: 각 draw의 외생 draws
        params: 구조모델 파라미터
        covariates: 공변량 리스트
        endogenous_lv: 내생 LV 이름 (병렬 구조에서만 사용)
        exogenous_lvs: 외생 LV 이름 리스트 (병렬 구조에서만 사용)
        weights: Importance weights (n_draws,)
        error_variance: 오차 분산
        is_hierarchical: 계층적 구조 여부
        hierarchical_paths: 계층적 경로 정보
        iteration_logger: 로거 (optional)
        log_level: 로깅 레벨 ('MINIMAL', 'MODERATE', 'DETAILED')

    Returns:
        병렬 구조: {'grad_gamma_lv': ..., 'grad_gamma_x': ...}
        계층적 구조: {'grad_gamma_{pred}_to_{target}': ...}
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    n_draws = len(lvs_list)
    weights_gpu = cp.asarray(weights)  # (n_draws,)

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info("\n[구조모델 그래디언트 계산]")
        iteration_logger.info(f"  구조 유형: {'계층적' if is_hierarchical else '병렬'}")

    if is_hierarchical:
        # ✅ 계층적 구조: 각 경로별로 gradient 계산
        gradients = {}

        for path_idx, path in enumerate(hierarchical_paths):
            target = path['target']
            predictors = path['predictors']
            param_key = f"gamma_{predictors[0]}_to_{target}"

            # 파라미터 추출
            gamma = params[param_key]

            if iteration_logger and log_level == 'DETAILED' and path_idx == 0:
                iteration_logger.info(f"\n  [경로 {path_idx + 1}] {predictors[0]} → {target}")
                iteration_logger.info(f"    - gamma: {gamma:.6f}")

            # LV 값 추출
            target_values = np.array([lvs[target] for lvs in lvs_list])
            pred_values = np.array([lvs[predictors[0]] for lvs in lvs_list])

            # GPU로 전송
            target_gpu = cp.asarray(target_values)  # (n_draws,)
            pred_gpu = cp.asarray(pred_values)  # (n_draws,)
            gamma_gpu = cp.asarray(gamma)  # 스칼라

            # 예측값 계산: target = gamma * predictor + error
            mu = gamma_gpu * pred_gpu  # (n_draws,)

            # 잔차
            residual = target_gpu - mu  # (n_draws,)

            if iteration_logger and log_level == 'DETAILED' and path_idx == 0:
                iteration_logger.info(f"    - predictor (draw 0): {float(pred_gpu[0]):.4f}")
                iteration_logger.info(f"    - target (draw 0): {float(target_gpu[0]):.4f}")
                iteration_logger.info(f"    - 예측값 μ (draw 0): {float(mu[0]):.4f}")
                iteration_logger.info(f"    - 잔차 (draw 0): {float(residual[0]):.4f}")
                iteration_logger.info(f"    - 잔차 범위: [{float(cp.min(residual)):.4f}, {float(cp.max(residual)):.4f}]")
                # ✅ 추가 디버깅: 모든 draws의 target 값 확인
                iteration_logger.info(f"    - target 값 (처음 5개 draws): {target_values[:5]}")
                iteration_logger.info(f"    - predictor 값 (처음 5개 draws): {pred_values[:5]}")
                iteration_logger.info(f"    - 잔차 (처음 5개 draws): {cp.asnumpy(residual[:5])}")

            # ∂ log L / ∂γ = Σ_r w_r * (target - μ)_r / σ² * predictor_r
            weighted_residual = weights_gpu * residual / error_variance  # (n_draws,)
            grad_gamma = cp.sum(weighted_residual * pred_gpu)  # 스칼라

            # NaN 체크
            if cp.isnan(grad_gamma):
                logger.warning(f"NaN detected in grad_{param_key}")
                grad_gamma = cp.asarray(0.0)

            # Gradient clipping
            grad_gamma = cp.clip(grad_gamma, -1e6, 1e6)

            gradients[f'grad_{param_key}'] = cp.asnumpy(grad_gamma).item()

            if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
                iteration_logger.info(f"    - grad_{param_key}: {gradients[f'grad_{param_key}']:.6f}")

        return gradients

    else:
        # 병렬 구조 (기존 방식)
        n_exo = len(exogenous_lvs)
        n_cov = len(covariates)

        gamma_lv = params['gamma_lv']
        gamma_x = params['gamma_x']

        # 배열로 변환
        lv_endo_values = np.array([lvs[endogenous_lv] for lvs in lvs_list])
        exo_lv_matrix = np.array([[lvs[lv_name] for lv_name in exogenous_lvs] for lvs in lvs_list])

        # 공변량 (모든 draw에서 동일)
        first_row = ind_data.iloc[0]
        X = np.array([first_row[cov] if cov in first_row.index and not pd.isna(first_row[cov]) else 0.0 for cov in covariates])

        # GPU로 전송
        lv_endo_gpu = cp.asarray(lv_endo_values)  # (n_draws,)
        exo_lv_gpu = cp.asarray(exo_lv_matrix)  # (n_draws, n_exo)
        X_gpu = cp.asarray(X)  # (n_cov,)
        gamma_lv_gpu = cp.asarray(gamma_lv)  # (n_exo,)
        gamma_x_gpu = cp.asarray(gamma_x)  # (n_cov,)

        # 예측값 계산
        mu = cp.dot(exo_lv_gpu, gamma_lv_gpu) + cp.dot(X_gpu, gamma_x_gpu)  # (n_draws,)

        # 잔차
        residual = lv_endo_gpu - mu  # (n_draws,)

        # ✅ 수정: 가중평균 적용
        # ∂ log L / ∂γ_lv = Σ_r w_r * (LV_endo - μ)_r / σ² * LV_exo_r
        weighted_residual = weights_gpu * residual / error_variance  # (n_draws,)
        grad_gamma_lv = cp.dot(exo_lv_gpu.T, weighted_residual)  # (n_exo,)

        # ∂ log L / ∂γ_x = Σ_r w_r * (LV_endo - μ)_r / σ² * X
        grad_gamma_x = cp.sum(weighted_residual) * X_gpu  # (n_cov,)

        # NaN 체크
        if cp.any(cp.isnan(grad_gamma_lv)):
            logger.warning("NaN detected in grad_gamma_lv")
            grad_gamma_lv = cp.nan_to_num(grad_gamma_lv, nan=0.0)

        if cp.any(cp.isnan(grad_gamma_x)):
            logger.warning("NaN detected in grad_gamma_x")
            grad_gamma_x = cp.nan_to_num(grad_gamma_x, nan=0.0)

        # Gradient clipping
        grad_gamma_lv = cp.clip(grad_gamma_lv, -1e6, 1e6)
        grad_gamma_x = cp.clip(grad_gamma_x, -1e6, 1e6)

        return {
            'grad_gamma_lv': cp.asnumpy(grad_gamma_lv),
            'grad_gamma_x': cp.asnumpy(grad_gamma_x)
        }


def compute_choice_gradient_batch_gpu(
    ind_data: pd.DataFrame,
    lvs_list: List[Dict[str, float]],
    params: Dict,
    endogenous_lv: str,
    choice_attributes: List[str],
    weights: np.ndarray,
    moderators: List[str] = None,
    iteration_logger=None,
    log_level: str = 'DETAILED'
) -> Dict[str, np.ndarray]:
    """
    선택모델 그래디언트를 GPU 배치로 계산 (가중평균 + 배치 처리)

    ✅ 조절효과 지원 추가

    CPU 구현 (gradient_calculator.py의 ChoiceGradient)을 따르면서:
    1. Importance weighting 적용
    2. GPU 배치 처리로 성능 향상 (for loop 제거)

    Args:
        ind_data: 개인의 선택 데이터
        lvs_list: 각 draw의 잠재변수 값
        params: 선택모델 파라미터
        endogenous_lv: 내생 LV 이름 (main LV)
        choice_attributes: 선택 속성 리스트
        weights: Importance weights (n_draws,)
        moderators: 조절변수 LV 이름 리스트 (optional)
        iteration_logger: 로거 (optional)
        log_level: 로깅 레벨 ('MINIMAL', 'MODERATE', 'DETAILED')

    Returns:
        기본: {'grad_intercept': ..., 'grad_beta': ..., 'grad_lambda': ...}
        조절효과: {'grad_intercept': ..., 'grad_beta': ..., 'grad_lambda_main': ..., 'grad_lambda_mod_{moderator}': ...}
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    n_draws = len(lvs_list)
    n_choice_situations = len(ind_data)
    n_attributes = len(choice_attributes)

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info("\n[선택모델 그래디언트 계산]")
        iteration_logger.info(f"  선택 상황 수: {n_choice_situations}")

    intercept = params['intercept']
    beta = params['beta']

    # ✅ 유연한 리스트 기반: lambda_ 파라미터 자동 추출
    lambda_lvs = {}
    gamma_interactions = {}

    for key in params.keys():
        if key.startswith('lambda_'):
            lv_name = key.replace('lambda_', '')
            lambda_lvs[lv_name] = params[key]
        elif key.startswith('gamma_') and '_to_' not in key:
            # LV-Attribute 상호작용: gamma_{lv_name}_{attr_name}
            # 구조모델 파라미터 (gamma_{lv1}_to_{lv2})는 제외
            gamma_interactions[key] = params[key]

    if iteration_logger and log_level == 'DETAILED':
        iteration_logger.info(f"  유연한 리스트 기반 모델:")
        iteration_logger.info(f"    - intercept: {intercept:.6f}")
        iteration_logger.info(f"    - beta: {beta[:min(3, len(beta))]}")
        if lambda_lvs:
            for lv_name, lambda_val in lambda_lvs.items():
                iteration_logger.info(f"    - lambda_{lv_name}: {lambda_val:.6f}")
        else:
            iteration_logger.info(f"    - lambda: 없음 (Base Model)")
        if gamma_interactions:
            for gamma_key, gamma_val in gamma_interactions.items():
                iteration_logger.info(f"    - {gamma_key}: {gamma_val:.6f}")

    weights_gpu = cp.asarray(weights)  # (n_draws,)

    # ✅ 유연한 리스트 기반: LV 값들 자동 추출
    lv_values = {}
    for lv_name in lambda_lvs.keys():
        lv_values[lv_name] = np.array([lvs[lv_name] for lvs in lvs_list])

    # 선택 변수 찾기
    choice_var = None
    for col in ['choice', 'chosen', 'choice_binary']:
        if col in ind_data.columns:
            choice_var = col
            break

    if choice_var is None:
        raise ValueError("선택 변수를 찾을 수 없습니다")

    # 속성 데이터와 선택 준비
    attributes_matrix = []
    choices = []

    for idx in range(n_choice_situations):
        row = ind_data.iloc[idx]
        attr_values = [row[attr] if attr in row.index and not pd.isna(row[attr]) else 0.0 for attr in choice_attributes]
        attributes_matrix.append(attr_values)
        choices.append(row[choice_var])

    attributes_matrix = np.array(attributes_matrix)  # (n_situations, n_attributes)
    choices = np.array(choices)  # (n_situations,)

    # GPU로 전송
    attr_gpu = cp.asarray(attributes_matrix)  # (n_situations, n_attributes)
    choices_gpu = cp.asarray(choices)  # (n_situations,)
    beta_gpu = cp.asarray(beta)  # (n_attributes,)

    # ✅ 유연한 리스트 기반: LV GPU 전송
    lv_gpu = {}
    for lv_name, lv_vals in lv_values.items():
        lv_gpu[lv_name] = cp.asarray(lv_vals)  # (n_draws,)

    # ✅ 개선: 배치 처리 (for loop 제거)
    # Broadcasting을 사용하여 모든 draws를 동시에 처리

    # attr_batch: (1, n_situations, n_attributes)
    attr_batch = attr_gpu[None, :, :]

    # V_batch: (n_draws, n_situations)
    # V = intercept + β'X
    V_batch = intercept + cp.dot(attr_batch, beta_gpu[:, None]).squeeze(-1)

    # ✅ 유연한 리스트 기반: V += Σ(λ_i * LV_i)
    # lambda_lvs가 빈 딕셔너리면 아무것도 추가 안 됨 (Base Model)
    for lv_name, lambda_val in lambda_lvs.items():
        lv_batch = lv_gpu[lv_name][:, None]  # (n_draws, 1)
        V_batch = V_batch + lambda_val * lv_batch

    # Φ(V): (n_draws, n_situations)
    prob_batch = cp_ndtr(V_batch)
    prob_batch = cp.clip(prob_batch, 1e-10, 1 - 1e-10)

    # φ(V): (n_draws, n_situations)
    phi_batch = cp_norm_pdf(V_batch)

    # 실제 선택에 따라: (n_draws, n_situations)
    prob_final_batch = cp.where(choices_gpu[None, :] == 1, prob_batch, 1 - prob_batch)

    # Mills ratio: (n_draws, n_situations)
    mills_batch = phi_batch / prob_final_batch

    # Sign: (n_draws, n_situations)
    sign_batch = cp.where(choices_gpu[None, :] == 1, 1.0, -1.0)

    if iteration_logger and log_level == 'DETAILED':
        iteration_logger.info(f"\n  [중간 계산 값 (draw 0, situation 0)]")
        iteration_logger.info(f"    - V: {float(V_batch[0, 0]):.4f}")
        iteration_logger.info(f"    - Φ(V): {float(prob_batch[0, 0]):.4f}")
        iteration_logger.info(f"    - φ(V): {float(phi_batch[0, 0]):.4f}")
        iteration_logger.info(f"    - choice: {int(choices_gpu[0])}")
        iteration_logger.info(f"    - Mills ratio: {float(mills_batch[0, 0]):.4f}")

    # Weighted mills: (n_draws, n_situations)
    weighted_mills = weights_gpu[:, None] * sign_batch * mills_batch

    # ✅ 수정: 가중평균 적용
    # ∂V/∂intercept = 1
    grad_intercept = cp.sum(weighted_mills).item()

    # ∂V/∂β = X
    # (n_situations, n_attributes).T @ (n_draws, n_situations).T
    # = (n_attributes, n_situations) @ (n_situations, n_draws)
    # = (n_attributes, n_draws) → sum over draws
    grad_beta = cp.dot(attr_gpu.T, weighted_mills.T).sum(axis=1)  # (n_attributes,)

    # ✅ 유연한 리스트 기반: Lambda gradient 계산
    # ∂V/∂λ_i = LV_i
    # lambda_lvs가 빈 딕셔너리면 아무것도 계산 안 됨 (Base Model)
    grad_lambda = {}
    for lv_name in lambda_lvs.keys():
        lv_batch = lv_gpu[lv_name][:, None]  # (n_draws, 1)
        grad_lambda[lv_name] = cp.sum(weighted_mills * lv_batch).item()

    # ✅ 유연한 리스트 기반: Gamma gradient 계산 (LV-Attribute 상호작용)
    # ∂V/∂γ_ij = LV_i × X_j
    grad_gamma = {}

    if iteration_logger and log_level == 'DETAILED':
        iteration_logger.info(f"\n  [Gamma Gradient 계산]")
        iteration_logger.info(f"    - gamma_interactions 키: {list(gamma_interactions.keys())}")
        iteration_logger.info(f"    - lv_gpu 키: {list(lv_gpu.keys())}")
        iteration_logger.info(f"    - choice_attributes: {choice_attributes}")

    for gamma_key in gamma_interactions.keys():
        # gamma_purchase_intention_health_label → lv_name='purchase_intention', attr_name='health_label'
        parts = gamma_key.replace('gamma_', '').rsplit('_', 1)
        if iteration_logger and log_level == 'DETAILED':
            iteration_logger.info(f"    - {gamma_key}: parts={parts}")

        if len(parts) == 2:
            lv_name, attr_name = parts
            if iteration_logger and log_level == 'DETAILED':
                iteration_logger.info(f"      lv_name={lv_name}, attr_name={attr_name}")
                iteration_logger.info(f"      lv_name in lv_gpu: {lv_name in lv_gpu}")
                iteration_logger.info(f"      attr_name in choice_attributes: {attr_name in choice_attributes}")

            if lv_name in lv_gpu and attr_name in choice_attributes:
                lv_batch = lv_gpu[lv_name][:, None]  # (n_draws, 1)
                attr_idx = choice_attributes.index(attr_name)
                attr_values = attr_gpu[:, attr_idx]  # (n_situations,)
                # (n_draws, n_situations) * (n_draws, 1) * (1, n_situations)
                interaction_batch = lv_batch * attr_values[None, :]  # (n_draws, n_situations)
                grad_gamma[gamma_key] = cp.sum(weighted_mills * interaction_batch).item()

                if iteration_logger and log_level == 'DETAILED':
                    iteration_logger.info(f"      ✅ grad_{gamma_key} 계산 완료: {grad_gamma[gamma_key]:.6f}")

    # NaN 체크
    if np.isnan(grad_intercept):
        logger.warning("NaN detected in grad_intercept")
        grad_intercept = 0.0

    if cp.any(cp.isnan(grad_beta)):
        logger.warning("NaN detected in grad_beta")
        grad_beta = cp.nan_to_num(grad_beta, nan=0.0)

    # ✅ 유연한 리스트 기반: Lambda NaN 체크
    for lv_name in grad_lambda.keys():
        if np.isnan(grad_lambda[lv_name]):
            logger.warning(f"NaN detected in grad_lambda_{lv_name}")
            grad_lambda[lv_name] = 0.0

    # ✅ 유연한 리스트 기반: Gamma NaN 체크
    for gamma_key in grad_gamma.keys():
        if np.isnan(grad_gamma[gamma_key]):
            logger.warning(f"NaN detected in grad_{gamma_key}")
            grad_gamma[gamma_key] = 0.0

    # Gradient clipping
    grad_intercept = np.clip(grad_intercept, -1e6, 1e6)
    grad_beta = cp.clip(grad_beta, -1e6, 1e6)

    # ✅ 유연한 리스트 기반: Lambda clipping
    for lv_name in grad_lambda.keys():
        grad_lambda[lv_name] = np.clip(grad_lambda[lv_name], -1e6, 1e6)

    # ✅ 유연한 리스트 기반: Gamma clipping
    for gamma_key in grad_gamma.keys():
        grad_gamma[gamma_key] = np.clip(grad_gamma[gamma_key], -1e6, 1e6)

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(f"\n  [최종 그래디언트]")
        iteration_logger.info(f"    - grad_intercept: {grad_intercept:.6f}")
        iteration_logger.info(f"    - grad_beta: {cp.asnumpy(grad_beta)[:min(3, len(grad_beta))]}")
        if grad_lambda:
            for lv_name, grad_val in grad_lambda.items():
                iteration_logger.info(f"    - grad_lambda_{lv_name}: {grad_val:.6f}")
        else:
            iteration_logger.info(f"    - grad_lambda: 없음 (Base Model)")
        if grad_gamma:
            for gamma_key, grad_val in grad_gamma.items():
                iteration_logger.info(f"    - grad_{gamma_key}: {grad_val:.6f}")

    # ✅ 유연한 리스트 기반: 결과 반환
    result = {
        'grad_intercept': grad_intercept,
        'grad_beta': cp.asnumpy(grad_beta)
    }
    for lv_name, grad_val in grad_lambda.items():
        result[f'grad_lambda_{lv_name}'] = grad_val
    for gamma_key, grad_val in grad_gamma.items():
        result[f'grad_{gamma_key}'] = grad_val

    return result


def compute_all_individuals_gradients_batch_gpu(
    gpu_measurement_model,
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
    모든 개인의 gradient를 GPU batch로 동시 계산 (개인별 순차 + draws GPU batch)

    ⚠️ 이 함수는 개인별 순차 처리입니다.
    완전 GPU Batch는 compute_all_individuals_gradients_full_batch_gpu 사용

    Args:
        gpu_measurement_model: GPU 측정모델
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
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    n_individuals = len(all_ind_data)
    n_draws = all_ind_draws.shape[1]

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"\n{'='*80}\n"
            f"GPU Batch Gradient 계산 (개인별 순차)\n"
            f"{'='*80}\n"
            f"  개인 수: {n_individuals}명\n"
            f"  Draws per individual: {n_draws}개\n"
            f"  총 계산: {n_individuals} × {n_draws} = {n_individuals * n_draws}개\n"
            f"{'='*80}"
        )

    # 계층적 구조 지원
    is_hierarchical = hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical

    if is_hierarchical:
        n_first_order = len(structural_model.exogenous_lvs)
        n_higher_order = len(structural_model.get_higher_order_lvs())
    else:
        n_exo = structural_model.n_exo

    # 모든 개인의 gradient 저장
    all_individual_gradients = []

    # 개인별로 처리 (각 개인 내부는 GPU batch)
    for ind_idx, (ind_data, ind_draws) in enumerate(zip(all_ind_data, all_ind_draws)):
        # 모든 draws의 LV 값 미리 계산
        lvs_list = []
        exo_draws_list = []

        for draw_idx in range(n_draws):
            if is_hierarchical:
                # 계층적 구조
                first_order_draws = ind_draws[draw_idx, :n_first_order]
                higher_order_errors = ind_draws[draw_idx, n_first_order:]

                higher_order_lvs = structural_model.get_higher_order_lvs()
                error_dict = {lv_name: higher_order_errors[i] for i, lv_name in enumerate(higher_order_lvs)}

                latent_vars = structural_model.predict(
                    ind_data, first_order_draws, params_dict['structural'],
                    endo_draw=None, higher_order_draws=error_dict
                )
                exo_draws_list.append(first_order_draws)
            else:
                # 병렬 구조
                exo_draws = ind_draws[draw_idx, :n_exo]
                endo_draw = ind_draws[draw_idx, n_exo]

                latent_vars = structural_model.predict(
                    ind_data, exo_draws, params_dict['structural'], endo_draw
                )
                exo_draws_list.append(exo_draws)

            lvs_list.append(latent_vars)

        # 1. 결합 likelihood 계산 (GPU batch)
        ll_batch = compute_joint_likelihood_batch_gpu(
            gpu_measurement_model,
            ind_data,
            lvs_list,
            ind_draws,
            params_dict,
            structural_model,
            choice_model
        )

        # 2. Importance weights 계산 (GPU)
        weights = compute_importance_weights_gpu(ll_batch, individual_id=ind_idx)

        # 3. 가중평균 gradient 계산 (GPU batch)
        grad_meas = compute_measurement_gradient_batch_gpu(
            gpu_measurement_model,
            ind_data,
            lvs_list,
            params_dict['measurement'],
            weights,
            iteration_logger=None,  # 개별 로깅 비활성화
            log_level='MINIMAL'
        )

        grad_struct = compute_structural_gradient_batch_gpu(
            ind_data,
            lvs_list,
            exo_draws_list,
            params_dict['structural'],
            structural_model.covariates,
            structural_model.endogenous_lv if not is_hierarchical else None,
            structural_model.exogenous_lvs if not is_hierarchical else None,
            weights,
            error_variance=1.0,
            is_hierarchical=is_hierarchical,
            hierarchical_paths=structural_model.hierarchical_paths if is_hierarchical else None,
            iteration_logger=None,
            log_level='MINIMAL'
        )

        # 선택모델 gradient
        if hasattr(choice_model.config, 'moderators') and choice_model.config.moderators:
            # 조절효과 모델
            grad_choice = compute_choice_gradient_batch_gpu(
                ind_data,
                lvs_list,
                params_dict['choice'],
                choice_model.config.main_lv,
                choice_model.config.choice_attributes,
                weights,
                moderators=choice_model.config.moderators,
                iteration_logger=None,
                log_level='MINIMAL'
            )
        else:
            # 기본 모델
            grad_choice = compute_choice_gradient_batch_gpu(
                ind_data,
                lvs_list,
                params_dict['choice'],
                structural_model.endogenous_lv,
                choice_model.config.choice_attributes,
                weights,
                moderators=None,
                iteration_logger=None,
                log_level='MINIMAL'
            )

        # 개인별 gradient 저장
        ind_grad_dict = {
            'measurement': grad_meas,
            'structural': grad_struct,
            'choice': grad_choice
        }

        all_individual_gradients.append(ind_grad_dict)

        # 진행 상황 로깅 (10% 단위)
        if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
            if (ind_idx + 1) % max(1, n_individuals // 10) == 0:
                progress = (ind_idx + 1) / n_individuals * 100
                iteration_logger.info(f"  진행: {ind_idx + 1}/{n_individuals} ({progress:.0f}%)")

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"{'='*80}\n"
            f"완전 GPU Batch Gradient 계산 완료: {n_individuals}명\n"
            f"{'='*80}"
        )

    return all_individual_gradients

def compute_all_individuals_gradients_full_batch_gpu(
    gpu_measurement_model,
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
    모든 개인의 gradient를 완전 GPU batch로 동시 계산

    🚀 완전 GPU Batch: 326명 × 100 draws × 80 params = 2,608,000개 동시 계산

    Args:
        gpu_measurement_model: GPU 측정모델
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
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    import time

    n_individuals = len(all_ind_data)
    n_draws = all_ind_draws.shape[1]

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"\n{'='*80}\n"
            f"🚀 완전 GPU Batch Gradient 계산\n"
            f"{'='*80}\n"
            f"  개인 수: {n_individuals}명\n"
            f"  Draws per individual: {n_draws}개\n"
            f"  총 계산: {n_individuals} × {n_draws} = {n_individuals * n_draws}개 동시 처리\n"
            f"{'='*80}"
        )

    total_start = time.time()

    # Step 1: 데이터 준비 - 모든 개인 데이터를 3D 배열로 변환
    prep_start = time.time()

    # 모든 개인이 동일한 행 수를 가진다고 가정 (18행)
    n_rows = len(all_ind_data[0])

    # 필요한 컬럼 추출 (선택 데이터만)
    # choice_column은 estimator의 config에 있음
    # 여기서는 사용하지 않으므로 제거

    prep_time = time.time() - prep_start

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  데이터 준비 완료 ({prep_time:.3f}초):\n"
            f"    - all_ind_draws shape: {all_ind_draws.shape}"
        )

    # Step 2: GPU로 데이터 전송
    transfer_start = time.time()

    all_draws_gpu = cp.asarray(all_ind_draws)

    transfer_time = time.time() - transfer_start

    # Step 3: 완전 GPU Batch로 모든 개인 × 모든 draws의 LV 계산
    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  Step 3: 모든 개인 × 모든 draws의 LV 계산 중..."
        )

    lv_start = time.time()

    # 모든 개인 × 모든 draws의 LV 계산
    # Shape: (326, 100, n_lvs)
    all_lvs_list = []  # List of List[Dict]: (326, 100)

    is_hierarchical = hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical

    for ind_idx, (ind_data, ind_draws) in enumerate(zip(all_ind_data, all_ind_draws)):
        ind_lvs_list = []

        for draw_idx in range(n_draws):
            draw = ind_draws[draw_idx]

            # LV 계산 (CPU - structural_model.predict)
            latent_vars = structural_model.predict(
                ind_data.iloc[0],
                draw,
                params_dict['structural']
            )

            ind_lvs_list.append(latent_vars)

        all_lvs_list.append(ind_lvs_list)

    lv_time = time.time() - lv_start

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  LV 계산 완료 ({lv_time:.3f}초)"
        )

    # Step 4: LV를 3D 배열로 변환 (326, 100, 5)
    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  Step 4: LV를 3D 배열로 변환 중..."
        )

    convert_start = time.time()

    # LV 이름 순서 정의
    lv_names = list(params_dict['measurement'].keys())
    n_lvs = len(lv_names)

    # 3D 배열 생성: (326, 100, 5)
    all_lvs_array = np.zeros((n_individuals, n_draws, n_lvs))

    for ind_idx, ind_lvs_list in enumerate(all_lvs_list):
        for draw_idx, lvs_dict in enumerate(ind_lvs_list):
            for lv_idx, lv_name in enumerate(lv_names):
                all_lvs_array[ind_idx, draw_idx, lv_idx] = lvs_dict[lv_name]

    convert_time = time.time() - convert_start

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  LV 배열 변환 완료 ({convert_time:.3f}초): shape = {all_lvs_array.shape}"
        )

    # Step 5: 완전 GPU Batch로 모든 개인 × 모든 draws의 gradient 계산
    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  Step 5: 완전 GPU Batch gradient 계산 중 (1번의 GPU 호출)..."
        )

    grad_start = time.time()

    # 균등 가중치 (326, 100)
    all_weights = np.ones((n_individuals, n_draws)) / n_draws

    # 🚀 완전 GPU Batch: 326명 × 100 draws × 80 params = 2,608,000개 동시 계산
    # 측정모델, 구조모델, 선택모델 gradient를 한 번에 계산
    all_individual_gradients = compute_full_batch_gradients_gpu(
        gpu_measurement_model,
        all_ind_data,
        all_lvs_array,
        all_ind_draws,
        params_dict,
        all_weights,
        structural_model,
        choice_model,
        lv_names,
        iteration_logger=iteration_logger,
        log_level=log_level
    )

    grad_time = time.time() - grad_start

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  Gradient 계산 완료 ({grad_time:.3f}초)"
        )

    total_time = time.time() - total_start

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"\n완전 GPU Batch 계산 완료:\n"
            f"  총 시간: {total_time:.3f}초\n"
            f"    - 데이터 준비: {prep_time:.3f}초\n"
            f"    - 데이터 전송 (GPU): {transfer_time:.3f}초\n"
            f"    - LV 계산: {lv_time:.3f}초\n"
            f"    - Gradient 계산: {grad_time:.3f}초\n"
            f"  개인당 시간: {total_time / n_individuals * 1000:.2f}ms\n"
            f"  처리량: {n_individuals / total_time:.1f} 개인/초"
        )

    return all_individual_gradients


def compute_full_batch_gradients_gpu(
    gpu_measurement_model,
    all_ind_data: List[pd.DataFrame],
    all_lvs_array: np.ndarray,  # (326, 100, 5)
    all_ind_draws: np.ndarray,  # (326, 100, 6)
    params_dict: Dict,
    all_weights: np.ndarray,  # (326, 100)
    structural_model,
    choice_model,
    lv_names: List[str],
    iteration_logger=None,
    log_level: str = 'MINIMAL'
) -> List[Dict]:
    """
    완전 GPU Batch: 326명 × 100 draws × 80 params = 2,608,000개 gradient 동시 계산

    Args:
        gpu_measurement_model: GPU 측정모델
        all_ind_data: 모든 개인 데이터 (326개)
        all_lvs_array: 모든 LV 값 (326, 100, 5)
        all_ind_draws: 모든 draws (326, 100, 6)
        params_dict: 파라미터 딕셔너리
        all_weights: 가중치 (326, 100)
        structural_model: 구조모델
        choice_model: 선택모델
        lv_names: LV 이름 리스트
        iteration_logger: 로거
        log_level: 로깅 레벨

    Returns:
        개인별 gradient 딕셔너리 리스트 (326개)
    """
    n_individuals, n_draws, n_lvs = all_lvs_array.shape

    # GPU로 전송
    all_lvs_gpu = cp.asarray(all_lvs_array)  # (326, 100, 5)
    all_weights_gpu = cp.asarray(all_weights)  # (326, 100)

    # 1. 측정모델 Gradient (완전 Batch)
    meas_grads = compute_measurement_full_batch_gpu(
        gpu_measurement_model,
        all_ind_data,
        all_lvs_gpu,
        params_dict['measurement'],
        all_weights_gpu,
        lv_names,
        iteration_logger,
        log_level
    )

    # 2. 구조모델 Gradient (완전 Batch)
    struct_grads = compute_structural_full_batch_gpu(
        all_lvs_gpu,
        params_dict['structural'],
        all_weights_gpu,
        structural_model,
        lv_names,
        iteration_logger,
        log_level
    )

    # 3. 선택모델 Gradient (완전 Batch)
    choice_grads = compute_choice_full_batch_gpu(
        all_ind_data,
        all_lvs_gpu,
        params_dict['choice'],
        all_weights_gpu,
        choice_model,
        lv_names,
        iteration_logger,
        log_level
    )

    # 개인별 gradient 딕셔너리로 변환
    all_individual_gradients = []
    for ind_idx in range(n_individuals):
        # 측정모델: {lv_name: {'grad_zeta': array, 'grad_sigma_sq': array}}
        meas_dict = {}
        for lv_name in meas_grads:
            meas_dict[lv_name] = {
                'grad_zeta': meas_grads[lv_name]['grad_zeta'][ind_idx],
                'grad_sigma_sq': meas_grads[lv_name]['grad_sigma_sq'][ind_idx]
            }

        # 구조모델: {param_name: scalar}
        struct_dict = {key: struct_grads[key][ind_idx].item() if hasattr(struct_grads[key][ind_idx], 'item') else struct_grads[key][ind_idx] for key in struct_grads}

        # 선택모델: {'grad_intercept': scalar, 'grad_beta': array, ...}
        choice_dict = {}
        for key in choice_grads:
            val = choice_grads[key][ind_idx]
            # grad_beta는 배열이므로 그대로 유지
            if key == 'grad_beta':
                choice_dict[key] = val
            elif hasattr(val, 'item'):
                choice_dict[key] = val.item()
            else:
                choice_dict[key] = val

        ind_grad_dict = {
            'measurement': meas_dict,
            'structural': struct_dict,
            'choice': choice_dict
        }
        all_individual_gradients.append(ind_grad_dict)

    return all_individual_gradients


def compute_measurement_full_batch_gpu(
    gpu_measurement_model,
    all_ind_data: List[pd.DataFrame],
    all_lvs_gpu,  # CuPy array (326, 100, 5)
    params: Dict,
    all_weights_gpu,  # CuPy array (326, 100)
    lv_names: List[str],
    iteration_logger=None,
    log_level: str = 'MINIMAL'
) -> Dict:
    """
    측정모델 Gradient - 완전 GPU Batch

    Returns:
        {lv_name: {'grad_zeta': (326, n_indicators), 'grad_sigma_sq': (326, n_indicators)}}
    """
    n_individuals, n_draws, n_lvs = all_lvs_gpu.shape

    gradients = {}

    for lv_idx, lv_name in enumerate(lv_names):
        zeta = params[lv_name]['zeta']
        sigma_sq = params[lv_name]['sigma_sq']
        n_indicators = len(zeta)

        config = gpu_measurement_model.models[lv_name].config

        # 모든 개인의 관측값 추출 (326, n_indicators)
        all_y = np.zeros((n_individuals, n_indicators))
        for ind_idx, ind_data in enumerate(all_ind_data):
            row = ind_data.iloc[0]
            for i, indicator in enumerate(config.indicators):
                if indicator in row.index and not pd.isna(row[indicator]):
                    all_y[ind_idx, i] = row[indicator]

        all_y_gpu = cp.asarray(all_y)  # (326, n_indicators)
        zeta_gpu = cp.asarray(zeta)  # (n_indicators,)
        sigma_sq_gpu = cp.asarray(sigma_sq)  # (n_indicators,)

        # LV 값 추출: (326, 100)
        lv_values_gpu = all_lvs_gpu[:, :, lv_idx]

        # Gradient 초기화
        grad_zeta_all = cp.zeros((n_individuals, n_indicators))
        grad_sigma_sq_all = cp.zeros((n_individuals, n_indicators))

        # 각 지표별로 계산
        for i in range(n_indicators):
            # 예측값: (326, 100)
            y_pred = zeta_gpu[i] * lv_values_gpu

            # 잔차: (326, 100)
            residual = all_y_gpu[:, i:i+1] - y_pred

            # Gradient (각 draw): (326, 100)
            grad_zeta_batch = residual * lv_values_gpu / sigma_sq_gpu[i]
            grad_sigma_sq_batch = -0.5 / sigma_sq_gpu[i] + 0.5 * (residual ** 2) / (sigma_sq_gpu[i] ** 2)

            # 가중평균: (326,)
            grad_zeta_all[:, i] = cp.sum(all_weights_gpu * grad_zeta_batch, axis=1)
            grad_sigma_sq_all[:, i] = cp.sum(all_weights_gpu * grad_sigma_sq_batch, axis=1)

        # ✅ fix_first_loading 고려: 첫 번째 loading이 고정되면 gradient 제외
        fix_first_loading = getattr(config, 'fix_first_loading', True)
        if fix_first_loading:
            # 첫 번째 zeta는 1.0으로 고정 (gradient 제외)
            grad_zeta_final = cp.asnumpy(grad_zeta_all[:, 1:])  # (326, n_indicators-1)
        else:
            grad_zeta_final = cp.asnumpy(grad_zeta_all)  # (326, n_indicators)

        gradients[lv_name] = {
            'zeta': grad_zeta_final,
            'sigma_sq': cp.asnumpy(grad_sigma_sq_all)
        }

    return gradients


def compute_structural_full_batch_gpu(
    all_lvs_gpu,  # CuPy array (326, 100, 5)
    params: Dict,
    all_weights_gpu,  # CuPy array (326, 100)
    structural_model,
    lv_names: List[str],
    iteration_logger=None,
    log_level: str = 'MINIMAL'
) -> Dict:
    """
    구조모델 Gradient - 완전 GPU Batch

    Returns:
        {param_name: (326,)}
    """
    n_individuals, n_draws, n_lvs = all_lvs_gpu.shape

    gradients = {}

    # 계층적 구조인 경우
    if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
        error_variance = 1.0

        for path in structural_model.hierarchical_paths:
            target = path['target']
            predictor = path['predictors'][0]  # 단일 predictor 가정
            param_key = f"gamma_{predictor}_to_{target}"
            gamma = params[param_key]

            # LV 인덱스 찾기
            target_idx = lv_names.index(target)
            pred_idx = lv_names.index(predictor)

            # LV 값 추출: (326, 100)
            target_values = all_lvs_gpu[:, :, target_idx]
            pred_values = all_lvs_gpu[:, :, pred_idx]

            # 예측값: (326, 100)
            mu = gamma * pred_values

            # 잔차: (326, 100)
            residual = target_values - mu

            # Gradient: (326, 100)
            weighted_residual = all_weights_gpu * residual / error_variance

            # 가중합: (326,)
            grad_gamma = cp.sum(weighted_residual * pred_values, axis=1)

            # 접두사 없이 저장
            gradients[param_key] = cp.asnumpy(grad_gamma)

    return gradients


def compute_choice_full_batch_gpu(
    all_ind_data: List[pd.DataFrame],
    all_lvs_gpu,  # CuPy array (326, 100, 5)
    params: Dict,
    all_weights_gpu,  # CuPy array (326, 100)
    choice_model,
    lv_names: List[str],
    iteration_logger=None,
    log_level: str = 'MINIMAL'
) -> Dict:
    """
    선택모델 Gradient - 완전 GPU Batch

    Returns:
        {'grad_intercept': (326,), 'grad_beta': (326, 3), 'grad_lambda_main': (326,), ...}
    """
    n_individuals, n_draws, n_lvs = all_lvs_gpu.shape

    # 파라미터 추출
    intercept = params['intercept']
    beta = params['beta']
    n_attributes = len(beta)

    # ✅ 모든 LV 주효과 vs 조절효과 vs 기본 모델 확인
    lambda_lv_keys = [key for key in params.keys() if key.startswith('lambda_') and key not in ['lambda_main']]

    all_lvs_as_main = len(lambda_lv_keys) > 1
    moderation_enabled = 'lambda_main' in params

    # 🔍 디버깅: params 키 확인
    if iteration_logger:
        iteration_logger.info(f"[GPU Choice Gradient] params 키: {list(params.keys())}")
        iteration_logger.info(f"[GPU Choice Gradient] lambda_lv_keys: {lambda_lv_keys}")
        iteration_logger.info(f"[GPU Choice Gradient] all_lvs_as_main: {all_lvs_as_main}")

    if all_lvs_as_main:
        # 모든 LV 주효과 모델
        lambda_lvs = {}
        for key in lambda_lv_keys:
            lv_name = key.replace('lambda_', '')
            lambda_lvs[lv_name] = params[key]
    elif moderation_enabled:
        # 조절효과 모델
        lambda_main = params['lambda_main']
        lambda_mod = {}
        for key in params:
            if key.startswith('lambda_mod_'):
                mod_lv_name = key.replace('lambda_mod_', '')
                lambda_mod[mod_lv_name] = params[key]
        main_lv = choice_model.config.main_lv
    else:
        # 기본 모델
        lambda_lv = params['lambda']
        # main_lv 찾기
        if hasattr(choice_model.config, 'main_lv'):
            main_lv = choice_model.config.main_lv
        else:
            main_lv = 'purchase_intention'  # 기본값

    choice_attributes = choice_model.config.choice_attributes

    # 모든 개인의 선택 데이터 추출
    n_situations = len(all_ind_data[0])
    all_choices = np.zeros((n_individuals, n_situations))
    all_attributes = np.zeros((n_individuals, n_situations, n_attributes))

    # 선택 변수 찾기
    choice_var = None
    for col in ['choice', 'chosen', 'choice_binary']:
        if col in all_ind_data[0].columns:
            choice_var = col
            break

    for ind_idx, ind_data in enumerate(all_ind_data):
        for sit_idx in range(n_situations):
            row = ind_data.iloc[sit_idx]
            all_choices[ind_idx, sit_idx] = row[choice_var]
            for attr_idx, attr in enumerate(choice_attributes):
                if attr in row.index and not pd.isna(row[attr]):
                    all_attributes[ind_idx, sit_idx, attr_idx] = row[attr]

    # GPU로 전송
    all_choices_gpu = cp.asarray(all_choices)  # (326, 18)
    all_attr_gpu = cp.asarray(all_attributes)  # (326, 18, 3)
    beta_gpu = cp.asarray(beta)  # (3,)

    # 속성 배치: (326, 1, 18, 3)
    attr_batch = all_attr_gpu[:, None, :, :]

    # 효용 계산: (326, 100, 18)
    # V = intercept + β'X
    V_batch = intercept + cp.sum(attr_batch * beta_gpu[None, None, None, :], axis=-1)

    if all_lvs_as_main:
        # ✅ 모든 LV 주효과: V += Σ(λ_i * LV_i)
        for lv_name, lambda_val in lambda_lvs.items():
            lv_idx = lv_names.index(lv_name)
            lv_batch = all_lvs_gpu[:, :, lv_idx:lv_idx+1]  # (326, 100, 1)
            V_batch = V_batch + lambda_val * lv_batch
    elif moderation_enabled:
        # 조절효과: V += λ_main * PI + Σ λ_mod_k * (PI × LV_k)
        main_lv_idx = lv_names.index(main_lv)
        main_lv_gpu = all_lvs_gpu[:, :, main_lv_idx]
        main_lv_batch = main_lv_gpu[:, :, None]  # (326, 100, 1)

        V_batch = V_batch + lambda_main * main_lv_batch

        for mod_lv_name, lambda_mod_val in lambda_mod.items():
            mod_lv_idx = lv_names.index(mod_lv_name)
            mod_lv_batch = all_lvs_gpu[:, :, mod_lv_idx:mod_lv_idx+1]  # (326, 100, 1)
            interaction = main_lv_batch * mod_lv_batch  # (326, 100, 1)
            V_batch = V_batch + lambda_mod_val * interaction
    else:
        # 기본 모델: V += λ * LV
        main_lv_idx = lv_names.index(main_lv)
        main_lv_gpu = all_lvs_gpu[:, :, main_lv_idx]
        main_lv_batch = main_lv_gpu[:, :, None]  # (326, 100, 1)
        V_batch = V_batch + lambda_lv * main_lv_batch

    # 확률 계산: (326, 100, 18)
    prob_batch = cp_ndtr(V_batch)
    prob_batch = cp.clip(prob_batch, 1e-10, 1 - 1e-10)
    phi_batch = cp_norm_pdf(V_batch)

    # 실제 선택에 따라: (326, 100, 18)
    choices_batch = all_choices_gpu[:, None, :]  # (326, 1, 18)
    prob_final = cp.where(choices_batch == 1, prob_batch, 1 - prob_batch)

    # Mills ratio: (326, 100, 18)
    mills_batch = phi_batch / prob_final
    sign_batch = cp.where(choices_batch == 1, 1.0, -1.0)

    # Weighted mills: (326, 100, 18)
    weighted_mills = all_weights_gpu[:, :, None] * sign_batch * mills_batch

    # Gradient 계산
    gradients = {}

    # intercept: (326,)
    gradients['intercept'] = cp.asnumpy(cp.sum(weighted_mills, axis=(1, 2)))

    # beta: (326, 3)
    grad_beta = cp.sum(weighted_mills[:, :, :, None] * attr_batch, axis=(1, 2))
    gradients['beta'] = cp.asnumpy(grad_beta)

    if all_lvs_as_main:
        # ✅ 모든 LV 주효과: lambda_{lv_name}
        for lv_name in lambda_lvs.keys():
            lv_idx = lv_names.index(lv_name)
            lv_batch = all_lvs_gpu[:, :, lv_idx:lv_idx+1]  # (326, 100, 1)
            grad_lambda_lv = cp.sum(weighted_mills * lv_batch, axis=(1, 2))
            gradients[f'lambda_{lv_name}'] = cp.asnumpy(grad_lambda_lv)

        # ✅ LV-Attribute 상호작용: gamma_{lv_name}_{attr_name}
        # params에서 gamma_ 파라미터 추출
        for key in params.keys():
            if key.startswith('gamma_') and '_to_' not in key:
                # gamma_purchase_intention_health_label → lv_name='purchase_intention', attr_name='health_label'
                # choice_attributes를 사용하여 파싱
                gamma_str = key.replace('gamma_', '')
                lv_name = None
                attr_name = None

                # 각 속성 이름으로 끝나는지 확인
                for attr in choice_attributes:
                    if gamma_str.endswith('_' + attr):
                        attr_name = attr
                        lv_name = gamma_str[:-(len(attr) + 1)]  # '_attr' 제거
                        break

                if lv_name and attr_name and lv_name in lv_names:
                    lv_idx = lv_names.index(lv_name)
                    attr_idx = choice_attributes.index(attr_name)
                    lv_batch = all_lvs_gpu[:, :, lv_idx]  # (326, 100)
                    attr_values = all_attr_gpu[:, :, attr_idx]  # (326, 18)
                    # (326, 100, 18) = (326, 100, 1) * (326, 1, 18)
                    interaction = lv_batch[:, :, None] * attr_values[:, None, :]  # (326, 100, 18)
                    grad_gamma = cp.sum(weighted_mills * interaction, axis=(1, 2))
                    gradients[key] = cp.asnumpy(grad_gamma)
    elif moderation_enabled:
        # 조절효과 모델
        main_lv_idx = lv_names.index(main_lv)
        main_lv_gpu = all_lvs_gpu[:, :, main_lv_idx]
        main_lv_batch = main_lv_gpu[:, :, None]  # (326, 100, 1)

        # lambda_main: (326,)
        gradients['lambda_main'] = cp.asnumpy(cp.sum(weighted_mills * main_lv_batch, axis=(1, 2)))

        # lambda_mod: (326,) for each moderator
        for mod_lv_name in lambda_mod.keys():
            mod_lv_idx = lv_names.index(mod_lv_name)
            mod_lv_batch = all_lvs_gpu[:, :, mod_lv_idx:mod_lv_idx+1]  # (326, 100, 1)
            interaction = main_lv_batch * mod_lv_batch  # (326, 100, 1)
            grad_lambda_mod = cp.sum(weighted_mills * interaction, axis=(1, 2))
            gradients[f'lambda_mod_{mod_lv_name}'] = cp.asnumpy(grad_lambda_mod)
    else:
        # 기본 모델
        main_lv_idx = lv_names.index(main_lv)
        main_lv_gpu = all_lvs_gpu[:, :, main_lv_idx]
        main_lv_batch = main_lv_gpu[:, :, None]  # (326, 100, 1)

        # lambda: (326,)
        gradients['lambda'] = cp.asnumpy(cp.sum(weighted_mills * main_lv_batch, axis=(1, 2)))

    return gradients


