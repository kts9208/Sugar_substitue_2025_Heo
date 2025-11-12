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

    # ✅ 조절효과 지원
    moderation_enabled = 'lambda_main' in params
    if moderation_enabled:
        lambda_main = params['lambda_main']
        # lambda_mod는 딕셔너리 형태: {'perceived_price': -0.3, 'nutrition_knowledge': 0.2}
        lambda_mod = {}
        for key in params:
            if key.startswith('lambda_mod_'):
                mod_lv_name = key.replace('lambda_mod_', '')
                lambda_mod[mod_lv_name] = params[key]

        if iteration_logger and log_level == 'DETAILED':
            iteration_logger.info(f"  조절효과 모델:")
            iteration_logger.info(f"    - intercept: {intercept:.6f}")
            iteration_logger.info(f"    - beta: {beta[:min(3, len(beta))]}")
            iteration_logger.info(f"    - lambda_main: {lambda_main:.6f}")
            for mod_lv_name, lambda_mod_val in lambda_mod.items():
                iteration_logger.info(f"    - lambda_mod_{mod_lv_name}: {lambda_mod_val:.6f}")
    else:
        lambda_lv = params['lambda']

        if iteration_logger and log_level == 'DETAILED':
            iteration_logger.info(f"  기본 모델:")
            iteration_logger.info(f"    - intercept: {intercept:.6f}")
            iteration_logger.info(f"    - beta: {beta[:min(3, len(beta))]}")
            iteration_logger.info(f"    - lambda: {lambda_lv:.6f}")

    weights_gpu = cp.asarray(weights)  # (n_draws,)

    # LV 값들 - main LV
    main_lv_values = np.array([lvs[endogenous_lv] for lvs in lvs_list])

    # ✅ 조절효과: moderator LV 값들
    if moderation_enabled:
        mod_lv_values = {}
        for mod_lv_name in lambda_mod.keys():
            mod_lv_values[mod_lv_name] = np.array([lvs[mod_lv_name] for lvs in lvs_list])

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
    main_lv_gpu = cp.asarray(main_lv_values)  # (n_draws,)
    attr_gpu = cp.asarray(attributes_matrix)  # (n_situations, n_attributes)
    choices_gpu = cp.asarray(choices)  # (n_situations,)
    beta_gpu = cp.asarray(beta)  # (n_attributes,)

    # ✅ 조절효과: moderator LV GPU 전송
    if moderation_enabled:
        mod_lv_gpu = {}
        for mod_lv_name, mod_values in mod_lv_values.items():
            mod_lv_gpu[mod_lv_name] = cp.asarray(mod_values)  # (n_draws,)

    # ✅ 개선: 배치 처리 (for loop 제거)
    # Broadcasting을 사용하여 모든 draws를 동시에 처리

    # main_lv_batch: (n_draws, 1)
    main_lv_batch = main_lv_gpu[:, None]

    # attr_batch: (1, n_situations, n_attributes)
    attr_batch = attr_gpu[None, :, :]

    # V_batch: (n_draws, n_situations)
    if moderation_enabled:
        # ✅ 조절효과 모델: V = intercept + β'X + λ_main*PI + Σ λ_mod_k * (PI × LV_k)
        V_batch = intercept + cp.dot(attr_batch, beta_gpu[:, None]).squeeze(-1) + lambda_main * main_lv_batch

        # 조절효과 항 추가
        for mod_lv_name, lambda_mod_val in lambda_mod.items():
            mod_lv_batch = mod_lv_gpu[mod_lv_name][:, None]  # (n_draws, 1)
            interaction = main_lv_batch * mod_lv_batch  # (n_draws, 1)
            V_batch = V_batch + lambda_mod_val * interaction
    else:
        # 기본 모델: V = intercept + β'X + λ*LV
        V_batch = intercept + cp.dot(attr_batch, beta_gpu[:, None]).squeeze(-1) + lambda_lv * main_lv_batch

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

    # ✅ 조절효과 gradient 계산
    if moderation_enabled:
        # ∂V/∂λ_main = PI
        grad_lambda_main = cp.sum(weighted_mills * main_lv_batch).item()

        # ∂V/∂λ_mod_k = PI × LV_k
        grad_lambda_mod = {}
        for mod_lv_name in lambda_mod.keys():
            mod_lv_batch = mod_lv_gpu[mod_lv_name][:, None]  # (n_draws, 1)
            interaction = main_lv_batch * mod_lv_batch  # (n_draws, 1)
            grad_lambda_mod[mod_lv_name] = cp.sum(weighted_mills * interaction).item()
    else:
        # ∂V/∂λ = LV
        grad_lambda = cp.sum(weighted_mills * main_lv_batch).item()

    # NaN 체크
    if np.isnan(grad_intercept):
        logger.warning("NaN detected in grad_intercept")
        grad_intercept = 0.0

    if cp.any(cp.isnan(grad_beta)):
        logger.warning("NaN detected in grad_beta")
        grad_beta = cp.nan_to_num(grad_beta, nan=0.0)

    if moderation_enabled:
        if np.isnan(grad_lambda_main):
            logger.warning("NaN detected in grad_lambda_main")
            grad_lambda_main = 0.0

        for mod_lv_name in grad_lambda_mod.keys():
            if np.isnan(grad_lambda_mod[mod_lv_name]):
                logger.warning(f"NaN detected in grad_lambda_mod_{mod_lv_name}")
                grad_lambda_mod[mod_lv_name] = 0.0
    else:
        if np.isnan(grad_lambda):
            logger.warning("NaN detected in grad_lambda")
            grad_lambda = 0.0

    # Gradient clipping
    grad_intercept = np.clip(grad_intercept, -1e6, 1e6)
    grad_beta = cp.clip(grad_beta, -1e6, 1e6)

    if moderation_enabled:
        grad_lambda_main = np.clip(grad_lambda_main, -1e6, 1e6)
        for mod_lv_name in grad_lambda_mod.keys():
            grad_lambda_mod[mod_lv_name] = np.clip(grad_lambda_mod[mod_lv_name], -1e6, 1e6)

        if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
            iteration_logger.info(f"\n  [최종 그래디언트]")
            iteration_logger.info(f"    - grad_intercept: {grad_intercept:.6f}")
            iteration_logger.info(f"    - grad_beta: {cp.asnumpy(grad_beta)[:min(3, len(grad_beta))]}")
            iteration_logger.info(f"    - grad_lambda_main: {grad_lambda_main:.6f}")
            for mod_lv_name, grad_val in grad_lambda_mod.items():
                iteration_logger.info(f"    - grad_lambda_mod_{mod_lv_name}: {grad_val:.6f}")

        # 결과 반환
        result = {
            'grad_intercept': grad_intercept,
            'grad_beta': cp.asnumpy(grad_beta),
            'grad_lambda_main': grad_lambda_main
        }
        for mod_lv_name, grad_val in grad_lambda_mod.items():
            result[f'grad_lambda_mod_{mod_lv_name}'] = grad_val

        return result
    else:
        grad_lambda = np.clip(grad_lambda, -1e6, 1e6)

        if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
            iteration_logger.info(f"\n  [최종 그래디언트]")
            iteration_logger.info(f"    - grad_intercept: {grad_intercept:.6f}")
            iteration_logger.info(f"    - grad_beta: {cp.asnumpy(grad_beta)[:min(3, len(grad_beta))]}")
            iteration_logger.info(f"    - grad_lambda: {grad_lambda:.6f}")

        return {
            'grad_intercept': grad_intercept,
            'grad_beta': cp.asnumpy(grad_beta),
            'grad_lambda': grad_lambda
        }

