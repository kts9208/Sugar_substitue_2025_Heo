"""
GPU 배치 처리 유틸리티 함수들

SimultaneousEstimator에서 사용할 GPU 배치 계산 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupyx.scipy.special import ndtr
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available. GPU acceleration disabled.")


def compute_measurement_batch_gpu(gpu_measurement_model,
                                   ind_data: pd.DataFrame,
                                   lvs_list: List[Dict[str, float]],
                                   params: Dict[str, Dict],
                                   iteration_logger=None) -> np.ndarray:
    """
    여러 draws에 대한 측정모델 우도를 GPU 배치로 계산

    Args:
        gpu_measurement_model: GPUMultiLatentMeasurement 인스턴스
        ind_data: 개인 데이터 (1행)
        lvs_list: 각 draw의 잠재변수 값 리스트 [{lv_name: value}, ...]
        params: 측정모델 파라미터 {lv_name: {'zeta': ..., 'tau': ...}}
        iteration_logger: 반복 로거 (상세 로깅용)

    Returns:
        각 draw의 로그우도 배열 (n_draws,)
    """
    if not CUPY_AVAILABLE or gpu_measurement_model is None:
        raise RuntimeError("GPU measurement model not available")

    # 파라미터 로깅 제거 (중복)

    # GPU 배치 처리
    ll_batch = gpu_measurement_model.log_likelihood_batch_draws(
        ind_data, lvs_list, params
    )

    return np.array(ll_batch)


def compute_choice_batch_gpu(ind_data: pd.DataFrame,
                             lvs_list: List[Dict[str, float]],
                             params: Dict[str, np.ndarray],
                             choice_model,
                             iteration_logger=None) -> np.ndarray:
    """
    여러 draws에 대한 선택모델 우도를 GPU 배치로 계산

    Args:
        ind_data: 개인의 선택 데이터 (여러 행)
        lvs_list: 각 draw의 잠재변수 값 리스트 [{lv_name: value}, ...]
        params: 선택모델 파라미터 {'intercept': ..., 'beta': ..., 'lambda': ...}
        choice_model: 선택모델 인스턴스
        iteration_logger: 반복 로거 (상세 로깅용)

    Returns:
        각 draw의 로그우도 배열 (n_draws,)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    n_draws = len(lvs_list)
    n_choice_situations = len(ind_data)
    
    # 파라미터 추출
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
    else:
        lambda_lv = params['lambda']

    # 선택 변수 찾기
    choice_var = None
    for col in ['choice', 'chosen', 'choice_binary']:
        if col in ind_data.columns:
            choice_var = col
            break

    if choice_var is None:
        raise ValueError(f"선택 변수를 찾을 수 없습니다. 가능한 컬럼: {ind_data.columns.tolist()}")

    # 속성 데이터 준비
    attributes = []
    choices = []
    valid_indices = []

    for idx in range(n_choice_situations):
        row = ind_data.iloc[idx]
        attr_values = [row[attr] for attr in choice_model.config.choice_attributes]
        choice_value = row[choice_var]

        # NaN 체크
        if not (pd.isna(choice_value) or any(pd.isna(v) for v in attr_values)):
            attributes.append(attr_values)
            choices.append(choice_value)
            valid_indices.append(idx)

    if len(attributes) == 0:
        # 모든 선택 상황이 NaN인 경우
        return np.full(n_draws, -1e10)

    attributes = np.array(attributes)  # (n_valid_situations, n_attributes)
    choices = np.array(choices)  # (n_valid_situations,)
    n_valid_situations = len(attributes)

    # GPU로 전송
    attributes_gpu = cp.asarray(attributes)
    choices_gpu = cp.asarray(choices)
    beta_gpu = cp.asarray(beta)

    # 첫 번째 draw에 대해서만 상세 로깅
    log_detail = iteration_logger is not None

    # 파라미터 로깅 제거 (중복)

    # 각 draw에 대한 우도 계산
    draw_lls = []

    for draw_idx in range(n_draws):
        lv_dict = lvs_list[draw_idx]

        # 내생 LV 값 (purchase_intention)
        if 'purchase_intention' in lv_dict:
            main_lv_value = lv_dict['purchase_intention']
        else:
            # 단일 LV인 경우
            main_lv_value = list(lv_dict.values())[0]

        # 상세 로깅 제거 (중복)

        # 효용 계산
        if moderation_enabled:
            # ✅ 조절효과 모델: V = intercept + beta*X + lambda_main*PI + Σ lambda_mod_k * (PI × LV_k)
            utility = intercept + cp.dot(attributes_gpu, beta_gpu) + lambda_main * main_lv_value

            # 조절효과 항 추가
            for mod_lv_name, lambda_mod_val in lambda_mod.items():
                if mod_lv_name in lv_dict:
                    mod_lv_value = lv_dict[mod_lv_name]
                    interaction = main_lv_value * mod_lv_value
                    utility = utility + lambda_mod_val * interaction
        else:
            # 기본 모델: V = intercept + beta*X + lambda*LV
            utility = intercept + cp.dot(attributes_gpu, beta_gpu) + lambda_lv * main_lv_value

        # 상세 로깅 제거 (중복)

        # 확률 계산: P = Φ(V) for choice=1, 1-Φ(V) for choice=0
        prob = ndtr(utility)

        # choice=0인 경우 1-prob
        prob = cp.where(choices_gpu == 1, prob, 1 - prob)

        # 확률 클리핑 (수치 안정성)
        prob = cp.clip(prob, 1e-10, 1 - 1e-10)

        # 로그우도 (모든 선택 상황의 곱 = 로그의 합)
        ll = cp.sum(cp.log(prob))

        # 유한성 체크
        if not cp.isfinite(ll):
            ll = -1e10

        draw_lls.append(float(ll))

    return np.array(draw_lls)


def compute_structural_batch_gpu(ind_data: pd.DataFrame,
                                 lvs_list: List[Dict[str, float]],
                                 params: Dict[str, np.ndarray],
                                 draws: np.ndarray,
                                 structural_model,
                                 iteration_logger=None) -> np.ndarray:
    """
    여러 draws에 대한 구조모델 우도를 GPU 배치로 계산

    Args:
        ind_data: 개인 데이터 (1행)
        lvs_list: 각 draw의 잠재변수 값 리스트 [{lv_name: value}, ...]
        params: 구조모델 파라미터 {'gamma_lv': ..., 'gamma_x': ...} or {'gamma_pred_to_target': ...}
        draws: 개인의 draws (n_draws, n_dimensions)
        structural_model: 구조모델 인스턴스
        iteration_logger: 반복 로거 (상세 로깅용)

    Returns:
        각 draw의 로그우도 배열 (n_draws,)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    n_draws = len(lvs_list)

    # ✅ 계층적 구조 확인
    if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
        return _compute_hierarchical_structural_batch_gpu(
            ind_data, lvs_list, params, draws, structural_model, iteration_logger
        )
    # 다중 잠재변수 구조모델인지 확인
    elif hasattr(structural_model, 'endogenous_lv'):
        # 병렬 구조 (하위 호환)
        return _compute_multi_latent_structural_batch_gpu(
            ind_data, lvs_list, params, draws, structural_model, iteration_logger
        )
    else:
        # 단일 잠재변수
        return _compute_single_latent_structural_batch_gpu(
            ind_data, lvs_list, params, draws, structural_model, iteration_logger
        )


def _compute_hierarchical_structural_batch_gpu(ind_data: pd.DataFrame,
                                               lvs_list: List[Dict[str, float]],
                                               params: Dict[str, Any],
                                               draws: np.ndarray,
                                               structural_model,
                                               iteration_logger=None) -> np.ndarray:
    """계층적 구조모델 우도 계산 (GPU 배치)"""

    n_draws = len(lvs_list)
    draw_lls = []

    log_detail = iteration_logger is not None

    # 계층적 경로 순회
    for draw_idx in range(n_draws):
        lv_dict = lvs_list[draw_idx]
        draw = draws[draw_idx]

        # 1차 LV draws
        n_first_order = len(structural_model.exogenous_lvs)
        exo_draws = draw[:n_first_order]

        # 2차+ LV 오차항
        higher_order_draws = {}
        higher_order_lvs = structural_model.get_higher_order_lvs()
        for i, lv_name in enumerate(higher_order_lvs):
            higher_order_draws[lv_name] = draw[n_first_order + i]

        # 각 경로에 대한 로그우도 계산
        total_ll = 0.0

        for path in structural_model.hierarchical_paths:
            target = path['target']
            predictors = path['predictors']

            # 예측값 계산
            lv_mean = 0.0
            for pred in predictors:
                param_name = f'gamma_{pred}_to_{target}'
                gamma = params[param_name]
                lv_mean += gamma * lv_dict[pred]

            # 실제값
            target_actual = lv_dict[target]

            # 잔차
            residual = target_actual - lv_mean

            # 로그우도: log N(target_actual | lv_mean, error_variance)
            error_var = structural_model.error_variance
            ll = -0.5 * np.log(2 * np.pi * error_var) - 0.5 * (residual**2) / error_var

            total_ll += ll

            if log_detail and draw_idx == 0:
                iteration_logger.info(f"  경로 {pred}->{target}: 예측={lv_mean:.4f}, 실제={target_actual:.4f}, LL={ll:.4f}")

        draw_lls.append(total_ll)

    return np.array(draw_lls)


def _compute_multi_latent_structural_batch_gpu(ind_data: pd.DataFrame,
                                               lvs_list: List[Dict[str, float]],
                                               params: Dict[str, np.ndarray],
                                               draws: np.ndarray,
                                               structural_model,
                                               iteration_logger=None) -> np.ndarray:
    """다중 잠재변수 구조모델 우도 계산 (GPU 배치)"""
    
    n_draws = len(lvs_list)
    gamma_lv = params['gamma_lv']
    gamma_x = params['gamma_x']
    
    # 공변량 효과 계산 (모든 draws에 동일)
    first_row = ind_data.iloc[0]
    x_effect = 0.0
    for i, var in enumerate(structural_model.covariates):
        if var in first_row.index:
            value = first_row[var]
            if pd.isna(value):
                value = 0.0
            x_effect += gamma_x[i] * value
    
    # GPU로 전송
    gamma_lv_gpu = cp.asarray(gamma_lv)
    
    draw_lls = []

    # 첫 번째 draw에 대해서만 상세 로깅
    log_detail = iteration_logger is not None

    # 파라미터 로깅 제거 (중복)

    for draw_idx in range(n_draws):
        lv_dict = lvs_list[draw_idx]
        draw = draws[draw_idx]

        # 외생 LV 효과
        n_exo = structural_model.n_exo
        exo_draws = draw[:n_exo]
        exo_draws_gpu = cp.asarray(exo_draws)

        lv_effect = float(cp.dot(gamma_lv_gpu, exo_draws_gpu))

        # 예측값
        endo_mean = lv_effect + x_effect

        # 실제값
        endo_actual = lv_dict[structural_model.endogenous_lv]

        # 잔차
        residual = endo_actual - endo_mean

        # 상세 로깅 제거 (중복)

        # 로그우도: log N(endo_actual | endo_mean, 1)
        ll = -0.5 * np.log(2 * np.pi) - 0.5 * residual**2

        draw_lls.append(ll)

    return np.array(draw_lls)


def _compute_single_latent_structural_batch_gpu(ind_data: pd.DataFrame,
                                                lvs_list: List[Dict[str, float]],
                                                params: Dict[str, np.ndarray],
                                                draws: np.ndarray,
                                                structural_model,
                                                iteration_logger=None) -> np.ndarray:
    """단일 잠재변수 구조모델 우도 계산 (GPU 배치)"""
    
    n_draws = len(lvs_list)
    gamma = params['gamma']
    
    # 공변량 효과 계산 (모든 draws에 동일)
    first_row = ind_data.iloc[0]
    x_effect = 0.0
    for i, var in enumerate(structural_model.config.sociodemographics):
        if var in first_row.index:
            value = first_row[var]
            if pd.isna(value):
                value = 0.0
            x_effect += gamma[i] * value
    
    draw_lls = []
    
    for draw_idx in range(n_draws):
        lv_dict = lvs_list[draw_idx]
        draw = draws[draw_idx]
        
        # 예측값
        lv_mean = x_effect
        
        # 실제값
        lv_actual = list(lv_dict.values())[0]
        
        # 잔차
        residual = lv_actual - lv_mean
        
        # 로그우도: log N(lv_actual | lv_mean, 1)
        ll = -0.5 * np.log(2 * np.pi) - 0.5 * residual**2
        
        draw_lls.append(ll)
    
    return np.array(draw_lls)

