"""
GPU 배치 처리 유틸리티 함수들

SimultaneousEstimator에서 사용할 GPU 배치 계산 함수들을 제공합니다.

Updated: 2025-11-20 - 에러 처리 개선
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

    # ✅ 모델 타입 확인: ASC 기반 multinomial logit vs binary probit
    use_alternative_specific = 'asc_sugar' in params or 'asc_A' in params

    # 파라미터 추출
    if use_alternative_specific:
        # Multinomial Logit with ASC
        asc_sugar = params.get('asc_sugar', params.get('asc_A', 0.0))
        asc_sugar_free = params.get('asc_sugar_free', params.get('asc_B', 0.0))

        # ✅ Beta 파라미터 (배열 또는 개별 키)
        if 'beta' in params:
            beta = params['beta']
        else:
            # 개별 beta 키에서 배열 생성 (choice_attributes 순서대로)
            if hasattr(choice_model, 'choice_attributes'):
                beta = np.array([params.get(f'beta_{attr}', 0.0) for attr in choice_model.choice_attributes])
            else:
                # choice_attributes가 없으면 알파벳 순서로
                beta_keys = sorted([k for k in params.keys() if k.startswith('beta_')])
                beta = np.array([params[k] for k in beta_keys])

        # ✅ 대안별 LV 계수 (theta_*)
        theta_params = {}  # {(alt, lv_name): theta_value}
        for key in params:
            if key.startswith('theta_sugar_'):
                lv_name = key.replace('theta_sugar_', '')
                theta_params[('sugar', lv_name)] = params[key]
            elif key.startswith('theta_sugar_free_'):
                lv_name = key.replace('theta_sugar_free_', '')
                theta_params[('sugar_free', lv_name)] = params[key]
            elif key.startswith('theta_A_'):
                lv_name = key.replace('theta_A_', '')
                theta_params[('A', lv_name)] = params[key]
            elif key.startswith('theta_B_'):
                lv_name = key.replace('theta_B_', '')
                theta_params[('B', lv_name)] = params[key]

        # ✅ 대안별 LV-Attribute 상호작용 (gamma_*)
        gamma_interactions = {}  # {(alt, lv_name, attr_name): gamma_value}
        for key in params:
            if key.startswith('gamma_sugar_') and not '_to_' in key:
                # gamma_sugar_purchase_intention_health_label → alt='sugar', lv_name='purchase_intention', attr_name='health_label'
                parts = key.replace('gamma_sugar_', '').rsplit('_', 1)
                if len(parts) == 2:
                    lv_name, attr_name = parts
                    gamma_interactions[('sugar', lv_name, attr_name)] = params[key]
            elif key.startswith('gamma_sugar_free_') and not '_to_' in key:
                parts = key.replace('gamma_sugar_free_', '').rsplit('_', 1)
                if len(parts) == 2:
                    lv_name, attr_name = parts
                    gamma_interactions[('sugar_free', lv_name, attr_name)] = params[key]
            elif key.startswith('gamma_A_') and not '_to_' in key:
                parts = key.replace('gamma_A_', '').rsplit('_', 1)
                if len(parts) == 2:
                    lv_name, attr_name = parts
                    gamma_interactions[('A', lv_name, attr_name)] = params[key]
            elif key.startswith('gamma_B_') and not '_to_' in key:
                parts = key.replace('gamma_B_', '').rsplit('_', 1)
                if len(parts) == 2:
                    lv_name, attr_name = parts
                    gamma_interactions[('B', lv_name, attr_name)] = params[key]
    else:
        # Binary Probit with intercept
        intercept = params['intercept']
        beta = params['beta']

        # ✅ 유연한 리스트 기반: lambda_* 파라미터 수집
        lambda_lvs = {}  # {lv_name: lambda_value}
        for key in params:
            if key.startswith('lambda_'):
                lv_name = key.replace('lambda_', '')
                lambda_lvs[lv_name] = params[key]

        # ✅ 유연한 리스트 기반: gamma_* 파라미터 수집 (LV-Attribute 상호작용)
        gamma_interactions = {}  # {(lv_name, attr_name): gamma_value}
        for key in params:
            if key.startswith('gamma_') and not '_to_' in key:
                # gamma_purchase_intention_health_label → lv_name='purchase_intention', attr_name='health_label'
                parts = key.replace('gamma_', '').rsplit('_', 1)
                if len(parts) == 2:
                    lv_name, attr_name = parts
                    gamma_interactions[(lv_name, attr_name)] = params[key]

    # 선택 변수 찾기
    choice_var = None
    for col in ['choice', 'chosen', 'choice_binary']:
        if col in ind_data.columns:
            choice_var = col
            break

    if choice_var is None:
        raise ValueError(f"선택 변수를 찾을 수 없습니다. 가능한 컬럼: {ind_data.columns.tolist()}")

    if use_alternative_specific:
        # ✅ Multinomial Logit: 대안별 데이터 준비
        # 데이터는 이미 long format (각 행이 하나의 대안)
        # sugar_content 컬럼으로 대안 구분

        # 유효한 choice set만 선택 (NaN 제외)
        valid_mask = ~ind_data[choice_var].isna()
        valid_data = ind_data[valid_mask].copy()

        if len(valid_data) == 0:
            return np.full(n_draws, -1e10)

        # 속성 데이터 추출
        attributes = valid_data[choice_model.config.choice_attributes].values
        choices = valid_data[choice_var].values

        # sugar_content 추출 (대안 구분용)
        if 'sugar_content' in valid_data.columns:
            sugar_contents = valid_data['sugar_content'].values
        elif 'alternative' in valid_data.columns:
            sugar_contents = valid_data['alternative'].values
        else:
            raise ValueError("sugar_content 또는 alternative 컬럼이 없습니다.")

        # GPU로 전송
        attributes_gpu = cp.asarray(attributes)
        choices_gpu = cp.asarray(choices)
        beta_gpu = cp.asarray(beta)

    else:
        # ✅ Binary Probit: 기존 방식
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

    # 속성 이름 리스트 (choice_model에서 가져오기)
    choice_attributes = choice_model.config.choice_attributes

    # 각 draw에 대한 우도 계산
    draw_lls = []

    for draw_idx in range(n_draws):
        lv_dict = lvs_list[draw_idx]

        if use_alternative_specific:
            # ✅ Multinomial Logit: 대안별 효용 계산
            # V_alt = ASC_alt + β*X_alt + Σ(θ_alt_i * LV_i) + Σ(γ_alt_ij * LV_i * X_j)

            utility = cp.zeros(len(attributes_gpu))

            for i in range(len(attributes_gpu)):
                sugar_content = sugar_contents[i]

                if pd.isna(sugar_content):
                    # opt-out (reference alternative)
                    utility[i] = 0.0
                elif sugar_content == '알반당' or sugar_content == 'A':
                    # 일반당 대안
                    utility[i] = asc_sugar + cp.dot(attributes_gpu[i], beta_gpu)

                    # 대안별 LV 주효과
                    for (alt, lv_name), theta_val in theta_params.items():
                        if (alt == 'sugar' or alt == 'A') and lv_name in lv_dict:
                            utility[i] += theta_val * lv_dict[lv_name]

                    # 대안별 LV-Attribute 상호작용
                    for (alt, lv_name, attr_name), gamma_val in gamma_interactions.items():
                        if (alt == 'sugar' or alt == 'A') and lv_name in lv_dict and attr_name in choice_attributes:
                            attr_idx = choice_attributes.index(attr_name)
                            utility[i] += gamma_val * lv_dict[lv_name] * attributes_gpu[i, attr_idx]

                elif sugar_content == '무설탕' or sugar_content == 'B':
                    # 무설탕 대안
                    utility[i] = asc_sugar_free + cp.dot(attributes_gpu[i], beta_gpu)

                    # 대안별 LV 주효과
                    for (alt, lv_name), theta_val in theta_params.items():
                        if (alt == 'sugar_free' or alt == 'B') and lv_name in lv_dict:
                            utility[i] += theta_val * lv_dict[lv_name]

                    # 대안별 LV-Attribute 상호작용
                    for (alt, lv_name, attr_name), gamma_val in gamma_interactions.items():
                        if (alt == 'sugar_free' or alt == 'B') and lv_name in lv_dict and attr_name in choice_attributes:
                            attr_idx = choice_attributes.index(attr_name)
                            utility[i] += gamma_val * lv_dict[lv_name] * attributes_gpu[i, attr_idx]

            # Multinomial Logit 확률 계산
            # P_i = exp(V_i) / Σ_j exp(V_j)
            # choice set별로 그룹화 (3개 대안씩)
            n_alternatives = 3
            n_choice_sets = len(utility) // n_alternatives

            ll = 0.0
            for cs_idx in range(n_choice_sets):
                start_idx = cs_idx * n_alternatives
                end_idx = start_idx + n_alternatives

                # 해당 choice set의 효용
                V_cs = utility[start_idx:end_idx]

                # 수치 안정성을 위해 최대값 빼기
                V_max = cp.max(V_cs)
                exp_V = cp.exp(V_cs - V_max)
                sum_exp_V = cp.sum(exp_V)

                # 선택된 대안의 확률
                # ✅ 최적화: cp.where() 대신 argmax 사용 (더 빠름)
                choices_cs = choices_gpu[start_idx:end_idx]  # (3,)
                chosen_alt_idx = int(cp.argmax(choices_cs))  # 선택된 대안의 인덱스 (0, 1, or 2)

                # 에러 체크: 선택된 대안이 없는 경우
                if float(choices_cs[chosen_alt_idx]) != 1.0:
                    print(f"\n{'='*80}")
                    print(f"❌ 선택모델 우도 계산 에러: Choice set {cs_idx}에 선택된 대안이 없습니다!")
                    print(f"{'='*80}")
                    print(f"Draw index: {draw_idx}/{n_draws}")
                    print(f"Choice set index: {cs_idx}/{n_choice_sets}")
                    print(f"Choice set range: [{start_idx}:{end_idx}]")
                    print(f"Choices in this set: {cp.asnumpy(choices_cs)}")
                    print(f"Utilities: {cp.asnumpy(V_cs)}")
                    print(f"개인 데이터 shape: {ind_data.shape}")
                    print(f"전체 choices shape: {choices_gpu.shape}")
                    print(f"{'='*80}\n")
                    raise ValueError(f"Choice set {cs_idx}에 선택된 대안이 없습니다!")

                prob_chosen = exp_V[chosen_alt_idx] / sum_exp_V

                # 로그우도 누적
                ll += cp.log(cp.clip(prob_chosen, 1e-10, 1.0))

        else:
            # ✅ Binary Probit: 기존 방식
            # V = intercept + beta*X + Σ(lambda_i * LV_i) + Σ(gamma_ij * LV_i * X_j)
            utility = intercept + cp.dot(attributes_gpu, beta_gpu)

            # 주효과: Σ(lambda_i * LV_i)
            for lv_name, lambda_val in lambda_lvs.items():
                if lv_name in lv_dict:
                    lv_value = lv_dict[lv_name]
                    utility = utility + lambda_val * lv_value

            # LV-Attribute 상호작용: Σ(gamma_ij * LV_i * X_j)
            for (lv_name, attr_name), gamma_val in gamma_interactions.items():
                if lv_name in lv_dict:
                    lv_value = lv_dict[lv_name]
                    # attr_name에 해당하는 속성 인덱스 찾기
                    if attr_name in choice_attributes:
                        attr_idx = choice_attributes.index(attr_name)
                        # attributes_gpu[:, attr_idx]는 (n_valid_situations,) 형태
                        attr_values = attributes_gpu[:, attr_idx]
                        interaction = lv_value * attr_values
                        utility = utility + gamma_val * interaction

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
            print(f"\n{'='*80}")
            print(f"❌ 선택모델 우도가 비유한값(inf/nan)입니다!")
            print(f"{'='*80}")
            print(f"Draw index: {draw_idx}/{n_draws}")
            print(f"Log-likelihood value: {ll}")
            print(f"개인 데이터 shape: {ind_data.shape}")
            if use_alternative_specific:
                print(f"Choice sets: {n_choice_sets}")
                print(f"Utilities (first 10): {cp.asnumpy(utility[:10])}")
                print(f"Utilities (last 10): {cp.asnumpy(utility[-10:])}")
                print(f"Utilities stats: min={float(cp.min(utility)):.4f}, max={float(cp.max(utility)):.4f}, mean={float(cp.mean(utility)):.4f}")
            else:
                print(f"Utilities (first 10): {cp.asnumpy(utility[:10])}")
                print(f"Probabilities (first 10): {cp.asnumpy(prob[:10])}")
            print(f"LV values: {lv_dict}")
            print(f"Parameters:")
            if use_alternative_specific:
                print(f"  asc_sugar: {asc_sugar}")
                print(f"  asc_sugar_free: {asc_sugar_free}")
                print(f"  beta: {beta}")
                print(f"  theta_params: {theta_params}")
            else:
                print(f"  intercept: {intercept}")
                print(f"  beta: {beta}")
                print(f"  lambda_lvs: {lambda_lvs}")
            print(f"{'='*80}\n")
            raise ValueError(f"선택모델 우도가 비유한값입니다: {ll}")

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

        # 🔍 디버깅: 첫 번째 draw에 대한 상세 로깅
        if log_detail and draw_idx == 0:
            iteration_logger.info(f"\n[구조모델 우도 계산 - Draw #0]")
            iteration_logger.info(f"  error_variance: {structural_model.error_variance:.4f}")
            iteration_logger.info(f"  경로 수: {len(structural_model.hierarchical_paths)}")

        for path_idx, path in enumerate(structural_model.hierarchical_paths):
            target = path['target']
            predictors = path['predictors']

            # 예측값 계산
            lv_mean = 0.0
            gamma_details = []
            for pred in predictors:
                param_name = f'gamma_{pred}_to_{target}'
                gamma = params[param_name]
                pred_lv = lv_dict[pred]
                contribution = gamma * pred_lv
                lv_mean += contribution
                gamma_details.append(f"{param_name}={gamma:.4f} × {pred}={pred_lv:.4f} = {contribution:.4f}")

            # 실제값
            target_actual = lv_dict[target]

            # 잔차
            residual = target_actual - lv_mean

            # 로그우도: log N(target_actual | lv_mean, error_variance)
            error_var = structural_model.error_variance
            ll = -0.5 * np.log(2 * np.pi * error_var) - 0.5 * (residual**2) / error_var

            total_ll += ll

            if log_detail and draw_idx == 0:
                iteration_logger.info(f"\n  [경로 #{path_idx+1}] {predictors} → {target}")
                for detail in gamma_details:
                    iteration_logger.info(f"    {detail}")
                iteration_logger.info(f"    lv_mean (합계) = {lv_mean:.4f}")
                iteration_logger.info(f"    target_actual = {target_actual:.4f}")
                iteration_logger.info(f"    residual = {residual:.4f}")
                iteration_logger.info(f"    ll = logpdf({target_actual:.4f} | μ={lv_mean:.4f}, σ²={error_var:.4f}) = {ll:.4f}")

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

