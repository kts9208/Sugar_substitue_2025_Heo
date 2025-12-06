"""
완전 병렬 GPU Gradient 계산 - Advanced Indexing 사용

측정모델의 모든 지표(38개)를 한 번에 계산하는 완전 병렬 구현
Zero-padding 없이 Advanced Indexing으로 각 지표에 맞는 LV를 자동 선택

성능:
- GPU 커널 호출: 1번 (기존 38번 → 38배 개선)
- 메모리: 9.45 MB (Zero-padding 24.87 MB 대비 62% 절약)
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def compute_measurement_full_parallel_gpu(
    gpu_measurement_model,
    all_ind_data: List[pd.DataFrame],
    all_lvs_gpu,  # CuPy array (326, 100, 5)
    params_dict: Dict,
    all_weights_gpu,  # CuPy array (326, 100)
    lv_names: List[str],
    iteration_logger=None,
    log_level: str = 'MINIMAL',
    measurement_params_fixed: bool = False
) -> Dict:
    """
    측정모델 Gradient - 완전 병렬 (모든 지표 한 번에)

    ⚠️ 주의: 동시추정에서는 이 함수를 호출하지 않습니다 (측정모델 고정)
    이 함수는 순차추정 또는 CFA에서만 사용됩니다.

    Advanced Indexing을 사용하여 38개 지표를 1번의 GPU 커널 호출로 계산

    Args:
        gpu_measurement_model: GPU 측정모델
        all_ind_data: 모든 개인 데이터 (326개)
        all_lvs_gpu: 모든 LV 값 (326, 100, 5)
        params_dict: 파라미터 딕셔너리
        all_weights_gpu: 가중치 (326, 100)
        lv_names: LV 이름 리스트
        iteration_logger: 로거
        log_level: 로깅 레벨
        measurement_params_fixed: 측정모델 파라미터 고정 여부 (순차추정용)

    Returns:
        {lv_name: {'zeta': (326, n_indicators), 'sigma_sq': (326, n_indicators)}}
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for full parallel computation")

    # ✅ 측정모델 파라미터 고정 시 그래디언트를 0으로 반환 (순차추정용)
    if measurement_params_fixed:
        gradients = {}
        for lv_name in lv_names:
            config = gpu_measurement_model.models[lv_name].config
            n_ind = len(config.indicators)
            n_individuals = len(all_ind_data)

            # fix_first_loading 고려
            fix_first_loading = getattr(config, 'fix_first_loading', True)
            n_zeta = n_ind - 1 if fix_first_loading else n_ind

            gradients[lv_name] = {
                'zeta': np.zeros((n_individuals, n_zeta)),
                'sigma_sq': np.zeros((n_individuals, n_ind))
            }

        if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
            iteration_logger.info("  ✅ 측정모델 파라미터 고정: 그래디언트 = 0")

        return gradients
    
    start_time = time.time()
    
    n_individuals, n_draws, n_lvs = all_lvs_gpu.shape
    
    # 1. 지표-LV 매핑 배열 생성
    indicator_to_lv = []
    indicator_names = []
    lv_for_indicator = []  # 각 지표가 속한 LV 이름
    
    for lv_idx, lv_name in enumerate(lv_names):
        config = gpu_measurement_model.models[lv_name].config
        n_indicators = len(config.indicators)
        
        indicator_to_lv.extend([lv_idx] * n_indicators)
        indicator_names.extend(config.indicators)
        lv_for_indicator.extend([lv_name] * n_indicators)
    
    n_total_indicators = len(indicator_names)

    # 파라미터 수 계산 (fix_first_loading 고려)
    n_zeta_params = sum(len(gpu_measurement_model.models[lv].config.indicators) - 1
                        for lv in lv_names)  # 각 LV의 첫 번째 제외
    n_sigma_sq_params = n_total_indicators  # 모든 sigma_sq
    n_total_params = n_zeta_params + n_sigma_sq_params

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  완전 병렬 측정모델 Gradient 계산 시작\n"
            f"    - 총 지표 수: {n_total_indicators}개\n"
            f"    - 총 파라미터 수: {n_total_params}개 ({n_zeta_params} zeta + {n_sigma_sq_params} sigma_sq)\n"
            f"    - LV 수: {n_lvs}개\n"
            f"    - 매핑: {dict(zip(lv_names, [indicator_to_lv.count(i) for i in range(n_lvs)]))}"
        )
    
    # 2. 모든 관측값 수집 (326, 38)
    all_y = np.zeros((n_individuals, n_total_indicators))
    
    for ind_idx, ind_data in enumerate(all_ind_data):
        row = ind_data.iloc[0]
        for i, indicator in enumerate(indicator_names):
            if indicator in row.index and not pd.isna(row[indicator]):
                all_y[ind_idx, i] = row[indicator]
    
    # 3. 모든 파라미터 수집 (38,)
    all_zeta = []
    all_sigma_sq = []
    
    for lv_name in lv_names:
        all_zeta.extend(params_dict['measurement'][lv_name]['zeta'])
        all_sigma_sq.extend(params_dict['measurement'][lv_name]['sigma_sq'])
    
    all_zeta = np.array(all_zeta)
    all_sigma_sq = np.array(all_sigma_sq)
    
    # 4. GPU로 전송
    all_y_gpu = cp.asarray(all_y)  # (326, 38)
    all_zeta_gpu = cp.asarray(all_zeta)  # (38,)
    all_sigma_sq_gpu = cp.asarray(all_sigma_sq)  # (38,)
    indicator_to_lv_gpu = cp.asarray(indicator_to_lv)  # (38,)
    
    if iteration_logger and log_level == 'DETAILED':
        iteration_logger.info(
            f"  데이터 준비 완료\n"
            f"    - all_y: {all_y_gpu.shape}\n"
            f"    - all_zeta: {all_zeta_gpu.shape}\n"
            f"    - all_sigma_sq: {all_sigma_sq_gpu.shape}\n"
            f"    - indicator_to_lv: {indicator_to_lv_gpu.shape}"
        )
    
    # 5. ✨ Advanced Indexing: 각 지표에 맞는 LV 선택
    # all_lvs_gpu: (326, 100, 5)
    # indicator_to_lv: [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,3,3,...,3,4,4,4]
    # → lv_for_indicators: (326, 100, 38)
    lv_for_indicators = all_lvs_gpu[:, :, indicator_to_lv_gpu]
    
    if iteration_logger and log_level == 'DETAILED':
        iteration_logger.info(
            f"  Advanced Indexing 완료\n"
            f"    - lv_for_indicators: {lv_for_indicators.shape}\n"
            f"    - 각 지표마다 해당 LV 값이 자동 선택됨"
        )
    
    # 6. 완전 병렬 Gradient 계산
    # 예측값: (326, 100, 38)
    y_pred_all = all_zeta_gpu[None, None, :] * lv_for_indicators
    
    # 잔차: (326, 100, 38)
    residual_all = all_y_gpu[:, None, :] - y_pred_all
    
    # Gradient (각 draw): (326, 100, 38)
    grad_zeta_batch = (residual_all * lv_for_indicators / 
                       all_sigma_sq_gpu[None, None, :])
    
    grad_sigma_sq_batch = (-0.5 / all_sigma_sq_gpu[None, None, :] + 
                           0.5 * (residual_all ** 2) / (all_sigma_sq_gpu[None, None, :] ** 2))
    
    # 가중평균: (326, 38)
    grad_zeta_all = cp.sum(all_weights_gpu[:, :, None] * grad_zeta_batch, axis=1)
    grad_sigma_sq_all = cp.sum(all_weights_gpu[:, :, None] * grad_sigma_sq_batch, axis=1)
    
    if iteration_logger and log_level == 'DETAILED':
        iteration_logger.info(
            f"  Gradient 계산 완료\n"
            f"    - grad_zeta_all: {grad_zeta_all.shape}\n"
            f"    - grad_sigma_sq_all: {grad_sigma_sq_all.shape}"
        )
    
    # 7. LV별로 분리
    gradients = {}
    idx = 0
    
    for lv_name in lv_names:
        config = gpu_measurement_model.models[lv_name].config
        n_ind = len(config.indicators)
        
        # fix_first_loading 고려
        fix_first_loading = getattr(config, 'fix_first_loading', True)
        
        if fix_first_loading:
            # 첫 번째 zeta는 1.0으로 고정 (gradient 제외)
            grad_zeta_lv = cp.asnumpy(grad_zeta_all[:, idx+1:idx+n_ind])
        else:
            grad_zeta_lv = cp.asnumpy(grad_zeta_all[:, idx:idx+n_ind])
        
        gradients[lv_name] = {
            'zeta': grad_zeta_lv,
            'sigma_sq': cp.asnumpy(grad_sigma_sq_all[:, idx:idx+n_ind])
        }
        
        idx += n_ind
    
    elapsed = time.time() - start_time

    # 실제 파라미터 수 계산 (fix_first_loading 고려)
    n_zeta_params = sum(len(gpu_measurement_model.models[lv].config.indicators) - 1
                        for lv in lv_names)  # 각 LV의 첫 번째 제외
    n_sigma_sq_params = n_total_indicators  # 모든 sigma_sq
    n_total_params = n_zeta_params + n_sigma_sq_params

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  ✅ 완전 병렬 측정모델 Gradient 계산 완료 ({elapsed:.4f}초)\n"
            f"    - GPU 커널 호출: 1번\n"
            f"    - 지표 수: {n_total_indicators}개\n"
            f"    - 파라미터 수: {n_total_params}개 ({n_zeta_params} zeta + {n_sigma_sq_params} sigma_sq)\n"
            f"    - 계산량: {n_individuals} × {n_draws} × {n_total_indicators} × 2 = "
            f"{n_individuals * n_draws * n_total_indicators * 2:,}개 (zeta + sigma_sq)"
        )

    return gradients


def compute_all_individuals_gradients_full_parallel_gpu(
    gpu_measurement_model,
    all_ind_data: List[pd.DataFrame],
    all_ind_draws: np.ndarray,
    params_dict: Dict,
    measurement_model,
    structural_model,
    choice_model,
    iteration_logger=None,
    log_level: str = 'MINIMAL',
    use_scaling: bool = False  # ✅ 측정모델 우도 스케일링 사용 여부
) -> List[Dict]:
    """
    모든 개인의 gradient를 완전 병렬로 계산 (Advanced Indexing 사용)

    ✅ 동시추정 전용: 측정모델 그래디언트는 계산하지 않음 (고정 파라미터)

    구조모델: 기존 방식 사용
    선택모델: 기존 방식 사용

    Args:
        gpu_measurement_model: GPU 측정모델
        all_ind_data: 모든 개인의 데이터 리스트
        all_ind_draws: 모든 개인의 draws (N, n_draws, n_dims)
        params_dict: 파라미터 딕셔너리
        measurement_model: 측정모델
        structural_model: 구조모델
        choice_model: 선택모델
        iteration_logger: 로거
        log_level: 로깅 레벨

    Returns:
        개인별 gradient 딕셔너리 리스트
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    start_time = time.time()
    
    n_individuals, n_draws, n_dims = all_ind_draws.shape
    
    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"\n{'='*70}\n"
            f"완전 병렬 Gradient 계산 (Advanced Indexing)\n"
            f"{'='*70}\n"
            f"  개인 수: {n_individuals}\n"
            f"  Draws: {n_draws}\n"
            f"  차원: {n_dims}"
        )
    
    # LV 이름 추출
    lv_names = list(params_dict['measurement'].keys())
    
    # 1. 모든 개인의 LV 값 계산 (326, 100, 5)
    lv_start = time.time()
    all_lvs_list = []

    is_hierarchical = hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical

    for ind_idx, ind_data in enumerate(all_ind_data):
        ind_draws = all_ind_draws[ind_idx]  # (100, 6)

        # 각 draw에 대한 LV 값 계산
        lvs_for_draws = []
        for draw_idx in range(n_draws):
            draw = ind_draws[draw_idx]

            if is_hierarchical:
                # 계층적 구조: exo_draws와 higher_order_draws 분리
                n_first_order = len(structural_model.exogenous_lvs)
                exo_draws = draw[:n_first_order]

                # 2차+ LV 오차항
                higher_order_draws = {}
                higher_order_lvs = structural_model.get_higher_order_lvs()
                for i, lv_name in enumerate(higher_order_lvs):
                    higher_order_draws[lv_name] = draw[n_first_order + i]

                lv_values = structural_model.predict(
                    ind_data, exo_draws, params_dict['structural'],
                    higher_order_draws=higher_order_draws
                )
            else:
                # 병렬 구조 (하위 호환)
                lv_values = structural_model.predict(ind_data, draw, params_dict['structural'])

            lvs_for_draws.append([lv_values[lv_name] for lv_name in lv_names])

        all_lvs_list.append(lvs_for_draws)

    all_lvs_array = np.array(all_lvs_list)  # (326, 100, 5)
    all_lvs_gpu = cp.asarray(all_lvs_array)

    lv_time = time.time() - lv_start

    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"  LV 계산 완료 ({lv_time:.3f}초)\n"
            f"    - all_lvs shape: {all_lvs_array.shape}"
        )
        # 첫 번째 개인의 첫 번째 draw LV 값 출력 (디버깅)
        if len(all_lvs_list) > 0 and len(all_lvs_list[0]) > 0:
            first_lv_values = all_lvs_list[0][0]
            iteration_logger.info(f"  [디버깅] 첫 번째 개인, 첫 번째 draw LV 값:")
            for lv_idx, lv_name in enumerate(lv_names):
                iteration_logger.info(f"    {lv_name}: {first_lv_values[lv_idx]:.6f}")
    
    # 2. 가중치 계산 (균등 가중치)
    all_weights = np.ones((n_individuals, n_draws)) / n_draws
    all_weights_gpu = cp.asarray(all_weights)

    # ✅ 측정모델 우도 스케일링 가중치 계산
    # Forward pass와 동일한 스케일링을 Backward pass에도 적용
    measurement_weight = 1.0
    if use_scaling:
        n_measurement_indicators = 0
        if hasattr(gpu_measurement_model, 'models'):
            for lv_name, model in gpu_measurement_model.models.items():
                n_measurement_indicators += len(model.config.indicators)

        if n_measurement_indicators > 0:
            measurement_weight = 1.0 / n_measurement_indicators

            if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
                iteration_logger.info(
                    f"\n{'='*80}\n"
                    f"📊 Gradient 스케일링 설정 (완전 병렬)\n"
                    f"{'='*80}\n"
                    f"  측정모델 지표 수: {n_measurement_indicators}개\n"
                    f"  측정모델 가중치 (ω): {measurement_weight:.6f}\n"
                    f"  ∇LL_total = ∇LL_choice + {measurement_weight:.6f} × ∇LL_measurement\n"
                    f"{'='*80}"
                )

    # ✅ 동시추정: 측정모델 그래디언트 계산 제외 (고정 파라미터)
    # 측정모델 그래디언트는 빈 딕셔너리로 설정
    meas_grads = {}
    meas_time = 0.0

    # 4. 구조모델 Gradient (기존 방식)
    from .gpu_gradient_batch import compute_structural_full_batch_gpu

    struct_start = time.time()
    struct_grads = compute_structural_full_batch_gpu(
        all_ind_data,
        all_lvs_gpu,
        params_dict,
        all_weights_gpu,
        structural_model,
        choice_model,
        gpu_measurement_model,
        lv_names,
        iteration_logger,
        log_level,
        measurement_weight=measurement_weight  # ✅ 스케일링 가중치 전달
    )
    struct_time = time.time() - struct_start
    
    # 5. 선택모델 Gradient (기존 방식)
    from .gpu_gradient_batch import compute_choice_full_batch_gpu
    
    choice_start = time.time()
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
    choice_time = time.time() - choice_start
    
    # 6. 개인별 gradient 딕셔너리로 변환
    all_individual_gradients = []
    
    for ind_idx in range(n_individuals):
        # 측정모델: {lv_name: {'zeta': array, 'sigma_sq': array}}
        meas_dict = {}
        for lv_name in meas_grads:
            meas_dict[lv_name] = {
                'zeta': meas_grads[lv_name]['zeta'][ind_idx],
                'sigma_sq': meas_grads[lv_name]['sigma_sq'][ind_idx]
            }

        # 구조모델: {param_name: scalar}
        struct_dict = {
            key: struct_grads[key][ind_idx].item() if hasattr(struct_grads[key][ind_idx], 'item')
            else struct_grads[key][ind_idx]
            for key in struct_grads
        }

        # 선택모델: {'intercept': scalar, 'beta': array, ...}
        choice_dict = {}
        for key in choice_grads:
            val = choice_grads[key][ind_idx]
            if key == 'beta':
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
    
    total_time = time.time() - start_time
    
    if iteration_logger and log_level in ['MODERATE', 'DETAILED']:
        iteration_logger.info(
            f"\n{'='*70}\n"
            f"완전 병렬 Gradient 계산 완료 ({total_time:.3f}초)\n"
            f"{'='*70}\n"
            f"  시간 분석:\n"
            f"    - LV 계산:      {lv_time:.3f}초 ({lv_time/total_time*100:.1f}%)\n"
            f"    - 측정모델:     {meas_time:.3f}초 ({meas_time/total_time*100:.1f}%)\n"
            f"    - 구조모델:     {struct_time:.3f}초 ({struct_time/total_time*100:.1f}%)\n"
            f"    - 선택모델:     {choice_time:.3f}초 ({choice_time/total_time*100:.1f}%)\n"
            f"  성능:\n"
            f"    - 개인당 시간:  {total_time / n_individuals * 1000:.2f}ms\n"
            f"    - 처리량:       {n_individuals / total_time:.1f} 개인/초\n"
            f"{'='*70}"
        )
    
    return all_individual_gradients

