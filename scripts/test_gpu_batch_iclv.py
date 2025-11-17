"""
동시추정 (Simultaneous Estimation) - GPU 배치 처리

이 파일 하나로 모든 동시추정을 수행합니다.
경로 설정과 선택모델 설정만 변경하면 다양한 모델을 테스트할 수 있습니다.

사용법:
1. PATHS 딕셔너리에서 원하는 경로를 True/False로 설정
2. MAIN_LVS, MODERATION_LVS, LV_ATTRIBUTE_INTERACTIONS에서 선택모델 설정
3. 실행하면 자동으로 경로 구성 및 파일명 생성
4. 결과 파일명: simultaneous_{경로명}_{선택모델LV}_results_{timestamp}.csv

주요 기능:
- 경로 설정: True/False로 간단하게 켜고 끄기
- 선택모델 설정: 순차추정과 동일한 방식
- 자동 파일명 생성: 경로와 선택모델에 따라 파일명 자동 생성
- 초기값 로드: 순차추정 결과를 초기값으로 사용 (선택사항)
- GPU 배치 처리: 고속 동시추정

Author: Sugar Substitute Research Team
Date: 2025-11-17
"""

import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 유틸리티 import
sys.path.insert(0, str(project_root / 'examples'))
from model_config_utils import (
    build_paths_from_config,
    build_choice_config_dict,
    generate_simultaneous_filename
)

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    ChoiceConfig,
    EstimationConfig
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    MultiLatentStructuralConfig,
    MultiLatentConfig,
    create_sugar_substitute_multi_lv_config
)
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_gpu_batch_estimator import SimultaneousGPUBatchEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice


# ============================================================================
# 🎯 사용자 설정 영역 - 여기만 수정하세요!
# ============================================================================

# 1. 경로 설정: True/False로 간단하게 켜고 끄기
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': False,  # 건강관심도 → 가격수준
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': False,  # 가격수준 → 구매의도
    'NK->PI': False,  # 영양지식 → 구매의도
}

# 2. 선택모델 설정 (순차추정 2단계와 동일한 설정)
# ✅ 순차추정 2단계(sequential_stage2_with_extended_model.py)와 동일하게 설정
# ✅ 초기값 파일(INITIAL_PARAMS_FILE)과 일치해야 함!

# 📌 잠재변수 주효과 (원하는 잠재변수만 추가)
# 예시: [] = Base Model (잠재변수 없음)
#      ['purchase_intention'] = Base + PI 주효과
#      ['purchase_intention', 'nutrition_knowledge'] = Base + PI + NK 주효과
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']  # Auto-generated

# 📌 조절효과 (잠재변수 2개 세트)
# 예시: [('perceived_price', 'nutrition_knowledge')] = PP와 NK의 조절효과
MODERATION_LVS = []  # Auto-generated

# 📌 LV-Attribute 상호작용 (잠재변수-속성 2개 세트)
# 예시: [('purchase_intention', 'price')] = PI × price 상호작용
#      [('purchase_intention', 'price'), ('nutrition_knowledge', 'health_label')]
LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]  # Auto-generated

# 3. 초기값 설정
# 순차추정 결과를 초기값으로 사용 (파일명 지정)
# None이면 자동 초기화 사용
# ⚠️ 중요: 초기값 파일을 사용하는 경우, 위의 선택모델 설정(MAIN_LVS, LV_ATTRIBUTE_INTERACTIONS)이
#          순차추정 2단계(sequential_stage2_with_extended_model.py)와 동일해야 합니다!
INITIAL_PARAMS_FILE = None  # ✅ 일단 자동 초기화로 테스트 (파일 형식 문제로 인해)

# 4. GPU 메모리 설정
CPU_MEMORY_THRESHOLD_MB = 2000  # CPU 메모리 임계값 (MB)
GPU_MEMORY_THRESHOLD_MB = 5000  # GPU 메모리 임계값 (MB)

# 5. 추정 설정
N_DRAWS = 100  # Halton draws 수
MAX_ITERATIONS = 1000  # 최대 반복 횟수

# ============================================================================
# 🤖 자동 처리 영역 - 수정 불필요
# ============================================================================


def main():
    """메인 실행 함수"""

    # 1. 경로 구성
    hierarchical_paths, path_name, model_description = build_paths_from_config(PATHS)

    print("=" * 70)
    print(f"동시추정 (GPU 배치): {model_description}")
    print("=" * 70)

    if hierarchical_paths:
        print(f"\n[1] 경로 구성 완료:")
        for i, path_dict in enumerate(hierarchical_paths, 1):
            target = path_dict['target']
            predictors = path_dict['predictors']
            pred_str = ' + '.join(predictors)
            print(f"    {i}. {pred_str} → {target}")
    else:
        print(f"\n[1] Base Model (경로 없음)")

    # 2. 선택모델 설정
    print(f"\n[2] 선택모델 설정:")
    if MAIN_LVS:
        print(f"    주효과 LV: {', '.join(MAIN_LVS)}")
    if MODERATION_LVS:
        print(f"    조절효과 LV: {', '.join(MODERATION_LVS)}")
    if LV_ATTRIBUTE_INTERACTIONS:
        print(f"    LV-속성 상호작용:")
        for lv, attr in LV_ATTRIBUTE_INTERACTIONS:
            print(f"      - {lv} × {attr}")
    if not MAIN_LVS and not MODERATION_LVS and not LV_ATTRIBUTE_INTERACTIONS:
        print(f"    Base Model (잠재변수 없음)")

    # 3. 데이터 로드
    print("\n[3] 데이터 로드:")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    n_individuals = data['respondent_id'].nunique()
    print(f"    데이터 shape: {data.shape}")
    print(f"    전체 개인 수: {n_individuals}")

    # 4. Config 생성
    print("\n[4] Config 생성:")

    # ✅ 유연한 리스트 기반: 선택모델 설정 딕셔너리 생성
    choice_config_dict = build_choice_config_dict(
        main_lvs=MAIN_LVS,
        lv_attribute_interactions=LV_ATTRIBUTE_INTERACTIONS
    )

    print(f"    선택모델 설정 딕셔너리: {choice_config_dict}")

    # Config 생성 (순차추정과 동일한 함수 사용)
    config = create_sugar_substitute_multi_lv_config(
        custom_paths=hierarchical_paths,
        choice_config_overrides=choice_config_dict,
        n_draws=N_DRAWS,
        max_iterations=MAX_ITERATIONS,
        optimizer='L-BFGS-B',  # ✅ BHHH → L-BFGS-B로 변경
        use_analytic_gradient=True,
        calculate_se=True,
        gradient_log_level='DETAILED',
        use_parameter_scaling=False  # ✅ 스케일링 비활성화
    )

    print(f"    Config 생성 완료")
    print(f"    - 잠재변수: 5개 (HC, PB, PP, NK, PI)")
    print(f"    - 측정 방법: 연속형 선형 (Continuous Linear)")
    print(f"    - Halton draws: {N_DRAWS}")
    print(f"    - 최대 반복: {MAX_ITERATIONS}")
    print(f"    - 최적화: L-BFGS-B (Analytic Gradient)")
    print(f"    - 파라미터 스케일링: 비활성화")
    print(f"    - GPU 배치 처리: 활성화")
    
    # 5. 모델 생성
    print("\n[5] 모델 생성:")
    with open("debug_choice_config.txt", "w", encoding="utf-8") as f:
        f.write(f"config.choice.main_lvs = {config.choice.main_lvs}\n")
        f.write(f"config.choice.lv_attribute_interactions = {config.choice.lv_attribute_interactions}\n")

    try:
        measurement_model = MultiLatentMeasurement(config.measurement_configs)
        structural_model = MultiLatentStructural(config.structural)
        choice_model = MultinomialLogitChoice(config.choice)
        print("    측정모델, 구조모델, 선택모델 생성 완료")
        print("    - 선택모델: Multinomial Logit (MNL)")

        with open("debug_choice_config.txt", "a", encoding="utf-8") as f:
            f.write(f"choice_model.main_lvs = {choice_model.main_lvs}\n")
            f.write(f"choice_model.lv_attribute_interactions = {choice_model.lv_attribute_interactions}\n")
    except Exception as e:
        print(f"    [ERROR] 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Estimator 생성
    print("\n[6] Estimator 생성:")
    try:
        estimator = SimultaneousGPUBatchEstimator(
            config,
            use_gpu=True,
            memory_monitor_cpu_threshold_mb=CPU_MEMORY_THRESHOLD_MB,
            memory_monitor_gpu_threshold_mb=GPU_MEMORY_THRESHOLD_MB
        )
        print(f"    동시추정 GPU 배치 Estimator 생성 완료")
        print(f"    - 메모리 모니터링: CPU {CPU_MEMORY_THRESHOLD_MB}MB, GPU {GPU_MEMORY_THRESHOLD_MB}MB")
    except Exception as e:
        print(f"    [ERROR] Estimator 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. 초기값 로드 (선택사항)
    print("\n[7] 초기값 로드:")
    initial_params = None

    if INITIAL_PARAMS_FILE:
        # 순차추정 결과 파일에서 초기값 로드
        initial_params_path = project_root / 'results' / 'sequential_stage_wise' / INITIAL_PARAMS_FILE

        if initial_params_path.exists():
            print(f"    초기값 파일: {INITIAL_PARAMS_FILE}")

            # .pkl 파일인 경우 (순차추정 1단계 결과)
            if INITIAL_PARAMS_FILE.endswith('.pkl'):
                import pickle
                with open(initial_params_path, 'rb') as f:
                    stage1_results = pickle.load(f)

                # 1단계 결과에서 파라미터 추출
                print(f"    순차추정 1단계 결과 로드 완료")

                # measurement_results와 structural_results에서 파라미터 추출
                if 'measurement_results' in stage1_results and 'structural_results' in stage1_results:
                    meas_params = stage1_results['measurement_results'].get('params', {})
                    struct_params = stage1_results['structural_results'].get('params', {})

                    # DataFrame이나 dict가 비어있지 않은지 확인
                    meas_valid = (isinstance(meas_params, dict) and len(meas_params) > 0) or \
                                 (hasattr(meas_params, 'empty') and not meas_params.empty)
                    struct_valid = (isinstance(struct_params, dict) and len(struct_params) > 0) or \
                                   (hasattr(struct_params, 'empty') and not struct_params.empty)

                    if meas_valid and struct_valid:
                        # 파라미터 딕셔너리 구성
                        param_dict = {
                            'measurement': meas_params,
                            'structural': struct_params,
                            'choice': None  # 선택모델은 자동 초기화
                        }

                        print(f"    측정모델 파라미터: {len(meas_params)} LVs")
                        if isinstance(meas_params, dict):
                            for lv_name, lv_params in meas_params.items():
                                if isinstance(lv_params, dict):
                                    print(f"      - {lv_name}: zeta={len(lv_params.get('zeta', []))}, sigma_sq={len(lv_params.get('sigma_sq', []))}")

                        print(f"    구조모델 파라미터:")
                        if isinstance(struct_params, dict):
                            for key, value in struct_params.items():
                                if isinstance(value, (int, float)):
                                    print(f"      - {key}: {value:.6f}")
                                else:
                                    print(f"      - {key}: {value}")
                        else:
                            print(f"      (DataFrame 형식: {len(struct_params)} rows)")

                        print(f"    선택모델 파라미터: 자동 초기화 사용")

                        # ParameterManager를 사용하여 배열로 변환
                        # 이 작업은 estimator 내부에서 수행되므로 여기서는 딕셔너리만 전달
                        initial_params = param_dict
                    else:
                        print(f"    [WARNING] 파라미터 정보가 불완전합니다.")
                        print(f"    자동 초기화를 사용합니다.")
                        initial_params = None
                else:
                    print(f"    [WARNING] .pkl 파일에 measurement_results 또는 structural_results가 없습니다.")
                    print(f"    자동 초기화를 사용합니다.")
                    initial_params = None

            # .csv 파일인 경우 (이전 동시추정 결과)
            elif INITIAL_PARAMS_FILE.endswith('.csv'):
                df_initial = pd.read_csv(initial_params_path)

                # Estimation statistics 행 제거 (빈 행 이후)
                first_empty_idx = df_initial[df_initial['Coefficient'].isna()].index
                if len(first_empty_idx) > 0:
                    df_initial = df_initial.iloc[:first_empty_idx[0]]

                # Estimate 값만 추출 (순서대로)
                initial_params = df_initial['Estimate'].values.astype(float)
                print(f"    초기값 개수: {len(initial_params)}")
                print(f"    초기값 범위: [{initial_params.min():.4f}, {initial_params.max():.4f}]")
            else:
                print(f"    [WARNING] 초기값 파일을 찾을 수 없습니다: {INITIAL_PARAMS_FILE}")
                print(f"    자동 초기화를 사용합니다.")
        else:
            print(f"    [WARNING] 초기값 파일을 찾을 수 없습니다: {INITIAL_PARAMS_FILE}")
            print(f"    자동 초기화를 사용합니다.")
    else:
        print(f"    초기값 파일 지정 안 됨 - 자동 초기화 사용")

    # 8. 추정 실행
    print("\n[8] 동시추정 실행:")
    print("    GPU 배치 처리로 모든 파라미터를 동시에 추정합니다.")
    print("    [주의] 5-10분 정도 소요될 수 있습니다...")

    # 로그 파일 경로 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = project_root / 'results' / f'simultaneous_estimation_log_{timestamp}.txt'
    print(f"    로그 파일: {log_file.name}")

    start_time = time.time()

    try:
        result = estimator.estimate(
            data=data,
            measurement_model=measurement_model,
            structural_model=structural_model,
            choice_model=choice_model,
            log_file=str(log_file),
            initial_params=initial_params
        )

        elapsed_time = time.time() - start_time

        # 9. 결과 출력
        print("\n" + "=" * 70)
        print("추정 결과")
        print("=" * 70)
        print(f"\n추정 시간: {elapsed_time/60:.2f}분 ({elapsed_time:.1f}초)")
        print(f"수렴 여부: {result['success']}")
        print(f"반복 횟수: {result.get('n_iterations', result.get('iterations', 'N/A'))}")
        print(f"최종 로그우도: {result['log_likelihood']:.4f}")

        # 메모리 사용 요약
        if hasattr(estimator, 'memory_monitor'):
            print("\n" + "=" * 70)
            print("메모리 사용 요약")
            print("=" * 70)
            mem_summary = estimator.memory_monitor.get_memory_summary()
            print(f"현재 CPU 메모리: {mem_summary['current_cpu_mb']:.1f}MB")
            if mem_summary['current_gpu_mb'] is not None:
                print(f"현재 GPU 메모리: {mem_summary['current_gpu_mb']:.1f}MB")
            if 'cpu_max_mb' in mem_summary:
                print(f"최대 CPU 메모리: {mem_summary['cpu_max_mb']:.1f}MB")
                print(f"평균 CPU 메모리: {mem_summary['cpu_avg_mb']:.1f}MB")
            if 'gpu_max_mb' in mem_summary:
                print(f"최대 GPU 메모리: {mem_summary['gpu_max_mb']:.1f}MB")
                print(f"평균 GPU 메모리: {mem_summary['gpu_avg_mb']:.1f}MB")

        # 10. 결과 저장
        print("\n[10] 결과 저장:")
        output_dir = project_root / 'results'
        output_dir.mkdir(exist_ok=True)

        # 파일명 생성 (순차추정과 동일한 규칙)
        csv_filename = generate_simultaneous_filename(path_name, config, timestamp)
        csv_file = output_dir / csv_filename

        # 파라미터 저장 (npy)
        npy_filename = csv_filename.replace('.csv', '.npy')
        params_file = output_dir / npy_filename
        np.save(params_file, result['raw_params'])

        # 파라미터 통계 추출
        print(f"    파라미터 통계 추출 중...")
        param_list = []

        if 'parameter_statistics' in result:
            stats = result['parameter_statistics']

            # 측정모델 파라미터 (다중 잠재변수)
            if 'measurement' in stats:
                for lv_name, lv_stats in stats['measurement'].items():
                    # zeta (요인적재량)
                    if 'zeta' in lv_stats:
                        zeta_stats = lv_stats['zeta']
                        for i in range(len(zeta_stats['estimate'])):
                            indicator_name = config.measurement_configs[lv_name].indicators[i]
                            param_list.append({
                                'Coefficient': f'ζ_{lv_name}_{indicator_name}',
                                'Estimate': zeta_stats['estimate'][i],
                                'Std. Err.': zeta_stats['std_error'][i],
                                'P. Value': zeta_stats['p_value'][i]
                            })

                    # sigma_sq (오차분산) - continuous_linear 방식
                    if 'sigma_sq' in lv_stats:
                        sigma_sq_stats = lv_stats['sigma_sq']
                        for i in range(len(sigma_sq_stats['estimate'])):
                            indicator_name = config.measurement_configs[lv_name].indicators[i]
                            param_list.append({
                                'Coefficient': f'σ²_{lv_name}_{indicator_name}',
                                'Estimate': sigma_sq_stats['estimate'][i],
                                'Std. Err.': sigma_sq_stats['std_error'][i],
                                'P. Value': sigma_sq_stats['p_value'][i]
                            })

            # 구조모델 파라미터 (계층적 구조)
            if 'structural' in stats:
                struct = stats['structural']

                # 계층적 파라미터 (gamma_pred_to_target)
                for key, value in struct.items():
                    if key.startswith('gamma_'):
                        param_list.append({
                            'Coefficient': f'γ_{key.replace("gamma_", "")}',
                            'Estimate': value['estimate'],
                            'Std. Err.': value['std_error'],
                            'P. Value': value['p_value']
                        })

            # 선택모델 파라미터
            if 'choice' in stats:
                choice = stats['choice']

                # intercept
                if 'intercept' in choice:
                    param_list.append({
                        'Coefficient': 'β_Intercept',
                        'Estimate': choice['intercept']['estimate'],
                        'Std. Err.': choice['intercept']['std_error'],
                        'P. Value': choice['intercept']['p_value']
                    })

                # beta
                if 'beta' in choice:
                    beta_stats = choice['beta']
                    for i, attr in enumerate(config.choice.choice_attributes):
                        param_list.append({
                            'Coefficient': f'β_{attr}',
                            'Estimate': beta_stats['estimate'][i],
                            'Std. Err.': beta_stats['std_error'][i],
                            'P. Value': beta_stats['p_value'][i]
                        })

                # lambda (주효과 LV)
                if 'lambda' in choice:
                    for lv_name, lv_stats in choice['lambda'].items():
                        param_list.append({
                            'Coefficient': f'λ_{lv_name}',
                            'Estimate': lv_stats['estimate'],
                            'Std. Err.': lv_stats['std_error'],
                            'P. Value': lv_stats['p_value']
                        })

                # lambda_interaction (LV-속성 상호작용)
                if 'lambda_interaction' in choice:
                    for interaction_name, interaction_stats in choice['lambda_interaction'].items():
                        param_list.append({
                            'Coefficient': f'λ_int_{interaction_name}',
                            'Estimate': interaction_stats['estimate'],
                            'Std. Err.': interaction_stats['std_error'],
                            'P. Value': interaction_stats['p_value']
                        })

            print(f"    ✓ {len(param_list)}개 파라미터 추출 완료")
        else:
            print(f"    [WARNING] parameter_statistics가 없습니다. 표준오차 없이 저장합니다.")

        # DataFrame 생성
        df_params = pd.DataFrame(param_list)

        # Estimation statistics 추가
        n_iter = result.get('n_iterations', result.get('iterations', 'N/A'))
        stats_list = [
            {'Coefficient': '', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
            {'Coefficient': 'Estimation statistics', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
            {'Coefficient': 'Iterations', 'Estimate': n_iter,
             'Std. Err.': 'LL (final)', 'P. Value': f"{result['log_likelihood']:.2f}"},
            {'Coefficient': 'AIC', 'Estimate': f"{result['aic']:.2f}",
             'Std. Err.': 'BIC', 'P. Value': f"{result['bic']:.2f}"}
        ]

        df_stats = pd.DataFrame(stats_list)
        df_combined = pd.concat([df_params, df_stats], ignore_index=True)

        # CSV 저장 (상세 파라미터)
        df_combined.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"    ✓ 결과 저장 완료: {csv_filename}")
        print(f"      - 파라미터 수: {len(param_list)}")
        print(f"      - 최종 LL: {result['log_likelihood']:.2f}")
        print(f"      - AIC: {result['aic']:.2f}, BIC: {result['bic']:.2f}")

    except Exception as e:
        print(f"\n[ERROR] 추정 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("동시추정 완료!")
    print("=" * 70)
    print(f"\n모델 구성:")
    print(f"  - 경로: {model_description}")
    if MAIN_LVS or MODERATION_LVS or LV_ATTRIBUTE_INTERACTIONS:
        print(f"  - 선택모델 LV: {', '.join(MAIN_LVS) if MAIN_LVS else 'None'}")
    print(f"\n결과 파일:")
    print(f"  - {csv_filename}")
    print(f"  - {npy_filename}")
    print(f"  - {log_file.name}")


if __name__ == '__main__':
    main()

