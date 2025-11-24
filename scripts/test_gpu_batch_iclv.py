"""
동시추정 (Simultaneous Estimation) - GPU 배치 처리

이 파일 하나로 모든 동시추정을 수행합니다.
순차추정 2단계 CSV 파일명만 지정하면 자동으로 설정이 파싱됩니다.

사용법:
1. INITIAL_PARAMS_CSV에 순차추정 2단계 결과 파일명 지정
   예: 'st2_HC-PB_PB-PI1_PI2_results.csv'
2. 실행하면 자동으로 경로 및 선택모델 설정 파싱
3. 결과 파일명: simultaneous_{경로명}_{선택모델LV}_results_{timestamp}.csv

주요 기능:
- 자동 설정: CSV 파일명에서 경로 및 선택모델 설정 자동 추출
- 초기값: PKL 파일에서 측정모델 파라미터 로드, 나머지는 0.1로 초기화
- GPU 배치 처리: 고속 동시추정

추정 대상:
- 측정모델: 추정 O (PKL 초기값 사용)
- 구조모델: 추정 O (초기값 0.1)
- 선택모델: 추정 O (초기값 0.1)

Author: Sugar Substitute Research Team
Date: 2025-11-18
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
    generate_iclv_filename,
    save_simultaneous_results,
    parse_csv_filename,
    parse_csv_content
)

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    create_sugar_substitute_multi_lv_config
)
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_gpu_batch_estimator import SimultaneousGPUBatchEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.iclv_models.initial_values_final import get_sigma_sq_initial_value


# ============================================================================
# 🎯 사용자 설정 영역 - 여기만 수정하세요!
# ============================================================================

# ============================================================================
# 설정 모드: 수동 설정
# ============================================================================

print("\n" + "=" * 70)
print("[MANUAL] 수동 설정 모드")
print("=" * 70)

# 1. 1단계 구조모델 경로 설정 (2path)
PATHS = {
    'HC->PB': True,   # health_concern → perceived_benefit
    'HC->PP': False,
    'HC->PI': False,
    'PB->PI': True,   # perceived_benefit → purchase_intention
    'PP->PI': False,
    'NK->PI': False,
}

# 2. 2단계 선택모델 설정
MAIN_LVS = ['nutrition_knowledge', 'purchase_intention', 'perceived_price']  # NK, PI, PP 주효과
MODERATION_LVS = []  # 조절효과 없음
LV_ATTRIBUTE_INTERACTIONS = []  # LV-Attribute 상호작용 없음

# 설정 출력
print(f"\n[모델 설정]")
print(f"  1단계 경로: {[k for k, v in PATHS.items() if v]}")
print(f"  주효과 LV: {MAIN_LVS}")
print(f"  조절효과: {MODERATION_LVS if MODERATION_LVS else '없음'}")
print(f"  LV-Attribute 상호작용: {LV_ATTRIBUTE_INTERACTIONS if LV_ATTRIBUTE_INTERACTIONS else '없음'}")
print("=" * 70 + "\n")

# ✅ CFA 결과 파일 사용 (측정모델만 추정된 결과)
# PKL 파일명도 자동 생성
from model_config_utils import build_paths_from_config
_, path_name, _, _ = build_paths_from_config(PATHS)
# INITIAL_PARAMS_PKL = f'stage1_{path_name}_results.pkl'  # SEM 결과 (구조모델 포함)
INITIAL_PARAMS_PKL = 'cfa_results.pkl'  # ✅ CFA 결과 (측정모델만)

# 4. GPU 메모리 설정
CPU_MEMORY_THRESHOLD_MB = 2000  # CPU 메모리 임계값 (MB)
GPU_MEMORY_THRESHOLD_MB = 7000  # GPU 메모리 임계값 (MB) - 5GB → 7GB로 증가

# 5. 추정 설정
N_DRAWS = 100  # Halton draws 수
MAX_ITERATIONS = 50  # 최대 반복 횟수 (1000 → 50으로 단축)

# ============================================================================
# 🤖 자동 처리 영역 - 수정 불필요
# ============================================================================

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""

    # 1. 경로 구성
    hierarchical_paths, path_name, model_description, n_paths = build_paths_from_config(PATHS)

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
        optimizer='trust-constr',  # ✅ Trust Region 사용
        use_analytic_gradient=True,
        calculate_se=True,
        se_method='robust',  # ✅ Sandwich Estimator (Robust SE) 사용
        gradient_log_level='MINIMAL',  # ✅ DETAILED → MINIMAL로 변경 (속도 최적화)
        use_parameter_scaling=False,  # ✅ 스케일링 비활성화 (z-score 표준화만 테스트)
        standardize_choice_attributes=True  # ✅ z-score 표준화 활성화
    )

    print(f"    Config 생성 완료")
    print(f"    - 잠재변수: 5개 (HC, PB, PP, NK, PI)")
    print(f"    - 측정 방법: 연속형 선형 (Continuous Linear)")
    print(f"    - Halton draws: {N_DRAWS}")
    print(f"    - 최대 반복: {MAX_ITERATIONS}")
    print(f"    - 최적화: Trust-Constr (Analytic Gradient)")
    print(f"    - 표준오차: Sandwich Estimator (Robust SE)")
    print(f"    - 파라미터 스케일링: ❌ 비활성화 (모든 스케일 = 1.0)")
    print(f"    - 데이터 표준화: ✅ 활성화 (z-score)")
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

    # ✅ 측정모델에 CFA 결과 로드 (동시추정 전용)
    # 이 단계는 초기값 설정 전에 수행되어야 함
    pkl_path = project_root / 'results' / 'final' / 'cfa_only' / INITIAL_PARAMS_PKL

    if pkl_path.exists():
        print(f"\n    [INFO] 측정모델에 CFA 결과 로드 중...")
        import pickle
        with open(pkl_path, 'rb') as f:
            cfa_results = pickle.load(f)

        if 'loadings' in cfa_results and 'measurement_errors' in cfa_results:
            loadings_df = cfa_results['loadings']
            errors_df = cfa_results['measurement_errors']
            intercepts_df = cfa_results.get('intercepts', None)  # ✅ 절편 로드

            # 각 잠재변수의 측정모델에 CFA 결과 설정
            for lv_name, model in measurement_model.models.items():
                lv_config = config.measurement_configs[lv_name]
                indicators = lv_config.indicators

                # zeta (요인적재량)
                zeta_values = []
                for indicator in indicators:
                    row = loadings_df[(loadings_df['lval'] == indicator) &
                                     (loadings_df['op'] == '~') &
                                     (loadings_df['rval'] == lv_name)]

                    if not row.empty:
                        zeta_values.append(float(row['Estimate'].iloc[0]))
                    else:
                        print(f"    [WARNING] {indicator} ~ {lv_name} 요인적재량을 찾을 수 없습니다. 기본값 1.0 사용")
                        zeta_values.append(1.0)

                # sigma_sq (오차분산)
                sigma_sq_values = []
                for indicator in indicators:
                    row = errors_df[(errors_df['lval'] == indicator) &
                                   (errors_df['op'] == '~~') &
                                   (errors_df['rval'] == indicator)]

                    if not row.empty:
                        sigma_sq_values.append(float(row['Estimate'].iloc[0]))
                    else:
                        print(f"    [WARNING] {indicator}의 오차분산을 찾을 수 없습니다. 기본값 0.5 사용")
                        sigma_sq_values.append(0.5)

                # ✅ alpha (절편)
                alpha_values = []
                if intercepts_df is not None:
                    for indicator in indicators:
                        row = intercepts_df[(intercepts_df['lval'] == indicator) &
                                           (intercepts_df['op'] == '~') &
                                           (intercepts_df['rval'] == '1')]

                        if not row.empty:
                            alpha_values.append(float(row['Estimate'].iloc[0]))
                        else:
                            print(f"    [WARNING] {indicator}의 절편을 찾을 수 없습니다. 기본값 0.0 사용")
                            alpha_values.append(0.0)
                else:
                    print(f"    [WARNING] CFA 결과에 절편이 없습니다. 모든 절편을 0.0으로 설정")
                    alpha_values = [0.0] * len(indicators)

                # 측정모델 config에 CFA 결과 설정
                model.config.zeta = np.array(zeta_values)
                model.config.sigma_sq = np.array(sigma_sq_values)
                model.config.alpha = np.array(alpha_values)  # ✅ 절편 추가

                print(f"    [INFO] {lv_name}: zeta={len(zeta_values)}개, sigma_sq={len(sigma_sq_values)}개, alpha={len(alpha_values)}개 로드 완료")

            print(f"    [SUCCESS] 측정모델에 CFA 결과 로드 완료 (절편 포함)")
        else:
            print(f"    [WARNING] CFA 결과 형식이 올바르지 않습니다.")

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

    # 7. 초기값 설정
    print("\n[7] 초기값 설정:")
    print("    [INFO] 측정모델 (zeta, sigma_sq): CFA 결과에서 로드 (이미 완료)")
    print("    [INFO] 구조모델 & 선택모델: 0.1로 초기화")

    initial_params = None

    if pkl_path.exists():
        print(f"\n    CFA 결과 로드: {INITIAL_PARAMS_PKL}")

        # 1. CFA 결과에서 측정모델 파라미터 로드
        import pickle
        with open(pkl_path, 'rb') as f:
            cfa_results = pickle.load(f)

        # CFA 결과는 직접 loadings와 measurement_errors를 포함
        if 'loadings' in cfa_results and 'measurement_errors' in cfa_results:
            print(f"    [INFO] CFA 결과에서 측정모델 파라미터 추출")

            loadings_df = cfa_results['loadings']
            errors_df = cfa_results['measurement_errors']
            intercepts_df = cfa_results.get('intercepts', None)  # ✅ 절편 로드

            # 측정모델 파라미터 딕셔너리 생성
            measurement_dict = {}
            for lv_name, lv_config in config.measurement_configs.items():
                indicators = lv_config.indicators

                # ✅ zeta (요인적재량) - CFA loadings에서 추출
                zeta_values = []
                for indicator in indicators:
                    row = loadings_df[(loadings_df['lval'] == indicator) &
                                     (loadings_df['op'] == '~') &
                                     (loadings_df['rval'] == lv_name)]

                    if not row.empty:
                        zeta_values.append(float(row['Estimate'].iloc[0]))
                    else:
                        print(f"    [WARNING] {indicator} ~ {lv_name} 요인적재량을 찾을 수 없습니다. 기본값 1.0 사용")
                        zeta_values.append(1.0)

                # ✅ sigma_sq (오차분산) - CFA measurement_errors에서 추출
                sigma_sq_values = []
                for indicator in indicators:
                    row = errors_df[(errors_df['lval'] == indicator) &
                                   (errors_df['op'] == '~~') &
                                   (errors_df['rval'] == indicator)]

                    if not row.empty:
                        sigma_sq_values.append(float(row['Estimate'].iloc[0]))
                    else:
                        print(f"    [WARNING] {indicator}의 오차분산을 찾을 수 없습니다. 기본값 0.5 사용")
                        sigma_sq_values.append(0.5)

                # ✅ alpha (절편) - CFA intercepts에서 추출
                alpha_values = []
                if intercepts_df is not None:
                    for indicator in indicators:
                        row = intercepts_df[(intercepts_df['lval'] == indicator) &
                                           (intercepts_df['op'] == '~') &
                                           (intercepts_df['rval'] == '1')]

                        if not row.empty:
                            alpha_values.append(float(row['Estimate'].iloc[0]))
                        else:
                            print(f"    [WARNING] {indicator}의 절편을 찾을 수 없습니다. 기본값 0.0 사용")
                            alpha_values.append(0.0)
                else:
                    print(f"    [WARNING] CFA 결과에 절편이 없습니다. 모든 절편을 0.0으로 설정")
                    alpha_values = [0.0] * len(indicators)

                measurement_dict[lv_name] = {
                    'zeta': np.array(zeta_values),
                    'sigma_sq': np.array(sigma_sq_values),
                    'alpha': np.array(alpha_values)  # ✅ 절편 추가
                }

                # ✅ 로드된 값 출력
                print(f"    [INFO] {lv_name} 측정모델 파라미터 로드:")
                print(f"      - zeta (요인적재량): {zeta_values}")
                print(f"      - sigma_sq (오차분산): {sigma_sq_values}")
                print(f"      - alpha (절편): {alpha_values}")

            # 2. 구조모델 파라미터: 0.1로 초기화
            print(f"    [INFO] 구조모델 파라미터: 0.1로 초기화")
            structural_dict = {}
            for path in config.structural.hierarchical_paths:
                target_lv = path['target']
                predictors = path['predictors']

                for pred_lv in predictors:
                    param_name = f'gamma_{pred_lv}_to_{target_lv}'
                    structural_dict[param_name] = 0.1
                    print(f"      - {param_name}: 0.1")

            # 3. 선택모델 파라미터: 0.1로 초기화
            print(f"    [INFO] 선택모델 파라미터: 0.1로 초기화")
            choice_dict = {}

            # Multinomial Logit의 대안 이름 (하드코딩)
            # opt-out은 기준 대안이므로 제외
            alternatives = ['sugar', 'sugar_free']  # opt-out 제외

            # ASC (Alternative-Specific Constants)
            for alt in alternatives:
                param_name = f'asc_{alt}'
                choice_dict[param_name] = 0.1
                print(f"      - {param_name}: 0.1")

            # beta (속성 계수) - 모든 대안에 공통 적용
            for attr in config.choice.choice_attributes:
                param_name = f'beta_{attr}'
                choice_dict[param_name] = 0.1
                print(f"      - {param_name}: 0.1")

            # theta (LV 주효과) - 각 대안별로
            if config.choice.main_lvs:
                for lv in config.choice.main_lvs:
                    for alt in alternatives:
                        param_name = f'theta_{alt}_{lv}'
                        choice_dict[param_name] = 0.1
                        print(f"      - {param_name}: 0.1")

            # gamma (LV-속성 상호작용) - 각 대안별로
            if config.choice.lv_attribute_interactions:
                for interaction in config.choice.lv_attribute_interactions:
                    lv = interaction['lv']
                    attr = interaction['attribute']
                    for alt in alternatives:
                        param_name = f'gamma_{alt}_{lv}_{attr}'
                        choice_dict[param_name] = 0.1
                        print(f"      - {param_name}: 0.1")

            # ✅ 최종 초기값 딕셔너리 구성 (측정모델 제외)
            # 측정모델 파라미터는 이미 measurement_model 객체에 로드되어 있음
            initial_params = {
                'structural': structural_dict,
                'choice': choice_dict
                # ❌ 'measurement' 키 제거: 동시추정에서는 불필요
                #    측정모델 파라미터는 measurement_model.models[lv_name].config에 이미 로드됨
            }

            # 결과 출력
            print(f"\n    [SUCCESS] 초기값 설정 완료:")
            print(f"      - 측정모델: {len(measurement_dict)} LVs (measurement_model 객체에 로드됨)")
            for lv_name in list(measurement_dict.keys())[:3]:
                lv_params = measurement_dict[lv_name]
                n_zeta = len(lv_params['zeta'])
                print(f"        * {lv_name}: zeta={n_zeta}개")

            print(f"      - 구조모델: {len(structural_dict)}개 파라미터 (0.1로 초기화)")
            print(f"      - 선택모델: {len(choice_dict)}개 파라미터 (0.1로 초기화)")

        else:
            print(f"    [ERROR] CFA 결과에 loadings 또는 measurement_errors가 없습니다.")
            raise ValueError("CFA 결과 형식이 올바르지 않습니다. loadings와 measurement_errors가 필요합니다.")
    else:
        print(f"    [WARNING] CFA 결과 파일을 찾을 수 없습니다: {INITIAL_PARAMS_PKL}")
        raise FileNotFoundError(f"CFA 결과 파일을 찾을 수 없습니다: {pkl_path}")

    # 8. 추정 실행
    print("\n[8] 동시추정 실행:")
    print("    [INFO] 추정 모드: 구조모델 + 선택모델만 추정")
    print("    [INFO] 측정모델: 고정값 사용 (추정 안 함)")
    print("    [INFO] GPU 배치 처리로 파라미터를 동시에 추정합니다.")
    print("    [주의] 5-10분 정도 소요될 수 있습니다...")

    # 로그 파일 경로 설정 (최종 결과 폴더)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / 'results' / 'final' / 'simultaneous' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'simultaneous_estimation_log_{timestamp}.txt'
    print(f"    로그 파일: {log_file.name}")

    start_time = time.time()

    try:
        print(f"\n    [INFO] 추정 시작...")
        print(f"    [INFO] 초기 파라미터 설정:")
        if initial_params:
            print(f"      - 측정모델: PKL에서 로드 (고정)")
            print(f"      - 구조모델: 0.1로 초기화 (추정 대상)")
            print(f"      - 선택모델: 0.1로 초기화 (추정 대상)")
        else:
            print(f"      - 자동 초기화 사용")

        result = estimator.estimate(
            data=data,
            measurement_model=measurement_model,
            structural_model=structural_model,
            choice_model=choice_model,
            log_file=str(log_file),
            initial_params=initial_params
            # ✅ 동시추정은 항상 측정모델 고정 (설정 불필요)
        )

        print(f"    [SUCCESS] 추정 완료!")

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
        # 최종 결과 폴더에 저장
        output_dir = project_root / 'results' / 'final' / 'simultaneous' / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 파일명 생성 (순차추정과 동일한 규칙, 타임스탬프 제거)
        csv_filename = generate_iclv_filename(
            choice_config=config,
            stage1_model_name=path_name,
            estimation_type='simultaneous'
        ) + '_results.csv'
        csv_file = output_dir / csv_filename

        # 파라미터 저장 (npy)
        npy_filename = csv_filename.replace('.csv', '.npy')
        params_file = output_dir / npy_filename
        np.save(params_file, result['raw_params'])
        print(f"    ✓ 원시 파라미터 저장: {npy_filename}")

        # CSV 저장 (범용 함수 사용)
        print(f"    CSV 결과 저장 중...")
        save_simultaneous_results(
            results=result,
            save_path=csv_file,
            cfa_results=cfa_results,
            config=config
        )
        print(f"    ✓ 결과 저장 완료: {csv_filename}")
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

