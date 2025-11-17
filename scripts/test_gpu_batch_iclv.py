"""
ICLV 모델 테스트 - 다중 잠재변수

다중 잠재변수 ICLV 모델을 추정합니다.
5개 잠재변수 (4개 외생 + 1개 내생)

추정 방법 선택:
- USE_SEQUENTIAL = False: 동시추정 (GPU 배치 처리)
- USE_SEQUENTIAL = True: 순차추정 (3단계 추정)
"""

import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# ============================================================================
# 추정 방법 선택
# ============================================================================
USE_SEQUENTIAL = False  # True: 순차추정, False: 동시추정 (GPU 배치)

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    ChoiceConfig,
    EstimationConfig
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    MultiLatentStructuralConfig,
    MultiLatentConfig
)

# 추정 방법에 따라 다른 Estimator import
if USE_SEQUENTIAL:
    from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
    print("✅ 순차추정 모드 (Sequential Estimation)")
else:
    from src.analysis.hybrid_choice_model.iclv_models.simultaneous_gpu_batch_estimator import SimultaneousGPUBatchEstimator
    print("✅ 동시추정 모드 (Simultaneous Estimation with GPU Batch)")

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice, MultinomialLogitChoice


# DataConfig를 직접 정의
from dataclasses import dataclass

@dataclass
class DataConfig:
    """데이터 설정"""
    individual_id: str = 'respondent_id'
    choice_id: str = 'choice_set'


def main():
    """메인 실행 함수"""

    print("="*70)
    print("GPU 배치 처리 ICLV 동시추정 - 다중 잠재변수 (5개)")
    print("="*70)

    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    print(f"   데이터 shape: {data.shape}")
    n_individuals = data['respondent_id'].nunique()
    print(f"   전체 개인 수: {n_individuals}")
    
    # 2. 설정
    print("\n2. ICLV 설정...")

    # 측정모델 설정 (5개 잠재변수)
    # 🔴 measurement_method='continuous_linear'을 디폴트로 설정 (SEM 방식)
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            n_categories=5,
            measurement_method='continuous_linear'  # 디폴트: 연속형 선형 측정모델
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            n_categories=5,
            measurement_method='continuous_linear'  # 디폴트: 연속형 선형 측정모델
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],
            n_categories=5,
            measurement_method='continuous_linear'  # 디폴트: 연속형 선형 측정모델
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=[f'q{i}' for i in range(30, 50)],  # q30-q49
            n_categories=5,
            measurement_method='continuous_linear'  # 디폴트: 연속형 선형 측정모델
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            n_categories=5,
            measurement_method='continuous_linear'  # 디폴트: 연속형 선형 측정모델
        )
    }

    # 구조모델 설정 (✅ 계층적 구조)
    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
        covariates=[],  # ✅ 사회인구학적 변수 제거 (새로운 디폴트)
        hierarchical_paths=[  # ✅ 계층적 경로 명시
            {'target': 'perceived_benefit', 'predictors': ['health_concern']},
            {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
        ],
        error_variance=1.0
    )

    # 선택모델 설정 (✅ 조절효과 - 디폴트 사용)
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        # ✅ 디폴트 값 사용 (명시하지 않아도 자동 적용):
        # moderation_enabled=True
        # moderator_lvs=['perceived_price', 'nutrition_knowledge']
        # main_lv='purchase_intention'
    )

    # 추정 설정
    estimation_config = EstimationConfig(
        optimizer='BHHH',  # ✅ BHHH 최적화 알고리즘 사용 (Newton-CG with OPG)
        use_analytic_gradient=True,  # ✅ Analytic gradient (CPU) 테스트
        n_draws=100,
        draw_type='halton',
        max_iterations=1000,
        calculate_se=True,  # 표준오차 계산 활성화 (BHHH 사용)
        use_parallel=False,  # GPU 배치는 자체적으로 병렬처리
        n_cores=None,
        early_stopping=False,  # ✅ 조기 종료 비활성화 (정상 종료 테스트)
        early_stopping_patience=999,
        early_stopping_tol=1e-6,
        gradient_log_level='DETAILED',  # ✅ 상세 그래디언트 로깅 활성화
        use_parameter_scaling=False  # ✅ 파라미터 스케일링 비활성화
    )

    # 통합 설정
    config = MultiLatentConfig(
        measurement_configs=measurement_configs,
        structural=structural_config,
        choice=choice_config,
        estimation=estimation_config,
        individual_id_column='respondent_id',
        choice_column='choice'
    )

    # config에 data 속성 추가
    config.data = DataConfig(
        individual_id='respondent_id',
        choice_id='choice_set'
    )

    print("   설정 완료")
    print(f"   - 잠재변수: {len(measurement_configs)}개 (3개 1차 LV + 2개 고차 LV)")
    total_indicators = sum(len(mc.indicators) for mc in measurement_configs.values())
    print(f"   - 지표 수: {total_indicators}")
    print(f"   - 측정 방법: 연속형 선형 (Continuous Linear)")
    print(f"   - 구조모델: 계층적 (HC → PB → PI)")
    print(f"   - 사회인구학적 변수: {len(structural_config.covariates)} (제거됨)")
    print(f"   - 선택모델: 조절효과 (PI × PP, PI × NK)")
    print(f"   - 선택 속성: {len(choice_config.choice_attributes)}")
    print(f"   - Halton draws: {estimation_config.n_draws}")
    print(f"   - 최대 반복: {estimation_config.max_iterations}")
    print(f"   - 전체 개인 수: {n_individuals}")
    print(f"   - GPU 배치 처리: 활성화")
    
    # 3. 모델 생성 (test_iclv_full_data.py와 동일)
    print("\n3. 모델 생성...")

    # ✅ 선택모델 타입 선택 (Binary Probit 또는 Multinomial Logit)
    USE_MNL = True  # True: MNL (이론적으로 올바름), False: Binary Probit (근사)

    try:
        measurement_model = MultiLatentMeasurement(measurement_configs)
        structural_model = MultiLatentStructural(structural_config)

        if USE_MNL:
            choice_model = MultinomialLogitChoice(choice_config)
            print("   - 선택모델: Multinomial Logit (MNL)")
        else:
            choice_model = BinaryProbitChoice(choice_config)
            print("   - 선택모델: Binary Probit")

        print("   - 측정모델, 구조모델, 선택모델 생성 완료")
    except Exception as e:
        print(f"   [ERROR] 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Estimator 생성 (추정 방법에 따라 다름)
    if USE_SEQUENTIAL:
        print("\n4. 순차 Estimator 생성...")
        try:
            estimator = SequentialEstimator(config)
            print("   - 순차 Estimator 생성 완료")
            print("   - 3단계 추정: 측정모델 → 구조모델 → 선택모델")
        except Exception as e:
            print(f"   [ERROR] Estimator 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("\n4. 동시추정 GPU 배치 Estimator 생성...")
        try:
            estimator = SimultaneousGPUBatchEstimator(
                config,
                use_gpu=True,
                memory_monitor_cpu_threshold_mb=2000,  # CPU 메모리 임계값 2GB
                memory_monitor_gpu_threshold_mb=5000   # GPU 메모리 임계값 5GB
            )
            print("   - 동시추정 GPU 배치 Estimator 생성 완료")
            print("   - 메모리 모니터링 활성화 (CPU: 2GB, GPU: 5GB 임계값)")
        except Exception as e:
            print(f"   [ERROR] Estimator 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return

    # 5. 초기값 로드 (gpu_batch_iclv_results_20251114_070950.csv)
    print("\n5. 초기값 로드...")
    initial_params_file = project_root / 'results' / 'gpu_batch_iclv_results_20251114_070950.csv'

    initial_params = None
    if initial_params_file.exists():
        print(f"   초기값 파일: {initial_params_file}")
        df_initial = pd.read_csv(initial_params_file)

        # Estimation statistics 행 제거 (빈 행 이후)
        first_empty_idx = df_initial[df_initial['Coefficient'].isna()].index
        if len(first_empty_idx) > 0:
            df_initial = df_initial.iloc[:first_empty_idx[0]]

        # Estimate 값만 추출 (순서대로)
        initial_params = df_initial['Estimate'].values.astype(float)
        print(f"   초기값 개수: {len(initial_params)}")
        print(f"   초기값 범위: [{initial_params.min():.4f}, {initial_params.max():.4f}]")
    else:
        print(f"   [경고] 초기값 파일을 찾을 수 없습니다: {initial_params_file}")
        print(f"   랜덤 초기값을 사용합니다.")

    # 6. ICLV 추정 실행
    if USE_SEQUENTIAL:
        print("\n6. ICLV 순차추정 실행...")
        print("   (3단계 추정 - 다중 잠재변수)")
        print("\n   [주의] 순차추정은 2-5분 정도 소요될 수 있습니다...")
    else:
        print("\n6. ICLV 동시추정 실행...")
        print("   (GPU 배치 처리 - 다중 잠재변수)")
        print("\n   [주의] GPU 배치 처리는 5-10분 정도 소요될 수 있습니다...")

    # 로그 파일 경로 설정
    log_file = project_root / 'results' / 'gpu_batch_iclv_estimation_log.txt'
    print(f"   로그 파일: {log_file}")

    start_time = time.time()

    try:
        result = estimator.estimate(
            data=data,
            measurement_model=measurement_model,
            structural_model=structural_model,
            choice_model=choice_model,
            log_file=str(log_file),
            initial_params=initial_params  # 초기값 전달
        )

        elapsed_time = time.time() - start_time

        # 7. 결과 출력
        print("\n" + "="*70)
        if USE_SEQUENTIAL:
            print("추정 결과 (순차추정 - 다중 잠재변수)")
        else:
            print("추정 결과 (GPU 배치 - 다중 잠재변수)")
        print("="*70)
        print(f"\n추정 시간: {elapsed_time/60:.2f}분 ({elapsed_time:.1f}초)")
        print(f"수렴 여부: {result['success']}")
        print(f"반복 횟수: {result.get('n_iterations', result.get('iterations', 'N/A'))}")
        print(f"최종 로그우도: {result['log_likelihood']:.4f}")

        # 순차추정인 경우 단계별 결과 출력
        if USE_SEQUENTIAL and 'stage_results' in result:
            print("\n" + "-"*70)
            print("단계별 결과")
            print("-"*70)
            stage_results = result['stage_results']
            print(f"1단계 (측정모델): LL = {stage_results['measurement']['log_likelihood']:.4f}")
            print(f"2단계 (구조모델): LL = {stage_results['structural']['log_likelihood']:.4f}")
            print(f"3단계 (선택모델): LL = {stage_results['choice']['log_likelihood']:.4f}")

        # 메모리 사용 요약 (동시추정만)
        if not USE_SEQUENTIAL and hasattr(estimator, 'memory_monitor'):
            print("\n" + "="*70)
            print("메모리 사용 요약")
            print("="*70)
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

        # 8. 결과 저장
        output_dir = project_root / 'results'
        output_dir.mkdir(exist_ok=True)

        # 타임스탬프 생성 (CSV와 동일한 타임스탬프 사용)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파일명 prefix 설정
        file_prefix = 'sequential_iclv' if USE_SEQUENTIAL else 'gpu_batch_iclv'

        # 파라미터 저장 (npy)
        params_file = output_dir / f'{file_prefix}_params_{timestamp}.npy'
        np.save(params_file, result['raw_params'])

        # ✅ 로그 파일에서 최종 파라미터 값 파싱
        param_list = []
        initial_ll_from_log = None
        log_file = output_dir / 'gpu_batch_iclv_estimation_log.txt'

        if log_file.exists():
            print("\n로그 파일에서 최종 파라미터 값 파싱 중...")
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # "Parameter Scaling Comparison" 섹션 찾기
                import re
                # 두 번째 ---- 라인 이후부터 세 번째 ---- 라인까지
                pattern = r'Parameter Scaling Comparison:.*?-{80}.*?-{80}\n(.*?)-{80}'
                match = re.search(pattern, content, re.DOTALL)

                if match:
                    param_section = match.group(1)
                    # 각 파라미터 라인 파싱 (영문 파라미터 이름)
                    # 형식: 2025-11-12 17:46:30 - zeta_health_concern_q7                1.821545     1.821545     1.000000
                    param_pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-\s+([a-zA-Z_][^\s]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)'

                    for line in param_section.strip().split('\n'):
                        param_match = re.match(param_pattern, line.strip())
                        if param_match:
                            param_name = param_match.group(1)
                            external_value = float(param_match.group(2))

                            param_list.append({
                                'Coefficient': param_name,
                                'Estimate': external_value,
                                'Std. Err.': 'N/A',
                                'P. Value': 'N/A'
                            })

                    print(f"   ✓ {len(param_list)}개 파라미터 파싱 완료")

                    # 초기 LL 파싱 (Major Iteration #1)
                    ll_pattern = r'\[Major Iteration #1 완료\].*?최종 LL:\s*([-+]?[\d.]+)'
                    ll_match = re.search(ll_pattern, content, re.DOTALL)
                    if ll_match:
                        initial_ll_from_log = f"{float(ll_match.group(1)):.2f}"
                        print(f"   ✓ 초기 LL 파싱: {initial_ll_from_log}")
                else:
                    print("   ⚠️  Parameter Scaling Comparison 섹션을 찾을 수 없습니다.")
            except Exception as e:
                print(f"   ⚠️  로그 파일 파싱 실패: {e}")
                import traceback
                traceback.print_exc()

        # 로그 파일 파싱 실패 시 기존 방식 사용
        if not param_list:
            print("\n로그 파일 파싱 실패 - result['parameters']에서 파라미터 추출 중...")

            # parameter_statistics가 있는 경우 (표준오차 계산됨)
            if 'parameter_statistics' in result:
                print("\n표준오차 및 통계량 포함하여 저장 중...")
                stats = result['parameter_statistics']

                # 측정모델 파라미터 (다중 잠재변수)
                if 'measurement' in stats:
                    for lv_name, lv_stats in stats['measurement'].items():
                        # zeta (요인적재량)
                        if 'zeta' in lv_stats:
                            zeta_stats = lv_stats['zeta']
                            for i in range(len(zeta_stats['estimate'])):
                                # 지표 이름 가져오기
                                indicator_name = measurement_configs[lv_name].indicators[i]
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
                                # 지표 이름 가져오기
                                indicator_name = measurement_configs[lv_name].indicators[i]
                                param_list.append({
                                    'Coefficient': f'σ²_{lv_name}_{indicator_name}',
                                    'Estimate': sigma_sq_stats['estimate'][i],
                                    'Std. Err.': sigma_sq_stats['std_error'][i],
                                    'P. Value': sigma_sq_stats['p_value'][i]
                                })

                        # tau (임계값) - ordered_probit 방식
                        if 'tau' in lv_stats:
                            tau_stats = lv_stats['tau']
                            for i in range(tau_stats['estimate'].shape[0]):
                                indicator_name = measurement_configs[lv_name].indicators[i]
                                for j in range(tau_stats['estimate'].shape[1]):
                                    param_list.append({
                                        'Coefficient': f'τ_{lv_name}_{indicator_name}_{j+1}',
                                        'Estimate': tau_stats['estimate'][i, j],
                                        'Std. Err.': tau_stats['std_error'][i, j],
                                        'P. Value': tau_stats['p_value'][i, j]
                                    })

                # 구조모델 파라미터 (✅ 계층적 구조)
                if 'structural' in stats:
                    struct = stats['structural']

                    # ✅ 계층적 파라미터 (gamma_pred_to_target)
                    for key, value in struct.items():
                        if key.startswith('gamma_'):
                            param_list.append({
                                'Coefficient': f'γ_{key.replace("gamma_", "")}',
                                'Estimate': value['estimate'],
                                'Std. Err.': value['std_error'],
                                'P. Value': value['p_value']
                            })

                    # 하위 호환: gamma_lv (병렬 구조)
                    if 'gamma_lv' in struct:
                        gamma_lv_stats = struct['gamma_lv']
                        lv_names = ['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge']
                        for i, lv in enumerate(lv_names):
                            param_list.append({
                                'Coefficient': f'γ_lv_{lv}',
                                'Estimate': gamma_lv_stats['estimate'][i],
                                'Std. Err.': gamma_lv_stats['std_error'][i],
                                'P. Value': gamma_lv_stats['p_value'][i]
                            })

                    # 하위 호환: gamma_x (사회인구학적 변수)
                    if 'gamma_x' in struct:
                        gamma_x_stats = struct['gamma_x']
                        sociodem_vars = ['age_std', 'gender', 'income_std']
                        for i, var in enumerate(sociodem_vars):
                            param_list.append({
                                'Coefficient': f'γ_x_{var}',
                                'Estimate': gamma_x_stats['estimate'][i],
                                'Std. Err.': gamma_x_stats['std_error'][i],
                                'P. Value': gamma_x_stats['p_value'][i]
                            })

                # 선택모델 파라미터 (✅ 조절효과 지원)
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
                        choice_attrs = ['sugar_free', 'health_label', 'price']
                        for i, attr in enumerate(choice_attrs):
                            param_list.append({
                                'Coefficient': f'β_{attr}',
                                'Estimate': beta_stats['estimate'][i],
                                'Std. Err.': beta_stats['std_error'][i],
                                'P. Value': beta_stats['p_value'][i]
                            })

                    # ✅ lambda_main (조절효과 모델)
                    if 'lambda_main' in choice:
                        param_list.append({
                            'Coefficient': 'λ_main',
                            'Estimate': choice['lambda_main']['estimate'],
                            'Std. Err.': choice['lambda_main']['std_error'],
                            'P. Value': choice['lambda_main']['p_value']
                        })

                    # ✅ lambda_mod (조절효과 계수)
                    for key in choice.keys():
                        if key.startswith('lambda_mod_'):
                            mod_name = key.replace('lambda_mod_', '')
                            param_list.append({
                                'Coefficient': f'λ_mod_{mod_name}',
                                'Estimate': choice[key]['estimate'],
                                'Std. Err.': choice[key]['std_error'],
                                'P. Value': choice[key]['p_value']
                            })

                    # 하위 호환: lambda (기본 모델)
                    if 'lambda' in choice:
                        param_list.append({
                            'Coefficient': 'λ',
                            'Estimate': choice['lambda']['estimate'],
                            'Std. Err.': choice['lambda']['std_error'],
                            'P. Value': choice['lambda']['p_value']
                        })

            else:
                # 기존 방식 (표준오차 없음)
                print("\n표준오차 없이 저장 중...")

                # 측정모델 파라미터 (다중 잠재변수)
                for lv_name, lv_params in result['parameters']['measurement'].items():
                    # zeta (요인적재량)
                    zeta = lv_params['zeta']
                    for i, val in enumerate(zeta):
                        # 지표 이름 가져오기
                        indicator_name = measurement_configs[lv_name].indicators[i]
                        param_list.append({
                            'Coefficient': f'ζ_{lv_name}_{indicator_name}',
                            'Estimate': val,
                            'Std. Err.': 'N/A',
                            'P. Value': 'N/A'
                        })

                    # sigma_sq (오차분산) - continuous_linear 방식
                    if 'sigma_sq' in lv_params:
                        sigma_sq = lv_params['sigma_sq']
                        for i, val in enumerate(sigma_sq):
                            # 지표 이름 가져오기
                            indicator_name = measurement_configs[lv_name].indicators[i]
                            param_list.append({
                                'Coefficient': f'σ²_{lv_name}_{indicator_name}',
                                'Estimate': val,
                                'Std. Err.': 'N/A',
                                'P. Value': 'N/A'
                            })

                    # tau (임계값) - ordered_probit 방식
                    if 'tau' in lv_params:
                        tau = lv_params['tau']
                        for i in range(tau.shape[0]):
                            indicator_name = measurement_configs[lv_name].indicators[i]
                            for j in range(tau.shape[1]):
                                param_list.append({
                                    'Coefficient': f'τ_{lv_name}_{indicator_name}_{j+1}',
                                    'Estimate': tau[i, j],
                                    'Std. Err.': 'N/A',
                                    'P. Value': 'N/A'
                                })

                # 구조모델 파라미터 (✅ 계층적 구조)
                struct_params = result['parameters']['structural']

                # ✅ 계층적 파라미터
                for key, value in struct_params.items():
                    if key.startswith('gamma_'):
                        param_list.append({
                            'Coefficient': f'γ_{key.replace("gamma_", "")}',
                            'Estimate': value,
                            'Std. Err.': 'N/A',
                            'P. Value': 'N/A'
                        })

                # 하위 호환: gamma_lv (병렬 구조)
                if 'gamma_lv' in struct_params:
                    gamma_lv = struct_params['gamma_lv']
                    lv_names = ['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge']
                    for i, lv in enumerate(lv_names):
                        param_list.append({
                            'Coefficient': f'γ_lv_{lv}',
                            'Estimate': gamma_lv[i],
                            'Std. Err.': 'N/A',
                            'P. Value': 'N/A'
                        })

                # 하위 호환: gamma_x (사회인구학적 변수)
                if 'gamma_x' in struct_params:
                    gamma_x = struct_params['gamma_x']
                    sociodem_vars = ['age_std', 'gender', 'income_std']
                    for i, var in enumerate(sociodem_vars):
                        param_list.append({
                            'Coefficient': f'γ_x_{var}',
                            'Estimate': gamma_x[i],
                            'Std. Err.': 'N/A',
                            'P. Value': 'N/A'
                        })

                # 선택모델 파라미터 (✅ 조절효과 지원)
                choice_params = result['parameters']['choice']

                param_list.append({
                    'Coefficient': 'β_Intercept',
                    'Estimate': choice_params['intercept'],
                    'Std. Err.': 'N/A',
                    'P. Value': 'N/A'
                })

                beta = choice_params['beta']
                choice_attrs = ['sugar_free', 'health_label', 'price']
                for i, attr in enumerate(choice_attrs):
                    param_list.append({
                        'Coefficient': f'β_{attr}',
                        'Estimate': beta[i],
                        'Std. Err.': 'N/A',
                        'P. Value': 'N/A'
                    })

                # ✅ lambda_main (조절효과 모델)
                if 'lambda_main' in choice_params:
                    param_list.append({
                        'Coefficient': 'λ_main',
                        'Estimate': choice_params['lambda_main'],
                        'Std. Err.': 'N/A',
                        'P. Value': 'N/A'
                    })

                # ✅ lambda_mod (조절효과 계수)
                for key in choice_params.keys():
                    if key.startswith('lambda_mod_'):
                        mod_name = key.replace('lambda_mod_', '')
                        param_list.append({
                            'Coefficient': f'λ_mod_{mod_name}',
                            'Estimate': choice_params[key],
                            'Std. Err.': 'N/A',
                            'P. Value': 'N/A'
                        })

                # 하위 호환: lambda (기본 모델)
                if 'lambda' in choice_params:
                    param_list.append({
                        'Coefficient': 'λ',
                        'Estimate': choice_params['lambda'],
                        'Std. Err.': 'N/A',
                        'P. Value': 'N/A'
                    })

        # DataFrame 생성
        df_params = pd.DataFrame(param_list)

        # 초기 LL 설정 (로그 파일에서 파싱된 값 사용, 없으면 기본값)
        initial_ll = initial_ll_from_log if initial_ll_from_log is not None else 'N/A'

        # Estimation statistics 추가
        n_iter = result.get('n_iterations', result.get('iterations', 'N/A'))
        stats_list = [
            {'Coefficient': '', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
            {'Coefficient': 'Estimation statistics', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
            {'Coefficient': 'Iterations', 'Estimate': n_iter,
             'Std. Err.': 'LL (start)', 'P. Value': initial_ll},
            {'Coefficient': 'AIC', 'Estimate': f"{result['aic']:.2f}",
             'Std. Err.': 'LL (final, whole model)', 'P. Value': f"{result['log_likelihood']:.2f}"},
            {'Coefficient': 'BIC', 'Estimate': f"{result['bic']:.2f}",
             'Std. Err.': 'LL (Choice)', 'P. Value': 'N/A'}
        ]

        df_stats = pd.DataFrame(stats_list)
        df_combined = pd.concat([df_params, df_stats], ignore_index=True)

        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV 저장 (상세 파라미터)
        csv_file = output_dir / f'{file_prefix}_results_{timestamp}.csv'
        df_combined.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n   ✓ 결과 저장 완료: {csv_file}")
        print(f"     - 파라미터 수: {len(param_list)}")
        print(f"     - 최종 LL: {result['log_likelihood']:.2f}")

        # ✅ Hessian 역행렬 저장 (별도 CSV 파일)
        if hasattr(estimator, 'hessian_inv_matrix') and estimator.hessian_inv_matrix is not None:
            print(f"\n   ✓ Hessian 역행렬 저장 중...")
            hess_inv = estimator.hessian_inv_matrix

            # 파라미터 이름 가져오기
            if hasattr(estimator, 'param_names') and estimator.param_names:
                param_names = estimator.param_names
            else:
                param_names = [f"param_{i}" for i in range(hess_inv.shape[0])]

            # DataFrame 생성 (행/열 모두 파라미터 이름)
            df_hessian = pd.DataFrame(
                hess_inv,
                index=param_names,
                columns=param_names
            )

            # CSV 저장
            hessian_file = output_dir / f'{file_prefix}_hessian_inv_{timestamp}.csv'
            df_hessian.to_csv(hessian_file, encoding='utf-8-sig')
            print(f"     - Hessian 역행렬 저장 완료: {hessian_file}")
            print(f"     - Shape: {hess_inv.shape}")
        elif 'hessian_inv' in result and result['hessian_inv'] is not None:
            print(f"\n   ✓ Hessian 역행렬 저장 중...")
            hess_inv = result['hessian_inv']

            # 파라미터 이름 가져오기
            if hasattr(estimator, 'param_names') and estimator.param_names:
                param_names = estimator.param_names
            else:
                param_names = [f"param_{i}" for i in range(hess_inv.shape[0])]

            # DataFrame 생성
            df_hessian = pd.DataFrame(
                hess_inv,
                index=param_names,
                columns=param_names
            )

            # CSV 저장
            hessian_file = output_dir / f'{file_prefix}_hessian_inv_{timestamp}.csv'
            df_hessian.to_csv(hessian_file, encoding='utf-8-sig')
            print(f"     - Hessian 역행렬 저장 완료: {hessian_file}")
            print(f"     - Shape: {hess_inv.shape}")
        else:
            print(f"\n   ⚠️  Hessian 역행렬 없음 (저장 건너뜀)")

        # 요약정보 저장 (CSV)
        optimizer_name = 'Sequential_3Step' if USE_SEQUENTIAL else 'BFGS_GPU_Batch'
        gpu_enabled = 'False' if USE_SEQUENTIAL else 'True'
        halton_draws = 'N/A' if USE_SEQUENTIAL else str(estimation_config.n_draws)

        summary_data = {
            'Metric': ['Estimation_Time_Minutes', 'N_Individuals', 'N_Observations',
                       'Halton_Draws', 'Optimizer', 'Log_Likelihood', 'N_Parameters',
                       'GPU_Enabled', 'AIC', 'BIC'],
            'Value': [f"{elapsed_time/60:.2f}", str(n_individuals), str(data.shape[0]),
                      halton_draws, optimizer_name,
                      f"{result['log_likelihood']:.4f}", str(result['n_parameters']),
                      gpu_enabled, f"{result['aic']:.2f}", f"{result['bic']:.2f}"]
        }

        if n_iter != 'N/A':
            summary_data['Metric'].append('N_Iterations')
            summary_data['Value'].append(str(n_iter))

        df_summary = pd.DataFrame(summary_data)
        summary_file = output_dir / f'{file_prefix}_summary_{timestamp}.csv'
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')

        print(f"\n결과 저장:")
        print(f"  - 파라미터 (통계량 포함): {csv_file}")
        print(f"  - 파라미터 (npy): {params_file}")
        print(f"  - 요약정보: {summary_file}")

    except Exception as e:
        print(f"   [ERROR] 추정 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    if USE_SEQUENTIAL:
        print("순차 추정 완료!")
    else:
        print("GPU 배치 추정 완료!")
    print("="*70)


if __name__ == '__main__':
    main()

