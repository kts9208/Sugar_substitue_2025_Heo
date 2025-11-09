"""
GPU 배치 처리 ICLV 모델 테스트 - 다중 잠재변수

완전한 GPU 배치 처리로 다중 잠재변수 ICLV 모델을 추정합니다.
5개 잠재변수 (4개 외생 + 1개 내생) 동시추정
"""

import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path

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
from src.analysis.hybrid_choice_model.iclv_models.gpu_batch_estimator import GPUBatchEstimator


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
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            n_categories=5  # test_iclv_full_data.py와 동일하게 5로 설정
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            n_categories=5
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],
            n_categories=5
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=[f'q{i}' for i in range(30, 50)],  # q30-q49
            n_categories=2
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            n_categories=5
        )
    }

    # 구조모델 설정
    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
        covariates=['age_std', 'gender', 'income_std'],  # test_iclv_full_data.py와 동일
        error_variance=1.0
    )

    # 선택모델 설정
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price']  # test_iclv_full_data.py와 동일
    )

    # 추정 설정
    estimation_config = EstimationConfig(
        optimizer='BFGS',
        use_analytic_gradient=False,  # GPU 배치는 수치 그래디언트 사용
        n_draws=100,
        draw_type='halton',
        max_iterations=1000,
        calculate_se=False,  # GPU 배치는 표준오차 계산 안 함 (속도 우선)
        use_parallel=False,  # GPU 배치는 자체적으로 병렬처리
        n_cores=None
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
    print(f"   - 잠재변수: {len(measurement_configs)}개 (4개 외생 + 1개 내생)")
    total_indicators = sum(len(mc.indicators) for mc in measurement_configs.values())
    print(f"   - 지표 수: {total_indicators}")
    print(f"   - 사회인구학적 변수: {len(structural_config.covariates)}")
    print(f"   - 선택 속성: {len(choice_config.choice_attributes)}")
    print(f"   - Halton draws: {estimation_config.n_draws}")
    print(f"   - 최대 반복: {estimation_config.max_iterations}")
    print(f"   - 전체 개인 수: {n_individuals}")
    print(f"   - GPU 배치 처리: 활성화")
    
    # 3. GPU 배치 Estimator 생성
    print("\n3. GPU 배치 Estimator 생성...")

    try:
        estimator = GPUBatchEstimator(config, data, use_gpu=True)
        print("   - GPU 배치 Estimator 생성 완료")
    except Exception as e:
        print(f"   [ERROR] Estimator 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. ICLV 동시추정 실행
    print("\n4. ICLV 동시추정 실행...")
    print("   (GPU 배치 처리 - 다중 잠재변수)")
    print("   (로깅: 매 5회 반복마다 LL 출력)")
    print("\n   [주의] GPU 배치 처리는 5-10분 정도 소요될 수 있습니다...")

    # 로그 파일 경로 설정
    log_file = project_root / 'results' / 'gpu_batch_iclv_estimation_log.txt'
    print(f"   로그 파일: {log_file}")

    start_time = time.time()

    try:
        result = estimator.estimate(
            initial_params=None,
            method='BFGS',
            maxiter=estimation_config.max_iterations
        )

        elapsed_time = time.time() - start_time

        # 5. 결과 출력
        print("\n" + "="*70)
        print("추정 결과 (GPU 배치 - 다중 잠재변수)")
        print("="*70)
        print(f"\n추정 시간: {elapsed_time/60:.2f}분 ({elapsed_time:.1f}초)")
        print(f"수렴 여부: {result['success']}")
        print(f"반복 횟수: {result['iterations']}")
        print(f"최종 로그우도: {result['log_likelihood']:.4f}")

        # 6. 결과 저장
        output_dir = project_root / 'results'
        output_dir.mkdir(exist_ok=True)

        # 파라미터 저장 (npy)
        params_file = output_dir / 'gpu_batch_iclv_params.npy'
        np.save(params_file, result['params'])

        # 요약정보 저장 (CSV)
        summary_data = {
            'Metric': ['Estimation_Time_Minutes', 'N_Individuals', 'N_Observations',
                       'Halton_Draws', 'Optimizer', 'Log_Likelihood', 'N_Parameters',
                       'Batch_Size', 'GPU_Enabled'],
            'Value': [f"{elapsed_time/60:.2f}", str(n_individuals), str(data.shape[0]),
                      str(estimation_config.n_draws), 'BFGS_GPU_Batch',
                      f"{result['log_likelihood']:.4f}", str(len(result['params'])),
                      str(estimator.batch_size), 'True']
        }

        if 'iterations' in result:
            summary_data['Metric'].append('N_Iterations')
            summary_data['Value'].append(str(result['iterations']))

        df_summary = pd.DataFrame(summary_data)
        summary_file = output_dir / 'gpu_batch_iclv_summary.csv'
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')

        print(f"\n결과 저장:")
        print(f"  - 파라미터: {params_file}")
        print(f"  - 요약정보: {summary_file}")

    except Exception as e:
        print(f"   [ERROR] 추정 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("GPU 배치 추정 완료!")
    print("="*70)


if __name__ == '__main__':
    main()

