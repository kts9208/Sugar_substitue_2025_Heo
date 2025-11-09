"""
ICLV 동시추정 - 전체 데이터셋 (Apollo 방식 Analytic Gradient)

전체 인원 (300명) + 더 많은 Halton draws로 최종 추정
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    ICLVConfig,
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig,
    EstimationConfig
)


# DataConfig를 직접 정의
from dataclasses import dataclass

@dataclass
class DataConfig:
    """데이터 설정"""
    individual_id: str = 'respondent_id'
    choice_id: str = 'choice_set'


def main():
    print("="*70)
    print("ICLV 동시추정 - 전체 데이터셋 (Apollo Analytic Gradient)")
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
    
    # 측정모델 설정 (1개 잠재변수, 6개 지표)
    measurement_config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        n_categories=5
    )
    
    # 구조모델 설정
    structural_config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std'],
        include_in_choice=False
    )
    
    # 선택모델 설정
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price']  # 설탕함량 추가
    )
    
    # 추정 설정 - 전체 데이터용
    estimation_config = EstimationConfig(
        optimizer='BFGS',                # Apollo 방식 BFGS
        use_analytic_gradient=True,      # Analytic gradient 사용
        n_draws=100,                     # 🔴 100 draws (Apollo 권장)
        draw_type='halton',
        max_iterations=1000,             # 🔴 전체 데이터는 더 많은 반복 필요
        calculate_se=True,               # 🔴 표준오차 계산 활성화
        use_parallel=True,               # 🔴 병렬처리 활성화
        n_cores=None                     # 🔴 자동으로 CPU 코어 수 감지
    )
    
    # 통합 설정
    config = ICLVConfig(
        measurement=measurement_config,
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
    print(f"   - 잠재변수: {measurement_config.latent_variable}")
    print(f"   - 지표 수: {len(measurement_config.indicators)}")
    print(f"   - 사회인구학적 변수: {len(structural_config.sociodemographics)}")
    print(f"   - 선택 속성: {len(choice_config.choice_attributes)}")
    print(f"   - Halton draws: {estimation_config.n_draws}")
    print(f"   - 최대 반복: {estimation_config.max_iterations}")
    print(f"   - 전체 개인 수: {n_individuals}")
    print(f"   - 병렬처리: {'활성화' if estimation_config.use_parallel else '비활성화'}")

    # 3. 모델 생성
    print("\n3. 모델 생성...")

    # 측정모델
    measurement_model = OrderedProbitMeasurement(measurement_config)
    print("   - 측정모델 생성 완료")

    # 구조모델
    structural_model = LatentVariableRegression(structural_config)
    print("   - 구조모델 생성 완료")

    # 선택모델
    choice_model = BinaryProbitChoice(choice_config)
    print("   - 선택모델 생성 완료")

    # 4. ICLV 동시추정 실행
    print("\n4. ICLV 동시추정 실행...")
    print("   (전체 데이터 + BFGS + Analytic Gradient (Apollo 방식) + 병렬처리)")
    print("   (로깅: 매 10회 반복마다 LL 출력, 개선 시 즉시 출력)")
    print("\n   ⚠️  전체 데이터 추정은 시간이 오래 걸릴 수 있습니다...")

    # 로그 파일 경로 설정
    log_file = project_root / 'results' / 'iclv_full_data_estimation_log.txt'
    print(f"   로그 파일: {log_file}")

    start_time = time.time()

    estimator = SimultaneousEstimator(config)
    results = estimator.estimate(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        choice_model=choice_model,
        log_file=str(log_file)
    )
    
    elapsed_time = time.time() - start_time

    # 5. 결과 출력
    print("\n" + "="*70)
    print("추정 결과 (전체 데이터)")
    print("="*70)
    print(f"\n추정 시간: {elapsed_time/60:.2f}분 ({elapsed_time:.1f}초)")

    # convergence 키가 있는지 확인
    if 'convergence' in results:
        print(f"수렴 여부: {results['convergence']['success']}")
        print(f"반복 횟수: {results['convergence']['n_iterations']}")
    else:
        print(f"수렴 여부: 미확인 (precision loss)")
        print(f"반복 횟수: 미확인")

    print(f"최종 로그우도: {results['log_likelihood']:.4f}")

    print("\n파라미터 추정값:")
    print("\n[측정모델]")
    print(f"  요인적재량 (zeta): {results['parameters']['measurement']['zeta']}")
    print(f"  임계값 (tau) shape: {results['parameters']['measurement']['tau'].shape}")

    print("\n[구조모델]")
    print(f"  gamma: {results['parameters']['structural']['gamma']}")

    print("\n[선택모델]")
    print(f"  intercept: {results['parameters']['choice']['intercept']:.4f}")
    print(f"  beta: {results['parameters']['choice']['beta']}")
    print(f"  lambda: {results['parameters']['choice']['lambda']:.4f}")

    # 6. 결과 저장 (CSV 형식)
    output_dir = project_root / 'results'
    output_dir.mkdir(exist_ok=True)

    # 파라미터를 DataFrame으로 변환
    param_list = []

    # parameter_statistics가 있는 경우 (표준오차 계산됨)
    if 'parameter_statistics' in results:
        print("\n표준오차 및 통계량 포함하여 저장 중...")
        stats = results['parameter_statistics']

        # 측정모델 파라미터
        if 'measurement' in stats:
            meas = stats['measurement']

            # zeta
            if 'zeta' in meas:
                zeta_stats = meas['zeta']
                for i in range(len(zeta_stats['estimate'])):
                    param_list.append({
                        'Coefficient': f'ζ_{i+1}',
                        'Estimate': zeta_stats['estimate'][i],
                        'Std. Err.': zeta_stats['std_error'][i],
                        'P. Value': zeta_stats['p_value'][i]
                    })

            # tau
            if 'tau' in meas:
                tau_stats = meas['tau']
                for i in range(tau_stats['estimate'].shape[0]):
                    for j in range(tau_stats['estimate'].shape[1]):
                        param_list.append({
                            'Coefficient': f'τ_{i+1},{j+1}',
                            'Estimate': tau_stats['estimate'][i, j],
                            'Std. Err.': tau_stats['std_error'][i, j],
                            'P. Value': tau_stats['p_value'][i, j]
                        })

        # 구조모델 파라미터
        if 'structural' in stats:
            struct = stats['structural']
            if 'gamma' in struct:
                gamma_stats = struct['gamma']
                sociodem_vars = ['age_std', 'gender', 'income_std']
                for i, var in enumerate(sociodem_vars):
                    param_list.append({
                        'Coefficient': f'γ_{var}',
                        'Estimate': gamma_stats['estimate'][i],
                        'Std. Err.': gamma_stats['std_error'][i],
                        'P. Value': gamma_stats['p_value'][i]
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
                choice_attrs = ['sugar_free', 'health_label', 'price']  # 설탕함량 추가
                for i, attr in enumerate(choice_attrs):
                    param_list.append({
                        'Coefficient': f'β_{attr}',
                        'Estimate': beta_stats['estimate'][i],
                        'Std. Err.': beta_stats['std_error'][i],
                        'P. Value': beta_stats['p_value'][i]
                    })

            # lambda
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

        # 측정모델 파라미터
        zeta = results['parameters']['measurement']['zeta']
        for i, val in enumerate(zeta):
            param_list.append({
                'Coefficient': f'ζ_{i+1}',
                'Estimate': val,
                'Std. Err.': 'N/A',
                'P. Value': 'N/A'
            })

        tau = results['parameters']['measurement']['tau']
        for i in range(tau.shape[0]):
            for j in range(tau.shape[1]):
                param_list.append({
                    'Coefficient': f'τ_{i+1},{j+1}',
                    'Estimate': tau[i, j],
                    'Std. Err.': 'N/A',
                    'P. Value': 'N/A'
                })

        # 구조모델 파라미터
        gamma = results['parameters']['structural']['gamma']
        sociodem_vars = ['age_std', 'gender', 'income_std']
        for i, var in enumerate(sociodem_vars):
            param_list.append({
                'Coefficient': f'γ_{var}',
                'Estimate': gamma[i],
                'Std. Err.': 'N/A',
                'P. Value': 'N/A'
            })

        # 선택모델 파라미터
        param_list.append({
            'Coefficient': 'β_Intercept',
            'Estimate': results['parameters']['choice']['intercept'],
            'Std. Err.': 'N/A',
            'P. Value': 'N/A'
        })

        beta = results['parameters']['choice']['beta']
        choice_attrs = ['sugar_free', 'health_label', 'price']  # 설탕함량 추가
        for i, attr in enumerate(choice_attrs):
            param_list.append({
                'Coefficient': f'β_{attr}',
                'Estimate': beta[i],
                'Std. Err.': 'N/A',
                'P. Value': 'N/A'
            })

        param_list.append({
            'Coefficient': 'λ',
            'Estimate': results['parameters']['choice']['lambda'],
            'Std. Err.': 'N/A',
            'P. Value': 'N/A'
        })

    # DataFrame 생성
    df_params = pd.DataFrame(param_list)

    # 로그 파일에서 초기 LL 읽기
    initial_ll = 'N/A'
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Iter    1:' in line and 'LL =' in line:
                    # "Iter    1: LL =   -7581.2098 (Best:   -7581.2098) [NEW BEST]"
                    ll_str = line.split('LL =')[1].split('(')[0].strip()
                    initial_ll = f"{float(ll_str):.2f}"
                    break
    except Exception as e:
        print(f"   ⚠️  초기 LL 읽기 실패: {e}")

    # Estimation statistics 추가
    stats_list = [
        {'Coefficient': '', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
        {'Coefficient': 'Estimation statistics', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
        {'Coefficient': 'Iterations', 'Estimate': results.get('n_iterations', 'N/A'),
         'Std. Err.': 'LL (start)', 'P. Value': initial_ll},
        {'Coefficient': 'AIC', 'Estimate': f"{results['aic']:.2f}",
         'Std. Err.': 'LL (final, whole model)', 'P. Value': f"{results['log_likelihood']:.2f}"},
        {'Coefficient': 'BIC', 'Estimate': f"{results['bic']:.2f}",
         'Std. Err.': 'LL (Choice)', 'P. Value': 'N/A'}
    ]

    df_stats = pd.DataFrame(stats_list)
    df_combined = pd.concat([df_params, df_stats], ignore_index=True)

    # CSV 저장
    csv_file = output_dir / 'iclv_full_data_results.csv'
    df_combined.to_csv(csv_file, index=False, encoding='utf-8-sig')

    # 요약 정보도 별도 CSV로 저장
    summary_data = {
        'Metric': ['Estimation_Time_Minutes', 'N_Individuals', 'N_Observations',
                   'Halton_Draws', 'Optimizer', 'Log_Likelihood', 'AIC', 'BIC'],
        'Value': [f"{elapsed_time/60:.2f}", str(n_individuals), str(data.shape[0]),
                  str(estimation_config.n_draws), f"{estimation_config.optimizer}_Analytic",
                  f"{results['log_likelihood']:.4f}", f"{results['aic']:.2f}", f"{results['bic']:.2f}"]
    }

    if 'n_iterations' in results:
        summary_data['Metric'].append('N_Iterations')
        summary_data['Value'].append(str(results['n_iterations']))

    df_summary = pd.DataFrame(summary_data)
    summary_file = output_dir / 'iclv_full_data_summary.csv'
    df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')

    print(f"\n결과 저장:")
    print(f"  - 파라미터 (통계량 포함): {csv_file}")
    print(f"  - 요약정보: {summary_file}")
    
    print("\n" + "="*70)
    print("전체 데이터 추정 완료!")
    print("="*70)


if __name__ == '__main__':
    main()

