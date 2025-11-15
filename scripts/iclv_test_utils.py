"""
ICLV 테스트 스크립트 공통 유틸리티

동시추정과 순차추정 테스트 스크립트에서 공통으로 사용하는 함수들
"""

import sys
from pathlib import Path
import pandas as pd
import multiprocessing

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent


def load_integrated_data():
    """
    통합 데이터 로드

    Returns:
        pd.DataFrame: 통합 데이터
    """
    print("\n데이터 로드 중...")
    # integrated_data_cleaned.csv 사용 (중복 제거된 버전)
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data_cleaned.csv'
    data = pd.read_csv(data_path)

    print(f"   데이터 shape: {data.shape}")
    n_individuals = data['respondent_id'].nunique()
    print(f"   전체 개인 수: {n_individuals}")

    return data


def get_cpu_info():
    """
    CPU 정보 반환
    
    Returns:
        tuple: (n_cpus, n_cores_to_use)
    """
    n_cpus = multiprocessing.cpu_count()
    n_cores = max(1, n_cpus - 1)  # 1개는 시스템용으로 남김
    
    return n_cpus, n_cores


def print_config_summary(config, use_parallel=False, n_cores=1, n_cpus=1):
    """
    설정 요약 출력
    
    Args:
        config: ICLV 설정 객체
        use_parallel: 병렬처리 사용 여부
        n_cores: 사용할 코어 수
        n_cpus: 전체 CPU 코어 수
    """
    print("\n설정 요약:")
    print(f"   - 잠재변수: 5개 (HC, PB, PI, PP, NK)")
    print(f"   - 측정모델 지표: 38개")
    
    # 구조모델 정보
    if hasattr(config, 'structural'):
        if hasattr(config.structural, 'hierarchical_paths'):
            print(f"   - 구조모델: 계층적 (HC -> PB -> PI)")
        else:
            print(f"   - 구조모델: 병렬")
    
    # 선택모델 정보
    if hasattr(config, 'choice'):
        print(f"   - 선택 속성: {len(config.choice.choice_attributes)}개")
        if hasattr(config.choice, 'moderation_enabled') and config.choice.moderation_enabled:
            print(f"   - 조절효과: {len(config.choice.moderator_lvs)}개 (PP, NK)")
    
    # 추정 설정 (동시추정만)
    if hasattr(config, 'estimation'):
        print(f"   - Halton draws: {config.estimation.n_draws}")
        print(f"   - 최대 반복: {config.estimation.max_iterations}")
        parallel_status = "활성화" if use_parallel else "비활성화"
        print(f"   - 병렬처리: {parallel_status}")
        if use_parallel:
            print(f"   - 사용 코어: {n_cores}/{n_cpus}개 ({n_cores/n_cpus*100:.1f}%)")


def _create_param_dict(model, latent_variable, parameter, estimate, std_err=None, p_value=None):
    """
    파라미터 딕셔너리 생성 (공통 함수)

    Args:
        model: 모델 타입 ('Measurement', 'Structural', 'Choice')
        latent_variable: 잠재변수 이름
        parameter: 파라미터 이름
        estimate: 추정값
        std_err: 표준오차 (선택)
        p_value: p-value (선택)

    Returns:
        파라미터 딕셔너리
    """
    return {
        'Model': model,
        'Latent_Variable': latent_variable,
        'Parameter': parameter,
        'Estimate': estimate,
        'Std_Err': std_err,
        'p_value': p_value
    }


def _extract_simultaneous_params(results):
    """
    동시추정 결과에서 파라미터 추출

    Args:
        results: 동시추정 결과 딕셔너리

    Returns:
        파라미터 리스트
    """
    param_list = []

    # 측정모델 파라미터 (Ordered Probit)
    for lv_name in ['health_concern', 'perceived_benefit', 'perceived_price',
                    'nutrition_knowledge', 'purchase_intention']:
        if lv_name in results['parameters']['measurement']:
            params = results['parameters']['measurement'][lv_name]

            # zeta (요인적재량)
            for i, val in enumerate(params['zeta']):
                param_list.append(_create_param_dict(
                    'Measurement', lv_name, f'ζ_{i+1}', val
                ))

            # tau (임계값)
            tau = params['tau']
            for i in range(tau.shape[0]):
                for j in range(tau.shape[1]):
                    param_list.append(_create_param_dict(
                        'Measurement', lv_name, f'τ_{i+1},{j+1}', tau[i, j]
                    ))

    # 구조모델 파라미터
    gamma_lv = results['parameters']['structural']['gamma_lv']
    lv_names = ['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge']
    for i, lv_name in enumerate(lv_names):
        param_list.append(_create_param_dict(
            'Structural', 'purchase_intention', f'γ_{lv_name}', gamma_lv[i]
        ))

    gamma_x = results['parameters']['structural']['gamma_x']
    covariate_names = ['age_std', 'gender', 'income_std', 'education_level']
    for i, cov_name in enumerate(covariate_names):
        param_list.append(_create_param_dict(
            'Structural', 'purchase_intention', f'γ_{cov_name}', gamma_x[i]
        ))

    # 선택모델 파라미터
    param_list.append(_create_param_dict(
        'Choice', 'N/A', 'β_Intercept', results['parameters']['choice']['intercept']
    ))

    beta = results['parameters']['choice']['beta']
    choice_attrs = ['sugar_free', 'health_label', 'price']
    for i, attr in enumerate(choice_attrs):
        param_list.append(_create_param_dict(
            'Choice', 'N/A', f'β_{attr}', beta[i]
        ))

    param_list.append(_create_param_dict(
        'Choice', 'N/A', 'λ', results['parameters']['choice']['lambda']
    ))

    return param_list


def _extract_sequential_params(results):
    """
    순차추정 결과에서 파라미터 추출

    Args:
        results: 순차추정 결과 딕셔너리

    Returns:
        파라미터 리스트
    """
    param_list = []

    # Step 1: SEM 결과
    if 'stage_results' in results and 'measurement' in results['stage_results']:
        sem_results = results['stage_results']['measurement'].get('full_results', {})

        # 요인적재량
        if 'loadings' in sem_results:
            loadings = sem_results['loadings']
            for _, row in loadings.iterrows():
                param_list.append(_create_param_dict(
                    'Measurement',
                    row['rval'],
                    f'λ_{row["lval"]}',
                    row['Estimate'],
                    row.get('Std. Err', None),
                    row.get('p-value', None)
                ))

        # 경로계수
        if 'paths' in sem_results:
            paths = sem_results['paths']
            for _, row in paths.iterrows():
                param_list.append(_create_param_dict(
                    'Structural',
                    row['lval'],
                    f'γ_{row["rval"]}',
                    row['Estimate'],
                    row.get('Std. Err', None),
                    row.get('p-value', None)
                ))

    # Step 2: 선택모델 결과
    if 'stage_results' in results and 'choice' in results['stage_results']:
        choice_results = results['stage_results']['choice']

        if 'parameter_statistics' in choice_results:
            stats = choice_results['parameter_statistics']

            # Intercept
            if 'intercept' in stats:
                s = stats['intercept']
                param_list.append(_create_param_dict(
                    'Choice', None, 'intercept',
                    s['estimate'], s['se'], s['p']
                ))

            # Beta
            if 'beta' in stats:
                for attr, s in stats['beta'].items():
                    param_list.append(_create_param_dict(
                        'Choice', None, f'β_{attr}',
                        s['estimate'], s['se'], s['p']
                    ))

            # Lambda
            for key in ['lambda_main', 'lambda_mod_perceived_price', 'lambda_mod_nutrition_knowledge']:
                if key in stats:
                    s = stats[key]
                    param_list.append(_create_param_dict(
                        'Choice', None, key,
                        s['estimate'], s['se'], s['p']
                    ))

    return param_list


def save_results_to_csv(results, output_path, estimation_type='simultaneous'):
    """
    결과를 CSV 파일로 저장

    Args:
        results: 추정 결과 딕셔너리
        output_path: 출력 파일 경로
        estimation_type: 'simultaneous' 또는 'sequential'
    """
    # 추정 방법에 따라 파라미터 추출
    if estimation_type == 'simultaneous':
        param_list = _extract_simultaneous_params(results)
    elif estimation_type == 'sequential':
        param_list = _extract_sequential_params(results)
    else:
        raise ValueError(f"Unknown estimation_type: {estimation_type}")

    # DataFrame 생성 및 저장
    if param_list:
        df = pd.DataFrame(param_list)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n결과 저장 완료: {output_path}")
        print(f"   - 파라미터 개수: {len(param_list)}개")
    else:
        print("\n저장할 파라미터가 없습니다.")

