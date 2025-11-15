"""
ICLV 테스트 스크립트 공통 유틸리티

동시추정과 순차추정 테스트 스크립트에서 공통으로 사용하는 함수들
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List
import logging

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


def _single_bootstrap_worker(args):
    """
    단일 부트스트랩 샘플 처리 (병렬 처리용 워커 함수)

    Args:
        args: (sample_idx, data, individual_ids, measurement_model, structural_model,
               choice_model, random_seed)

    Returns:
        Dict: 파라미터 추정치 또는 None (실패 시)
    """
    (sample_idx, data, individual_ids, measurement_model,
     structural_model, choice_model, random_seed) = args

    try:
        # 시드 설정 (각 샘플마다 다른 시드)
        np.random.seed(random_seed + sample_idx)

        # 1. 개인별 리샘플링 (with replacement)
        bootstrap_ids = np.random.choice(individual_ids, size=len(individual_ids), replace=True)

        # 2. 리샘플링된 개인들의 데이터 추출
        bootstrap_data = pd.concat([
            data[data['respondent_id'] == id_val].copy()
            for id_val in bootstrap_ids
        ]).reset_index(drop=True)

        # 3. Step 1: SEM 추정
        from src.analysis.hybrid_choice_model.iclv_models.sem_estimator import SEMEstimator

        sem_estimator = SEMEstimator()
        sem_results = sem_estimator.fit(
            data=bootstrap_data,
            measurement_model=measurement_model,
            structural_model=structural_model
        )

        # 요인점수 추출
        factor_scores = sem_results['factor_scores']

        # 4. Step 2: 선택모델 추정
        choice_results = choice_model.fit(
            data=bootstrap_data,
            factor_scores=factor_scores
        )

        # 5. 결과 통합
        results = {
            'stage_results': {
                'measurement': {
                    'full_results': sem_results,
                    'parameters': sem_results.get('loadings', [])
                },
                'structural': {
                    'parameters': sem_results.get('paths', [])
                },
                'choice': choice_results
            }
        }

        # 6. 파라미터 추출
        params = _extract_sequential_params(results)

        return params

    except Exception as e:
        import traceback
        logger.warning(f"Bootstrap sample {sample_idx} failed: {e}")
        logger.debug(traceback.format_exc())
        return None


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


def _format_pvalue(p):
    """
    p-value를 적절한 형식으로 표기

    Args:
        p: p-value (float, str, 또는 None)

    Returns:
        str: 포맷된 p-value 문자열
    """
    if p is None or p == '-':
        return '-'

    # 문자열인 경우 float로 변환 시도
    if isinstance(p, str):
        try:
            p = float(p)
        except (ValueError, TypeError):
            return p

    # float인 경우 포맷팅
    if isinstance(p, (int, float)):
        if p < 0.001:
            return '<0.001'
        else:
            return f'{p:.4f}'

    return str(p)


def _create_param_dict(model, latent_variable, parameter, estimate, std_err=None, p_value=None, std_est=None):
    """
    파라미터 딕셔너리 생성 (공통 함수)

    Args:
        model: 모델 타입 ('Measurement', 'Structural', 'Choice')
        latent_variable: 잠재변수 이름
        parameter: 파라미터 이름
        estimate: 추정값 (비표준화)
        std_err: 표준오차 (선택)
        p_value: p-value (선택)
        std_est: 표준화 계수 (선택)

    Returns:
        파라미터 딕셔너리
    """
    # p-value 포맷팅
    formatted_p = _format_pvalue(p_value)

    return {
        'Model': model,
        'Latent_Variable': latent_variable,
        'Parameter': parameter,
        'Estimate': estimate,
        'Std_Est': std_est,
        'Std_Err': std_err,
        'p_value': formatted_p
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
                    row.get('p-value', None),
                    row.get('Est. Std', None)  # 표준화 계수 추가
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
                    row.get('p-value', None),
                    row.get('Est. Std', None)  # 표준화 계수 추가
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


def bootstrap_sequential_estimation(
    data: pd.DataFrame,
    measurement_model,
    structural_model,
    choice_model,
    n_bootstrap: int = 100,
    n_workers: int = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    순차추정 부트스트래핑 (CPU 병렬 처리)

    각 부트스트랩 샘플마다:
    1. 개인별 리샘플링 (with replacement)
    2. Step 1: SEM 재추정 -> 요인점수 추출
    3. Step 2: 선택모델 재추정
    4. 파라미터 저장

    Args:
        data: 전체 데이터 (5,868행)
        measurement_model: 측정모델 설정
        structural_model: 구조모델 설정
        choice_model: 선택모델 설정
        n_bootstrap: 부트스트랩 샘플 수 (기본 100)
        n_workers: 병렬 작업 수 (None이면 CPU 코어 수 - 1)
        confidence_level: 신뢰수준 (기본 0.95)
        random_seed: 랜덤 시드
        show_progress: 진행 상황 표시

    Returns:
        Dict containing:
        - bootstrap_estimates: 각 샘플의 파라미터 추정치 (List[Dict])
        - confidence_intervals: 파라미터별 신뢰구간 (DataFrame)
        - bootstrap_statistics: 평균, 표준편차, 편향 등 (DataFrame)
        - n_successful: 성공한 샘플 수
        - n_failed: 실패한 샘플 수
    """
    from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator

    # 병렬 작업 수 설정
    if n_workers is None:
        n_cpus = multiprocessing.cpu_count()
        n_workers = max(1, n_cpus - 1)

    print(f"\n부트스트래핑 설정:")
    print(f"   샘플 수: {n_bootstrap}")
    print(f"   병렬 작업 수: {n_workers}")
    print(f"   신뢰수준: {confidence_level}")

    # 개인 ID 추출
    individual_ids = data['respondent_id'].unique()
    n_individuals = len(individual_ids)

    print(f"   개인 수: {n_individuals}")

    # 워커 함수에 전달할 인자 준비
    worker_args = [
        (i, data, individual_ids, measurement_model, structural_model, choice_model, random_seed)
        for i in range(n_bootstrap)
    ]

    # 병렬 실행
    print(f"\n부트스트래핑 시작...")
    start_time = pd.Timestamp.now()

    bootstrap_results = []

    if n_workers > 1:
        # 병렬 처리
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_single_bootstrap_worker, args) for args in worker_args]

            if show_progress:
                # 진행 상황 표시
                for future in tqdm(as_completed(futures), total=n_bootstrap, desc="Bootstrap"):
                    result = future.result()
                    if result is not None:
                        bootstrap_results.append(result)
            else:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        bootstrap_results.append(result)
    else:
        # 순차 처리
        iterator = tqdm(worker_args, desc="Bootstrap") if show_progress else worker_args
        for args in iterator:
            result = _single_bootstrap_worker(args)
            if result is not None:
                bootstrap_results.append(result)

    end_time = pd.Timestamp.now()
    elapsed = (end_time - start_time).total_seconds()

    n_successful = len(bootstrap_results)
    n_failed = n_bootstrap - n_successful

    print(f"\n부트스트래핑 완료!")
    print(f"   성공: {n_successful}/{n_bootstrap}")
    print(f"   실패: {n_failed}/{n_bootstrap}")
    print(f"   소요 시간: {elapsed:.1f}초")

    if n_successful == 0:
        raise RuntimeError("모든 부트스트랩 샘플이 실패했습니다.")

    # 5. 신뢰구간 계산
    print(f"\n신뢰구간 계산 중...")
    confidence_intervals = _calculate_bootstrap_ci(bootstrap_results, confidence_level)

    # 6. 부트스트랩 통계량 계산
    bootstrap_stats = _calculate_bootstrap_stats(bootstrap_results)

    return {
        'bootstrap_estimates': bootstrap_results,
        'confidence_intervals': confidence_intervals,
        'bootstrap_statistics': bootstrap_stats,
        'n_successful': n_successful,
        'n_failed': n_failed,
        'elapsed_seconds': elapsed
    }


def _calculate_bootstrap_ci(bootstrap_results: List[List[Dict]],
                            confidence_level: float = 0.95) -> pd.DataFrame:
    """
    부트스트랩 결과로부터 신뢰구간 및 p-value 계산

    Args:
        bootstrap_results: 부트스트랩 파라미터 리스트
        confidence_level: 신뢰수준

    Returns:
        DataFrame: 파라미터별 신뢰구간 및 통계량
    """
    from scipy import stats

    # 파라미터별로 값 수집 (비표준화 + 표준화)
    param_values = {}
    param_values_std = {}  # 표준화 계수

    for sample_params in bootstrap_results:
        for param_dict in sample_params:
            key = (param_dict['Model'], param_dict['Latent_Variable'], param_dict['Parameter'])

            if key not in param_values:
                param_values[key] = []
                param_values_std[key] = []

            # Estimate 값 추출 (비표준화)
            estimate = param_dict['Estimate']
            if isinstance(estimate, str):
                try:
                    estimate = float(estimate)
                except (ValueError, TypeError):
                    continue

            param_values[key].append(estimate)

            # Std_Est 값 추출 (표준화)
            std_est = param_dict.get('Std_Est', None)
            if std_est is not None and not isinstance(std_est, str):
                param_values_std[key].append(std_est)
            elif isinstance(std_est, str):
                try:
                    param_values_std[key].append(float(std_est))
                except (ValueError, TypeError):
                    pass

    # 신뢰구간 계산
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_results = []

    for key, values in param_values.items():
        if len(values) == 0:
            continue

        values_array = np.array(values)

        # Percentile method (비표준화)
        lower_ci = np.percentile(values_array, lower_percentile)
        upper_ci = np.percentile(values_array, upper_percentile)
        mean_val = np.mean(values_array)
        se_val = np.std(values_array, ddof=1)  # 표준오차

        # 표준화 계수 통계량
        std_values = param_values_std.get(key, [])
        if len(std_values) > 0:
            std_values_array = np.array(std_values)
            mean_std = np.mean(std_values_array)
            se_std = np.std(std_values_array, ddof=1)
            lower_ci_std = np.percentile(std_values_array, lower_percentile)
            upper_ci_std = np.percentile(std_values_array, upper_percentile)
        else:
            mean_std = None
            se_std = None
            lower_ci_std = None
            upper_ci_std = None

        # p-value 계산 (3가지 방법)

        # 1. Bootstrap p-value (양측검정)
        # H0: parameter = 0
        n_bootstrap = len(values_array)
        if mean_val >= 0:
            # 양수인 경우: 0 이하인 값의 비율 × 2
            n_extreme = np.sum(values_array <= 0)
        else:
            # 음수인 경우: 0 이상인 값의 비율 × 2
            n_extreme = np.sum(values_array >= 0)

        p_value_bootstrap = 2 * (n_extreme + 1) / (n_bootstrap + 1)

        # 2. Normal approximation p-value
        if se_val > 0:
            z_stat = mean_val / se_val
            p_value_normal = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            p_value_normal = 1.0

        # 3. t-distribution p-value (더 보수적)
        if se_val > 0:
            t_stat = mean_val / se_val
            df = n_bootstrap - 1
            p_value_t = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            p_value_t = 1.0

        # 0을 포함하는지 확인 (유의성)
        significant = not (lower_ci <= 0 <= upper_ci)

        ci_results.append({
            'Model': key[0],
            'Latent_Variable': key[1],
            'Parameter': key[2],
            'Mean': mean_val,
            'SE': se_val,
            'CI_Lower': lower_ci,
            'CI_Upper': upper_ci,
            'Mean_Std': mean_std,
            'SE_Std': se_std,
            'CI_Lower_Std': lower_ci_std,
            'CI_Upper_Std': upper_ci_std,
            'p_value_bootstrap': p_value_bootstrap,
            'p_value_normal': p_value_normal,
            'p_value_t': p_value_t,
            'Significant': significant
        })

    return pd.DataFrame(ci_results)


def _calculate_bootstrap_stats(bootstrap_results: List[List[Dict]]) -> pd.DataFrame:
    """
    부트스트랩 통계량 계산

    Args:
        bootstrap_results: 부트스트랩 파라미터 리스트

    Returns:
        DataFrame: 파라미터별 통계량 (평균, 표준편차, 편향 등)
    """
    # 파라미터별로 값 수집
    param_values = {}

    for sample_params in bootstrap_results:
        for param_dict in sample_params:
            key = (param_dict['Model'], param_dict['Latent_Variable'], param_dict['Parameter'])

            if key not in param_values:
                param_values[key] = []

            # Estimate 값 추출
            estimate = param_dict['Estimate']
            if isinstance(estimate, str):
                try:
                    estimate = float(estimate)
                except (ValueError, TypeError):
                    continue

            param_values[key].append(estimate)

    # 통계량 계산
    stats_results = []

    for key, values in param_values.items():
        if len(values) == 0:
            continue

        values_array = np.array(values)

        stats_results.append({
            'Model': key[0],
            'Latent_Variable': key[1],
            'Parameter': key[2],
            'Bootstrap_Mean': np.mean(values_array),
            'Bootstrap_SE': np.std(values_array, ddof=1),
            'Bootstrap_Min': np.min(values_array),
            'Bootstrap_Max': np.max(values_array),
            'N_Samples': len(values_array)
        })

    return pd.DataFrame(stats_results)

