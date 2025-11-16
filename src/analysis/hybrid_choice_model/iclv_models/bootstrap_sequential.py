"""
순차추정 부트스트래핑 모듈

3가지 부트스트래핑 지원:
1. Stage 1 Only: SEM만 부트스트래핑 (측정모델 + 구조모델)
2. Stage 2 Only: 선택모델만 부트스트래핑 (1단계 요인점수 고정)
3. Stage 1+2: 순차추정 전체 부트스트래핑

Author: ICLV Team
Date: 2025-01-16
"""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)


class SequentialBootstrap:
    """
    순차추정 부트스트래핑 클래스
    
    3가지 부트스트래핑 모드:
    - 'stage1': 1단계만 (SEM)
    - 'stage2': 2단계만 (선택모델, 요인점수 고정)
    - 'both': 1+2단계 전체
    """
    
    def __init__(
        self,
        n_bootstrap: int = 100,
        n_workers: Optional[int] = None,
        confidence_level: float = 0.95,
        random_seed: int = 42,
        show_progress: bool = True
    ):
        """
        초기화
        
        Args:
            n_bootstrap: 부트스트랩 샘플 수
            n_workers: 병렬 작업 수 (None이면 CPU 코어 수 - 1)
            confidence_level: 신뢰수준
            random_seed: 랜덤 시드
            show_progress: 진행 상황 표시
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.show_progress = show_progress
        
        # 병렬 작업 수 설정
        if n_workers is None:
            n_cpus = multiprocessing.cpu_count()
            self.n_workers = max(1, n_cpus - 1)
        else:
            self.n_workers = n_workers
        
        logger.info(f"부트스트래핑 설정: {n_bootstrap}개 샘플, {self.n_workers}개 워커")
    
    def run_stage1_bootstrap(
        self,
        data: pd.DataFrame,
        measurement_model,
        structural_model
    ) -> Dict[str, Any]:
        """
        1단계만 부트스트래핑 (SEM)
        
        각 부트스트랩 샘플마다:
        1. 개인별 리샘플링
        2. SEM 재추정 (측정모델 + 구조모델)
        3. 파라미터 저장
        
        Args:
            data: 전체 데이터
            measurement_model: 측정모델 설정
            structural_model: 구조모델 설정
        
        Returns:
            부트스트랩 결과 딕셔너리
        """
        logger.info("=" * 70)
        logger.info("1단계 부트스트래핑 시작 (SEM Only)")
        logger.info("=" * 70)
        
        # 개인 ID 추출
        individual_ids = data['respondent_id'].unique()
        n_individuals = len(individual_ids)
        
        print(f"\n[1단계 부트스트래핑]")
        print(f"  샘플 수: {self.n_bootstrap}")
        print(f"  병렬 작업 수: {self.n_workers}")
        print(f"  신뢰수준: {self.confidence_level}")
        print(f"  개인 수: {n_individuals}")
        
        # 워커 함수 인자 준비
        worker_args = [
            (i, data, individual_ids, measurement_model, structural_model, None, self.random_seed, 'stage1')
            for i in range(self.n_bootstrap)
        ]
        
        # 부트스트래핑 실행
        bootstrap_results = self._run_parallel_bootstrap(worker_args)
        
        # 결과 처리
        return self._process_results(bootstrap_results, 'stage1')
    
    def run_stage2_bootstrap(
        self,
        choice_data: pd.DataFrame,
        factor_scores: Dict[str, np.ndarray],
        choice_model
    ) -> Dict[str, Any]:
        """
        2단계만 부트스트래핑 (선택모델, 요인점수 고정)
        
        각 부트스트랩 샘플마다:
        1. 개인별 리샘플링 (선택 데이터 + 요인점수)
        2. 선택모델 재추정
        3. 파라미터 저장
        
        Args:
            choice_data: 선택 데이터
            factor_scores: 1단계에서 추출한 요인점수 (고정)
            choice_model: 선택모델 설정
        
        Returns:
            부트스트랩 결과 딕셔너리
        """
        logger.info("=" * 70)
        logger.info("2단계 부트스트래핑 시작 (Choice Model Only)")
        logger.info("=" * 70)

        # 개인 ID 추출
        individual_ids = choice_data['respondent_id'].unique()
        n_individuals = len(individual_ids)

        print(f"\n[2단계 부트스트래핑]")
        print(f"  샘플 수: {self.n_bootstrap}")
        print(f"  병렬 작업 수: {self.n_workers}")
        print(f"  신뢰수준: {self.confidence_level}")
        print(f"  개인 수: {n_individuals}")

        # 워커 함수 인자 준비
        worker_args = [
            (i, choice_data, individual_ids, None, None, choice_model, self.random_seed, 'stage2', factor_scores)
            for i in range(self.n_bootstrap)
        ]

        # 부트스트래핑 실행
        bootstrap_results = self._run_parallel_bootstrap(worker_args)

        # 결과 처리
        return self._process_results(bootstrap_results, 'stage2')

    def run_both_stages_bootstrap(
        self,
        data: pd.DataFrame,
        measurement_model,
        structural_model,
        choice_model
    ) -> Dict[str, Any]:
        """
        1+2단계 전체 부트스트래핑 (순차추정)

        각 부트스트랩 샘플마다:
        1. 개인별 리샘플링
        2. Step 1: SEM 재추정 -> 요인점수 추출
        3. Step 2: 선택모델 재추정
        4. 파라미터 저장

        Args:
            data: 전체 데이터
            measurement_model: 측정모델 설정
            structural_model: 구조모델 설정
            choice_model: 선택모델 설정

        Returns:
            부트스트랩 결과 딕셔너리
        """
        logger.info("=" * 70)
        logger.info("1+2단계 부트스트래핑 시작 (Full Sequential)")
        logger.info("=" * 70)

        # 개인 ID 추출
        individual_ids = data['respondent_id'].unique()
        n_individuals = len(individual_ids)

        print(f"\n[1+2단계 부트스트래핑]")
        print(f"  샘플 수: {self.n_bootstrap}")
        print(f"  병렬 작업 수: {self.n_workers}")
        print(f"  신뢰수준: {self.confidence_level}")
        print(f"  개인 수: {n_individuals}")

        # 워커 함수 인자 준비
        worker_args = [
            (i, data, individual_ids, measurement_model, structural_model, choice_model, self.random_seed, 'both')
            for i in range(self.n_bootstrap)
        ]

        # 부트스트래핑 실행
        bootstrap_results = self._run_parallel_bootstrap(worker_args)

        # 결과 처리
        return self._process_results(bootstrap_results, 'both')

    def _run_parallel_bootstrap(self, worker_args: List[Tuple]) -> List[Dict]:
        """
        병렬 부트스트래핑 실행

        Args:
            worker_args: 워커 함수 인자 리스트

        Returns:
            부트스트랩 결과 리스트
        """
        import time
        start_time = time.time()

        bootstrap_results = []

        if self.n_workers > 1:
            # 병렬 처리
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(_bootstrap_worker, args) for args in worker_args]

                if self.show_progress:
                    for future in tqdm(as_completed(futures), total=self.n_bootstrap, desc="Bootstrap"):
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
            iterator = tqdm(worker_args, desc="Bootstrap") if self.show_progress else worker_args
            for args in iterator:
                result = _bootstrap_worker(args)
                if result is not None:
                    bootstrap_results.append(result)

        elapsed = time.time() - start_time

        n_successful = len(bootstrap_results)
        n_failed = self.n_bootstrap - n_successful

        print(f"\n부트스트래핑 완료!")
        print(f"  성공: {n_successful}/{self.n_bootstrap}")
        print(f"  실패: {n_failed}/{self.n_bootstrap}")
        print(f"  소요 시간: {elapsed:.1f}초")

        if n_successful == 0:
            raise RuntimeError("모든 부트스트랩 샘플이 실패했습니다.")

        return bootstrap_results

    def _process_results(self, bootstrap_results: List[Dict], mode: str) -> Dict[str, Any]:
        """
        부트스트랩 결과 처리

        Args:
            bootstrap_results: 부트스트랩 결과 리스트
            mode: 'stage1', 'stage2', 'both'

        Returns:
            처리된 결과 딕셔너리
        """
        print(f"\n신뢰구간 계산 중...")

        # 신뢰구간 계산
        confidence_intervals = self._calculate_confidence_intervals(bootstrap_results, mode)

        # 부트스트랩 통계량 계산
        bootstrap_stats = self._calculate_statistics(bootstrap_results, mode)

        return {
            'bootstrap_estimates': bootstrap_results,
            'confidence_intervals': confidence_intervals,
            'bootstrap_statistics': bootstrap_stats,
            'n_successful': len(bootstrap_results),
            'n_failed': self.n_bootstrap - len(bootstrap_results),
            'mode': mode
        }

    def _calculate_confidence_intervals(
        self,
        bootstrap_results: List[Dict],
        mode: str
    ) -> pd.DataFrame:
        """
        신뢰구간 계산

        Args:
            bootstrap_results: 부트스트랩 결과 리스트
            mode: 'stage1', 'stage2', 'both'

        Returns:
            신뢰구간 DataFrame
        """
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        # 파라미터별로 값 수집
        param_values = {}

        for result in bootstrap_results:
            if mode == 'stage1':
                params = result.get('stage1_params', {})
            elif mode == 'stage2':
                params = result.get('stage2_params', {})
            else:  # both
                params = {**result.get('stage1_params', {}), **result.get('stage2_params', {})}

            for param_name, param_value in params.items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)

        # 신뢰구간 계산
        ci_data = []
        for param_name, values in param_values.items():
            values_array = np.array(values)

            lower_ci = np.percentile(values_array, lower_percentile)
            upper_ci = np.percentile(values_array, upper_percentile)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)

            # p-value 계산 (부트스트랩 분포 기반)
            # 귀무가설: 파라미터 = 0
            # 양측 검정
            n_samples = len(values_array)
            if mean_val >= 0:
                # 평균이 양수면, 0보다 작거나 같은 샘플 비율
                p_lower = np.sum(values_array <= 0) / n_samples
                p_value = 2 * p_lower
            else:
                # 평균이 음수면, 0보다 크거나 같은 샘플 비율
                p_upper = np.sum(values_array >= 0) / n_samples
                p_value = 2 * p_upper

            # p-value는 최대 1.0
            p_value = min(p_value, 1.0)

            # 0을 포함하지 않으면 유의
            significant = not (lower_ci <= 0 <= upper_ci)

            ci_data.append({
                'parameter': param_name,
                'mean': mean_val,
                'std': std_val,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'p_value': p_value,
                'significant': significant
            })

        return pd.DataFrame(ci_data)

    def _calculate_statistics(
        self,
        bootstrap_results: List[Dict],
        mode: str
    ) -> pd.DataFrame:
        """
        부트스트랩 통계량 계산

        Args:
            bootstrap_results: 부트스트랩 결과 리스트
            mode: 'stage1', 'stage2', 'both'

        Returns:
            통계량 DataFrame
        """
        # 파라미터별로 값 수집
        param_values = {}

        for result in bootstrap_results:
            if mode == 'stage1':
                params = result.get('stage1_params', {})
            elif mode == 'stage2':
                params = result.get('stage2_params', {})
            else:  # both
                params = {**result.get('stage1_params', {}), **result.get('stage2_params', {})}

            for param_name, param_value in params.items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)

        # 통계량 계산
        stats_data = []
        for param_name, values in param_values.items():
            values_array = np.array(values)

            stats_data.append({
                'parameter': param_name,
                'mean': np.mean(values_array),
                'std': np.std(values_array, ddof=1),
                'median': np.median(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array)
            })

        return pd.DataFrame(stats_data)


def _bootstrap_worker(args: Tuple) -> Optional[Dict]:
    """
    부트스트랩 워커 함수 (병렬 처리용)

    Args:
        args: (sample_idx, data, individual_ids, measurement_model, structural_model,
               choice_model, random_seed, mode, factor_scores)

    Returns:
        부트스트랩 결과 딕셔너리 또는 None (실패 시)
    """
    try:
        if len(args) == 8:
            # stage1 또는 both
            sample_idx, data, individual_ids, measurement_model, structural_model, choice_model, random_seed, mode = args
            factor_scores = None
        else:
            # stage2
            sample_idx, data, individual_ids, measurement_model, structural_model, choice_model, random_seed, mode, factor_scores = args

        # 시드 설정
        np.random.seed(random_seed + sample_idx)

        # 개인별 리샘플링
        bootstrap_ids = np.random.choice(individual_ids, size=len(individual_ids), replace=True)

        # 리샘플링된 데이터 생성
        bootstrap_data_list = []
        for boot_id in bootstrap_ids:
            individual_data = data[data['respondent_id'] == boot_id].copy()
            bootstrap_data_list.append(individual_data)

        bootstrap_data = pd.concat(bootstrap_data_list, ignore_index=True)

        result = {'sample_idx': sample_idx}

        if mode in ['stage1', 'both']:
            # 1단계 추정
            stage1_result = _run_stage1(bootstrap_data, measurement_model, structural_model)
            result['stage1_params'] = stage1_result['params']
            result['stage1_ll'] = stage1_result['log_likelihood']

            if mode == 'both':
                # 요인점수 추출
                factor_scores = stage1_result['factor_scores']

        if mode in ['stage2', 'both']:
            # 2단계 추정
            if mode == 'stage2':
                # 요인점수도 리샘플링
                resampled_factor_scores = {}
                for lv_name, scores in factor_scores.items():
                    # 개인 ID 순서대로 매핑
                    id_to_score = {individual_ids[i]: scores[i] for i in range(len(individual_ids))}
                    # 리샘플링된 ID 순서로 요인점수 추출
                    resampled_scores = np.array([id_to_score[boot_id] for boot_id in bootstrap_ids])
                    resampled_factor_scores[lv_name] = resampled_scores

                stage2_result = _run_stage2(bootstrap_data, resampled_factor_scores, choice_model)
            else:
                # both 모드: 1단계에서 추출한 요인점수 사용
                stage2_result = _run_stage2(bootstrap_data, factor_scores, choice_model)

            result['stage2_params'] = stage2_result['params']
            result['stage2_ll'] = stage2_result['log_likelihood']

        return result

    except Exception as e:
        logger.warning(f"부트스트랩 샘플 {sample_idx} 실패: {e}")
        return None


def _run_stage1(data: pd.DataFrame, measurement_model, structural_model) -> Dict[str, Any]:
    """
    1단계 추정 (SEM)

    Args:
        data: 부트스트랩 데이터
        measurement_model: 측정모델 설정
        structural_model: 구조모델 설정

    Returns:
        1단계 추정 결과
    """
    from src.analysis.hybrid_choice_model.iclv_models.sem_estimator import SEMEstimator

    # SEM 추정
    sem_estimator = SEMEstimator(measurement_model, structural_model)
    sem_results = sem_estimator.estimate(data)

    # 파라미터 추출
    params = {}

    # 측정모델 파라미터
    if 'measurement' in sem_results:
        for lv_name, lv_params in sem_results['measurement'].items():
            if 'zeta' in lv_params:
                for i, zeta_val in enumerate(lv_params['zeta']):
                    params[f'zeta_{lv_name}_{i}'] = zeta_val
            if 'sigma_sq' in lv_params:
                for i, sigma_val in enumerate(lv_params['sigma_sq']):
                    params[f'sigma_sq_{lv_name}_{i}'] = sigma_val

    # 구조모델 파라미터
    if 'structural' in sem_results:
        for param_name, param_value in sem_results['structural'].items():
            params[f'gamma_{param_name}'] = param_value

    # 요인점수 추출
    factor_scores = sem_results.get('factor_scores', {})

    return {
        'params': params,
        'log_likelihood': sem_results.get('log_likelihood', np.nan),
        'factor_scores': factor_scores
    }


def _run_stage2(
    data: pd.DataFrame,
    factor_scores: Dict[str, np.ndarray],
    choice_model
) -> Dict[str, Any]:
    """
    2단계 추정 (선택모델)

    Args:
        data: 부트스트랩 선택 데이터
        factor_scores: 요인점수
        choice_model: 선택모델 설정

    Returns:
        2단계 추정 결과
    """
    from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice

    # 선택모델 추정
    mnl_model = MultinomialLogitChoice(choice_model)
    choice_results = mnl_model.fit(data, factor_scores)

    # 파라미터 추출
    params = {}
    if 'params' in choice_results:
        choice_params = choice_results['params']

        # ASC
        if 'asc_sugar' in choice_params:
            params['asc_sugar'] = choice_params['asc_sugar']
        if 'asc_sugar_free' in choice_params:
            params['asc_sugar_free'] = choice_params['asc_sugar_free']

        # Beta
        if 'beta' in choice_params:
            beta_array = choice_params['beta']
            for i, beta_val in enumerate(beta_array):
                params[f'beta_{i}'] = beta_val

        # Theta (대안별 LV 계수)
        for key, value in choice_params.items():
            if key.startswith('theta_'):
                params[key] = value

        # Gamma (LV-Attribute 상호작용)
        for key, value in choice_params.items():
            if key.startswith('gamma_'):
                params[key] = value

        # Lambda (LV 주효과)
        for key, value in choice_params.items():
            if key.startswith('lambda_'):
                params[key] = value

    return {
        'params': params,
        'log_likelihood': choice_results.get('log_likelihood', np.nan)
    }


# 편의 함수들
def bootstrap_stage1_only(
    data: pd.DataFrame,
    measurement_model,
    structural_model,
    n_bootstrap: int = 100,
    n_workers: Optional[int] = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    1단계만 부트스트래핑 (편의 함수)

    Args:
        data: 전체 데이터
        measurement_model: 측정모델 설정
        structural_model: 구조모델 설정
        n_bootstrap: 부트스트랩 샘플 수
        n_workers: 병렬 작업 수
        confidence_level: 신뢰수준
        random_seed: 랜덤 시드
        show_progress: 진행 상황 표시

    Returns:
        부트스트랩 결과
    """
    bootstrapper = SequentialBootstrap(
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
        confidence_level=confidence_level,
        random_seed=random_seed,
        show_progress=show_progress
    )

    return bootstrapper.run_stage1_bootstrap(data, measurement_model, structural_model)


def bootstrap_stage2_only(
    choice_data: pd.DataFrame,
    factor_scores: Dict[str, np.ndarray],
    choice_model,
    n_bootstrap: int = 100,
    n_workers: Optional[int] = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    2단계만 부트스트래핑 (편의 함수)

    Args:
        choice_data: 선택 데이터
        factor_scores: 1단계 요인점수 (고정)
        choice_model: 선택모델 설정
        n_bootstrap: 부트스트랩 샘플 수
        n_workers: 병렬 작업 수
        confidence_level: 신뢰수준
        random_seed: 랜덤 시드
        show_progress: 진행 상황 표시

    Returns:
        부트스트랩 결과
    """
    bootstrapper = SequentialBootstrap(
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
        confidence_level=confidence_level,
        random_seed=random_seed,
        show_progress=show_progress
    )

    return bootstrapper.run_stage2_bootstrap(choice_data, factor_scores, choice_model)


def bootstrap_both_stages(
    data: pd.DataFrame,
    measurement_model,
    structural_model,
    choice_model,
    n_bootstrap: int = 100,
    n_workers: Optional[int] = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    1+2단계 전체 부트스트래핑 (편의 함수)

    Args:
        data: 전체 데이터
        measurement_model: 측정모델 설정
        structural_model: 구조모델 설정
        choice_model: 선택모델 설정
        n_bootstrap: 부트스트랩 샘플 수
        n_workers: 병렬 작업 수
        confidence_level: 신뢰수준
        random_seed: 랜덤 시드
        show_progress: 진행 상황 표시

    Returns:
        부트스트랩 결과
    """
    bootstrapper = SequentialBootstrap(
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
        confidence_level=confidence_level,
        random_seed=random_seed,
        show_progress=show_progress
    )

    return bootstrapper.run_both_stages_bootstrap(
        data, measurement_model, structural_model, choice_model
    )

