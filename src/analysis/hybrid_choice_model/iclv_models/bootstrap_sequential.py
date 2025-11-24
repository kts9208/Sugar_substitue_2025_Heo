"""
순차추정 부트스트래핑 모듈

⚠️ 중요: 항상 1+2단계 통합 부트스트래핑을 수행합니다.
- 각 부트스트랩 샘플마다 1단계(SEM) → 2단계(선택모델)를 순차적으로 실행
- 1단계의 불확실성을 2단계 신뢰구간에 반영
- 이론적으로 올바른 순차추정 부트스트래핑

사용 예제:
    from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_both_stages

    results = bootstrap_both_stages(
        data=data,
        measurement_model=config.measurement_configs,
        structural_model=config.structural,
        choice_model=choice_config,
        n_bootstrap=1000,
        n_workers=6,
        confidence_level=0.95,
        random_seed=42
    )

Author: ICLV Team
Date: 2025-01-16
Updated: 2025-11-23 (항상 both 모드로 통합)
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

    ⚠️ 항상 1+2단계 통합 부트스트래핑을 수행합니다.
    - 각 부트스트랩 샘플마다 1단계(SEM) → 2단계(선택모델)를 순차 실행
    - 1단계의 불확실성을 2단계 신뢰구간에 반영
    - 이론적으로 올바른 순차추정 표준오차 추정

    Note: run_stage1_bootstrap, run_stage2_bootstrap은 deprecated되었습니다.
          항상 run_both_stages_bootstrap을 사용하세요.
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
        ⚠️ DEPRECATED: 이 메서드는 더 이상 권장되지 않습니다.

        1단계만 부트스트래핑하면 1단계의 불확실성이 2단계에 반영되지 않습니다.
        대신 run_both_stages_bootstrap()을 사용하세요.

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
        import warnings
        warnings.warn(
            "run_stage1_bootstrap()은 deprecated되었습니다. "
            "1단계의 불확실성을 2단계에 반영하려면 run_both_stages_bootstrap()을 사용하세요.",
            DeprecationWarning,
            stacklevel=2
        )

        logger.info("=" * 70)
        logger.info("⚠️  1단계만 부트스트래핑 (SEM Only) - DEPRECATED")
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
        ⚠️ DEPRECATED: 이 메서드는 더 이상 권장되지 않습니다.

        2단계만 부트스트래핑하면 1단계의 불확실성이 반영되지 않아
        신뢰구간이 과소추정됩니다.
        대신 run_both_stages_bootstrap()을 사용하세요.

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
        import warnings
        warnings.warn(
            "run_stage2_bootstrap()은 deprecated되었습니다. "
            "1단계의 불확실성을 반영하려면 run_both_stages_bootstrap()을 사용하세요.",
            DeprecationWarning,
            stacklevel=2
        )

        logger.info("=" * 70)
        logger.info("⚠️  2단계만 부트스트래핑 (Choice Model Only) - DEPRECATED")
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
        ✅ 권장: 1+2단계 전체 부트스트래핑 (순차추정)

        각 부트스트랩 샘플마다:
        1. 개인별 리샘플링
        2. Step 1: SEM 재추정 → 요인점수 추출
        3. Step 2: 선택모델 재추정 (1단계 요인점수 사용)
        4. 파라미터 저장

        이 방법은 1단계의 불확실성을 2단계 신뢰구간에 반영하므로
        이론적으로 올바른 순차추정 표준오차를 제공합니다.

        ✅ Sign Correction이 자동으로 적용됩니다.

        Args:
            data: 전체 데이터
            measurement_model: 측정모델 설정
            structural_model: 구조모델 설정
            choice_model: 선택모델 설정

        Returns:
            부트스트랩 결과 딕셔너리:
            - bootstrap_estimates: 각 샘플의 파라미터 추정치
            - confidence_intervals: 파라미터별 신뢰구간
            - bootstrap_statistics: 평균, 표준편차 등
            - n_successful: 성공한 샘플 수
            - n_failed: 실패한 샘플 수
            - mode: 'both'
            - sign_flip_statistics: 부호 반전 통계 (Sign Correction)
        """
        logger.info("=" * 70)
        logger.info("✅ 1+2단계 통합 부트스트래핑 시작 (Full Sequential)")
        logger.info("=" * 70)

        # 개인 ID 추출
        individual_ids = data['respondent_id'].unique()
        n_individuals = len(individual_ids)

        print(f"\n[1+2단계 부트스트래핑]")
        print(f"  샘플 수: {self.n_bootstrap}")
        print(f"  병렬 작업 수: {self.n_workers}")
        print(f"  신뢰수준: {self.confidence_level}")
        print(f"  개인 수: {n_individuals}")

        # ✅ Sign Correction을 위한 원본 적재량 추출
        logger.info("\n원본 데이터로 1단계 추정 중 (Sign Correction 기준점)...")
        original_sem_results = _run_stage1(data, measurement_model, structural_model)
        original_loadings = original_sem_results['loadings']  # DataFrame
        logger.info(f"원본 적재량 추출 완료")
        logger.info("✅ Sign Correction 활성화됨 (요인적재량 내적 기반)")

        # 워커 함수 인자 준비 (원본 적재량 추가)
        worker_args = [
            (i, data, individual_ids, measurement_model, structural_model, choice_model,
             self.random_seed, 'both', original_loadings)
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

        # ✅ Sign Flip 통계 계산 (both 모드에서만)
        sign_flip_stats = None
        if mode == 'both':
            sign_flip_stats = self._calculate_sign_flip_statistics(bootstrap_results)
            if sign_flip_stats is not None and len(sign_flip_stats) > 0:
                print(f"\n✅ Sign Correction 통계:")
                print(f"  총 잠재변수 수: {len(sign_flip_stats)}")
                n_flipped_total = sign_flip_stats['n_flipped'].sum()
                n_total = sign_flip_stats['n_total'].sum()
                flip_rate_avg = sign_flip_stats['flip_rate'].mean()
                print(f"  총 부호 반전 횟수: {n_flipped_total}/{n_total} ({flip_rate_avg*100:.1f}%)")

                # 반전율이 높은 변수 경고
                high_flip_vars = sign_flip_stats[sign_flip_stats['flip_rate'] > 0.3]
                if len(high_flip_vars) > 0:
                    print(f"  ⚠️  부호 반전율 > 30%인 변수: {len(high_flip_vars)}개")
                    for _, row in high_flip_vars.iterrows():
                        print(f"    - {row['lv_name']}: {row['flip_rate']*100:.1f}%")

        result_dict = {
            'bootstrap_estimates': bootstrap_results,
            'confidence_intervals': confidence_intervals,
            'bootstrap_statistics': bootstrap_stats,
            'n_successful': len(bootstrap_results),
            'n_failed': self.n_bootstrap - len(bootstrap_results),
            'mode': mode
        }

        if sign_flip_stats is not None:
            result_dict['sign_flip_statistics'] = sign_flip_stats

        return result_dict

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

    def _calculate_sign_flip_statistics(self, bootstrap_results: List[Dict]) -> Optional[pd.DataFrame]:
        """
        부호 반전 통계 계산 (Sign Correction)

        Args:
            bootstrap_results: 부트스트랩 결과 리스트

        Returns:
            부호 반전 통계 DataFrame
            - lv_name: 잠재변수 이름
            - n_flipped: 부호 반전된 횟수
            - n_total: 총 샘플 수
            - flip_rate: 부호 반전율 (0~1)
        """
        flip_counts = {}

        for result in bootstrap_results:
            if result is None or 'sign_flip_status' not in result:
                continue

            for lv_name, flipped in result['sign_flip_status'].items():
                if lv_name not in flip_counts:
                    flip_counts[lv_name] = {'flipped': 0, 'total': 0}

                flip_counts[lv_name]['total'] += 1
                if flipped:
                    flip_counts[lv_name]['flipped'] += 1

        if not flip_counts:
            return None

        # DataFrame 생성
        stats_list = []
        for lv_name, counts in flip_counts.items():
            stats_list.append({
                'lv_name': lv_name,
                'n_flipped': counts['flipped'],
                'n_total': counts['total'],
                'flip_rate': counts['flipped'] / counts['total'] if counts['total'] > 0 else 0.0
            })

        return pd.DataFrame(stats_list)


def _bootstrap_worker(args: Tuple) -> Optional[Dict]:
    """
    부트스트랩 워커 함수 (병렬 처리용)

    Args:
        args: (sample_idx, data, individual_ids, measurement_model, structural_model,
               choice_model, random_seed, mode, original_loadings)

        original_loadings: Sign Correction을 위한 원본 요인적재량 DataFrame (both 모드에서만)

    Returns:
        부트스트랩 결과 딕셔너리 또는 None (실패 시)
    """
    try:
        if len(args) == 8:
            # stage1 (deprecated)
            sample_idx, data, individual_ids, measurement_model, structural_model, choice_model, random_seed, mode = args
            factor_scores = None
            original_loadings = None
        elif len(args) == 9:
            # both (with Sign Correction) 또는 stage2
            sample_idx, data, individual_ids, measurement_model, structural_model, choice_model, random_seed, mode, loadings_or_scores = args
            if mode == 'both':
                # both 모드: loadings_or_scores는 원본 적재량 DataFrame
                original_loadings = loadings_or_scores
                factor_scores = None
            else:
                # stage2 모드: loadings_or_scores는 요인점수
                factor_scores = loadings_or_scores
                original_loadings = None
        else:
            raise ValueError(f"잘못된 인자 개수: {len(args)}")

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

                # ✅ Sign Correction 적용 (요인적재량 내적 기반)
                if original_loadings is not None and len(original_loadings) > 0:
                    flip_status = {}
                    debug_info = {}  # 디버깅 정보 저장
                    correlations = {}  # 내적 값 저장

                    # 부트스트랩 적재량
                    boot_loadings = stage1_result['loadings']

                    # 각 잠재변수별로 원본 적재량과 부트스트랩 적재량의 내적 계산
                    for lv_name in factor_scores.keys():
                        # 원본 적재량 (해당 잠재변수)
                        orig_lv_loadings = original_loadings[
                            (original_loadings['op'] == '~') &
                            (original_loadings['rval'] == lv_name)
                        ].copy()

                        # 부트스트랩 적재량 (해당 잠재변수)
                        boot_lv_loadings = boot_loadings[
                            (boot_loadings['op'] == '~') &
                            (boot_loadings['rval'] == lv_name)
                        ].copy()

                        # Marker Variable 제외 (첫 번째 지표는 1.0으로 고정)
                        orig_lv_loadings = orig_lv_loadings.iloc[1:]
                        boot_lv_loadings = boot_lv_loadings.iloc[1:]

                        # 적재량의 내적 계산
                        if len(orig_lv_loadings) > 0:
                            dot_product = np.dot(
                                orig_lv_loadings['Estimate'].values,
                                boot_lv_loadings['Estimate'].values
                            )
                            correlations[lv_name] = dot_product

                            # 내적이 음수면 부호 반전
                            if dot_product < 0:
                                factor_scores[lv_name] = -factor_scores[lv_name]
                                flip_status[lv_name] = True
                                if sample_idx < 5:
                                    debug_info[f'{lv_name}_flipped'] = True
                                    debug_info[f'{lv_name}_dot_product'] = dot_product
                            else:
                                flip_status[lv_name] = False
                        else:
                            # 지표가 1개뿐이면 부호 반전 불가
                            correlations[lv_name] = 1.0
                            flip_status[lv_name] = False

                    result['sign_flip_status'] = flip_status
                    result['dot_products'] = correlations  # 내적 값 저장
                    if sample_idx < 5:
                        result['debug_info'] = debug_info  # 디버깅 정보 저장

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
                # factor_scores는 Dict[str, np.ndarray] 형태 (개인별, N=328)
                # 선택 데이터(N=5904)에 맞게 확장 필요

                # 개인 ID 리스트 추출 (부트스트랩 데이터의 고유 ID)
                unique_ids = bootstrap_data['respondent_id'].unique()

                # 요인점수를 선택 데이터에 맞게 확장
                expanded_factor_scores = {}
                for lv_name, scores_array in factor_scores.items():
                    # scores_array는 np.ndarray (개인별, N=328)
                    # unique_ids와 scores_array의 길이가 같아야 함
                    if len(scores_array) != len(unique_ids):
                        raise ValueError(f"요인점수 길이({len(scores_array)})와 고유 ID 수({len(unique_ids)})가 다릅니다.")

                    # ID별 요인점수 매핑
                    id_to_score = {unique_ids[i]: scores_array[i] for i in range(len(unique_ids))}

                    # 선택 데이터의 각 행에 대응하는 요인점수 추출
                    expanded_scores = np.array([id_to_score[rid] for rid in bootstrap_data['respondent_id']])
                    expanded_factor_scores[lv_name] = expanded_scores

                stage2_result = _run_stage2(bootstrap_data, expanded_factor_scores, choice_model)

            result['stage2_params'] = stage2_result['params']
            result['stage2_ll'] = stage2_result['log_likelihood']

        return result

    except Exception as e:
        import traceback
        logger.warning(f"부트스트랩 샘플 {sample_idx} 실패: {e}")
        logger.warning(f"상세 에러:\n{traceback.format_exc()}")
        return None


def _run_stage1(data: pd.DataFrame, measurement_model, structural_model) -> Dict[str, Any]:
    """
    1단계 추정 (SEM)

    sequential_stage1_example.py와 동일한 방식으로 추정

    Args:
        data: 부트스트랩 데이터
        measurement_model: Dict[str, MeasurementConfig] (config.measurement_configs)
        structural_model: MultiLatentStructuralConfig (config.structural)

    Returns:
        1단계 추정 결과
    """
    from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import MultiLatentConfig, EstimationConfig
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural

    # sequential_stage1_example.py와 동일하게 객체 생성
    measurement_obj = MultiLatentMeasurement(measurement_model)
    structural_obj = MultiLatentStructural(structural_model)

    # Config 재구성 (estimation 설정 추가)
    config = MultiLatentConfig(
        measurement_configs=measurement_model,
        structural=structural_model,
        choice=None,  # 1단계에서는 선택모델 불필요
        estimation=EstimationConfig()  # 기본 추정 설정
    )

    estimator = SequentialEstimator(config)

    # 1단계만 추정
    sem_results = estimator.estimate_stage1_only(
        data=data,
        measurement_model=measurement_obj,
        structural_model=structural_obj,
        save_path=None,  # 부트스트랩에서는 저장 안 함
        log_file=None  # 로그 파일 없음
    )

    # 파라미터 추출
    params = {}

    # 경로계수 추출
    if 'paths' in sem_results:
        paths_df = sem_results['paths']
        for _, row in paths_df.iterrows():
            param_name = f"{row['lval']}~{row['rval']}"
            params[param_name] = row['Estimate']

    # 요인적재량 추출
    if 'loadings' in sem_results:
        loadings_df = sem_results['loadings']
        for _, row in loadings_df.iterrows():
            param_name = f"{row['lval']}=~{row['rval']}"
            params[param_name] = row['Estimate']

    # 요인점수 추출
    factor_scores = sem_results.get('factor_scores', {})

    # 요인적재량 DataFrame 추출
    loadings_df = sem_results.get('loadings', pd.DataFrame())

    return {
        'params': params,
        'log_likelihood': sem_results.get('log_likelihood', np.nan),
        'factor_scores': factor_scores,
        'loadings': loadings_df  # ✅ Sign Correction을 위해 추가
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
    ⚠️ DEPRECATED: 이 함수는 더 이상 권장되지 않습니다.

    1단계만 부트스트래핑하면 1단계의 불확실성이 2단계에 반영되지 않습니다.
    대신 bootstrap_both_stages()를 사용하세요.

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
    import warnings
    warnings.warn(
        "bootstrap_stage1_only()은 deprecated되었습니다. "
        "1단계의 불확실성을 2단계에 반영하려면 bootstrap_both_stages()를 사용하세요.",
        DeprecationWarning,
        stacklevel=2
    )

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
    ⚠️ DEPRECATED: 이 함수는 더 이상 권장되지 않습니다.

    2단계만 부트스트래핑하면 1단계의 불확실성이 반영되지 않아
    신뢰구간이 과소추정됩니다.
    대신 bootstrap_both_stages()를 사용하세요.

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
    import warnings
    warnings.warn(
        "bootstrap_stage2_only()은 deprecated되었습니다. "
        "1단계의 불확실성을 반영하려면 bootstrap_both_stages()를 사용하세요.",
        DeprecationWarning,
        stacklevel=2
    )

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
    ✅ 권장: 1+2단계 전체 부트스트래핑 (편의 함수)

    각 부트스트랩 샘플마다:
    1. 개인별 리샘플링
    2. 1단계 SEM 재추정 → 요인점수 추출
    3. 2단계 선택모델 재추정 (1단계 요인점수 사용)
    4. 파라미터 저장

    이 방법은 1단계의 불확실성을 2단계 신뢰구간에 반영하므로
    이론적으로 올바른 순차추정 표준오차를 제공합니다.

    Args:
        data: 전체 데이터
        measurement_model: 측정모델 설정
        structural_model: 구조모델 설정
        choice_model: 선택모델 설정
        n_bootstrap: 부트스트랩 샘플 수 (권장: 1000 이상)
        n_workers: 병렬 작업 수 (None이면 CPU 코어 수 - 1)
        confidence_level: 신뢰수준 (기본: 0.95)
        random_seed: 랜덤 시드
        show_progress: 진행 상황 표시

    Returns:
        부트스트랩 결과 딕셔너리:
        - bootstrap_estimates: 각 샘플의 파라미터 추정치
        - confidence_intervals: 파라미터별 신뢰구간
        - bootstrap_statistics: 평균, 표준편차 등
        - n_successful: 성공한 샘플 수
        - n_failed: 실패한 샘플 수
        - mode: 'both'
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

