"""
전체 순차추정 테스트 스크립트 (실제 데이터)

Step 1: SEM 추정 (측정모델 + 구조모델)
Step 2: 선택모델 추정 (요인점수 사용)

Author: Sugar Substitute Research Team
Date: 2025-11-15
"""

import sys
from pathlib import Path
import time
import logging

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ✅ 로깅 설정
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'sequential_estimation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 공통 유틸리티 임포트
from iclv_test_utils import (
    load_integrated_data,
    get_cpu_info,
    print_config_summary,
    save_results_to_csv
)

from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    create_default_multi_lv_config
)


def format_pvalue(p):
    """
    p-value를 적절한 형식으로 표기

    Args:
        p: p-value

    Returns:
        str: 포맷된 p-value 문자열
    """
    if p < 0.001:
        return "<0.001"
    else:
        return f"{p:.4f}"


def create_config():
    """
    5개 잠재변수 설정 생성

    동시추정(test_multi_latent_iclv.py)과 동일한 설정 사용
    """
    print("\n설정 생성 중...")

    # create_default_multi_lv_config() 사용 (동시추정과 동일)
    # 순차추정에서는 n_draws, max_iterations 등은 사용하지 않음
    config = create_default_multi_lv_config(
        n_draws=100,  # 순차추정에서는 사용 안 함
        max_iterations=1000,  # 순차추정에서는 사용 안 함
        use_parallel=False,
        n_cores=1
    )

    print("   설정 완료")

    return config


def print_results(results, elapsed_time):
    """
    결과 출력

    Args:
        results: 추정 결과
        elapsed_time: 소요 시간 (초)
    """
    print("\n" + "="*70)
    print("순차추정 결과")
    print("="*70)
    print(f"\n추정 시간: {elapsed_time/60:.2f}분 ({elapsed_time:.1f}초)")

    # 결과 구조 확인
    print(f"\n결과 키: {list(results.keys())}")

    # Step 1: SEM 결과
    if 'stage_results' in results and 'measurement' in results['stage_results']:
        sem_results = results['stage_results']['measurement']['full_results']
        print("\n[Step 1: SEM 추정 결과]")
        print(f"  로그우도: {sem_results.get('log_likelihood', 'N/A')}")

        if 'fit_indices' in sem_results:
            print("\n  적합도 지수:")
            for index, value in sem_results['fit_indices'].items():
                print(f"    {index}: {value:.4f}")

        if 'loadings' in sem_results:
            print(f"\n  요인적재량: {len(sem_results['loadings'])}개")
        if 'paths' in sem_results:
            print(f"  경로계수: {len(sem_results['paths'])}개")

    # Step 2: 선택모델 결과
    if 'stage_results' in results and 'choice' in results['stage_results']:
        choice_results = results['stage_results']['choice']
        print("\n[Step 2: 선택모델 추정 결과]")
        print(f"  로그우도: {choice_results.get('log_likelihood', 'N/A'):.2f}")
        print(f"  AIC: {choice_results.get('aic', 'N/A'):.2f}")
        print(f"  BIC: {choice_results.get('bic', 'N/A'):.2f}")

        if 'parameter_statistics' in choice_results and choice_results['parameter_statistics'] is not None:
            print("\n  파라미터 (유의성 포함):")
            stats = choice_results['parameter_statistics']

            # Intercept
            if 'intercept' in stats:
                s = stats['intercept']
                sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
                p_str = format_pvalue(s['p'])
                print(f"    intercept: {s['estimate']:.4f} (p={p_str}) {sig}")

            # Beta
            if 'beta' in stats:
                for attr, s in stats['beta'].items():
                    sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
                    p_str = format_pvalue(s['p'])
                    print(f"    β_{attr}: {s['estimate']:.4f} (p={p_str}) {sig}")

            # Lambda
            if 'lambda_main' in stats:
                s = stats['lambda_main']
                sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
                p_str = format_pvalue(s['p'])
                print(f"    λ_main: {s['estimate']:.4f} (p={p_str}) {sig}")

            for key in ['lambda_mod_perceived_price', 'lambda_mod_nutrition_knowledge']:
                if key in stats:
                    s = stats[key]
                    sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
                    p_str = format_pvalue(s['p'])
                    print(f"    {key}: {s['estimate']:.4f} (p={p_str}) {sig}")

            print("\n  유의수준: *** p<0.001, ** p<0.01, * p<0.05")


def main():
    """메인 실행 함수"""
    print("="*70)
    print("ICLV 순차추정 (5개 잠재변수)")
    print("="*70)

    # 1. 데이터 로드 (공통 유틸리티 사용)
    data = load_integrated_data()

    # 2. CPU 정보 (공통 유틸리티 사용)
    n_cpus, n_cores = get_cpu_info()
    print(f"\n사용 가능한 CPU 코어: {n_cpus}개")

    # 3. 설정 생성
    config = create_config()

    # 4. 설정 요약 출력 (공통 유틸리티 사용)
    print_config_summary(config, use_parallel=False, n_cores=1, n_cpus=n_cpus)

    # 5. 모델 생성
    print("\n모델 생성 중...")
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    print("   - 측정모델 생성 완료 (5개 잠재변수)")

    structural_model = MultiLatentStructural(config.structural)
    print("   - 구조모델 생성 완료 (계층적: HC → PB → PI)")

    choice_model = MultinomialLogitChoice(config.choice)
    print("   - 선택모델 생성 완료 (조절효과 포함)")

    # 6. 순차추정 실행
    print("\n순차추정 실행 중...")
    print("   (Step 1: SEM, Step 2: Choice Model)")
    print("\n   [주의] 순차추정은 2-5분 정도 소요될 수 있습니다...")

    start_time = time.time()

    estimator = SequentialEstimator(config)
    results = estimator.estimate(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        choice_model=choice_model
    )

    elapsed_time = time.time() - start_time

    # 7. 결과 출력
    print_results(results, elapsed_time)

    # 8. 결과 저장 (공통 유틸리티 사용)
    output_dir = project_root / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'sequential_estimation_results.csv'

    save_results_to_csv(results, output_path, estimation_type='sequential')

    print("\n" + "="*70)
    print("[완료] 순차추정 완료!")
    print("="*70)

    return results


if __name__ == '__main__':
    results = main()

