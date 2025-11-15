"""
순차추정 부트스트래핑 테스트 스크립트

CPU 병렬 처리를 사용한 부트스트래핑 신뢰구간 계산

Author: Sugar Substitute Research Team
Date: 2025-11-15
"""

import sys
from pathlib import Path
import time

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 유틸리티 임포트
from iclv_test_utils import (
    load_integrated_data,
    get_cpu_info,
    print_config_summary,
    bootstrap_sequential_estimation
)

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    create_default_multi_lv_config
)


def main():
    """메인 함수"""
    print("=" * 80)
    print("순차추정 부트스트래핑 테스트 (CPU 병렬)")
    print("=" * 80)
    
    # CPU 정보
    n_cpus, n_cores = get_cpu_info()
    print(f"\nCPU 정보:")
    print(f"   전체 CPU 코어: {n_cpus}")
    print(f"   사용할 코어: {n_cores}")
    
    # 1. 데이터 로드
    data = load_integrated_data()
    
    # 2. 모델 설정
    print("\n모델 설정 중...")
    config = create_default_multi_lv_config(
        n_draws=100,  # 순차추정에서는 사용 안 함
        max_iterations=1000,  # 순차추정에서는 사용 안 함
        use_parallel=False,
        n_cores=1
    )

    # 측정모델
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    print("   - 측정모델 생성 완료 (5개 잠재변수)")

    # 구조모델
    structural_model = MultiLatentStructural(config.structural)
    print("   - 구조모델 생성 완료 (계층적: HC -> PB -> PI)")

    # 선택모델
    choice_model = MultinomialLogitChoice(config.choice)
    print("   - 선택모델 생성 완료 (조절효과 포함)")

    print_config_summary(config, use_parallel=True, n_cores=n_cores, n_cpus=n_cpus)
    
    # 3. 부트스트래핑 실행
    print("\n" + "=" * 80)
    print("부트스트래핑 시작")
    print("=" * 80)
    
    start_time = time.time()
    
    bootstrap_results = bootstrap_sequential_estimation(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        choice_model=choice_model,
        n_bootstrap=1000,  # 1000개 샘플
        n_workers=n_cores,
        confidence_level=0.95,
        random_seed=42,
        show_progress=True
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 4. 결과 출력
    print("\n" + "=" * 80)
    print("부트스트래핑 결과")
    print("=" * 80)
    
    print(f"\n총 소요 시간: {elapsed:.1f}초")
    print(f"성공한 샘플: {bootstrap_results['n_successful']}")
    print(f"실패한 샘플: {bootstrap_results['n_failed']}")
    
    # 신뢰구간 출력
    ci_df = bootstrap_results['confidence_intervals']
    stats_df = bootstrap_results['bootstrap_statistics']
    
    print(f"\n신뢰구간 계산 완료:")
    print(f"   파라미터 수: {len(ci_df)}")
    
    # 유의한 파라미터만 출력
    significant_params = ci_df[ci_df['Significant'] == True]
    print(f"   유의한 파라미터: {len(significant_params)}")
    
    # 선택모델 파라미터 출력
    print("\n선택모델 파라미터 신뢰구간 및 p-value:")
    choice_ci = ci_df[ci_df['Model'] == 'Choice'].copy()

    if len(choice_ci) > 0:
        for _, row in choice_ci.iterrows():
            param_name = row['Parameter']
            mean_val = row['Mean']
            se_val = row['SE']
            ci_lower = row['CI_Lower']
            ci_upper = row['CI_Upper']
            p_boot = row['p_value_bootstrap']
            p_norm = row['p_value_normal']
            sig = '*' if row['Significant'] else ''

            # p-value 포맷팅
            if p_boot < 0.001:
                p_boot_str = "<0.001"
            else:
                p_boot_str = f"{p_boot:.3f}"

            if p_norm < 0.001:
                p_norm_str = "<0.001"
            else:
                p_norm_str = f"{p_norm:.3f}"

            print(f"   {param_name:30s}: {mean_val:7.4f} (SE={se_val:.4f})  "
                  f"95% CI: [{ci_lower:7.4f}, {ci_upper:7.4f}]  "
                  f"p={p_boot_str} {sig}")
    
    # 5. 결과 저장
    print("\n결과 저장 중...")
    
    # 신뢰구간 저장
    ci_output_path = project_root / 'results' / 'sequential_bootstrap_ci.csv'
    ci_df.to_csv(ci_output_path, index=False, encoding='utf-8-sig')
    print(f"   신뢰구간 저장: {ci_output_path}")
    
    # 부트스트랩 통계량 저장
    stats_output_path = project_root / 'results' / 'sequential_bootstrap_stats.csv'
    stats_df.to_csv(stats_output_path, index=False, encoding='utf-8-sig')
    print(f"   통계량 저장: {stats_output_path}")
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)


if __name__ == '__main__':
    main()

