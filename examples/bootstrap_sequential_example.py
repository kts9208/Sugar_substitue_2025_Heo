"""
순차추정 부트스트래핑 예제

⚠️ 중요 (2025-11-23 업데이트):
- example_both_stages_bootstrap() 사용 권장 (--mode both)
- example_stage1_bootstrap(), example_stage2_bootstrap()는 deprecated

3가지 부트스트래핑 모드:
1. Stage 1 Only: SEM만 부트스트래핑 (❌ Deprecated)
2. Stage 2 Only: 선택모델만 부트스트래핑 (❌ Deprecated)
3. Both Stages: 1+2단계 전체 부트스트래핑 (✅ 권장)

Author: ICLV Team
Date: 2025-01-16
Updated: 2025-11-23
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import (
    bootstrap_stage1_only,
    bootstrap_stage2_only,
    bootstrap_both_stages
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig


def example_stage1_bootstrap():
    """
    1단계만 부트스트래핑 예제

    ⚠️ DEPRECATED: 이 예제는 더 이상 권장되지 않습니다.
    대신 example_both_stages_bootstrap()을 사용하세요.
    """
    import warnings
    warnings.warn(
        "example_stage1_bootstrap()은 deprecated되었습니다. "
        "1단계의 불확실성을 2단계에 반영하려면 example_both_stages_bootstrap()을 사용하세요.",
        DeprecationWarning,
        stacklevel=2
    )

    print("=" * 70)
    print("예제 1: 1단계만 부트스트래핑 (SEM) - ⚠️ DEPRECATED")
    print("=" * 70)
    
    # 데이터 로드
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    data = pd.read_csv(data_path)
    print(f"데이터 로드 완료: {len(data)}행")
    
    # 설정 생성
    config = create_sugar_substitute_multi_lv_config()
    
    # 부트스트래핑 실행
    results = bootstrap_stage1_only(
        data=data,
        measurement_model=config.measurement_configs,
        structural_model=config.structural,
        n_bootstrap=50,  # 예제용으로 적게 설정
        n_workers=4,
        confidence_level=0.95,
        random_seed=42,
        show_progress=True
    )
    
    # 결과 출력
    print("\n[신뢰구간]")
    print(results['confidence_intervals'].head(10))
    
    print("\n[부트스트랩 통계량]")
    print(results['bootstrap_statistics'].head(10))
    
    # 결과 저장
    save_dir = project_root / "results" / "bootstrap"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results['confidence_intervals'].to_csv(save_dir / "stage1_bootstrap_ci.csv", index=False)
    results['bootstrap_statistics'].to_csv(save_dir / "stage1_bootstrap_stats.csv", index=False)
    
    print(f"\n결과 저장: {save_dir}")


def example_stage2_bootstrap():
    """
    2단계만 부트스트래핑 예제

    ⚠️ DEPRECATED: 이 예제는 더 이상 권장되지 않습니다.
    대신 example_both_stages_bootstrap()을 사용하세요.
    """
    import warnings
    warnings.warn(
        "example_stage2_bootstrap()은 deprecated되었습니다. "
        "1단계의 불확실성을 2단계에 반영하려면 example_both_stages_bootstrap()을 사용하세요.",
        DeprecationWarning,
        stacklevel=2
    )

    print("\n" + "=" * 70)
    print("예제 2: 2단계만 부트스트래핑 (선택모델, 요인점수 고정) - ⚠️ DEPRECATED")
    print("=" * 70)

    # 데이터 로드
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    data = pd.read_csv(data_path)
    print(f"데이터 로드 완료: {len(data)}행")

    # 1단계 결과 로드 (요인점수)
    results_dir = project_root / "results" / "final" / "sequential" / "stage1"
    stage1_file = results_dir / "stage1_HC-PB_HC-PP_PB-PI_PP-PI_results.pkl"

    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    factor_scores = stage1_results['factor_scores']
    print(f"요인점수 로드 완료: {list(factor_scores.keys())}")

    # ✅ 선택모델 설정: Base Model (잠재변수 없음)
    choice_config = ChoiceConfig(
        choice_attributes=['health_label', 'price'],
        choice_type='binary',
        price_variable='price',
        all_lvs_as_main=False,  # 잠재변수 주효과 사용 안 함
        main_lvs=None,  # 잠재변수 없음
        moderation_enabled=False,
        lv_attribute_interactions=None  # 상호작용 없음
    )

    print(f"\n선택모델 설정:")
    print(f"   - 모델 유형: Base Model (잠재변수 없음)")
    print(f"   - 선택 속성만 사용: {choice_config.choice_attributes}")

    # 부트스트래핑 실행
    from datetime import datetime
    start_time = datetime.now()
    print(f"\n시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = bootstrap_stage2_only(
        choice_data=data,
        factor_scores=factor_scores,
        choice_model=choice_config,
        n_bootstrap=1000,  # 1000회 부트스트래핑
        n_workers=6,
        confidence_level=0.95,
        random_seed=42,
        show_progress=True
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.0f}초)")
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("부트스트래핑 결과")
    print("=" * 70)

    print(f"\n성공: {results['n_successful']}/{results['n_successful'] + results['n_failed']}")
    print(f"실패: {results['n_failed']}/{results['n_successful'] + results['n_failed']}")

    print("\n[신뢰구간]")
    print(results['confidence_intervals'].to_string(index=False))

    print("\n[부트스트랩 통계량]")
    print(results['bootstrap_statistics'].to_string(index=False))

    # 결과 저장
    save_dir = project_root / "results" / "bootstrap"
    save_dir.mkdir(parents=True, exist_ok=True)

    results['confidence_intervals'].to_csv(save_dir / "stage2_base_model_bootstrap_ci.csv", index=False)
    results['bootstrap_statistics'].to_csv(save_dir / "stage2_base_model_bootstrap_stats.csv", index=False)

    # 전체 결과 저장 (pickle)
    with open(save_dir / "stage2_base_model_bootstrap_full.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✅ 결과 저장: {save_dir}")
    print(f"   - stage2_base_model_bootstrap_ci.csv")
    print(f"   - stage2_base_model_bootstrap_stats.csv")
    print(f"   - stage2_base_model_bootstrap_full.pkl")


def example_both_stages_bootstrap():
    """
    1+2단계 전체 부트스트래핑 예제

    ✅ 권장: 항상 이 방법을 사용하세요!
    - 1단계의 불확실성을 2단계 신뢰구간에 반영
    - 이론적으로 올바른 순차추정 표준오차
    """
    print("\n" + "=" * 70)
    print("예제 3: 1+2단계 전체 부트스트래핑 (1000회) - ✅ 권장")
    print("=" * 70)

    # 데이터 로드
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    data = pd.read_csv(data_path)
    print(f"데이터 로드 완료: {len(data)}행, {data['respondent_id'].nunique()}명")

    # 설정 생성 (디폴트 계층적 구조)
    config = create_sugar_substitute_multi_lv_config()
    print(f"\n1단계 설정:")
    print(f"   - 구조모델: 계층적 구조 (디폴트)")
    print(f"     * perceived_benefit <- health_concern")
    print(f"     * purchase_intention <- perceived_benefit")

    # 선택모델 설정: Base Model + PI 주효과
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='multinomial',
        all_lvs_as_main=True,  # True로 설정해야 main_lvs가 작동
        main_lvs=['purchase_intention'],  # PI 주효과만
        lv_attribute_interactions=[]  # 상호작용 없음
    )

    print(f"\n2단계 설정:")
    print(f"   - 모델 유형: Base Model + PI 주효과")
    print(f"   - 속성변수: sugar_free, health_label, price")
    print(f"   - LV 주효과: purchase_intention (PI만)")
    print(f"   - LV-Attribute 상호작용: 없음")

    # 부트스트래핑 실행
    from datetime import datetime
    start_time = datetime.now()
    print(f"\n시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = bootstrap_both_stages(
        data=data,
        measurement_model=config.measurement_configs,
        structural_model=config.structural,
        choice_model=choice_config,
        n_bootstrap=1000,  # 1000회 부트스트래핑
        n_workers=6,
        confidence_level=0.95,
        random_seed=42,
        show_progress=True
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.0f}초)")

    # 결과 출력
    print("\n" + "=" * 70)
    print("부트스트래핑 결과")
    print("=" * 70)

    print(f"\n성공: {results['n_successful']}/{results['n_successful'] + results['n_failed']}")
    print(f"실패: {results['n_failed']}/{results['n_successful'] + results['n_failed']}")
    print(f"성공률: {results['n_successful']/(results['n_successful'] + results['n_failed'])*100:.1f}%")

    print("\n[신뢰구간 (상위 20개)]")
    print(results['confidence_intervals'].head(20).to_string(index=False))

    print("\n[부트스트랩 통계량 (상위 20개)]")
    print(results['bootstrap_statistics'].head(20).to_string(index=False))

    # 결과 저장
    save_dir = project_root / "results" / "bootstrap"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 타임스탬프 추가
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')

    ci_file = save_dir / f"both_stages_bootstrap_ci_{timestamp}.csv"
    stats_file = save_dir / f"both_stages_bootstrap_stats_{timestamp}.csv"
    full_file = save_dir / f"both_stages_bootstrap_full_{timestamp}.pkl"

    results['confidence_intervals'].to_csv(ci_file, index=False)
    results['bootstrap_statistics'].to_csv(stats_file, index=False)

    # 전체 결과 저장 (pickle)
    with open(save_dir / f"both_stages_bootstrap_full_{timestamp}.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✅ 결과 저장: {save_dir}")
    print(f"   - {ci_file.name}")
    print(f"   - {stats_file.name}")
    print(f"   - {full_file.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="순차추정 부트스트래핑 예제")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['stage1', 'stage2', 'both', 'all'],
        default='both',  # ✅ 기본값을 'both'로 변경
        help='부트스트래핑 모드 (stage1: 1단계만 [Deprecated], stage2: 2단계만 [Deprecated], both: 전체 [권장], all: 모든 예제)'
    )

    args = parser.parse_args()

    if args.mode == 'stage1':
        example_stage1_bootstrap()
    elif args.mode == 'stage2':
        example_stage2_bootstrap()
    elif args.mode == 'both':
        example_both_stages_bootstrap()
    else:  # all
        example_stage1_bootstrap()
        example_stage2_bootstrap()
        example_both_stages_bootstrap()

    print("\n" + "=" * 70)
    print("모든 부트스트래핑 완료!")
    print("=" * 70)

