"""
1단계 추정: HC→PB→PI

경로 구조:
- HC → PB (건강관심도 → 건강유익성)
- PB → PI (건강유익성 → 구매의도)

Author: ICLV Team
Date: 2025-11-23
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_config_utils import (
    build_paths_from_config,
    LV_NAMES,
    LV_KOREAN
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
import pandas as pd


# ============================================================================
# 🎯 사용자 설정 영역 - 여기만 수정하세요!
# ============================================================================

# 경로 설정: HC→PB→PI
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': False,  # 건강관심도 → 가격수준
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': False,  # 가격수준 → 구매의도
    'NK->PI': False,  # 영양지식 → 구매의도
}

# 요인점수 변환 방법
STANDARDIZATION_METHOD = 'zscore'  # Z-score 표준화 사용

# 수정지수 계산 여부
CALCULATE_MODIFICATION_INDICES = False


def main():
    # 1. 경로 구성
    hierarchical_paths, path_name, model_description = build_paths_from_config(PATHS)

    print("=" * 70)
    print(f"1단계 추정: {model_description}")
    print("=" * 70)

    if hierarchical_paths:
        print(f"\n[1] 경로 구성 완료:")
        for i, path_dict in enumerate(hierarchical_paths, 1):
            target = path_dict['target']
            predictors = path_dict['predictors']
            target_abbr = [k for k, v in LV_NAMES.items() if v == target][0]
            predictor_abbrs = [k for k, v in LV_NAMES.items() if v in predictors]
            target_kor = LV_KOREAN[target_abbr]
            predictor_kors = [LV_KOREAN[p] for p in predictor_abbrs]
            print(f"   {i}. {' + '.join(predictor_abbrs)} → {target_abbr}  ({', '.join(predictor_kors)} → {target_kor})")
    else:
        print(f"\n[1] 경로 없음 (Base Model)")

    # 2. 데이터 로드
    print("\n[2] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "sugar_substitute_choice_data.csv"
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {data['respondent_id'].nunique()}명")

    # 3. 설정 생성
    print("\n[3] 설정 생성 중...")
    config = create_sugar_substitute_multi_lv_config(
        n_draws=100,
        max_iterations=1000,
        use_hierarchical=True,
        all_lvs_as_main=False,
        custom_paths=hierarchical_paths
    )
    print("✅ 설정 생성 완료")

    # 4. 모델 생성
    print("\n[4] 모델 생성 중...")
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    estimator = SequentialEstimator(config, standardization_method=STANDARDIZATION_METHOD)
    print("✅ 모델 생성 완료")
    print(f"   - 요인점수 변환 방법: {STANDARDIZATION_METHOD}")

    # 5. 1단계 추정
    print("\n[5] 1단계 추정 실행 중...")

    save_dir = project_root / "results" / "final" / "sequential" / "stage1"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"stage1_{path_name}_results.pkl"

    sem_results = estimator.estimate_stage1_only(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        save_path=str(save_path),
        log_file=None
    )

    print("\n✅ 1단계 추정 완료!")

    # 6. 결과 출력
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    if 'fit_indices' in sem_results:
        fit = sem_results['fit_indices']
        print(f"\n[적합도 지수]")
        print(f"  CFI:   {fit.get('CFI', 'N/A'):.4f}")
        print(f"  TLI:   {fit.get('TLI', 'N/A'):.4f}")
        print(f"  RMSEA: {fit.get('RMSEA', 'N/A'):.4f}")

    if 'paths' in sem_results:
        paths_df = sem_results['paths']
        print(f"\n[경로계수] ({len(paths_df)}개)")
        print(paths_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("저장된 파일")
    print("=" * 70)
    print(f"\n  📁 {save_path.parent / f'stage1_{path_name}_results_paths.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_loadings.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_fit_indices.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_factor_scores.csv'}")


if __name__ == "__main__":
    main()

