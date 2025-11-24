"""
1단계 추정 (통합 버전)

이 파일 하나로 모든 1단계 추정을 수행합니다.
경로 설정만 변경하면 다양한 구조모델을 테스트할 수 있습니다.

사용법:
1. PATHS 딕셔너리에서 원하는 경로를 True/False로 설정
2. 실행하면 자동으로 경로 구성 및 파일명 생성ㅇ
3. 결과 파일명: stage1_{경로명}_results.*

주요 기능:
- 경로 설정: True/False로 간단하게 켜고 끄기
- 자동 파일명 생성: 경로에 따라 파일명 자동 생성
- 모델 설명 자동 출력: 어떤 경로가 추정되는지 명확히 표시
- 수정지수 계산: 경로 추가 제안 (선택사항)

Author: Sugar Substitute Research Team
Date: 2025-11-16
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 유틸리티 import
from model_config_utils import build_paths_from_config, LV_NAMES, LV_KOREAN

from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config


# ============================================================================
# 🎯 사용자 설정 영역 - 여기만 수정하세요!
# ============================================================================

# 경로 설정: 3경로 모델 (HC→PB→PI + HC→PP)
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': True,   # 건강관심도 → 가격수준
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': False,  # 가격수준 → 구매의도
    'NK->PI': False,  # 영양지식 → 구매의도
}

# 요인점수 변환 방법
# 'zscore': Z-score 표준화 (평균 0, 표준편차 1) - 기본값
# 'center': 중심화 (평균 0, 표준편차는 원본 유지)
STANDARDIZATION_METHOD = 'zscore'  # ✅ Z-score 표준화 사용

# 수정지수 계산 여부 (True: 경로 추가 제안, False: 제안 안 함)
CALCULATE_MODIFICATION_INDICES = False

# ============================================================================
# 🤖 자동 처리 영역 - 수정 불필요
# ============================================================================


def main():
    # 1. 경로 구성
    hierarchical_paths, path_name, model_description, n_paths = build_paths_from_config(PATHS)

    print("=" * 70)
    print(f"1단계 추정: {model_description}")
    print("=" * 70)

    if hierarchical_paths:
        print(f"\n[1] 경로 구성 완료:")
        # 경로를 보기 좋게 출력
        for i, path_dict in enumerate(hierarchical_paths, 1):
            target = path_dict['target']
            predictors = path_dict['predictors']
            # 약어로 변환
            target_abbr = [k for k, v in LV_NAMES.items() if v == target][0]
            predictor_abbrs = [k for k, v in LV_NAMES.items() if v in predictors]
            # 한글 이름도 표시
            target_kor = LV_KOREAN[target_abbr]
            predictor_kors = [LV_KOREAN[p] for p in predictor_abbrs]
            print(f"   {i}. {' + '.join(predictor_abbrs)} → {target_abbr}  ({', '.join(predictor_kors)} → {target_kor})")
    else:
        print(f"\n[1] 경로 없음 (Base Model)")

    print(f"\n📁 결과 파일명: stage1_{path_name}_results.*")

    # 2. 데이터 로드
    print("\n[2] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")

    # 3. 설정 생성
    print("\n[3] 모델 설정 중...")
    if hierarchical_paths:
        config = create_sugar_substitute_multi_lv_config(custom_paths=hierarchical_paths)
    else:
        # Base model: 경로 없이 CFA만
        config = create_sugar_substitute_multi_lv_config(use_hierarchical=False)
    print("✅ 설정 완료")

    # 4. 모델 생성
    print("\n[4] 모델 생성 중...")
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    estimator = SequentialEstimator(config, standardization_method=STANDARDIZATION_METHOD)
    print("✅ 모델 생성 완료")
    print(f"   - 요인점수 변환 방법: {STANDARDIZATION_METHOD}")

    # 5. 1단계 추정
    print("\n[5] 1단계 추정 실행 중...")

    # 최종 결과 폴더에 저장 (경로 개수별로 폴더 분리)
    save_dir = project_root / "results" / "final" / "sequential" / path_name / "stage1"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"stage1_{path_name}_results.pkl"

    results = estimator.estimate_stage1_only(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        save_path=str(save_path),
        calculate_modification_indices=CALCULATE_MODIFICATION_INDICES
    )

    print("\n✅ 1단계 추정 완료!")

    # 6. 결과 출력
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    # 로그우도
    print(f"\n[로그우도] {results['log_likelihood']:.2f}")

    # 적합도 지수
    fit = results['fit_indices']
    print("\n[적합도 지수]")
    for key, value in fit.items():
        print(f"  {key:8s}: {value:7.4f}")

    # 경로계수
    if hierarchical_paths:
        paths = results['paths']
        print(f"\n[잠재변수 간 경로계수] {len(paths)}개")
        print("\n" + "-" * 100)
        print(f"{'종속변수':20s} {'←':3s} {'예측변수':20s} {'계수':>10s} {'표준오차':>10s} {'p-value':>10s} {'유의성':>8s}")
        print("-" * 100)

        for _, row in paths.iterrows():
            sig = "***" if row['p-value'] < 0.001 else "**" if row['p-value'] < 0.01 else "*" if row['p-value'] < 0.05 else ""
            print(f"{row['lval']:20s} {'←':3s} {row['rval']:20s} {row['Estimate']:10.4f} {row['Std. Err']:10.4f} {row['p-value']:10.4f} {sig:>8s}")

        print("-" * 100)

        # 유의한 경로 개수
        n_sig = (paths['p-value'] < 0.05).sum()
        print(f"\n유의한 경로 (p<0.05): {n_sig}/{len(paths)}개")

    # 수정지수 (요청한 경우)
    if CALCULATE_MODIFICATION_INDICES and 'modification_indices' in results:
        mod_indices = results['modification_indices']
        if mod_indices is not None and len(mod_indices) > 0:
            print("\n[수정지수 (상위 5개)]")
            print("추가하면 모델 적합도가 개선될 수 있는 경로:")
            print("-" * 70)
            for _, row in mod_indices.head(5).iterrows():
                print(f"  {row['lhs']:20s} → {row['rhs']:20s}: MI = {row['mi']:7.2f}")

    print("\n" + "=" * 70)
    print("저장된 파일")
    print("=" * 70)

    print(f"\n  📁 {save_path.parent / f'stage1_{path_name}_results_paths.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_loadings.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_fit_indices.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_factor_scores.csv'}")

    print("\n" + "=" * 70)
    print("다음 단계")
    print("=" * 70)
    print(f"\n2단계 선택모델을 추정하려면:")
    print(f"  1. examples/sequential_stage2_with_extended_model.py 열기")
    print(f"  2. STAGE1_RESULT_FILE = 'stage1_{path_name}_results.pkl' 설정")
    print(f"  3. python examples/sequential_stage2_with_extended_model.py 실행")


if __name__ == "__main__":
    main()

