"""
1단계 추정 (유연한 경로 설정)

경로를 쉽게 추가/삭제하고, 결과 파일명에 자동으로 반영됩니다.

사용법:
1. PATHS 딕셔너리에서 원하는 경로를 True/False로 설정
2. 실행하면 자동으로 경로 구성 및 파일명 생성

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

from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config


# ============================================================================
# 경로 설정: True/False로 간단하게 켜고 끄기
# ============================================================================
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': True,   # 건강관심도 → 가격수준 (NEW)
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': True,   # 가격수준 → 구매의도 (NEW)
    'NK->PI': False,  # 영양지식 → 구매의도
}

# 수정지수 계산 여부
CALCULATE_MODIFICATION_INDICES = False  # True: 경로 추가 제안, False: 제안 안 함

# 약어 매핑
LV_NAMES = {
    'HC': 'health_concern',
    'PB': 'perceived_benefit',
    'PP': 'perceived_price',
    'NK': 'nutrition_knowledge',
    'PI': 'purchase_intention'
}


def build_paths_from_config(paths_config):
    """
    경로 설정에서 hierarchical_paths 생성
    
    Args:
        paths_config: {'HC->PB': True, ...} 형태의 딕셔너리
    
    Returns:
        hierarchical_paths: [{'target': ..., 'predictors': [...]}, ...]
        path_name: 파일명용 경로 이름 (예: 'HC-PB_PB-PI')
    """
    # 활성화된 경로만 필터링
    active_paths = {k: v for k, v in paths_config.items() if v}
    
    if not active_paths:
        raise ValueError("최소 1개 이상의 경로를 활성화해야 합니다.")
    
    # 경로를 target별로 그룹화
    target_predictors = {}
    
    for path_str in active_paths.keys():
        # 'HC->PB' 형태를 파싱
        parts = path_str.split('->')
        if len(parts) != 2:
            raise ValueError(f"잘못된 경로 형식: {path_str}. 'LV1->LV2' 형태여야 합니다.")
        
        predictor_abbr, target_abbr = parts
        predictor = LV_NAMES.get(predictor_abbr)
        target = LV_NAMES.get(target_abbr)
        
        if predictor is None or target is None:
            raise ValueError(f"알 수 없는 잠재변수: {path_str}")
        
        if target not in target_predictors:
            target_predictors[target] = []
        target_predictors[target].append(predictor)
    
    # hierarchical_paths 생성
    hierarchical_paths = []
    for target, predictors in target_predictors.items():
        hierarchical_paths.append({
            'target': target,
            'predictors': predictors
        })
    
    # 파일명용 경로 이름 생성 (예: 'HC-PB_PB-PI_PP-PI_NK-PI')
    path_name = '_'.join(sorted(active_paths.keys())).replace('->', '-')
    
    return hierarchical_paths, path_name, active_paths


def main():
    print("=" * 70)
    print("1단계 추정: 유연한 경로 설정")
    print("=" * 70)
    
    # 1. 경로 구성
    print("\n[1] 경로 구성 중...")
    hierarchical_paths, path_name, active_paths = build_paths_from_config(PATHS)
    
    print(f"✅ 활성화된 경로 ({len(active_paths)}개):")
    for i, path_str in enumerate(sorted(active_paths.keys()), 1):
        print(f"   {i}. {path_str}")
    
    print(f"\n📁 결과 파일명: stage1_{path_name}_results.*")
    
    # 2. 데이터 로드
    print("\n[2] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data_cleaned.csv"
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")
    
    # 3. 설정 생성
    print("\n[3] 모델 설정 중...")
    config = create_sugar_substitute_multi_lv_config(custom_paths=hierarchical_paths)
    print("✅ 설정 완료")
    
    # 4. 모델 생성
    print("\n[4] 모델 생성 중...")
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    estimator = SequentialEstimator(config)
    print("✅ 모델 생성 완료")
    
    # 5. 1단계 추정
    print("\n[5] 1단계 추정 실행 중...")

    save_path = project_root / "results" / "sequential_stage_wise" / f"stage1_{path_name}_results.pkl"

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
    print(f"  CFI:   {fit['CFI']:.4f}")
    print(f"  TLI:   {fit['TLI']:.4f}")
    print(f"  RMSEA: {fit['RMSEA']:.4f}")
    print(f"  AIC:   {fit['AIC']:.4f}")
    print(f"  BIC:   {fit['BIC']:.4f}")

    # 경로계수
    paths = results['paths']
    print(f"\n[잠재변수 간 경로계수] {len(paths)}개")
    print("\n" + "-" * 80)
    print(f"{'종속변수':20s} {'←':3s} {'예측변수':20s} {'계수':>10s} {'표준오차':>10s} {'p-value':>10s} {'유의성':>8s}")
    print("-" * 80)

    for _, row in paths.iterrows():
        sig = "***" if row['p-value'] < 0.001 else "**" if row['p-value'] < 0.01 else "*" if row['p-value'] < 0.05 else ""
        print(f"{row['lval']:20s} {'←':3s} {row['rval']:20s} {row['Estimate']:10.4f} {row['Std. Err']:10.4f} {row['p-value']:10.4f} {sig:>8s}")

    print("-" * 80)

    # 유의한 경로 개수
    n_sig = (paths['p-value'] < 0.05).sum()
    print(f"\n유의한 경로 (p<0.05): {n_sig}/{len(paths)}개")

    print("\n" + "=" * 70)
    print("저장된 파일")
    print("=" * 70)

    print(f"\n  📁 {save_path.parent / f'stage1_{path_name}_results_paths.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_loadings.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_fit_indices.csv'}")
    print(f"  📁 {save_path.parent / f'stage1_{path_name}_results_factor_scores.csv'}")

    # 7. 수정지수 결과 출력
    if CALCULATE_MODIFICATION_INDICES and 'modification_indices' in results:
        print("\n" + "=" * 70)
        print("수정지수 (Modification Indices) - 경로 추가 제안")
        print("=" * 70)

        mi_results = results['modification_indices']
        suggestions = mi_results.get('suggestions', [])

        if len(suggestions) > 0:
            print(f"\n💡 {len(suggestions)}개 경로 추가를 제안합니다:\n")
            print("-" * 80)
            print(f"{'순위':>4s} {'경로':30s} {'MI':>10s} {'p-value':>10s} {'예상 계수':>12s} {'추천':20s}")
            print("-" * 80)

            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i:4d} {suggestion['path']:30s} {suggestion['MI']:10.2f} {suggestion['p_value']:10.4f} "
                      f"{suggestion['expected_change']:12.4f} {suggestion['recommendation']:20s}")

            print("-" * 80)
            print(f"\n📌 해석:")
            print(f"  - MI > 10.83: 강력 추천 (p<0.001)")
            print(f"  - MI > 6.63:  추천 (p<0.01)")
            print(f"  - MI > 3.84:  고려 가능 (p<0.05)")
            print(f"\n💡 제안된 경로를 PATHS 딕셔너리에 추가하여 재실행하세요!")
        else:
            print(f"\n✅ {mi_results.get('message', '제안할 경로가 없습니다.')}")


if __name__ == "__main__":
    main()

