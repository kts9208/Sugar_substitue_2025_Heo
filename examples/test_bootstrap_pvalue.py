"""
부트스트래핑 p-value 테스트 (100회)
"""

import sys
from pathlib import Path
import pandas as pd
import pickle

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_stage2_only
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig


def main():
    print("=" * 70)
    print("부트스트래핑 p-value 테스트 (100회)")
    print("=" * 70)
    
    # 데이터 로드
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data_cleaned.csv"
    data = pd.read_csv(data_path)
    
    # 1단계 결과 로드
    stage1_path = project_root / "results" / "sequential_stage_wise" / "stage1_HC-PB_HC-PP_PB-PI_PP-PI_results.pkl"
    with open(stage1_path, 'rb') as f:
        stage1_results = pickle.load(f)
    factor_scores = stage1_results['factor_scores']
    
    # Base Model 설정
    choice_config = ChoiceConfig(
        choice_attributes=['health_label', 'price'],
        choice_type='binary',
        price_variable='price',
        all_lvs_as_main=False,
        main_lvs=None,
        moderation_enabled=False,
        lv_attribute_interactions=None
    )
    
    print("\n부트스트래핑 실행 중 (100회)...")
    
    results = bootstrap_stage2_only(
        choice_data=data,
        factor_scores=factor_scores,
        choice_model=choice_config,
        n_bootstrap=100,
        n_workers=6,
        confidence_level=0.95,
        random_seed=42,
        show_progress=True
    )
    
    print("\n" + "=" * 70)
    print("결과")
    print("=" * 70)
    
    print(f"\n성공: {results['n_successful']}/{results['n_successful'] + results['n_failed']}")
    
    print("\n[신뢰구간 + p-value]")
    ci_df = results['confidence_intervals']
    print(ci_df.to_string(index=False))
    
    print("\n[해석]")
    for _, row in ci_df.iterrows():
        sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['parameter']:20s}: mean={row['mean']:7.3f}, p={row['p_value']:.4f} {sig_marker}")


if __name__ == "__main__":
    main()

