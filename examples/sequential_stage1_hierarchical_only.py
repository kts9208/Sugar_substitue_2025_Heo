"""
1단계 추정 (순수 계층적 구조 - 2경로만)

경로 구조:
1. HC → PB (건강관심도 → 건강유익성)
2. PB → PI (건강유익성 → 구매의도)

총 2개 경로만 추정

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


def main():
    print("=" * 70)
    print("1단계 추정: 순수 계층적 구조 (2경로)")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data_cleaned.csv"
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")
    
    # 2. 순수 계층적 경로 설정 (2개만)
    print("\n[2] 경로 설정 중...")
    hierarchical_paths = [
        # HC → PB
        {
            'target': 'perceived_benefit',
            'predictors': ['health_concern']
        },
        # PB → PI
        {
            'target': 'purchase_intention',
            'predictors': ['perceived_benefit']
        }
    ]
    
    print("✅ 경로 설정 완료:")
    print("   1. health_concern → perceived_benefit")
    print("   2. perceived_benefit → purchase_intention")
    
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
    print("    (측정모델 + 구조모델)")
    
    save_path = project_root / "results" / "sequential_stage_wise" / "stage1_hierarchical_2paths_results.pkl"
    
    results = estimator.estimate_stage1_only(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        save_path=str(save_path)
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
        print(f"  {key}: {value:.4f}")
    
    # 경로계수
    paths = results['paths']
    print(f"\n[잠재변수 간 경로계수] {len(paths)}개")
    print("\n" + "-" * 70)
    print(f"{'종속변수':20s} {'←':3s} {'예측변수':20s} {'계수':>10s} {'표준오차':>10s} {'p-value':>10s} {'유의성':>8s}")
    print("-" * 70)
    
    for _, row in paths.iterrows():
        sig = "***" if row['p-value'] < 0.001 else "**" if row['p-value'] < 0.01 else "*" if row['p-value'] < 0.05 else ""
        print(f"{row['lval']:20s} {'←':3s} {row['rval']:20s} {row['Estimate']:10.4f} {row['Std. Err']:10.4f} {row['p-value']:10.4f} {sig:>8s}")
    
    print("-" * 70)
    
    # 유의한 경로 개수
    n_sig = (paths['p-value'] < 0.05).sum()
    print(f"\n유의한 경로 (p<0.05): {n_sig}/{len(paths)}개")
    
    print("\n" + "=" * 70)
    print("저장된 파일")
    print("=" * 70)
    
    print(f"\n  📁 {save_path.parent / 'stage1_hierarchical_2paths_results_paths.csv'}")
    print(f"  📁 {save_path.parent / 'stage1_hierarchical_2paths_results_fit_indices.csv'}")


if __name__ == "__main__":
    main()

