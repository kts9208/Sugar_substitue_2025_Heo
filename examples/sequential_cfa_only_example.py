"""
CFA 전용 추정 예제: 잠재변수 간 상관관계 확인

구조모델 없이 측정모델(CFA)만 추정하여 5개 잠재변수 간 
10개 상관관계(5C2)를 모두 확인합니다.

이를 통해 어떤 잠재변수 간 관계가 유의한지 확인하고,
이후 구조모델 설정에 활용할 수 있습니다.

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
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config


def main():
    print("=" * 70)
    print("CFA 전용 추정: 잠재변수 간 상관관계 확인")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data_cleaned.csv"
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")
    
    # 2. 설정 생성
    print("\n[2] 모델 설정 중...")
    config = create_sugar_substitute_multi_lv_config()
    print("✅ 설정 완료")
    print(f"   - 잠재변수: {list(config.measurement_configs.keys())}")
    
    # 3. 모델 생성
    print("\n[3] 모델 생성 중...")
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    estimator = SequentialEstimator(config)
    print("✅ 모델 생성 완료")
    
    # 4. CFA 추정
    print("\n[4] CFA 추정 실행 중...")
    print("    (측정모델만 추정, 구조모델 없음)")
    
    save_path = project_root / "results" / "sequential_stage_wise" / "cfa_results.pkl"
    
    results = estimator.estimate_cfa_only(
        data=data,
        measurement_model=measurement_model,
        save_path=str(save_path)
    )
    
    print("\n✅ CFA 추정 완료!")
    
    # 5. 결과 출력
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
    
    # 상관관계
    corr = results['correlations']
    print(f"\n[잠재변수 간 상관관계] {len(corr)}개")
    print("\n" + "-" * 70)
    print(f"{'LV 1':20s} {'LV 2':20s} {'상관계수':>10s} {'p-value':>10s} {'유의성':>8s}")
    print("-" * 70)
    
    # 유의도 순으로 정렬
    corr_sorted = corr.sort_values('p-value')
    for _, row in corr_sorted.iterrows():
        sig = "***" if row['p-value'] < 0.001 else "**" if row['p-value'] < 0.01 else "*" if row['p-value'] < 0.05 else ""
        print(f"{row['lval']:20s} {row['rval']:20s} {row['Est. Std']:10.4f} {row['p-value']:10.4f} {sig:>8s}")
    
    print("-" * 70)
    
    # 유의한 상관관계 개수
    n_sig = (corr['p-value'] < 0.05).sum()
    n_very_sig = (corr['p-value'] < 0.01).sum()
    print(f"\n유의한 상관관계 (p<0.05): {n_sig}/{len(corr)}개")
    print(f"매우 유의한 상관관계 (p<0.01): {n_very_sig}/{len(corr)}개")
    
    # 요인적재량
    loadings = results['loadings']
    print(f"\n[요인적재량] {len(loadings)}개")
    print(loadings.to_string(index=False))
    
    # 상관관계 행렬 출력
    print("\n" + "=" * 70)
    print("상관관계 행렬 (Correlation Matrix)")
    print("=" * 70)

    # 행렬 생성
    corr_matrix_path = save_path.parent / 'cfa_results_correlation_matrix.csv'
    if corr_matrix_path.exists():
        corr_matrix = pd.read_csv(corr_matrix_path, index_col=0)
        print("\n" + corr_matrix.to_string())

        # p-value 행렬
        print("\n" + "=" * 70)
        print("p-value 행렬")
        print("=" * 70)
        pvalue_matrix_path = save_path.parent / 'cfa_results_pvalue_matrix.csv'
        if pvalue_matrix_path.exists():
            pvalue_matrix = pd.read_csv(pvalue_matrix_path, index_col=0)
            print("\n" + pvalue_matrix.to_string())

    print("\n" + "=" * 70)
    print("다음 단계")
    print("=" * 70)

    print(f"\n결과가 저장되었습니다:")
    print(f"  📁 {save_path.parent / 'cfa_results_correlations.csv'}")
    print(f"  📁 {save_path.parent / 'cfa_results_correlation_matrix.csv'} ⭐")
    print(f"  📁 {save_path.parent / 'cfa_results_pvalue_matrix.csv'} ⭐")
    print(f"  📁 {save_path.parent / 'cfa_results_loadings.csv'}")
    print(f"  📁 {save_path.parent / 'cfa_results_fit_indices.csv'}")
    print(f"  📁 {save_path.parent / 'cfa_results_factor_scores.csv'}")

    print("\n유의한 상관관계를 바탕으로 구조모델을 설정하여")
    print("1단계 추정(SEM)을 진행할 수 있습니다.")


if __name__ == "__main__":
    main()

