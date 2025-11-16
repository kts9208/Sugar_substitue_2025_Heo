"""
1단계 추정 예제: 측정모델 + 구조모델 (SEM)

이 스크립트는 순차추정의 1단계만 실행하여 잠재변수 간 관계를 확인합니다.
결과를 파일로 저장하여 나중에 2단계에서 재사용할 수 있습니다.

사용법:
    python examples/sequential_stage1_example.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config


def main():
    print("="*70)
    print("1단계 추정: 측정모델 + 구조모델 (SEM)")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data_cleaned.csv"

    if not data_path.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("먼저 데이터 전처리를 실행하세요.")
        return

    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")
    
    # 2. 설정 생성
    print("\n[2] 모델 설정 중...")
    # use_full_paths=True: 모든 잠재변수 간 경로 추정
    config = create_sugar_substitute_multi_lv_config(use_full_paths=True)
    print("✅ 설정 완료")
    print(f"   - 잠재변수: {list(config.measurement_configs.keys())}")
    print(f"   - 구조모델: 모든 경로 추정 (완전 연결)")
    print(f"     * perceived_benefit <- health_concern, perceived_price, nutrition_knowledge")
    print(f"     * purchase_intention <- health_concern, perceived_benefit, perceived_price, nutrition_knowledge")

    # 3. 모델 생성
    print("\n[3] 모델 생성 중...")
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    estimator = SequentialEstimator(config)
    print("✅ 모델 생성 완료")
    
    # 4. 1단계 추정 실행
    print("\n[4] 1단계 추정 실행 중...")
    print("    (측정모델 + 구조모델 통합 추정)")

    results_dir = project_root / "results" / "sequential_stage_wise"
    results_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    stage1_results = estimator.estimate_stage1_only(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        save_path=str(results_dir / "stage1_results.pkl"),
        log_file=str(logs_dir / "stage1_estimation.log")
    )
    
    print("\n✅ 1단계 추정 완료!")
    
    # 5. 결과 확인
    print("\n" + "="*70)
    print("결과 요약")
    print("="*70)
    
    print(f"\n[로그우도] {stage1_results['log_likelihood']:.2f}")
    
    print("\n[적합도 지수]")
    for key, value in stage1_results['fit_indices'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n[잠재변수 간 경로계수]")
    print(stage1_results['paths'])
    
    print("\n[요인적재량]")
    print(stage1_results['loadings'])
    
    print("\n[요인점수 통계]")
    for lv_name, scores in stage1_results['factor_scores'].items():
        print(f"  {lv_name}:")
        print(f"    Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        print(f"    Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    # 6. 저장 경로 안내
    print("\n" + "="*70)
    print("다음 단계")
    print("="*70)
    print(f"\n1단계 결과가 저장되었습니다:")
    print(f"  📁 {stage1_results['save_path']}")
    print(f"\n2단계를 실행하려면:")
    print(f"  python examples/sequential_stage2_example.py")


if __name__ == "__main__":
    main()

