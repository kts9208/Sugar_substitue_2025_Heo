"""
모든 잠재변수 간 경로 추정 예제 (20개 경로)

각 잠재변수를 종속변수로 하는 5개 모델을 순차적으로 추정하여
5×4 = 20개의 방향성 경로를 모두 확인합니다.

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
    print("모든 잠재변수 간 경로 추정 (20개)")
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
    
    # 4. 모든 경로 추정
    print("\n[4] 모든 경로 추정 실행 중...")
    print("    (각 잠재변수를 종속변수로 하는 5개 모델 추정)")
    
    save_path = project_root / "results" / "sequential_stage_wise" / "all_paths_results.pkl"
    
    results = estimator.estimate_all_paths(
        data=data,
        measurement_model=measurement_model,
        save_path=str(save_path)
    )
    
    print("\n✅ 모든 경로 추정 완료!")
    
    # 5. 결과 출력
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    
    all_paths = results['all_paths']
    summary = results['summary']
    
    print(f"\n[총 경로 수] {len(all_paths)}개")
    print(f"[유의한 경로 (p<0.05)] {(all_paths['p-value'] < 0.05).sum()}개")
    print(f"[매우 유의한 경로 (p<0.01)] {(all_paths['p-value'] < 0.01).sum()}개")
    
    print("\n[종속변수별 요약]")
    print(summary.to_string(index=False))
    
    print("\n[모든 경로 (유의도 순)]")
    all_paths_sorted = all_paths.sort_values('p-value')
    for _, row in all_paths_sorted.iterrows():
        sig = "***" if row['p-value'] < 0.001 else "**" if row['p-value'] < 0.01 else "*" if row['p-value'] < 0.05 else ""
        print(f"  {row['predictor']:20s} → {row['target']:20s}: {row['Estimate']:7.4f} (p={row['p-value']:.4f}) {sig}")
    
    print("\n" + "=" * 70)
    print("다음 단계")
    print("=" * 70)
    
    print(f"\n결과가 저장되었습니다:")
    print(f"  📁 {save_path.parent / 'all_paths_results_all_20_paths.csv'}")
    print(f"  📁 {save_path.parent / 'all_paths_results_summary.csv'}")
    
    print("\n유의한 경로만 포함하여 1단계 추정을 진행하거나,")
    print("현재 결과를 바탕으로 2단계 선택모델을 추정할 수 있습니다.")


if __name__ == "__main__":
    main()

