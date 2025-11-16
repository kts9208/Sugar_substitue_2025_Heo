"""
2단계 추정 예제: 선택모델

이 스크립트는 1단계에서 저장한 요인점수를 사용하여 선택모델을 추정합니다.
1단계 결과를 검토한 후 실행하세요.

사용법:
    python examples/sequential_stage2_example.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.choice_model import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.config import MultiLatentConfig


def main():
    print("="*70)
    print("2단계 추정: 선택모델")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "integrated_data.csv"
    
    if not data_path.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행")
    
    # 2. 1단계 결과 확인
    print("\n[2] 1단계 결과 확인 중...")
    stage1_path = project_root / "results" / "stage1_results.pkl"
    
    if not stage1_path.exists():
        print(f"❌ 1단계 결과 파일을 찾을 수 없습니다: {stage1_path}")
        print("먼저 1단계를 실행하세요:")
        print("  python examples/sequential_stage1_example.py")
        return
    
    # 1단계 결과 로드 (요약 정보만)
    stage1_results = SequentialEstimator.load_stage1_results(str(stage1_path))
    print(f"✅ 1단계 결과 로드 완료")
    print(f"   - 요인점수: {list(stage1_results['factor_scores'].keys())}")
    print(f"   - 로그우도: {stage1_results['log_likelihood']:.2f}")
    
    # 3. 설정 생성
    print("\n[3] 모델 설정 중...")
    config = MultiLatentConfig(
        latent_variables={
            'purchase_intention': ['PI1', 'PI2', 'PI3'],
            'perceived_price': ['PP1', 'PP2', 'PP3'],
            'nutrition_knowledge': ['NK1', 'NK2', 'NK3'],
            'health_concern': ['HC1', 'HC2', 'HC3'],
            'perceived_benefit': ['PB1', 'PB2', 'PB3']
        },
        structural_paths={
            'health_concern': [],
            'perceived_benefit': ['health_concern'],
            'nutrition_knowledge': [],
            'perceived_price': ['nutrition_knowledge'],
            'purchase_intention': ['perceived_benefit', 'perceived_price']
        },
        choice_attributes=['price', 'sugar_content', 'brand'],
        choice_column='choice',
        individual_id_column='respondent_id'
    )
    print("✅ 설정 완료")
    
    # 4. 선택모델 생성
    print("\n[4] 선택모델 생성 중...")
    choice_model = MultinomialLogitChoice(
        choice_attributes=config.choice_attributes,
        latent_variable='purchase_intention',  # 주요 잠재변수
        choice_column=config.choice_column,
        individual_id_column=config.individual_id_column
    )
    estimator = SequentialEstimator(config)
    print("✅ 선택모델 생성 완료")
    
    # 5. 2단계 추정 실행
    print("\n[5] 2단계 추정 실행 중...")
    print("    (선택모델 추정)")
    
    stage2_results = estimator.estimate_stage2_only(
        data=data,
        choice_model=choice_model,
        factor_scores=str(stage1_path),  # 파일 경로로 전달
        log_file=str(project_root / "logs" / "stage2_estimation.log")
    )
    
    print("\n✅ 2단계 추정 완료!")
    
    # 6. 결과 확인
    print("\n" + "="*70)
    print("결과 요약")
    print("="*70)
    
    print(f"\n[로그우도] {stage2_results['log_likelihood']:.2f}")
    print(f"[AIC] {stage2_results.get('aic', 'N/A')}")
    print(f"[BIC] {stage2_results.get('bic', 'N/A')}")
    
    print("\n[선택모델 파라미터]")
    for param_name, param_value in stage2_results['params'].items():
        if isinstance(param_value, np.ndarray):
            print(f"  {param_name}: {param_value}")
        else:
            print(f"  {param_name}: {param_value:.4f}")
    
    if 'parameter_statistics' in stage2_results and stage2_results['parameter_statistics'] is not None:
        print("\n[파라미터 통계]")
        print(stage2_results['parameter_statistics'])
    
    # 7. 전체 결과 요약
    print("\n" + "="*70)
    print("전체 추정 완료")
    print("="*70)
    print(f"\n1단계 로그우도: {stage1_results['log_likelihood']:.2f}")
    print(f"2단계 로그우도: {stage2_results['log_likelihood']:.2f}")
    print(f"\n결과 파일:")
    print(f"  📁 1단계: {stage1_path}")
    print(f"  📄 로그: logs/stage1_estimation.log")
    print(f"  📄 로그: logs/stage2_estimation.log")


if __name__ == "__main__":
    main()

