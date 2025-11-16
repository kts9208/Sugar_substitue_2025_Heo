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
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import MultiLatentConfig


def main():
    print("="*70)
    print("2단계 추정: 선택모델")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"

    if not data_path.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return

    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")
    
    # 2. 1단계 결과 확인
    print("\n[2] 1단계 결과 확인 중...")
    stage1_path = project_root / "results" / "sequential_stage_wise" / "stage1_results.pkl"

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

    # 선택모델 설정
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='multinomial',
        all_lvs_as_main=True,  # 모든 잠재변수를 주효과로 사용
        main_lvs=['health_concern', 'perceived_benefit', 'perceived_price',
                  'nutrition_knowledge', 'purchase_intention']
    )
    print("✅ 설정 완료")

    # 4. 선택모델 생성
    print("\n[4] 선택모델 생성 중...")
    choice_model = MultinomialLogitChoice(choice_config)

    # MultiLatentConfig는 SequentialEstimator 생성용 (간단한 더미 설정)
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement, MeasurementConfig
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural

    # 더미 측정모델 설정
    measurement_configs = {
        'health_concern': MeasurementConfig(indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'], n_categories=5),
        'perceived_benefit': MeasurementConfig(indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'], n_categories=5),
        'perceived_price': MeasurementConfig(indicators=['q27', 'q28', 'q29'], n_categories=5),
        'nutrition_knowledge': MeasurementConfig(indicators=['q30', 'q31', 'q32', 'q33', 'q34', 'q35', 'q36', 'q37', 'q38', 'q39', 'q40', 'q41', 'q42', 'q43', 'q44', 'q45', 'q46', 'q47', 'q48', 'q49'], n_categories=5),
        'purchase_intention': MeasurementConfig(indicators=['q18', 'q19', 'q20'], n_categories=5)
    }
    measurement_model = MultiLatentMeasurement(measurement_configs)

    # 더미 구조모델 설정
    structural_model = MultiLatentStructural(
        hierarchical_paths=[
            {'target': 'perceived_benefit', 'predictors': ['health_concern', 'perceived_price', 'nutrition_knowledge']},
            {'target': 'purchase_intention', 'predictors': ['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge']}
        ]
    )

    config = MultiLatentConfig(
        measurement_configs=measurement_configs,
        structural=structural_model,
        choice=choice_config,
        estimation={'method': 'sequential'}
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

