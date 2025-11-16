"""
모든 LV 주효과 모델 테스트 스크립트

5개 잠재변수가 모두 선택모델의 main effect로 추정되는지 확인
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    create_sugar_substitute_multi_lv_config
)

def test_config():
    """설정 확인"""
    print("="*70)
    print("모든 LV 주효과 모델 설정 테스트")
    print("="*70)
    
    # 설정 생성
    config = create_sugar_substitute_multi_lv_config(
        n_draws=100,
        max_iterations=1000,
        all_lvs_as_main=True,  # 모든 LV 주효과
        use_moderation=False   # 조절효과 비활성화
    )
    
    # 선택모델 설정 확인
    print("\n[선택모델 설정]")
    print(f"  all_lvs_as_main: {config.choice.all_lvs_as_main}")
    print(f"  moderation_enabled: {config.choice.moderation_enabled}")
    print(f"  main_lvs: {config.choice.main_lvs}")
    print(f"  n_main_lvs: {config.choice.n_main_lvs}")
    
    # 파라미터 이름 확인
    print("\n[예상 파라미터 이름]")
    print("  선택모델:")
    print("    - intercept")
    print("    - beta_sugar_free")
    print("    - beta_health_label")
    print("    - beta_price")
    
    for lv_name in config.choice.main_lvs:
        print(f"    - lambda_{lv_name}")
    
    # 구조모델 경로 확인
    print("\n[구조모델 경로]")
    for path in config.structural.hierarchical_paths:
        target = path['target']
        predictors = path['predictors']
        for pred in predictors:
            print(f"    - gamma_{pred}_to_{target}")
    
    print("\n✅ 설정 테스트 완료!")
    print("="*70)
    
    return config


def test_parameter_names():
    """파라미터 이름 생성 테스트"""
    from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator
    
    print("\n" + "="*70)
    print("파라미터 이름 생성 테스트")
    print("="*70)
    
    config = create_sugar_substitute_multi_lv_config(
        n_draws=100,
        all_lvs_as_main=True
    )
    
    estimator = SimultaneousEstimator(config)
    
    # 더미 모델 생성
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
    from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
    
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    choice_model = MultinomialLogitChoice(config.choice)
    
    # 파라미터 이름 생성
    param_names = estimator._get_parameter_names(
        measurement_model, structural_model, choice_model
    )
    
    print("\n[생성된 파라미터 이름]")
    lambda_params = [name for name in param_names if name.startswith('lambda_')]
    
    print(f"\n  Lambda 파라미터 ({len(lambda_params)}개):")
    for name in lambda_params:
        print(f"    - {name}")
    
    # 검증
    expected_lambdas = [
        'lambda_health_concern',
        'lambda_perceived_benefit',
        'lambda_perceived_price',
        'lambda_nutrition_knowledge',
        'lambda_purchase_intention'
    ]
    
    print("\n[검증]")
    all_found = all(name in param_names for name in expected_lambdas)
    
    if all_found:
        print("  ✅ 모든 LV lambda 파라미터가 생성되었습니다!")
    else:
        print("  ❌ 일부 lambda 파라미터가 누락되었습니다!")
        missing = [name for name in expected_lambdas if name not in param_names]
        print(f"  누락: {missing}")
    
    print("\n✅ 파라미터 이름 테스트 완료!")
    print("="*70)


if __name__ == '__main__':
    # 설정 테스트
    config = test_config()
    
    # 파라미터 이름 테스트
    test_parameter_names()
    
    print("\n" + "="*70)
    print("모든 테스트 완료!")
    print("="*70)

