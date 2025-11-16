"""
LV-Attribute 상호작용 기능 테스트

PI × price, PI × health_label, NK × health_label 상호작용항이
올바르게 효용함수에 추가되는지 확인합니다.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig


def test_lv_attribute_interaction_config():
    """ChoiceConfig에 LV-Attribute 상호작용 설정이 올바르게 저장되는지 테스트"""
    print("=" * 70)
    print("테스트 1: ChoiceConfig 설정")
    print("=" * 70)
    
    config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='multinomial',
        all_lvs_as_main=True,
        main_lvs=['purchase_intention', 'nutrition_knowledge'],
        lv_attribute_interactions=[
            {'lv': 'purchase_intention', 'attribute': 'price'},
            {'lv': 'purchase_intention', 'attribute': 'health_label'},
            {'lv': 'nutrition_knowledge', 'attribute': 'health_label'}
        ]
    )
    
    assert config.lv_attribute_interactions is not None
    assert len(config.lv_attribute_interactions) == 3
    assert config.n_lv_attr_interactions == 3
    
    print("✅ ChoiceConfig 설정 성공")
    print(f"   - 주효과 LV: {config.main_lvs}")
    print(f"   - 상호작용 개수: {config.n_lv_attr_interactions}")
    print(f"   - 상호작용 항목: {config.lv_attribute_interactions}")
    print()


def test_choice_model_initialization():
    """선택모델이 LV-Attribute 상호작용 설정을 올바르게 초기화하는지 테스트"""
    print("=" * 70)
    print("테스트 2: 선택모델 초기화")
    print("=" * 70)
    
    config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='multinomial',
        all_lvs_as_main=True,
        main_lvs=['purchase_intention', 'nutrition_knowledge'],
        lv_attribute_interactions=[
            {'lv': 'purchase_intention', 'attribute': 'price'},
            {'lv': 'purchase_intention', 'attribute': 'health_label'},
            {'lv': 'nutrition_knowledge', 'attribute': 'health_label'}
        ]
    )
    
    model = MultinomialLogitChoice(config)
    
    assert model.lv_attribute_interactions is not None
    assert len(model.lv_attribute_interactions) == 3
    
    print("✅ 선택모델 초기화 성공")
    print(f"   - 모델 타입: {model.__class__.__name__}")
    print(f"   - 상호작용 설정: {model.lv_attribute_interactions}")
    print()


def test_utility_computation_with_interactions():
    """효용함수가 LV-Attribute 상호작용을 올바르게 계산하는지 테스트"""
    print("=" * 70)
    print("테스트 3: 효용함수 계산")
    print("=" * 70)
    
    # 설정
    config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='multinomial',
        all_lvs_as_main=True,
        main_lvs=['purchase_intention', 'nutrition_knowledge'],
        lv_attribute_interactions=[
            {'lv': 'purchase_intention', 'attribute': 'price'},
            {'lv': 'purchase_intention', 'attribute': 'health_label'},
            {'lv': 'nutrition_knowledge', 'attribute': 'health_label'}
        ]
    )
    
    model = MultinomialLogitChoice(config)
    
    # 테스트 데이터 (3개 대안 × 2개 선택상황 = 6행)
    data = pd.DataFrame({
        'sugar_free': [1, 0, np.nan, 1, 0, np.nan],
        'health_label': [1, 1, np.nan, 0, 1, np.nan],
        'price': [1000, 1200, np.nan, 1100, 1300, np.nan],
        'choice': [1, 0, 0, 0, 1, 0]
    })
    
    # 잠재변수 값 (개인별)
    lv_dict = {
        'purchase_intention': np.array([0.5, 0.5, 0.5, -0.3, -0.3, -0.3]),
        'nutrition_knowledge': np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2])
    }
    
    # 파라미터
    params = {
        'beta': np.array([-0.5, 0.3, -0.001]),  # sugar_free, health_label, price
        'intercept': 0.0,
        'lambda_purchase_intention': 1.0,
        'lambda_nutrition_knowledge': 0.5,
        # 상호작용 계수
        'gamma_purchase_intention_price': -0.0002,  # PI × price (음수: 가격 민감도 증가)
        'gamma_purchase_intention_health_label': 0.4,  # PI × health_label (양수: 건강라벨 선호 증가)
        'gamma_nutrition_knowledge_health_label': 0.3  # NK × health_label (양수: 지식 높을수록 라벨 선호)
    }
    
    # 효용 계산
    V = model._compute_utilities(data, lv_dict, params)
    
    print("✅ 효용함수 계산 성공")
    print(f"   - 계산된 효용: {V}")
    print(f"   - 효용 shape: {V.shape}")
    print(f"   - opt-out 효용 (인덱스 2, 5): {V[2]}, {V[5]} (모두 0이어야 함)")
    
    # opt-out 대안의 효용이 0인지 확인
    assert V[2] == 0.0, "opt-out 대안의 효용이 0이 아닙니다"
    assert V[5] == 0.0, "opt-out 대안의 효용이 0이 아닙니다"
    
    # 상호작용 효과 확인 (수동 계산)
    # 첫 번째 대안 (인덱스 0): sugar_free=1, health_label=1, price=1000
    # V[0] = intercept + beta @ X + lambda_PI * PI + lambda_NK * NK 
    #        + gamma_PI_price * PI * price + gamma_PI_health_label * PI * health_label 
    #        + gamma_NK_health_label * NK * health_label
    expected_V0 = (0.0 + 
                   (-0.5 * 1 + 0.3 * 1 + -0.001 * 1000) +  # beta @ X
                   1.0 * 0.5 +  # lambda_PI * PI
                   0.5 * 0.8 +  # lambda_NK * NK
                   -0.0002 * 0.5 * 1000 +  # gamma_PI_price * PI * price
                   0.4 * 0.5 * 1 +  # gamma_PI_health_label * PI * health_label
                   0.3 * 0.8 * 1)  # gamma_NK_health_label * NK * health_label
    
    print(f"\n   - 수동 계산 V[0]: {expected_V0:.6f}")
    print(f"   - 모델 계산 V[0]: {V[0]:.6f}")
    print(f"   - 차이: {abs(V[0] - expected_V0):.10f}")
    
    assert np.isclose(V[0], expected_V0, atol=1e-6), f"효용 계산 오류: {V[0]} != {expected_V0}"
    
    print("\n✅ 모든 테스트 통과!")
    print()


if __name__ == "__main__":
    test_lv_attribute_interaction_config()
    test_choice_model_initialization()
    test_utility_computation_with_interactions()
    
    print("=" * 70)
    print("전체 테스트 완료")
    print("=" * 70)

