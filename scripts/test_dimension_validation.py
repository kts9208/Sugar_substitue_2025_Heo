"""
차원 검증 테스트
벡터화된 효용 계산에서 배열 차원이 올바른지 확인
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice, ChoiceConfig


def test_dimension_validation():
    """차원 검증 테스트"""
    
    print("=" * 80)
    print("차원 검증 테스트")
    print("=" * 80)
    
    # 테스트 데이터 생성
    print("\n[1] 테스트 데이터 생성...")
    n_individuals = 100
    n_choice_situations = 8
    n_alternatives = 3
    n_obs = n_individuals * n_choice_situations * n_alternatives
    
    print(f"  - 개인 수: {n_individuals}")
    print(f"  - 선택 상황 수: {n_choice_situations}")
    print(f"  - 대안 수: {n_alternatives}")
    print(f"  - 총 관측치 수: {n_obs}")
    
    # 데이터 생성
    data = pd.DataFrame({
        'price': np.random.uniform(1000, 5000, n_obs),
        'health_label': np.random.choice([0, 1], n_obs),
        'calorie': np.random.uniform(50, 200, n_obs),
        'sugar_content': np.tile(['알반당', '무설탕', np.nan], n_individuals * n_choice_situations),
        'choice': np.random.choice([0, 1], n_obs)
    })
    
    # 잠재변수 (개인 수준)
    lv = {
        'purchase_intention': np.random.randn(n_individuals),
        'nutrition_knowledge': np.random.randn(n_individuals)
    }
    
    # 파라미터
    params = {
        'asc_sugar': 0.5,
        'asc_sugar_free': 0.3,
        'beta': np.array([-0.001, 0.5, -0.01]),  # [price, health_label, calorie]
        'theta_sugar_purchase_intention': 0.8,
        'theta_sugar_free_purchase_intention': 0.6,
        'gamma_sugar_purchase_intention_price': -0.0005,
        'gamma_sugar_free_nutrition_knowledge_health_label': 0.3
    }
    
    # 모델 초기화
    print("\n[2] 모델 초기화...")
    config = ChoiceConfig(
        choice_attributes=['price', 'health_label', 'calorie'],
        price_variable='price',
        main_lvs=['purchase_intention', 'nutrition_knowledge'],
        lv_attribute_interactions=[
            {'lv': 'purchase_intention', 'attribute': 'price'},
            {'lv': 'nutrition_knowledge', 'attribute': 'health_label'}
        ],
        n_alternatives=3
    )
    model = MultinomialLogitChoice(config)
    
    # 효용 계산
    print("\n[3] 효용 계산 및 차원 검증...")
    try:
        V = model._compute_utilities(data, lv, params)
        print("  ✅ 차원 검증 통과!")
        
        # 결과 확인
        print(f"\n[4] 결과 확인...")
        print(f"  - V shape: {V.shape}")
        print(f"  - V ndim: {V.ndim}")
        print(f"  - V dtype: {V.dtype}")
        print(f"  - V 범위: [{V.min():.4f}, {V.max():.4f}]")
        print(f"  - NaN 개수: {np.isnan(V).sum()}")
        print(f"  - Inf 개수: {np.isinf(V).sum()}")
        
        # 차원 검증
        assert V.ndim == 1, f"V should be 1D, got {V.ndim}D"
        assert V.shape[0] == n_obs, f"V length should be {n_obs}, got {V.shape[0]}"
        assert not np.isnan(V).any(), "V contains NaN"
        assert not np.isinf(V).any(), "V contains Inf"
        
        print("\n✅ 모든 차원 검증 통과!")
        
    except AssertionError as e:
        print(f"\n❌ 차원 검증 실패: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        raise


def test_edge_cases():
    """엣지 케이스 테스트"""
    
    print("\n" + "=" * 80)
    print("엣지 케이스 테스트")
    print("=" * 80)
    
    # 케이스 1: 모든 대안이 opt-out
    print("\n[케이스 1] 모든 대안이 opt-out")
    data = pd.DataFrame({
        'price': [np.nan] * 9,
        'health_label': [np.nan] * 9,
        'calorie': [np.nan] * 9,
        'sugar_content': [np.nan] * 9,
        'choice': [0] * 9
    })
    
    lv = {'purchase_intention': np.array([1.0, 1.0, 1.0])}
    params = {'asc_sugar': 0.5, 'beta': np.array([-0.001, 0.5, -0.01])}
    
    config = ChoiceConfig(
        choice_attributes=['price', 'health_label', 'calorie'],
        main_lvs=['purchase_intention'],
        n_alternatives=3
    )
    model = MultinomialLogitChoice(config)
    
    V = model._compute_utilities(data, lv, params)
    assert np.all(V == 0.0), "All opt-out utilities should be 0"
    print("  ✅ 통과: 모든 효용이 0")
    
    print("\n✅ 모든 엣지 케이스 테스트 통과!")


if __name__ == '__main__':
    test_dimension_validation()
    test_edge_cases()

