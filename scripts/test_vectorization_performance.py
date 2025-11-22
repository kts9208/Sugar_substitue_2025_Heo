"""
벡터화 성능 테스트

효용 계산의 for 루프 vs 벡터화 성능 비교
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

def generate_test_data(n_individuals=1000, n_choice_situations=8, n_alternatives=3):
    """테스트 데이터 생성"""
    n_obs = n_individuals * n_choice_situations * n_alternatives
    
    data = pd.DataFrame({
        'individual_id': np.repeat(np.arange(n_individuals), n_choice_situations * n_alternatives),
        'choice_situation': np.tile(np.repeat(np.arange(n_choice_situations), n_alternatives), n_individuals),
        'sugar_content': np.tile(['알반당', '무설탕', np.nan], n_individuals * n_choice_situations),
        'price': np.random.uniform(1000, 5000, n_obs),
        'health_label': np.random.choice([0, 1], n_obs),
        'calorie': np.random.uniform(50, 200, n_obs),
        'choice': np.tile([0, 0, 0], n_individuals * n_choice_situations)
    })
    
    # 각 선택 상황에서 하나씩 선택
    for i in range(n_individuals):
        for j in range(n_choice_situations):
            idx_start = (i * n_choice_situations + j) * n_alternatives
            chosen_alt = np.random.choice(n_alternatives)
            data.loc[idx_start + chosen_alt, 'choice'] = 1
    
    return data

def test_performance():
    """성능 테스트"""
    print("=" * 80)
    print("벡터화 성능 테스트")
    print("=" * 80)
    
    # 테스트 데이터 생성
    print("\n[1] 테스트 데이터 생성...")
    data = generate_test_data(n_individuals=1000, n_choice_situations=8, n_alternatives=3)
    print(f"  - 개인 수: 1000")
    print(f"  - 선택 상황 수: 8")
    print(f"  - 대안 수: 3")
    print(f"  - 총 관측치 수: {len(data)}")
    
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
    
    # 파라미터 설정
    params = {
        'asc_sugar': 0.5,
        'asc_sugar_free': 0.3,
        'beta': np.array([-0.001, 0.5, -0.01]),  # price, health_label, calorie
        'theta_sugar_purchase_intention': 1.2,
        'theta_sugar_nutrition_knowledge': 0.8,
        'theta_sugar_free_purchase_intention': 1.5,
        'theta_sugar_free_nutrition_knowledge': 1.0,
        'gamma_sugar_purchase_intention_price': -0.0002,
        'gamma_sugar_nutrition_knowledge_health_label': 0.3,
        'gamma_sugar_free_purchase_intention_price': -0.0003,
        'gamma_sugar_free_nutrition_knowledge_health_label': 0.4
    }
    
    # 잠재변수 값 생성
    n_individuals = 1000
    lv = {
        'purchase_intention': np.random.randn(n_individuals),
        'nutrition_knowledge': np.random.randn(n_individuals)
    }
    
    # 성능 측정
    print("\n[3] 효용 계산 성능 측정...")
    n_iterations = 100
    
    start_time = time.time()
    for _ in range(n_iterations):
        V = model._compute_utilities(data, lv, params)
    elapsed_time = time.time() - start_time
    
    avg_time = elapsed_time / n_iterations * 1000  # ms
    
    print(f"\n{'='*80}")
    print(f"결과:")
    print(f"{'='*80}")
    print(f"  - 반복 횟수: {n_iterations}")
    print(f"  - 총 소요 시간: {elapsed_time:.4f}초")
    print(f"  - 평균 소요 시간: {avg_time:.4f}ms")
    print(f"  - 효용 벡터 shape: {V.shape}")
    print(f"  - 효용 범위: [{V.min():.4f}, {V.max():.4f}]")
    print(f"{'='*80}")
    
    # 결과 검증
    print("\n[4] 결과 검증...")
    print(f"  - NaN 개수: {np.isnan(V).sum()}")
    print(f"  - Inf 개수: {np.isinf(V).sum()}")
    opt_out_utilities = V[data['sugar_content'].isna()]
    print(f"  - opt-out 효용 (0이어야 함): {np.unique(opt_out_utilities)}")
    
    print("\n✅ 테스트 완료!")

if __name__ == '__main__':
    test_performance()

