"""
효용함수 계산 테스트
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

# 데이터 로드
data_path = project_root / "data" / "processed" / "iclv" / "integrated_data_cleaned.csv"
data = pd.read_csv(data_path)

print("=" * 70)
print("효용함수 계산 테스트")
print("=" * 70)

# 선택모델 설정
config = ChoiceConfig(
    choice_attributes=['health_label', 'price'],
    choice_type='binary',
    price_variable='price',
    all_lvs_as_main=True,
    main_lvs=['purchase_intention', 'nutrition_knowledge'],
    moderation_enabled=False
)

# 선택모델 생성
choice_model = MultinomialLogitChoice(config)

# 추정된 파라미터 (최종 결과에서 가져옴)
params = {
    'beta': np.array([0.633849, -0.660346]),  # [health_label, price]
    'asc_sugar': 1.763071,
    'asc_sugar_free': 2.284814,
    'theta_sugar_purchase_intention': -0.337067,
    'theta_sugar_free_purchase_intention': -0.149795,
    'theta_sugar_nutrition_knowledge': -0.150707,
    'theta_sugar_free_nutrition_knowledge': -0.171752
}
print("\n추정된 파라미터:")
for key, value in params.items():
    print(f"  {key}: {value}")

# 더미 잠재변수 (모두 0)
lv = {
    'purchase_intention': np.zeros(len(data) // 3),
    'nutrition_knowledge': np.zeros(len(data) // 3)
}

# 효용 계산
V = choice_model._compute_utilities(data, lv, params)

print(f"\n효용 계산 결과:")
print(f"  V shape: {V.shape}")
print(f"  V first 30 values:")
for i in range(30):
    sugar = str(data['sugar_content'].iloc[i])
    health = data['health_label'].iloc[i]
    price = data['price'].iloc[i]
    choice = data['choice'].iloc[i]
    print(f"    [{i:2d}] sugar={sugar:10s} health={health:6.1f} price={price:7.3f} V={V[i]:8.4f} choice={choice}")

# 선택된 대안들의 효용 확인
print(f"\n선택된 대안들의 효용:")
chosen = data[data['choice'] == 1].head(20)
for idx in chosen.index[:20]:
    i = idx
    sugar = str(data['sugar_content'].iloc[i])
    health = data['health_label'].iloc[i]
    price = data['price'].iloc[i]
    print(f"    [{i:4d}] sugar={sugar:10s} health={health:6.1f} price={price:7.3f} V={V[i]:8.4f}")

