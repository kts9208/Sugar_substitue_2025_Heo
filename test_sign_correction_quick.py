"""
빠른 Sign Correction 테스트 (10 샘플만)
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# 설정 로드
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config

# 경로 설정
PATHS = {
    'HC->PB': True,
    'HC->PP': False,
    'HC->PI': False,
    'PB->PI': True,
    'PP->PI': False,
    'NK->PI': False,
}

MAIN_LVS = ['nutrition_knowledge', 'purchase_intention', 'perceived_price']
LV_ATTRIBUTE_INTERACTIONS = []

# 데이터 로드
data = pd.read_csv('data/processed/choice_data_with_latent_indicators.csv')

# 모델 설정
hierarchical_paths = [
    {'from': 'health_concern', 'to': 'perceived_benefit'},
    {'from': 'perceived_benefit', 'to': 'purchase_intention'}
]
config = create_sugar_substitute_multi_lv_config(hierarchical_paths=hierarchical_paths)

# 부트스트랩 (10 샘플만)
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_both_stages
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

choice_config = ChoiceConfig(
    choice_var='choice',
    alternative_var='alternative',
    choice_attributes=['sugar_free', 'health_label', 'price'],
    choice_type='binary',
    main_lvs=MAIN_LVS,
    lv_attribute_interactions=LV_ATTRIBUTE_INTERACTIONS
)

print("=" * 80)
print("빠른 Sign Correction 테스트 (10 샘플)")
print("=" * 80)

results = bootstrap_both_stages(
    data=data,
    measurement_model=config.measurement_configs,
    structural_model=config.structural,
    choice_model=choice_config,
    n_bootstrap=10,
    n_workers=2,
    confidence_level=0.95,
    random_seed=42,
    show_progress=True
)

# Sign Flip 통계 확인
if 'sign_flip_statistics' in results:
    print("\n" + "=" * 80)
    print("Sign Flip 통계")
    print("=" * 80)
    print(results['sign_flip_statistics'])
    
    # flip_rate 확인
    flip_stats = results['sign_flip_statistics']
    print("\n각 잠재변수의 부호 반전 비율:")
    for _, row in flip_stats.iterrows():
        print(f"  {row['lv_name']:30s}: {row['n_flips']:3d}/10 ({row['flip_rate']:.1%})")
else:
    print("\n⚠️ sign_flip_statistics가 없습니다!")

# 첫 3개 샘플의 flip_status 확인
print("\n" + "=" * 80)
print("첫 3개 샘플의 Sign Flip Status")
print("=" * 80)
for i in range(min(3, len(results['bootstrap_estimates']))):
    sample = results['bootstrap_estimates'][i]
    if 'sign_flip_status' in sample:
        print(f"\n샘플 {i}:")
        for lv_name, flipped in sample['sign_flip_status'].items():
            status = "✅ 반전" if flipped else "  유지"
            print(f"  {lv_name:30s}: {status}")
    else:
        print(f"\n샘플 {i}: sign_flip_status 없음")

print("\n" + "=" * 80)
print("테스트 완료!")
print("=" * 80)

