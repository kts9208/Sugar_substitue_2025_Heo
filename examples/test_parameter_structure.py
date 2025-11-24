"""
파라미터 구조 확인 테스트
"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice

# 데이터 로드
data = pd.read_csv(project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv')

# Stage1 결과 로드
stage1_path = project_root / 'results' / 'final' / 'sequential' / '2path' / 'stage1' / 'stage1_2path_results.pkl'

# 설정 생성 (2path: HC->PB->PI)
config = create_sugar_substitute_multi_lv_config(
    custom_paths=[
        {'source': 'health_concern', 'target': 'perceived_benefit'},
        {'source': 'perceived_benefit', 'target': 'purchase_intention'}
    ],
    include_pp=False,
    include_nk=False
)

# 선택모델 설정 (PI 주효과만)
config.choice = ChoiceConfig(
    choice_attributes=['health_label', 'price'],
    choice_type='multinomial',
    n_alternatives=3,
    main_lvs=['purchase_intention'],
    lv_attribute_interactions=[]
)

# 선택모델 생성
choice_model = MultinomialLogitChoice(config.choice)

# Estimator 생성
estimator = SequentialEstimator(config, standardization_method='zscore')

# 2단계 추정
print("2단계 추정 시작...")
results = estimator.estimate_stage2_only(
    data=data,
    choice_model=choice_model,
    factor_scores=str(stage1_path)
)

# 결과 구조 출력
print("\n" + "="*70)
print("Results 구조")
print("="*70)
print(f"Keys: {list(results.keys())}")

if 'parameter_statistics' in results:
    print("\n" + "="*70)
    print("parameter_statistics 구조")
    print("="*70)
    param_stats = results['parameter_statistics']
    print(f"Type: {type(param_stats)}")
    print(f"Keys: {list(param_stats.keys())}")
    
    for key in param_stats.keys():
        print(f"\n[{key}]")
        print(f"  Type: {type(param_stats[key])}")
        if isinstance(param_stats[key], dict):
            print(f"  Keys: {list(param_stats[key].keys())}")
            # 첫 번째 항목 출력
            first_key = list(param_stats[key].keys())[0] if param_stats[key] else None
            if first_key:
                print(f"  Example ({first_key}): {param_stats[key][first_key]}")
        else:
            print(f"  Value: {param_stats[key]}")

