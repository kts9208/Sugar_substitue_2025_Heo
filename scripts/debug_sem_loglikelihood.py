"""
SEM 로그우도 디버깅 스크립트

semopy 모델의 속성을 확인하여 로그우도를 추출하는 방법을 찾습니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.analysis.hybrid_choice_model.iclv_models.sem_estimator import SEMEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_default_multi_lv_config

print("="*70)
print("SEM 로그우도 디버깅")
print("="*70)

# 데이터 로드
print("\n데이터 로드 중...")
data = pd.read_csv(project_root / 'data/processed/iclv/integrated_data.csv')
print(f"   데이터 shape: {data.shape}")

# 설정 생성
print("\n설정 생성 중...")
config = create_default_multi_lv_config()

# 모델 생성
print("\n모델 생성 중...")
measurement_model = MultiLatentMeasurement(config.measurement_configs)
structural_model = MultiLatentStructural(config.structural)

# SEM 추정
print("\nSEM 추정 중...")
estimator = SEMEstimator()
results = estimator.fit(data, measurement_model, structural_model)

print("\n" + "="*70)
print("semopy 모델 속성 확인")
print("="*70)

# 모든 속성 출력
print("\n[모든 속성]")
all_attrs = [attr for attr in dir(estimator.model) if not attr.startswith('_')]
for attr in all_attrs:
    print(f"  - {attr}")

# 로그우도 관련 속성 확인
print("\n[로그우도 관련 속성 확인]")
for attr in ['obj', 'loglike', 'objective', 'last_result', 'fun', 'fmin']:
    if hasattr(estimator.model, attr):
        value = getattr(estimator.model, attr)
        print(f"  ✅ {attr}: {value}")
    else:
        print(f"  ❌ {attr}: 없음")

# last_result 상세 확인
if hasattr(estimator.model, 'last_result'):
    print("\n[last_result 상세]")
    last_result = estimator.model.last_result
    print(f"  타입: {type(last_result)}")
    if hasattr(last_result, '__dict__'):
        for key, value in last_result.__dict__.items():
            print(f"  - {key}: {value}")

# inspect() 결과 확인
print("\n[inspect() 결과]")
params = estimator.model.inspect()
print(f"  컬럼: {list(params.columns)}")
print(f"  행 수: {len(params)}")

print("\n" + "="*70)
print("결과")
print("="*70)
print(f"\n로그우도: {results['log_likelihood']}")
print(f"적합도 지수: {results['fit_indices']}")

