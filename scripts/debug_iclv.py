"""
ICLV 디버깅 스크립트 - 로그우도 계산 확인
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig
)


def main():
    print("="*70)
    print("ICLV 디버깅 - 로그우도 계산 확인")
    print("="*70)
    
    # 데이터 로드
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    
    # 첫 번째 개인만 추출
    first_id = data['respondent_id'].unique()[0]
    person_data = data[data['respondent_id'] == first_id]
    
    print(f"\n개인 ID: {first_id}")
    print(f"선택 상황 수: {len(person_data)}")
    print(f"\n데이터 샘플:")
    print(person_data[['choice', 'price', 'health_label', 'q6', 'q7', 'q8']].head())
    
    # 설정
    measurement_config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        n_categories=5
    )
    
    structural_config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std'],
        include_in_choice=False
    )
    
    choice_config = ChoiceConfig(
        choice_attributes=['price', 'health_label']
    )
    
    # 모델 생성
    measurement_model = OrderedProbitMeasurement(measurement_config)
    structural_model = LatentVariableRegression(structural_config)
    choice_model = BinaryProbitChoice(choice_config)
    
    # 초기 파라미터
    measurement_params = {
        'zeta': np.array([1.0] * 6),
        'tau': np.array([[-2, -1, 1, 2]] * 6)
    }
    
    structural_params = {
        'gamma': np.array([0.0, 0.0, 0.0])
    }
    
    choice_params = {
        'intercept': 0.0,
        'beta': np.array([0.0, 0.0]),
        'lambda': 1.0
    }
    
    # 잠재변수 예측
    draw = 0.0  # 표준정규분포 draw
    lv = structural_model.predict(person_data, structural_params, draw)
    
    print(f"\n잠재변수 (LV): {lv}")
    print(f"LV 타입: {type(lv)}")
    
    # 측정모델 로그우도
    ll_measurement = measurement_model.log_likelihood(
        person_data, lv, measurement_params
    )
    print(f"\n측정모델 로그우도: {ll_measurement:.4f}")
    
    # 구조모델 로그우도
    ll_structural = structural_model.log_likelihood(
        person_data, lv, structural_params, draw
    )
    print(f"구조모델 로그우도: {ll_structural:.4f}")
    
    # 선택모델 로그우도 (각 선택 상황)
    print(f"\n선택모델 로그우도 (각 선택 상황):")
    choice_lls = []
    for idx in range(len(person_data)):
        ll_choice = choice_model.log_likelihood(
            person_data.iloc[idx:idx+1],
            lv,
            choice_params
        )
        choice_lls.append(ll_choice)
        print(f"  선택 {idx+1}: {ll_choice:.4f}")
    
    total_choice_ll = sum(choice_lls)
    print(f"  합계: {total_choice_ll:.4f}")
    
    # 결합 로그우도
    total_ll = ll_measurement + ll_structural + total_choice_ll
    print(f"\n결합 로그우도: {total_ll:.4f}")
    
    # exp 변환
    likelihood = np.exp(total_ll)
    print(f"우도 (exp): {likelihood:.6e}")
    
    if likelihood == 0:
        print("\n⚠️ 문제: 우도가 0입니다!")
        print("   원인: 로그우도가 너무 작아서 exp() 시 underflow 발생")
        print(f"   로그우도: {total_ll}")
        print(f"   exp({total_ll}) = 0 (underflow)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

