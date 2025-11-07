"""
NaN을 반환하는 개인들 디버깅
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
    # 데이터 로드
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    
    # NaN을 반환하는 개인들
    nan_ids = [14, 16, 23, 29, 36]
    
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
    
    for ind_id in nan_ids:
        print("="*70)
        print(f"개인 ID: {ind_id}")
        print("="*70)
        
        person_data = data[data['respondent_id'] == ind_id]
        
        print(f"\n데이터 shape: {person_data.shape}")
        print(f"\n지표 데이터:")
        print(person_data[['q6', 'q7', 'q8', 'q9', 'q10', 'q11']].head())
        
        print(f"\n사회인구학적 변수:")
        print(person_data[['age_std', 'gender', 'income_std']].head())
        
        print(f"\n선택 데이터:")
        print(person_data[['choice', 'price', 'health_label']].head(10))
        
        # 로그우도 계산
        draw = 0.0
        lv = structural_model.predict(person_data, structural_params, draw)
        
        print(f"\n잠재변수 (LV): {lv}")
        
        ll_measurement = measurement_model.log_likelihood(
            person_data, lv, measurement_params
        )
        print(f"측정모델 로그우도: {ll_measurement}")
        
        ll_structural = structural_model.log_likelihood(
            person_data, lv, structural_params, draw
        )
        print(f"구조모델 로그우도: {ll_structural}")
        
        print(f"\n선택모델 로그우도 (각 선택 상황):")
        choice_lls = []
        for idx in range(len(person_data)):
            ll_choice = choice_model.log_likelihood(
                person_data.iloc[idx:idx+1],
                lv,
                choice_params
            )
            choice_lls.append(ll_choice)
            if idx < 10:
                print(f"  선택 {idx+1}: {ll_choice:.4f}")
        
        ll_choice_total = sum(choice_lls)
        print(f"  합계: {ll_choice_total}")
        
        person_ll = ll_measurement + ll_structural + ll_choice_total
        print(f"\n결합 로그우도: {person_ll}")
        print()


if __name__ == '__main__':
    main()

