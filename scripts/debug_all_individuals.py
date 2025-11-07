"""
모든 개인의 로그우도 확인
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.special import logsumexp

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
    
    # 30명 추출
    test_ids = data['respondent_id'].unique()[:30]
    test_data = data[data['respondent_id'].isin(test_ids)]
    
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
    
    # 각 개인에 대해 로그우도 계산
    print("개인별 로그우도:")
    print("="*70)
    
    total_ll = 0.0
    problem_count = 0
    
    for ind_id in test_ids:
        person_data = test_data[test_data['respondent_id'] == ind_id]
        
        # 1개 draw만 사용 (간단히)
        draw = 0.0
        lv = structural_model.predict(person_data, structural_params, draw)
        
        ll_measurement = measurement_model.log_likelihood(
            person_data, lv, measurement_params
        )
        
        ll_structural = structural_model.log_likelihood(
            person_data, lv, structural_params, draw
        )
        
        choice_lls = []
        for idx in range(len(person_data)):
            ll_choice = choice_model.log_likelihood(
                person_data.iloc[idx:idx+1],
                lv,
                choice_params
            )
            choice_lls.append(ll_choice)
        
        ll_choice_total = sum(choice_lls)
        person_ll = ll_measurement + ll_structural + ll_choice_total
        
        total_ll += person_ll
        
        if person_ll < -100:
            problem_count += 1
            print(f"⚠️ ID {ind_id}: {person_ll:.2f} (측정: {ll_measurement:.2f}, 구조: {ll_structural:.2f}, 선택: {ll_choice_total:.2f})")
        else:
            print(f"✅ ID {ind_id}: {person_ll:.2f}")
    
    print("="*70)
    print(f"총 로그우도: {total_ll:.2f}")
    print(f"문제 있는 개인 수: {problem_count}/30")
    print(f"평균 로그우도: {total_ll/30:.2f}")


if __name__ == '__main__':
    main()

