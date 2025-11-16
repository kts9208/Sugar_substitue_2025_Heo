"""
통합 상관관계 분석 테스트 스크립트

간단한 샘플 데이터로 통합 상관관계 분석 기능을 테스트합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-16
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.analysis.hybrid_choice_model.iclv_models.correlation_analyzer import IntegratedCorrelationAnalyzer
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import (
    MultiLatentMeasurement, MeasurementConfig
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import (
    MultiLatentStructural, StructuralConfig as MultiStructuralConfig
)
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig


def create_sample_data(n_respondents=100):
    """샘플 데이터 생성"""
    np.random.seed(42)
    
    data_list = []
    
    for resp_id in range(1, n_respondents + 1):
        # 사회인구통계변수
        age = np.random.normal(40, 10)
        gender = np.random.choice([0, 1])
        income = np.random.normal(0, 1)
        education = np.random.choice([1, 2, 3, 4, 5, 6])
        
        # 잠재변수 지표 (5점 척도)
        # 건강관심도
        hc_base = np.random.normal(3.5, 0.8)
        q6_11 = [min(5, max(1, int(hc_base + np.random.normal(0, 0.5)))) for _ in range(6)]
        
        # 건강유익성
        pb_base = np.random.normal(3.8, 0.7)
        q12_17 = [min(5, max(1, int(pb_base + np.random.normal(0, 0.5)))) for _ in range(6)]
        
        # 구매의도
        pi_base = np.random.normal(3.2, 0.9)
        q18_20 = [min(5, max(1, int(pi_base + np.random.normal(0, 0.5)))) for _ in range(3)]
        
        # 가격수준
        pp_base = np.random.normal(3.0, 0.8)
        q21_24 = [min(5, max(1, int(pp_base + np.random.normal(0, 0.5)))) for _ in range(4)]
        
        # 영양지식 (2점 척도)
        nk_base = np.random.binomial(1, 0.6)
        q30_49 = [np.random.binomial(1, 0.6) for _ in range(20)]
        
        # 선택 상황 (각 응답자당 12개)
        for choice_set in range(1, 13):
            for alt in range(1, 4):  # 3개 대안
                row = {
                    'respondent_id': resp_id,
                    'choice_set': choice_set,
                    'alternative': alt,
                    'age': age,
                    'age_std': (age - 40) / 10,
                    'gender': gender,
                    'income_std': income,
                    'education_level': education,
                    'sugar_free': np.random.choice([0, 1]),
                    'health_label': np.random.choice([0, 1]),
                    'price': np.random.choice([1000, 1500, 2000, 2500]),
                    'choice': 1 if alt == np.random.choice([1, 2, 3]) else 0
                }
                
                # 지표 추가
                for i, val in enumerate(q6_11, start=6):
                    row[f'q{i}'] = val
                for i, val in enumerate(q12_17, start=12):
                    row[f'q{i}'] = val
                for i, val in enumerate(q18_20, start=18):
                    row[f'q{i}'] = val
                for i, val in enumerate(q21_24, start=21):
                    row[f'q{i}'] = val
                for i, val in enumerate(q30_49, start=30):
                    row[f'q{i}'] = val
                
                data_list.append(row)
    
    return pd.DataFrame(data_list)


def main():
    """메인 테스트 함수"""
    
    print("="*80)
    print("통합 상관관계 분석 테스트")
    print("="*80)
    
    # 1. 샘플 데이터 생성
    print("\n[1] 샘플 데이터 생성 중...")
    data = create_sample_data(n_respondents=50)
    print(f"✅ 샘플 데이터 생성 완료: {data.shape}")
    
    # 2. 모델 설정
    print("\n[2] 모델 설정 중...")
    
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            n_categories=5
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            n_categories=5
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            n_categories=5
        )
    }
    
    measurement_model = MultiLatentMeasurement(measurement_configs)
    
    structural_configs = {
        'health_concern': MultiStructuralConfig(
            latent_variable='health_concern',
            sociodemographics=['age_std', 'gender', 'income_std']
        ),
        'perceived_benefit': MultiStructuralConfig(
            latent_variable='perceived_benefit',
            sociodemographics=['age_std', 'gender']
        ),
        'purchase_intention': MultiStructuralConfig(
            latent_variable='purchase_intention',
            sociodemographics=['age_std', 'income_std']
        )
    }
    
    structural_model = MultiLatentStructural(structural_configs)
    
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='binary',
        all_lvs_as_main=True,
        main_lvs=['health_concern', 'perceived_benefit', 'purchase_intention']
    )
    
    print("✅ 모델 설정 완료")
    
    # 3. 통합 상관관계 분석 실행
    print("\n[3] 통합 상관관계 분석 실행 중...")
    
    analyzer = IntegratedCorrelationAnalyzer()
    
    results = analyzer.analyze_all_correlations(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        choice_config=choice_config,
        save_path=None  # 테스트이므로 저장하지 않음
    )
    
    print("\n✅ 분석 완료!")
    
    # 4. 결과 확인
    print("\n[4] 결과 확인...")
    
    analyzer.print_summary()
    
    print("\n" + "="*80)
    print("테스트 성공!")
    print("="*80)


if __name__ == "__main__":
    main()

