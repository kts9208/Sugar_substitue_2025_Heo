"""
통합 상관관계 분석 예제

1단계 SEM 변수와 2단계 선택모델 변수를 모두 포함하는 상관관계 분석

이 스크립트는 다음을 수행합니다:
1. 데이터 로드
2. 측정모델, 구조모델, 선택모델 설정
3. 통합 상관관계 분석 실행
4. 결과 저장 및 출력

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
from src.analysis.hybrid_choice_model.iclv_models.sem_estimator import SEMEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig, ChoiceConfig
)


def main():
    """메인 실행 함수"""
    
    print("="*80)
    print("통합 상관관계 분석 예제")
    print("="*80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    
    if not data_path.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("먼저 데이터 전처리를 실행하세요.")
        return
    
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {data.shape}")
    print(f"   - 응답자 수: {data['respondent_id'].nunique()}")
    print(f"   - 총 관측치 수: {len(data)}")
    
    # 2. 측정모델 설정 (5개 잠재변수)
    print("\n[2] 측정모델 설정 중...")
    
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
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],  # 실제 데이터에 있는 컬럼
            n_categories=5
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=[f'q{i}' for i in range(30, 50)],
            n_categories=2
        )
    }
    
    measurement_model = MultiLatentMeasurement(measurement_configs)
    print(f"✅ 측정모델 설정 완료: {len(measurement_configs)}개 잠재변수")
    
    # 3. 구조모델 설정 (계층적 구조)
    print("\n[3] 구조모델 설정 중...")

    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import MultiLatentStructuralConfig

    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_price', 'nutrition_knowledge'],
        covariates=['age_std', 'gender', 'income_std', 'education_level'],  # 사회인구통계변수
        hierarchical_paths=[
            {'target': 'perceived_benefit', 'predictors': ['health_concern']},
            {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
        ]
    )

    structural_model = MultiLatentStructural(structural_config)
    print(f"✅ 구조모델 설정 완료")
    
    # 4. 선택모델 설정
    print("\n[4] 선택모델 설정 중...")
    
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='binary',
        price_variable='price',
        all_lvs_as_main=True,
        main_lvs=['health_concern', 'perceived_benefit', 'perceived_price',
                  'nutrition_knowledge', 'purchase_intention']
    )
    print(f"✅ 선택모델 설정 완료")
    print(f"   - 속성변수: {choice_config.choice_attributes}")
    print(f"   - 주효과 잠재변수: {choice_config.main_lvs}")
    
    # 5. SEMEstimator 생성
    print("\n[5] SEMEstimator 생성 중...")
    sem_estimator = SEMEstimator()
    print("✅ SEMEstimator 생성 완료")
    
    # 6. 통합 상관관계 분석 실행
    print("\n[6] 통합 상관관계 분석 실행 중...")
    print("   (이 작업은 몇 분 정도 소요될 수 있습니다)")
    
    save_path = project_root / 'results' / 'correlation_analysis'
    
    results = sem_estimator.analyze_correlations(
        data=data,
        measurement_model=measurement_model,
        structural_model=structural_model,
        choice_config=choice_config,
        save_path=str(save_path)
    )
    
    print("\n✅ 통합 상관관계 분석 완료!")
    print(f"\n결과가 {save_path}에 저장되었습니다.")
    
    # 7. 주요 결과 출력
    print("\n" + "="*80)
    print("주요 결과")
    print("="*80)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"\n변수 개수:")
        print(f"  - 잠재변수: {summary.get('n_latent_variables', 0)}개")
        print(f"  - 선택모델 속성변수: {summary.get('n_attributes', 0)}개")
        print(f"  - 사회인구통계변수: {summary.get('n_sociodem_variables', 0)}개")
        
        print(f"\n상관관계 강도 분포:")
        print(f"  - 강한 상관관계 (|r| > 0.5): {summary.get('n_strong_correlations', 0)}개")
        print(f"  - 중간 상관관계 (0.3 < |r| ≤ 0.5): {summary.get('n_moderate_correlations', 0)}개")
        print(f"  - 약한 상관관계 (|r| ≤ 0.3): {summary.get('n_weak_correlations', 0)}개")


if __name__ == "__main__":
    main()

