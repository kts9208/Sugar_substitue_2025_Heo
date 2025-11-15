"""
SEMEstimator 파라미터 추출 테스트
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.sem_estimator import SEMEstimator
from analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from analysis.hybrid_choice_model.iclv_models.iclv_config import MeasurementConfig
from analysis.hybrid_choice_model.iclv_models.multi_latent_config import MultiLatentStructuralConfig


def main():
    print("="*70)
    print("SEMEstimator 파라미터 추출 테스트")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드...")
    data_paths = [
        project_root / "data" / "processed" / "survey",
        project_root / "processed_data" / "survey_data"
    ]
    
    data_dir = None
    for path in data_paths:
        if path.exists():
            data_dir = path
            break
    
    if data_dir is None:
        print("❌ 데이터 디렉토리를 찾을 수 없습니다.")
        return
    
    # 데이터 로드 (5개 잠재변수) - 326명 원본 데이터 사용
    dfs = []
    factors = [
        'health_concern',
        'perceived_benefit',
        'purchase_intention',
        'perceived_price',
        'nutrition_knowledge'  # 원본 데이터 사용 후 역코딩 처리
    ]

    for factor_name in factors:
        file_path = data_dir / f"{factor_name}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'no' in df.columns:
                df = df.set_index('no')
            dfs.append(df)
            print(f"  - {factor_name}: {df.shape}")
        else:
            print(f"  ⚠️ {factor_name}.csv 파일을 찾을 수 없습니다.")

    data = pd.concat(dfs, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    # 영양지식 역코딩 처리 (Q33, Q47만)
    # 참고: docs/NUTRITION_KNOWLEDGE_CODING.md
    reverse_items = ['q33', 'q47']
    for item in reverse_items:
        if item in data.columns:
            data[item] = 6 - data[item]  # 5점 척도 역코딩: 6 - 원본값
            print(f"  ✅ {item} 역코딩 완료")

    print(f"✅ 데이터 로드 완료: {data.shape} (326명 원본 + Q33, Q47 역코딩)")
    
    # 2. 모델 설정 (5개 잠재변수)
    print("\n[2] 모델 설정...")
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            measurement_method='continuous_linear',
            n_categories=5
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            measurement_method='continuous_linear',
            n_categories=5
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            measurement_method='continuous_linear',
            n_categories=5
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],
            measurement_method='continuous_linear',
            n_categories=5
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=['q30', 'q31', 'q32', 'q33', 'q34', 'q35', 'q36', 'q37', 'q38', 'q39',
                       'q40', 'q41', 'q42', 'q43', 'q44', 'q45', 'q46', 'q47', 'q48', 'q49'],
            measurement_method='continuous_linear',
            n_categories=5
        )
    }
    print(f"  측정모델: {len(measurement_configs)}개 잠재변수")
    
    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_price', 'nutrition_knowledge'],
        covariates=[],
        hierarchical_paths=[
            {'target': 'perceived_benefit', 'predictors': ['health_concern']},
            {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
        ]
    )
    print(f"  구조모델: 외생 LV {len(structural_config.exogenous_lvs)}개, 경로 {len(structural_config.hierarchical_paths)}개")

    measurement_model = MultiLatentMeasurement(measurement_configs)
    structural_model = MultiLatentStructural(structural_config)
    print("✅ 모델 설정 완료")
    
    # 3. SEM 추정
    print("\n[3] SEM 추정...")
    sem_estimator = SEMEstimator()
    results = sem_estimator.fit(data, measurement_model, structural_model)
    print("✅ SEM 추정 완료")
    
    # 4. 파라미터 출력
    print("\n" + "="*70)
    print("파라미터 추출 결과")
    print("="*70)
    
    print(f"\n[측정모델]")
    print(f"  요인적재량: {len(results['loadings'])}개")
    if len(results['loadings']) > 0:
        print("\n" + results['loadings'][['lval', 'rval', 'Estimate']].to_string())
    
    print(f"\n  측정 오차분산: {len(results['measurement_errors'])}개")
    if len(results['measurement_errors']) > 0:
        print("\n" + results['measurement_errors'][['lval', 'Estimate']].head(10).to_string())
    
    print(f"\n[구조모델]")
    print(f"  경로계수: {len(results['paths'])}개")
    if len(results['paths']) > 0:
        print("\n" + results['paths'][['lval', 'rval', 'Estimate']].to_string())
    
    print(f"\n  구조 오차분산: {len(results['structural_errors'])}개")
    if len(results['structural_errors']) > 0:
        print("\n" + results['structural_errors'][['lval', 'Estimate']].to_string())
    
    print(f"\n  외생 LV 분산: {len(results['lv_variances'])}개")
    if len(results['lv_variances']) > 0:
        print("\n" + results['lv_variances'][['lval', 'Estimate']].to_string())
    
    print("\n" + "="*70)
    print("✅ 테스트 완료!")
    print("="*70)


if __name__ == "__main__":
    main()

