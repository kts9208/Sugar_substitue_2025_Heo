"""
ICLV 동시추정 테스트 (수정 버전 v2)

수정사항:
1. Panel Product 구현
2. L-BFGS-B 알고리즘 + bounds
3. 수치 안정성 강화
4. maxiter 증가
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    ICLVConfig,
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig,
    EstimationConfig
)


# DataConfig를 직접 정의
from dataclasses import dataclass

@dataclass
class DataConfig:
    """데이터 설정"""
    individual_id: str = 'respondent_id'
    choice_id: str = 'choice_set'


def main():
    print("="*70)
    print("ICLV 동시추정 테스트 (수정 버전 v2)")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    print(f"   데이터 shape: {data.shape}")
    print(f"   개인 수: {data['respondent_id'].nunique()}")
    
    # 2. 소규모 테스트용 데이터 추출
    print("\n2. 소규모 테스트 데이터 추출...")
    n_test_individuals = 30  # 🔴 30명으로 증가 (원래 5)
    test_ids = data['respondent_id'].unique()[:n_test_individuals]
    test_data = data[data['respondent_id'].isin(test_ids)].copy()
    print(f"   테스트 개인 수: {n_test_individuals}")
    print(f"   테스트 데이터 shape: {test_data.shape}")
    
    # 3. 설정
    print("\n3. ICLV 설정...")
    
    # 측정모델 설정 (1개 잠재변수, 6개 지표)
    measurement_config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        n_categories=5
    )
    
    # 구조모델 설정
    structural_config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std'],
        include_in_choice=False
    )
    
    # 선택모델 설정
    choice_config = ChoiceConfig(
        choice_attributes=['price', 'health_label']
    )
    
    # 추정 설정
    estimation_config = EstimationConfig(
        optimizer='BFGS',                # 🔴 BFGS 사용
        use_analytic_gradient=True,      # 🔴 Analytic gradient 사용 (Apollo 방식)
        n_draws=50,                      # 🔴 50 draws로 증가 (원래 10)
        draw_type='halton',
        max_iterations=500,              # 🔴 BFGS는 빠르므로 500회로 충분
        calculate_se=False
    )
    
    # 통합 설정 (individual_id_column 사용)
    config = ICLVConfig(
        measurement=measurement_config,
        structural=structural_config,
        choice=choice_config,
        estimation=estimation_config,
        individual_id_column='respondent_id',  # 이 필드 사용
        choice_column='choice'
    )
    
    # config에 data 속성 추가 (SimultaneousEstimator가 사용)
    config.data = DataConfig(
        individual_id='respondent_id',
        choice_id='choice_set'
    )
    
    print("   설정 완료")
    print(f"   - 잠재변수: {measurement_config.latent_variable}")
    print(f"   - 지표 수: {len(measurement_config.indicators)}")
    print(f"   - 사회인구학적 변수: {len(structural_config.sociodemographics)}")
    print(f"   - 선택 속성: {len(choice_config.choice_attributes)}")
    print(f"   - Halton draws: {estimation_config.n_draws}")
    print(f"   - 최대 반복: {estimation_config.max_iterations}")
    print(f"   - 테스트 개인 수: {n_test_individuals}")
    
    # 4. 모델 생성
    print("\n4. 모델 생성...")

    # 측정모델
    measurement_model = OrderedProbitMeasurement(measurement_config)
    print("   - 측정모델 생성 완료")

    # 구조모델
    structural_model = LatentVariableRegression(structural_config)
    print("   - 구조모델 생성 완료")

    # 선택모델
    choice_model = BinaryProbitChoice(choice_config)
    print("   - 선택모델 생성 완료")

    # 5. 추정
    print("\n5. ICLV 동시추정 실행...")
    print("   (수정사항: Panel Product + BFGS + Analytic Gradient (Apollo 방식) + -inf 클리핑)")
    print("   (로깅: 매 5회 반복마다 LL 출력, 개선 시 즉시 출력)")
    print()

    estimator = SimultaneousEstimator(config)

    try:
        results = estimator.estimate(
            test_data,
            measurement_model,
            structural_model,
            choice_model
        )

        # 6. 결과 출력
        print("\n" + "="*70)
        print("추정 결과")
        print("="*70)
        
        print(f"\n수렴 여부: {results['success']}")
        print(f"최종 로그우도: {results['log_likelihood']:.4f}")
        print(f"반복 횟수: {results['n_iterations']}")
        
        print("\n파라미터 추정값:")
        params = results['parameters']
        
        print("\n[측정모델]")
        print(f"  요인적재량 (zeta): {params['measurement']['zeta']}")
        print(f"  임계값 (tau) shape: {params['measurement']['tau'].shape}")
        
        print("\n[구조모델]")
        print(f"  gamma: {params['structural']['gamma']}")
        
        print("\n[선택모델]")
        print(f"  intercept: {params['choice']['intercept']:.4f}")
        print(f"  beta: {params['choice']['beta']}")
        print(f"  lambda: {params['choice']['lambda']:.4f}")
        
        # 결과 저장
        output_path = project_root / 'results' / 'iclv_test_fixed_results.txt'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("ICLV 동시추정 결과 (수정 버전)\n")
            f.write("="*70 + "\n\n")
            f.write(f"수렴 여부: {results['success']}\n")
            f.write(f"최종 로그우도: {results['log_likelihood']:.4f}\n")
            f.write(f"반복 횟수: {results['n_iterations']}\n")
            f.write(f"\n파라미터:\n{params}\n")
        
        print(f"\n결과 저장: {output_path}")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("테스트 완료!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())

