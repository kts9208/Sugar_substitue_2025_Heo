"""
Gradient 계산 방법 벤치마크: Numerical vs Analytic

각 방법의 실제 계산 시간을 측정하여 비교합니다.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig, StructuralConfig, ChoiceConfig, EstimationConfig, ICLVConfig
)
from analysis.hybrid_choice_model.iclv_models.gpu_batch_estimator import GPUBatchEstimator


def load_test_data():
    """테스트 데이터 로드"""
    data_path = project_root / 'data' / 'processed' / 'choice_data_with_indicators.csv'
    data = pd.read_csv(data_path)
    print(f"데이터 로드 완료: {data.shape}")
    return data


def create_iclv_config():
    """ICLV 설정 생성"""
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=[f'q{i}' for i in range(1, 11)],
            n_categories=5
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=[f'q{i}' for i in range(11, 21)],
            n_categories=5
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=[f'q{i}' for i in range(21, 30)],
            n_categories=5
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=[f'q{i}' for i in range(30, 50)],
            n_categories=5
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=[f'q{i}' for i in range(50, 60)],
            n_categories=5
        )
    }
    
    structural_config = StructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
        sociodemographics=['age', 'gender', 'income']
    )
    
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price']
    )
    
    return ICLVConfig(
        measurement=measurement_configs,
        structural=structural_config,
        choice=choice_config
    )


def benchmark_gradient_calculation(data, config, method='numerical', n_iterations=1):
    """
    그래디언트 계산 벤치마크
    
    Args:
        data: 데이터
        config: ICLV 설정
        method: 'numerical' 또는 'analytic'
        n_iterations: 측정 반복 횟수
    
    Returns:
        평균 시간 (초)
    """
    print(f"\n{'='*70}")
    print(f"벤치마크: {method.upper()} Gradient")
    print(f"{'='*70}")
    
    # Estimation config
    estimation_config = EstimationConfig(
        optimizer='BFGS',
        use_analytic_gradient=(method == 'analytic'),
        n_draws=100,
        draw_type='halton',
        max_iterations=2,  # 2 iterations만 실행
        calculate_se=False,  # 표준오차 계산 비활성화 (속도 향상)
        use_parallel=False,
        n_cores=None
    )
    
    # 전체 config
    full_config = ICLVConfig(
        measurement=config.measurement,
        structural=config.structural,
        choice=config.choice,
        estimation=estimation_config
    )
    
    # Estimator 생성
    estimator = GPUBatchEstimator(full_config)
    
    times = []
    
    for i in range(n_iterations):
        print(f"\n반복 {i+1}/{n_iterations}...")
        
        start_time = time.time()
        
        try:
            # 추정 실행 (2 iterations만)
            result = estimator.estimate(data)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"  완료: {elapsed:.1f}초")
            print(f"  최종 LL: {result.log_likelihood:.4f}")
            
        except KeyboardInterrupt:
            print("\n사용자가 중단했습니다.")
            break
        except Exception as e:
            print(f"  에러 발생: {e}")
            break
    
    if times:
        avg_time = np.mean(times)
        print(f"\n평균 시간: {avg_time:.1f}초")
        return avg_time
    else:
        return None


def main():
    """메인 함수"""
    print("="*70)
    print("Gradient 계산 방법 벤치마크")
    print("="*70)
    
    # 데이터 로드
    print("\n1. 데이터 로드 중...")
    data = load_test_data()
    
    # 설정 생성
    print("\n2. ICLV 설정 생성 중...")
    config = create_iclv_config()
    
    # 벤치마크 실행
    results = {}
    
    # Numerical gradient
    print("\n3. Numerical Gradient 벤치마크...")
    numerical_time = benchmark_gradient_calculation(
        data, config, method='numerical', n_iterations=1
    )
    if numerical_time:
        results['Numerical'] = numerical_time
    
    # Analytic gradient (CPU)
    print("\n4. Analytic Gradient (CPU) 벤치마크...")
    analytic_time = benchmark_gradient_calculation(
        data, config, method='analytic', n_iterations=1
    )
    if analytic_time:
        results['Analytic (CPU)'] = analytic_time
    
    # 결과 요약
    print("\n" + "="*70)
    print("벤치마크 결과 요약")
    print("="*70)
    
    if results:
        print(f"\n{'방법':<20} {'시간 (초)':<15} {'시간 (분)':<15} {'상대 속도':<15}")
        print("-"*70)
        
        baseline = results.get('Numerical', 1.0)
        
        for method, time_sec in results.items():
            time_min = time_sec / 60
            relative = time_sec / baseline
            print(f"{method:<20} {time_sec:>10.1f}초     {time_min:>10.1f}분     {relative:>10.2f}×")
        
        print("\n" + "="*70)
        
        # 전체 최적화 예상 시간 (20 iterations)
        print("\n전체 최적화 예상 시간 (20 iterations):")
        print("-"*70)
        
        for method, time_sec in results.items():
            # 2 iterations 측정했으므로 10배
            total_time_sec = time_sec * 10
            total_time_min = total_time_sec / 60
            total_time_hour = total_time_min / 60
            
            print(f"{method:<20} {total_time_hour:>10.1f}시간 ({total_time_min:>10.1f}분)")
    
    else:
        print("\n벤치마크 결과가 없습니다.")
    
    print("\n" + "="*70)
    print("벤치마크 완료")
    print("="*70)


if __name__ == '__main__':
    main()

