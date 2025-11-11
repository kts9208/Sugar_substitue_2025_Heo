"""
Continuous Linear Measurement 테스트

연속형 선형 측정모델의 기본 기능을 테스트합니다.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import MeasurementConfig
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import (
    ContinuousLinearMeasurement,
    OrderedProbitMeasurement
)
from src.analysis.hybrid_choice_model.iclv_models.gpu_measurement_equations import (
    GPUContinuousLinearMeasurement
)


def test_continuous_linear_basic():
    """기본 기능 테스트"""
    print("\n" + "="*70)
    print("테스트 1: ContinuousLinearMeasurement 기본 기능")
    print("="*70)
    
    config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8'],
        measurement_method='continuous_linear',
        fix_first_loading=True,
        fix_error_variance=False
    )
    
    model = ContinuousLinearMeasurement(config)
    
    # 파라미터 초기화
    params = model.initialize_parameters()
    print(f"\n✅ 파라미터 초기화:")
    print(f"   - zeta: {params['zeta']}")
    print(f"   - sigma_sq: {params['sigma_sq']}")
    
    assert 'zeta' in params
    assert 'sigma_sq' in params
    assert len(params['zeta']) == 3
    assert len(params['sigma_sq']) == 3
    assert params['zeta'][0] == 1.0  # 첫 번째 고정
    
    # 파라미터 수
    n_params = model.get_n_parameters()
    print(f"\n✅ 파라미터 수: {n_params}개")
    print(f"   - zeta: 2개 (첫 번째 고정)")
    print(f"   - sigma_sq: 3개")
    assert n_params == 5  # zeta: 2 (첫 번째 고정) + sigma_sq: 3
    
    # 로그우도 계산
    data = pd.DataFrame({'q6': [3], 'q7': [4], 'q8': [5]})
    ll = model.log_likelihood(data, 4.0, params)
    print(f"\n✅ 로그우도 계산: {ll:.4f}")
    assert isinstance(ll, float)
    
    print("\n✅ 테스트 1 통과!")


def test_parameter_count_comparison():
    """파라미터 수 비교 테스트"""
    print("\n" + "="*70)
    print("테스트 2: ContinuousLinear vs OrderedProbit 파라미터 수 비교")
    print("="*70)
    
    # ContinuousLinear
    config_cont = MeasurementConfig(
        latent_variable='test',
        indicators=['q1', 'q2', 'q3'],
        measurement_method='continuous_linear',
        fix_first_loading=True,
        fix_error_variance=False
    )
    model_cont = ContinuousLinearMeasurement(config_cont)
    n_params_cont = model_cont.get_n_parameters()
    
    # OrderedProbit
    config_op = MeasurementConfig(
        latent_variable='test',
        indicators=['q1', 'q2', 'q3'],
        measurement_method='ordered_probit',
        n_categories=5
    )
    model_op = OrderedProbitMeasurement(config_op)
    n_params_op = model_op.get_n_parameters()
    
    print(f"\n📊 파라미터 수 비교 (3개 지표 기준):")
    print(f"   - ContinuousLinear: {n_params_cont}개")
    print(f"     * zeta: 2개 (첫 번째 고정)")
    print(f"     * sigma_sq: 3개")
    print(f"   - OrderedProbit: {n_params_op}개")
    print(f"     * zeta: 3개")
    print(f"     * tau: 12개 (3 × 4)")
    print(f"   - 감소량: {n_params_op - n_params_cont}개 ({(n_params_op - n_params_cont) / n_params_op * 100:.1f}%)")
    
    assert n_params_cont == 5  # zeta: 2 + sigma_sq: 3
    assert n_params_op == 15   # zeta: 3 + tau: 12
    
    print("\n✅ 테스트 2 통과!")


def test_gpu_continuous_linear():
    """GPU 버전 테스트"""
    print("\n" + "="*70)
    print("테스트 3: GPUContinuousLinearMeasurement")
    print("="*70)
    
    config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8'],
        measurement_method='continuous_linear',
        fix_first_loading=True,
        fix_error_variance=False
    )
    
    # CPU 버전
    model_cpu = GPUContinuousLinearMeasurement(config, use_gpu=False)
    params_cpu = model_cpu.initialize_parameters()
    data = pd.DataFrame({'q6': [3], 'q7': [4], 'q8': [5]})
    ll_cpu = model_cpu.log_likelihood(data, 4.0, params_cpu)
    
    print(f"\n✅ CPU 버전:")
    print(f"   - 파라미터 수: {model_cpu.get_n_parameters()}개")
    print(f"   - 로그우도: {ll_cpu:.4f}")
    
    # GPU 버전 (사용 가능한 경우)
    try:
        import cupy as cp
        model_gpu = GPUContinuousLinearMeasurement(config, use_gpu=True)
        params_gpu = model_gpu.initialize_parameters()
        ll_gpu = model_gpu.log_likelihood(data, 4.0, params_gpu)
        
        print(f"\n✅ GPU 버전:")
        print(f"   - 파라미터 수: {model_gpu.get_n_parameters()}개")
        print(f"   - 로그우도: {ll_gpu:.4f}")
        print(f"   - CPU vs GPU 차이: {abs(ll_cpu - ll_gpu):.6f}")
        
        assert abs(ll_cpu - ll_gpu) < 1e-6
    except ImportError:
        print("\n⚠️  GPU (CuPy) 사용 불가 - CPU 버전만 테스트")
    
    print("\n✅ 테스트 3 통과!")


def test_multi_latent_variable():
    """다중 잠재변수 파라미터 수 계산"""
    print("\n" + "="*70)
    print("테스트 4: 다중 잠재변수 파라미터 수 (5개 LV)")
    print("="*70)
    
    # 5개 잠재변수 설정
    lv_configs = {
        'health_concern': 6,      # 6개 지표
        'perceived_benefit': 6,   # 6개 지표
        'perceived_price': 3,     # 3개 지표
        'nutrition_knowledge': 20, # 20개 지표
        'purchase_intention': 3   # 3개 지표
    }
    
    total_indicators = sum(lv_configs.values())
    
    # ContinuousLinear 파라미터 수
    # zeta: (n_indicators - 5) (각 LV의 첫 번째 고정)
    # sigma_sq: n_indicators
    n_params_cont = (total_indicators - 5) + total_indicators
    
    # OrderedProbit 파라미터 수
    # zeta: n_indicators
    # tau: n_indicators * 4 (5점 척도)
    n_params_op = total_indicators + (total_indicators * 4)
    
    print(f"\n📊 5개 잠재변수 (총 {total_indicators}개 지표):")
    print(f"\n   ContinuousLinear:")
    print(f"   - zeta: {total_indicators - 5}개 (각 LV 첫 번째 고정)")
    print(f"   - sigma_sq: {total_indicators}개")
    print(f"   - 합계: {n_params_cont}개")
    
    print(f"\n   OrderedProbit:")
    print(f"   - zeta: {total_indicators}개")
    print(f"   - tau: {total_indicators * 4}개 ({total_indicators} × 4)")
    print(f"   - 합계: {n_params_op}개")
    
    print(f"\n   📉 감소량: {n_params_op - n_params_cont}개 ({(n_params_op - n_params_cont) / n_params_op * 100:.1f}%)")
    
    assert n_params_cont == 71  # (38-5) + 38 = 71
    assert n_params_op == 190   # 38 + 152 = 190
    
    print("\n✅ 테스트 4 통과!")


def test_bounds():
    """파라미터 bounds 테스트"""
    print("\n" + "="*70)
    print("테스트 5: 파라미터 Bounds")
    print("="*70)
    
    config = MeasurementConfig(
        latent_variable='test',
        indicators=['q1', 'q2', 'q3'],
        measurement_method='continuous_linear',
        fix_first_loading=True,
        fix_error_variance=False
    )
    
    model = ContinuousLinearMeasurement(config)
    bounds = model.get_parameter_bounds()
    
    print(f"\n✅ Bounds (5개 파라미터):")
    for i, (lower, upper) in enumerate(bounds):
        if i < 2:
            print(f"   - zeta[{i+1}]: [{lower}, {upper}]")
        else:
            print(f"   - sigma_sq[{i-2}]: [{lower}, {upper}]")
    
    assert len(bounds) == 5  # zeta: 2 + sigma_sq: 3
    assert bounds[0] == (-10.0, 10.0)  # zeta
    assert bounds[2] == (0.01, 100.0)  # sigma_sq
    
    print("\n✅ 테스트 5 통과!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Continuous Linear Measurement 테스트 시작")
    print("="*70)
    
    test_continuous_linear_basic()
    test_parameter_count_comparison()
    test_gpu_continuous_linear()
    test_multi_latent_variable()
    test_bounds()
    
    print("\n" + "="*70)
    print("✅ 모든 테스트 통과!")
    print("="*70)

