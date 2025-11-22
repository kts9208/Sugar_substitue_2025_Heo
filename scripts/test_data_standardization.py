"""
데이터 표준화 통합 테스트

DataStandardizer가 simultaneous_estimator_fixed.py에 제대로 통합되었는지 확인합니다.

Author: Sugar Substitute Research Team
Date: 2025-01-22
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.analysis.hybrid_choice_model.iclv_models.data_standardizer import DataStandardizer


def test_data_standardizer_basic():
    """DataStandardizer 기본 기능 테스트"""
    print("=" * 80)
    print("테스트 1: DataStandardizer 기본 기능")
    print("=" * 80)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    data = pd.DataFrame({
        'respondent_id': np.repeat(range(1, 101), 3),
        'price': np.random.choice([2000, 4000, 6000], 300),
        'health_label': np.random.choice([0, 1], 300),
        'chosen': np.random.choice([0, 1], 300)
    })
    
    print(f"\n[원본 데이터]")
    print(f"  행 수: {len(data)}")
    print(f"  price 통계: mean={data['price'].mean():.2f}, std={data['price'].std():.2f}")
    print(f"  health_label 통계: mean={data['health_label'].mean():.2f}, std={data['health_label'].std():.2f}")
    
    # DataStandardizer 생성
    standardizer = DataStandardizer(
        variables_to_standardize=['price', 'health_label']
    )
    
    # Fit & Transform
    data_standardized = standardizer.fit_transform(data)
    
    print(f"\n[표준화된 데이터]")
    print(f"  price 통계: mean={data_standardized['price'].mean():.6f}, std={data_standardized['price'].std():.6f}")
    print(f"  health_label 통계: mean={data_standardized['health_label'].mean():.6f}, std={data_standardized['health_label'].std():.6f}")
    
    # 검증
    price_mean = data_standardized['price'].mean()
    price_std = data_standardized['price'].std(ddof=0)  # 모집단 표준편차
    health_mean = data_standardized['health_label'].mean()
    health_std = data_standardized['health_label'].std(ddof=0)  # 모집단 표준편차

    assert abs(price_mean) < 1e-10, f"price 평균이 0이 아닙니다: {price_mean}"
    assert abs(price_std - 1.0) < 1e-6, f"price 표준편차가 1이 아닙니다: {price_std}"
    assert abs(health_mean) < 1e-10, f"health_label 평균이 0이 아닙니다: {health_mean}"
    assert abs(health_std - 1.0) < 1e-6, f"health_label 표준편차가 1이 아닙니다: {health_std}"
    
    print("\n✅ 테스트 1 통과: 표준화가 올바르게 작동합니다.")
    
    return standardizer, data, data_standardized


def test_inverse_transform(standardizer, data_original, data_standardized):
    """역변환 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: 역변환")
    print("=" * 80)
    
    # 역변환
    data_inverse = standardizer.inverse_transform(data_standardized)
    
    print(f"\n[역변환 데이터]")
    print(f"  price 통계: mean={data_inverse['price'].mean():.2f}, std={data_inverse['price'].std():.2f}")
    print(f"  health_label 통계: mean={data_inverse['health_label'].mean():.2f}, std={data_inverse['health_label'].std():.2f}")
    
    # 검증: 원본과 역변환 데이터가 동일해야 함
    price_diff = np.abs(data_original['price'] - data_inverse['price']).max()
    health_diff = np.abs(data_original['health_label'] - data_inverse['health_label']).max()
    
    print(f"\n[원본 vs 역변환 차이]")
    print(f"  price 최대 차이: {price_diff:.6e}")
    print(f"  health_label 최대 차이: {health_diff:.6e}")
    
    assert price_diff < 1e-10, f"price 역변환 오차가 큽니다: {price_diff}"
    assert health_diff < 1e-10, f"health_label 역변환 오차가 큽니다: {health_diff}"
    
    print("\n✅ 테스트 2 통과: 역변환이 올바르게 작동합니다.")


def test_config_integration():
    """Config 통합 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: Config 통합")
    print("=" * 80)
    
    from src.analysis.hybrid_choice_model.iclv_models.iclv_config import EstimationConfig
    
    # Config 생성
    config = EstimationConfig()
    
    # 기본값 확인
    assert hasattr(config, 'standardize_choice_attributes'), \
        "EstimationConfig에 standardize_choice_attributes 속성이 없습니다."
    
    print(f"\n[Config 설정]")
    print(f"  use_parameter_scaling: {config.use_parameter_scaling}")
    print(f"  standardize_choice_attributes: {config.standardize_choice_attributes}")
    
    # 기본값이 True인지 확인
    assert config.standardize_choice_attributes == True, \
        "standardize_choice_attributes 기본값이 True가 아닙니다."
    
    print("\n✅ 테스트 3 통과: Config 통합이 올바릅니다.")


def test_estimator_import():
    """Estimator import 테스트"""
    print("\n" + "=" * 80)
    print("테스트 4: Estimator Import")
    print("=" * 80)

    try:
        from src.analysis.hybrid_choice_model.iclv_models import simultaneous_estimator_fixed
        from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator
        print("\n✅ SimultaneousEstimator import 성공")

        # DataStandardizer import 확인 (모듈 소스 확인)
        import inspect
        module_source = inspect.getsource(simultaneous_estimator_fixed)

        assert 'from .data_standardizer import DataStandardizer' in module_source, \
            "DataStandardizer import 문이 없습니다."

        print("✅ DataStandardizer import 문 확인")

        # estimate 메서드에 표준화 로직 확인
        estimate_source = inspect.getsource(SimultaneousEstimator.estimate)
        
        assert 'standardize_choice_attributes' in estimate_source, \
            "estimate 메서드에 standardize_choice_attributes 로직이 없습니다."
        
        assert 'DataStandardizer' in estimate_source, \
            "estimate 메서드에 DataStandardizer 사용 로직이 없습니다."
        
        print("✅ estimate 메서드에 표준화 로직 확인")
        
        print("\n✅ 테스트 4 통과: Estimator 통합이 올바릅니다.")
        
    except Exception as e:
        print(f"\n❌ 테스트 4 실패: {e}")
        raise


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 80)
    print("데이터 표준화 통합 테스트")
    print("=" * 80)
    
    try:
        # 테스트 1: 기본 기능
        standardizer, data_original, data_standardized = test_data_standardizer_basic()
        
        # 테스트 2: 역변환
        test_inverse_transform(standardizer, data_original, data_standardized)
        
        # 테스트 3: Config 통합
        test_config_integration()
        
        # 테스트 4: Estimator import
        test_estimator_import()
        
        print("\n" + "=" * 80)
        print("✅ 모든 테스트 통과!")
        print("=" * 80)
        print("\n통합 완료:")
        print("  1. DataStandardizer 클래스 작동 확인")
        print("  2. 역변환 기능 확인")
        print("  3. Config 설정 확인")
        print("  4. Estimator 통합 확인")
        print("\n다음 단계:")
        print("  - test_gpu_batch_iclv.py 실행하여 실제 추정 테스트")
        print("  - 표준화 전후 그래디언트 크기 비교")
        print("  - 최적화 수렴 속도 비교")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 테스트 실패: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

