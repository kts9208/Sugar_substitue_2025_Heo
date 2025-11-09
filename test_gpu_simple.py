"""
GPU 측정모델 테스트
"""

import os
import sys

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 70)
print("GPU 측정모델 테스트")
print("=" * 70)

# GPU 측정모델 임포트
try:
    from src.analysis.hybrid_choice_model.iclv_models.gpu_measurement_equations import (
        GPUOrderedProbitMeasurement,
        GPUMultiLatentMeasurement,
        GPU_AVAILABLE
    )

    print(f"\n✅ GPU 측정모델 임포트 성공")
    print(f"   GPU 사용 가능: {GPU_AVAILABLE}")

    if GPU_AVAILABLE:
        import cupy as cp
        print(f"   CuPy 버전: {cp.__version__}")
        try:
            cp.cuda.Device(0).use()
            props = cp.cuda.runtime.getDeviceProperties(0)
            print(f"   GPU: {props['name'].decode()}")
            print(f"   GPU 메모리: {props['totalGlobalMem'] / 1024**3:.2f} GB")
        except Exception as e:
            print(f"   ⚠️ GPU 정보 확인 실패: {e}")
    else:
        print("   ⚠️ CPU 모드로 작동")

    # 간단한 측정모델 테스트
    print("\n" + "-" * 70)
    print("측정모델 기능 테스트")
    print("-" * 70)

    from src.analysis.hybrid_choice_model.iclv_models.iclv_config import MeasurementConfig
    import numpy as np
    import pandas as pd

    # 테스트 설정
    config = MeasurementConfig(
        latent_variable='test_lv',
        indicators=['Q1', 'Q2', 'Q3'],
        n_categories=5
    )

    # 모델 생성
    model = GPUOrderedProbitMeasurement(config, use_gpu=GPU_AVAILABLE)
    print(f"✅ 측정모델 생성 성공 (GPU 모드: {model.use_gpu})")

    # 테스트 데이터
    test_data = pd.DataFrame({
        'Q1': [3, 4, 2, 5, 1],
        'Q2': [4, 3, 3, 4, 2],
        'Q3': [2, 5, 4, 3, 1]
    })

    # 파라미터
    params = {
        'zeta': np.array([1.0, 0.8, 1.2]),
        'tau': np.array([
            [-2.0, -1.0, 0.0, 1.0],
            [-1.5, -0.5, 0.5, 1.5],
            [-2.5, -1.5, -0.5, 0.5]
        ])
    }

    # 우도 계산 테스트
    latent_var = 0.5
    ll = model.log_likelihood(test_data, latent_var, params)
    print(f"✅ 로그우도 계산 성공: {ll:.4f}")

    print("\n" + "=" * 70)
    print("🎉 GPU 측정모델 테스트 성공!")
    print("=" * 70)

    if not GPU_AVAILABLE:
        print("\n💡 참고: GPU를 사용하려면 CUDA 호환성 문제를 해결하세요.")
        print("   현재는 CPU 모드로 정상 작동합니다.")

except ImportError as e:
    print(f"\n❌ 임포트 실패: {e}")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"\n❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

