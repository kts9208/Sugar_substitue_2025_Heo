"""
CUDA 13.0 + CuPy 호환성 테스트
"""

import os
import sys

# CUDA 경로 설정
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
cuda_bin = os.path.join(os.environ['CUDA_PATH'], 'bin')
if cuda_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')

print("=" * 70)
print("CUDA 13.0 + CuPy 호환성 테스트")
print("=" * 70)

try:
    import cupy as cp
    print(f"\n✅ CuPy 버전: {cp.__version__}")
    
    # CUDA 사용 가능 여부
    cuda_available = cp.cuda.is_available()
    print(f"✅ CUDA 사용 가능: {cuda_available}")
    
    if cuda_available:
        # GPU 정보
        cp.cuda.Device(0).use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"✅ GPU: {props['name'].decode()}")
        print(f"✅ GPU 메모리: {props['totalGlobalMem'] / 1024**3:.2f} GB")
        
        # 간단한 배열 연산
        print("\n" + "-" * 70)
        print("GPU 배열 연산 테스트")
        print("-" * 70)
        a = cp.array([1, 2, 3, 4, 5])
        b = cp.array([10, 20, 30, 40, 50])
        c = a + b
        print(f"a = {a}")
        print(f"b = {b}")
        print(f"c = a + b = {c}")
        print("✅ GPU 배열 연산 성공!")
        
        # 정규분포 CDF 테스트
        print("\n" + "-" * 70)
        print("GPU 정규분포 CDF 테스트")
        print("-" * 70)
        from cupyx.scipy.special import ndtr
        x = cp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        cdf = ndtr(x)
        print(f"x = {x}")
        print(f"Φ(x) = {cdf}")
        print("✅ GPU 정규분포 CDF 성공!")
        
        # 행렬 연산 테스트
        print("\n" + "-" * 70)
        print("GPU 행렬 연산 테스트")
        print("-" * 70)
        A = cp.random.randn(100, 100)
        B = cp.random.randn(100, 100)
        C = cp.dot(A, B)
        print(f"A shape: {A.shape}")
        print(f"B shape: {B.shape}")
        print(f"C = A @ B shape: {C.shape}")
        print(f"C mean: {cp.mean(C):.4f}")
        print("✅ GPU 행렬 연산 성공!")
        
        print("\n" + "=" * 70)
        print("🎉 모든 GPU 테스트 성공!")
        print("=" * 70)
        print("\n✅ GPU를 사용한 측정모델 계산이 가능합니다!")
        
    else:
        print("\n❌ CUDA를 사용할 수 없습니다.")
        sys.exit(1)
        
except ImportError as e:
    print(f"\n❌ CuPy 임포트 실패: {e}")
    print("\nCuPy를 설치하세요:")
    print("  pip install cupy-cuda12x")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ GPU 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

