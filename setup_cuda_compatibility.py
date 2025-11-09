"""
CUDA 13.0과 CuPy-CUDA12x 호환성 설정

CUDA 13.0 DLL을 CUDA 12.0 이름으로 심볼릭 링크 생성
"""

import os
import sys
import shutil
from pathlib import Path

def setup_cuda_compatibility():
    """CUDA 13.0 DLL을 CUDA 12.0 이름으로 복사"""
    
    print("=" * 70)
    print("CUDA 호환성 설정")
    print("=" * 70)
    
    # CUDA 경로
    cuda_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
    cuda_bin = cuda_path / "bin"
    
    if not cuda_bin.exists():
        print(f"❌ CUDA 경로를 찾을 수 없습니다: {cuda_bin}")
        return False
    
    print(f"✅ CUDA 경로: {cuda_path}")
    
    # DLL 매핑 (CUDA 13.0 -> CUDA 12.0 이름)
    dll_mappings = {
        # CUDA 13.0 파일 -> CUDA 12.0 이름
        'nvrtc64_130_0.dll': 'nvrtc64_120_0.dll',
        'cublas64_13.dll': 'cublas64_12.dll',
        'cublasLt64_13.dll': 'cublasLt64_12.dll',
        'cufft64_11.dll': 'cufft64_11.dll',  # 동일
        'cufftw64_11.dll': 'cufftw64_11.dll',  # 동일
        'curand64_10.dll': 'curand64_10.dll',  # 동일
        'cusolver64_11.dll': 'cusolver64_11.dll',  # 동일
        'cusolverMg64_11.dll': 'cusolverMg64_11.dll',  # 동일
        'cusparse64_13.dll': 'cusparse64_12.dll',
        'cudnn64_9.dll': 'cudnn64_8.dll',
        'nvJitLink_130_0.dll': 'nvJitLink_120_0.dll',
    }

    # x64 서브디렉토리의 DLL도 bin으로 복사
    x64_dlls_to_copy = [
        'cublas64_13.dll',
        'cublasLt64_13.dll',
        'cufft64_11.dll',
        'curand64_10.dll',
        'cusolver64_11.dll',
        'cusparse64_13.dll',
    ]
    
    # x64 서브디렉토리도 확인
    cuda_bin_x64 = cuda_bin / "x64"

    copied_count = 0
    failed_count = 0

    # 1. x64 DLL을 bin으로 복사
    print("\n[1단계] x64 DLL을 bin으로 복사")
    print("-" * 70)
    if cuda_bin_x64.exists():
        for dll_name in x64_dlls_to_copy:
            src_file = cuda_bin_x64 / dll_name
            dst_file = cuda_bin / dll_name

            if not src_file.exists():
                continue

            if dst_file.exists():
                print(f"✓ 이미 존재: {dll_name}")
                continue

            try:
                shutil.copy2(src_file, dst_file)
                print(f"✅ 복사 성공: x64/{dll_name} -> bin/{dll_name}")
                copied_count += 1
            except Exception as e:
                print(f"❌ 복사 실패: {dll_name}: {e}")
                failed_count += 1

    # 2. CUDA 13.0 -> 12.0 이름 변경
    print("\n[2단계] CUDA 13.0 DLL을 12.0 이름으로 복사")
    print("-" * 70)
    for src_name, dst_name in dll_mappings.items():
        # 소스 파일 찾기
        src_file = None
        if (cuda_bin / src_name).exists():
            src_file = cuda_bin / src_name
        elif cuda_bin_x64.exists() and (cuda_bin_x64 / src_name).exists():
            src_file = cuda_bin_x64 / src_name
        
        if src_file is None:
            print(f"⚠️ 소스 파일 없음: {src_name}")
            continue
        
        # 대상 파일 경로
        dst_file = cuda_bin / dst_name
        
        # 이미 존재하면 건너뛰기
        if dst_file.exists():
            print(f"✓ 이미 존재: {dst_name}")
            copied_count += 1
            continue
        
        # 복사
        try:
            shutil.copy2(src_file, dst_file)
            print(f"✅ 복사 성공: {src_name} -> {dst_name}")
            copied_count += 1
        except Exception as e:
            print(f"❌ 복사 실패: {src_name} -> {dst_name}: {e}")
            failed_count += 1
    
    print("\n" + "=" * 70)
    print(f"복사 완료: {copied_count}개 성공, {failed_count}개 실패")
    print("=" * 70)
    
    # 환경변수 설정
    print("\n환경변수 설정:")
    os.environ['CUDA_PATH'] = str(cuda_path)
    os.environ['PATH'] = str(cuda_bin) + os.pathsep + os.environ.get('PATH', '')
    print(f"  CUDA_PATH = {cuda_path}")
    print(f"  PATH에 추가 = {cuda_bin}")
    
    return copied_count > 0


def test_cupy():
    """CuPy 테스트"""
    print("\n" + "=" * 70)
    print("CuPy 테스트")
    print("=" * 70)
    
    try:
        import cupy as cp
        print(f"✅ CuPy 버전: {cp.__version__}")
        
        # CUDA 사용 가능 여부
        cuda_available = cp.cuda.is_available()
        print(f"✅ CUDA 사용 가능: {cuda_available}")
        
        if cuda_available:
            # GPU 정보
            cp.cuda.Device(0).use()
            props = cp.cuda.runtime.getDeviceProperties(0)
            print(f"✅ GPU: {props['name'].decode()}")
            print(f"✅ GPU 메모리: {props['totalGlobalMem'] / 1024**3:.2f} GB")
            
            # 간단한 연산
            a = cp.array([1, 2, 3, 4, 5])
            b = cp.array([10, 20, 30, 40, 50])
            c = a + b
            print(f"✅ GPU 배열 연산: {c}")
            
            # 정규분포 CDF
            from cupyx.scipy.special import ndtr
            x = cp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            cdf = ndtr(x)
            print(f"✅ GPU 정규분포 CDF: {cdf}")
            
            print("\n" + "=" * 70)
            print("🎉 GPU 테스트 성공!")
            print("=" * 70)
            return True
        else:
            print("❌ CUDA를 사용할 수 없습니다.")
            return False
            
    except ImportError as e:
        print(f"❌ CuPy 임포트 실패: {e}")
        print("\nCuPy를 설치하세요:")
        print("  pip install cupy-cuda12x")
        return False
        
    except Exception as e:
        print(f"❌ GPU 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 관리자 권한 확인
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False
    
    if not is_admin:
        print("⚠️ 경고: 관리자 권한이 없습니다.")
        print("   DLL 복사가 실패할 수 있습니다.")
        print("   관리자 권한으로 실행하려면:")
        print("   1. PowerShell을 관리자 권한으로 실행")
        print("   2. python setup_cuda_compatibility.py")
        print("\n계속하시겠습니까? (y/n): ", end='')
        
        response = input().strip().lower()
        if response != 'y':
            print("중단되었습니다.")
            sys.exit(0)
    
    # CUDA 호환성 설정
    success = setup_cuda_compatibility()
    
    if success:
        print("\n✅ CUDA 호환성 설정 완료!")
        print("\n다음 단계:")
        print("  1. CuPy 설치 (아직 안 했다면):")
        print("     pip install cupy-cuda12x")
        print("  2. GPU 테스트:")
        print("     python test_gpu_simple.py")
        
        # CuPy 테스트
        print("\nCuPy가 설치되어 있다면 테스트를 진행합니다...")
        test_cupy()
    else:
        print("\n❌ CUDA 호환성 설정 실패")

