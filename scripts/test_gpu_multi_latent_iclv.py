"""
GPU 가속 다중 잠재변수 ICLV 모델 테스트

CuPy를 사용하여 GPU에서 모델을 추정합니다.
"""

import sys
from pathlib import Path
import pandas as pd
import multiprocessing

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_default_multi_lv_config
from src.analysis.hybrid_choice_model.iclv_models.gpu_multi_latent_estimator import GPUMultiLatentSimultaneousEstimator
from src.analysis.hybrid_choice_model.iclv_models.gpu_measurement_equations import GPU_AVAILABLE


def main():
    print("=" * 70)
    print("🚀 GPU 가속 다중 잠재변수 ICLV 동시추정 (5개 잠재변수)")
    print("=" * 70)
    
    # GPU 사용 가능 여부 확인
    if GPU_AVAILABLE:
        print("\n✅ GPU 사용 가능!")
        try:
            import cupy as cp
            print(f"   CuPy 버전: {cp.__version__}")
            print(f"   CUDA 사용 가능: {cp.cuda.is_available()}")
            if cp.cuda.is_available():
                print(f"   GPU 개수: {cp.cuda.runtime.getDeviceCount()}")
                cp.cuda.Device(0).use()
                props = cp.cuda.runtime.getDeviceProperties(0)
                print(f"   GPU 이름: {props['name'].decode()}")
                print(f"   GPU 메모리: {props['totalGlobalMem'] / 1024**3:.2f} GB")
        except Exception as e:
            print(f"   ⚠️ GPU 정보 확인 중 오류: {e}")
    else:
        print("\n⚠️ CuPy 미설치 - CPU 모드로 작동")
    
    # 1. 데이터 로드
    print("\n" + "=" * 70)
    print("1. 데이터 로드")
    print("=" * 70)
    
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    print(f"데이터 경로: {data_path}")
    
    data = pd.read_csv(data_path)
    n_individuals = data['respondent_id'].nunique()
    
    print(f"✓ 데이터 로드 완료")
    print(f"  - 개인 수: {n_individuals:,}")
    print(f"  - 관측치 수: {len(data):,}")
    print(f"  - 개인당 선택 상황: {len(data) / n_individuals:.1f}")
    
    # 2. 설정
    print("\n" + "=" * 70)
    print("2. 모델 설정")
    print("=" * 70)
    
    n_cpus = multiprocessing.cpu_count()
    use_parallel = True
    n_cores = max(1, n_cpus - 1)
    
    print(f"CPU 코어: {n_cpus}개")
    print(f"병렬처리: {use_parallel}")
    print(f"사용 코어: {n_cores}개")
    
    # GPU 사용 여부 선택
    use_gpu = GPU_AVAILABLE
    if use_gpu:
        print(f"GPU 가속: ✅ 활성화")
    else:
        print(f"GPU 가속: ❌ 비활성화 (CuPy 미설치)")
    
    config = create_default_multi_lv_config(
        n_draws=100,
        max_iterations=1000,
        use_parallel=use_parallel,
        n_cores=n_cores
    )
    
    print(f"✓ 설정 완료")
    print(f"  - Halton draws: {config.estimation.n_draws}")
    print(f"  - 최대 반복: {config.estimation.max_iterations}")
    print(f"  - 최적화 방법: {config.estimation.optimizer}")
    
    # 3. 모델 생성
    print("\n" + "=" * 70)
    print("3. GPU 모델 생성")
    print("=" * 70)
    
    estimator = GPUMultiLatentSimultaneousEstimator(config, data, use_gpu=use_gpu)
    
    print(f"✓ 모델 생성 완료")
    
    # 4. 추정
    print("\n" + "=" * 70)
    print("4. 모델 추정 시작")
    print("=" * 70)
    
    results = estimator.estimate()
    
    # 5. 결과 출력
    print("\n" + "=" * 70)
    print("5. 추정 결과")
    print("=" * 70)
    
    print(f"\n최종 로그우도: {results['log_likelihood']:.4f}")
    print(f"수렴 여부: {results['success']}")
    print(f"반복 횟수: {results['n_iterations']}")
    print(f"소요 시간: {results['time_elapsed']/60:.1f}분")
    
    # 파라미터 요약
    print("\n" + "-" * 70)
    print("파라미터 요약")
    print("-" * 70)
    
    params = results['params']
    
    # 구조모델 파라미터
    print("\n[구조모델]")
    print(f"  gamma_lv (잠재변수 계수): {params['structural']['gamma_lv']}")
    print(f"  gamma_x (공변량 계수): {params['structural']['gamma_x']}")
    
    # 선택모델 파라미터
    print("\n[선택모델]")
    print(f"  intercept: {params['choice']['intercept']:.4f}")
    print(f"  beta (속성 계수): {params['choice']['beta']}")
    print(f"  lambda (잠재변수 계수): {params['choice']['lambda']:.4f}")
    
    # 측정모델 파라미터 (요약)
    print("\n[측정모델]")
    for lv_name, lv_params in params['measurement'].items():
        print(f"  {lv_name}:")
        print(f"    zeta (요인적재량): {lv_params['zeta'][:3]}... (처음 3개)")
        print(f"    tau (임계값): {lv_params['tau'].shape}")
    
    print("\n" + "=" * 70)
    print("✅ 추정 완료!")
    print("=" * 70)
    
    # GPU vs CPU 성능 비교 정보
    if use_gpu:
        print("\n💡 GPU 가속이 적용되었습니다.")
        print("   측정모델의 정규분포 CDF 계산이 GPU에서 수행되었습니다.")
    else:
        print("\n💡 CPU 모드로 실행되었습니다.")
        print("   GPU 가속을 사용하려면 CuPy를 설치하세요:")
        print("   pip install cupy-cuda12x")


if __name__ == '__main__':
    main()

