"""
완전 GPU Batch 계산 유닛 테스트

326명 × 100 draws × 80 파라미터 = 2,608,000개 gradient를 한 번에 계산하는 테스트
"""

import numpy as np
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy 사용 가능")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ CuPy 사용 불가")
    sys.exit(1)


def generate_dummy_data():
    """
    더미 데이터 생성
    
    Returns:
        all_data: (326, 18, 10) - 326명 × 18행 × 10 features
        all_draws: (326, 100, 6) - 326명 × 100 draws × 6 LVs
        params: (80,) - 80개 파라미터
    """
    print("\n" + "="*80)
    print("더미 데이터 생성")
    print("="*80)
    
    np.random.seed(42)
    
    n_individuals = 326
    n_rows_per_ind = 18
    n_features = 10
    n_draws = 100
    n_lvs = 6
    n_params = 80
    
    # 개인 데이터 (326명 × 18행 × 10 features)
    all_data = np.random.randn(n_individuals, n_rows_per_ind, n_features)
    
    # Draws (326명 × 100 draws × 6 LVs)
    all_draws = np.random.randn(n_individuals, n_draws, n_lvs)
    
    # 파라미터 (80개)
    params = np.random.randn(n_params) * 0.1
    
    print(f"  all_data shape: {all_data.shape}")
    print(f"  all_draws shape: {all_draws.shape}")
    print(f"  params shape: {params.shape}")
    print(f"  총 계산량: {n_individuals} × {n_draws} × {n_params} = {n_individuals * n_draws * n_params:,}개")
    
    return all_data, all_draws, params


def compute_gradients_sequential(all_data, all_draws, params):
    """
    순차 계산 (현재 방식 시뮬레이션)
    
    각 개인마다:
        각 draw마다:
            각 파라미터마다 gradient 계산
    
    Returns:
        individual_grads: (326, 80)
    """
    print("\n" + "="*80)
    print("순차 계산 (Baseline)")
    print("="*80)
    
    n_individuals, n_rows, n_features = all_data.shape
    n_draws = all_draws.shape[1]
    n_params = len(params)
    
    individual_grads = np.zeros((n_individuals, n_params))
    
    start_time = time.time()
    
    for ind_idx in range(n_individuals):
        # 각 개인의 데이터
        ind_data = all_data[ind_idx]  # (18, 10)
        ind_draws = all_draws[ind_idx]  # (100, 6)
        
        # 각 draw의 gradient 계산
        draw_grads = np.zeros((n_draws, n_params))
        
        for draw_idx in range(n_draws):
            draw = ind_draws[draw_idx]  # (6,)
            
            # 간단한 gradient 계산 (더미)
            # 실제로는 측정모델 + 구조모델 + 선택모델 gradient
            grad = np.zeros(n_params)
            
            # 측정모델 gradient (파라미터 0-40)
            for p in range(40):
                grad[p] = np.sum(ind_data[:, 0] * draw[0] * params[p])
            
            # 구조모델 gradient (파라미터 40-70)
            for p in range(40, 70):
                grad[p] = np.sum(draw * params[p])
            
            # 선택모델 gradient (파라미터 70-80)
            for p in range(70, 80):
                grad[p] = np.sum(ind_data[:, 1] * draw[1] * params[p])
            
            draw_grads[draw_idx] = grad
        
        # 가중평균 (균등 가중치)
        individual_grads[ind_idx] = np.mean(draw_grads, axis=0)
        
        # 진행 상황 (10% 단위)
        if (ind_idx + 1) % max(1, n_individuals // 10) == 0:
            progress = (ind_idx + 1) / n_individuals * 100
            elapsed = time.time() - start_time
            print(f"  진행: {ind_idx + 1}/{n_individuals} ({progress:.0f}%) - {elapsed:.2f}초")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n순차 계산 완료:")
    print(f"  총 시간: {elapsed_time:.3f}초")
    print(f"  개인당 시간: {elapsed_time / n_individuals * 1000:.2f}ms")
    print(f"  처리량: {n_individuals / elapsed_time:.1f} 개인/초")
    
    return individual_grads, elapsed_time


def compute_gradients_full_gpu_batch(all_data, all_draws, params):
    """
    완전 GPU Batch 계산
    
    326명 × 100 draws × 80 파라미터 = 2,608,000개를 한 번에 계산
    
    Returns:
        individual_grads: (326, 80)
    """
    print("\n" + "="*80)
    print("완전 GPU Batch 계산")
    print("="*80)
    
    n_individuals, n_rows, n_features = all_data.shape
    n_draws = all_draws.shape[1]
    n_params = len(params)
    
    start_time = time.time()
    
    # GPU로 데이터 전송
    print("  GPU로 데이터 전송 중...")
    transfer_start = time.time()
    
    all_data_gpu = cp.asarray(all_data)      # (326, 18, 10)
    all_draws_gpu = cp.asarray(all_draws)    # (326, 100, 6)
    params_gpu = cp.asarray(params)          # (80,)
    
    transfer_time = time.time() - transfer_start
    print(f"  데이터 전송 완료: {transfer_time:.3f}초")
    
    # GPU Batch 계산
    print("  GPU Batch gradient 계산 중...")
    compute_start = time.time()
    
    # 모든 개인 × 모든 draws의 gradient 계산
    # Shape: (326, 100, 80)
    all_grads_gpu = cp.zeros((n_individuals, n_draws, n_params))
    
    # 측정모델 gradient (파라미터 0-40)
    # (326, 18, 1) × (326, 1, 100, 1) × (40,) → (326, 100, 40)
    for p in range(40):
        # (326, 18, 1) × (326, 1, 100, 1) → (326, 18, 100)
        # Broadcasting: data[:, :, None, :] × draws[:, None, :, :]
        data_expanded = all_data_gpu[:, :, None, 0]  # (326, 18, 1)
        draws_expanded = all_draws_gpu[:, None, :, 0]  # (326, 1, 100)

        # (326, 18, 100) → sum over rows → (326, 100)
        grad_p = cp.sum(data_expanded * draws_expanded * params_gpu[p], axis=1)
        all_grads_gpu[:, :, p] = grad_p
    
    # 구조모델 gradient (파라미터 40-70)
    # (326, 100, 6) × (30,) → (326, 100, 30)
    for p in range(40, 70):
        grad_p = cp.sum(all_draws_gpu * params_gpu[p], axis=2)
        all_grads_gpu[:, :, p] = grad_p
    
    # 선택모델 gradient (파라미터 70-80)
    # (326, 18, 1) × (326, 1, 100, 1) × (10,) → (326, 100, 10)
    for p in range(70, 80):
        # Broadcasting
        data_expanded = all_data_gpu[:, :, None, 1]  # (326, 18, 1)
        draws_expanded = all_draws_gpu[:, None, :, 1]  # (326, 1, 100)

        # (326, 18, 100) → sum over rows → (326, 100)
        grad_p = cp.sum(data_expanded * draws_expanded * params_gpu[p], axis=1)
        all_grads_gpu[:, :, p] = grad_p
    
    compute_time = time.time() - compute_start
    print(f"  GPU 계산 완료: {compute_time:.3f}초")
    
    # 개인별 가중평균 (GPU reduction)
    print("  개인별 가중평균 계산 중...")
    reduction_start = time.time()
    
    # 균등 가중치로 평균
    individual_grads_gpu = cp.mean(all_grads_gpu, axis=1)  # (326, 80)
    
    reduction_time = time.time() - reduction_start
    print(f"  가중평균 완료: {reduction_time:.3f}초")
    
    # CPU로 결과 전송
    print("  CPU로 결과 전송 중...")
    transfer_back_start = time.time()
    
    individual_grads = cp.asnumpy(individual_grads_gpu)
    
    transfer_back_time = time.time() - transfer_back_start
    print(f"  결과 전송 완료: {transfer_back_time:.3f}초")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n완전 GPU Batch 계산 완료:")
    print(f"  총 시간: {elapsed_time:.3f}초")
    print(f"    - 데이터 전송 (GPU): {transfer_time:.3f}초")
    print(f"    - GPU 계산: {compute_time:.3f}초")
    print(f"    - 가중평균: {reduction_time:.3f}초")
    print(f"    - 결과 전송 (CPU): {transfer_back_time:.3f}초")
    print(f"  개인당 시간: {elapsed_time / n_individuals * 1000:.2f}ms")
    print(f"  처리량: {n_individuals / elapsed_time:.1f} 개인/초")
    
    return individual_grads, elapsed_time


def main():
    """메인 테스트 함수"""
    print("\n" + "="*80)
    print("완전 GPU Batch 계산 유닛 테스트")
    print("="*80)
    print("목표: 326명 × 100 draws × 80 파라미터 = 2,608,000개 gradient 동시 계산")
    print("="*80)
    
    # 1. 더미 데이터 생성
    all_data, all_draws, params = generate_dummy_data()
    
    # 2. 순차 계산 (Baseline)
    grads_seq, time_seq = compute_gradients_sequential(all_data, all_draws, params)
    
    # 3. 완전 GPU Batch 계산
    grads_gpu, time_gpu = compute_gradients_full_gpu_batch(all_data, all_draws, params)
    
    # 4. 결과 비교
    print("\n" + "="*80)
    print("결과 비교")
    print("="*80)
    
    # 수치 비교
    diff = np.abs(grads_seq - grads_gpu)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  최대 차이: {max_diff:.6e}")
    print(f"  평균 차이: {mean_diff:.6e}")
    
    if max_diff < 1e-5:
        print("  ✅ 수치 일치 (오차 < 1e-5)")
    else:
        print("  ⚠️ 수치 불일치 (오차 >= 1e-5)")
    
    # 성능 비교
    print("\n" + "="*80)
    print("성능 비교")
    print("="*80)
    
    speedup = time_seq / time_gpu
    
    print(f"  순차 계산: {time_seq:.3f}초")
    print(f"  GPU Batch: {time_gpu:.3f}초")
    print(f"  가속 비율: {speedup:.1f}배")
    
    if speedup > 10:
        print(f"  ✅ 목표 달성! ({speedup:.1f}배 > 10배)")
    else:
        print(f"  ⚠️ 목표 미달 ({speedup:.1f}배 < 10배)")
    
    print("\n" + "="*80)
    print("테스트 완료")
    print("="*80)


if __name__ == "__main__":
    main()

