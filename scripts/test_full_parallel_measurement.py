"""
완전 병렬 측정모델 Gradient 계산 테스트

Advanced Indexing을 사용하여 38개 지표를 1번의 GPU 커널 호출로 계산
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy 사용 가능")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ CuPy 미설치")
    sys.exit(1)

from src.analysis.hybrid_choice_model.iclv_models.gpu_gradient_full_parallel import (
    compute_measurement_full_parallel_gpu
)
from src.analysis.hybrid_choice_model.iclv_models.gpu_measurement_equations import (
    GPUMultiLatentMeasurement
)
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import MeasurementConfig

print("\n" + "="*80)
print("완전 병렬 측정모델 Gradient 계산 테스트")
print("="*80)

# 1. 측정모델 설정 (5개 LV, 38개 지표)
print("\n1. 측정모델 설정...")

measurement_configs = {
    'health_concern': MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        measurement_method='continuous_linear',
        n_categories=None,
        fix_first_loading=True
    ),
    'perceived_benefit': MeasurementConfig(
        latent_variable='perceived_benefit',
        indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
        measurement_method='continuous_linear',
        n_categories=None,
        fix_first_loading=True
    ),
    'perceived_price': MeasurementConfig(
        latent_variable='perceived_price',
        indicators=['q27', 'q28', 'q29'],
        measurement_method='continuous_linear',
        n_categories=None,
        fix_first_loading=True
    ),
    'nutrition_knowledge': MeasurementConfig(
        latent_variable='nutrition_knowledge',
        indicators=[f'q{i}' for i in range(30, 50)],  # q30-q49 (20개)
        measurement_method='continuous_linear',
        n_categories=None,
        fix_first_loading=True
    ),
    'purchase_intention': MeasurementConfig(
        latent_variable='purchase_intention',
        indicators=['q18', 'q19', 'q20'],
        measurement_method='continuous_linear',
        n_categories=None,
        fix_first_loading=True
    )
}

lv_names = list(measurement_configs.keys())
n_indicators_per_lv = {lv: len(cfg.indicators) for lv, cfg in measurement_configs.items()}
total_indicators = sum(n_indicators_per_lv.values())

print(f"   - LV 수: {len(lv_names)}")
print(f"   - 총 지표 수: {total_indicators}")
print(f"   - LV별 지표 수: {n_indicators_per_lv}")

# 2. GPU 측정모델 생성
print("\n2. GPU 측정모델 생성...")
gpu_measurement_model = GPUMultiLatentMeasurement(measurement_configs, use_gpu=True)
print("   - GPU 측정모델 생성 완료")

# 3. 테스트 데이터 생성
print("\n3. 테스트 데이터 생성...")

n_individuals = 326
n_draws = 100
n_lvs = 5

# 모든 개인의 데이터 생성
all_ind_data = []
for i in range(n_individuals):
    # 각 개인의 지표 데이터
    ind_dict = {'respondent_id': i}
    
    # 모든 지표에 대한 관측값 (랜덤)
    for lv_name, config in measurement_configs.items():
        for indicator in config.indicators:
            ind_dict[indicator] = np.random.randn() * 2 + 3  # 평균 3, 표준편차 2
    
    all_ind_data.append(pd.DataFrame([ind_dict]))

# LV 값 생성 (326, 100, 5)
all_lvs = np.random.randn(n_individuals, n_draws, n_lvs)
all_lvs_gpu = cp.asarray(all_lvs)

# 가중치 (균등)
all_weights = np.ones((n_individuals, n_draws)) / n_draws
all_weights_gpu = cp.asarray(all_weights)

print(f"   - 개인 수: {n_individuals}")
print(f"   - Draws: {n_draws}")
print(f"   - LV 수: {n_lvs}")
print(f"   - all_lvs shape: {all_lvs.shape}")

# 4. 파라미터 설정
print("\n4. 파라미터 설정...")

params_dict = {'measurement': {}}

for lv_name, config in measurement_configs.items():
    n_ind = len(config.indicators)
    params_dict['measurement'][lv_name] = {
        'zeta': np.ones(n_ind),  # 모두 1.0
        'sigma_sq': np.ones(n_ind)  # 모두 1.0
    }

print("   - 파라미터 초기화 완료")

# 5. ✨ 완전 병렬 Gradient 계산
print("\n" + "="*80)
print("5. ✨ 완전 병렬 Gradient 계산 (Advanced Indexing)")
print("="*80)

start_time = time.time()

gradients = compute_measurement_full_parallel_gpu(
    gpu_measurement_model,
    all_ind_data,
    all_lvs_gpu,
    params_dict,
    all_weights_gpu,
    lv_names,
    iteration_logger=None,
    log_level='DETAILED'
)

elapsed = time.time() - start_time

print(f"\n✅ 완전 병렬 Gradient 계산 완료!")
print(f"   - 소요 시간: {elapsed:.4f}초")
print(f"   - 처리량: {n_individuals * n_draws * total_indicators / elapsed:,.0f} 계산/초")

# 6. 결과 검증
print("\n" + "="*80)
print("6. 결과 검증")
print("="*80)

for lv_name in lv_names:
    grad_zeta = gradients[lv_name]['grad_zeta']
    grad_sigma_sq = gradients[lv_name]['grad_sigma_sq']
    
    print(f"\n{lv_name}:")
    print(f"   - grad_zeta shape: {grad_zeta.shape}")
    print(f"   - grad_sigma_sq shape: {grad_sigma_sq.shape}")
    print(f"   - grad_zeta 범위: [{grad_zeta.min():.4f}, {grad_zeta.max():.4f}]")
    print(f"   - grad_sigma_sq 범위: [{grad_sigma_sq.min():.4f}, {grad_sigma_sq.max():.4f}]")

# 7. 성능 분석
print("\n" + "="*80)
print("7. 성능 분석")
print("="*80)

print(f"\n계산량:")
print(f"   - 개인 수: {n_individuals}")
print(f"   - Draws: {n_draws}")
print(f"   - 지표 수: {total_indicators}")
print(f"   - 총 계산: {n_individuals * n_draws * total_indicators:,}개")

print(f"\nGPU 커널 호출:")
print(f"   - 기존 (지표별 순차): {total_indicators}번")
print(f"   - 제안 (LV별 순차): {len(lv_names)}번")
print(f"   - 완전 병렬 (Advanced Indexing): 1번 ✅")

print(f"\n예상 속도 향상:")
print(f"   - 기존 대비: {total_indicators}배")
print(f"   - 제안 대비: {len(lv_names)}배")

print(f"\n메모리 사용:")
memory_mb = n_individuals * n_draws * total_indicators * 8 / 1024 / 1024
print(f"   - 완전 병렬: {memory_mb:.2f} MB")

zero_padding_mb = n_individuals * n_draws * len(lv_names) * 20 * 8 / 1024 / 1024
print(f"   - Zero-padding: {zero_padding_mb:.2f} MB")
print(f"   - 절약: {zero_padding_mb - memory_mb:.2f} MB ({(1 - memory_mb/zero_padding_mb)*100:.1f}%)")

print("\n" + "="*80)
print("✅ 테스트 완료!")
print("="*80)

