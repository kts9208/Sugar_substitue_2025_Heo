"""
GPU 경로 통합 테스트: 실제 함수 호출로 검증
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("통합 테스트: GPU 경로에서 실제 함수 호출")
print("="*80)

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'examples'))

print(f"\n프로젝트 루트: {project_root}")

# CuPy 사용 가능 여부 확인
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy 사용 가능")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ CuPy 사용 불가 - GPU 테스트 스킵")
    sys.exit(0)

# 필요한 모듈 import
from src.analysis.hybrid_choice_model.iclv_models.gpu_gradient_batch import (
    compute_structural_gradient_batch_gpu,
    compute_choice_gradient_batch_gpu
)
from src.analysis.hybrid_choice_model.iclv_models.gpu_measurement_equations import GPUMultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import MeasurementConfig

print("✅ 모듈 import 성공")

def create_test_data():
    """테스트용 데이터 생성

    Note:
        compute_structural_gradient_batch_gpu()는 **단일 개인**의 데이터를 처리합니다.
        - ind_data: 단일 개인 (1행)
        - lvs_list: 각 draw의 LV 값 (길이 n_draws)
        - exo_draws_list: 각 draw의 외생 draws (길이 n_draws)
        - weights: draw별 가중치 (n_draws,)
    """
    print("\n[1] 테스트 데이터 생성")

    n_draws = 100  # Halton draws 수

    # ✅ 단일 개인 데이터 (1행)
    ind_data = pd.DataFrame({
        'respondent_id': [0],
        'age_std': [0.5],
        'gender': [1],
        'income_std': [-0.3]
    })

    # ✅ 잠재변수 점수: 각 draw별 (길이 n_draws)
    lvs_list = []
    for r in range(n_draws):
        lvs_list.append({
            'health_concern': np.random.randn(),
            'perceived_benefit': np.random.randn(),
            'purchase_intention': np.random.randn()
        })

    # ✅ 외생변수 draws: 각 draw별 (길이 n_draws)
    exo_draws_list = [np.random.randn() for _ in range(n_draws)]

    # ✅ 가중치: (n_draws,) shape - importance weights
    weights = np.ones(n_draws) / n_draws  # 정규화된 가중치

    print(f"  개인 수: 1 (단일 개인)")
    print(f"  Draw 수: {n_draws}")
    print(f"  lvs_list 길이: {len(lvs_list)}")
    print(f"  exo_draws_list 길이: {len(exo_draws_list)}")
    print(f"  가중치 shape: {weights.shape}")

    return ind_data, lvs_list, exo_draws_list, weights

def create_test_params():
    """테스트용 파라미터 생성"""
    print("\n[2] 테스트 파라미터 생성")
    
    params_dict = {
        'measurement': {
            'health_concern': {
                'zeta': np.array([1.0, 0.8, 0.9]),
                'sigma_sq': np.array([0.5, 0.6, 0.4]),
                'alpha': None
            },
            'perceived_benefit': {
                'zeta': np.array([1.0, 0.85, 0.75]),
                'sigma_sq': np.array([0.4, 0.5, 0.6]),
                'alpha': None
            },
            'purchase_intention': {
                'zeta': np.array([1.0, 0.95, 0.90]),
                'sigma_sq': np.array([0.3, 0.4, 0.5]),
                'alpha': None
            }
        },
        'structural': {
            'gamma_health_concern_to_perceived_benefit': 0.5,
            'gamma_perceived_benefit_to_purchase_intention': 0.3
        },
        'choice': {
            'asc_sugar': 1.45,
            'asc_sugar_free': 2.44,
            'beta_health_label': 0.50,
            'beta_price': -0.56,
            'theta_sugar_nutrition_knowledge': -0.02,
            'theta_sugar_free_nutrition_knowledge': -0.01,
            'theta_sugar_purchase_intention': -0.02,
            'theta_sugar_free_purchase_intention': -0.01,
            'theta_sugar_perceived_price': -0.02,
            'theta_sugar_free_perceived_price': -0.01
        }
    }

    print(f"  측정모델 LV 수: {len(params_dict['measurement'])}")
    print(f"  구조모델 경로 수: {len(params_dict['structural'])}")
    print(f"  선택모델 파라미터 수: {len(params_dict['choice'])}")

    return params_dict

def create_gpu_measurement_model(params_dict):
    """GPU 측정모델 생성"""
    print("\n[3] GPU 측정모델 생성")

    # 측정모델 설정
    configs = {}
    for lv_name in ['health_concern', 'perceived_benefit', 'purchase_intention']:
        n_indicators = len(params_dict['measurement'][lv_name]['zeta'])
        configs[lv_name] = MeasurementConfig(
            latent_variable=lv_name,
            indicators=[f'q{i}' for i in range(n_indicators)],
            measurement_method='ordered_probit',
            n_categories=5  # ✅ 5점 척도
        )

    # ✅ GPUMultiLatentMeasurement 사용
    gpu_measurement_model = GPUMultiLatentMeasurement(configs, use_gpu=True)

    print(f"  ✅ GPU 측정모델 생성 완료")

    return gpu_measurement_model

def test_compute_structural_gradient():
    """compute_structural_gradient_batch_gpu() 통합 테스트"""
    print("\n" + "="*80)
    print("TEST: compute_structural_gradient_batch_gpu() 통합 테스트")
    print("="*80)
    
    try:
        # 데이터 생성
        ind_data, lvs_list, exo_draws_list, weights = create_test_data()
        params_dict = create_test_params()
        gpu_measurement_model = create_gpu_measurement_model(params_dict)
        
        # 계층적 경로 정의
        hierarchical_paths = [
            {
                'target': 'perceived_benefit',
                'predictors': ['health_concern']
            },
            {
                'target': 'purchase_intention',
                'predictors': ['perceived_benefit']
            }
        ]
        
        print("\n[4] compute_structural_gradient_batch_gpu() 호출")
        print(f"  경로 수: {len(hierarchical_paths)}")
        
        # ✅ 실제 함수 호출
        grad_struct = compute_structural_gradient_batch_gpu(
            ind_data=ind_data,
            lvs_list=lvs_list,
            exo_draws_list=exo_draws_list,
            params=params_dict,  # ✅ 전체 파라미터 딕셔너리 전달
            covariates=['age_std', 'gender', 'income_std'],
            endogenous_lv='purchase_intention',
            exogenous_lvs=['health_concern'],
            weights=weights,
            is_hierarchical=True,
            hierarchical_paths=hierarchical_paths,
            gpu_measurement_model=gpu_measurement_model,  # ✅ GPU 측정모델 전달
            choice_model=None,  # 선택모델은 None (구조모델만 테스트)
            iteration_logger=None,
            log_level='MINIMAL'
        )
        
        print(f"\n[5] 결과 확인")
        print(f"  Gradient 딕셔너리 키: {list(grad_struct.keys())}")
        
        for key, value in grad_struct.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, mean={value.mean():.6f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print("✅ 통합 테스트 성공!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_choice_data():
    """선택모델 테스트용 데이터 생성"""
    # 3개 선택 상황 (각 개인이 3번 선택)
    choice_data = pd.DataFrame({
        'respondent_id': [0, 0, 0],
        'choice_situation': [0, 1, 2],
        'choice': [1, 2, 1],  # ✅ 'choice' 컬럼 사용 (sugar=1, sugar_free=2, opt-out=3)
        'health_label': [0, 1, 1],
        'price': [1.5, 2.0, 1.8]
    })
    return choice_data


def test_compute_choice_gradient():
    """compute_choice_gradient_batch_gpu() 통합 테스트"""
    print("\n" + "="*80)
    print("TEST 2: compute_choice_gradient_batch_gpu() 통합 테스트")
    print("="*80)

    try:
        # 선택 데이터 생성
        choice_data = create_choice_data()
        print(f"\n[1] 선택 데이터 생성")
        print(f"  선택 상황 수: {len(choice_data)}")

        # LV 값 생성
        n_draws = 100
        lvs_list = []
        for r in range(n_draws):
            lvs_list.append({
                'nutrition_knowledge': np.random.randn(),
                'purchase_intention': np.random.randn(),
                'perceived_price': np.random.randn()
            })

        # 파라미터 생성
        params_dict = create_test_params()

        # 가중치
        weights = np.ones(n_draws) / n_draws

        print(f"\n[2] compute_choice_gradient_batch_gpu() 호출")
        print(f"  선택모델 파라미터: {list(params_dict['choice'].keys())}")

        # compute_choice_gradient_batch_gpu() 호출
        grad_choice = compute_choice_gradient_batch_gpu(
            choice_data,
            lvs_list,
            params_dict['choice'],  # ✅ choice 파라미터만 전달
            endogenous_lv='purchase_intention',
            choice_attributes=['health_label', 'price'],
            weights=weights
        )

        # 결과 확인
        print(f"\n[3] 결과 확인")
        print(f"  Gradient 딕셔너리 키: {list(grad_choice.keys())}")
        for key, value in grad_choice.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, mean={value.mean():.6f}")
            else:
                print(f"  {key}: {value}")

        print("\n" + "="*80)
        print("✅ TEST 2 성공!")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n❌ TEST 2 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_compute_structural_gradient()
    success2 = test_compute_choice_gradient()

    if success1 and success2:
        print("\n✅ 모든 GPU 경로 통합 테스트 성공")
        sys.exit(0)
    else:
        print("\n❌ GPU 경로 통합 테스트 실패")
        sys.exit(1)

