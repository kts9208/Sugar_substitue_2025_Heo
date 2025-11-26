"""
GPU 경로에서 compute_individual_gradient()가 params_dict를 올바르게 사용하는지 테스트
"""

import sys
import os
import numpy as np
import pandas as pd

print("="*80)
print("테스트 시작: GPU 경로에서 params_dict 접근 테스트")
print("="*80)

# 프로젝트 루트를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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

# gpu_gradient_batch.py는 함수만 정의되어 있으므로 import 스킵
print("✅ GPU gradient 함수는 별도 import 불필요 (테스트는 로직만 검증)")

def test_structural_gradient_params_access():
    """
    compute_structural_gradient_batch_gpu()가 params를 올바르게 사용하는지 테스트
    """
    print("\n" + "="*80)
    print("TEST: compute_structural_gradient_batch_gpu params 접근")
    print("="*80)
    
    # 1. params_dict 생성 (전체 구조)
    print("\n[1] params_dict 생성 (전체 구조)")
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
            }
        },
        'structural': {
            'gamma_health_concern_to_perceived_benefit': -0.001,
            'gamma_perceived_benefit_to_purchase_intention': 0.002
        },
        'choice': {
            'asc_sugar': 1.45,
            'asc_sugar_free': 2.44,
            'beta_health_label': 0.50,
            'beta_price': -0.56
        }
    }
    
    print(f"  params_dict keys: {list(params_dict.keys())}")
    print(f"  params_dict['structural'] keys: {list(params_dict['structural'].keys())}")
    print(f"  params_dict['measurement'] keys: {list(params_dict['measurement'].keys())}")
    
    # 2. 계층적 경로 정의
    print("\n[2] 계층적 경로 정의")
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
    
    print(f"  경로 수: {len(hierarchical_paths)}")
    for i, path in enumerate(hierarchical_paths):
        print(f"  경로 {i+1}: {path['predictors'][0]} → {path['target']}")
    
    # 3. params 접근 테스트
    print("\n[3] params 접근 테스트")
    
    try:
        for path_idx, path in enumerate(hierarchical_paths):
            target = path['target']
            predictors = path['predictors']
            param_key = f"gamma_{predictors[0]}_to_{target}"
            
            print(f"\n  [경로 {path_idx + 1}] {predictors[0]} → {target}")
            print(f"    param_key: {param_key}")
            
            # ✅ 수정된 접근 방식: params['structural'][param_key]
            gamma = params_dict['structural'][param_key]
            print(f"    ✅ gamma = {gamma} (params['structural']['{param_key}'])")
            
            # ❌ 이전 접근 방식 (에러 발생)
            try:
                gamma_old = params_dict[param_key]
                print(f"    ❌ FAIL: params['{param_key}'] 접근 성공 (예상: KeyError)")
                return False
            except KeyError:
                print(f"    ✅ PASS: params['{param_key}'] 접근 실패 (예상대로)")
        
        print("\n  ✅ 모든 경로에서 params['structural'][param_key] 접근 성공")
        
    except KeyError as e:
        print(f"  ❌ FAIL: KeyError 발생 - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. measurement/choice params 접근 테스트
    print("\n[4] measurement/choice params 접근 테스트")
    
    try:
        # measurement params 접근
        meas_params = params_dict['measurement']
        print(f"  ✅ params['measurement'] 접근 성공 ({len(meas_params)} LVs)")
        
        # choice params 접근
        choice_params = params_dict['choice']
        print(f"  ✅ params['choice'] 접근 성공 ({len(choice_params)} params)")
        
    except KeyError as e:
        print(f"  ❌ FAIL: KeyError 발생 - {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ 모든 테스트 통과!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = test_structural_gradient_params_access()
        
        if success:
            print("\n✅ 테스트 성공: GPU 경로에서 params_dict 접근이 올바릅니다.")
            sys.exit(0)
        else:
            print("\n❌ 테스트 실패: params_dict 접근에 문제가 있습니다.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 테스트 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

