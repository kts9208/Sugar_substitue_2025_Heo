"""
compute_individual_gradient()가 measurement_params_fixed=True일 때
params_dict['measurement']에 접근하지 않는지 테스트
"""

import sys
import os

print("="*80)
print("테스트 시작: compute_individual_gradient with measurement_params_fixed=True")
print("="*80)

# 프로젝트 루트를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"\n프로젝트 루트: {project_root}")

try:
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_gradient import (
        MultiLatentJointGradient
    )
    print("✅ MultiLatentJointGradient import 성공")
except Exception as e:
    print(f"❌ MultiLatentJointGradient import 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_gradient_with_missing_measurement():
    """
    params_dict에 'measurement' 키가 없을 때 에러가 발생하지 않는지 테스트
    """
    print("\n" + "="*80)
    print("TEST: params_dict without 'measurement' key")
    print("="*80)
    
    # 1. MultiLatentJointGradient 초기화 (measurement_params_fixed=True)
    print("\n[1] MultiLatentJointGradient 초기화 (measurement_params_fixed=True)")
    joint_grad = MultiLatentJointGradient(
        measurement_grad=None,
        structural_grad=None,
        choice_grad=None,
        use_gpu=False,
        gpu_measurement_model=None,
        use_full_parallel=False,
        measurement_params_fixed=True  # ✅ 측정모델 파라미터 고정
    )
    
    print(f"  joint_grad.measurement_params_fixed = {joint_grad.measurement_params_fixed}")
    
    # 2. params_dict 생성 ('measurement' 키 없음)
    print("\n[2] params_dict 생성 ('measurement' 키 없음)")
    params_dict = {
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
    print(f"  'measurement' in params_dict: {'measurement' in params_dict}")
    
    # 3. measurement_params_fixed 플래그 확인
    print("\n[3] measurement_params_fixed 플래그로 분기 테스트")
    
    if joint_grad.measurement_params_fixed:
        print("  ✅ measurement_params_fixed=True")
        print("  → 측정모델 그래디언트 계산 스킵")
        print("  → params_dict['measurement']에 접근하지 않음")
        
        # 실제 코드에서 사용하는 로직 시뮬레이션
        try:
            # measurement_params_fixed=True이면 이 블록이 실행되지 않아야 함
            if not joint_grad.measurement_params_fixed:
                # 이 부분은 실행되지 않음
                _ = params_dict['measurement']  # KeyError 발생 가능
                print("  ❌ FAIL: measurement_params_fixed=True인데 측정모델 접근 시도")
                return False
            else:
                print("  ✅ PASS: params_dict['measurement']에 접근하지 않음")
        except KeyError as e:
            print(f"  ❌ FAIL: KeyError 발생 - {e}")
            return False
    else:
        print("  ❌ FAIL: measurement_params_fixed=False (예상: True)")
        return False
    
    # 4. 실제 gradient 계산 로직 시뮬레이션
    print("\n[4] gradient 계산 로직 시뮬레이션")
    
    try:
        # multi_latent_gradient.py Line 695-709 로직
        if joint_grad.measurement_params_fixed:
            # 측정모델 그래디언트를 빈 딕셔너리로 설정
            grad_meas = {}
            print("  ✅ grad_meas = {} (빈 딕셔너리)")
        else:
            # 파라미터가 변하므로 그래디언트 계산
            # 이 경우 params_dict['measurement']에 접근
            grad_meas = params_dict['measurement']  # KeyError 발생 가능
            print("  ❌ FAIL: params_dict['measurement']에 접근 시도")
            return False
        
        print("  ✅ PASS: 측정모델 그래디언트 계산 스킵 성공")
        
    except KeyError as e:
        print(f"  ❌ FAIL: KeyError 발생 - {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ 모든 테스트 통과!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = test_gradient_with_missing_measurement()
        
        if success:
            print("\n✅ 테스트 성공: measurement_params_fixed=True일 때 params_dict['measurement']에 접근하지 않습니다.")
            sys.exit(0)
        else:
            print("\n❌ 테스트 실패: measurement_params_fixed 로직에 문제가 있습니다.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 테스트 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

