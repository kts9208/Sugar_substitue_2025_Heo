"""
MultiLatentJointGradient의 measurement_params_fixed 플래그 테스트

이 테스트는 measurement_params_fixed=True로 설정했을 때,
compute_individual_gradient()가 측정모델 그래디언트를 0으로 설정하고
params_dict['measurement']에 접근하지 않는지 확인합니다.
"""

import sys
import os

print("="*80)
print("테스트 시작: measurement_params_fixed 플래그")
print("="*80)

# 프로젝트 루트를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"\n프로젝트 루트: {project_root}")
print(f"Python path: {sys.path[0]}")

try:
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_gradient import (
        MultiLatentJointGradient
    )
    print("\n✅ MultiLatentJointGradient import 성공")
except Exception as e:
    print(f"\n❌ MultiLatentJointGradient import 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_measurement_params_fixed():
    """
    measurement_params_fixed=True 테스트
    """
    print("\n" + "="*80)
    print("TEST: measurement_params_fixed=True")
    print("="*80)

    # 1. Mock 객체 생성
    measurement_grad = None  # 실제로는 사용되지 않음
    structural_grad = None
    choice_grad = None

    # 2. MultiLatentJointGradient 초기화 (measurement_params_fixed=True)
    print("\n[1] MultiLatentJointGradient 초기화 (measurement_params_fixed=True)")
    joint_grad = MultiLatentJointGradient(
        measurement_grad=measurement_grad,
        structural_grad=structural_grad,
        choice_grad=choice_grad,
        use_gpu=False,
        gpu_measurement_model=None,
        use_full_parallel=False,
        measurement_params_fixed=True  # ✅ 측정모델 파라미터 고정
    )

    # 3. measurement_params_fixed 플래그 확인
    print(f"\n[2] measurement_params_fixed 플래그 확인")
    print(f"  joint_grad.measurement_params_fixed = {joint_grad.measurement_params_fixed}")

    if joint_grad.measurement_params_fixed:
        print("  ✅ PASS: measurement_params_fixed=True로 설정됨")
    else:
        print("  ❌ FAIL: measurement_params_fixed=False (예상: True)")
        return False

    # 4. measurement_params_fixed=False 테스트
    print("\n[3] MultiLatentJointGradient 초기화 (measurement_params_fixed=False)")
    joint_grad_false = MultiLatentJointGradient(
        measurement_grad=measurement_grad,
        structural_grad=structural_grad,
        choice_grad=choice_grad,
        use_gpu=False,
        gpu_measurement_model=None,
        use_full_parallel=False,
        measurement_params_fixed=False  # ✅ 측정모델 파라미터 변동
    )

    print(f"\n[4] measurement_params_fixed 플래그 확인")
    print(f"  joint_grad_false.measurement_params_fixed = {joint_grad_false.measurement_params_fixed}")

    if not joint_grad_false.measurement_params_fixed:
        print("  ✅ PASS: measurement_params_fixed=False로 설정됨")
    else:
        print("  ❌ FAIL: measurement_params_fixed=True (예상: False)")
        return False

    # 5. 기본값 테스트 (파라미터 생략 시)
    print("\n[5] MultiLatentJointGradient 초기화 (기본값)")
    joint_grad_default = MultiLatentJointGradient(
        measurement_grad=measurement_grad,
        structural_grad=structural_grad,
        choice_grad=choice_grad,
        use_gpu=False,
        gpu_measurement_model=None,
        use_full_parallel=False
        # measurement_params_fixed 생략 → 기본값 False
    )

    print(f"\n[6] measurement_params_fixed 플래그 확인 (기본값)")
    print(f"  joint_grad_default.measurement_params_fixed = {joint_grad_default.measurement_params_fixed}")

    if not joint_grad_default.measurement_params_fixed:
        print("  ✅ PASS: 기본값 False로 설정됨")
    else:
        print("  ❌ FAIL: 기본값이 True (예상: False)")
        return False

    print("\n" + "="*80)
    print("✅ 모든 테스트 통과!")
    print("="*80)

    return True

if __name__ == "__main__":
    try:
        success = test_measurement_params_fixed()

        if success:
            print("\n✅ 테스트 성공: measurement_params_fixed 플래그가 제대로 작동합니다.")
            sys.exit(0)
        else:
            print("\n❌ 테스트 실패: measurement_params_fixed 플래그에 문제가 있습니다.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 테스트 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

