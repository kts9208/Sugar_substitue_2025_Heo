"""
Hessian 처리 로직 테스트

L-BFGS-B와 BFGS의 hess_inv 처리가 올바른지 확인
"""
import numpy as np
import scipy.optimize as opt


def test_hessian_handling():
    """
    L-BFGS-B와 BFGS의 hess_inv 처리 테스트
    """
    print("="*80)
    print("Hessian 처리 로직 테스트")
    print("="*80)
    
    # 간단한 최적화 문제
    def objective(x):
        return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-3)**2
    
    def gradient(x):
        return np.array([2*(x[0]-1), 2*(x[1]-2), 2*(x[2]-3)])
    
    initial_params = np.array([0.0, 0.0, 0.0])
    
    # 1. L-BFGS-B 테스트
    print("\n" + "="*80)
    print("1. L-BFGS-B 테스트")
    print("="*80)
    
    result_lbfgsb = opt.minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        jac=gradient
    )
    
    print(f"최적화 성공: {result_lbfgsb.success}")
    print(f"최종 파라미터: {result_lbfgsb.x}")
    print(f"최종 함수값: {result_lbfgsb.fun}")
    print()
    
    # Hessian 처리 (우리 코드와 동일한 로직)
    if hasattr(result_lbfgsb, 'hess_inv') and result_lbfgsb.hess_inv is not None:
        hess_inv = result_lbfgsb.hess_inv
        
        if hasattr(hess_inv, 'todense'):
            # L-BFGS-B의 경우
            print("✅ Hessian 역행렬: L-BFGS-B에서 자동 제공 (LbfgsInvHessProduct)")
            print("  → todense()로 numpy 배열로 변환 중...")
            hess_inv_array = hess_inv.todense()
            print(f"  ✅ 변환 완료 (shape: {hess_inv_array.shape})")
        else:
            # BFGS의 경우
            print("✅ Hessian 역행렬: BFGS에서 자동 제공 (numpy.ndarray)")
            hess_inv_array = hess_inv
        
        print("  → 추가 계산 0회! (optimizer가 최적화 중 자동 계산)")
        print()
        print(f"Hessian 역행렬 타입: {type(hess_inv_array)}")
        print(f"Hessian 역행렬 shape: {hess_inv_array.shape}")
        print(f"Hessian 역행렬:\n{hess_inv_array}")
        print()
        
        # 표준오차 계산
        diag_elements = np.diag(hess_inv_array)
        se = np.sqrt(np.abs(diag_elements))
        print(f"표준오차: {se}")
    else:
        print("❌ Hessian 역행렬 없음")
    
    # 2. BFGS 테스트
    print("\n" + "="*80)
    print("2. BFGS 테스트")
    print("="*80)
    
    result_bfgs = opt.minimize(
        objective,
        initial_params,
        method='BFGS',
        jac=gradient
    )
    
    print(f"최적화 성공: {result_bfgs.success}")
    print(f"최종 파라미터: {result_bfgs.x}")
    print(f"최종 함수값: {result_bfgs.fun}")
    print()
    
    # Hessian 처리 (우리 코드와 동일한 로직)
    if hasattr(result_bfgs, 'hess_inv') and result_bfgs.hess_inv is not None:
        hess_inv = result_bfgs.hess_inv
        
        if hasattr(hess_inv, 'todense'):
            # L-BFGS-B의 경우
            print("✅ Hessian 역행렬: L-BFGS-B에서 자동 제공 (LbfgsInvHessProduct)")
            print("  → todense()로 numpy 배열로 변환 중...")
            hess_inv_array = hess_inv.todense()
            print(f"  ✅ 변환 완료 (shape: {hess_inv_array.shape})")
        else:
            # BFGS의 경우
            print("✅ Hessian 역행렬: BFGS에서 자동 제공 (numpy.ndarray)")
            hess_inv_array = hess_inv
        
        print("  → 추가 계산 0회! (optimizer가 최적화 중 자동 계산)")
        print()
        print(f"Hessian 역행렬 타입: {type(hess_inv_array)}")
        print(f"Hessian 역행렬 shape: {hess_inv_array.shape}")
        print(f"Hessian 역행렬:\n{hess_inv_array}")
        print()
        
        # 표준오차 계산
        diag_elements = np.diag(hess_inv_array)
        se = np.sqrt(np.abs(diag_elements))
        print(f"표준오차: {se}")
    else:
        print("❌ Hessian 역행렬 없음")
    
    # 3. 비교
    print("\n" + "="*80)
    print("3. L-BFGS-B vs BFGS 비교")
    print("="*80)
    
    if (hasattr(result_lbfgsb, 'hess_inv') and result_lbfgsb.hess_inv is not None and
        hasattr(result_bfgs, 'hess_inv') and result_bfgs.hess_inv is not None):
        
        hess_inv_lbfgsb = result_lbfgsb.hess_inv.todense() if hasattr(result_lbfgsb.hess_inv, 'todense') else result_lbfgsb.hess_inv
        hess_inv_bfgs = result_bfgs.hess_inv.todense() if hasattr(result_bfgs.hess_inv, 'todense') else result_bfgs.hess_inv
        
        print(f"L-BFGS-B Hessian 역행렬:\n{hess_inv_lbfgsb}")
        print()
        print(f"BFGS Hessian 역행렬:\n{hess_inv_bfgs}")
        print()
        
        diff = np.abs(hess_inv_lbfgsb - hess_inv_bfgs)
        print(f"차이 (절대값):\n{diff}")
        print(f"최대 차이: {np.max(diff):.6e}")
        
        if np.allclose(hess_inv_lbfgsb, hess_inv_bfgs, rtol=1e-3):
            print("✅ 두 방법의 Hessian 역행렬이 거의 동일합니다")
        else:
            print("⚠️ 두 방법의 Hessian 역행렬이 다릅니다")


if __name__ == "__main__":
    test_hessian_handling()

