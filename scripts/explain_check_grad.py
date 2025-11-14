"""
scipy.optimize.check_grad의 내부 동작 설명

check_grad가 어떻게 numerical gradient를 계산하고 
analytic gradient와 비교하는지 보여줍니다.

Author: Sugar Substitute Research Team
Date: 2025-11-14
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scipy.optimize import check_grad

from src.analysis.hybrid_choice_model.iclv_models.gradient_core import compute_score_gradient


def manual_numerical_gradient(func, x, epsilon=1.4901161193847656e-08):
    """
    scipy.optimize.check_grad가 내부적으로 사용하는 방식
    
    Finite difference method (중앙차분):
    ∂f/∂x_i ≈ [f(x + ε*e_i) - f(x - ε*e_i)] / (2ε)
    
    여기서 e_i는 i번째 단위벡터
    """
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        # x + ε*e_i
        x_plus = x.copy()
        x_plus[i] += epsilon
        
        # x - ε*e_i
        x_minus = x.copy()
        x_minus[i] -= epsilon
        
        # 중앙차분
        grad[i] = (func(x_plus) - func(x_minus)) / (2.0 * epsilon)
    
    return grad


def demonstrate_check_grad():
    """
    check_grad의 동작을 단계별로 보여줍니다
    """
    print("="*70)
    print("scipy.optimize.check_grad 내부 동작 설명")
    print("="*70)
    
    # 간단한 예제: f(x) = x₁² + 2*x₂² + 3*x₁*x₂
    print("\n예제 함수: f(x) = x₁² + 2*x₂² + 3*x₁*x₂")
    print("\n해석적 gradient:")
    print("  ∂f/∂x₁ = 2*x₁ + 3*x₂")
    print("  ∂f/∂x₂ = 4*x₂ + 3*x₁")
    
    # 목적함수
    def objective(x):
        return x[0]**2 + 2*x[1]**2 + 3*x[0]*x[1]
    
    # Analytic gradient
    def analytic_grad(x):
        return np.array([
            2*x[0] + 3*x[1],  # ∂f/∂x₁
            4*x[1] + 3*x[0]   # ∂f/∂x₂
        ])
    
    # 테스트 포인트
    x0 = np.array([1.5, 2.0])
    epsilon = 1e-7
    
    print(f"\n테스트 포인트: x = {x0}")
    print(f"Epsilon (step size): {epsilon}")
    
    # 1. Analytic gradient 계산
    print("\n" + "-"*70)
    print("1. Analytic Gradient 계산")
    print("-"*70)
    grad_analytic = analytic_grad(x0)
    print(f"Analytic gradient = {grad_analytic}")
    print(f"  ∂f/∂x₁ = 2*{x0[0]} + 3*{x0[1]} = {grad_analytic[0]}")
    print(f"  ∂f/∂x₂ = 4*{x0[1]} + 3*{x0[0]} = {grad_analytic[1]}")
    
    # 2. Numerical gradient 계산 (수동)
    print("\n" + "-"*70)
    print("2. Numerical Gradient 계산 (Finite Difference)")
    print("-"*70)
    
    print("\n파라미터 1 (x₁):")
    x_plus_1 = x0.copy()
    x_plus_1[0] += epsilon
    x_minus_1 = x0.copy()
    x_minus_1[0] -= epsilon
    f_plus_1 = objective(x_plus_1)
    f_minus_1 = objective(x_minus_1)
    numerical_grad_1 = (f_plus_1 - f_minus_1) / (2 * epsilon)
    
    print(f"  x + ε*e₁ = {x_plus_1}")
    print(f"  f(x + ε*e₁) = {f_plus_1:.10f}")
    print(f"  x - ε*e₁ = {x_minus_1}")
    print(f"  f(x - ε*e₁) = {f_minus_1:.10f}")
    print(f"  ∂f/∂x₁ ≈ ({f_plus_1:.10f} - {f_minus_1:.10f}) / (2*{epsilon})")
    print(f"  ∂f/∂x₁ ≈ {numerical_grad_1:.10f}")
    
    print("\n파라미터 2 (x₂):")
    x_plus_2 = x0.copy()
    x_plus_2[1] += epsilon
    x_minus_2 = x0.copy()
    x_minus_2[1] -= epsilon
    f_plus_2 = objective(x_plus_2)
    f_minus_2 = objective(x_minus_2)
    numerical_grad_2 = (f_plus_2 - f_minus_2) / (2 * epsilon)
    
    print(f"  x + ε*e₂ = {x_plus_2}")
    print(f"  f(x + ε*e₂) = {f_plus_2:.10f}")
    print(f"  x - ε*e₂ = {x_minus_2}")
    print(f"  f(x - ε*e₂) = {f_minus_2:.10f}")
    print(f"  ∂f/∂x₂ ≈ ({f_plus_2:.10f} - {f_minus_2:.10f}) / (2*{epsilon})")
    print(f"  ∂f/∂x₂ ≈ {numerical_grad_2:.10f}")
    
    grad_numerical = np.array([numerical_grad_1, numerical_grad_2])
    print(f"\nNumerical gradient = {grad_numerical}")
    
    # 3. 비교
    print("\n" + "-"*70)
    print("3. Analytic vs Numerical Gradient 비교")
    print("-"*70)
    print(f"{'Parameter':<15} {'Analytic':>15} {'Numerical':>15} {'Difference':>15}")
    print("-"*60)
    print(f"{'x₁':<15} {grad_analytic[0]:>15.10f} {grad_numerical[0]:>15.10f} {abs(grad_analytic[0] - grad_numerical[0]):>15.10e}")
    print(f"{'x₂':<15} {grad_analytic[1]:>15.10f} {grad_numerical[1]:>15.10f} {abs(grad_analytic[1] - grad_numerical[1]):>15.10e}")
    
    # 4. L2 norm 계산 (check_grad가 반환하는 값)
    print("\n" + "-"*70)
    print("4. Error 계산 (L2 Norm)")
    print("-"*70)
    diff = grad_analytic - grad_numerical
    error = np.sqrt(np.sum(diff**2))
    print(f"Difference vector = {diff}")
    print(f"L2 norm = sqrt({diff[0]**2:.10e} + {diff[1]**2:.10e})")
    print(f"L2 norm = {error:.10e}")
    
    # 5. scipy.optimize.check_grad 사용
    print("\n" + "-"*70)
    print("5. scipy.optimize.check_grad 결과")
    print("-"*70)
    error_scipy = check_grad(objective, analytic_grad, x0, epsilon=epsilon)
    print(f"check_grad() 반환값 = {error_scipy:.10e}")
    print(f"수동 계산 결과 = {error:.10e}")
    print(f"차이 = {abs(error_scipy - error):.10e}")
    
    if abs(error_scipy - error) < 1e-10:
        print("\n✅ 수동 계산과 check_grad() 결과가 일치합니다!")


def demonstrate_with_score_gradient():
    """
    compute_score_gradient를 사용한 실제 예제
    """
    print("\n\n" + "="*70)
    print("compute_score_gradient() 함수 검증 예제")
    print("="*70)
    
    # 선형 회귀 모델
    print("\n모델: y = θ*x + ε, ε ~ N(0, σ²)")
    print("Log-likelihood: Σ[-0.5*log(2πσ²) - (y - θ*x)²/(2σ²)]")
    print("Gradient: ∂ log L / ∂θ = Σ[(y - θ*x) / σ² * x]")
    
    # 데이터
    np.random.seed(42)
    n = 5  # 작은 샘플
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.1, 4.3, 5.8, 8.2, 10.1])
    sigma_sq = 1.0
    
    print(f"\n데이터:")
    print(f"  x = {x}")
    print(f"  y = {y}")
    print(f"  σ² = {sigma_sq}")
    
    # 테스트 파라미터
    theta = 2.0
    epsilon = 1e-7
    
    print(f"\n테스트 파라미터: θ = {theta}")
    print(f"Epsilon: {epsilon}")
    
    # 목적함수
    def objective(theta_val):
        theta_val = theta_val[0]
        predicted = theta_val * x
        residual = y - predicted
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma_sq) + residual**2 / sigma_sq)
        return -ll  # 최소화
    
    # Analytic gradient (compute_score_gradient 사용)
    def analytic_grad(theta_val):
        theta_val = theta_val[0]
        predicted = theta_val * x
        
        # compute_score_gradient 사용
        grad = compute_score_gradient(
            observed=y,
            predicted=predicted,
            variance=sigma_sq,
            derivative_term=x
        )
        
        total_grad = np.sum(grad)
        return -np.array([total_grad])
    
    # 1. Analytic gradient
    print("\n" + "-"*70)
    print("1. Analytic Gradient (compute_score_gradient 사용)")
    print("-"*70)
    theta_array = np.array([theta])
    grad_analytic = analytic_grad(theta_array)
    
    predicted = theta * x
    print(f"\n예측값 (θ*x) = {predicted}")
    print(f"잔차 (y - θ*x) = {y - predicted}")
    
    individual_grads = compute_score_gradient(
        observed=y,
        predicted=predicted,
        variance=sigma_sq,
        derivative_term=x
    )
    print(f"\n개별 gradient = {individual_grads}")
    print(f"  각 관측치: (y_i - θ*x_i) / σ² * x_i")
    for i in range(n):
        print(f"    관측치 {i+1}: ({y[i]:.1f} - {predicted[i]:.1f}) / {sigma_sq} * {x[i]} = {individual_grads[i]:.6f}")
    
    print(f"\n총 gradient = Σ(개별 gradient) = {np.sum(individual_grads):.6f}")
    print(f"Analytic gradient = {grad_analytic[0]:.10f} (부호 반전)")
    
    # 2. Numerical gradient
    print("\n" + "-"*70)
    print("2. Numerical Gradient (Finite Difference)")
    print("-"*70)
    
    theta_plus = np.array([theta + epsilon])
    theta_minus = np.array([theta - epsilon])
    f_plus = objective(theta_plus)
    f_minus = objective(theta_minus)
    grad_numerical = (f_plus - f_minus) / (2 * epsilon)
    
    print(f"\nf(θ + ε) = f({theta + epsilon:.10f}) = {f_plus:.10f}")
    print(f"f(θ - ε) = f({theta - epsilon:.10f}) = {f_minus:.10f}")
    print(f"Numerical gradient = ({f_plus:.10f} - {f_minus:.10f}) / (2*{epsilon})")
    print(f"Numerical gradient = {grad_numerical:.10f}")
    
    # 3. 비교
    print("\n" + "-"*70)
    print("3. 비교")
    print("-"*70)
    print(f"Analytic gradient  = {grad_analytic[0]:.10f}")
    print(f"Numerical gradient = {grad_numerical:.10f}")
    print(f"Difference         = {abs(grad_analytic[0] - grad_numerical):.10e}")
    
    # 4. check_grad
    print("\n" + "-"*70)
    print("4. scipy.optimize.check_grad")
    print("-"*70)
    error = check_grad(objective, analytic_grad, theta_array, epsilon=epsilon)
    print(f"Error (L2 norm) = {error:.10e}")
    
    if error < 1e-5:
        print("\n✅ PASS: compute_score_gradient가 정확합니다!")
    else:
        print("\n❌ FAIL: 오차가 있습니다!")


if __name__ == '__main__':
    # 1. 기본 동작 설명
    demonstrate_check_grad()
    
    # 2. compute_score_gradient 예제
    demonstrate_with_score_gradient()

