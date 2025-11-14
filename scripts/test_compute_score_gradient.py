"""
compute_score_gradient() 함수 검증

scipy.optimize.check_grad를 사용하여 
compute_score_gradient() 함수의 정확성을 검증합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-14
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scipy.optimize import check_grad

from src.analysis.hybrid_choice_model.iclv_models.gradient_core import (
    compute_score_gradient,
    compute_probit_gradient_common_term,
    compute_variance_gradient
)


def test_score_gradient_simple():
    """
    간단한 선형 회귀 모델로 Score Function 검증
    
    모델: y = θ * x + ε, ε ~ N(0, σ²)
    Log-likelihood: log L = -0.5 * log(2πσ²) - (y - θ*x)² / (2σ²)
    Gradient: ∂ log L / ∂θ = (y - θ*x) / σ² * x
    """
    print("\n" + "="*70)
    print("Test 1: 간단한 선형 회귀 모델")
    print("="*70)
    
    # 데이터 생성
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    true_theta = 2.5
    sigma_sq = 1.0
    y = true_theta * x + np.random.randn(n) * np.sqrt(sigma_sq)
    
    # 테스트할 파라미터 값
    theta = 2.0
    
    # 목적함수: -log-likelihood
    def objective(theta_val):
        theta_val = theta_val[0]  # scalar로 변환
        predicted = theta_val * x
        residual = y - predicted
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma_sq) + residual**2 / sigma_sq)
        return -ll  # 최소화 문제
    
    # Analytic gradient
    def analytic_grad(theta_val):
        theta_val = theta_val[0]  # scalar로 변환
        predicted = theta_val * x
        
        # compute_score_gradient 사용
        grad = compute_score_gradient(
            observed=y,
            predicted=predicted,
            variance=sigma_sq,
            derivative_term=x
        )
        
        total_grad = np.sum(grad)
        return -np.array([total_grad])  # 최소화 문제
    
    # scipy.optimize.check_grad로 검증
    theta_init = np.array([theta])
    error = check_grad(objective, analytic_grad, theta_init, epsilon=1e-7)
    
    print(f"\n파라미터: θ = {theta}")
    print(f"Gradient Error (L2 norm): {error:.10f}")
    
    if error < 1e-5:
        print("✅ PASS: compute_score_gradient가 정확합니다!")
    else:
        print("❌ FAIL: compute_score_gradient에 오차가 있습니다!")
    
    return error


def test_score_gradient_multiple_params():
    """
    다중 파라미터 선형 회귀로 Score Function 검증
    
    모델: y = β₀ + β₁*x₁ + β₂*x₂ + ε
    """
    print("\n" + "="*70)
    print("Test 2: 다중 파라미터 선형 회귀")
    print("="*70)
    
    # 데이터 생성
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)  # 2개 설명변수
    true_beta = np.array([1.0, 2.0, -1.5])  # [intercept, β₁, β₂]
    sigma_sq = 1.0
    
    X_with_intercept = np.column_stack([np.ones(n), X])
    y = X_with_intercept @ true_beta + np.random.randn(n) * np.sqrt(sigma_sq)
    
    # 목적함수
    def objective(beta):
        predicted = X_with_intercept @ beta
        residual = y - predicted
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma_sq) + residual**2 / sigma_sq)
        return -ll
    
    # Analytic gradient
    def analytic_grad(beta):
        predicted = X_with_intercept @ beta
        
        # 각 파라미터에 대한 gradient 계산
        grad = np.zeros(3)
        for j in range(3):
            grad[j] = np.sum(compute_score_gradient(
                observed=y,
                predicted=predicted,
                variance=sigma_sq,
                derivative_term=X_with_intercept[:, j]
            ))
        
        return -grad
    
    # 검증
    beta_init = np.array([0.5, 1.5, -1.0])
    error = check_grad(objective, analytic_grad, beta_init, epsilon=1e-7)
    
    print(f"\n파라미터: β = {beta_init}")
    print(f"Gradient Error (L2 norm): {error:.10f}")
    
    if error < 1e-5:
        print("✅ PASS: 다중 파라미터에서도 정확합니다!")
    else:
        print("❌ FAIL: 다중 파라미터에서 오차가 있습니다!")
    
    return error


def test_probit_gradient():
    """
    Binary Probit 모델로 compute_probit_gradient_common_term 검증
    
    모델: P(y=1) = Φ(V), V = β*x
    """
    print("\n" + "="*70)
    print("Test 3: Binary Probit 모델")
    print("="*70)
    
    from scipy.stats import norm
    
    # 데이터 생성
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    true_beta = 1.5
    V_true = true_beta * x
    prob_true = norm.cdf(V_true)
    y = (np.random.rand(n) < prob_true).astype(float)
    
    # 목적함수: -log-likelihood
    def objective(beta_val):
        beta_val = beta_val[0]
        V = beta_val * x
        prob = norm.cdf(V)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        ll = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        return -ll
    
    # Analytic gradient
    def analytic_grad(beta_val):
        beta_val = beta_val[0]
        V = beta_val * x
        
        # compute_probit_gradient_common_term 사용
        common_term = compute_probit_gradient_common_term(
            choice=y,
            utility=V
        )
        
        # ∂V/∂β = x
        grad = np.sum(common_term * x)
        
        return -np.array([grad])
    
    # 검증
    beta_init = np.array([1.0])
    error = check_grad(objective, analytic_grad, beta_init, epsilon=1e-7)
    
    print(f"\n파라미터: β = {beta_init[0]}")
    print(f"Gradient Error (L2 norm): {error:.10f}")
    
    if error < 1e-5:
        print("✅ PASS: Probit gradient가 정확합니다!")
    else:
        print("❌ FAIL: Probit gradient에 오차가 있습니다!")
    
    return error


def test_variance_gradient():
    """
    분산 파라미터 gradient 검증
    
    모델: y ~ N(μ, σ²)
    ∂ log L / ∂σ² = -1/(2σ²) + (y-μ)²/(2σ⁴)
    """
    print("\n" + "="*70)
    print("Test 4: 분산 파라미터 Gradient")
    print("="*70)
    
    # 데이터 생성
    np.random.seed(42)
    n = 100
    mu = 2.0
    true_sigma_sq = 1.5
    y = np.random.randn(n) * np.sqrt(true_sigma_sq) + mu
    
    # 목적함수
    def objective(sigma_sq_val):
        sigma_sq_val = sigma_sq_val[0]
        if sigma_sq_val <= 0:
            return 1e10
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma_sq_val) + (y - mu)**2 / sigma_sq_val)
        return -ll
    
    # Analytic gradient
    def analytic_grad(sigma_sq_val):
        sigma_sq_val = sigma_sq_val[0]
        
        # compute_variance_gradient 사용
        grad = compute_variance_gradient(
            observed=y,
            predicted=mu,
            variance=sigma_sq_val
        )
        
        total_grad = np.sum(grad)
        return -np.array([total_grad])
    
    # 검증
    sigma_sq_init = np.array([1.0])
    error = check_grad(objective, analytic_grad, sigma_sq_init, epsilon=1e-7)
    
    print(f"\n파라미터: σ² = {sigma_sq_init[0]}")
    print(f"Gradient Error (L2 norm): {error:.10f}")
    
    if error < 1e-5:
        print("✅ PASS: 분산 gradient가 정확합니다!")
    else:
        print("❌ FAIL: 분산 gradient에 오차가 있습니다!")
    
    return error


def main():
    """모든 테스트 실행"""
    print("\n" + "="*70)
    print("compute_score_gradient() 함수 검증")
    print("scipy.optimize.check_grad 사용")
    print("="*70)
    
    errors = []
    
    # Test 1: 간단한 선형 회귀
    errors.append(test_score_gradient_simple())
    
    # Test 2: 다중 파라미터
    errors.append(test_score_gradient_multiple_params())
    
    # Test 3: Binary Probit
    errors.append(test_probit_gradient())
    
    # Test 4: 분산 파라미터
    errors.append(test_variance_gradient())
    
    # 최종 결과
    print("\n" + "="*70)
    print("최종 결과")
    print("="*70)
    print(f"Test 1 (선형 회귀):        Error = {errors[0]:.10f}")
    print(f"Test 2 (다중 파라미터):    Error = {errors[1]:.10f}")
    print(f"Test 3 (Binary Probit):    Error = {errors[2]:.10f}")
    print(f"Test 4 (분산 파라미터):    Error = {errors[3]:.10f}")
    print(f"\nMax Error: {max(errors):.10f}")
    
    if max(errors) < 1e-5:
        print("\n✅ 모든 테스트 PASS!")
        print("compute_score_gradient() 함수가 정확하게 작동합니다!")
    else:
        print("\n⚠️  일부 테스트에서 오차가 발견되었습니다.")
    
    print("="*70)


if __name__ == '__main__':
    main()

