"""
Core Gradient Calculation Functions

모든 ICLV 모델 gradient 계산의 공통 통계적 원리를 구현합니다.

핵심 원리: Score Function (점수 함수)
∂ log L / ∂θ = Σ (y_i - μ_i(θ)) / Var(y_i) × ∂μ_i/∂θ

Author: Sugar Substitute Research Team
Date: 2025-11-14
"""

import numpy as np
from scipy.stats import norm
from typing import Union


def compute_score_gradient(observed: Union[float, np.ndarray],
                          predicted: Union[float, np.ndarray],
                          variance: Union[float, np.ndarray],
                          derivative_term: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Score Function을 이용한 gradient 계산
    
    정규분포 기반 최대우도추정의 일반 공식:
    ∂ log L / ∂θ = (y - μ(θ)) / Var(y) × ∂μ/∂θ
    
    Args:
        observed: 관측값 (y)
        predicted: 예측값 (μ(θ))
        variance: 분산 (Var(y))
        derivative_term: 미분항 (∂μ/∂θ)
    
    Returns:
        gradient: ∂ log L / ∂θ
    
    Examples:
        # 측정모델 (Continuous Linear): Y = ζ*LV + ε
        grad_zeta = compute_score_gradient(
            observed=y,
            predicted=zeta * lv,
            variance=sigma_sq,
            derivative_term=lv
        )
        
        # 구조모델: target = γ*predictor + error
        grad_gamma = compute_score_gradient(
            observed=target_value,
            predicted=gamma * pred_value,
            variance=error_variance,
            derivative_term=pred_value
        )
        
        # 선택모델: choice ~ Φ(V)
        grad_beta = compute_score_gradient(
            observed=choice,
            predicted=prob,
            variance=prob * (1 - prob),
            derivative_term=phi * X
        )
    """
    # 잔차 계산
    residual = observed - predicted
    
    # Score gradient
    gradient = residual / variance * derivative_term
    
    return gradient


def compute_probit_gradient_common_term(choice: float,
                                       utility: float) -> float:
    """
    Binary Probit 모델의 gradient 공통항 계산
    
    V = utility
    P = Φ(V)
    φ = φ(V)
    
    common_term = (choice - P) / (P * (1-P)) × φ
    
    이 common_term에 각 파라미터의 미분항을 곱하면 해당 파라미터의 gradient가 됩니다:
    - ∂ log L / ∂intercept = common_term × 1
    - ∂ log L / ∂β = common_term × X
    - ∂ log L / ∂λ = common_term × LV
    
    Args:
        choice: 선택 결과 (0 or 1)
        utility: 효용 V
    
    Returns:
        common_term: gradient 계산에 사용할 공통항
    """
    # 확률 계산
    prob = norm.cdf(utility)
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    
    # 확률밀도함수
    phi = norm.pdf(utility)
    
    # 공통항: (choice - prob) / (prob * (1 - prob)) * phi
    variance = prob * (1 - prob)
    common_term = (choice - prob) / variance * phi
    
    return common_term


def compute_ordered_probit_gradient_terms(observed_category: int,
                                         latent_value: float,
                                         thresholds: np.ndarray,
                                         n_categories: int) -> tuple:
    """
    Ordered Probit 모델의 gradient 계산을 위한 항들
    
    Y* = ζ*LV (latent continuous variable)
    Y = k if τ_{k-1} < Y* ≤ τ_k
    
    P(Y=k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
    
    ∂ log L / ∂ζ = (φ(τ_{k-1} - ζ*LV) - φ(τ_k - ζ*LV)) / P(Y=k) × LV
    ∂ log L / ∂τ_k = φ(τ_k - ζ*LV) / P(Y=k)
    
    Args:
        observed_category: 관측된 카테고리 (0-based index)
        latent_value: 잠재변수 값 (ζ*LV)
        thresholds: 임계값 배열 τ
        n_categories: 카테고리 개수
    
    Returns:
        (prob, phi_lower, phi_upper): 확률과 두 경계의 PDF 값
    """
    k = observed_category
    
    # P(Y=k) 계산
    if k == 0:
        # 첫 번째 카테고리: P(Y=0) = Φ(τ_0 - V)
        prob = norm.cdf(thresholds[0] - latent_value)
        phi_upper = norm.pdf(thresholds[0] - latent_value)
        phi_lower = 0.0
    elif k == n_categories - 1:
        # 마지막 카테고리: P(Y=K) = 1 - Φ(τ_{K-1} - V)
        prob = 1 - norm.cdf(thresholds[-1] - latent_value)
        phi_upper = 0.0
        phi_lower = norm.pdf(thresholds[-1] - latent_value)
    else:
        # 중간 카테고리: P(Y=k) = Φ(τ_k - V) - Φ(τ_{k-1} - V)
        prob = norm.cdf(thresholds[k] - latent_value) - norm.cdf(thresholds[k-1] - latent_value)
        phi_upper = norm.pdf(thresholds[k] - latent_value)
        phi_lower = norm.pdf(thresholds[k-1] - latent_value)
    
    # 수치 안정성
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    
    return prob, phi_lower, phi_upper


def compute_variance_gradient(observed: Union[float, np.ndarray],
                             predicted: Union[float, np.ndarray],
                             variance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    분산 파라미터에 대한 gradient 계산
    
    정규분포의 log-likelihood:
    log L = -0.5 * log(2π * σ²) - 0.5 * (y - μ)² / σ²
    
    ∂ log L / ∂σ² = -1/(2σ²) + (y - μ)² / (2σ⁴)
    
    Args:
        observed: 관측값 (y)
        predicted: 예측값 (μ)
        variance: 분산 (σ²)
    
    Returns:
        gradient: ∂ log L / ∂σ²
    """
    residual = observed - predicted
    
    # ∂ log L / ∂σ² = -1/(2σ²) + (y - μ)² / (2σ⁴)
    gradient = -1.0 / (2.0 * variance) + (residual ** 2) / (2.0 * variance ** 2)
    
    return gradient

