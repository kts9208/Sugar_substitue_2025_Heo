"""
ICLV 컴포넌트별 단위 테스트

King (2022) R 코드의 각 컴포넌트를 Python으로 구현하고 검증합니다.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
import matplotlib.pyplot as plt


# ============================================================================
# 1. Halton Draws 생성 및 검증
# ============================================================================

def test_halton_draws():
    """
    Halton draws 생성 테스트
    
    Apollo R 코드:
    apollo_draws = list(
        interDrawsType="halton",
        interNDraws=1000,
        interNormDraws=c("eta")
    )
    """
    print("\n" + "="*70)
    print("TEST 1: Halton Draws 생성")
    print("="*70)
    
    n_draws = 1000
    
    # Halton 시퀀스 생성
    sampler = qmc.Halton(d=1, scramble=True, seed=42)
    uniform_draws = sampler.random(n=n_draws)
    halton_draws = norm.ppf(uniform_draws).flatten()
    
    print(f"생성된 draws 수: {len(halton_draws)}")
    print(f"평균: {halton_draws.mean():.6f} (기대값: 0)")
    print(f"표준편차: {halton_draws.std():.6f} (기대값: 1)")
    print(f"최소값: {halton_draws.min():.3f}")
    print(f"최대값: {halton_draws.max():.3f}")
    
    # 정규분포 검증
    from scipy.stats import kstest
    ks_stat, p_value = kstest(halton_draws, 'norm')
    print(f"\nKolmogorov-Smirnov 검정:")
    print(f"  통계량: {ks_stat:.6f}")
    print(f"  p-value: {p_value:.6f}")
    
    if p_value > 0.05:
        print("  ✓ 정규분포를 따릅니다 (p > 0.05)")
    else:
        print("  ✗ 정규분포를 따르지 않습니다 (p < 0.05)")
    
    # 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(halton_draws, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(-4, 4, 100)
    plt.plot(x, norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Halton Draws Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    from scipy.stats import probplot
    probplot(halton_draws, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(halton_draws[:100], 'o-', markersize=3)
    plt.xlabel('Draw Index')
    plt.ylabel('Value')
    plt.title('First 100 Draws')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tests/halton_draws_validation.png', dpi=150)
    print(f"\n그래프 저장: tests/halton_draws_validation.png")
    
    return halton_draws


# ============================================================================
# 2. Ordered Probit 측정모델 검증
# ============================================================================

def test_ordered_probit():
    """
    Ordered Probit 모델 테스트
    
    Apollo R 코드:
    op_settings = list(
        outcomeOrdered = Q13CurrentThreatToSelf,
        V = zeta_Q13*LV,
        tau = c(tau_Q13_1, tau_Q13_2, tau_Q13_3, tau_Q13_4)
    )
    P[["indic_Q13"]] = apollo_op(op_settings, functionality)
    """
    print("\n" + "="*70)
    print("TEST 2: Ordered Probit 측정모델")
    print("="*70)
    
    # 파라미터 설정
    zeta = 1.0  # 요인적재량
    tau = np.array([-2.0, -1.0, 1.0, 2.0])  # 임계값 (4개 = 5점 척도)
    
    print(f"요인적재량 (ζ): {zeta}")
    print(f"임계값 (τ): {tau}")
    
    # 잠재변수 값 범위
    lv_values = np.linspace(-3, 3, 100)
    
    # 각 범주의 확률 계산
    probs = {k: [] for k in range(1, 6)}
    
    for lv in lv_values:
        # P(Y=k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
        for k in range(1, 6):
            if k == 1:
                prob = norm.cdf(tau[0] - zeta * lv)
            elif k == 5:
                prob = 1 - norm.cdf(tau[3] - zeta * lv)
            else:
                prob = norm.cdf(tau[k-1] - zeta * lv) - norm.cdf(tau[k-2] - zeta * lv)
            
            probs[k].append(prob)
    
    # 시각화
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for k in range(1, 6):
        plt.plot(lv_values, probs[k], label=f'Category {k}', 
                linewidth=2, color=colors[k-1])
    
    plt.xlabel('Latent Variable (LV)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Ordered Probit: Category Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 임계값 표시
    for i, t in enumerate(tau):
        plt.axvline(x=t/zeta, color='red', linestyle=':', alpha=0.5)
        plt.text(t/zeta, 0.95, f'τ{i+1}', fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('tests/ordered_probit_validation.png', dpi=150)
    print(f"\n그래프 저장: tests/ordered_probit_validation.png")
    
    # 확률 합 검증
    print("\n확률 합 검증 (각 LV 값에서 모든 범주 확률의 합 = 1):")
    for i in [0, 25, 50, 75, 99]:
        lv = lv_values[i]
        total_prob = sum(probs[k][i] for k in range(1, 6))
        print(f"  LV={lv:6.2f}: Σp = {total_prob:.6f}")
    
    return probs


# ============================================================================
# 3. 구조방정식 검증
# ============================================================================

def test_structural_equation():
    """
    구조방정식 테스트
    
    Apollo R 코드:
    apollo_randCoeff=function(apollo_beta, apollo_inputs){
        randcoeff = list()
        randcoeff[["LV"]] = gamma_Age*Age + gamma_Gender*Q1Gender + 
                           gamma_Distance*Distance + gamma_Income*IncomeDummy + 
                           gamma_Experts*Experts + gamma_BP*BP + 
                           gamma_Charity*Charity + gamma_Certainty*Q12CECertainty +
                           gamma_Cons*Consequentiality + eta
        return(randcoeff)
    }
    """
    print("\n" + "="*70)
    print("TEST 3: 구조방정식")
    print("="*70)
    
    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    n = 500
    
    data = pd.DataFrame({
        'Age': np.random.normal(45, 15, n),
        'Gender': np.random.binomial(1, 0.5, n),
        'Income': np.random.binomial(1, 0.6, n),
    })
    
    # 표준화
    data['Age_std'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
    
    # 파라미터
    gamma_Age = 0.3
    gamma_Gender = -0.2
    gamma_Income = 0.4
    
    print(f"파라미터:")
    print(f"  γ_Age: {gamma_Age}")
    print(f"  γ_Gender: {gamma_Gender}")
    print(f"  γ_Income: {gamma_Income}")
    
    # 잠재변수 생성
    eta = np.random.normal(0, 1, n)
    data['LV'] = (
        gamma_Age * data['Age_std'] +
        gamma_Gender * data['Gender'] +
        gamma_Income * data['Income'] +
        eta
    )
    
    print(f"\n잠재변수 통계:")
    print(f"  평균: {data['LV'].mean():.3f}")
    print(f"  표준편차: {data['LV'].std():.3f}")
    print(f"  최소값: {data['LV'].min():.3f}")
    print(f"  최대값: {data['LV'].max():.3f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # LV 분포
    axes[0, 0].hist(data['LV'], bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(data['LV'].min(), data['LV'].max(), 100)
    axes[0, 0].plot(x, norm.pdf(x, data['LV'].mean(), data['LV'].std()), 
                    'r-', linewidth=2, label='Normal fit')
    axes[0, 0].set_xlabel('Latent Variable')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('LV Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Age vs LV
    axes[0, 1].scatter(data['Age_std'], data['LV'], alpha=0.5, s=20)
    axes[0, 1].set_xlabel('Age (standardized)')
    axes[0, 1].set_ylabel('Latent Variable')
    axes[0, 1].set_title(f'Age vs LV (γ={gamma_Age})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gender vs LV
    axes[1, 0].boxplot([data[data['Gender']==0]['LV'], 
                        data[data['Gender']==1]['LV']],
                       labels=['Male (0)', 'Female (1)'])
    axes[1, 0].set_ylabel('Latent Variable')
    axes[1, 0].set_title(f'Gender vs LV (γ={gamma_Gender})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Income vs LV
    axes[1, 1].boxplot([data[data['Income']==0]['LV'], 
                        data[data['Income']==1]['LV']],
                       labels=['Low (0)', 'High (1)'])
    axes[1, 1].set_ylabel('Latent Variable')
    axes[1, 1].set_title(f'Income vs LV (γ={gamma_Income})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tests/structural_equation_validation.png', dpi=150)
    print(f"\n그래프 저장: tests/structural_equation_validation.png")
    
    return data


# ============================================================================
# 4. Binary Probit 선택모델 검증
# ============================================================================

def test_binary_probit():
    """
    Binary Probit 선택모델 테스트
    
    Apollo R 코드:
    op_settings = list(
        outcomeOrdered = Q6ResearchResponse,
        V = intercept + b_bid*Q6Bid + lambda*LV,
        tau = list(-100, 0),
        coding = c(-1, 0, 1)
    )
    """
    print("\n" + "="*70)
    print("TEST 4: Binary Probit 선택모델")
    print("="*70)
    
    # 파라미터
    intercept = 0.5
    b_bid = -2.0
    lambda_lv = 1.5
    
    print(f"파라미터:")
    print(f"  절편: {intercept}")
    print(f"  β_bid: {b_bid}")
    print(f"  λ: {lambda_lv}")
    
    # 가격 범위
    bid_values = np.linspace(0, 1.5, 100)
    
    # 다양한 LV 값에 대한 선택 확률
    lv_values = [-1, 0, 1, 2]
    
    plt.figure(figsize=(10, 6))
    
    for lv in lv_values:
        probs = []
        for bid in bid_values:
            V = intercept + b_bid * bid + lambda_lv * lv
            prob = norm.cdf(V)
            probs.append(prob)
        
        plt.plot(bid_values, probs, label=f'LV = {lv}', linewidth=2)
    
    plt.xlabel('Bid (Price)', fontsize=12)
    plt.ylabel('P(Accept)', fontsize=12)
    plt.title('Binary Probit: Choice Probability', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('tests/binary_probit_validation.png', dpi=150)
    print(f"\n그래프 저장: tests/binary_probit_validation.png")
    
    # WTP 계산
    print(f"\nWTP 계산 (P(Accept) = 0.5일 때의 가격):")
    for lv in lv_values:
        # V = 0일 때 P = 0.5
        # intercept + b_bid * WTP + lambda * LV = 0
        # WTP = -(intercept + lambda * LV) / b_bid
        wtp = -(intercept + lambda_lv * lv) / b_bid
        print(f"  LV={lv:2d}: WTP = {wtp:.3f}")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ICLV 컴포넌트 검증 테스트")
    print("King (2022) Apollo R 코드 → Python 변환")
    print("="*70)
    
    # 1. Halton Draws
    halton_draws = test_halton_draws()
    
    # 2. Ordered Probit
    probs = test_ordered_probit()
    
    # 3. 구조방정식
    data = test_structural_equation()
    
    # 4. Binary Probit
    test_binary_probit()
    
    print("\n" + "="*70)
    print("모든 컴포넌트 테스트 완료! ✓")
    print("="*70)
    print("\n생성된 파일:")
    print("  - tests/halton_draws_validation.png")
    print("  - tests/ordered_probit_validation.png")
    print("  - tests/structural_equation_validation.png")
    print("  - tests/binary_probit_validation.png")

