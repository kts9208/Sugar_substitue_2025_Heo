"""
조절효과 구현 예시

이 스크립트는 ICLV 선택모델에서 조절효과를 구현하는 방법을 보여줍니다.

예시:
- 주 잠재변수: 구매의도 (Purchase Intention)
- 조절변수 1: 가격수준 (Perceived Price) - 부적 조절 예상
- 조절변수 2: 영양지식 (Nutrition Knowledge) - 정적 조절 예상
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# ============================================================================
# 1. 조절효과 없는 기본 모델
# ============================================================================

def basic_choice_model(data, lv_main, params):
    """
    기본 선택모델 (조절효과 없음)
    
    V = intercept + β·X + λ·LV_main
    P(choice=1) = Φ(V)
    """
    intercept = params['intercept']
    beta = params['beta']
    lambda_lv = params['lambda']
    
    X = data[['sugar_free', 'health_label', 'price']].values
    
    # 효용 계산
    V = intercept + X @ beta + lambda_lv * lv_main
    
    # 확률 계산
    prob = norm.cdf(V)
    
    return prob


# ============================================================================
# 2. 조절효과 포함 모델
# ============================================================================

def moderation_choice_model(data, latent_vars, params):
    """
    조절효과 포함 선택모델
    
    V = intercept + β·X + λ_main·PI + λ_mod_price·(PI×PP) + λ_mod_knowledge·(PI×NK)
    P(choice=1) = Φ(V)
    
    Args:
        data: 선택 속성 데이터
        latent_vars: 잠재변수 딕셔너리
            {
                'purchase_intention': float,
                'perceived_price': float,
                'nutrition_knowledge': float
            }
        params: 파라미터 딕셔너리
            {
                'intercept': float,
                'beta': np.ndarray,
                'lambda_main': float,
                'lambda_mod_price': float,
                'lambda_mod_knowledge': float
            }
    """
    intercept = params['intercept']
    beta = params['beta']
    lambda_main = params['lambda_main']
    lambda_mod_price = params['lambda_mod_price']
    lambda_mod_knowledge = params['lambda_mod_knowledge']
    
    X = data[['sugar_free', 'health_label', 'price']].values
    
    # 잠재변수 추출
    lv_main = latent_vars['purchase_intention']
    lv_mod_price = latent_vars['perceived_price']
    lv_mod_knowledge = latent_vars['nutrition_knowledge']
    
    # 효용 계산 (조절효과 포함)
    V = (intercept + 
         X @ beta + 
         lambda_main * lv_main +                              # 주효과
         lambda_mod_price * (lv_main * lv_mod_price) +        # 가격수준 조절
         lambda_mod_knowledge * (lv_main * lv_mod_knowledge)) # 영양지식 조절
    
    # 확률 계산
    prob = norm.cdf(V)
    
    return prob


# ============================================================================
# 3. 예시 실행
# ============================================================================

def example_1_basic_vs_moderation():
    """예시 1: 기본 모델 vs 조절효과 모델 비교"""
    
    print("="*70)
    print("예시 1: 기본 모델 vs 조절효과 모델")
    print("="*70)
    
    # 데이터 생성
    data = pd.DataFrame({
        'sugar_free': [1],
        'health_label': [1],
        'price': [0.5]
    })
    
    # 기본 모델 파라미터
    params_basic = {
        'intercept': 0.0,
        'beta': np.array([0.5, 0.3, -1.0]),
        'lambda': 1.0
    }
    
    # 조절효과 모델 파라미터
    params_mod = {
        'intercept': 0.0,
        'beta': np.array([0.5, 0.3, -1.0]),
        'lambda_main': 1.0,
        'lambda_mod_price': -0.3,      # 부적 조절
        'lambda_mod_knowledge': 0.2    # 정적 조절
    }
    
    # 시나리오 1: 가격수준 낮음, 영양지식 높음
    print("\n시나리오 1: 가격수준 낮음 (-1), 영양지식 높음 (+1)")
    print("-" * 70)
    
    lv_main = 1.0
    latent_vars_1 = {
        'purchase_intention': 1.0,
        'perceived_price': -1.0,      # 낮음 (저렴하다고 인식)
        'nutrition_knowledge': 1.0    # 높음
    }
    
    prob_basic = basic_choice_model(data, lv_main, params_basic)
    prob_mod = moderation_choice_model(data, latent_vars_1, params_mod)
    
    print(f"  기본 모델 선택 확률: {prob_basic[0]:.4f}")
    print(f"  조절효과 모델 선택 확률: {prob_mod[0]:.4f}")
    print(f"  차이: {prob_mod[0] - prob_basic[0]:+.4f}")
    
    # 시나리오 2: 가격수준 높음, 영양지식 낮음
    print("\n시나리오 2: 가격수준 높음 (+1), 영양지식 낮음 (-1)")
    print("-" * 70)
    
    latent_vars_2 = {
        'purchase_intention': 1.0,
        'perceived_price': 1.0,       # 높음 (비싸다고 인식)
        'nutrition_knowledge': -1.0   # 낮음
    }
    
    prob_mod_2 = moderation_choice_model(data, latent_vars_2, params_mod)
    
    print(f"  기본 모델 선택 확률: {prob_basic[0]:.4f}")
    print(f"  조절효과 모델 선택 확률: {prob_mod_2[0]:.4f}")
    print(f"  차이: {prob_mod_2[0] - prob_basic[0]:+.4f}")
    
    print("\n해석:")
    print("  - 시나리오 1: 가격 저렴 + 지식 높음 → 선택 확률 증가")
    print("  - 시나리오 2: 가격 비쌈 + 지식 낮음 → 선택 확률 감소")


def example_2_simple_slopes():
    """예시 2: Simple Slopes Analysis"""
    
    print("\n" + "="*70)
    print("예시 2: Simple Slopes Analysis (단순 기울기 분석)")
    print("="*70)
    
    lambda_main = 1.0
    lambda_mod_price = -0.3
    lambda_mod_knowledge = 0.2
    
    # 가격수준의 조절효과
    print("\n가격수준의 조절효과:")
    print("-" * 70)
    
    for price_level, price_val in [('낮음 (-1SD)', -1), ('평균', 0), ('높음 (+1SD)', 1)]:
        # 구매의도의 효과 = λ_main + λ_mod × M
        slope = lambda_main + lambda_mod_price * price_val
        print(f"  가격수준 {price_level}: 구매의도 효과 = {slope:.3f}")
    
    print("\n해석: 가격수준이 높을수록 구매의도의 효과 감소 (부적 조절)")
    
    # 영양지식의 조절효과
    print("\n영양지식의 조절효과:")
    print("-" * 70)
    
    for knowledge_level, knowledge_val in [('낮음 (-1SD)', -1), ('평균', 0), ('높음 (+1SD)', 1)]:
        slope = lambda_main + lambda_mod_knowledge * knowledge_val
        print(f"  영양지식 {knowledge_level}: 구매의도 효과 = {slope:.3f}")
    
    print("\n해석: 영양지식이 높을수록 구매의도의 효과 증가 (정적 조절)")


def example_3_visualization():
    """예시 3: 조절효과 시각화"""
    
    print("\n" + "="*70)
    print("예시 3: 조절효과 시각화")
    print("="*70)
    
    lambda_main = 1.0
    lambda_mod_price = -0.3
    lambda_mod_knowledge = 0.2
    
    # 구매의도 범위
    purchase_intention = np.linspace(-2, 2, 100)
    
    # 그림 1: 가격수준의 조절효과
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 가격수준 수준별
    for label, price_val, color in [
        ('가격 낮음 (-1SD)', -1, 'green'),
        ('가격 평균', 0, 'blue'),
        ('가격 높음 (+1SD)', 1, 'red')
    ]:
        # 효용 기여도 = (λ_main + λ_mod × M) × PI
        slope = lambda_main + lambda_mod_price * price_val
        utility = slope * purchase_intention
        
        ax1.plot(purchase_intention, utility, label=label, linewidth=2, color=color)
    
    ax1.set_xlabel('구매의도 (Purchase Intention)', fontsize=12)
    ax1.set_ylabel('효용 기여도 (Utility Contribution)', fontsize=12)
    ax1.set_title('가격수준의 조절효과 (부적 조절)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # 영양지식 수준별
    for label, knowledge_val, color in [
        ('지식 낮음 (-1SD)', -1, 'red'),
        ('지식 평균', 0, 'blue'),
        ('지식 높음 (+1SD)', 1, 'green')
    ]:
        slope = lambda_main + lambda_mod_knowledge * knowledge_val
        utility = slope * purchase_intention
        
        ax2.plot(purchase_intention, utility, label=label, linewidth=2, color=color)
    
    ax2.set_xlabel('구매의도 (Purchase Intention)', fontsize=12)
    ax2.set_ylabel('효용 기여도 (Utility Contribution)', fontsize=12)
    ax2.set_title('영양지식의 조절효과 (정적 조절)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('moderation_effects.png', dpi=300, bbox_inches='tight')
    print("\n그래프 저장: moderation_effects.png")
    plt.show()


def example_4_probability_surface():
    """예시 4: 확률 표면 (3D)"""
    
    print("\n" + "="*70)
    print("예시 4: 선택 확률 표면 (구매의도 × 가격수준)")
    print("="*70)
    
    # 데이터 설정
    data = pd.DataFrame({
        'sugar_free': [1],
        'health_label': [1],
        'price': [0.5]
    })
    
    params = {
        'intercept': 0.0,
        'beta': np.array([0.5, 0.3, -1.0]),
        'lambda_main': 1.0,
        'lambda_mod_price': -0.3,
        'lambda_mod_knowledge': 0.0  # 영양지식 고정
    }
    
    # 그리드 생성
    pi_range = np.linspace(-2, 2, 50)
    pp_range = np.linspace(-2, 2, 50)
    PI, PP = np.meshgrid(pi_range, pp_range)
    
    # 확률 계산
    PROB = np.zeros_like(PI)
    for i in range(len(pi_range)):
        for j in range(len(pp_range)):
            latent_vars = {
                'purchase_intention': PI[j, i],
                'perceived_price': PP[j, i],
                'nutrition_knowledge': 0.0
            }
            PROB[j, i] = moderation_choice_model(data, latent_vars, params)[0]
    
    # 3D 플롯
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(PI, PP, PROB, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('구매의도 (PI)', fontsize=12)
    ax.set_ylabel('가격수준 (PP)', fontsize=12)
    ax.set_zlabel('선택 확률', fontsize=12)
    ax.set_title('조절효과: 선택 확률 표면', fontsize=14, fontweight='bold')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig('probability_surface.png', dpi=300, bbox_inches='tight')
    print("\n그래프 저장: probability_surface.png")
    plt.show()


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("조절효과 구현 예시")
    print("="*70)
    
    # 예시 1: 기본 vs 조절효과 모델
    example_1_basic_vs_moderation()
    
    # 예시 2: Simple Slopes
    example_2_simple_slopes()
    
    # 예시 3: 시각화
    example_3_visualization()
    
    # 예시 4: 확률 표면
    example_4_probability_surface()
    
    print("\n" + "="*70)
    print("모든 예시 완료!")
    print("="*70)

