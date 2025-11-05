"""
Standalone Choice Model Test

King (2022) Apollo R 코드 기반 Binary Probit 선택모델 단독 테스트

Author: Sugar Substitute Research Team
Date: 2025-11-05
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 직접 import
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import (
    BinaryProbitChoice,
    ChoiceConfig
)


def test_basic_functionality():
    """기본 기능 테스트"""
    print("\n" + "="*70)
    print("TEST 1: 기본 기능 테스트")
    print("="*70)
    
    # 설정
    config = ChoiceConfig(
        choice_attributes=['price', 'quality'],
        choice_type='binary',
        price_variable='price'
    )
    
    # 모델 생성
    model = BinaryProbitChoice(config)
    print("\n✅ 모델 생성 성공")
    
    # 테스트 데이터
    data = pd.DataFrame({
        'price': [0.5, 1.0, 1.5],
        'quality': [0.3, 0.5, 0.7],
        'choice': [1, 1, 0]
    })
    
    # 파라미터
    params = {
        'intercept': 0.5,
        'beta': np.array([-2.0, 0.3]),
        'lambda': 1.5
    }
    
    # 잠재변수
    lv = np.array([0.5, 0.0, -0.5])
    
    # 로그우도 계산
    ll = model.log_likelihood(data, lv, params)
    print(f"\n로그우도: {ll:.4f}")
    
    # 확률 예측
    probs = model.predict_probabilities(data, lv, params)
    print(f"\n예측 확률:")
    for i, prob in enumerate(probs):
        print(f"  관측 {i+1}: {prob:.4f} (실제 선택: {data['choice'].iloc[i]})")
    
    # 선택 예측
    predictions = model.predict(data, lv, params)
    print(f"\n예측 선택: {predictions}")
    print(f"실제 선택: {data['choice'].values}")
    
    print("\n✅ 기본 기능 테스트 통과!")


def test_wtp_calculation():
    """WTP 계산 테스트"""
    print("\n" + "="*70)
    print("TEST 2: WTP 계산 테스트")
    print("="*70)
    
    # 설정
    config = ChoiceConfig(
        choice_attributes=['price', 'quality'],
        choice_type='binary',
        price_variable='price'
    )
    
    model = BinaryProbitChoice(config)
    
    # 파라미터
    beta_price = -2.0
    beta_quality = 0.6
    
    params = {
        'intercept': 0.0,
        'beta': np.array([beta_price, beta_quality]),
        'lambda': 1.0
    }
    
    # WTP 계산
    wtp = model.calculate_wtp(params, 'quality')
    
    # 이론적 WTP
    theoretical_wtp = -beta_quality / beta_price
    
    print(f"\nWTP for Quality:")
    print(f"  계산된 WTP: {wtp:.4f}")
    print(f"  이론적 WTP: {theoretical_wtp:.4f}")
    print(f"  β_quality: {beta_quality}")
    print(f"  β_price: {beta_price}")
    
    assert np.isclose(wtp, theoretical_wtp), "WTP 계산 오류!"
    
    print("\n✅ WTP 계산 테스트 통과!")


def test_price_sensitivity():
    """가격 민감도 분석"""
    print("\n" + "="*70)
    print("TEST 3: 가격 민감도 분석")
    print("="*70)
    
    # 설정
    config = ChoiceConfig(
        choice_attributes=['price'],
        choice_type='binary',
        price_variable='price'
    )
    
    model = BinaryProbitChoice(config)
    
    # 파라미터
    intercept = 0.5
    beta_price = -2.0
    lambda_lv = 1.5
    
    params = {
        'intercept': intercept,
        'beta': np.array([beta_price]),
        'lambda': lambda_lv
    }
    
    # 가격 범위
    price_values = np.linspace(0, 1.5, 100)
    
    # 다양한 LV 값
    lv_values = [-1, 0, 1, 2]
    
    # 시각화
    plt.figure(figsize=(10, 6))
    
    for lv in lv_values:
        probs = []
        for price in price_values:
            test_data = pd.DataFrame({'price': [price]})
            prob = model.predict_probabilities(test_data, lv, params)[0]
            probs.append(prob)
        
        plt.plot(price_values, probs, label=f'LV = {lv}', linewidth=2)
    
    plt.xlabel('Price', fontsize=12)
    plt.ylabel('P(Choice = Yes)', fontsize=12)
    plt.title('Binary Probit Choice Model: Price Sensitivity by Latent Variable', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tests/choice_model_price_sensitivity.png', dpi=300, bbox_inches='tight')
    print("\n시각화 저장: tests/choice_model_price_sensitivity.png")
    
    print("\n✅ 가격 민감도 분석 완료!")


def test_apollo_r_validation():
    """
    Apollo R 코드와의 검증
    
    King (2022) 논문의 예시:
    V = intercept + b_bid*Bid + lambda*LV
    """
    print("\n" + "="*70)
    print("TEST 4: Apollo R 코드 검증")
    print("="*70)
    
    # Apollo R 코드 파라미터
    intercept = 0.5
    b_bid = -2.0
    lambda_lv = 1.5
    
    # 테스트 케이스
    test_cases = [
        {'bid': 0.0, 'lv': 0.0, 'expected_V': 0.5},
        {'bid': 1.0, 'lv': 0.0, 'expected_V': -1.5},
        {'bid': 0.0, 'lv': 1.0, 'expected_V': 2.0},
        {'bid': 1.0, 'lv': 1.0, 'expected_V': 0.0},
    ]
    
    # 설정
    config = ChoiceConfig(
        choice_attributes=['bid'],
        choice_type='binary',
        price_variable='bid'
    )
    
    model = BinaryProbitChoice(config)
    
    params = {
        'intercept': intercept,
        'beta': np.array([b_bid]),
        'lambda': lambda_lv
    }
    
    print(f"\n파라미터:")
    print(f"  절편: {intercept}")
    print(f"  β_bid: {b_bid}")
    print(f"  λ: {lambda_lv}")
    
    print(f"\n검증 결과:")
    for i, case in enumerate(test_cases):
        data = pd.DataFrame({'bid': [case['bid']]})
        lv = case['lv']
        
        # 효용 계산 (수동)
        V_manual = intercept + b_bid * case['bid'] + lambda_lv * lv
        
        # 확률 계산
        prob = model.predict_probabilities(data, lv, params)[0]
        expected_prob = norm.cdf(case['expected_V'])
        
        print(f"\n  케이스 {i+1}:")
        print(f"    Bid: {case['bid']}, LV: {lv}")
        print(f"    V (계산): {V_manual:.4f}")
        print(f"    V (예상): {case['expected_V']:.4f}")
        print(f"    P(Yes): {prob:.4f}")
        print(f"    P(Yes) 예상: {expected_prob:.4f}")
        
        assert np.isclose(V_manual, case['expected_V']), f"케이스 {i+1} 효용 계산 오류!"
        assert np.isclose(prob, expected_prob), f"케이스 {i+1} 확률 계산 오류!"
    
    print("\n✅ Apollo R 코드 검증 통과!")


def test_latent_variable_effect():
    """잠재변수 효과 분석"""
    print("\n" + "="*70)
    print("TEST 5: 잠재변수 효과 분석")
    print("="*70)
    
    # 설정
    config = ChoiceConfig(
        choice_attributes=['price'],
        choice_type='binary',
        price_variable='price'
    )
    
    model = BinaryProbitChoice(config)
    
    # 고정 가격
    fixed_price = 1.0
    
    # 다양한 λ 값
    lambda_values = [0.5, 1.0, 1.5, 2.0]
    
    # LV 범위
    lv_range = np.linspace(-2, 2, 100)
    
    # 시각화
    plt.figure(figsize=(10, 6))
    
    for lambda_val in lambda_values:
        params = {
            'intercept': 0.0,
            'beta': np.array([-2.0]),
            'lambda': lambda_val
        }
        
        probs = []
        for lv in lv_range:
            test_data = pd.DataFrame({'price': [fixed_price]})
            prob = model.predict_probabilities(test_data, lv, params)[0]
            probs.append(prob)
        
        plt.plot(lv_range, probs, label=f'λ = {lambda_val}', linewidth=2)
    
    plt.xlabel('Latent Variable (LV)', fontsize=12)
    plt.ylabel('P(Choice = Yes)', fontsize=12)
    plt.title(f'Latent Variable Effect on Choice Probability (Price = {fixed_price})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('tests/choice_model_lv_effect.png', dpi=300, bbox_inches='tight')
    print("\n시각화 저장: tests/choice_model_lv_effect.png")
    
    print("\n✅ 잠재변수 효과 분석 완료!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Binary Probit 선택모델 단독 테스트")
    print("King (2022) Apollo R 코드 기반")
    print("="*70)
    
    # 테스트 실행
    test_basic_functionality()
    test_wtp_calculation()
    test_price_sensitivity()
    test_apollo_r_validation()
    test_latent_variable_effect()
    
    print("\n" + "="*70)
    print("✅ 모든 테스트 통과!")
    print("="*70)
    print("\n생성된 파일:")
    print("  - tests/choice_model_price_sensitivity.png")
    print("  - tests/choice_model_lv_effect.png")

