"""
Simultaneous Estimation Choice Model Test

King (2022) Apollo R 코드 기반 동시 추정 선택모델 테스트

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

# ICLV 모델 컴포넌트 import
from src.analysis.hybrid_choice_model.iclv_models import (
    OrderedProbitMeasurement,
    LatentVariableRegression,
    BinaryProbitChoice,
    SimultaneousEstimator,
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig,
    ICLVConfig,
    create_iclv_config
)


def generate_synthetic_data(n_individuals=100, n_indicators=5, seed=42):
    """
    합성 데이터 생성
    
    King (2022) 스타일의 ICLV 데이터:
    1. 사회인구학적 변수 → 잠재변수 (구조모델)
    2. 잠재변수 → 관측지표 (측정모델)
    3. 잠재변수 + 속성 → 선택 (선택모델)
    """
    np.random.seed(seed)
    
    print("\n" + "="*70)
    print("합성 데이터 생성")
    print("="*70)
    
    # 1. 사회인구학적 변수
    age = np.random.normal(45, 15, n_individuals)
    age_std = (age - age.mean()) / age.std()
    
    gender = np.random.binomial(1, 0.5, n_individuals)
    
    income = np.random.normal(50000, 20000, n_individuals)
    income_std = (income - income.mean()) / income.std()
    
    # 2. 구조방정식: LV = γ*X + η
    gamma_age = 0.5
    gamma_gender = -0.3
    gamma_income = 0.2
    
    eta = np.random.normal(0, 1, n_individuals)
    
    latent_var = (
        gamma_age * age_std +
        gamma_gender * gender +
        gamma_income * income_std +
        eta
    )
    
    print(f"\n구조모델 파라미터:")
    print(f"  γ_age: {gamma_age}")
    print(f"  γ_gender: {gamma_gender}")
    print(f"  γ_income: {gamma_income}")
    print(f"  LV 평균: {latent_var.mean():.3f}, 표준편차: {latent_var.std():.3f}")
    
    # 3. 측정모델: Ordered Probit
    zeta = np.array([1.0, 1.2, 0.9, 1.1, 0.8])
    tau = np.array([
        [-2.0, -1.0, 1.0, 2.0],
        [-2.0, -1.0, 1.0, 2.0],
        [-2.0, -1.0, 1.0, 2.0],
        [-2.0, -1.0, 1.0, 2.0],
        [-2.0, -1.0, 1.0, 2.0]
    ])
    
    indicators = np.zeros((n_individuals, n_indicators))
    
    for i in range(n_individuals):
        lv = latent_var[i]
        
        for j in range(n_indicators):
            V = zeta[j] * lv
            
            # 각 범주의 확률 계산
            probs = []
            for k in range(5):
                if k == 0:
                    prob = norm.cdf(tau[j, 0] - V)
                elif k == 4:
                    prob = 1 - norm.cdf(tau[j, 3] - V)
                else:
                    prob = norm.cdf(tau[j, k] - V) - norm.cdf(tau[j, k-1] - V)
                probs.append(prob)
            
            # 범주 샘플링
            indicators[i, j] = np.random.choice([1, 2, 3, 4, 5], p=probs)
    
    print(f"\n측정모델 파라미터:")
    print(f"  ζ (요인적재량): {zeta}")
    print(f"  지표 평균: {indicators.mean(axis=0)}")
    
    # 4. 선택모델: Binary Probit
    intercept = 0.5
    beta_price = -2.0
    beta_quality = 0.3
    lambda_lv = 1.5
    
    price = np.random.uniform(0, 1.5, n_individuals)
    quality = np.random.uniform(0, 1, n_individuals)
    
    V_choice = intercept + beta_price * price + beta_quality * quality + lambda_lv * latent_var
    prob_yes = norm.cdf(V_choice)
    
    choice = np.random.binomial(1, prob_yes)
    
    print(f"\n선택모델 파라미터:")
    print(f"  절편: {intercept}")
    print(f"  β_price: {beta_price}")
    print(f"  β_quality: {beta_quality}")
    print(f"  λ: {lambda_lv}")
    print(f"  선택 비율: {choice.mean():.3f}")
    
    # 5. 데이터프레임 생성
    data = pd.DataFrame({
        'individual_id': range(n_individuals),
        'age_std': age_std,
        'gender': gender,
        'income_std': income_std,
        'price': price,
        'quality': quality,
        'choice': choice
    })
    
    # 지표 추가
    for j in range(n_indicators):
        data[f'indicator_{j+1}'] = indicators[:, j]
    
    # 진짜 잠재변수 (검증용)
    data['true_lv'] = latent_var
    
    # 진짜 파라미터
    true_params = {
        'measurement': {
            'zeta': zeta,
            'tau': tau
        },
        'structural': {
            'gamma': np.array([gamma_age, gamma_gender, gamma_income])
        },
        'choice': {
            'intercept': intercept,
            'beta': np.array([beta_price, beta_quality]),
            'lambda': lambda_lv
        }
    }
    
    return data, true_params


def test_individual_components():
    """개별 컴포넌트 테스트"""
    print("\n" + "="*70)
    print("TEST 1: 개별 컴포넌트 테스트")
    print("="*70)
    
    # 데이터 생성
    data, true_params = generate_synthetic_data(n_individuals=100)
    
    # 1. 측정모델 테스트
    print("\n[1] 측정모델 테스트")
    measurement_config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['indicator_1', 'indicator_2', 'indicator_3', 'indicator_4', 'indicator_5'],
        n_categories=5
    )
    
    measurement_model = OrderedProbitMeasurement(measurement_config)
    
    ll_measurement = measurement_model.log_likelihood(
        data,
        data['true_lv'].values,
        true_params['measurement']
    )
    
    print(f"  측정모델 로그우도: {ll_measurement:.2f}")
    
    # 2. 구조모델 테스트
    print("\n[2] 구조모델 테스트")
    structural_config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std']
    )
    
    structural_model = LatentVariableRegression(structural_config)
    
    # Sequential 추정
    results_structural = structural_model.fit(data, data['true_lv'].values)
    
    print(f"  추정된 γ: {results_structural['gamma']}")
    print(f"  진짜 γ: {true_params['structural']['gamma']}")
    print(f"  R²: {results_structural['r_squared']:.3f}")
    
    # 3. 선택모델 테스트
    print("\n[3] 선택모델 테스트")
    choice_config = ChoiceConfig(
        choice_attributes=['price', 'quality'],
        price_variable='price'
    )
    
    choice_model = BinaryProbitChoice(choice_config)
    
    ll_choice = choice_model.log_likelihood(
        data,
        data['true_lv'].values,
        true_params['choice']
    )
    
    print(f"  선택모델 로그우도: {ll_choice:.2f}")
    
    # 확률 예측
    probs = choice_model.predict_probabilities(
        data,
        data['true_lv'].values,
        true_params['choice']
    )
    
    print(f"  예측 확률 평균: {probs.mean():.3f}")
    print(f"  실제 선택 비율: {data['choice'].mean():.3f}")
    
    print("\n✅ 모든 개별 컴포넌트 테스트 통과!")


def test_choice_model_wtp():
    """선택모델 WTP 계산 테스트"""
    print("\n" + "="*70)
    print("TEST 2: WTP 계산 테스트")
    print("="*70)
    
    # 데이터 생성
    data, true_params = generate_synthetic_data(n_individuals=100)
    
    # 선택모델 생성
    choice_config = ChoiceConfig(
        choice_attributes=['price', 'quality'],
        price_variable='price'
    )
    
    choice_model = BinaryProbitChoice(choice_config)
    
    # WTP 계산
    wtp_quality = choice_model.calculate_wtp(true_params['choice'], 'quality')
    
    print(f"\nWTP for Quality:")
    print(f"  계산된 WTP: {wtp_quality:.3f}")
    print(f"  예상 WTP: {-true_params['choice']['beta'][1] / true_params['choice']['beta'][0]:.3f}")
    
    # 이론적 WTP
    beta_quality = true_params['choice']['beta'][1]
    beta_price = true_params['choice']['beta'][0]
    theoretical_wtp = -beta_quality / beta_price
    
    print(f"  이론적 WTP: {theoretical_wtp:.3f}")
    
    assert np.isclose(wtp_quality, theoretical_wtp), "WTP 계산 오류!"
    
    print("\n✅ WTP 계산 테스트 통과!")


def test_choice_model_sensitivity():
    """선택모델 민감도 분석"""
    print("\n" + "="*70)
    print("TEST 3: 선택모델 민감도 분석")
    print("="*70)
    
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
    
    # 선택모델 생성
    choice_config = ChoiceConfig(
        choice_attributes=['price'],
        price_variable='price'
    )
    
    choice_model = BinaryProbitChoice(choice_config)
    
    # 시각화
    plt.figure(figsize=(10, 6))
    
    for lv in lv_values:
        probs = []
        for price in price_values:
            test_data = pd.DataFrame({'price': [price]})
            prob = choice_model.predict_probabilities(test_data, lv, params)[0]
            probs.append(prob)
        
        plt.plot(price_values, probs, label=f'LV = {lv}', linewidth=2)
    
    plt.xlabel('Price', fontsize=12)
    plt.ylabel('P(Choice = Yes)', fontsize=12)
    plt.title('Binary Probit Choice Model: Price Sensitivity by Latent Variable', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tests/choice_model_sensitivity.png', dpi=300, bbox_inches='tight')
    print("\n시각화 저장: tests/choice_model_sensitivity.png")
    
    print("\n✅ 민감도 분석 완료!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ICLV 동시 추정 선택모델 테스트")
    print("King (2022) Apollo R 코드 기반")
    print("="*70)
    
    # 테스트 실행
    test_individual_components()
    test_choice_model_wtp()
    test_choice_model_sensitivity()
    
    print("\n" + "="*70)
    print("✅ 모든 테스트 통과!")
    print("="*70)

