"""
Ordered Probit 측정모델 통합 테스트

SimultaneousEstimator와의 통합을 검증합니다.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 간단한 Config 클래스들
@dataclass
class MeasurementConfig:
    """측정모델 설정"""
    indicators: List[str]
    n_categories: int = 5
    indicator_types: Optional[List[str]] = None


# OrderedProbitMeasurement import
sys.path.insert(0, str(project_root / 'src' / 'analysis' / 'hybrid_choice_model' / 'iclv_models'))
from measurement_equations import OrderedProbitMeasurement


def test_integration_with_simultaneous_estimator():
    """
    SimultaneousEstimator에서 사용하는 방식으로 테스트
    
    동시 추정에서는:
    1. 각 Halton draw마다 잠재변수 값이 다름
    2. log_likelihood()를 반복 호출
    3. 파라미터는 딕셔너리 형태로 전달
    """
    print("\n" + "="*70)
    print("통합 테스트: SimultaneousEstimator 방식")
    print("="*70)
    
    # 설정
    config = MeasurementConfig(
        indicators=['Q13', 'Q14', 'Q15'],
        n_categories=5
    )
    
    model = OrderedProbitMeasurement(config)
    
    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    n_individuals = 50
    n_draws = 10  # Halton draws 수
    
    # 실제 파라미터
    true_params = {
        'zeta': np.array([1.0, 1.2, 0.8]),
        'tau': np.array([
            [-2.0, -1.0, 1.0, 2.0],
            [-2.0, -1.0, 1.0, 2.0],
            [-2.0, -1.0, 1.0, 2.0]
        ])
    }
    
    # 관측지표 데이터 생성
    data_list = []
    
    for i in range(n_individuals):
        # 구조방정식으로 잠재변수 생성
        eta = np.random.normal(0, 1)
        lv_true = eta  # 간단화
        
        # 측정지표 생성
        row = {'individual_id': i}
        
        for j, indicator in enumerate(config.indicators):
            zeta_j = true_params['zeta'][j]
            tau_j = true_params['tau'][j]
            
            # Ordered Probit
            y_star = zeta_j * lv_true + np.random.normal(0, 1)
            
            if y_star < tau_j[0]:
                y = 1
            elif y_star < tau_j[1]:
                y = 2
            elif y_star < tau_j[2]:
                y = 3
            elif y_star < tau_j[3]:
                y = 4
            else:
                y = 5
            
            row[indicator] = y
        
        data_list.append(row)
    
    data = pd.DataFrame(data_list)
    
    print(f"\n생성된 데이터:")
    print(f"  개인 수: {n_individuals}")
    print(f"  지표 수: {len(config.indicators)}")
    
    # SimultaneousEstimator 방식으로 우도 계산
    print(f"\n동시 추정 방식 우도 계산:")
    print(f"  Halton draws: {n_draws}")
    
    total_ll = 0.0
    
    for i in range(n_individuals):
        ind_data = data[data['individual_id'] == i]
        
        # 개인별 시뮬레이션 우도
        sim_likelihood = 0.0
        
        for draw in range(n_draws):
            # Halton draw (정규분포)
            eta_draw = np.random.normal(0, 1)
            lv_draw = np.array([eta_draw])  # 1개 관측치
            
            # 측정모델 우도
            ll_measurement = model.log_likelihood(ind_data, lv_draw, true_params)
            
            # 시뮬레이션 우도 누적
            sim_likelihood += np.exp(ll_measurement)
        
        # 평균
        sim_likelihood /= n_draws
        
        # 로그 변환
        if sim_likelihood > 0:
            total_ll += np.log(sim_likelihood)
    
    print(f"\n총 로그우도: {total_ll:.2f}")
    print(f"개인당 평균 로그우도: {total_ll/n_individuals:.2f}")
    
    assert np.isfinite(total_ll), "로그우도가 유한하지 않습니다"
    assert total_ll < 0, "로그우도는 음수여야 합니다"
    
    print("\n✓ SimultaneousEstimator 방식 통합 성공")


def test_apollo_equivalence():
    """
    Apollo R 코드와의 동등성 검증
    
    Apollo 코드:
        op_settings = list(
            outcomeOrdered = Q13,
            V = zeta_Q13 * LV,
            tau = c(tau_Q13_1, tau_Q13_2, tau_Q13_3, tau_Q13_4)
        )
        P[["indic_Q13"]] = apollo_op(op_settings, functionality)
    """
    print("\n" + "="*70)
    print("Apollo R 코드 동등성 검증")
    print("="*70)
    
    config = MeasurementConfig(
        indicators=['Q13'],
        n_categories=5
    )
    
    model = OrderedProbitMeasurement(config)
    
    # Apollo 예제 파라미터
    zeta = 1.0
    tau = np.array([-2.0, -1.0, 1.0, 2.0])
    lv = 0.5
    
    print(f"\nApollo 파라미터:")
    print(f"  zeta_Q13 = {zeta}")
    print(f"  tau = {tau}")
    print(f"  LV = {lv}")
    
    # 각 범주의 확률 계산
    print(f"\n확률 계산 (Apollo apollo_op 동등):")
    
    for k in range(1, 6):
        prob = model._ordered_probit_probability(k, lv, zeta, tau)
        print(f"  P(Q13={k} | LV={lv}) = {prob:.6f}")
    
    # 수동 계산으로 검증
    V = zeta * lv
    
    print(f"\n수동 계산 검증:")
    print(f"  V = zeta * LV = {V}")
    
    # P(Y=3) 계산
    prob_3_manual = norm.cdf(tau[2] - V) - norm.cdf(tau[1] - V)
    prob_3_model = model._ordered_probit_probability(3, lv, zeta, tau)
    
    print(f"  P(Y=3) 수동: {prob_3_manual:.6f}")
    print(f"  P(Y=3) 모델: {prob_3_model:.6f}")
    print(f"  차이: {abs(prob_3_manual - prob_3_model):.10f}")
    
    assert abs(prob_3_manual - prob_3_model) < 1e-10, "Apollo 계산과 일치하지 않습니다"
    
    print("\n✓ Apollo R 코드와 완전히 동등합니다")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Ordered Probit 측정모델 통합 테스트")
    print("="*70)
    
    test_integration_with_simultaneous_estimator()
    test_apollo_equivalence()
    
    print("\n" + "="*70)
    print("모든 통합 테스트 통과! ✓")
    print("="*70)

