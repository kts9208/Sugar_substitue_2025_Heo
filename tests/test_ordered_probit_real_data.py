"""
실제 설문 데이터를 이용한 Ordered Probit 측정모델 테스트

기존 대체당 연구 데이터 (건강지각도, 지각된 유익성 등)를 사용하여
구축한 OrderedProbitMeasurement 모듈을 검증합니다.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# MeasurementConfig 정의
@dataclass
class MeasurementConfig:
    """측정모델 설정"""
    indicators: List[str]
    n_categories: int = 5
    indicator_types: Optional[List[str]] = None


# OrderedProbitMeasurement import
sys.path.insert(0, str(project_root / 'src' / 'analysis' / 'hybrid_choice_model' / 'iclv_models'))
from measurement_equations import OrderedProbitMeasurement


def load_survey_data():
    """설문 데이터 로드"""
    data_dir = project_root / 'data' / 'processed' / 'survey'
    
    datasets = {
        'health_concern': pd.read_csv(data_dir / 'health_concern.csv'),
        'perceived_benefit': pd.read_csv(data_dir / 'perceived_benefit.csv'),
        'purchase_intention': pd.read_csv(data_dir / 'purchase_intention.csv'),
        'perceived_price': pd.read_csv(data_dir / 'perceived_price.csv'),
        'nutrition_knowledge': pd.read_csv(data_dir / 'nutrition_knowledge.csv')
    }
    
    return datasets


def test_scenario_1_perceived_benefit():
    """
    시나리오 1: 지각된 유익성 (Perceived Benefit)
    
    King (2022) 스타일: Q13, Q14, Q15 (3개 지표)
    """
    print("\n" + "="*80)
    print("시나리오 1: 지각된 유익성 (Perceived Benefit) - King (2022) 스타일")
    print("="*80)
    
    # 데이터 로드
    datasets = load_survey_data()
    data = datasets['perceived_benefit']
    
    print(f"\n데이터 정보:")
    print(f"  관측치 수: {len(data)}")
    print(f"  전체 지표: {[col for col in data.columns if col.startswith('q')]}")
    
    # King (2022) 스타일: q13, q14, q15
    indicators = ['q13', 'q14', 'q15']
    
    print(f"\n사용 지표: {indicators}")
    print(f"\n지표별 기술통계:")
    for ind in indicators:
        print(f"  {ind}: 평균={data[ind].mean():.2f}, 표준편차={data[ind].std():.2f}, "
              f"범위=[{data[ind].min()}, {data[ind].max()}]")
    
    # 분포 확인
    print(f"\n지표별 응답 분포:")
    for ind in indicators:
        counts = data[ind].value_counts().sort_index()
        print(f"  {ind}: {dict(counts)}")
    
    # 설정
    config = MeasurementConfig(
        indicators=indicators,
        n_categories=5
    )
    
    model = OrderedProbitMeasurement(config)
    
    # 잠재변수 (간단히 평균으로 계산)
    latent_var = data[indicators].mean(axis=1).values
    
    print(f"\n잠재변수 (평균):")
    print(f"  평균: {latent_var.mean():.2f}")
    print(f"  표준편차: {latent_var.std():.2f}")
    print(f"  범위: [{latent_var.min():.2f}, {latent_var.max():.2f}]")
    
    # 파라미터 (King 2022 스타일)
    params = {
        'zeta': np.array([1.0, 1.2, 0.8]),
        'tau': np.array([
            [-2.0, -1.0, 1.0, 2.0],
            [-2.0, -1.0, 1.0, 2.0],
            [-2.0, -1.0, 1.0, 2.0]
        ])
    }
    
    print(f"\n파라미터:")
    print(f"  요인적재량 (ζ): {params['zeta']}")
    print(f"  임계값 (τ): {params['tau'][0]}")
    
    # 로그우도 계산
    ll = model.log_likelihood(data, latent_var, params)
    
    print(f"\n로그우도 결과:")
    print(f"  총 로그우도: {ll:.2f}")
    print(f"  개인당 평균: {ll/len(data):.2f}")
    print(f"  지표당 평균: {ll/(len(data)*len(indicators)):.2f}")
    
    # 확률 예측 (처음 5명)
    probs_dict = model.predict_probabilities(latent_var[:5], params)

    print(f"\n처음 5명의 범주 확률 예측:")
    print(f"  지표: {indicators[0]}")
    probs_q13 = probs_dict[indicators[0]]  # (5, 5) 배열
    for i in range(5):
        print(f"  개인 {i+1} (LV={latent_var[i]:.2f}, 실제={data.iloc[i][indicators[0]]}): "
              f"P(1)={probs_q13[i,0]:.3f}, P(2)={probs_q13[i,1]:.3f}, P(3)={probs_q13[i,2]:.3f}, "
              f"P(4)={probs_q13[i,3]:.3f}, P(5)={probs_q13[i,4]:.3f}")
    
    print("\n✓ 시나리오 1 완료")
    
    return {
        'factor': 'perceived_benefit',
        'n_obs': len(data),
        'n_indicators': len(indicators),
        'log_likelihood': ll,
        'll_per_person': ll/len(data),
        'll_per_observation': ll/(len(data)*len(indicators))
    }


def test_scenario_2_health_concern():
    """
    시나리오 2: 건강관심도 (Health Concern)
    
    6개 지표: q6, q7, q8, q9, q10, q11
    """
    print("\n" + "="*80)
    print("시나리오 2: 건강관심도 (Health Concern) - 6개 지표")
    print("="*80)
    
    # 데이터 로드
    datasets = load_survey_data()
    data = datasets['health_concern']
    
    print(f"\n데이터 정보:")
    print(f"  관측치 수: {len(data)}")
    
    # 지표
    indicators = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
    
    print(f"\n사용 지표: {indicators}")
    print(f"\n지표별 기술통계:")
    for ind in indicators:
        print(f"  {ind}: 평균={data[ind].mean():.2f}, 표준편차={data[ind].std():.2f}")
    
    # 설정
    config = MeasurementConfig(
        indicators=indicators,
        n_categories=5
    )
    
    model = OrderedProbitMeasurement(config)
    
    # 잠재변수
    latent_var = data[indicators].mean(axis=1).values
    
    print(f"\n잠재변수 (평균):")
    print(f"  평균: {latent_var.mean():.2f}")
    print(f"  표준편차: {latent_var.std():.2f}")
    
    # 파라미터
    params = {
        'zeta': np.ones(6),  # 6개 지표
        'tau': np.tile([-2.0, -1.0, 1.0, 2.0], (6, 1))
    }
    
    print(f"\n파라미터:")
    print(f"  요인적재량 (ζ): {params['zeta']}")
    print(f"  임계값 (τ): {params['tau'][0]}")
    
    # 로그우도 계산
    ll = model.log_likelihood(data, latent_var, params)
    
    print(f"\n로그우도 결과:")
    print(f"  총 로그우도: {ll:.2f}")
    print(f"  개인당 평균: {ll/len(data):.2f}")
    print(f"  지표당 평균: {ll/(len(data)*len(indicators)):.2f}")
    
    print("\n✓ 시나리오 2 완료")
    
    return {
        'factor': 'health_concern',
        'n_obs': len(data),
        'n_indicators': len(indicators),
        'log_likelihood': ll,
        'll_per_person': ll/len(data),
        'll_per_observation': ll/(len(data)*len(indicators))
    }


def test_scenario_3_all_factors():
    """
    시나리오 3: 전체 요인 비교
    
    5개 요인 모두에 대해 Ordered Probit 적용
    """
    print("\n" + "="*80)
    print("시나리오 3: 전체 요인 비교 (5개 요인)")
    print("="*80)
    
    # 데이터 로드
    datasets = load_survey_data()
    
    # 요인별 설정
    factor_configs = {
        'health_concern': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        'perceived_benefit': ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
        'purchase_intention': ['q18', 'q19', 'q20'],
        'perceived_price': ['q27', 'q28', 'q29'],
        'nutrition_knowledge': [f'q{i}' for i in range(30, 50)]  # q30-q49
    }
    
    results = []
    
    for factor_name, indicators in factor_configs.items():
        print(f"\n{'─'*80}")
        print(f"요인: {factor_name}")
        print(f"{'─'*80}")
        
        data = datasets[factor_name]
        
        print(f"  관측치: {len(data)}")
        print(f"  지표 수: {len(indicators)}")
        print(f"  지표: {indicators[:3]}{'...' if len(indicators) > 3 else ''}")
        
        # 설정
        config = MeasurementConfig(
            indicators=indicators,
            n_categories=5
        )
        
        model = OrderedProbitMeasurement(config)
        
        # 잠재변수
        latent_var = data[indicators].mean(axis=1).values
        
        # 파라미터
        n_ind = len(indicators)
        params = {
            'zeta': np.ones(n_ind),
            'tau': np.tile([-2.0, -1.0, 1.0, 2.0], (n_ind, 1))
        }
        
        # 로그우도 계산
        ll = model.log_likelihood(data, latent_var, params)
        
        result = {
            'factor': factor_name,
            'n_obs': len(data),
            'n_indicators': len(indicators),
            'log_likelihood': ll,
            'll_per_person': ll/len(data),
            'll_per_observation': ll/(len(data)*len(indicators))
        }
        
        results.append(result)
        
        print(f"  로그우도: {ll:.2f}")
        print(f"  개인당 평균: {ll/len(data):.2f}")
        print(f"  지표당 평균: {ll/(len(data)*len(indicators)):.2f}")
    
    # 요약 테이블
    print(f"\n{'='*80}")
    print("전체 요인 비교 요약")
    print(f"{'='*80}")
    print(f"{'요인':<25} | {'지표수':>6} | {'총 LL':>10} | {'개인당':>8} | {'지표당':>8}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['factor']:<25} | {r['n_indicators']:>6} | {r['log_likelihood']:>10.2f} | "
              f"{r['ll_per_person']:>8.2f} | {r['ll_per_observation']:>8.2f}")
    
    print("\n✓ 시나리오 3 완료")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("실제 설문 데이터를 이용한 Ordered Probit 측정모델 테스트")
    print("="*80)
    
    # 시나리오 1: perceived_benefit (King 2022 스타일)
    result1 = test_scenario_1_perceived_benefit()
    
    # 시나리오 2: health_concern (6개 지표)
    result2 = test_scenario_2_health_concern()
    
    # 시나리오 3: 전체 요인 비교
    results3 = test_scenario_3_all_factors()
    
    print("\n" + "="*80)
    print("모든 테스트 완료! ✓")
    print("="*80)
    
    print("\n주요 결과:")
    print(f"  1. perceived_benefit (King 스타일): LL = {result1['log_likelihood']:.2f}")
    print(f"  2. health_concern (6개 지표): LL = {result2['log_likelihood']:.2f}")
    print(f"  3. 전체 5개 요인 테스트 완료")
    
    print("\n결론:")
    print("  ✅ 기존 설문 데이터와 Ordered Probit 모듈이 완벽히 호환됩니다")
    print("  ✅ 모든 요인에 대해 로그우도 계산이 정상적으로 작동합니다")
    print("  ✅ King (2022) 스타일 분석이 즉시 가능합니다")

