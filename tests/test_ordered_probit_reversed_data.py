#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
역코딩 데이터를 이용한 Ordered Probit 측정모델 재테스트

이 스크립트는 역코딩된 설문 데이터를 사용하여
OrderedProbitMeasurement 모듈의 성능 개선을 검증합니다.

목적:
1. 역코딩 전후 로그우도 비교
2. 적합도 개선 확인
3. 요인 내 일관성 검증
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
    latent_variable_name: Optional[str] = None


# OrderedProbitMeasurement 직접 import
measurement_equations_path = project_root / "src" / "analysis" / "hybrid_choice_model" / "iclv_models" / "measurement_equations.py"
import importlib.util
spec = importlib.util.spec_from_file_location("measurement_equations", measurement_equations_path)
measurement_equations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement_equations)
OrderedProbitMeasurement = measurement_equations.OrderedProbitMeasurement


def test_scenario_1_perceived_benefit_comparison():
    """
    시나리오 1: perceived_benefit 역코딩 전후 비교
    
    목적: q13 역코딩이 로그우도에 미치는 영향 확인
    """
    print("=" * 80)
    print("시나리오 1: perceived_benefit 역코딩 전후 비교")
    print("=" * 80)
    
    # 원본 데이터
    data_original = pd.read_csv('data/processed/survey/perceived_benefit.csv')
    
    # 역코딩 데이터
    data_reversed = pd.read_csv('data/processed/survey/perceived_benefit_reversed.csv')
    
    # 지표 설정 (King 2022 스타일: q13, q14, q15)
    indicators = ['q13', 'q14', 'q15']
    
    print(f"\n데이터 로드 완료:")
    print(f"  - 관측치: {len(data_original)}명")
    print(f"  - 지표: {indicators}")
    
    # 데이터 분포 비교
    print(f"\n=== 원본 데이터 분포 ===")
    for ind in indicators:
        mean_val = data_original[ind].mean()
        std_val = data_original[ind].std()
        print(f"{ind}: 평균={mean_val:.2f}, 표준편차={std_val:.2f}")
    
    print(f"\n=== 역코딩 데이터 분포 ===")
    for ind in indicators:
        mean_val = data_reversed[ind].mean()
        std_val = data_reversed[ind].std()
        print(f"{ind}: 평균={mean_val:.2f}, 표준편차={std_val:.2f}")
    
    # 요인 내 일관성 (표준편차)
    original_means = [data_original[ind].mean() for ind in indicators]
    reversed_means = [data_reversed[ind].mean() for ind in indicators]
    
    original_consistency = np.std(original_means)
    reversed_consistency = np.std(reversed_means)
    
    print(f"\n=== 요인 내 일관성 (지표 평균들의 표준편차) ===")
    print(f"원본 데이터: {original_consistency:.3f}")
    print(f"역코딩 데이터: {reversed_consistency:.3f}")
    print(f"개선율: {(1 - reversed_consistency/original_consistency)*100:.1f}%")
    
    # 설정
    config = MeasurementConfig(
        indicators=indicators,
        n_categories=5
    )
    
    # 모델 생성
    model = OrderedProbitMeasurement(config)
    
    # 잠재변수 (단순 평균)
    latent_var_original = data_original[indicators].mean(axis=1).values
    latent_var_reversed = data_reversed[indicators].mean(axis=1).values
    
    print(f"\n잠재변수 통계:")
    print(f"  원본 - 평균: {latent_var_original.mean():.2f}, 표준편차: {latent_var_original.std():.2f}")
    print(f"  역코딩 - 평균: {latent_var_reversed.mean():.2f}, 표준편차: {latent_var_reversed.std():.2f}")
    
    # 파라미터 (King 2022 스타일)
    params = {
        'zeta': np.array([1.0, 1.2, 0.8]),
        'tau': np.array([
            [-2.0, -1.0, 1.0, 2.0],
            [-2.0, -1.0, 1.0, 2.0],
            [-2.0, -1.0, 1.0, 2.0]
        ])
    }
    
    # 로그우도 계산 - 원본
    ll_original = model.log_likelihood(data_original, latent_var_original, params)
    ll_per_person_original = ll_original / len(data_original)
    ll_per_indicator_original = ll_original / (len(data_original) * len(indicators))
    
    # 로그우도 계산 - 역코딩
    ll_reversed = model.log_likelihood(data_reversed, latent_var_reversed, params)
    ll_per_person_reversed = ll_reversed / len(data_reversed)
    ll_per_indicator_reversed = ll_reversed / (len(data_reversed) * len(indicators))
    
    print(f"\n=== 로그우도 비교 ===")
    print(f"\n원본 데이터:")
    print(f"  총 로그우도: {ll_original:.2f}")
    print(f"  개인당 평균: {ll_per_person_original:.2f}")
    print(f"  지표당 평균: {ll_per_indicator_original:.2f}")
    
    print(f"\n역코딩 데이터:")
    print(f"  총 로그우도: {ll_reversed:.2f}")
    print(f"  개인당 평균: {ll_per_person_reversed:.2f}")
    print(f"  지표당 평균: {ll_per_indicator_reversed:.2f}")
    
    # 개선율
    improvement = ll_reversed - ll_original
    improvement_pct = (improvement / abs(ll_original)) * 100
    
    print(f"\n=== 개선 효과 ===")
    print(f"로그우도 개선: {improvement:.2f} ({improvement_pct:.2f}%)")
    print(f"지표당 LL 개선: {ll_per_indicator_reversed - ll_per_indicator_original:.3f}")
    
    if ll_reversed > ll_original:
        print(f"✅ 역코딩 적용으로 모델 적합도가 개선되었습니다!")
    else:
        print(f"⚠️ 역코딩 적용 후 적합도 변화 없음")
    
    # 확률 예측 비교 (처음 5명)
    print(f"\n=== 확률 예측 비교 (처음 5명) ===")

    probs_original = model.predict_probabilities(latent_var_original[:5], params)
    probs_reversed = model.predict_probabilities(latent_var_reversed[:5], params)

    for i in range(5):
        print(f"\n개인 {i+1}:")
        print(f"  원본 LV={latent_var_original[i]:.2f}, 역코딩 LV={latent_var_reversed[i]:.2f}")
        print(f"  실제값 (원본): q13={data_original.iloc[i]['q13']:.0f}, 역코딩: q13={data_reversed.iloc[i]['q13']:.0f}")

        # q13 확률 비교
        print(f"  q13 확률 (원본):", end=" ")
        for k in range(5):
            prob = probs_original['q13'][i, k]
            print(f"P({k+1})={prob:.3f}", end=" ")

        print(f"\n  q13 확률 (역코딩):", end=" ")
        for k in range(5):
            prob = probs_reversed['q13'][i, k]
            print(f"P({k+1})={prob:.3f}", end=" ")
        print()
    
    print("\n" + "=" * 80)
    
    return {
        'original_ll': ll_original,
        'reversed_ll': ll_reversed,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }


def test_scenario_2_all_factors_comparison():
    """
    시나리오 2: 전체 요인 역코딩 전후 비교
    
    목적: 모든 요인에 대한 역코딩 효과 확인
    """
    print("=" * 80)
    print("시나리오 2: 전체 요인 역코딩 전후 비교")
    print("=" * 80)
    
    # 요인 정의
    factors = {
        'health_concern': {
            'indicators': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            'has_reversed': False
        },
        'perceived_benefit': {
            'indicators': ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            'has_reversed': True
        },
        'purchase_intention': {
            'indicators': ['q18', 'q19', 'q20'],
            'has_reversed': False
        },
        'perceived_price': {
            'indicators': ['q27', 'q28', 'q29'],
            'has_reversed': True
        },
        'nutrition_knowledge': {
            'indicators': [f'q{i}' for i in range(30, 50)],
            'has_reversed': True
        }
    }
    
    results = []
    
    for factor_name, factor_info in factors.items():
        print(f"\n{'='*60}")
        print(f"{factor_name}")
        print(f"{'='*60}")
        
        indicators = factor_info['indicators']
        has_reversed = factor_info['has_reversed']
        
        # 원본 데이터
        data_original = pd.read_csv(f'data/processed/survey/{factor_name}.csv')
        
        # 역코딩 데이터 (있는 경우)
        if has_reversed:
            reversed_file = f'data/processed/survey/{factor_name}_reversed.csv'
            if Path(reversed_file).exists():
                data_reversed = pd.read_csv(reversed_file)
            else:
                print(f"⚠️ 역코딩 파일 없음: {reversed_file}")
                data_reversed = data_original
        else:
            data_reversed = data_original
        
        # 설정
        config = MeasurementConfig(
            indicators=indicators,
            n_categories=5
        )
        
        # 모델
        model = OrderedProbitMeasurement(config)
        
        # 잠재변수
        latent_var_original = data_original[indicators].mean(axis=1).values
        latent_var_reversed = data_reversed[indicators].mean(axis=1).values
        
        # 파라미터
        n_indicators = len(indicators)
        params = {
            'zeta': np.ones(n_indicators),
            'tau': np.tile([-2.0, -1.0, 1.0, 2.0], (n_indicators, 1))
        }
        
        # 로그우도
        ll_original = model.log_likelihood(data_original, latent_var_original, params)
        ll_reversed = model.log_likelihood(data_reversed, latent_var_reversed, params)
        
        ll_per_indicator_original = ll_original / (len(data_original) * n_indicators)
        ll_per_indicator_reversed = ll_reversed / (len(data_reversed) * n_indicators)
        
        improvement = ll_reversed - ll_original
        
        print(f"지표 수: {n_indicators}개")
        print(f"역문항 여부: {'있음' if has_reversed else '없음'}")
        print(f"\n원본 데이터:")
        print(f"  총 LL: {ll_original:.2f}")
        print(f"  지표당 LL: {ll_per_indicator_original:.3f}")
        
        if has_reversed:
            print(f"\n역코딩 데이터:")
            print(f"  총 LL: {ll_reversed:.2f}")
            print(f"  지표당 LL: {ll_per_indicator_reversed:.3f}")
            print(f"  개선: {improvement:.2f} ({ll_per_indicator_reversed - ll_per_indicator_original:.3f} per indicator)")
        
        results.append({
            'factor': factor_name,
            'n_indicators': n_indicators,
            'has_reversed': has_reversed,
            'll_original': ll_original,
            'll_reversed': ll_reversed,
            'll_per_ind_original': ll_per_indicator_original,
            'll_per_ind_reversed': ll_per_indicator_reversed,
            'improvement': improvement
        })
    
    # 요약 테이블
    print(f"\n{'='*80}")
    print("전체 요인 비교 요약")
    print(f"{'='*80}")
    
    print(f"\n{'요인':<25} {'지표수':>6} {'역문항':>6} {'원본 LL/지표':>12} {'역코딩 LL/지표':>14} {'개선':>8}")
    print("-" * 80)
    
    for r in results:
        reversed_mark = "✓" if r['has_reversed'] else "-"
        improvement_mark = f"+{r['improvement']:.1f}" if r['improvement'] > 0 else f"{r['improvement']:.1f}"
        
        print(f"{r['factor']:<25} {r['n_indicators']:>6} {reversed_mark:>6} "
              f"{r['ll_per_ind_original']:>12.3f} {r['ll_per_ind_reversed']:>14.3f} {improvement_mark:>8}")
    
    print("\n" + "=" * 80)
    
    return results


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("역코딩 데이터를 이용한 Ordered Probit 측정모델 재테스트")
    print("=" * 80)
    
    # 시나리오 1: perceived_benefit 상세 비교
    result1 = test_scenario_1_perceived_benefit_comparison()
    
    # 시나리오 2: 전체 요인 비교
    result2 = test_scenario_2_all_factors_comparison()
    
    # 최종 결론
    print("\n" + "=" * 80)
    print("최종 결론")
    print("=" * 80)
    
    print(f"\n1. perceived_benefit (q13, q14, q15) 역코딩 효과:")
    print(f"   - 로그우도 개선: {result1['improvement']:.2f} ({result1['improvement_pct']:.2f}%)")
    
    if result1['improvement'] > 0:
        print(f"   ✅ 역코딩 적용으로 모델 적합도가 개선되었습니다!")
    
    print(f"\n2. 전체 요인 분석:")
    reversed_factors = [r for r in result2 if r['has_reversed']]
    total_improvement = sum(r['improvement'] for r in reversed_factors)
    
    print(f"   - 역문항이 있는 요인: {len(reversed_factors)}개")
    print(f"   - 총 로그우도 개선: {total_improvement:.2f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

