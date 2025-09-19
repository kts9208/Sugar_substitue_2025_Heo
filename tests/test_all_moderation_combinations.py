#!/usr/bin/env python3
"""
5개 요인 간 모든 조절효과 조합 분석 테스트
"""

import sys
import pandas as pd
import numpy as np
from itertools import permutations, combinations
from pathlib import Path
import time

def calculate_all_combinations():
    """5개 요인으로 가능한 모든 조절효과 조합 계산"""
    factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 'perceived_price', 'nutrition_knowledge']
    
    print("=== 5개 요인 간 조절효과 분석 가능한 모든 조합 ===")
    print(f"요인: {factors}")
    print()
    
    # 조절효과 분석: 독립변수 × 조절변수 → 종속변수
    # 3개 요인이 필요: 독립변수, 종속변수, 조절변수
    
    combinations_list = []
    
    for dependent in factors:
        for independent in factors:
            if independent != dependent:
                for moderator in factors:
                    if moderator != dependent and moderator != independent:
                        combinations_list.append({
                            'independent': independent,
                            'dependent': dependent,
                            'moderator': moderator,
                            'name': f"{independent}_x_{moderator}_to_{dependent}"
                        })
    
    print(f"📊 총 가능한 조절효과 조합 수: {len(combinations_list)}개")
    print(f"   계산식: 5(종속) × 4(독립) × 3(조절) = {5*4*3}개")
    print()
    
    return combinations_list


def test_sample_combinations(combinations_list, sample_size=10):
    """샘플 조합들에 대해 조절효과 분석 테스트"""
    print(f"=== 샘플 {sample_size}개 조합 조절효과 분석 테스트 ===")
    
    # 샘플 선택 (다양한 조합 포함)
    sample_combinations = []
    
    # 각 요인이 종속변수인 경우를 포함하도록 샘플링
    factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 'perceived_price', 'nutrition_knowledge']
    
    for i, factor in enumerate(factors):
        # 각 요인이 종속변수인 경우 2개씩 선택
        factor_combinations = [c for c in combinations_list if c['dependent'] == factor]
        if len(factor_combinations) >= 2:
            sample_combinations.extend(factor_combinations[:2])
    
    print(f"선택된 샘플 조합: {len(sample_combinations)}개")
    print()
    
    results_summary = []
    
    for i, combo in enumerate(sample_combinations, 1):
        print(f"🔄 분석 {i}/{len(sample_combinations)}: {combo['name']}")
        print(f"   {combo['independent']} × {combo['moderator']} → {combo['dependent']}")
        
        try:
            from moderation_analysis import analyze_moderation_effects
            
            start_time = time.time()
            
            # 조절효과 분석 실행
            results = analyze_moderation_effects(
                independent_var=combo['independent'],
                dependent_var=combo['dependent'],
                moderator_var=combo['moderator']
            )
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            # 결과 추출
            moderation_test = results.get('moderation_test', {})
            interaction_coef = moderation_test.get('interaction_coefficient', 0)
            p_value = moderation_test.get('p_value', 1)
            significant = moderation_test.get('significant', False)
            
            # 결과 요약
            result_summary = {
                'combination': combo['name'],
                'independent': combo['independent'],
                'dependent': combo['dependent'],
                'moderator': combo['moderator'],
                'interaction_coefficient': interaction_coef,
                'p_value': p_value,
                'significant': significant,
                'analysis_time': analysis_time
            }
            
            results_summary.append(result_summary)
            
            # 결과 출력
            status = "✅ 유의함" if significant else "❌ 유의하지 않음"
            print(f"   결과: 계수={interaction_coef:.6f}, p={p_value:.6f}, {status}")
            print(f"   분석시간: {analysis_time:.2f}초")
            print()
            
        except Exception as e:
            print(f"   ❌ 분석 실패: {e}")
            result_summary = {
                'combination': combo['name'],
                'independent': combo['independent'],
                'dependent': combo['dependent'],
                'moderator': combo['moderator'],
                'interaction_coefficient': None,
                'p_value': None,
                'significant': False,
                'analysis_time': None,
                'error': str(e)
            }
            results_summary.append(result_summary)
            print()
    
    return results_summary


def analyze_results_summary(results_summary):
    """결과 요약 분석"""
    print("=== 📊 조절효과 분석 결과 요약 ===")
    print("=" * 80)
    
    total_analyses = len(results_summary)
    successful_analyses = len([r for r in results_summary if r.get('interaction_coefficient') is not None])
    significant_effects = len([r for r in results_summary if r.get('significant', False)])
    
    print(f"총 분석 수: {total_analyses}")
    print(f"성공한 분석: {successful_analyses}/{total_analyses} ({successful_analyses/total_analyses*100:.1f}%)")
    print(f"유의한 조절효과: {significant_effects}/{successful_analyses} ({significant_effects/successful_analyses*100:.1f}%)")
    print()
    
    # 성공한 분석들의 상세 결과
    successful_results = [r for r in results_summary if r.get('interaction_coefficient') is not None]
    
    if successful_results:
        print("📋 상세 결과:")
        print("-" * 100)
        print(f"{'조합':<40} {'계수':<12} {'P값':<10} {'유의성':<8} {'시간(초)'}")
        print("-" * 100)
        
        for result in successful_results:
            combo_name = result['combination'][:38] + '..' if len(result['combination']) > 40 else result['combination']
            coef = f"{result['interaction_coefficient']:.6f}"
            p_val = f"{result['p_value']:.6f}"
            sig = "✅" if result['significant'] else "❌"
            time_str = f"{result['analysis_time']:.2f}" if result['analysis_time'] else "N/A"
            
            print(f"{combo_name:<40} {coef:<12} {p_val:<10} {sig:<8} {time_str}")
    
    # 유의한 조절효과들
    significant_results = [r for r in successful_results if r.get('significant', False)]
    if significant_results:
        print(f"\n🎯 유의한 조절효과 ({len(significant_results)}개):")
        print("-" * 60)
        for result in significant_results:
            print(f"• {result['independent']} × {result['moderator']} → {result['dependent']}")
            print(f"  계수: {result['interaction_coefficient']:.6f}, p={result['p_value']:.6f}")
    else:
        print("\n💡 유의한 조절효과가 발견되지 않았습니다.")
    
    # 평균 분석 시간
    analysis_times = [r['analysis_time'] for r in successful_results if r['analysis_time']]
    if analysis_times:
        avg_time = np.mean(analysis_times)
        total_time = sum(analysis_times)
        print(f"\n⏱️ 분석 시간:")
        print(f"   평균 분석 시간: {avg_time:.2f}초")
        print(f"   총 분석 시간: {total_time:.2f}초")
        
        # 전체 조합 분석 예상 시간
        total_combinations = 5 * 4 * 3  # 60개
        estimated_total_time = total_combinations * avg_time
        print(f"   전체 {total_combinations}개 조합 분석 예상 시간: {estimated_total_time:.1f}초 ({estimated_total_time/60:.1f}분)")


def main():
    """메인 함수"""
    print("🔍 5개 요인 간 조절효과 분석 전체 검토")
    print("=" * 80)
    
    # 1. 모든 가능한 조합 계산
    all_combinations = calculate_all_combinations()
    
    # 2. 샘플 조합들에 대해 분석 테스트
    results_summary = test_sample_combinations(all_combinations, sample_size=10)
    
    # 3. 결과 요약 분석
    analyze_results_summary(results_summary)
    
    print("\n" + "=" * 80)
    print("🎉 조절효과 분석 모듈 검토 완료!")
    print("=" * 80)
    
    # 최종 권장사항
    print("💡 권장사항:")
    print("1. 모든 60개 조합을 분석하려면 별도 스크립트 실행")
    print("2. 연구 목적에 맞는 특정 조합들을 선별하여 분석")
    print("3. 유의한 조절효과가 발견된 조합들에 대해 심화 분석")


if __name__ == "__main__":
    main()
