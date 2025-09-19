"""
Moderation Analysis Example Usage

조절효과 분석 모듈의 사용 예제들을 제공합니다.
다양한 조절효과 분석 시나리오와 고급 기능 사용법을 포함합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

# 조절효과 분석 모듈 임포트
from moderation_analysis import (
    # 핵심 분석 함수
    analyze_moderation_effects,
    calculate_simple_slopes,
    calculate_conditional_effects,
    test_moderation_significance,
    
    # 데이터 로딩
    load_moderation_data,
    get_available_factors,
    combine_factor_data,
    
    # 상호작용 모델링
    create_interaction_terms,
    build_moderation_model,
    
    # 결과 저장
    export_moderation_results,
    create_moderation_report,
    
    # 시각화
    create_moderation_plot,
    create_simple_slopes_plot,
    create_interaction_heatmap,
    visualize_moderation_analysis,
    
    # 설정
    create_default_moderation_config,
    create_custom_moderation_config
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_moderation_analysis():
    """예제 1: 기본 조절효과 분석"""
    print("=" * 60)
    print("예제 1: 기본 조절효과 분석")
    print("=" * 60)
    
    try:
        # 건강관심도 → 구매의도 (영양지식 조절)
        print("분석: 건강관심도 → 구매의도 (영양지식의 조절효과)")
        
        results = analyze_moderation_effects(
            independent_var='health_concern',
            dependent_var='purchase_intention',
            moderator_var='nutrition_knowledge'
        )
        
        # 결과 출력
        print("\n📊 분석 결과:")
        moderation_test = results['moderation_test']
        print(f"상호작용 계수: {moderation_test['interaction_coefficient']:.4f}")
        print(f"P값: {moderation_test['p_value']:.4f}")
        print(f"유의성: {'유의함' if moderation_test['significant'] else '유의하지 않음'}")
        print(f"해석: {moderation_test['interpretation']}")
        
        # 단순기울기 결과
        print("\n📈 단순기울기 분석:")
        simple_slopes = results['simple_slopes']
        for level, slope_info in simple_slopes.items():
            print(f"{level}: {slope_info['simple_slope']:.4f} (p={slope_info['p_value']:.4f})")
        
        return results
        
    except Exception as e:
        print(f"❌ 예제 1 실행 실패: {e}")
        return None


def example_2_custom_configuration():
    """예제 2: 사용자 정의 설정을 사용한 분석"""
    print("=" * 60)
    print("예제 2: 사용자 정의 설정을 사용한 분석")
    print("=" * 60)
    
    try:
        # 사용자 정의 설정 생성
        custom_config = create_custom_moderation_config(
            results_dir="custom_moderation_results",
            bootstrap_samples=1000,  # 빠른 테스트를 위해 감소
            confidence_level=0.99,   # 99% 신뢰구간
            center_variables=True,
            simple_slopes_values=[-2.0, -1.0, 0.0, 1.0, 2.0]  # 더 많은 수준
        )
        
        print("사용자 정의 설정:")
        print(f"- 결과 디렉토리: {custom_config.results_dir}")
        print(f"- 부트스트래핑 샘플: {custom_config.bootstrap_samples}")
        print(f"- 신뢰수준: {custom_config.confidence_level}")
        print(f"- 단순기울기 수준: {custom_config.simple_slopes_values}")
        
        # 분석 실행
        print("\n분석: 지각된혜택 → 구매의도 (지각된가격 조절)")
        
        # 데이터 로드 (사용자 정의 설정 사용)
        data = load_moderation_data(
            independent_var='perceived_benefit',
            dependent_var='purchase_intention',
            moderator_var='perceived_price',
            config=custom_config
        )
        
        print(f"로드된 데이터: {data.shape}")
        
        # 분석 실행
        results = analyze_moderation_effects(
            independent_var='perceived_benefit',
            dependent_var='purchase_intention',
            moderator_var='perceived_price',
            data=data
        )
        
        # 결과 저장 (사용자 정의 설정 사용)
        saved_files = export_moderation_results(
            results, 
            analysis_name="custom_benefit_price_moderation"
        )
        
        print(f"\n💾 저장된 파일: {len(saved_files)}개")
        for file_type, file_path in saved_files.items():
            print(f"   - {file_type}: {file_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ 예제 2 실행 실패: {e}")
        return None


def example_3_step_by_step_analysis():
    """예제 3: 단계별 상세 분석"""
    print("=" * 60)
    print("예제 3: 단계별 상세 분석")
    print("=" * 60)
    
    try:
        # 1단계: 데이터 로드
        print("1단계: 데이터 로드")
        data = combine_factor_data([
            'health_concern', 'perceived_benefit', 'nutrition_knowledge'
        ])
        print(f"   데이터 크기: {data.shape}")
        print(f"   기술통계:\n{data.describe()}")
        
        # 2단계: 상호작용항 생성
        print("\n2단계: 상호작용항 생성")
        interaction_data = create_interaction_terms(
            data=data,
            independent_var='health_concern',
            moderator_var='nutrition_knowledge',
            method='product'
        )
        
        interaction_name = 'health_concern_x_nutrition_knowledge'
        print(f"   상호작용항 '{interaction_name}' 생성 완료")
        print(f"   상호작용항 통계: 평균={interaction_data[interaction_name].mean():.4f}, "
              f"표준편차={interaction_data[interaction_name].std():.4f}")
        
        # 3단계: 모델 스펙 생성
        print("\n3단계: 모델 스펙 생성")
        model_spec = build_moderation_model(
            independent_var='health_concern',
            dependent_var='perceived_benefit',
            moderator_var='nutrition_knowledge'
        )
        print(f"   모델 스펙:\n{model_spec}")
        
        # 4단계: 조절효과 유의성 검정
        print("\n4단계: 조절효과 유의성 검정")
        moderation_test = test_moderation_significance(
            independent_var='health_concern',
            dependent_var='perceived_benefit',
            moderator_var='nutrition_knowledge',
            data=interaction_data
        )
        
        print(f"   상호작용 계수: {moderation_test['interaction_coefficient']:.4f}")
        print(f"   유의성: {'유의함' if moderation_test['significant'] else '유의하지 않음'}")
        
        # 5단계: 단순기울기 분석
        print("\n5단계: 단순기울기 분석")
        simple_slopes = calculate_simple_slopes(
            independent_var='health_concern',
            dependent_var='perceived_benefit',
            moderator_var='nutrition_knowledge',
            data=interaction_data
        )
        
        for level, slope_info in simple_slopes.items():
            significance = "유의함" if slope_info['significant'] else "유의하지 않음"
            print(f"   {level}: 기울기={slope_info['simple_slope']:.4f}, {significance}")
        
        # 6단계: 조건부 효과 분석
        print("\n6단계: 조건부 효과 분석")
        conditional_effects = calculate_conditional_effects(
            independent_var='health_concern',
            dependent_var='perceived_benefit',
            moderator_var='nutrition_knowledge',
            data=interaction_data
        )
        
        for percentile, effect_info in conditional_effects.items():
            print(f"   {percentile}: 효과={effect_info['simple_slope']:.4f}")
        
        return {
            'data': interaction_data,
            'moderation_test': moderation_test,
            'simple_slopes': simple_slopes,
            'conditional_effects': conditional_effects
        }
        
    except Exception as e:
        print(f"❌ 예제 3 실행 실패: {e}")
        return None


def example_4_comprehensive_visualization():
    """예제 4: 포괄적 시각화"""
    print("=" * 60)
    print("예제 4: 포괄적 시각화")
    print("=" * 60)
    
    try:
        # 분석 실행
        print("분석 실행: 건강관심도 → 지각된혜택 (지각된가격 조절)")
        
        results = analyze_moderation_effects(
            independent_var='health_concern',
            dependent_var='perceived_benefit',
            moderator_var='perceived_price'
        )
        
        # 데이터 로드 (시각화용)
        data = load_moderation_data(
            independent_var='health_concern',
            dependent_var='perceived_benefit',
            moderator_var='perceived_price'
        )
        
        # 상호작용항 추가
        data = create_interaction_terms(
            data=data,
            independent_var='health_concern',
            moderator_var='perceived_price'
        )
        
        # 포괄적 시각화 생성
        print("\n📊 시각화 생성 중...")
        plot_files = visualize_moderation_analysis(
            data=data,
            results=results,
            analysis_name="example_health_price_moderation"
        )
        
        print(f"생성된 그래프: {len(plot_files)}개")
        for plot_type, plot_path in plot_files.items():
            print(f"   - {plot_type}: {plot_path}")
        
        # 개별 그래프도 생성 가능
        print("\n개별 그래프 생성:")
        
        # 조절효과 플롯
        moderation_plot = create_moderation_plot(data, results)
        print(f"   조절효과 플롯: {moderation_plot}")
        
        # 단순기울기 플롯
        slopes_plot = create_simple_slopes_plot(results)
        print(f"   단순기울기 플롯: {slopes_plot}")
        
        # 상호작용 히트맵
        heatmap_plot = create_interaction_heatmap(data, results)
        print(f"   상호작용 히트맵: {heatmap_plot}")
        
        return plot_files
        
    except Exception as e:
        print(f"❌ 예제 4 실행 실패: {e}")
        return None


def example_5_multiple_moderations():
    """예제 5: 다중 조절효과 분석"""
    print("=" * 60)
    print("예제 5: 다중 조절효과 분석")
    print("=" * 60)
    
    try:
        # 여러 조절효과 조합 정의
        moderation_combinations = [
            ('health_concern', 'purchase_intention', 'nutrition_knowledge'),
            ('health_concern', 'purchase_intention', 'perceived_price'),
            ('perceived_benefit', 'purchase_intention', 'nutrition_knowledge'),
            ('perceived_benefit', 'purchase_intention', 'perceived_price'),
        ]
        
        results_summary = []
        
        print(f"분석할 조절효과 조합: {len(moderation_combinations)}개")
        
        for i, (independent, dependent, moderator) in enumerate(moderation_combinations, 1):
            print(f"\n{i}. {independent} → {dependent} (조절: {moderator})")
            
            try:
                # 조절효과 분석
                results = analyze_moderation_effects(
                    independent_var=independent,
                    dependent_var=dependent,
                    moderator_var=moderator
                )
                
                # 결과 요약
                moderation_test = results['moderation_test']
                summary = {
                    'independent': independent,
                    'dependent': dependent,
                    'moderator': moderator,
                    'interaction_coeff': moderation_test['interaction_coefficient'],
                    'p_value': moderation_test['p_value'],
                    'significant': moderation_test['significant'],
                    'interpretation': moderation_test['interpretation']
                }
                
                results_summary.append(summary)
                
                # 간단한 결과 출력
                status = "✅ 유의함" if summary['significant'] else "❌ 유의하지 않음"
                print(f"   결과: {status} (p={summary['p_value']:.4f})")
                
            except Exception as e:
                print(f"   ❌ 분석 실패: {e}")
                continue
        
        # 전체 결과 요약
        print("\n" + "=" * 50)
        print("다중 조절효과 분석 요약")
        print("=" * 50)
        
        significant_count = sum(1 for r in results_summary if r['significant'])
        total_count = len(results_summary)
        
        print(f"총 분석 수: {total_count}")
        print(f"유의한 조절효과: {significant_count}개 ({significant_count/total_count*100:.1f}%)")
        print()
        
        print("상세 결과:")
        print("-" * 80)
        print(f"{'독립변수':<15} {'종속변수':<15} {'조절변수':<15} {'계수':<8} {'P값':<8} {'유의성'}")
        print("-" * 80)
        
        for summary in results_summary:
            status = "유의함" if summary['significant'] else "유의하지 않음"
            print(f"{summary['independent']:<15} {summary['dependent']:<15} "
                  f"{summary['moderator']:<15} {summary['interaction_coeff']:<8.4f} "
                  f"{summary['p_value']:<8.4f} {status}")
        
        return results_summary
        
    except Exception as e:
        print(f"❌ 예제 5 실행 실패: {e}")
        return None


def run_all_examples():
    """모든 예제 실행"""
    print("🚀 조절효과 분석 예제 실행")
    print("=" * 80)
    
    examples = [
        ("기본 조절효과 분석", example_1_basic_moderation_analysis),
        ("사용자 정의 설정", example_2_custom_configuration),
        ("단계별 상세 분석", example_3_step_by_step_analysis),
        ("포괄적 시각화", example_4_comprehensive_visualization),
        ("다중 조절효과 분석", example_5_multiple_moderations)
    ]
    
    results = {}
    
    for example_name, example_func in examples:
        print(f"\n🔄 실행 중: {example_name}")
        try:
            result = example_func()
            results[example_name] = result
            print(f"✅ {example_name} 완료")
        except Exception as e:
            print(f"❌ {example_name} 실패: {e}")
            results[example_name] = None
        
        print("-" * 60)
    
    # 전체 요약
    successful_examples = sum(1 for r in results.values() if r is not None)
    total_examples = len(examples)
    
    print(f"\n🎉 예제 실행 완료: {successful_examples}/{total_examples}개 성공")
    
    return results


if __name__ == "__main__":
    # 개별 예제 실행 또는 전체 예제 실행
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_moderation_analysis()
        elif example_num == "2":
            example_2_custom_configuration()
        elif example_num == "3":
            example_3_step_by_step_analysis()
        elif example_num == "4":
            example_4_comprehensive_visualization()
        elif example_num == "5":
            example_5_multiple_moderations()
        else:
            print(f"알 수 없는 예제 번호: {example_num}")
            print("사용법: python example_usage.py [1-5]")
    else:
        run_all_examples()
