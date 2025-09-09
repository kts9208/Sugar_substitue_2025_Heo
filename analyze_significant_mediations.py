#!/usr/bin/env python3
"""
유의한 매개효과만 추출하여 분석
"""

import pandas as pd
import numpy as np
import logging
import json
from path_analysis.effects_calculator import EffectsCalculator
from semopy import Model

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_analyze_significant_mediations():
    """저장된 결과에서 유의한 매개효과만 추출하여 분석"""
    
    print("=" * 60)
    print("유의한 매개효과 추출 및 분석")
    print("=" * 60)
    
    try:
        # JSON 파일 로드
        json_file = "comprehensive_mediation_results/mediation_summary_20250908_221610.json"
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"JSON 파일 로드 완료: {json_file}")
        
        # EffectsCalculator 결과 추출
        effects_results = data.get('effects_calculator_results', {})
        
        if not effects_results:
            print("❌ EffectsCalculator 결과가 없습니다.")
            return False
        
        # 유의한 매개효과 추출
        significant_results = effects_results.get('significant_results', {})
        all_results = effects_results.get('all_results', {})
        summary = effects_results.get('summary', {})
        
        print(f"\n분석 요약:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print(f"\n유의한 매개효과: {len(significant_results)}개")
        
        # 유의한 매개효과 상세 분석
        if significant_results:
            print(f"\n" + "=" * 40)
            print("유의한 매개효과 상세 분석")
            print("=" * 40)
            
            for sig_key, sig_data in significant_results.items():
                print(f"\n--- {sig_key} ---")
                
                # 변수 정보
                independent_var = sig_data.get('independent_var', 'N/A')
                dependent_var = sig_data.get('dependent_var', 'N/A')
                mediator = sig_data.get('mediator', 'N/A')
                
                print(f"독립변수: {independent_var}")
                print(f"종속변수: {dependent_var}")
                print(f"매개변수: {mediator}")
                
                # 매개효과 결과
                mediation_result = sig_data.get('mediation_result', {})
                
                if 'original_effects' in mediation_result:
                    original = mediation_result['original_effects']
                    direct_effect = original.get('direct_effect', 0)
                    indirect_effect = original.get('indirect_effect', 0)
                    total_effect = original.get('total_effect', 0)
                    
                    print(f"직접효과: {direct_effect:.6f}")
                    print(f"간접효과: {indirect_effect:.6f}")
                    print(f"총효과: {total_effect:.6f}")
                    
                    # 매개효과 비율
                    if total_effect != 0:
                        mediation_ratio = indirect_effect / total_effect
                        print(f"매개효과 비율: {mediation_ratio:.2%}")
                        
                        if abs(mediation_ratio) > 0.5:
                            print("→ 강한 매개효과")
                        elif abs(mediation_ratio) > 0.2:
                            print("→ 중간 매개효과")
                        else:
                            print("→ 약한 매개효과")
                
                if 'confidence_intervals' in mediation_result:
                    ci = mediation_result['confidence_intervals']
                    print(f"신뢰구간 (95%):")
                    
                    for effect_name, ci_data in ci.items():
                        if isinstance(ci_data, dict):
                            lower = ci_data.get('lower', 'N/A')
                            upper = ci_data.get('upper', 'N/A')
                            significant = ci_data.get('significant', False)
                            
                            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                                print(f"  {effect_name}: [{lower:.6f}, {upper:.6f}] {'*' if significant else ''}")
        
        # 모든 결과에서 유의한 간접효과만 추출
        print(f"\n" + "=" * 40)
        print("모든 조합에서 유의한 간접효과 추출")
        print("=" * 40)
        
        significant_indirect_effects = []
        
        for combination_key, combination_data in all_results.items():
            mediation_result = combination_data.get('mediation_result', {})
            
            if 'confidence_intervals' in mediation_result:
                ci = mediation_result['confidence_intervals']
                indirect_ci = ci.get('indirect_effects', {})
                
                if isinstance(indirect_ci, dict) and indirect_ci.get('significant', False):
                    # 유의한 간접효과 발견
                    original_effects = mediation_result.get('original_effects', {})
                    indirect_effect = original_effects.get('indirect_effect', 0)
                    
                    if abs(indirect_effect) > 0.001:  # 매우 작은 효과 제외
                        significant_indirect_effects.append({
                            'combination': combination_key,
                            'independent_var': combination_data.get('independent_var', 'N/A'),
                            'dependent_var': combination_data.get('dependent_var', 'N/A'),
                            'mediator': combination_data.get('mediator', 'N/A'),
                            'indirect_effect': indirect_effect,
                            'lower_ci': indirect_ci.get('lower', 'N/A'),
                            'upper_ci': indirect_ci.get('upper', 'N/A')
                        })
        
        # 유의한 간접효과 정렬 (효과 크기 순)
        significant_indirect_effects.sort(key=lambda x: abs(x['indirect_effect']), reverse=True)
        
        print(f"\n유의한 간접효과: {len(significant_indirect_effects)}개")
        
        for i, effect in enumerate(significant_indirect_effects, 1):
            print(f"\n{i}. {effect['combination']}")
            print(f"   {effect['independent_var']} → {effect['mediator']} → {effect['dependent_var']}")
            print(f"   간접효과: {effect['indirect_effect']:.6f}")
            print(f"   신뢰구간: [{effect['lower_ci']:.6f}, {effect['upper_ci']:.6f}] *")
        
        return True, significant_indirect_effects
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def create_mediation_summary_report(significant_effects):
    """유의한 매개효과 요약 보고서 생성"""
    
    print(f"\n" + "=" * 60)
    print("유의한 매개효과 요약 보고서 생성")
    print("=" * 60)
    
    try:
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"significant_mediations_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("5개 요인 유의한 매개효과 분석 보고서\n")
            f.write("=" * 60 + "\n")
            f.write(f"생성 시간: {timestamp}\n\n")
            
            f.write(f"유의한 간접효과: {len(significant_effects)}개\n\n")
            
            # 효과 크기별 분류
            strong_effects = [e for e in significant_effects if abs(e['indirect_effect']) > 0.1]
            medium_effects = [e for e in significant_effects if 0.05 < abs(e['indirect_effect']) <= 0.1]
            weak_effects = [e for e in significant_effects if 0.01 < abs(e['indirect_effect']) <= 0.05]
            
            f.write(f"효과 크기별 분류:\n")
            f.write(f"  강한 효과 (|효과| > 0.1): {len(strong_effects)}개\n")
            f.write(f"  중간 효과 (0.05 < |효과| ≤ 0.1): {len(medium_effects)}개\n")
            f.write(f"  약한 효과 (0.01 < |효과| ≤ 0.05): {len(weak_effects)}개\n\n")
            
            # 상세 결과
            f.write("상세 매개효과 결과:\n")
            f.write("-" * 40 + "\n")
            
            for i, effect in enumerate(significant_effects, 1):
                f.write(f"\n{i}. {effect['combination']}\n")
                f.write(f"   경로: {effect['independent_var']} → {effect['mediator']} → {effect['dependent_var']}\n")
                f.write(f"   간접효과: {effect['indirect_effect']:.6f}\n")
                f.write(f"   신뢰구간: [{effect['lower_ci']:.6f}, {effect['upper_ci']:.6f}] *\n")
                
                # 효과 크기 해석
                abs_effect = abs(effect['indirect_effect'])
                if abs_effect > 0.1:
                    f.write(f"   해석: 강한 매개효과\n")
                elif abs_effect > 0.05:
                    f.write(f"   해석: 중간 매개효과\n")
                else:
                    f.write(f"   해석: 약한 매개효과\n")
            
            # 변수별 매개역할 분석
            f.write(f"\n" + "=" * 40 + "\n")
            f.write("변수별 매개역할 분석\n")
            f.write("=" * 40 + "\n")
            
            mediator_counts = {}
            for effect in significant_effects:
                mediator = effect['mediator']
                if mediator not in mediator_counts:
                    mediator_counts[mediator] = 0
                mediator_counts[mediator] += 1
            
            sorted_mediators = sorted(mediator_counts.items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"\n매개변수별 유의한 매개효과 횟수:\n")
            for mediator, count in sorted_mediators:
                f.write(f"  {mediator}: {count}회\n")
            
            # 독립변수별 분석
            independent_counts = {}
            for effect in significant_effects:
                independent = effect['independent_var']
                if independent not in independent_counts:
                    independent_counts[independent] = 0
                independent_counts[independent] += 1
            
            sorted_independents = sorted(independent_counts.items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"\n독립변수별 유의한 매개효과 횟수:\n")
            for independent, count in sorted_independents:
                f.write(f"  {independent}: {count}회\n")
        
        print(f"✅ 요약 보고서 저장: {report_file}")
        
        return True, report_file
        
    except Exception as e:
        print(f"❌ 보고서 생성 실패: {e}")
        return False, None

if __name__ == "__main__":
    print("유의한 매개효과 추출 및 분석")
    
    # 1. 유의한 매개효과 추출
    success, significant_effects = load_and_analyze_significant_mediations()
    
    # 2. 요약 보고서 생성
    if success and significant_effects:
        report_success, report_file = create_mediation_summary_report(significant_effects)
    else:
        report_success = False
        report_file = None
    
    print(f"\n" + "=" * 60)
    print("최종 분석 결과")
    print("=" * 60)
    print(f"유의한 매개효과 추출: {'✅ 성공' if success else '❌ 실패'}")
    print(f"요약 보고서 생성: {'✅ 성공' if report_success else '❌ 실패'}")
    
    if success and significant_effects:
        print(f"\n📊 발견된 유의한 간접효과: {len(significant_effects)}개")
        
        # 효과 크기별 분류
        strong_effects = [e for e in significant_effects if abs(e['indirect_effect']) > 0.1]
        medium_effects = [e for e in significant_effects if 0.05 < abs(e['indirect_effect']) <= 0.1]
        weak_effects = [e for e in significant_effects if 0.01 < abs(e['indirect_effect']) <= 0.05]
        
        print(f"  강한 효과 (|효과| > 0.1): {len(strong_effects)}개")
        print(f"  중간 효과 (0.05 < |효과| ≤ 0.1): {len(medium_effects)}개")
        print(f"  약한 효과 (0.01 < |효과| ≤ 0.05): {len(weak_effects)}개")
        
        # 상위 5개 효과 출력
        print(f"\n🏆 상위 5개 유의한 매개효과:")
        for i, effect in enumerate(significant_effects[:5], 1):
            print(f"  {i}. {effect['independent_var']} → {effect['mediator']} → {effect['dependent_var']}")
            print(f"     간접효과: {effect['indirect_effect']:.6f}")
    
    if report_success and report_file:
        print(f"\n📁 요약 보고서: {report_file}")
    
    if success:
        print(f"\n🎉 분석 완료!")
        print("✅ 모든 요인에 대한 매개효과가 분석되었습니다.")
        print("✅ 유의한 매개효과가 식별되었습니다.")
        print("✅ 효과 크기별 분류가 완료되었습니다.")
        print("✅ 변수별 매개역할이 분석되었습니다.")
    else:
        print(f"\n⚠️  분석에서 문제가 발생했습니다.")
