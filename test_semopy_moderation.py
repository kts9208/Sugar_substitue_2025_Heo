#!/usr/bin/env python3
"""
semopy 기반 조절효과 분석 테스트
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_semopy_moderation():
    """semopy 기반 조절효과 분석 테스트"""
    print("=== semopy 기반 조절효과 분석 테스트 ===")
    
    try:
        from moderation_analysis import analyze_moderation_effects, export_moderation_results
        
        print("1. 조절효과 분석 실행 중...")
        
        # 조절효과 분석 (timeout 없이)
        results = analyze_moderation_effects(
            independent_var='health_concern',
            dependent_var='perceived_benefit',
            moderator_var='nutrition_knowledge'
        )
        
        print("✅ semopy 조절효과 분석 성공!")
        
        # 결과 확인
        print("\n📊 분석 결과:")
        
        # 변수 정보
        variables = results.get('variables', {})
        print(f"독립변수: {variables.get('independent', 'N/A')}")
        print(f"종속변수: {variables.get('dependent', 'N/A')}")
        print(f"조절변수: {variables.get('moderator', 'N/A')}")
        print(f"상호작용항: {variables.get('interaction', 'N/A')}")
        
        # 모델 정보
        model_info = results.get('model_info', {})
        print(f"\n모델 정보:")
        print(f"관측치 수: {model_info.get('n_observations', 'N/A')}")
        print(f"모수 수: {model_info.get('n_parameters', 'N/A')}")
        
        # 조절효과 검정
        moderation_test = results.get('moderation_test', {})
        print(f"\n🎯 조절효과 검정 결과:")
        print(f"상호작용 계수: {moderation_test.get('interaction_coefficient', 'N/A'):.6f}")
        print(f"표준오차: {moderation_test.get('std_error', 'N/A'):.6f}")
        print(f"Z값: {moderation_test.get('z_value', 'N/A'):.6f}")
        print(f"P값: {moderation_test.get('p_value', 'N/A'):.6f}")
        print(f"유의성: {'✅ 유의함' if moderation_test.get('significant', False) else '❌ 유의하지 않음'}")
        print(f"해석: {moderation_test.get('interpretation', 'N/A')}")
        
        # 전체 계수 테이블
        coefficients = results.get('coefficients', {})
        print(f"\n📋 전체 회귀계수:")
        for var_name, coeff_info in coefficients.items():
            estimate = coeff_info.get('estimate', 0)
            p_value = coeff_info.get('p_value', 1)
            significant = '✅' if coeff_info.get('significant', False) else '❌'
            print(f"  {var_name}: {estimate:.6f} (p={p_value:.6f}) {significant}")
        
        # 단순기울기 분석
        simple_slopes = results.get('simple_slopes', {})
        if simple_slopes:
            print(f"\n📈 단순기울기 분석:")
            for level, slope_info in simple_slopes.items():
                slope = slope_info.get('simple_slope', 0)
                p_val = slope_info.get('p_value', 1)
                sig = '✅' if slope_info.get('significant', False) else '❌'
                print(f"  {level}: {slope:.6f} (p={p_val:.6f}) {sig}")
        
        # 적합도 지수
        fit_indices = results.get('fit_indices', {})
        if fit_indices:
            print(f"\n📏 모델 적합도:")
            for index_name, value in fit_indices.items():
                print(f"  {index_name}: {value:.4f}")
        
        # 결과 저장
        print(f"\n2. 결과 저장 중...")
        saved_files = export_moderation_results(results, analysis_name='semopy_test')
        
        print(f"✅ 결과 저장 성공: {len(saved_files)}개 파일")
        for file_type, file_path in saved_files.items():
            print(f"   - {file_type}: {file_path.name}")
        
        return results
        
    except Exception as e:
        print(f"❌ semopy 조절효과 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """메인 함수"""
    print("🔍 semopy 기반 조절효과 분석 테스트")
    print("=" * 60)
    
    results = test_semopy_moderation()
    
    if results:
        print("\n" + "=" * 60)
        print("🎉 semopy 조절효과 분석 성공!")
        print("=" * 60)
        
        # 핵심 결과 요약
        moderation_test = results.get('moderation_test', {})
        interaction_coef = moderation_test.get('interaction_coefficient', 0)
        p_value = moderation_test.get('p_value', 1)
        significant = moderation_test.get('significant', False)
        
        print(f"🎯 핵심 결과:")
        print(f"   상호작용 계수: {interaction_coef:.6f}")
        print(f"   P값: {p_value:.6f}")
        print(f"   유의성: {'✅ 유의함' if significant else '❌ 유의하지 않음'}")
        
        if significant:
            print(f"   해석: {moderation_test.get('interpretation', 'N/A')}")
            print("\n💡 조절효과가 통계적으로 유의합니다!")
            print("   영양지식이 건강관심도와 지각된혜택 간의 관계를 조절합니다.")
        else:
            print("\n💡 조절효과가 통계적으로 유의하지 않습니다.")
            print("   영양지식의 조절효과를 확인할 수 없습니다.")
    else:
        print("\n" + "=" * 60)
        print("❌ semopy 조절효과 분석 실패")
        print("=" * 60)
        print("💡 가능한 원인:")
        print("   - semopy 라이브러리 설치 문제")
        print("   - 데이터 형식 문제")
        print("   - 모델 수렴 문제")


if __name__ == "__main__":
    main()
