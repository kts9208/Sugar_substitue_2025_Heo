"""
조절효과 분석 모듈의 60개 조합 분석 기능 테스트
"""

def test_comprehensive_moderation_analysis():
    """종합 조절효과 분석 테스트"""
    print("=" * 80)
    print("조절효과 분석 모듈 - 60개 조합 분석 기능 테스트")
    print("=" * 80)
    
    try:
        from moderation_analysis import analyze_all_moderation_combinations
        
        print("📊 5개 요인 간 모든 조절효과 조합 분석 시작...")
        
        # 모든 조합 분석 실행
        results = analyze_all_moderation_combinations(
            variables=None,  # 기본 5개 요인 사용
            save_results=True,
            show_progress=True
        )
        
        print("\n✅ 분석 완료!")
        
        # 결과 요약 출력
        summary = results.get('summary', {})
        print(f"\n📋 분석 결과 요약:")
        print(f"   총 조합 수: {summary.get('total_combinations', 0)}개")
        print(f"   성공한 분석: {summary.get('successful_analyses', 0)}개")
        print(f"   유의한 조절효과: {summary.get('significant_effects', 0)}개")
        print(f"   성공률: {summary.get('success_rate', 0):.1f}%")
        
        # 유의한 조절효과가 있는 경우 상세 정보 출력
        if summary.get('significant_effects', 0) > 0:
            print(f"\n🎯 유의한 조절효과 발견:")
            detailed_results = results.get('detailed_results', [])
            significant_results = [r for r in detailed_results if r.get('significant', False)]
            
            for result in significant_results:
                print(f"   • {result['independent']} × {result['moderator']} → {result['dependent']}")
                print(f"     계수: {result['interaction_coefficient']:.6f}, p-value: {result['p_value']:.6f}")
        else:
            print(f"\n💡 유의한 조절효과가 발견되지 않았습니다.")
        
        # 저장된 파일 정보
        if 'saved_files' in results:
            saved_files = results['saved_files']
            print(f"\n💾 저장된 파일:")
            for file_type, file_path in saved_files.items():
                print(f"   {file_type}: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_analysis():
    """개별 조절효과 분석 테스트"""
    print("\n" + "=" * 80)
    print("개별 조절효과 분석 테스트")
    print("=" * 80)
    
    try:
        from moderation_analysis import analyze_moderation_effects
        
        print("📊 건강관심도 → 구매의도 (영양지식 조절) 분석...")
        
        results = analyze_moderation_effects(
            independent_var='health_concern',
            dependent_var='purchase_intention',
            moderator_var='nutrition_knowledge'
        )
        
        # 결과 출력
        moderation_test = results.get('moderation_test', {})
        print(f"   상호작용 계수: {moderation_test.get('interaction_coefficient', 'N/A')}")
        print(f"   p-value: {moderation_test.get('p_value', 'N/A')}")
        print(f"   유의성: {'유의함' if moderation_test.get('significant', False) else '유의하지 않음'}")
        
        # 단순기울기 분석 결과
        simple_slopes = results.get('simple_slopes', {})
        if simple_slopes:
            print(f"\n   단순기울기 분석:")
            for level, slope_info in simple_slopes.items():
                if isinstance(slope_info, dict):
                    slope = slope_info.get('simple_slope', 'N/A')
                    p_val = slope_info.get('p_value', 'N/A')
                    print(f"     {level}: 기울기={slope}, p-value={p_val}")
        
        print("✅ 개별 분석 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 개별 분석 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("조절효과 분석 모듈 종합 테스트 시작")
    
    # 1. 개별 분석 테스트
    individual_success = test_individual_analysis()
    
    # 2. 종합 분석 테스트
    comprehensive_success = test_comprehensive_moderation_analysis()
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    print(f"개별 분석 테스트: {'✅ 성공' if individual_success else '❌ 실패'}")
    print(f"종합 분석 테스트: {'✅ 성공' if comprehensive_success else '❌ 실패'}")
    
    if individual_success and comprehensive_success:
        print("\n🎉 모든 테스트 성공!")
        print("💡 조절효과 분석 모듈이 60개 조합을 모두 분석할 수 있도록 구성되었습니다.")
    else:
        print("\n⚠️ 일부 테스트 실패")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
