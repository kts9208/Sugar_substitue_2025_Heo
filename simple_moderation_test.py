"""
간단한 조절효과 분석 테스트
"""

def test_single_moderation():
    """단일 조절효과 분석 테스트"""
    print("조절효과 분석 테스트 시작...")
    
    try:
        from moderation_analysis import analyze_moderation_effects
        
        print("1. 건강관심도 → 구매의도 (영양지식 조절)")
        results1 = analyze_moderation_effects(
            independent_var='health_concern',
            dependent_var='purchase_intention',
            moderator_var='nutrition_knowledge'
        )
        
        moderation_test1 = results1.get('moderation_test', {})
        print(f"   상호작용 계수: {moderation_test1.get('interaction_coefficient', 'N/A')}")
        print(f"   p-value: {moderation_test1.get('p_value', 'N/A')}")
        print(f"   유의성: {'유의함' if moderation_test1.get('significant', False) else '유의하지 않음'}")
        
        print("\n2. 지각된 혜택 → 구매의도 (지각된 가격 조절)")
        results2 = analyze_moderation_effects(
            independent_var='perceived_benefit',
            dependent_var='purchase_intention',
            moderator_var='perceived_price'
        )
        
        moderation_test2 = results2.get('moderation_test', {})
        print(f"   상호작용 계수: {moderation_test2.get('interaction_coefficient', 'N/A')}")
        print(f"   p-value: {moderation_test2.get('p_value', 'N/A')}")
        print(f"   유의성: {'유의함' if moderation_test2.get('significant', False) else '유의하지 않음'}")
        
        print("\n✅ 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_moderation()
