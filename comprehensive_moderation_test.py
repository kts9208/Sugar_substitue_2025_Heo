"""
포괄적 조절효과 분석 테스트

다양한 변수 조합으로 조절효과 분석을 수행하고 결과를 비교합니다.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_comprehensive_moderation_analysis():
    """포괄적 조절효과 분석 실행"""
    
    # 분석할 변수 조합들
    analysis_combinations = [
        {
            'name': 'health_nutrition_purchase',
            'independent': 'health_concern',
            'dependent': 'purchase_intention',
            'moderator': 'nutrition_knowledge',
            'description': '건강관심도 → 구매의도 (영양지식 조절)'
        },
        {
            'name': 'health_price_purchase',
            'independent': 'health_concern',
            'dependent': 'purchase_intention',
            'moderator': 'perceived_price',
            'description': '건강관심도 → 구매의도 (지각된 가격 조절)'
        },
        {
            'name': 'benefit_nutrition_purchase',
            'independent': 'perceived_benefit',
            'dependent': 'purchase_intention',
            'moderator': 'nutrition_knowledge',
            'description': '지각된 혜택 → 구매의도 (영양지식 조절)'
        },
        {
            'name': 'benefit_price_purchase',
            'independent': 'perceived_benefit',
            'dependent': 'purchase_intention',
            'moderator': 'perceived_price',
            'description': '지각된 혜택 → 구매의도 (지각된 가격 조절)'
        }
    ]
    
    results_summary = []
    
    print("=" * 80)
    print("포괄적 조절효과 분석 시작")
    print("=" * 80)
    
    for i, combo in enumerate(analysis_combinations, 1):
        print(f"\n{i}. {combo['description']}")
        print("-" * 60)
        
        try:
            from moderation_analysis import analyze_moderation_effects
            
            # 조절효과 분석 실행
            results = analyze_moderation_effects(
                independent_var=combo['independent'],
                dependent_var=combo['dependent'],
                moderator_var=combo['moderator']
            )
            
            # 주요 결과 추출
            moderation_test = results.get('moderation_test', {})
            fit_indices = results.get('fit_indices', {})
            
            summary = {
                'analysis_name': combo['name'],
                'description': combo['description'],
                'independent_var': combo['independent'],
                'dependent_var': combo['dependent'],
                'moderator_var': combo['moderator'],
                'interaction_coefficient': moderation_test.get('interaction_coefficient', 'N/A'),
                'p_value': moderation_test.get('p_value', 'N/A'),
                'significant': moderation_test.get('significant', False),
                'interpretation': moderation_test.get('interpretation', 'N/A'),
                'cfi': fit_indices.get('CFI', 'N/A'),
                'rmsea': fit_indices.get('RMSEA', 'N/A'),
                'aic': fit_indices.get('AIC', 'N/A')
            }
            
            results_summary.append(summary)
            
            # 결과 출력
            print(f"✅ 분석 성공")
            print(f"   상호작용 계수: {summary['interaction_coefficient']}")
            print(f"   p-value: {summary['p_value']}")
            print(f"   유의성: {'유의함' if summary['significant'] else '유의하지 않음'}")
            print(f"   CFI: {summary['cfi']}")
            print(f"   RMSEA: {summary['rmsea']}")
            
            # 결과 저장
            try:
                from moderation_analysis import export_moderation_results
                saved_files = export_moderation_results(results, analysis_name=combo['name'])
                print(f"   결과 저장: {len(saved_files)}개 파일")
            except Exception as e:
                print(f"   ⚠️ 결과 저장 실패: {e}")
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            summary = {
                'analysis_name': combo['name'],
                'description': combo['description'],
                'error': str(e)
            }
            results_summary.append(summary)
    
    # 종합 결과 저장
    save_comprehensive_summary(results_summary)
    
    # 결과 비교 출력
    print_comparison_results(results_summary)

def save_comprehensive_summary(results_summary):
    """종합 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV 저장
    df = pd.DataFrame(results_summary)
    csv_path = f"moderation_analysis_results/comprehensive_analysis_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # JSON 저장
    json_path = f"moderation_analysis_results/comprehensive_analysis_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n종합 결과 저장:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")

def print_comparison_results(results_summary):
    """결과 비교 출력"""
    print("\n" + "=" * 80)
    print("조절효과 분석 결과 비교")
    print("=" * 80)
    
    # 유의한 조절효과가 있는 분석들
    significant_results = [r for r in results_summary if r.get('significant', False)]
    
    print(f"\n📊 전체 분석 수: {len(results_summary)}")
    print(f"📈 유의한 조절효과: {len(significant_results)}개")
    
    if significant_results:
        print("\n🎯 유의한 조절효과 결과:")
        for result in significant_results:
            print(f"  • {result['description']}")
            print(f"    계수: {result['interaction_coefficient']:.6f}, p-value: {result['p_value']:.6f}")
    
    # 모델 적합도 비교
    print(f"\n📋 모델 적합도 비교:")
    print(f"{'분석명':<25} {'CFI':<8} {'RMSEA':<8} {'AIC':<12}")
    print("-" * 55)
    
    for result in results_summary:
        if 'error' not in result:
            cfi = f"{result['cfi']:.3f}" if isinstance(result['cfi'], (int, float)) else "N/A"
            rmsea = f"{result['rmsea']:.3f}" if isinstance(result['rmsea'], (int, float)) else "N/A"
            aic = f"{result['aic']:.1f}" if isinstance(result['aic'], (int, float)) else "N/A"
            print(f"{result['analysis_name']:<25} {cfi:<8} {rmsea:<8} {aic:<12}")

def analyze_interaction_patterns():
    """상호작용 패턴 분석"""
    print("\n" + "=" * 80)
    print("상호작용 패턴 상세 분석")
    print("=" * 80)
    
    try:
        from moderation_analysis import load_moderation_data, create_interaction_terms
        
        # 가장 흥미로운 조합 선택 (건강관심도 → 구매의도, 영양지식 조절)
        data = load_moderation_data(
            independent_var='health_concern',
            dependent_var='purchase_intention',
            moderator_var='nutrition_knowledge'
        )
        
        # 상호작용항 생성
        interaction_data = create_interaction_terms(
            data=data,
            independent_var='health_concern',
            moderator_var='nutrition_knowledge'
        )
        
        # 기술통계 분석
        print("\n📈 변수별 기술통계:")
        desc_stats = interaction_data.describe()
        print(desc_stats)
        
        # 상관관계 분석
        print("\n🔗 변수간 상관관계:")
        corr_matrix = interaction_data.corr()
        print(corr_matrix)
        
        # 조절변수 수준별 분석
        print("\n📊 조절변수 수준별 분석:")
        
        # 영양지식을 3분위로 나누기
        nutrition_tertiles = pd.qcut(interaction_data['nutrition_knowledge'], 3, labels=['Low', 'Medium', 'High'])
        
        for level in ['Low', 'Medium', 'High']:
            subset = interaction_data[nutrition_tertiles == level]
            corr = subset['health_concern'].corr(subset['purchase_intention'])
            print(f"  영양지식 {level}: 건강관심도-구매의도 상관관계 = {corr:.4f} (n={len(subset)})")
        
    except Exception as e:
        print(f"❌ 상호작용 패턴 분석 실패: {e}")

def main():
    """메인 함수"""
    try:
        # 1. 포괄적 조절효과 분석
        run_comprehensive_moderation_analysis()
        
        # 2. 상호작용 패턴 분석
        analyze_interaction_patterns()
        
        print("\n" + "=" * 80)
        print("포괄적 조절효과 분석 완료")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 전체 분석 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
