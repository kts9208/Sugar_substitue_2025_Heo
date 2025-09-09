#!/usr/bin/env python3
"""
조절효과 분석 메인 실행 스크립트

5개 요인 간 조절효과 분석을 수행하고 결과를 저장합니다.
- health_concern (건강관심도): q6~q11
- perceived_benefit (지각된혜택): q16~q17  
- purchase_intention (구매의도): q18~q19
- perceived_price (지각된가격): q20~q21
- nutrition_knowledge (영양지식): q30~q49
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
sys.path.append('.')

# 조절효과 분석 모듈 임포트
from moderation_analysis import (
    analyze_moderation_effects,
    export_moderation_results,
    visualize_moderation_analysis,
    create_default_moderation_config,
    get_available_factors
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moderation_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🔍 조절효과 분석 (Moderation Analysis) 시스템")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. 사용 가능한 요인 확인
        available_factors = get_available_factors()
        print(f"📋 사용 가능한 요인: {len(available_factors)}개")
        for i, factor in enumerate(available_factors, 1):
            print(f"   {i}. {factor}")
        print()
        
        # 2. 분석할 조절효과 조합 정의
        moderation_analyses = [
            {
                'name': 'health_nutrition_purchase',
                'independent': 'health_concern',
                'dependent': 'purchase_intention',
                'moderator': 'nutrition_knowledge',
                'description': '건강관심도 → 구매의도 (영양지식 조절)'
            },
            {
                'name': 'benefit_price_purchase',
                'independent': 'perceived_benefit',
                'dependent': 'purchase_intention',
                'moderator': 'perceived_price',
                'description': '지각된혜택 → 구매의도 (지각된가격 조절)'
            },
            {
                'name': 'health_price_benefit',
                'independent': 'health_concern',
                'dependent': 'perceived_benefit',
                'moderator': 'perceived_price',
                'description': '건강관심도 → 지각된혜택 (지각된가격 조절)'
            }
        ]
        
        print(f"🎯 분석할 조절효과: {len(moderation_analyses)}개")
        for i, analysis in enumerate(moderation_analyses, 1):
            print(f"   {i}. {analysis['description']}")
        print()
        
        # 3. 각 조절효과 분석 실행
        all_results = {}
        
        for i, analysis_config in enumerate(moderation_analyses, 1):
            print(f"🔄 분석 {i}/{len(moderation_analyses)}: {analysis_config['name']}")
            print(f"   {analysis_config['description']}")
            
            try:
                # 조절효과 분석 실행
                from moderation_analysis import load_moderation_data, create_interaction_terms

                # 데이터 로드
                data = load_moderation_data(
                    independent_var=analysis_config['independent'],
                    dependent_var=analysis_config['dependent'],
                    moderator_var=analysis_config['moderator']
                )

                # 상호작용항 추가
                data = create_interaction_terms(
                    data=data,
                    independent_var=analysis_config['independent'],
                    moderator_var=analysis_config['moderator']
                )

                # 조절효과 분석 실행
                results = analyze_moderation_effects(
                    independent_var=analysis_config['independent'],
                    dependent_var=analysis_config['dependent'],
                    moderator_var=analysis_config['moderator'],
                    data=data
                )

                # 결과 저장
                saved_files = export_moderation_results(
                    results,
                    analysis_name=analysis_config['name']
                )

                # 시각화 생성
                plot_files = visualize_moderation_analysis(
                    data=data,
                    results=results,
                    analysis_name=analysis_config['name']
                )
                
                # 결과 요약 출력
                print_analysis_summary(results, analysis_config['name'])
                
                # 저장된 파일 정보 출력
                print(f"   💾 저장된 파일: {len(saved_files)}개")
                for file_type, file_path in saved_files.items():
                    print(f"      - {file_type}: {file_path.name}")
                
                if plot_files:
                    print(f"   📊 생성된 그래프: {len(plot_files)}개")
                    for plot_type, plot_path in plot_files.items():
                        print(f"      - {plot_type}: {plot_path.name}")
                
                all_results[analysis_config['name']] = results
                print("   ✅ 분석 완료!")
                
            except Exception as e:
                print(f"   ❌ 분석 실패: {e}")
                logger.error(f"분석 {analysis_config['name']} 실패: {e}")
                continue
            
            print()
        
        # 4. 전체 결과 요약
        print_overall_summary(all_results)
        
        print("🎉 모든 조절효과 분석 완료!")
        print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ 전체 분석 실패: {e}")
        logger.error(f"전체 분석 실패: {e}")
        sys.exit(1)


def print_analysis_summary(results: Dict[str, Any], analysis_name: str):
    """개별 분석 결과 요약 출력"""
    print(f"   📊 {analysis_name} 분석 결과:")
    
    # 모델 정보
    model_info = results.get('model_info', {})
    print(f"      관측치 수: {model_info.get('n_observations', 'N/A')}")
    
    # 조절효과 검정
    moderation_test = results.get('moderation_test', {})
    interaction_coeff = moderation_test.get('interaction_coefficient', 0)
    p_value = moderation_test.get('p_value', 1)
    significant = moderation_test.get('significant', False)
    
    print(f"      상호작용 계수: {interaction_coeff:.4f}")
    print(f"      P값: {p_value:.4f}")
    print(f"      유의성: {'✅ 유의함' if significant else '❌ 유의하지 않음'}")
    
    if significant:
        interpretation = moderation_test.get('interpretation', '')
        print(f"      해석: {interpretation}")
    
    # 단순기울기 요약
    simple_slopes = results.get('simple_slopes', {})
    if simple_slopes:
        significant_slopes = sum(1 for slope in simple_slopes.values() 
                               if slope.get('significant', False))
        print(f"      단순기울기: {significant_slopes}/{len(simple_slopes)}개 유의함")


def print_overall_summary(all_results: Dict[str, Dict[str, Any]]):
    """전체 결과 요약 출력"""
    print("=" * 60)
    print("📋 전체 조절효과 분석 요약")
    print("=" * 60)
    
    total_analyses = len(all_results)
    significant_moderations = 0
    
    print(f"총 분석 수: {total_analyses}")
    print()
    
    print("분석별 결과:")
    print("-" * 40)
    
    for analysis_name, results in all_results.items():
        moderation_test = results.get('moderation_test', {})
        significant = moderation_test.get('significant', False)
        p_value = moderation_test.get('p_value', 1)
        
        if significant:
            significant_moderations += 1
        
        status = "✅ 유의함" if significant else "❌ 유의하지 않음"
        print(f"{analysis_name}: {status} (p={p_value:.4f})")
    
    print()
    print(f"유의한 조절효과: {significant_moderations}/{total_analyses}개 ({significant_moderations/total_analyses*100:.1f}%)")
    
    # 권장사항
    print()
    print("💡 권장사항:")
    if significant_moderations > 0:
        print("- 유의한 조절효과가 발견되었습니다.")
        print("- 단순기울기 분석 결과를 자세히 검토하세요.")
        print("- 시각화 그래프를 통해 상호작용 패턴을 확인하세요.")
    else:
        print("- 유의한 조절효과가 발견되지 않았습니다.")
        print("- 다른 조절변수를 고려해보세요.")
        print("- 표본 크기나 측정 방법을 검토해보세요.")


def run_custom_analysis():
    """사용자 정의 조절효과 분석"""
    print("\n🔧 사용자 정의 조절효과 분석")
    print("-" * 40)
    
    available_factors = get_available_factors()
    print(f"사용 가능한 요인: {', '.join(available_factors)}")
    
    try:
        # 사용자 입력
        independent_var = input("독립변수를 입력하세요: ").strip()
        dependent_var = input("종속변수를 입력하세요: ").strip()
        moderator_var = input("조절변수를 입력하세요: ").strip()
        
        # 유효성 검증
        for var in [independent_var, dependent_var, moderator_var]:
            if var not in available_factors:
                print(f"❌ 잘못된 요인명: {var}")
                return
        
        if independent_var == dependent_var:
            print("❌ 독립변수와 종속변수는 달라야 합니다.")
            return
        
        # 분석 실행
        print(f"\n🔄 분석 실행: {independent_var} × {moderator_var} → {dependent_var}")
        
        results = analyze_moderation_effects(
            independent_var=independent_var,
            dependent_var=dependent_var,
            moderator_var=moderator_var
        )
        
        # 결과 저장
        analysis_name = f"custom_{independent_var}_x_{moderator_var}_to_{dependent_var}"
        saved_files = export_moderation_results(results, analysis_name)
        
        # 결과 출력
        print_analysis_summary(results, analysis_name)
        
        print(f"\n💾 결과 저장 완료: {len(saved_files)}개 파일")
        for file_type, file_path in saved_files.items():
            print(f"   - {file_type}: {file_path}")
        
    except KeyboardInterrupt:
        print("\n분석이 취소되었습니다.")
    except Exception as e:
        print(f"\n❌ 분석 실패: {e}")


if __name__ == "__main__":
    # 명령행 인수 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--custom":
        run_custom_analysis()
    else:
        main()
