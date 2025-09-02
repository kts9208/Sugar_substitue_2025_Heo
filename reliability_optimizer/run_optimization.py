"""
신뢰도 최적화 실행 스크립트

기존 신뢰도 분석 결과를 입력받아 AVE 기준을 만족하지 못하는 요인의
문항들을 체계적으로 제거하여 최적의 문항 조합을 찾습니다.

사용법:
    python run_optimization.py

Author: Reliability Optimization System
Date: 2025-01-02
"""

import sys
import logging
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from reliability_optimizer import ReliabilityOptimizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    print("🔧 === 신뢰도 최적화 시스템 실행 ===")
    
    try:
        # 1. 최적화기 초기화
        print("\n📋 1단계: 최적화기 초기화")
        optimizer = ReliabilityOptimizer("reliability_analysis_results")
        
        # 2. 기존 신뢰도 분석 결과 로드
        print("\n📊 2단계: 기존 신뢰도 분석 결과 로드")
        if not optimizer.load_reliability_results():
            print("❌ 신뢰도 분석 결과 로드 실패")
            return False
        
        # 3. 원시 데이터 로드 (크론바흐 알파 계산용)
        print("\n📁 3단계: 원시 데이터 로드")
        # nutrition_knowledge 요인의 원시 데이터 로드
        raw_data_path = "processed_data/survey_data/nutrition_knowledge.csv"
        if not optimizer.load_raw_data(raw_data_path):
            print("❌ 원시 데이터 로드 실패")
            return False
        
        # 4. 문제 요인 식별
        print("\n🔍 4단계: 문제 요인 식별")
        problematic_factors = optimizer.identify_problematic_factors()
        
        if not problematic_factors:
            print("✅ 모든 요인이 신뢰도 기준을 만족합니다!")
            return True
        
        print(f"📋 문제 요인 발견: {len(problematic_factors)}개")
        for factor in problematic_factors:
            print(f"   - {factor}")
        
        # 5. 신뢰도 최적화 실행
        print("\n⚡ 5단계: 신뢰도 최적화 실행")
        optimization_results = optimizer.optimize_all_problematic_factors(max_removals=10)
        
        # 6. 결과 출력
        print("\n📈 6단계: 최적화 결과 출력")
        optimizer.print_optimization_summary(optimization_results)
        
        # 7. 보고서 생성
        print("\n📄 7단계: 최적화 보고서 생성")
        if optimizer.generate_optimization_report(optimization_results):
            print("✅ 보고서 생성 완료: reliability_optimization_results/")
        else:
            print("❌ 보고서 생성 실패")
        
        print("\n🎉 === 신뢰도 최적화 완료! ===")
        return True
        
    except Exception as e:
        logger.error(f"최적화 실행 중 오류: {e}")
        print(f"\n❌ 오류 발생: {e}")
        return False


def run_specific_factor_optimization(factor_name: str, max_removals: int = 10):
    """
    특정 요인에 대한 최적화 실행
    
    Args:
        factor_name (str): 최적화할 요인명
        max_removals (int): 최대 제거할 문항 수
    """
    print(f"🔧 === '{factor_name}' 요인 신뢰도 최적화 ===")
    
    try:
        # 최적화기 초기화
        optimizer = ReliabilityOptimizer("reliability_analysis_results")
        
        # 결과 로드
        if not optimizer.load_reliability_results():
            print("❌ 신뢰도 분석 결과 로드 실패")
            return False
        
        # 원시 데이터 로드
        if factor_name == "nutrition_knowledge":
            raw_data_path = "processed_data/survey_data/nutrition_knowledge.csv"
        else:
            raw_data_path = f"processed_data/survey_data/{factor_name}.csv"
        
        if not optimizer.load_raw_data(raw_data_path):
            print("❌ 원시 데이터 로드 실패")
            return False
        
        # 특정 요인 최적화
        result = optimizer.optimize_factor_reliability(factor_name, max_removals)
        
        # 결과 출력
        if 'error' in result:
            print(f"❌ 오류: {result['error']}")
            return False
        
        print(f"\n📊 '{factor_name}' 최적화 결과:")
        
        original_stats = result['original_stats']
        print(f"\n📈 원본 신뢰도:")
        print(f"   - 문항 수: {original_stats['n_items']}개")
        print(f"   - Cronbach's α: {original_stats['cronbach_alpha']:.4f}")
        print(f"   - CR: {original_stats['composite_reliability']:.4f}")
        print(f"   - AVE: {original_stats['ave']:.4f}")
        
        best_solution = result['best_solution']
        if best_solution:
            print(f"\n✨ 최적화 결과:")
            print(f"   - 제거 문항: {len(best_solution['items_removed'])}개")
            print(f"   - 제거할 문항들: {', '.join(best_solution['items_removed'])}")
            print(f"   - 남은 문항: {best_solution['n_remaining']}개")
            print(f"   - Cronbach's α: {best_solution['cronbach_alpha']:.4f}")
            print(f"   - CR: {best_solution['composite_reliability']:.4f}")
            print(f"   - AVE: {best_solution['ave']:.4f}")
            print(f"   - 모든 기준 충족: {'✅' if best_solution['meets_all_criteria'] else '❌'}")
        else:
            print(f"\n❌ 최적화 실패: 기준을 만족하는 해결책을 찾지 못했습니다.")
        
        return True
        
    except Exception as e:
        logger.error(f"특정 요인 최적화 중 오류: {e}")
        print(f"\n❌ 오류 발생: {e}")
        return False


if __name__ == "__main__":
    # 전체 최적화 실행
    success = main()
    
    # nutrition_knowledge 요인에 대한 상세 분석도 실행
    print("\n" + "="*80)
    print("🔍 nutrition_knowledge 요인 상세 분석")
    print("="*80)
    run_specific_factor_optimization("nutrition_knowledge", max_removals=15)
    
    if success:
        print("\n✨ 신뢰도 최적화가 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n💥 신뢰도 최적화 실행 중 오류가 발생했습니다!")
        sys.exit(1)
