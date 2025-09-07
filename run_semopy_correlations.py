#!/usr/bin/env python3
"""
semopy 요인간 상관계수 분석 실행 스크립트

이 스크립트는 다음 작업을 수행합니다:
1. 5개 요인 데이터 로드
2. semopy를 이용한 상관계수 및 p값 추출
3. 결과 저장

사용법:
    python run_semopy_correlations.py

Author: Sugar Substitute Research Team
Date: 2025-01-06
"""

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from factor_analysis.semopy_correlations import SemopyCorrelationExtractor


def main():
    """메인 실행 함수"""
    print("🔬 semopy 요인간 상관계수 분석")
    print("="*50)
    
    try:
        # SemopyCorrelationExtractor 인스턴스 생성
        extractor = SemopyCorrelationExtractor()
        
        # 분석 실행
        file_info = extractor.run_analysis()
        
        # 성공 메시지
        print("\n" + "="*50)
        print("✅ 분석 성공!")
        print("="*50)
        
        print(f"\n📁 생성된 파일:")
        print(f"  📊 상관계수 매트릭스: {file_info['correlation_file'].name}")
        print(f"  📈 p값 매트릭스: {file_info['pvalue_file'].name}")
        print(f"  📋 종합 결과 JSON: {file_info['json_file'].name}")
        
        print(f"\n📂 저장 위치: factor_correlations_results/")
        
        print(f"\n🎯 다음 단계:")
        print(f"  1. 생성된 CSV 파일을 Excel에서 열어 확인")
        print(f"  2. JSON 파일에서 유의한 상관관계 확인")
        print(f"  3. p < 0.05인 관계들을 중심으로 해석")
        
        return True
        
    except ImportError as e:
        print(f"❌ 라이브러리 오류: {e}")
        print("💡 해결방법: pip install semopy")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
        print("💡 해결방법: processed_data/survey_data/ 디렉토리에 데이터 파일이 있는지 확인")
        return False
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 semopy 요인간 상관계수 분석이 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n💥 분석 실행 중 오류가 발생했습니다.")
        sys.exit(1)
