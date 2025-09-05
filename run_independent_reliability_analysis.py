#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
독립적인 신뢰도 및 타당도 분석 실행 스크립트

이 스크립트는 저장된 요인분석 결과 파일들을 읽어서
신뢰도 및 타당도 분석을 수행하고 결과를 시각화합니다.
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from factor_analysis.reliability_calculator import IndependentReliabilityCalculator, run_independent_reliability_analysis
from factor_analysis.reliability_visualizer import ReliabilityVisualizer, visualize_reliability_results

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('independent_reliability_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    print('=' * 80)
    print('독립적인 신뢰도 및 타당도 분석 실행')
    print('=' * 80)
    
    start_time = datetime.now()
    print(f'분석 시작 시간: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    try:
        # 1. 디렉토리 설정
        results_dir = "factor_analysis_results"
        survey_data_dir = "processed_data/survey_data"
        output_dir = "reliability_analysis_results"
        
        print(f"요인분석 결과 디렉토리: {results_dir}")
        print(f"설문 데이터 디렉토리: {survey_data_dir}")
        print(f"출력 디렉토리: {output_dir}")
        print()
        
        # 디렉토리 존재 확인
        if not os.path.exists(results_dir):
            logger.error(f"요인분석 결과 디렉토리를 찾을 수 없습니다: {results_dir}")
            return
        
        if not os.path.exists(survey_data_dir):
            logger.warning(f"설문 데이터 디렉토리를 찾을 수 없습니다: {survey_data_dir}")
            logger.warning("크론바흐 알파 계산이 제한될 수 있습니다.")
        
        # 출력 디렉토리 생성
        Path(output_dir).mkdir(exist_ok=True)
        
        # 2. 신뢰도 분석 실행
        print("신뢰도 및 타당도 분석 시작...")
        print("-" * 50)
        
        reliability_results = run_independent_reliability_analysis(
            results_dir=results_dir,
            survey_data_dir=survey_data_dir
        )
        
        if 'error' in reliability_results:
            logger.error(f"신뢰도 분석 실패: {reliability_results['error']}")
            return
        
        print("✓ 신뢰도 분석 완료")
        
        # 3. 결과 요약 출력
        print("\n신뢰도 분석 결과 요약:")
        print("-" * 50)
        
        reliability_stats = reliability_results.get('reliability_stats', {})
        for factor_name, stats in reliability_stats.items():
            print(f"\n[{factor_name}]")
            print(f"  문항 수: {stats.get('n_items', 0)}")
            print(f"  Cronbach's Alpha: {stats.get('cronbach_alpha', 'N/A'):.4f}" if not pd.isna(stats.get('cronbach_alpha')) else "  Cronbach's Alpha: N/A")
            print(f"  Composite Reliability: {stats.get('composite_reliability', 'N/A'):.4f}" if not pd.isna(stats.get('composite_reliability')) else "  Composite Reliability: N/A")
            print(f"  AVE: {stats.get('ave', 'N/A'):.4f}" if not pd.isna(stats.get('ave')) else "  AVE: N/A")
            
            # 수용성 판단
            alpha_ok = stats.get('cronbach_alpha', 0) >= 0.7 if not pd.isna(stats.get('cronbach_alpha', float('nan'))) else False
            cr_ok = stats.get('composite_reliability', 0) >= 0.7 if not pd.isna(stats.get('composite_reliability', float('nan'))) else False
            ave_ok = stats.get('ave', 0) >= 0.5 if not pd.isna(stats.get('ave', float('nan'))) else False
            
            print(f"  수용성: Alpha({'✓' if alpha_ok else '✗'}), CR({'✓' if cr_ok else '✗'}), AVE({'✓' if ave_ok else '✗'})")
        
        # 4. 시각화 실행
        print("\n\n시각화 생성 중...")
        print("-" * 50)
        
        visualize_reliability_results(reliability_results, output_dir)
        print("✓ 시각화 완료")
        
        # 5. 결과 파일 목록 출력
        print(f"\n생성된 결과 파일들 ({output_dir}):")
        print("-" * 50)
        
        output_path = Path(output_dir)
        if output_path.exists():
            for file_path in sorted(output_path.glob("*")):
                if file_path.is_file():
                    print(f"  - {file_path.name}")
        
        # 6. 완료 메시지
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n분석 완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 소요 시간: {duration.total_seconds():.2f}초")
        print("\n" + "=" * 80)
        print("독립적인 신뢰도 및 타당도 분석이 성공적으로 완료되었습니다!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"분석 실행 중 오류 발생: {e}")
        print(f"\n❌ 오류 발생: {e}")
        print("자세한 내용은 로그 파일을 확인하세요: independent_reliability_analysis.log")


def print_usage():
    """사용법 출력"""
    print("사용법:")
    print("  python run_independent_reliability_analysis.py")
    print()
    print("설명:")
    print("  저장된 요인분석 결과 파일들을 읽어서 신뢰도 및 타당도 분석을 수행합니다.")
    print()
    print("필요한 파일들:")
    print("  - factor_analysis_results/ 디렉토리의 요인분석 결과 파일들")
    print("  - processed_data/survey_data/ 디렉토리의 원본 설문 데이터 (선택사항)")
    print()
    print("생성되는 결과:")
    print("  - reliability_analysis_results/ 디렉토리에 분석 결과 및 시각화 파일들")


if __name__ == "__main__":
    # pandas import 추가 (isna 함수 사용을 위해)
    import pandas as pd
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        main()
