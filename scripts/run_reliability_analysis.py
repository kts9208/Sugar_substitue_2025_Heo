#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 신뢰도 분석 실행 스크립트

이 스크립트는 다음 단계를 순차적으로 실행합니다:
1. 역문항(역코딩) 처리 (필요시)
2. 요인분석 실행
3. 신뢰도 및 타당도 분석
4. 결과 시각화 및 보고서 생성

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')
sys.path.append('..')

try:
    from src.analysis.factor_analysis.factor_analyzer import analyze_factor_loading
    from src.analysis.factor_analysis.results_exporter import export_factor_results
    from src.analysis.factor_analysis.reliability_calculator import run_independent_reliability_analysis
    from src.analysis.factor_analysis.reliability_visualizer import visualize_reliability_results
    from src.utils.results_manager import save_results, archive_previous_results
except ImportError:
    # Fallback to current structure
    from factor_analysis.factor_analyzer import analyze_factor_loading
    from factor_analysis.results_exporter import export_factor_results
    from factor_analysis.reliability_calculator import run_independent_reliability_analysis
    from factor_analysis.reliability_visualizer import visualize_reliability_results

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reliability_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_reverse_coding_needed():
    """역문항 처리 필요성 확인"""
    try:
        config_path = Path("data/config/reverse_items_config.json")
        if not config_path.exists():
            # Fallback to current structure
            config_path = Path("processed_data/reverse_items_config.json")
        
        if config_path.exists():
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return len(config.get('reverse_items', {})) > 0
        return False
    except Exception as e:
        logger.warning(f"역문항 설정 확인 실패: {e}")
        return False


def run_reverse_items_processing():
    """역문항 처리 실행"""
    try:
        print("1단계: 역문항 처리")
        print("-" * 50)
        
        from processed_data.modules.reverse_items_processor import ReverseItemsProcessor
        
        processor = ReverseItemsProcessor()
        success = processor.process_reverse_items()
        
        if success:
            print("✓ 역문항 처리 완료")
            return True
        else:
            print("❌ 역문항 처리 실패")
            return False
            
    except Exception as e:
        logger.error(f"역문항 처리 오류: {e}")
        print(f"❌ 역문항 처리 오류: {e}")
        return False


def run_factor_analysis():
    """요인분석 실행"""
    try:
        print("2단계: 요인분석 실행")
        print("-" * 50)

        # 분석할 요인들 (DCE 관련 요인 제외)
        factors_to_analyze = [
            'health_concern',
            'perceived_benefit', 
            'purchase_intention',
            'perceived_price',
            'nutrition_knowledge'
        ]

        print(f"분석 대상 요인: {', '.join(factors_to_analyze)}")

        # 다중 요인 분석 실행
        results = analyze_factor_loading(factors_to_analyze)

        if 'error' in results:
            logger.error(f"요인분석 실패: {results['error']}")
            return False

        print("✓ 요인분석 완료")

        # 결과 저장
        saved_files = export_factor_results(results)
        if saved_files:
            print(f"  - 결과 저장 완료: {len(saved_files)}개 파일")
            for file_type, file_path in saved_files.items():
                print(f"    * {file_type}: {os.path.basename(file_path)}")

        return True

    except Exception as e:
        logger.error(f"요인분석 오류: {e}")
        print(f"❌ 요인분석 오류: {e}")
        return False


def run_reliability_analysis():
    """신뢰도 및 타당도 분석 실행"""
    try:
        print("3단계: 신뢰도 및 타당도 분석")
        print("-" * 50)
        
        results = run_independent_reliability_analysis()
        
        if 'error' in results:
            logger.error(f"신뢰도 분석 실패: {results['error']}")
            return {}
        
        print("✓ 신뢰도 분석 완료")
        
        # 결과 요약 출력
        reliability_stats = results.get('reliability_stats', {})
        for factor_name, stats in reliability_stats.items():
            alpha = stats.get('cronbach_alpha', 0)
            cr = stats.get('composite_reliability', 0)
            ave = stats.get('ave', 0)
            
            alpha_ok = alpha >= 0.7 if not pd.isna(alpha) else False
            cr_ok = cr >= 0.7 if not pd.isna(cr) else False
            ave_ok = ave >= 0.5 if not pd.isna(ave) else False
            
            status = "✓" if all([alpha_ok, cr_ok, ave_ok]) else "⚠"
            print(f"  {status} {factor_name}: α={alpha:.3f}, CR={cr:.3f}, AVE={ave:.3f}")
        
        return results

    except Exception as e:
        logger.error(f"신뢰도 분석 오류: {e}")
        print(f"❌ 신뢰도 분석 오류: {e}")
        return {}


def run_visualization():
    """결과 시각화"""
    try:
        print("4단계: 결과 시각화")
        print("-" * 50)
        
        viz_results = visualize_reliability_results()
        
        if viz_results:
            print("✓ 시각화 완료")
            print(f"  - 생성된 차트: {len(viz_results)}개")
            for chart_name, chart_path in viz_results.items():
                print(f"    * {chart_name}: {os.path.basename(chart_path)}")
        
        return viz_results

    except Exception as e:
        logger.error(f"시각화 오류: {e}")
        print(f"❌ 시각화 오류: {e}")
        return {}


def main():
    """메인 실행 함수"""
    print('=' * 80)
    print('통합 신뢰도 분석 실행')
    print('=' * 80)
    
    start_time = datetime.now()
    print(f'분석 시작 시간: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    try:
        # 0. 사전 확인
        print("사전 확인 중...")
        print("-" * 50)
        
        # 필요한 파일들 확인
        required_paths = [
            "processed_data/survey_data",
            "factor_analysis_results"
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            print("❌ 필요한 파일/디렉토리가 없습니다:")
            for path in missing_paths:
                print(f"  - {path}")
            return
        
        print("✓ 필요한 파일들 확인 완료")
        
        # 역문항 처리 필요성 확인
        needs_reverse_coding = check_reverse_coding_needed()
        print(f"✓ 역문항 처리 필요: {'예' if needs_reverse_coding else '아니오'}")
        print()
        
        # 1. 역문항 처리 (필요한 경우)
        if needs_reverse_coding:
            reverse_success = run_reverse_items_processing()
            if not reverse_success:
                print("❌ 역문항 처리 실패로 인해 분석을 중단합니다.")
                return
        else:
            print("1단계: 역문항 처리 건너뛰기")
            print("-" * 50)
            print("✓ 역문항이 없어 처리를 건너뜁니다.")
            print()
        
        # 2. 요인분석 실행
        factor_analysis_success = run_factor_analysis()
        if not factor_analysis_success:
            print("❌ 요인분석 실패로 인해 분석을 중단합니다.")
            return

        # 3. 신뢰도 분석
        reliability_results = run_reliability_analysis()
        if not reliability_results:
            print("❌ 신뢰도 분석 실패로 인해 분석을 중단합니다.")
            return

        # 4. 시각화
        viz_results = run_visualization()
        
        # 5. 최종 요약
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("✅ 통합 신뢰도 분석 완료!")
        print("=" * 80)
        print(f"총 소요 시간: {duration}")
        print(f"완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📁 결과 파일 위치:")
        print(f"  📊 요인분석 결과: factor_analysis_results/")
        print(f"  📈 신뢰도 분석 결과: reliability_analysis_results/")
        if viz_results:
            print(f"  📉 시각화 결과: {len(viz_results)}개 차트 생성")
        
        print(f"\n🎯 다음 단계 권장:")
        print(f"  1. 신뢰도 지수 확인 (α ≥ 0.7, CR ≥ 0.7, AVE ≥ 0.5)")
        print(f"  2. 판별타당도 검증 실행")
        print(f"  3. 경로분석 실행")

    except Exception as e:
        logger.error(f"분석 실행 중 오류: {e}")
        print(f"❌ 분석 실행 중 오류: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
    else:
        main()
