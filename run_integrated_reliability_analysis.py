#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 신뢰도 분석 실행 스크립트

이 스크립트는 다음 단계를 순차적으로 실행합니다:
1. 역문항(역코딩) 처리
2. 요인분석 실행 (필수)
3. 신뢰도 및 타당도 분석
4. 결과 시각화 및 보고서 생성

주의: 신뢰도 분석은 요인분석 결과를 기반으로 하므로, 역문항 처리 후 반드시 요인분석을 먼저 실행해야 합니다.
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from processed_data.modules.reverse_items_processor import ReverseItemsProcessor
from factor_analysis.factor_analyzer import analyze_factor_loading
from factor_analysis.results_exporter import export_factor_results
from factor_analysis.reliability_calculator import run_independent_reliability_analysis
from factor_analysis.reliability_visualizer import visualize_reliability_results
from factor_analysis.comparison_analyzer import run_comparison_analysis

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_reliability_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def check_reverse_coding_needed() -> bool:
    """
    역문항 처리가 필요한지 확인
    
    Returns:
        bool: 역문항 처리 필요 여부
    """
    try:
        processor = ReverseItemsProcessor()
        config = processor.config['reverse_items']
        
        # 역문항이 있는 요인이 있는지 확인
        for factor_name, factor_config in config.items():
            reverse_items = factor_config.get('reverse_items', [])
            if reverse_items:
                return True
        
        return False
        
    except Exception as e:
        logger.warning(f"역문항 처리 필요성 확인 중 오류: {e}")
        return False


def run_reverse_items_processing() -> bool:
    """
    역문항 처리 실행
    
    Returns:
        bool: 처리 성공 여부
    """
    try:
        print("1단계: 역문항(역코딩) 처리")
        print("-" * 50)
        
        processor = ReverseItemsProcessor()
        results = processor.process_all_factors()
        
        if 'error' in results:
            logger.error(f"역문항 처리 실패: {results['error']}")
            return False
        
        # 결과 요약
        total_processed = results.get('total_reverse_items_processed', 0)
        total_errors = results.get('total_errors', 0)
        
        print(f"✓ 역문항 처리 완료")
        print(f"  - 처리된 역문항: {total_processed}개")
        print(f"  - 오류: {total_errors}개")
        
        if total_errors > 0:
            logger.warning(f"역문항 처리 중 {total_errors}개 오류 발생")
        
        # 보고서 생성
        report = processor.generate_processing_report(results)
        report_file = f"reverse_items_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  - 보고서: {report_file}")
        print()
        
        return True
        
    except Exception as e:
        logger.error(f"역문항 처리 중 오류: {e}")
        return False


def run_factor_analysis() -> bool:
    """
    요인분석 실행

    Returns:
        bool: 분석 성공 여부
    """
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

        # 결과 요약
        factor_loadings = results.get('factor_loadings')
        if factor_loadings is not None:
            print(f"  - 분석된 요인 수: {len(factor_loadings['Factor'].unique())}")
            print(f"  - 총 문항 수: {len(factor_loadings)}")

            # 요인별 문항 수 출력
            factor_counts = factor_loadings['Factor'].value_counts()
            for factor, count in factor_counts.items():
                print(f"    * {factor}: {count}개 문항")

        # 적합도 지수 출력
        fit_indices = results.get('fit_indices')
        if fit_indices is not None:
            print("  - 모델 적합도:")
            if isinstance(fit_indices, pd.DataFrame) and not fit_indices.empty:
                for _, row in fit_indices.iterrows():
                    metric = row['Metric']
                    value = row['Value']
                    print(f"    * {metric}: {value:.4f}")
            elif isinstance(fit_indices, dict):
                for metric, value in fit_indices.items():
                    if isinstance(value, (int, float)):
                        print(f"    * {metric}: {value:.4f}")
                    else:
                        print(f"    * {metric}: {value}")

        print()
        return True

    except Exception as e:
        logger.error(f"요인분석 중 오류: {e}")
        return False


def run_reliability_analysis() -> dict:
    """
    신뢰도 분석 실행

    Returns:
        dict: 분석 결과
    """
    try:
        print("3단계: 신뢰도 및 타당도 분석")
        print("-" * 50)
        
        results = run_independent_reliability_analysis()
        
        if 'error' in results:
            logger.error(f"신뢰도 분석 실패: {results['error']}")
            return {}
        
        print("✓ 신뢰도 분석 완료")
        
        # 결과 요약
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
        
        print()
        return results
        
    except Exception as e:
        logger.error(f"신뢰도 분석 중 오류: {e}")
        return {}


def run_comparison_analysis() -> bool:
    """
    역문항 처리 전후 비교 분석 실행

    Returns:
        bool: 분석 성공 여부
    """
    try:
        print("4단계: 역문항 처리 전후 비교 분석")
        print("-" * 50)

        comparison_results = run_comparison_analysis()

        if 'error' in comparison_results:
            logger.warning(f"비교 분석 실패: {comparison_results['error']}")
            print("⚠️ 비교 분석을 건너뜁니다 (비교할 파일이 부족할 수 있습니다)")
            return True  # 비교 분석 실패는 전체 워크플로우를 중단하지 않음

        print("✓ 비교 분석 완료")

        # 결과 요약
        loadings_comparison = comparison_results.get('loadings_comparison')
        if loadings_comparison is not None and not loadings_comparison.empty:
            total_items = len(loadings_comparison)
            improved_items = loadings_comparison['Improvement'].sum()
            print(f"  - 분석 문항: {total_items}개")
            print(f"  - 개선 문항: {improved_items}개 ({improved_items/total_items:.1%})")

            # 부호 변경 문항
            sign_changed = loadings_comparison['Sign_Changed'].sum()
            if sign_changed > 0:
                print(f"  - 부호 변경: {sign_changed}개 문항")

        fit_comparison = comparison_results.get('fit_comparison')
        if fit_comparison is not None and not fit_comparison.empty:
            total_indices = len(fit_comparison)
            improved_indices = fit_comparison['Improvement'].sum()
            print(f"  - 적합도 지수: {improved_indices}/{total_indices}개 개선")

        output_dir = comparison_results.get('output_dir', '')
        if output_dir:
            print(f"  - 결과 저장: {output_dir}")

        print()
        return True

    except Exception as e:
        logger.error(f"비교 분석 중 오류: {e}")
        print("⚠️ 비교 분석에 실패했지만 계속 진행합니다.")
        return True


def run_visualization(reliability_results: dict) -> bool:
    """
    결과 시각화 실행
    
    Args:
        reliability_results (dict): 신뢰도 분석 결과
        
    Returns:
        bool: 시각화 성공 여부
    """
    try:
        print("5단계: 결과 시각화 및 보고서 생성")
        print("-" * 50)
        
        output_dir = f"integrated_reliability_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        visualize_reliability_results(reliability_results, output_dir)
        
        print("✓ 시각화 완료")
        print(f"  - 출력 디렉토리: {output_dir}")
        
        # 생성된 파일 목록
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.glob("*"))
            print(f"  - 생성된 파일: {len(files)}개")
            for file_path in sorted(files):
                print(f"    * {file_path.name}")
        
        print()
        return True
        
    except Exception as e:
        logger.error(f"시각화 중 오류: {e}")
        return False


def main():
    """메인 실행 함수"""
    print('=' * 80)
    print('통합 신뢰도 분석 실행 (역문항 처리 + 신뢰도 분석)')
    print('=' * 80)
    
    start_time = datetime.now()
    print(f'분석 시작 시간: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    try:
        # pandas import (isna 함수 사용을 위해)
        import pandas as pd
        
        # 0. 사전 확인
        print("사전 확인 중...")
        print("-" * 50)
        
        # 필요한 파일들 확인
        required_files = [
            "processed_data/reverse_items_config.json",
            "processed_data/survey_data",
            "factor_analysis_results"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ 필요한 파일/디렉토리가 없습니다:")
            for file_path in missing_files:
                print(f"  - {file_path}")
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
            print("1단계: 역문항 처리 건너뛰기 (역문항 없음)")
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
        
        # 4. 비교 분석
        comparison_success = run_comparison_analysis()
        # 비교 분석 실패는 전체 워크플로우를 중단하지 않음

        # 5. 시각화
        viz_success = run_visualization(reliability_results)
        if not viz_success:
            print("⚠️ 시각화에 실패했지만 분석은 완료되었습니다.")
        
        # 4. 완료 메시지
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("=" * 80)
        print("🎉 통합 신뢰도 분석이 성공적으로 완료되었습니다!")
        print("=" * 80)
        print(f"분석 완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 소요 시간: {duration.total_seconds():.2f}초")
        print()
        
        # 5. 결과 요약
        print("📊 최종 결과 요약:")
        print("-" * 50)
        
        reliability_stats = reliability_results.get('reliability_stats', {})
        acceptable_factors = 0
        total_factors = len(reliability_stats)
        
        for factor_name, stats in reliability_stats.items():
            alpha = stats.get('cronbach_alpha', 0)
            cr = stats.get('composite_reliability', 0)
            ave = stats.get('ave', 0)
            
            alpha_ok = alpha >= 0.7 if not pd.isna(alpha) else False
            cr_ok = cr >= 0.7 if not pd.isna(cr) else False
            ave_ok = ave >= 0.5 if not pd.isna(ave) else False
            
            if all([alpha_ok, cr_ok, ave_ok]):
                acceptable_factors += 1
        
        print(f"전체 요인 수: {total_factors}")
        print(f"신뢰도 기준 통과: {acceptable_factors}/{total_factors} ({acceptable_factors/total_factors*100:.1f}%)")
        
        # 판별타당도
        discriminant_validity = reliability_results.get('discriminant_validity', {})
        if discriminant_validity:
            valid_pairs = 0
            total_pairs = 0
            factors = list(discriminant_validity.keys())
            
            for i, factor1 in enumerate(factors):
                for j, factor2 in enumerate(factors):
                    if i < j:
                        total_pairs += 1
                        if discriminant_validity[factor1].get(factor2, False):
                            valid_pairs += 1
            
            print(f"판별타당도 통과: {valid_pairs}/{total_pairs} ({valid_pairs/total_pairs*100:.1f}%)")
        
        print()
        print("📁 생성된 결과 파일들을 확인하시기 바랍니다.")
        
    except Exception as e:
        logger.error(f"통합 분석 실행 중 오류 발생: {e}")
        print(f"\n❌ 오류 발생: {e}")
        print("자세한 내용은 로그 파일을 확인하세요: integrated_reliability_analysis.log")


def print_usage():
    """사용법 출력"""
    print("사용법:")
    print("  python run_integrated_reliability_analysis.py")
    print()
    print("설명:")
    print("  역문항 처리부터 신뢰도 분석까지 통합 실행합니다.")
    print()
    print("실행 단계:")
    print("  1. 역문항(역코딩) 처리")
    print("  2. 요인분석 실행")
    print("  3. 신뢰도 및 타당도 분석")
    print("  4. 역문항 처리 전후 비교 분석")
    print("  5. 결과 시각화 및 보고서 생성")
    print()
    print("필요한 파일들:")
    print("  - processed_data/reverse_items_config.json")
    print("  - processed_data/survey_data/*.csv")
    print()
    print("생성되는 결과:")
    print("  - factor_analysis_results/ (요인분석 결과)")
    print("  - integrated_reliability_results_*/ (통합 분석 결과)")
    print("  - comparison_analysis_results/ (비교 분석 결과)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        main()
