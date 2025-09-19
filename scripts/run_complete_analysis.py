#!/usr/bin/env python3
"""
통합 분석 파이프라인 실행 스크립트

이 스크립트는 전체 분석 파이프라인을 순차적으로 실행합니다:
1. 요인분석 (Factor Analysis)
2. 신뢰도 분석 (Reliability Analysis)
3. 판별타당도 검증 (Discriminant Validity)
4. 상관관계 분석 (Correlation Analysis)
5. 경로분석 (Path Analysis)
6. 조절효과 분석 (Moderation Analysis) - 선택사항
7. 다항로짓 분석 (Multinomial Logit) - 선택사항

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

import sys
import os
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')
sys.path.append('..')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_script(script_path, args=None, description=""):
    """스크립트 실행 함수"""
    try:
        print(f"\n🚀 {description} 실행 중...")
        print("-" * 60)
        
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(f"✅ {description} 완료")
            if result.stdout:
                print("출력:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} 실패")
            if result.stderr:
                print("오류:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ {description} 실행 오류: {e}")
        logger.error(f"{description} 실행 오류: {e}")
        return False


def check_prerequisites():
    """사전 요구사항 확인"""
    print("🔍 사전 요구사항 확인 중...")
    print("-" * 60)
    
    # 필요한 디렉토리 확인
    required_dirs = [
        "processed_data/survey_data",
        "scripts"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ 필요한 디렉토리가 없습니다:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    
    # 필요한 스크립트 확인
    required_scripts = [
        "scripts/run_factor_analysis.py",
        "scripts/run_reliability_analysis.py",
        "scripts/run_path_analysis.py"
    ]
    
    missing_scripts = []
    for script_path in required_scripts:
        if not Path(script_path).exists():
            missing_scripts.append(script_path)
    
    if missing_scripts:
        print("❌ 필요한 스크립트가 없습니다:")
        for script_path in missing_scripts:
            print(f"  - {script_path}")
        return False
    
    print("✅ 사전 요구사항 확인 완료")
    return True


def run_factor_analysis():
    """요인분석 실행"""
    return run_script(
        "scripts/run_factor_analysis.py",
        ["--all"],
        "요인분석 (Factor Analysis)"
    )


def run_reliability_analysis():
    """신뢰도 분석 실행"""
    return run_script(
        "scripts/run_reliability_analysis.py",
        None,
        "신뢰도 분석 (Reliability Analysis)"
    )


def run_discriminant_validity():
    """판별타당도 검증 실행"""
    script_paths = [
        "scripts/run_discriminant_validity.py",
        "run_discriminant_validity_analysis.py"  # Fallback
    ]
    
    for script_path in script_paths:
        if Path(script_path).exists():
            return run_script(
                script_path,
                None,
                "판별타당도 검증 (Discriminant Validity)"
            )
    
    print("⚠️ 판별타당도 검증 스크립트를 찾을 수 없습니다. 건너뜁니다.")
    return True


def run_correlation_analysis():
    """상관관계 분석 실행"""
    script_paths = [
        "scripts/run_correlation_analysis.py",
        "run_semopy_correlations.py"  # Fallback
    ]
    
    for script_path in script_paths:
        if Path(script_path).exists():
            return run_script(
                script_path,
                None,
                "상관관계 분석 (Correlation Analysis)"
            )
    
    print("⚠️ 상관관계 분석 스크립트를 찾을 수 없습니다. 건너뜁니다.")
    return True


def run_path_analysis():
    """경로분석 실행"""
    return run_script(
        "scripts/run_path_analysis.py",
        ["--model", "all"],
        "경로분석 (Path Analysis)"
    )


def run_moderation_analysis():
    """조절효과 분석 실행"""
    script_paths = [
        "scripts/run_moderation_analysis.py",
        "run_moderation_analysis.py"  # Fallback
    ]
    
    for script_path in script_paths:
        if Path(script_path).exists():
            return run_script(
                script_path,
                None,
                "조절효과 분석 (Moderation Analysis)"
            )
    
    print("⚠️ 조절효과 분석 스크립트를 찾을 수 없습니다. 건너뜁니다.")
    return True


def run_multinomial_logit():
    """다항로짓 분석 실행"""
    script_paths = [
        "scripts/run_multinomial_logit.py",
        "multinomial_logit/mnl_analysis.py"  # Fallback
    ]
    
    for script_path in script_paths:
        if Path(script_path).exists():
            return run_script(
                script_path,
                None,
                "다항로짓 분석 (Multinomial Logit)"
            )
    
    print("⚠️ 다항로짓 분석 스크립트를 찾을 수 없습니다. 건너뜁니다.")
    return True


def generate_final_report():
    """최종 보고서 생성"""
    print("\n📊 최종 보고서 생성 중...")
    print("-" * 60)
    
    try:
        # 결과 디렉토리들 확인
        result_dirs = [
            "factor_analysis_results",
            "reliability_analysis_results", 
            "path_analysis_results",
            "discriminant_validity_results",
            "correlation_visualization_results"
        ]
        
        existing_dirs = []
        for dir_path in result_dirs:
            if Path(dir_path).exists():
                existing_dirs.append(dir_path)
        
        print(f"✅ 생성된 결과 디렉토리: {len(existing_dirs)}개")
        for dir_path in existing_dirs:
            file_count = len(list(Path(dir_path).glob("*")))
            print(f"  📁 {dir_path}: {file_count}개 파일")
        
        return True
        
    except Exception as e:
        print(f"❌ 최종 보고서 생성 오류: {e}")
        return False


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='통합 분석 파이프라인 실행')
    parser.add_argument('--skip-factor', action='store_true', help='요인분석 건너뛰기')
    parser.add_argument('--skip-reliability', action='store_true', help='신뢰도 분석 건너뛰기')
    parser.add_argument('--skip-path', action='store_true', help='경로분석 건너뛰기')
    parser.add_argument('--include-moderation', action='store_true', help='조절효과 분석 포함')
    parser.add_argument('--include-mnl', action='store_true', help='다항로짓 분석 포함')
    parser.add_argument('--core-only', action='store_true', help='핵심 분석만 실행 (요인분석, 신뢰도, 경로분석)')
    
    args = parser.parse_args()
    
    print("🎯 통합 분석 파이프라인 실행")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    # 사전 요구사항 확인
    if not check_prerequisites():
        print("❌ 사전 요구사항을 만족하지 않습니다. 실행을 중단합니다.")
        return
    
    # 분석 실행 계획
    analysis_plan = []
    
    if not args.skip_factor:
        analysis_plan.append(("요인분석", run_factor_analysis))
    
    if not args.skip_reliability:
        analysis_plan.append(("신뢰도 분석", run_reliability_analysis))
    
    if not args.core_only:
        analysis_plan.append(("판별타당도 검증", run_discriminant_validity))
        analysis_plan.append(("상관관계 분석", run_correlation_analysis))
    
    if not args.skip_path:
        analysis_plan.append(("경로분석", run_path_analysis))
    
    if args.include_moderation and not args.core_only:
        analysis_plan.append(("조절효과 분석", run_moderation_analysis))
    
    if args.include_mnl and not args.core_only:
        analysis_plan.append(("다항로짓 분석", run_multinomial_logit))
    
    print(f"\n📋 실행 계획: {len(analysis_plan)}개 분석")
    for i, (name, _) in enumerate(analysis_plan, 1):
        print(f"  {i}. {name}")
    
    # 분석 실행
    successful_analyses = []
    failed_analyses = []
    
    for analysis_name, analysis_func in analysis_plan:
        print(f"\n{'='*80}")
        print(f"단계 {len(successful_analyses) + len(failed_analyses) + 1}: {analysis_name}")
        print(f"{'='*80}")
        
        if analysis_func():
            successful_analyses.append(analysis_name)
        else:
            failed_analyses.append(analysis_name)
            print(f"⚠️ {analysis_name} 실패. 다음 단계로 진행합니다.")
    
    # 최종 보고서 생성
    generate_final_report()
    
    # 최종 요약
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("🎉 통합 분석 파이프라인 완료!")
    print(f"{'='*80}")
    print(f"총 소요 시간: {duration}")
    print(f"완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 실행 결과:")
    print(f"  ✅ 성공: {len(successful_analyses)}개")
    for analysis in successful_analyses:
        print(f"    - {analysis}")
    
    if failed_analyses:
        print(f"  ❌ 실패: {len(failed_analyses)}개")
        for analysis in failed_analyses:
            print(f"    - {analysis}")
    
    print(f"\n📁 결과 파일 위치:")
    print(f"  📊 요인분석: factor_analysis_results/")
    print(f"  📈 신뢰도 분석: reliability_analysis_results/")
    print(f"  📉 경로분석: path_analysis_results/")
    print(f"  📋 기타 분석: 각 분석별 디렉토리")
    
    print(f"\n🎯 권장 후속 작업:")
    print(f"  1. 각 분석 결과 검토")
    print(f"  2. 적합도 지수 및 유의성 확인")
    print(f"  3. 연구 보고서 작성")


if __name__ == "__main__":
    main()
