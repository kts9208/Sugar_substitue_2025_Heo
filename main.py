#!/usr/bin/env python3
"""
Sugar Substitute Research - 통합 분석 시스템

이 스크립트는 설탕 대체재 연구의 모든 분석을 통합적으로 관리합니다:
1. 요인분석 (Factor Analysis)
2. 신뢰도 분석 (Reliability Analysis)  
3. 판별타당도 검증 (Discriminant Validity)
4. 상관관계 분석 (Correlation Analysis)
5. 경로분석 (Path Analysis)
6. 조절효과 분석 (Moderation Analysis)
7. 다항로짓 분석 (Multinomial Logit)
8. 결과 관리 (Results Management)

Author: Sugar Substitute Research Team
Date: 2025-09-18
Version: 2.0 (Reorganized)
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
sys.path.append('src')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """필요한 디렉토리들 생성"""
    directories = [
        "logs",
        "results/current",
        "results/archive", 
        "data/processed/survey",
        "src/analysis",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_prerequisites():
    """사전 요구사항 확인"""
    print("🔍 사전 요구사항 확인 중...")
    print("-" * 60)
    
    # 필요한 스크립트 확인
    required_scripts = [
        "scripts/run_factor_analysis.py",
        "scripts/run_reliability_analysis.py", 
        "scripts/run_path_analysis.py",
        "scripts/run_complete_analysis.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print("❌ 필요한 스크립트가 없습니다:")
        for script in missing_scripts:
            print(f"  - {script}")
        return False
    
    # 데이터 디렉토리 확인
    data_dirs = [
        "data/processed/survey",
        "processed_data/survey_data"  # Fallback
    ]
    
    data_available = False
    for data_dir in data_dirs:
        if Path(data_dir).exists() and any(Path(data_dir).iterdir()):
            data_available = True
            break
    
    if not data_available:
        print("❌ 분석 데이터를 찾을 수 없습니다.")
        return False
    
    print("✅ 사전 요구사항 확인 완료")
    return True


def run_script_safely(script_path, args=None, description=""):
    """스크립트 안전 실행"""
    try:
        print(f"\n🚀 {description} 실행 중...")
        print("-" * 60)
        
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # 인코딩 오류 처리
            timeout=300  # 5분 타임아웃
        )
        
        if result.returncode == 0:
            print(f"✅ {description} 완료")
            if result.stdout:
                # 출력이 너무 길면 요약만 표시
                lines = result.stdout.split('\n')
                if len(lines) > 20:
                    print("출력 요약:")
                    print('\n'.join(lines[:10]))
                    print(f"... ({len(lines)-20}줄 생략) ...")
                    print('\n'.join(lines[-10:]))
                else:
                    print("출력:")
                    print(result.stdout)
            return True
        else:
            print(f"❌ {description} 실패 (코드: {result.returncode})")
            if result.stderr:
                print("오류:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 타임아웃 (5분 초과)")
        return False
    except Exception as e:
        print(f"❌ {description} 실행 오류: {e}")
        logger.error(f"{description} 실행 오류: {e}")
        return False


def run_factor_analysis():
    """요인분석 실행"""
    return run_script_safely(
        "scripts/run_factor_analysis.py",
        ["--all"],
        "요인분석 (Factor Analysis)"
    )


def run_reliability_analysis():
    """신뢰도 분석 실행"""
    return run_script_safely(
        "scripts/run_reliability_analysis.py",
        None,
        "신뢰도 분석 (Reliability Analysis)"
    )


def run_path_analysis():
    """경로분석 실행"""
    return run_script_safely(
        "scripts/run_path_analysis.py",
        ["--model", "comprehensive"],
        "경로분석 (Path Analysis)"
    )


def run_complete_pipeline():
    """전체 분석 파이프라인 실행"""
    return run_script_safely(
        "scripts/run_complete_analysis.py",
        ["--core-only"],
        "전체 분석 파이프라인"
    )


def show_results_summary():
    """결과 요약 표시"""
    print("\n📊 분석 결과 요약")
    print("=" * 60)
    
    try:
        from src.utils.results_manager import ResultsManager
        manager = ResultsManager()
        summary = manager.get_summary()
        
        print(f"분석 유형 수: {summary['total_analysis_types']}")
        print(f"총 아카이브 버전: {summary['total_archived_versions']}")
        
        if summary['latest_results']:
            print(f"\n📋 최신 결과:")
            for analysis_type, timestamp in summary['latest_results'].items():
                print(f"  🔹 {analysis_type}: {timestamp}")
        
        print(f"\n📁 결과 파일 위치:")
        print(f"  📊 현재 결과: results/current/")
        print(f"  📦 아카이브: results/archive/")
        
    except ImportError:
        print("⚠️ 결과 관리 시스템을 사용할 수 없습니다.")
        
        # 기본 결과 디렉토리 확인
        result_dirs = [
            "results/current/factor_analysis",
            "results/current/reliability_analysis",
            "results/current/path_analysis"
        ]
        
        for result_dir in result_dirs:
            if Path(result_dir).exists():
                file_count = len(list(Path(result_dir).glob("*")))
                print(f"  📁 {result_dir}: {file_count}개 파일")


def interactive_menu():
    """대화형 메뉴"""
    while True:
        print("\n" + "=" * 60)
        print("🎯 Sugar Substitute Research - 분석 메뉴")
        print("=" * 60)
        print("1. 요인분석 (Factor Analysis)")
        print("2. 신뢰도 분석 (Reliability Analysis)")
        print("3. 경로분석 (Path Analysis)")
        print("4. 전체 분석 파이프라인 (Core Analysis)")
        print("5. 결과 관리 (Results Management)")
        print("6. 결과 요약 보기")
        print("0. 종료")
        
        try:
            choice = input("\n선택하세요 (0-6): ").strip()
            
            if choice == "0":
                print("👋 분석 시스템을 종료합니다.")
                break
            elif choice == "1":
                run_factor_analysis()
            elif choice == "2":
                run_reliability_analysis()
            elif choice == "3":
                run_path_analysis()
            elif choice == "4":
                run_complete_pipeline()
            elif choice == "5":
                run_script_safely(
                    "scripts/manage_results.py",
                    ["--status"],
                    "결과 관리"
                )
            elif choice == "6":
                show_results_summary()
            else:
                print("❌ 잘못된 선택입니다. 0-6 사이의 숫자를 입력하세요.")
                
        except KeyboardInterrupt:
            print("\n\n👋 사용자가 종료했습니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='Sugar Substitute Research - 통합 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                          # 대화형 메뉴
  python main.py --factor                 # 요인분석만 실행
  python main.py --reliability            # 신뢰도 분석만 실행
  python main.py --path                   # 경로분석만 실행
  python main.py --all                    # 전체 분석 실행
  python main.py --results                # 결과 요약 보기
        """
    )
    
    parser.add_argument('--factor', action='store_true', help='요인분석 실행')
    parser.add_argument('--reliability', action='store_true', help='신뢰도 분석 실행')
    parser.add_argument('--path', action='store_true', help='경로분석 실행')
    parser.add_argument('--all', action='store_true', help='전체 분석 파이프라인 실행')
    parser.add_argument('--results', action='store_true', help='결과 요약 보기')
    parser.add_argument('--interactive', action='store_true', help='대화형 메뉴 실행')
    
    args = parser.parse_args()
    
    # 헤더 출력
    print("🎯 Sugar Substitute Research - 통합 분석 시스템")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"버전: 2.0 (Reorganized)")
    
    # 디렉토리 생성
    ensure_directories()
    
    # 사전 요구사항 확인
    if not check_prerequisites():
        print("❌ 사전 요구사항을 만족하지 않습니다. 실행을 중단합니다.")
        return
    
    # 명령행 인수에 따른 실행
    if args.factor:
        run_factor_analysis()
    elif args.reliability:
        run_reliability_analysis()
    elif args.path:
        run_path_analysis()
    elif args.all:
        run_complete_pipeline()
    elif args.results:
        show_results_summary()
    elif args.interactive:
        interactive_menu()
    else:
        # 기본값: 대화형 메뉴
        interactive_menu()
    
    print(f"\n🎉 분석 시스템 실행 완료!")
    print(f"📁 결과 확인: results/current/")
    print(f"📋 로그 확인: logs/main_analysis.log")


if __name__ == "__main__":
    main()
