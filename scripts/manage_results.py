#!/usr/bin/env python3
"""
결과 파일 관리 스크립트

이 스크립트는 분석 결과 파일들을 관리합니다:
1. 현재 결과 조회
2. 버전 히스토리 확인
3. 결과 아카이브
4. 버전 복원
5. 오래된 버전 정리

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')
sys.path.append('..')

try:
    from src.utils.results_manager import ResultsManager
except ImportError:
    # Fallback - 현재 구조에서 실행하는 경우
    print("⚠️ 새로운 구조의 results_manager를 찾을 수 없습니다.")
    print("현재 디렉토리에서 기본 기능만 제공합니다.")
    sys.exit(1)


def show_current_status():
    """현재 결과 상태 표시"""
    print("📊 현재 결과 파일 상태")
    print("=" * 60)
    
    manager = ResultsManager()
    summary = manager.get_summary()
    
    print(f"분석 유형 수: {summary['total_analysis_types']}")
    print(f"총 아카이브 버전: {summary['total_archived_versions']}")
    
    print(f"\n📋 최신 결과:")
    if summary['latest_results']:
        for analysis_type, timestamp in summary['latest_results'].items():
            print(f"  🔹 {analysis_type}: {timestamp}")
    else:
        print("  ❌ 최신 결과가 없습니다.")
    
    print(f"\n📦 버전 수:")
    if summary['version_counts']:
        for analysis_type, count in summary['version_counts'].items():
            print(f"  🔹 {analysis_type}: {count}개 버전")
    else:
        print("  ❌ 아카이브된 버전이 없습니다.")


def show_version_history(analysis_type: str):
    """특정 분석 유형의 버전 히스토리 표시"""
    print(f"📜 {analysis_type} 버전 히스토리")
    print("=" * 60)
    
    manager = ResultsManager()
    versions = manager.list_versions(analysis_type)
    
    if not versions:
        print("❌ 버전 히스토리가 없습니다.")
        return
    
    # 최신순으로 정렬
    versions.sort(key=lambda x: x['timestamp'], reverse=True)
    
    for i, version in enumerate(versions, 1):
        timestamp = version['timestamp']
        description = version.get('description', '설명 없음')
        file_count = version.get('file_count', 0)
        
        print(f"{i}. {timestamp}")
        print(f"   📝 설명: {description}")
        print(f"   📄 파일 수: {file_count}")
        print(f"   📁 경로: {version['archived_path']}")
        print()


def archive_results(analysis_type: str, description: str = ""):
    """결과 아카이브"""
    print(f"📦 {analysis_type} 결과 아카이브 중...")
    print("-" * 50)
    
    manager = ResultsManager()
    archive_path = manager.archive_current_results(analysis_type, description)
    
    if archive_path:
        print(f"✅ 아카이브 완료: {archive_path}")
    else:
        print("❌ 아카이브할 결과가 없거나 실패했습니다.")


def restore_version(analysis_type: str, timestamp: str):
    """특정 버전 복원"""
    print(f"🔄 {analysis_type} 버전 복원 중...")
    print(f"대상 버전: {timestamp}")
    print("-" * 50)
    
    manager = ResultsManager()
    
    # 복원 전 확인
    versions = manager.list_versions(analysis_type)
    target_version = None
    
    for version in versions:
        if version['timestamp'] == timestamp:
            target_version = version
            break
    
    if not target_version:
        print(f"❌ 버전을 찾을 수 없습니다: {timestamp}")
        return
    
    print(f"복원할 버전 정보:")
    print(f"  📅 타임스탬프: {target_version['timestamp']}")
    print(f"  📝 설명: {target_version.get('description', '설명 없음')}")
    print(f"  📄 파일 수: {target_version.get('file_count', 0)}")
    
    # 사용자 확인
    confirm = input("\n복원을 진행하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 복원이 취소되었습니다.")
        return
    
    # 복원 실행
    success = manager.restore_version(analysis_type, timestamp)
    
    if success:
        print("✅ 버전 복원 완료!")
    else:
        print("❌ 버전 복원 실패!")


def cleanup_old_versions(analysis_type: str, keep_count: int = 5):
    """오래된 버전 정리"""
    print(f"🧹 {analysis_type} 오래된 버전 정리 중...")
    print(f"유지할 버전 수: {keep_count}")
    print("-" * 50)
    
    manager = ResultsManager()
    versions = manager.list_versions(analysis_type)
    
    if len(versions) <= keep_count:
        print(f"✅ 정리할 버전이 없습니다. (현재 {len(versions)}개)")
        return
    
    print(f"현재 버전 수: {len(versions)}")
    print(f"제거될 버전 수: {len(versions) - keep_count}")
    
    # 사용자 확인
    confirm = input("\n정리를 진행하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 정리가 취소되었습니다.")
        return
    
    # 정리 실행
    manager.cleanup_old_versions(analysis_type, keep_count)
    print("✅ 오래된 버전 정리 완료!")


def list_analysis_types():
    """사용 가능한 분석 유형 목록"""
    analysis_types = [
        "factor_analysis",
        "path_analysis", 
        "reliability_analysis",
        "discriminant_validity",
        "correlations",
        "moderation_analysis",
        "multinomial_logit",
        "utility_function",
        "visualizations"
    ]
    
    print("📋 사용 가능한 분석 유형:")
    for i, analysis_type in enumerate(analysis_types, 1):
        print(f"  {i}. {analysis_type}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='결과 파일 관리')
    parser.add_argument('--status', action='store_true', help='현재 상태 표시')
    parser.add_argument('--history', type=str, help='특정 분석 유형의 버전 히스토리 표시')
    parser.add_argument('--archive', type=str, help='특정 분석 유형 결과 아카이브')
    parser.add_argument('--description', type=str, default="", help='아카이브 설명')
    parser.add_argument('--restore', nargs=2, metavar=('TYPE', 'TIMESTAMP'), 
                       help='특정 버전 복원 (분석유형 타임스탬프)')
    parser.add_argument('--cleanup', type=str, help='오래된 버전 정리')
    parser.add_argument('--keep', type=int, default=5, help='유지할 버전 수 (기본값: 5)')
    parser.add_argument('--list-types', action='store_true', help='사용 가능한 분석 유형 목록')
    
    args = parser.parse_args()
    
    print("🗂️ 결과 파일 관리 시스템")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.status:
        show_current_status()
        
    elif args.history:
        show_version_history(args.history)
        
    elif args.archive:
        archive_results(args.archive, args.description)
        
    elif args.restore:
        analysis_type, timestamp = args.restore
        restore_version(analysis_type, timestamp)
        
    elif args.cleanup:
        cleanup_old_versions(args.cleanup, args.keep)
        
    elif args.list_types:
        list_analysis_types()
        
    else:
        # 기본값: 현재 상태 표시
        show_current_status()
        
        print(f"\n🎯 사용법:")
        print(f"  python scripts/manage_results.py --status")
        print(f"  python scripts/manage_results.py --history factor_analysis")
        print(f"  python scripts/manage_results.py --archive path_analysis --description '새로운 모델'")
        print(f"  python scripts/manage_results.py --restore factor_analysis 20250918_143022")
        print(f"  python scripts/manage_results.py --cleanup factor_analysis --keep 3")
        print(f"  python scripts/manage_results.py --list-types")


if __name__ == "__main__":
    main()
