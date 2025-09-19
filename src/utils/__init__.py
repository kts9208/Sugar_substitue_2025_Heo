"""
유틸리티 모듈 패키지

이 패키지는 분석에 필요한 유틸리티 기능들을 제공합니다:
- 결과 파일 관리
- 데이터 전처리
- 설정 관리

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

# 유틸리티 모듈들 임포트
try:
    from .results_manager import *
except ImportError:
    pass

__all__ = [
    # Results Management
    "ResultsManager",
    "save_results",
    "archive_previous_results",
    "get_latest_results",
    "list_all_versions"
]
