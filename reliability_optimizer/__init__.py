"""
신뢰도 최적화 모듈

이 패키지는 기존 신뢰도 분석 결과를 바탕으로 AVE 기준을 만족하지 못하는 요인의
문항들을 체계적으로 제거하여 크론바흐 알파, CR, AVE 기준을 모두 만족하는
최적의 문항 조합을 찾는 기능을 제공합니다.

주요 기능:
- 기존 신뢰도 분석 결과 로드
- 문제 요인 자동 식별
- 체계적인 문항 제거를 통한 신뢰도 최적화
- 최적화 결과 보고서 생성

Author: Reliability Optimization System
Date: 2025-01-02
"""

from .item_optimizer import ReliabilityOptimizer

__version__ = "1.0.0"
__author__ = "Reliability Optimization System"

__all__ = [
    'ReliabilityOptimizer'
]
