"""
분석 모듈 패키지

이 패키지는 다양한 통계 분석 기능들을 제공합니다:
- 요인분석 (Factor Analysis)
- 경로분석 (Path Analysis)  
- 조절효과 분석 (Moderation Analysis)
- 다항로짓 분석 (Multinomial Logit)
- 효용함수 분석 (Utility Function)

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

# 분석 모듈들 임포트
try:
    from .factor_analysis import *
except ImportError:
    pass

try:
    from .path_analysis import *
except ImportError:
    pass

try:
    from .moderation_analysis import *
except ImportError:
    pass

try:
    from .multinomial_logit import *
except ImportError:
    pass

try:
    from .utility_function import *
except ImportError:
    pass

__all__ = [
    # Factor Analysis
    "analyze_factor_loading",
    "export_factor_results",
    
    # Path Analysis
    "PathAnalyzer",
    "analyze_path_model",
    "create_path_model",
    "export_path_results",
    
    # Moderation Analysis
    "analyze_moderation_effects",
    
    # Multinomial Logit
    "run_mnl_analysis",
    
    # Utility Function
    "analyze_utility_function"
]
