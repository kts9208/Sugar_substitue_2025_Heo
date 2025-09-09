"""
Moderation Analysis Module using semopy

이 모듈은 semopy를 사용하여 조절효과 분석(Moderation Analysis)을 수행하는 기능을 제공합니다.
상호작용항을 통한 조절효과 검정, 단순기울기 분석, 조건부 효과 분석 등을 포함합니다.

주요 기능:
1. 상호작용항 생성 및 모델 구축
2. 조절효과 유의성 검정
3. 단순기울기 분석 (Simple Slopes Analysis)
4. 조건부 효과 계산 (Conditional Effects)
5. 조절효과 시각화
6. 결과 저장 및 보고서 생성

Author: Sugar Substitute Research Team
Date: 2025-09-09
"""

from .data_loader import (
    ModerationDataLoader,
    load_moderation_data,
    get_available_factors,
    combine_factor_data
)
from .interaction_builder import (
    InteractionBuilder,
    create_interaction_terms,
    build_moderation_model,
    create_interaction_model_spec
)
from .moderation_analyzer import (
    ModerationAnalyzer,
    analyze_moderation_effects,
    calculate_simple_slopes,
    calculate_conditional_effects,
    test_moderation_significance
)
from .results_exporter import (
    ModerationResultsExporter,
    export_moderation_results,
    create_moderation_report
)
from .visualizer import (
    ModerationVisualizer,
    create_moderation_plot,
    create_simple_slopes_plot,
    create_interaction_heatmap,
    visualize_moderation_analysis
)
from .config import (
    ModerationAnalysisConfig,
    create_default_moderation_config,
    create_custom_moderation_config,
    get_factor_items_mapping,
    get_factor_descriptions
)

__version__ = "1.0.0"
__author__ = "Sugar Substitute Research Team"
__all__ = [
    # Data loading
    'ModerationDataLoader',
    'load_moderation_data',
    'get_available_factors',
    'combine_factor_data',
    
    # Interaction building
    'InteractionBuilder',
    'create_interaction_terms',
    'build_moderation_model',
    'create_interaction_model_spec',
    
    # Core analysis
    'ModerationAnalyzer',
    'analyze_moderation_effects',
    'calculate_simple_slopes',
    'calculate_conditional_effects',
    'test_moderation_significance',
    
    # Results export
    'ModerationResultsExporter',
    'export_moderation_results',
    'create_moderation_report',
    
    # Visualization
    'ModerationVisualizer',
    'create_moderation_plot',
    'create_simple_slopes_plot',
    'create_interaction_heatmap',
    'visualize_moderation_analysis',
    
    # Configuration
    'ModerationAnalysisConfig',
    'create_default_moderation_config',
    'create_custom_moderation_config',
    'get_factor_items_mapping',
    'get_factor_descriptions'
]

# 모듈 정보
__doc__ = """
Moderation Analysis Module

이 모듈은 설탕 대체재 연구를 위한 조절효과 분석 기능을 제공합니다.
semopy를 기반으로 하여 상호작용항을 통한 조절효과를 분석하고,
단순기울기 분석, 조건부 효과 계산 등의 고급 기능을 포함합니다.

사용 예제:
    from moderation_analysis import analyze_moderation_effects
    
    # 기본 조절효과 분석
    results = analyze_moderation_effects(
        independent_var='health_concern',
        dependent_var='purchase_intention',
        moderator_var='nutrition_knowledge'
    )
    
    # 결과 저장
    from moderation_analysis import export_moderation_results
    export_moderation_results(results)
"""
