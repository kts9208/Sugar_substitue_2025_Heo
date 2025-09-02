"""
Factor Analysis 패키지

semopy를 이용한 Factor Loading 분석을 위한 모듈들을 제공합니다.
전처리된 CSV 파일들을 불러와서 확인적 요인분석(CFA)을 수행합니다.

Author: Sugar Substitute Research Team
Date: 2025-08-31
"""

from .data_loader import FactorDataLoader, load_factor_data, get_available_factors
from .factor_analyzer import FactorAnalyzer, SemopyAnalyzer, analyze_factor_loading
from .results_exporter import FactorResultsExporter, export_factor_results
from .config import FactorAnalysisConfig, create_factor_model_spec, create_custom_config
from .visualizer import (
    visualize_factor_analysis,
    create_loading_heatmap,
    create_loading_barplot,
    create_fit_indices_plot,
    create_model_diagram,
    FactorAnalysisVisualizer,
    FactorLoadingPlotter,
    ModelDiagramGenerator,
    FitIndicesVisualizer
)
from .semopy_native_visualizer import (
    SemopyNativeVisualizer,
    SemopyModelExtractor,
    IntegratedSemopyVisualizer,
    create_sem_diagram,
    visualize_with_semopy,
    create_diagrams_for_factors
)
from .reliability_calculator import (
    ReliabilityCalculator,
    calculate_reliability_from_results
)

__version__ = "1.0.0"
__author__ = "Sugar Substitute Research Team"

__all__ = [
    # Data loading
    'FactorDataLoader',
    'load_factor_data',
    'get_available_factors',

    # Factor analysis
    'FactorAnalyzer',
    'SemopyAnalyzer',
    'analyze_factor_loading',

    # Results export
    'FactorResultsExporter',
    'export_factor_results',

    # Configuration
    'FactorAnalysisConfig',
    'create_factor_model_spec',
    'create_custom_config',

    # Visualization
    'visualize_factor_analysis',
    'create_loading_heatmap',
    'create_loading_barplot',
    'create_fit_indices_plot',
    'create_model_diagram',
    'FactorAnalysisVisualizer',
    'FactorLoadingPlotter',
    'ModelDiagramGenerator',
    'FitIndicesVisualizer',

    # semopy Native Visualization
    'SemopyNativeVisualizer',
    'SemopyModelExtractor',
    'IntegratedSemopyVisualizer',
    'create_sem_diagram',
    'visualize_with_semopy',
    'create_diagrams_for_factors',

    # Reliability and Validity
    'ReliabilityCalculator',
    'calculate_reliability_from_results'
]
