"""
Multinomial Logit Model 패키지

DCE(Discrete Choice Experiment) 데이터를 사용한 Multinomial Logit Model 구현
"""

from .data_loader import DCEDataLoader, load_dce_data, get_dce_summary
from .data_preprocessor import DCEDataPreprocessor, preprocess_dce_data, get_preprocessing_summary
from .model_config import ModelConfig, ModelConfigManager, create_default_config, create_custom_config
from .model_estimator import MultinomialLogitEstimator, estimate_multinomial_logit
from .results_analyzer import ResultsAnalyzer, analyze_results, create_quick_report

__version__ = "1.0.0"
__author__ = "DCE Analysis Team"

__all__ = [
    # Data loading
    'DCEDataLoader',
    'load_dce_data',
    'get_dce_summary',
    
    # Data preprocessing
    'DCEDataPreprocessor',
    'preprocess_dce_data',
    'get_preprocessing_summary',
    
    # Model configuration
    'ModelConfig',
    'ModelConfigManager',
    'create_default_config',
    'create_custom_config',
    
    # Model estimation
    'MultinomialLogitEstimator',
    'estimate_multinomial_logit',
    
    # Results analysis
    'ResultsAnalyzer',
    'analyze_results',
    'create_quick_report'
]
