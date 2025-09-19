"""
Configuration settings for utility function module.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
UTILITY_MODULE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR = UTILITY_MODULE_DIR / "outputs"

# Data paths
DCE_DATA_DIR = PROCESSED_DATA_DIR / "dce_data"
SURVEY_DATA_DIR = PROCESSED_DATA_DIR / "survey_data"
PATH_ANALYSIS_DIR = BASE_DIR / "path_analysis_results"

# DCE data files
DCE_RAW_DATA_FILE = DCE_DATA_DIR / "dce_raw_data.csv"
DCE_ATTRIBUTE_DATA_FILE = DCE_DATA_DIR / "dce_attribute_data.csv"
DCE_CHOICE_MATRIX_FILE = DCE_DATA_DIR / "dce_choice_matrix.csv"

# Survey data files
FACTORS_SUMMARY_FILE = SURVEY_DATA_DIR / "factors_summary.csv"
HEALTH_CONCERN_FILE = SURVEY_DATA_DIR / "health_concern.csv"
PERCEIVED_BENEFIT_FILE = SURVEY_DATA_DIR / "perceived_benefit.csv"
NUTRITION_KNOWLEDGE_FILE = SURVEY_DATA_DIR / "nutrition_knowledge.csv"
PERCEIVED_PRICE_FILE = SURVEY_DATA_DIR / "perceived_price.csv"
PURCHASE_INTENTION_FILE = SURVEY_DATA_DIR / "purchase_intention.csv"

# SEM results files (latest)
SEM_STRUCTURAL_PATHS_FILE = PATH_ANALYSIS_DIR / "comprehensive_structural_structural_paths_20250910_084833.csv"
SEM_PATH_ANALYSIS_FILE = PATH_ANALYSIS_DIR / "comprehensive_structural_path_analysis_20250910_084833.csv"
SEM_FIT_INDICES_FILE = PATH_ANALYSIS_DIR / "comprehensive_structural_fit_indices_20250910_084833.csv"

# Output settings
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Utility function parameters
UTILITY_COMPONENTS = {
    'dce_variables': ['sugar_presence', 'health_label', 'price'],
    'sem_factors': ['health_concern', 'perceived_benefit', 'nutrition_knowledge', 'perceived_price'],
    'error_term': 'random_utility'
}

# Calculation settings
DEFAULT_RANDOM_SEED = 42
N_BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
