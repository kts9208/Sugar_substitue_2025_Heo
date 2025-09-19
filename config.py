#!/usr/bin/env python3
"""
Sugar Substitute Research - 설정 파일

이 파일은 전체 프로젝트의 설정을 관리합니다.

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

# 데이터 디렉토리 설정
DATA_CONFIG = {
    "raw_data_dir": PROJECT_ROOT / "data" / "raw",
    "processed_data_dir": PROJECT_ROOT / "data" / "processed",
    "survey_data_dir": PROJECT_ROOT / "data" / "processed" / "survey",
    "dce_data_dir": PROJECT_ROOT / "data" / "processed" / "dce",
    "config_dir": PROJECT_ROOT / "data" / "config",
    
    # Fallback paths (기존 구조 호환성)
    "fallback_survey_dir": PROJECT_ROOT / "processed_data" / "survey_data",
    "fallback_config_file": PROJECT_ROOT / "processed_data" / "reverse_items_config.json"
}

# 결과 디렉토리 설정
RESULTS_CONFIG = {
    "current_results_dir": PROJECT_ROOT / "results" / "current",
    "archive_results_dir": PROJECT_ROOT / "results" / "archive",
    "metadata_file": PROJECT_ROOT / "results" / "metadata.json",
    
    # 분석별 결과 디렉토리
    "factor_analysis_dir": PROJECT_ROOT / "results" / "current" / "factor_analysis",
    "path_analysis_dir": PROJECT_ROOT / "results" / "current" / "path_analysis",
    "reliability_analysis_dir": PROJECT_ROOT / "results" / "current" / "reliability_analysis",
    "discriminant_validity_dir": PROJECT_ROOT / "results" / "current" / "discriminant_validity",
    "correlations_dir": PROJECT_ROOT / "results" / "current" / "correlations",
    "moderation_analysis_dir": PROJECT_ROOT / "results" / "current" / "moderation_analysis",
    "multinomial_logit_dir": PROJECT_ROOT / "results" / "current" / "multinomial_logit",
    "utility_function_dir": PROJECT_ROOT / "results" / "current" / "utility_function",
    "visualizations_dir": PROJECT_ROOT / "results" / "current" / "visualizations"
}

# 스크립트 디렉토리 설정
SCRIPTS_CONFIG = {
    "scripts_dir": PROJECT_ROOT / "scripts",
    "main_script": PROJECT_ROOT / "main.py",
    "factor_analysis_script": PROJECT_ROOT / "scripts" / "run_factor_analysis.py",
    "reliability_analysis_script": PROJECT_ROOT / "scripts" / "run_reliability_analysis.py",
    "path_analysis_script": PROJECT_ROOT / "scripts" / "run_path_analysis.py",
    "complete_analysis_script": PROJECT_ROOT / "scripts" / "run_complete_analysis.py",
    "results_management_script": PROJECT_ROOT / "scripts" / "manage_results.py"
}

# 로그 설정
LOGGING_CONFIG = {
    "log_dir": PROJECT_ROOT / "logs",
    "main_log_file": PROJECT_ROOT / "logs" / "main_analysis.log",
    "factor_analysis_log": PROJECT_ROOT / "logs" / "factor_analysis.log",
    "path_analysis_log": PROJECT_ROOT / "logs" / "path_analysis.log",
    "reliability_analysis_log": PROJECT_ROOT / "logs" / "reliability_analysis.log",
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# 분석 설정
ANALYSIS_CONFIG = {
    # 요인분석 설정
    "factor_analysis": {
        "default_factors": [
            "health_concern",
            "perceived_benefit", 
            "purchase_intention",
            "perceived_price",
            "nutrition_knowledge"
        ],
        "min_loading_threshold": 0.5,
        "good_loading_threshold": 0.7,
        "fit_indices_thresholds": {
            "CFI": 0.9,
            "TLI": 0.9,
            "RMSEA": 0.08,
            "SRMR": 0.08
        }
    },
    
    # 신뢰도 분석 설정
    "reliability_analysis": {
        "cronbach_alpha_threshold": 0.7,
        "composite_reliability_threshold": 0.7,
        "ave_threshold": 0.5
    },
    
    # 경로분석 설정
    "path_analysis": {
        "bootstrap_samples": 1000,
        "confidence_level": 0.95,
        "standardized": True,
        "include_indirect_effects": True
    },
    
    # 판별타당도 설정
    "discriminant_validity": {
        "fornell_larcker_criterion": True,
        "htmt_threshold": 0.85
    }
}

# 시각화 설정
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "font_family": "Arial",
    "font_size": 12,
    "color_palette": "viridis",
    "save_formats": ["png", "pdf"],
    "korean_font": "Malgun Gothic"  # Windows 기본 한글 폰트
}

# 파일 명명 규칙
NAMING_CONFIG = {
    "timestamp_format": "%Y%m%d_%H%M%S",
    "date_format": "%Y-%m-%d",
    "file_encoding": "utf-8",
    "csv_separator": ",",
    "decimal_places": 4
}

# 버전 관리 설정
VERSION_CONFIG = {
    "max_archive_versions": 10,
    "auto_archive": True,
    "archive_description_required": False,
    "cleanup_old_versions": True
}

# 환경 변수 설정
ENVIRONMENT_CONFIG = {
    "python_path": [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "src"),
        str(PROJECT_ROOT / "scripts")
    ],
    "required_packages": [
        "pandas",
        "numpy", 
        "scipy",
        "semopy",
        "matplotlib",
        "seaborn",
        "pathlib",
        "json"
    ]
}


def get_data_path(data_type: str = "survey"):
    """데이터 경로 반환 (fallback 포함)"""
    if data_type == "survey":
        primary_path = DATA_CONFIG["survey_data_dir"]
        fallback_path = DATA_CONFIG["fallback_survey_dir"]
    elif data_type == "dce":
        primary_path = DATA_CONFIG["dce_data_dir"]
        fallback_path = None
    else:
        return None
    
    if primary_path.exists():
        return primary_path
    elif fallback_path and fallback_path.exists():
        return fallback_path
    else:
        return None


def get_config_file():
    """설정 파일 경로 반환 (fallback 포함)"""
    primary_path = DATA_CONFIG["config_dir"] / "reverse_items_config.json"
    fallback_path = DATA_CONFIG["fallback_config_file"]
    
    if primary_path.exists():
        return primary_path
    elif fallback_path.exists():
        return fallback_path
    else:
        return None


def ensure_directories():
    """필요한 디렉토리들 생성"""
    directories_to_create = []
    
    # 데이터 디렉토리
    for key, path in DATA_CONFIG.items():
        if key.endswith("_dir"):
            directories_to_create.append(path)
    
    # 결과 디렉토리
    for key, path in RESULTS_CONFIG.items():
        if key.endswith("_dir"):
            directories_to_create.append(path)
    
    # 로그 디렉토리
    directories_to_create.append(LOGGING_CONFIG["log_dir"])
    
    # 디렉토리 생성
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("🔧 Sugar Substitute Research - 설정 정보")
    print("=" * 60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"데이터 디렉토리: {get_data_path('survey')}")
    print(f"결과 디렉토리: {RESULTS_CONFIG['current_results_dir']}")
    print(f"로그 디렉토리: {LOGGING_CONFIG['log_dir']}")
    
    # 디렉토리 생성
    ensure_directories()
    print("\n✅ 필요한 디렉토리들이 생성되었습니다.")
