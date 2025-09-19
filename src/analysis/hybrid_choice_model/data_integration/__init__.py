"""
Data Integration Module

DCE 데이터와 SEM 데이터를 통합하는 모듈입니다.
하이브리드 선택 모델을 위한 데이터 준비 및 검증 기능을 제공합니다.
"""

from .hybrid_data_integrator import (
    HybridDataIntegrator,
    integrate_dce_sem_data,
    validate_hybrid_data
)

from .dce_data_processor import (
    DCEDataProcessor,
    process_dce_data,
    validate_dce_structure
)

from .sem_data_processor import (
    SEMDataProcessor,
    process_sem_data,
    validate_sem_structure
)

from .data_validator import (
    DataValidator,
    ValidationResult,
    validate_data_compatibility,
    check_data_quality
)

from .data_merger import (
    DataMerger,
    merge_dce_sem_data,
    align_individual_ids
)

__all__ = [
    # 메인 통합기
    "HybridDataIntegrator",
    "integrate_dce_sem_data",
    "validate_hybrid_data",
    
    # 데이터 처리기
    "DCEDataProcessor",
    "process_dce_data",
    "validate_dce_structure",
    "SEMDataProcessor", 
    "process_sem_data",
    "validate_sem_structure",
    
    # 검증기
    "DataValidator",
    "ValidationResult",
    "validate_data_compatibility",
    "check_data_quality",
    
    # 병합기
    "DataMerger",
    "merge_dce_sem_data",
    "align_individual_ids"
]
