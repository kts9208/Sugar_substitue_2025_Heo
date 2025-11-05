"""
Data Integration Module

DCE 데이터와 SEM 데이터를 통합하는 모듈입니다.
하이브리드 선택 모델을 위한 데이터 준비 및 검증 기능을 제공합니다.
"""

# 사회인구학적 데이터 로더 (새로 추가)
from .sociodemographic_loader import (
    SociodemographicLoader,
    load_sociodemographic_data
)

# 기존 모듈들 (선택적 임포트)
try:
    from .hybrid_data_integrator import (
        HybridDataIntegrator,
        integrate_dce_sem_data,
        validate_hybrid_data
    )
    HYBRID_INTEGRATOR_AVAILABLE = True
except ImportError:
    HYBRID_INTEGRATOR_AVAILABLE = False

try:
    from .dce_data_processor import (
        DCEDataProcessor,
        process_dce_data,
        validate_dce_structure
    )
    DCE_PROCESSOR_AVAILABLE = True
except ImportError:
    DCE_PROCESSOR_AVAILABLE = False

try:
    from .sem_data_processor import (
        SEMDataProcessor,
        process_sem_data,
        validate_sem_structure
    )
    SEM_PROCESSOR_AVAILABLE = True
except ImportError:
    SEM_PROCESSOR_AVAILABLE = False

try:
    from .data_validator import (
        DataValidator,
        ValidationResult,
        validate_data_compatibility,
        check_data_quality
    )
    DATA_VALIDATOR_AVAILABLE = True
except ImportError:
    DATA_VALIDATOR_AVAILABLE = False

try:
    from .data_merger import (
        DataMerger,
        merge_dce_sem_data,
        align_individual_ids
    )
    DATA_MERGER_AVAILABLE = True
except ImportError:
    DATA_MERGER_AVAILABLE = False

__all__ = [
    # 사회인구학적 데이터 로더 (새로 추가)
    "SociodemographicLoader",
    "load_sociodemographic_data",
]

# 기존 모듈들을 조건부로 추가
if HYBRID_INTEGRATOR_AVAILABLE:
    __all__.extend([
        "HybridDataIntegrator",
        "integrate_dce_sem_data",
        "validate_hybrid_data",
    ])

if DCE_PROCESSOR_AVAILABLE:
    __all__.extend([
        "DCEDataProcessor",
        "process_dce_data",
        "validate_dce_structure",
    ])

if SEM_PROCESSOR_AVAILABLE:
    __all__.extend([
        "SEMDataProcessor",
        "process_sem_data",
        "validate_sem_structure",
    ])

if DATA_VALIDATOR_AVAILABLE:
    __all__.extend([
        "DataValidator",
        "ValidationResult",
        "validate_data_compatibility",
        "check_data_quality",
    ])

if DATA_MERGER_AVAILABLE:
    __all__.extend([
        "DataMerger",
        "merge_dce_sem_data",
        "align_individual_ids"
    ])
