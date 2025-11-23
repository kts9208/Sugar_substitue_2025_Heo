"""
ICLV Model Configuration

ICLV 모델의 설정을 관리하는 모듈입니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from pathlib import Path


@dataclass
class MeasurementConfig:
    """측정모델 설정"""

    # 잠재변수
    latent_variable: str

    # 관측지표
    indicators: List[str]

    # 측정 방법
    measurement_method: Literal['ordered_probit', 'continuous_linear'] = 'continuous_linear'

    # 지표 유형 (하위 호환성 유지)
    indicator_type: Literal['ordered', 'continuous', 'binary'] = 'continuous'

    # Ordered Probit 설정
    n_categories: int = 5  # 리커트 척도 범주 수

    # Continuous Linear 설정
    fix_first_loading: bool = True      # 첫 번째 적재량 고정 (식별)
    fix_error_variance: bool = False    # 오차분산 고정 여부
    initial_error_variance: float = 1.0 # 초기 오차분산

    # 초기값
    initial_loadings: Optional[Dict[str, float]] = None
    initial_thresholds: Optional[List[float]] = None  # Ordered Probit용

    # 제약조건 (하위 호환성)
    fix_lv_variance: bool = True    # 잠재변수 분산 고정


@dataclass
class StructuralConfig:
    """구조모델 설정"""
    
    # 사회인구학적 변수
    sociodemographics: List[str]
    
    # 선택모델에도 포함 여부
    include_in_choice: bool = True
    
    # 초기값
    initial_gammas: Optional[Dict[str, float]] = None
    
    # 오차항 분산
    error_variance: float = 1.0
    fix_error_variance: bool = True


@dataclass
class ChoiceConfig:
    """
    선택모델 설정 (유연한 리스트 기반 시스템)

    ✅ 핵심 원칙: 플래그 없이 리스트만으로 모든 모델 표현

    Base Model (잠재변수 없음):
    - V = intercept + β·X
    - main_lvs = []
    - lv_attribute_interactions = []

    주효과 모델:
    - V = intercept + β·X + Σ(λ_i · LV_i)
    - main_lvs = ['purchase_intention', 'nutrition_knowledge']
    - lv_attribute_interactions = []

    LV-Attribute 상호작용 모델:
    - V = intercept + β·X + Σ(λ_i · LV_i) + Σ(γ_ij · LV_i · X_j)
    - main_lvs = ['purchase_intention', 'nutrition_knowledge']
    - lv_attribute_interactions = [{'lv': 'purchase_intention', 'attribute': 'price'}, ...]

    Example (Base Model):
        >>> config = ChoiceConfig(
        ...     choice_attributes=['sugar_free', 'health_label', 'price'],
        ...     main_lvs=[],  # 빈 리스트 = 잠재변수 없음
        ...     lv_attribute_interactions=[]
        ... )

    Example (주효과 모델):
        >>> config = ChoiceConfig(
        ...     choice_attributes=['sugar_free', 'health_label', 'price'],
        ...     main_lvs=['purchase_intention', 'nutrition_knowledge']
        ... )

    Example (LV-Attribute 상호작용):
        >>> config = ChoiceConfig(
        ...     choice_attributes=['sugar_free', 'health_label', 'price'],
        ...     main_lvs=['purchase_intention', 'nutrition_knowledge'],
        ...     lv_attribute_interactions=[
        ...         {'lv': 'purchase_intention', 'attribute': 'price'},
        ...         {'lv': 'purchase_intention', 'attribute': 'health_label'},
        ...         {'lv': 'nutrition_knowledge', 'attribute': 'health_label'}
        ...     ]
        ... )
    """

    # 선택 속성
    choice_attributes: List[str]

    # 선택 유형
    choice_type: Literal['binary', 'multinomial', 'ordered'] = 'binary'

    # ✅ 대안 개수 (Multinomial Logit용)
    # - binary: 2개 (선택/비선택)
    # - multinomial: 3개 이상 (일반당, 무설탕, opt-out 등)
    n_alternatives: Optional[int] = None

    # 가격 변수 (WTP 계산용)
    price_variable: str = 'price'

    # ✅ 유연한 리스트 기반 설정
    # 빈 리스트 = Base Model (잠재변수 없음)
    # 1개 이상 = 해당 잠재변수들의 주효과
    main_lvs: List[str] = field(default_factory=list)

    # ✅ LV-Attribute 상호작용 설정
    # 각 항목: {'lv': 잠재변수명, 'attribute': 속성명}
    # 빈 리스트 = 상호작용 없음
    lv_attribute_interactions: List[Dict[str, str]] = field(default_factory=list)

    # 초기값
    initial_betas: Optional[Dict[str, float]] = None
    initial_lambdas: Optional[Dict[str, float]] = None  # 각 LV별 초기값
    initial_lv_attr_interactions: Optional[Dict[str, float]] = None  # LV-Attribute 상호작용 초기값

    # Ordered Probit 설정 (binary choice의 경우)
    thresholds: Optional[List[float]] = None

    @property
    def n_main_lvs(self) -> int:
        """주효과 잠재변수 개수"""
        return len(self.main_lvs)

    @property
    def n_lv_attr_interactions(self) -> int:
        """LV-Attribute 상호작용 개수"""
        return len(self.lv_attribute_interactions)


# ================================================================================
# Optimizer 분류
# ================================================================================

# Gradient-based optimizers (gradient 필요)
GRADIENT_BASED_OPTIMIZERS = [
    # Quasi-Newton methods
    'BFGS', 'L-BFGS-B',

    # Newton methods
    'Newton-CG', 'CG',

    # Trust Region methods
    'trust-constr', 'trust-ncg', 'trust-exact', 'trust-krylov', 'dogleg',

    # Sequential Quadratic Programming
    'SLSQP',

    # Custom methods
    'BHHH'  # Berndt-Hall-Hall-Hausman (Newton-CG with OPG Hessian)
]

# Gradient-free optimizers (gradient 불필요)
GRADIENT_FREE_OPTIMIZERS = [
    'Nelder-Mead',  # Simplex method
    'Powell',       # Powell's method
    'COBYLA'        # Constrained Optimization BY Linear Approximation
]


@dataclass
class EstimationConfig:
    """추정 설정"""

    # 추정 방법
    method: Literal['simultaneous', 'sequential'] = 'simultaneous'
    
    # 시뮬레이션 설정
    n_draws: int = 1000
    draw_type: Literal['halton', 'random', 'mlhs'] = 'halton'
    scramble_halton: bool = True
    
    # 최적화 설정
    optimizer: str = 'BFGS'
    max_iterations: int = 2000
    convergence_tolerance: float = 1e-6

    # Gradient 설정 (Apollo 방식)
    use_analytic_gradient: bool = True  # True: analytic gradient, False: numerical gradient

    # Parameter Scaling 설정
    use_parameter_scaling: bool = True  # True: parameter scaling 활성화, False: 비활성화

    # Data Standardization 설정
    standardize_choice_attributes: bool = True  # True: 선택 속성 z-score 표준화, False: 원본 사용
    """
    선택 속성(price, health_label 등)을 z-score 표준화하여 그래디언트 균형 개선
    - True: z = (x - mean) / std (권장)
    - False: 원본 데이터 사용
    """

    # Gradient 로깅 설정
    gradient_log_level: Literal['MINIMAL', 'MODERATE', 'DETAILED'] = 'DETAILED'
    # - MINIMAL: 최종 그래디언트 norm만 로깅
    # - MODERATE: 개인별 그래디언트 요약 로깅
    # - DETAILED: 모든 중간 계산 과정 포함 (디폴트)

    # 병렬처리 설정
    use_parallel: bool = False  # 개인별 우도 계산 병렬화
    n_cores: Optional[int] = None  # None이면 CPU 코어 수 자동 감지

    # 표준오차 계산
    calculate_se: bool = True
    se_method: Literal['hessian', 'bootstrap', 'robust'] = 'hessian'

    # 부트스트랩 설정 (se_method='bootstrap'인 경우)
    n_bootstrap: int = 500

    # 조기 종료 설정
    early_stopping: bool = False  # 조기 종료 활성화 여부
    early_stopping_patience: int = 5  # 개선 없는 연속 횟수
    early_stopping_tol: float = 1e-6  # LL 변화 허용 오차

    def __post_init__(self):
        """Optimizer 검증"""
        all_optimizers = GRADIENT_BASED_OPTIMIZERS + GRADIENT_FREE_OPTIMIZERS
        if self.optimizer not in all_optimizers:
            import warnings
            warnings.warn(
                f"Unknown optimizer: '{self.optimizer}'. "
                f"Supported optimizers: {all_optimizers}. "
                f"Assuming gradient-based optimizer."
            )

    def is_gradient_based(self) -> bool:
        """Gradient-based optimizer 여부 확인"""
        return self.optimizer not in GRADIENT_FREE_OPTIMIZERS


@dataclass
class ICLVConfig:
    """ICLV 모델 전체 설정"""
    
    # 하위 설정
    measurement: MeasurementConfig
    structural: StructuralConfig
    choice: ChoiceConfig
    estimation: EstimationConfig
    
    # 데이터 설정
    individual_id_column: str = 'individual_id'
    choice_column: str = 'choice'
    alternative_column: Optional[str] = None
    
    # 결과 저장
    save_results: bool = True
    output_dir: Optional[Path] = None
    
    # 로깅
    log_file: Optional[Path] = None
    log_level: str = 'INFO'
    
    # 검증
    validate_data: bool = True
    
    def __post_init__(self):
        """설정 검증"""
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.log_file is not None:
            self.log_file = Path(self.log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


def create_iclv_config(
    latent_variable: str,
    indicators: List[str],
    sociodemographics: List[str],
    choice_attributes: List[str],
    price_variable: str = 'price',
    **kwargs
) -> ICLVConfig:
    """
    ICLV 설정 생성 헬퍼 함수
    
    Args:
        latent_variable: 잠재변수 이름
        indicators: 관측지표 리스트
        sociodemographics: 사회인구학적 변수 리스트
        choice_attributes: 선택 속성 리스트
        price_variable: 가격 변수 이름
        **kwargs: 추가 설정
    
    Returns:
        ICLVConfig 객체
    
    Example:
        >>> config = create_iclv_config(
        ...     latent_variable='health_concern',
        ...     indicators=['hc_1', 'hc_2', 'hc_3'],
        ...     sociodemographics=['age', 'gender', 'income'],
        ...     choice_attributes=['price', 'sugar_content'],
        ...     n_draws=1000
        ... )
    """
    
    # 측정모델 설정
    measurement = MeasurementConfig(
        latent_variable=latent_variable,
        indicators=indicators,
        indicator_type=kwargs.get('indicator_type', 'ordered'),
        n_categories=kwargs.get('n_categories', 5)
    )
    
    # 구조모델 설정
    structural = StructuralConfig(
        sociodemographics=sociodemographics,
        include_in_choice=kwargs.get('include_in_choice', True)
    )
    
    # 선택모델 설정
    choice = ChoiceConfig(
        choice_attributes=choice_attributes,
        choice_type=kwargs.get('choice_type', 'binary'),
        price_variable=price_variable
    )
    
    # 추정 설정
    estimation = EstimationConfig(
        method=kwargs.get('estimation_method', 'simultaneous'),
        n_draws=kwargs.get('n_draws', 1000),
        draw_type=kwargs.get('draw_type', 'halton'),
        optimizer=kwargs.get('optimizer', 'BFGS'),
        max_iterations=kwargs.get('max_iterations', 2000)
    )
    
    # 전체 설정
    config = ICLVConfig(
        measurement=measurement,
        structural=structural,
        choice=choice,
        estimation=estimation,
        individual_id_column=kwargs.get('individual_id_column', 'individual_id'),
        choice_column=kwargs.get('choice_column', 'choice'),
        save_results=kwargs.get('save_results', True),
        output_dir=kwargs.get('output_dir'),
        log_file=kwargs.get('log_file')
    )
    
    return config


def create_king2022_config(
    latent_variable: str = 'risk_perception',
    indicators: List[str] = None,
    **kwargs
) -> ICLVConfig:
    """
    King (2022) 논문 스타일의 ICLV 설정 생성
    
    마이크로플라스틱 위험인식 연구 설정을 재현합니다.
    
    Args:
        latent_variable: 잠재변수 (기본: 'risk_perception')
        indicators: 관측지표 (기본: Q13, Q14, Q15)
        **kwargs: 추가 설정
    
    Returns:
        ICLVConfig 객체
    """
    
    if indicators is None:
        indicators = [
            'Q13_current_threat',
            'Q14_future_threat',
            'Q15_environment_threat'
        ]
    
    sociodemographics = kwargs.get('sociodemographics', [
        'age', 'gender', 'distance', 'income',
        'experts', 'bp', 'charity', 'certainty', 'consequentiality'
    ])
    
    choice_attributes = kwargs.get('choice_attributes', ['bid'])
    
    config = create_iclv_config(
        latent_variable=latent_variable,
        indicators=indicators,
        sociodemographics=sociodemographics,
        choice_attributes=choice_attributes,
        price_variable='bid',
        indicator_type='ordered',
        n_categories=5,
        choice_type='binary',
        estimation_method='simultaneous',
        n_draws=1000,
        draw_type='halton',
        include_in_choice=True,  # 사회인구학적 변수를 양쪽에 포함
        **kwargs
    )
    
    return config


def create_sugar_substitute_config(**kwargs) -> ICLVConfig:
    """
    설탕 대체재 연구용 ICLV 설정 생성
    
    현재 프로젝트에 맞춘 기본 설정입니다.
    
    Returns:
        ICLVConfig 객체
    """
    
    config = create_iclv_config(
        latent_variable=kwargs.get('latent_variable', 'health_concern'),
        indicators=kwargs.get('indicators', [
            'health_concern_1', 'health_concern_2', 'health_concern_3',
            'health_concern_4', 'health_concern_5', 'health_concern_6',
            'health_concern_7'
        ]),
        sociodemographics=kwargs.get('sociodemographics', [
            'age', 'gender', 'income', 'education'
        ]),
        choice_attributes=kwargs.get('choice_attributes', [
            'price', 'sugar_content', 'health_label', 'brand'
        ]),
        price_variable='price',
        indicator_type='ordered',
        n_categories=7,  # 7점 척도
        choice_type='multinomial',
        estimation_method='simultaneous',
        n_draws=1000,
        **kwargs
    )
    
    return config

