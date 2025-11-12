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
    선택모델 설정

    ✅ 디폴트: 조절효과 활성화

    조절효과 모드:
    - V = intercept + β·X + λ_main·LV_main + Σ(λ_mod_i · LV_main · LV_mod_i)

    기본 모드 (moderation_enabled=False):
    - V = intercept + β·X + λ·LV

    Example (조절효과):
        >>> config = ChoiceConfig(
        ...     choice_attributes=['sugar_free', 'health_label', 'price'],
        ...     moderation_enabled=True,
        ...     moderator_lvs=['perceived_price', 'nutrition_knowledge'],
        ...     main_lv='purchase_intention'
        ... )
    """

    # 선택 속성
    choice_attributes: List[str]

    # 선택 유형
    choice_type: Literal['binary', 'multinomial', 'ordered'] = 'binary'

    # 가격 변수 (WTP 계산용)
    price_variable: str = 'price'

    # ✅ 조절효과 설정 (디폴트: 활성화)
    moderation_enabled: bool = True
    moderator_lvs: Optional[List[str]] = field(default_factory=lambda: ['perceived_price', 'nutrition_knowledge'])
    main_lv: str = 'purchase_intention'

    # 초기값
    initial_betas: Optional[Dict[str, float]] = None
    initial_lambda: float = 1.0  # 잠재변수 계수 (하위 호환성)
    initial_lambda_main: float = 1.0  # 주효과 계수
    initial_lambda_mod: Optional[List[float]] = None  # 조절효과 계수

    # Ordered Probit 설정 (binary choice의 경우)
    thresholds: Optional[List[float]] = None

    @property
    def n_moderators(self) -> int:
        """조절변수 개수"""
        if not self.moderation_enabled or self.moderator_lvs is None:
            return 0
        return len(self.moderator_lvs)


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

