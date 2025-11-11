"""
Multi-Latent Variable ICLV Configuration

다중 잠재변수 ICLV 모델의 설정을 관리하는 모듈입니다.
기존 단일 LV 설정을 확장하여 여러 잠재변수를 지원합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-09
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Import 순환 참조 방지를 위해 TYPE_CHECKING 사용
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .iclv_config import MeasurementConfig, ChoiceConfig, EstimationConfig
else:
    # 런타임에는 직접 import
    try:
        from .iclv_config import MeasurementConfig, ChoiceConfig, EstimationConfig
    except ImportError:
        # Fallback: 상대 경로로 import
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from iclv_config import MeasurementConfig, ChoiceConfig, EstimationConfig


@dataclass
class MultiLatentStructuralConfig:
    """
    다중 잠재변수 구조모델 설정

    ✅ 디폴트: 계층적 구조 (건강관심도 → 건강유익성 → 구매의도)

    구조:
    1. 계층적 구조 (hierarchical_paths 지정 시):
       - 1차 LV (외생): LV_i = η_i ~ N(0, 1)
       - 2차+ LV (내생): LV_j = Σ(γ_k * LV_k) + η

    2. 병렬 구조 (hierarchical_paths=None, 하위 호환):
       - 외생 LV: LV_i = η_i ~ N(0, 1)
       - 내생 LV: LV_endo = Σ(γ_i * LV_i) + Σ(γ_j * X_j) + η

    Example (계층적 구조):
        >>> config = MultiLatentStructuralConfig(
        ...     endogenous_lv='purchase_intention',
        ...     exogenous_lvs=['health_concern', 'perceived_price', 'nutrition_knowledge'],
        ...     covariates=[],
        ...     hierarchical_paths=[
        ...         {'target': 'perceived_benefit', 'predictors': ['health_concern']},
        ...         {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
        ...     ]
        ... )
    """

    # 내생 잠재변수 (구매의도)
    endogenous_lv: str

    # 외생 잠재변수 (1차 LV)
    exogenous_lvs: List[str]

    # 공변량 (사회인구학적 변수) - 디폴트: 빈 리스트
    covariates: List[str] = field(default_factory=list)

    # ✅ 계층적 경로 (디폴트: 건강관심도 → 건강유익성 → 구매의도)
    hierarchical_paths: Optional[List[Dict[str, Any]]] = None

    # 오차항 분산
    error_variance: float = 1.0
    fix_error_variance: bool = True

    # 초기값 (하위 호환성)
    initial_gamma_lv: Optional[List[float]] = None  # 잠재변수 계수 (병렬 구조용)
    initial_gamma_x: Optional[List[float]] = None   # 공변량 계수

    @property
    def n_exo(self) -> int:
        """외생 잠재변수 개수 (1차 LV)"""
        return len(self.exogenous_lvs)

    @property
    def n_cov(self) -> int:
        """공변량 개수"""
        return len(self.covariates)

    @property
    def is_hierarchical(self) -> bool:
        """계층적 구조 여부"""
        return self.hierarchical_paths is not None and len(self.hierarchical_paths) > 0

    def get_all_lvs(self) -> List[str]:
        """모든 잠재변수 이름 (1차 + 계층적)"""
        all_lvs = list(self.exogenous_lvs)

        if self.is_hierarchical:
            # 계층적 경로의 target들 추가
            for path in self.hierarchical_paths:
                target = path['target']
                if target not in all_lvs:
                    all_lvs.append(target)
        else:
            # 병렬 구조: 내생 LV 추가
            if self.endogenous_lv not in all_lvs:
                all_lvs.append(self.endogenous_lv)

        return all_lvs

    def get_first_order_lvs(self) -> List[str]:
        """1차 잠재변수 (외생 LV)"""
        return list(self.exogenous_lvs)

    def get_higher_order_lvs(self) -> List[str]:
        """2차 이상 잠재변수 (계층적 경로의 target들)"""
        if not self.is_hierarchical:
            return [self.endogenous_lv]

        higher_lvs = []
        for path in self.hierarchical_paths:
            target = path['target']
            if target not in higher_lvs:
                higher_lvs.append(target)
        return higher_lvs

    def __post_init__(self):
        """검증"""
        # 1. 내생 LV가 외생 LV에 포함되지 않는지 확인
        if self.endogenous_lv in self.exogenous_lvs:
            raise ValueError(f"내생 LV '{self.endogenous_lv}'가 외생 LV 목록에 포함되어 있습니다.")

        # 2. 계층적 경로 검증
        if self.is_hierarchical:
            self._validate_hierarchical_paths()

    def _validate_hierarchical_paths(self):
        """계층적 경로 검증"""
        if not self.hierarchical_paths:
            return

        all_lvs = set(self.exogenous_lvs)

        for i, path in enumerate(self.hierarchical_paths):
            # 필수 키 확인
            if 'target' not in path:
                raise ValueError(f"경로 {i}: 'target' 키가 없습니다.")
            if 'predictors' not in path:
                raise ValueError(f"경로 {i}: 'predictors' 키가 없습니다.")

            target = path['target']
            predictors = path['predictors']

            # predictors가 리스트인지 확인
            if not isinstance(predictors, list):
                raise ValueError(f"경로 {i}: 'predictors'는 리스트여야 합니다.")

            # 모든 predictor가 이미 정의된 LV인지 확인
            for pred in predictors:
                if pred not in all_lvs:
                    raise ValueError(
                        f"경로 {i}: predictor '{pred}'가 정의되지 않았습니다. "
                        f"사용 가능한 LV: {sorted(all_lvs)}"
                    )

            # target을 정의된 LV에 추가
            all_lvs.add(target)

        # 최종 target이 endogenous_lv인지 확인
        final_targets = [path['target'] for path in self.hierarchical_paths]
        if self.endogenous_lv not in final_targets:
            raise ValueError(
                f"계층적 경로의 최종 target이 '{self.endogenous_lv}'가 아닙니다. "
                f"현재 targets: {final_targets}"
            )


@dataclass
class MultiLatentConfig:
    """
    다중 잠재변수 ICLV 모델 전체 설정
    
    5개 잠재변수:
    1. 건강관심도 (health_concern)
    2. 건강유익성 (perceived_benefit)
    3. 가격수준 (perceived_price)
    4. 영양지식 (nutrition_knowledge)
    5. 구매의도 (purchase_intention) - 내생변수
    """
    
    # 측정모델 설정 (5개)
    measurement_configs: Dict[str, MeasurementConfig]
    
    # 구조모델 설정
    structural: MultiLatentStructuralConfig
    
    # 선택모델 설정
    choice: ChoiceConfig
    
    # 추정 설정
    estimation: EstimationConfig
    
    # 데이터 설정
    individual_id_column: str = 'respondent_id'
    choice_column: str = 'choice'
    
    def __post_init__(self):
        """검증"""
        # 모든 잠재변수가 측정모델에 정의되어 있는지 확인
        all_lvs = self.structural.exogenous_lvs + [self.structural.endogenous_lv]
        
        for lv in all_lvs:
            if lv not in self.measurement_configs:
                raise ValueError(f"잠재변수 '{lv}'의 측정모델 설정이 없습니다.")
        
        # 측정모델 설정의 잠재변수 이름 확인
        for lv_name, config in self.measurement_configs.items():
            if config.latent_variable != lv_name:
                raise ValueError(
                    f"측정모델 설정의 잠재변수 이름 불일치: "
                    f"키='{lv_name}', config.latent_variable='{config.latent_variable}'"
                )
    
    def get_all_latent_variables(self) -> List[str]:
        """모든 잠재변수 이름 반환"""
        return self.structural.exogenous_lvs + [self.structural.endogenous_lv]
    
    def get_n_latent_variables(self) -> int:
        """잠재변수 개수"""
        return len(self.get_all_latent_variables())
    
    def get_n_exogenous(self) -> int:
        """외생 잠재변수 개수"""
        return len(self.structural.exogenous_lvs)


def create_sugar_substitute_multi_lv_config(
    n_draws: int = 100,
    max_iterations: int = 1000,
    use_hierarchical: bool = True,  # ✅ 디폴트: 계층적 구조
    use_moderation: bool = True,    # ✅ 디폴트: 조절효과
    **kwargs
) -> MultiLatentConfig:
    """
    설탕 대체재 연구용 다중 LV ICLV 설정 생성

    ✅ 디폴트 구조:
    - 계층적 구조: 건강관심도 → 건강유익성 → 구매의도
    - 조절효과: 가격수준, 영양지식이 구매의도→선택 관계 조절
    - 사회인구학적 변수: 제거

    5개 잠재변수:
    1. 건강관심도 (Q6-Q11) - 1차 LV
    2. 건강유익성 (Q12-Q17) - 2차 LV
    3. 가격수준 (Q27-Q29) - 1차 LV (조절변수)
    4. 영양지식 (Q30-Q49) - 1차 LV (조절변수)
    5. 구매의도 (Q18-Q20) - 3차 LV (내생변수)

    Args:
        n_draws: Halton draws 수
        max_iterations: 최대 반복 횟수
        use_hierarchical: 계층적 구조 사용 여부 (디폴트: True)
        use_moderation: 조절효과 사용 여부 (디폴트: True)
        **kwargs: 추가 설정

    Returns:
        MultiLatentConfig 객체
    """

    # 1. 측정모델 설정 (5개)
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            n_categories=5,
            indicator_type='ordered'
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            n_categories=5,
            indicator_type='ordered'
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],
            n_categories=5,
            indicator_type='ordered'
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=[f'q{i}' for i in range(30, 50)],
            n_categories=5,
            indicator_type='ordered'
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            n_categories=5,
            indicator_type='ordered'
        )
    }

    # 2. 구조모델 설정
    if use_hierarchical:
        # ✅ 디폴트: 계층적 구조
        structural_config = MultiLatentStructuralConfig(
            endogenous_lv='purchase_intention',
            exogenous_lvs=[
                'health_concern',      # 1차 LV
                'perceived_price',     # 1차 LV (조절변수)
                'nutrition_knowledge'  # 1차 LV (조절변수)
            ],
            covariates=[],  # 사회인구학적 변수 제거
            hierarchical_paths=[
                {
                    'target': 'perceived_benefit',
                    'predictors': ['health_concern']
                },
                {
                    'target': 'purchase_intention',
                    'predictors': ['perceived_benefit']
                }
            ],
            error_variance=1.0,
            fix_error_variance=True
        )
    else:
        # 하위 호환: 병렬 구조
        structural_config = MultiLatentStructuralConfig(
            endogenous_lv='purchase_intention',
            exogenous_lvs=[
                'health_concern',
                'perceived_benefit',
                'perceived_price',
                'nutrition_knowledge'
            ],
            covariates=kwargs.get('covariates', ['age_std', 'gender', 'income_std']),
            error_variance=1.0,
            fix_error_variance=True
        )

    # 3. 선택모델 설정
    if use_moderation:
        # ✅ 디폴트: 조절효과
        choice_config = ChoiceConfig(
            choice_attributes=['sugar_free', 'health_label', 'price'],
            choice_type='binary',
            price_variable='price',
            moderation_enabled=True,
            moderator_lvs=['perceived_price', 'nutrition_knowledge'],
            main_lv='purchase_intention'
        )
    else:
        # 하위 호환: 기본 선택모델
        choice_config = ChoiceConfig(
            choice_attributes=['sugar_free', 'health_label', 'price'],
            choice_type='binary',
            price_variable='price',
            moderation_enabled=False
        )

    # 4. 추정 설정
    estimation_config = EstimationConfig(
        optimizer='BFGS',
        use_analytic_gradient=True,
        n_draws=n_draws,
        draw_type='halton',
        max_iterations=max_iterations,
        convergence_tolerance=1e-6,
        use_parallel=kwargs.get('use_parallel', False),
        n_cores=kwargs.get('n_cores', None)
    )

    # 5. 전체 설정
    config = MultiLatentConfig(
        measurement_configs=measurement_configs,
        structural=structural_config,
        choice=choice_config,
        estimation=estimation_config,
        individual_id_column='respondent_id',
        choice_column='choice'
    )

    return config


# 편의 함수
def create_default_multi_lv_config(**kwargs) -> MultiLatentConfig:
    """기본 다중 LV 설정 생성 (설탕 대체재 연구용)"""
    return create_sugar_substitute_multi_lv_config(**kwargs)

