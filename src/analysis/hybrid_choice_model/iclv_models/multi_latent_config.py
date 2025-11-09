"""
Multi-Latent Variable ICLV Configuration

다중 잠재변수 ICLV 모델의 설정을 관리하는 모듈입니다.
기존 단일 LV 설정을 확장하여 여러 잠재변수를 지원합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-09
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

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
    
    구조:
    - 외생 잠재변수: LV_i = η_i ~ N(0, 1)
    - 내생 잠재변수: LV_endo = Σ(γ_i * LV_i) + Σ(γ_j * X_j) + η
    """
    
    # 내생 잠재변수 (구매의도)
    endogenous_lv: str
    
    # 외생 잠재변수 (건강관심도, 건강유익성, 가격수준, 영양지식)
    exogenous_lvs: List[str]
    
    # 공변량 (사회인구학적 변수)
    covariates: List[str]
    
    # 오차항 분산
    error_variance: float = 1.0
    fix_error_variance: bool = True
    
    # 초기값
    initial_gamma_lv: Optional[List[float]] = None  # 잠재변수 계수
    initial_gamma_x: Optional[List[float]] = None   # 공변량 계수
    
    @property
    def n_exo(self) -> int:
        """외생 잠재변수 개수"""
        return len(self.exogenous_lvs)

    @property
    def n_cov(self) -> int:
        """공변량 개수"""
        return len(self.covariates)

    def __post_init__(self):
        """검증"""
        if self.endogenous_lv in self.exogenous_lvs:
            raise ValueError(f"내생 LV '{self.endogenous_lv}'가 외생 LV 목록에 포함되어 있습니다.")


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
    **kwargs
) -> MultiLatentConfig:
    """
    설탕 대체재 연구용 다중 LV ICLV 설정 생성
    
    5개 잠재변수:
    1. 건강관심도 (Q6-Q11)
    2. 건강유익성 (Q12-Q17)
    3. 가격수준 (Q27-Q29)
    4. 영양지식 (Q30-Q49)
    5. 구매의도 (Q18-Q20) - 내생변수
    
    Args:
        n_draws: Halton draws 수
        max_iterations: 최대 반복 횟수
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
    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=[
            'health_concern',
            'perceived_benefit',
            'perceived_price',
            'nutrition_knowledge'
        ],
        covariates=['age_std', 'gender', 'income_std', 'education_level'],
        error_variance=1.0,
        fix_error_variance=True
    )
    
    # 3. 선택모델 설정
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='binary',
        price_variable='price'
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

