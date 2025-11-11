# 계층적 구조 및 조절효과 구현 가이드

**작성일**: 2025-11-11  
**목적**: 제안된 ICLV 구조 수정사항 구현 방법

---

## 📋 구현 개요

### 목표 구조
```
1차 LV (외생):
  건강관심도 = η₁ ~ N(0,1)
  가격수준 = η₂ ~ N(0,1)
  영양지식 = η₃ ~ N(0,1)

2차 LV (중간 내생):
  건강유익성 = γ₁·건강관심도 + η₂

3차 LV (최종 내생):
  구매의도 = γ₂·건강유익성 + η₃

선택모델 (조절효과):
  V = intercept + β·X + λ₁·구매의도 + λ₂·(구매의도×가격수준) + λ₃·(구매의도×영양지식)
```

---

## 🔧 Phase 1: 계층적 구조 구현

### Step 1.1: 새로운 Config 클래스 생성

**파일**: `src/analysis/hybrid_choice_model/iclv_models/hierarchical_config.py`

```python
"""
Hierarchical Structural Model Configuration
계층적 구조모델 설정
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class HierarchicalPath:
    """
    계층적 경로 정의
    
    Example:
        # 건강유익성 = γ₁ * 건강관심도 + η
        HierarchicalPath(
            target='perceived_benefit',
            predictors=['health_concern'],
            error_variance=1.0
        )
    """
    target: str  # 목표 잠재변수
    predictors: List[str]  # 예측 잠재변수들
    error_variance: float = 1.0
    fix_error_variance: bool = True


@dataclass
class HierarchicalStructuralConfig:
    """
    계층적 구조모델 설정
    
    구조:
    - 1차 LV (외생): LV_i = η_i ~ N(0, 1)
    - 2차+ LV (내생): LV_j = Σ(γ_k * LV_k) + η_j
    
    Example:
        config = HierarchicalStructuralConfig(
            first_order_lvs=['health_concern', 'perceived_price', 'nutrition_knowledge'],
            hierarchical_paths=[
                HierarchicalPath(
                    target='perceived_benefit',
                    predictors=['health_concern'],
                    error_variance=1.0
                ),
                HierarchicalPath(
                    target='purchase_intention',
                    predictors=['perceived_benefit'],
                    error_variance=1.0
                )
            ]
        )
    """
    
    # 1차 잠재변수 (외생)
    first_order_lvs: List[str]
    
    # 계층적 경로
    hierarchical_paths: List[HierarchicalPath]
    
    # 사회인구학적 변수 (선택사항)
    covariates: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """검증"""
        # 모든 target이 first_order_lvs에 없는지 확인
        for path in self.hierarchical_paths:
            if path.target in self.first_order_lvs:
                raise ValueError(
                    f"Target '{path.target}'는 first_order_lvs에 있을 수 없습니다."
                )
            
            # 모든 predictor가 정의되어 있는지 확인
            all_lvs = self.first_order_lvs + [p.target for p in self.hierarchical_paths]
            for pred in path.predictors:
                if pred not in all_lvs:
                    raise ValueError(
                        f"Predictor '{pred}'가 정의되지 않았습니다."
                    )
    
    def get_all_latent_variables(self) -> List[str]:
        """모든 잠재변수 반환"""
        return self.first_order_lvs + [p.target for p in self.hierarchical_paths]
    
    def get_n_parameters(self) -> int:
        """구조모델 파라미터 수"""
        n_params = 0
        for path in self.hierarchical_paths:
            n_params += len(path.predictors)  # gamma 계수
        return n_params
    
    def get_parameter_names(self) -> List[str]:
        """파라미터 이름 리스트"""
        names = []
        for path in self.hierarchical_paths:
            for pred in path.predictors:
                names.append(f'gamma_{pred}_to_{path.target}')
        return names
```

---

### Step 1.2: 계층적 구조모델 클래스 생성

**파일**: `src/analysis/hybrid_choice_model/iclv_models/hierarchical_structural.py`

```python
"""
Hierarchical Structural Model
계층적 구조모델
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import norm
import logging

from .hierarchical_config import HierarchicalStructuralConfig, HierarchicalPath

logger = logging.getLogger(__name__)


class HierarchicalStructural:
    """
    계층적 구조모델
    
    구조:
    - 1차 LV: LV_i = η_i ~ N(0, 1)
    - 2차+ LV: LV_j = Σ(γ_k * LV_k) + η_j
    
    Example:
        건강관심도 (1차) → 건강유익성 (2차) → 구매의도 (3차)
    """
    
    def __init__(self, config: HierarchicalStructuralConfig):
        self.config = config
        self.first_order_lvs = config.first_order_lvs
        self.hierarchical_paths = config.hierarchical_paths
        self.covariates = config.covariates
        
        self.n_first_order = len(self.first_order_lvs)
        self.n_params = config.get_n_parameters()
        
        logger.info(f"HierarchicalStructural 초기화")
        logger.info(f"  1차 LV ({self.n_first_order}개): {self.first_order_lvs}")
        logger.info(f"  계층 경로 ({len(self.hierarchical_paths)}개)")
        for path in self.hierarchical_paths:
            logger.info(f"    {path.predictors} → {path.target}")
        logger.info(f"  총 파라미터: {self.n_params}개")
    
    def predict(self, data: pd.DataFrame,
                first_order_draws: np.ndarray,
                params: Dict[str, float],
                higher_order_draws: Dict[str, float]) -> Dict[str, float]:
        """
        계층적 잠재변수 예측
        
        Args:
            data: 개인 데이터
            first_order_draws: 1차 LV draws (n_first_order,)
            params: 구조모델 파라미터
                {
                    'gamma_health_concern_to_perceived_benefit': float,
                    'gamma_perceived_benefit_to_purchase_intention': float,
                    ...
                }
            higher_order_draws: 2차+ LV 오차항 draws
                {
                    'perceived_benefit': float,
                    'purchase_intention': float
                }
        
        Returns:
            모든 잠재변수 값
        """
        latent_vars = {}
        
        # 1차 LV (외생)
        for i, lv_name in enumerate(self.first_order_lvs):
            latent_vars[lv_name] = first_order_draws[i]
        
        # 2차+ LV (내생) - 순서대로 계산
        for path in self.hierarchical_paths:
            # 예측값 계산
            lv_mean = 0.0
            for pred in path.predictors:
                param_name = f'gamma_{pred}_to_{path.target}'
                gamma = params[param_name]
                lv_mean += gamma * latent_vars[pred]
            
            # 오차항 추가
            error_draw = higher_order_draws[path.target]
            latent_vars[path.target] = (
                lv_mean + np.sqrt(path.error_variance) * error_draw
            )
        
        return latent_vars
    
    def log_likelihood(self, data: pd.DataFrame,
                      latent_vars: Dict[str, float],
                      first_order_draws: np.ndarray,
                      params: Dict[str, float],
                      higher_order_draws: Dict[str, float]) -> float:
        """
        계층적 구조모델 로그우도
        
        LL = Σ log P(LV_1st) + Σ log P(LV_higher | LV_predictors)
        """
        ll = 0.0
        
        # 1차 LV: N(0, 1)
        for lv_name in self.first_order_lvs:
            ll += norm.logpdf(latent_vars[lv_name], loc=0, scale=1)
        
        # 2차+ LV: N(Σ(γ * LV_pred), σ²)
        for path in self.hierarchical_paths:
            # 평균 계산
            lv_mean = 0.0
            for pred in path.predictors:
                param_name = f'gamma_{pred}_to_{path.target}'
                gamma = params[param_name]
                lv_mean += gamma * latent_vars[pred]
            
            # 로그우도
            ll += norm.logpdf(
                latent_vars[path.target],
                loc=lv_mean,
                scale=np.sqrt(path.error_variance)
            )
        
        return ll
    
    def initialize_parameters(self) -> Dict[str, float]:
        """파라미터 초기화"""
        params = {}
        for path in self.hierarchical_paths:
            for pred in path.predictors:
                param_name = f'gamma_{pred}_to_{path.target}'
                params[param_name] = 0.0
        return params
    
    def get_parameter_names(self) -> List[str]:
        """파라미터 이름 리스트"""
        return self.config.get_parameter_names()
```

---

## 🔧 Phase 2: 조절효과 구현

### Step 2.1: 조절효과 포함 선택모델 클래스

**파일**: `src/analysis/hybrid_choice_model/iclv_models/choice_with_moderation.py`

```python
"""
Choice Model with Moderation Effects
조절효과가 포함된 선택모델
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import norm
import logging

from .iclv_config import ChoiceConfig

logger = logging.getLogger(__name__)


class BinaryProbitChoiceWithModeration:
    """
    조절효과가 포함된 Binary Probit 선택모델
    
    Model:
        V = intercept + β·X + λ_main·LV_main + Σ(λ_mod_i · LV_main · LV_mod_i)
        P(Yes) = Φ(V)
    
    Example:
        V = intercept + β·X + λ₁·PI + λ₂·(PI×PP) + λ₃·(PI×NK)
        
        여기서:
        - PI: 구매의도 (주 잠재변수)
        - PP: 가격수준 (조절변수 1)
        - NK: 영양지식 (조절변수 2)
    """
    
    def __init__(self, config: ChoiceConfig,
                 main_lv: str = 'purchase_intention',
                 moderator_lvs: Optional[List[str]] = None):
        """
        초기화
        
        Args:
            config: 선택모델 설정
            main_lv: 주 잠재변수 이름
            moderator_lvs: 조절변수 잠재변수 이름 리스트
        """
        self.config = config
        self.choice_attributes = config.choice_attributes
        self.price_variable = config.price_variable
        self.main_lv = main_lv
        self.moderator_lvs = moderator_lvs or []
        
        self.n_attributes = len(self.choice_attributes)
        self.n_moderators = len(self.moderator_lvs)
        
        logger.info(f"BinaryProbitChoiceWithModeration 초기화")
        logger.info(f"  선택 속성: {self.choice_attributes}")
        logger.info(f"  주 잠재변수: {self.main_lv}")
        logger.info(f"  조절변수: {self.moderator_lvs}")
    
    def log_likelihood(self, data: pd.DataFrame,
                      latent_vars: Dict[str, float],
                      params: Dict) -> float:
        """
        조절효과 포함 로그우도
        
        Args:
            data: 선택 데이터
            latent_vars: 모든 잠재변수 값
            params: {
                'intercept': float,
                'beta': np.ndarray (n_attributes,),
                'lambda_main': float,
                'lambda_mod': np.ndarray (n_moderators,)
            }
        """
        intercept = params['intercept']
        beta = params['beta']
        lambda_main = params['lambda_main']
        lambda_mod = params.get('lambda_mod', np.zeros(self.n_moderators))
        
        # 선택 속성
        X = data[self.choice_attributes].values
        choice = data['choice'].values
        
        # 주 잠재변수
        lv_main = latent_vars[self.main_lv]
        
        # 효용 계산
        V = intercept + X @ beta + lambda_main * lv_main
        
        # 조절효과 추가
        for i, mod_lv_name in enumerate(self.moderator_lvs):
            lv_mod = latent_vars[mod_lv_name]
            V += lambda_mod[i] * (lv_main * lv_mod)
        
        # 확률 및 로그우도
        prob_yes = norm.cdf(V)
        prob_yes = np.clip(prob_yes, 1e-10, 1 - 1e-10)
        
        ll = np.sum(
            choice * np.log(prob_yes) +
            (1 - choice) * np.log(1 - prob_yes)
        )
        
        return ll
    
    def get_n_parameters(self) -> int:
        """파라미터 수"""
        return 1 + self.n_attributes + 1 + self.n_moderators
    
    def initialize_parameters(self) -> Dict:
        """파라미터 초기화"""
        params = {
            'intercept': 0.0,
            'beta': np.zeros(self.n_attributes),
            'lambda_main': 1.0,
            'lambda_mod': np.zeros(self.n_moderators)
        }
        
        # 가격 변수 음수 초기화
        if self.price_variable in self.choice_attributes:
            price_idx = self.choice_attributes.index(self.price_variable)
            params['beta'][price_idx] = -1.0
        
        return params
```

---

## 📝 Phase 3: 통합 및 테스트

### Step 3.1: 설정 예시

```python
from src.analysis.hybrid_choice_model.iclv_models import (
    MeasurementConfig,
    ChoiceConfig,
    EstimationConfig
)
from src.analysis.hybrid_choice_model.iclv_models.hierarchical_config import (
    HierarchicalStructuralConfig,
    HierarchicalPath
)

# 측정모델 설정 (기존과 동일)
measurement_configs = {
    'health_concern': MeasurementConfig(...),
    'perceived_benefit': MeasurementConfig(...),
    'perceived_price': MeasurementConfig(...),
    'nutrition_knowledge': MeasurementConfig(...),
    'purchase_intention': MeasurementConfig(...)
}

# 계층적 구조모델 설정 (NEW)
structural_config = HierarchicalStructuralConfig(
    first_order_lvs=['health_concern', 'perceived_price', 'nutrition_knowledge'],
    hierarchical_paths=[
        HierarchicalPath(
            target='perceived_benefit',
            predictors=['health_concern'],
            error_variance=1.0
        ),
        HierarchicalPath(
            target='purchase_intention',
            predictors=['perceived_benefit'],
            error_variance=1.0
        )
    ],
    covariates=[]  # 사회인구학적 변수 제거
)

# 선택모델 설정 (기존과 동일)
choice_config = ChoiceConfig(
    choice_attributes=['sugar_free', 'health_label', 'price']
)
```

---

## ✅ 검증 체크리스트

- [ ] HierarchicalStructuralConfig 클래스 생성
- [ ] HierarchicalStructural 클래스 생성
- [ ] BinaryProbitChoiceWithModeration 클래스 생성
- [ ] 시뮬레이션 데이터로 파라미터 복원 테스트
- [ ] 실제 데이터로 수렴성 확인
- [ ] 기존 모델과 적합도 비교 (AIC, BIC)
- [ ] 간접효과 계산 (건강관심도 → 건강유익성 → 구매의도)
- [ ] 조절효과 해석 (Simple Slopes Analysis)

---

## 📊 예상 결과

### 파라미터 추정 결과 예시

```
구조모델:
  γ₁ (건강관심도 → 건강유익성): 0.65 (SE=0.08, p<0.001)
  γ₂ (건강유익성 → 구매의도): 0.72 (SE=0.09, p<0.001)

선택모델:
  λ₁ (구매의도 주효과): 1.23 (SE=0.15, p<0.001)
  λ₂ (구매의도 × 가격수준): -0.34 (SE=0.12, p<0.01)
  λ₃ (구매의도 × 영양지식): 0.28 (SE=0.11, p<0.05)

간접효과:
  건강관심도 → 건강유익성 → 구매의도: 0.47 (0.65 × 0.72)
```

---

## 🎯 다음 단계

1. **이론적 타당성 확인**: 연구 가설과 일치하는지 검토
2. **구현 시작**: Phase 1부터 단계적 구현
3. **테스트**: 시뮬레이션 및 실제 데이터 검증
4. **결과 해석**: 간접효과 및 조절효과 분석

