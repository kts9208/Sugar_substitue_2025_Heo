"""
Choice Models Module

다양한 선택모델 구현체들을 제공하는 모듈입니다.
팩토리 패턴을 사용하여 확장 가능한 구조로 설계되었습니다.
"""

from .base_choice_model import (
    BaseChoiceModel,
    ChoiceModelType,
    ChoiceModelResults
)

from .choice_model_factory import (
    ChoiceModelFactory,
    create_choice_model,
    get_available_models,
    get_model_info,
    register_choice_model
)

from .multinomial_logit_model import (
    MultinomialLogitModel,
    MNLResults
)

from .random_parameters_logit_model import (
    RandomParametersLogitModel,
    RPLResults
)

from .mixed_logit_model import (
    MixedLogitModel,
    MixedLogitResults
)

from .nested_logit_model import (
    NestedLogitModel,
    NestedLogitResults
)

from .multinomial_probit_model import (
    MultinomialProbitModel,
    ProbitResults
)

__all__ = [
    # 기본 클래스
    "BaseChoiceModel",
    "ChoiceModelType", 
    "ChoiceModelResults",
    
    # 팩토리
    "ChoiceModelFactory",
    "create_choice_model",
    "get_available_models",
    "get_model_info",
    "register_choice_model",
    
    # 구체적인 모델들
    "MultinomialLogitModel",
    "MNLResults",
    "RandomParametersLogitModel", 
    "RPLResults",
    "MixedLogitModel",
    "MixedLogitResults",
    "NestedLogitModel",
    "NestedLogitResults",
    "MultinomialProbitModel",
    "ProbitResults"
]
