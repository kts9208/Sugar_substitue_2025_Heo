"""
Choice Model Factory

선택모델 생성을 위한 팩토리 패턴 구현입니다.
다양한 선택모델을 동적으로 생성하고 관리할 수 있습니다.
"""

import logging
from typing import Dict, Type, List, Any, Optional
from .base_choice_model import BaseChoiceModel, ChoiceModelType

logger = logging.getLogger(__name__)


class ChoiceModelFactory:
    """선택모델 팩토리 클래스"""
    
    _models: Dict[ChoiceModelType, Type[BaseChoiceModel]] = {}
    _model_info: Dict[ChoiceModelType, Dict[str, Any]] = {}
    
    @classmethod
    def register_model(cls, model_type: ChoiceModelType, model_class: Type[BaseChoiceModel], 
                      info: Optional[Dict[str, Any]] = None):
        """
        새로운 선택모델 등록
        
        Args:
            model_type: 모델 타입
            model_class: 모델 클래스
            info: 모델 정보
        """
        cls._models[model_type] = model_class
        cls._model_info[model_type] = info or {}
        logger.info(f"선택모델 등록됨: {model_type.value}")
    
    @classmethod
    def create_model(cls, model_type: ChoiceModelType, config: Dict[str, Any]) -> BaseChoiceModel:
        """
        선택모델 생성
        
        Args:
            model_type: 생성할 모델 타입
            config: 모델 설정
            
        Returns:
            생성된 모델 인스턴스
        """
        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"지원되지 않는 모델 타입: {model_type.value}. "
                           f"사용 가능한 모델: {[m.value for m in available_models]}")
        
        model_class = cls._models[model_type]
        
        try:
            model = model_class(config)
            logger.info(f"선택모델 생성됨: {model_type.value}")
            return model
        except Exception as e:
            logger.error(f"모델 생성 실패: {model_type.value}, 오류: {e}")
            raise
    
    @classmethod
    def get_available_models(cls) -> List[ChoiceModelType]:
        """사용 가능한 모델 타입 목록 반환"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_type: ChoiceModelType) -> Dict[str, Any]:
        """특정 모델의 정보 반환"""
        if model_type not in cls._model_info:
            return {}
        return cls._model_info[model_type].copy()
    
    @classmethod
    def get_all_model_info(cls) -> Dict[ChoiceModelType, Dict[str, Any]]:
        """모든 모델의 정보 반환"""
        return cls._model_info.copy()
    
    @classmethod
    def is_model_available(cls, model_type: ChoiceModelType) -> bool:
        """모델 사용 가능 여부 확인"""
        return model_type in cls._models


# 편의 함수들
def create_choice_model(model_type: str, config: Dict[str, Any]) -> BaseChoiceModel:
    """
    문자열로 선택모델 생성
    
    Args:
        model_type: 모델 타입 문자열
        config: 모델 설정
        
    Returns:
        생성된 모델 인스턴스
    """
    try:
        model_enum = ChoiceModelType(model_type)
        return ChoiceModelFactory.create_model(model_enum, config)
    except ValueError as e:
        available_models = [m.value for m in ChoiceModelFactory.get_available_models()]
        raise ValueError(f"잘못된 모델 타입: {model_type}. "
                        f"사용 가능한 모델: {available_models}") from e


def get_available_models() -> List[str]:
    """사용 가능한 모델 타입 문자열 목록 반환"""
    return [model.value for model in ChoiceModelFactory.get_available_models()]


def get_model_info(model_type: str) -> Dict[str, Any]:
    """모델 정보 반환 (문자열 입력)"""
    try:
        model_enum = ChoiceModelType(model_type)
        return ChoiceModelFactory.get_model_info(model_enum)
    except ValueError:
        return {}


def register_choice_model(model_type: str, model_class: Type[BaseChoiceModel], 
                         info: Optional[Dict[str, Any]] = None):
    """선택모델 등록 (문자열 입력)"""
    try:
        model_enum = ChoiceModelType(model_type)
        ChoiceModelFactory.register_model(model_enum, model_class, info)
    except ValueError as e:
        raise ValueError(f"잘못된 모델 타입: {model_type}") from e


# 모델 자동 등록 함수
def _register_default_models():
    """기본 모델들 자동 등록"""
    try:
        # MNL 모델 등록
        from .multinomial_logit_model import MultinomialLogitModel
        ChoiceModelFactory.register_model(
            ChoiceModelType.MULTINOMIAL_LOGIT,
            MultinomialLogitModel,
            {
                "name": "Multinomial Logit",
                "description": "다항로짓 모델 - 가장 기본적인 선택모델",
                "parameters": ["coefficients"],
                "estimation_method": "maximum_likelihood",
                "complexity": "low",
                "computational_cost": "low"
            }
        )
    except ImportError:
        logger.warning("MultinomialLogitModel을 등록할 수 없습니다.")
    
    try:
        # RPL 모델 등록
        from .random_parameters_logit_model import RandomParametersLogitModel
        ChoiceModelFactory.register_model(
            ChoiceModelType.RANDOM_PARAMETERS_LOGIT,
            RandomParametersLogitModel,
            {
                "name": "Random Parameters Logit",
                "description": "확률모수 로짓 모델 - 개체 이질성을 고려한 모델",
                "parameters": ["mean_coefficients", "std_coefficients"],
                "estimation_method": "simulated_maximum_likelihood",
                "complexity": "high",
                "computational_cost": "high"
            }
        )
    except ImportError:
        logger.warning("RandomParametersLogitModel을 등록할 수 없습니다.")
    
    try:
        # Mixed Logit 모델 등록
        from .mixed_logit_model import MixedLogitModel
        ChoiceModelFactory.register_model(
            ChoiceModelType.MIXED_LOGIT,
            MixedLogitModel,
            {
                "name": "Mixed Logit",
                "description": "혼합로짓 모델 - 잠재 클래스와 확률모수를 결합한 모델",
                "parameters": ["class_coefficients", "mixing_coefficients"],
                "estimation_method": "simulated_maximum_likelihood",
                "complexity": "very_high",
                "computational_cost": "very_high"
            }
        )
    except ImportError:
        logger.warning("MixedLogitModel을 등록할 수 없습니다.")
    
    try:
        # Nested Logit 모델 등록
        from .nested_logit_model import NestedLogitModel
        ChoiceModelFactory.register_model(
            ChoiceModelType.NESTED_LOGIT,
            NestedLogitModel,
            {
                "name": "Nested Logit",
                "description": "중첩로짓 모델 - 계층적 선택구조를 고려한 모델",
                "parameters": ["coefficients", "nest_parameters"],
                "estimation_method": "maximum_likelihood",
                "complexity": "medium",
                "computational_cost": "medium"
            }
        )
    except ImportError:
        logger.warning("NestedLogitModel을 등록할 수 없습니다.")
    
    try:
        # Probit 모델 등록
        from .multinomial_probit_model import MultinomialProbitModel
        ChoiceModelFactory.register_model(
            ChoiceModelType.MULTINOMIAL_PROBIT,
            MultinomialProbitModel,
            {
                "name": "Multinomial Probit",
                "description": "다항프로빗 모델 - 정규분포 기반 선택모델",
                "parameters": ["coefficients", "covariance_matrix"],
                "estimation_method": "simulated_maximum_likelihood",
                "complexity": "high",
                "computational_cost": "very_high"
            }
        )
    except ImportError:
        logger.warning("MultinomialProbitModel을 등록할 수 없습니다.")


# 모듈 로드 시 기본 모델들 등록
_register_default_models()


# 팩토리 상태 확인 함수
def get_factory_status() -> Dict[str, Any]:
    """팩토리 상태 정보 반환"""
    available_models = ChoiceModelFactory.get_available_models()
    return {
        "total_models": len(available_models),
        "available_models": [model.value for model in available_models],
        "model_details": {
            model.value: ChoiceModelFactory.get_model_info(model)
            for model in available_models
        }
    }


def print_available_models():
    """사용 가능한 모델 정보 출력"""
    status = get_factory_status()
    
    print("=" * 60)
    print("사용 가능한 선택모델")
    print("=" * 60)
    print(f"총 {status['total_models']}개 모델 등록됨\n")
    
    for model_type in status['available_models']:
        info = status['model_details'][model_type]
        print(f"📊 {info.get('name', model_type)}")
        print(f"   타입: {model_type}")
        print(f"   설명: {info.get('description', 'N/A')}")
        print(f"   복잡도: {info.get('complexity', 'N/A')}")
        print(f"   계산비용: {info.get('computational_cost', 'N/A')}")
        print(f"   추정방법: {info.get('estimation_method', 'N/A')}")
        print()


if __name__ == "__main__":
    # 테스트 실행
    print_available_models()
