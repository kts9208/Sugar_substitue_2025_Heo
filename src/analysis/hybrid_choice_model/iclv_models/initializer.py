"""
Parameter Initializer for ICLV Models

ICLV 모델의 초기값 설정 전략 패턴입니다.
동시추정과 순차추정에 따라 다른 초기값 전략을 사용합니다.

단일책임 원칙:
- SimultaneousInitializer: 동시추정용 초기값 (모든 파라미터 한 번에)
- SequentialInitializer: 순차추정용 초기값 (단계별 초기값)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseInitializer(ABC):
    """
    초기값 설정 베이스 클래스
    """
    
    @abstractmethod
    def initialize(self, measurement_model, structural_model, 
                  choice_model, **kwargs) -> np.ndarray:
        """
        초기 파라미터 설정
        
        Args:
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            **kwargs: 추가 인자
        
        Returns:
            초기 파라미터 벡터
        """
        pass
    
    @abstractmethod
    def get_parameter_names(self, measurement_model, structural_model,
                           choice_model) -> List[str]:
        """
        파라미터 이름 리스트 생성
        
        Args:
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
        
        Returns:
            파라미터 이름 리스트
        """
        pass


class SimultaneousInitializer(BaseInitializer):
    """
    동시추정용 초기값 설정
    
    모든 모델의 파라미터를 한 번에 초기화합니다.
    - 측정모델 파라미터
    - 구조모델 파라미터
    - 선택모델 파라미터
    """
    
    def __init__(self, config):
        """
        Args:
            config: ICLVConfig 또는 MultiLatentConfig
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def initialize(self, measurement_model, structural_model, 
                  choice_model, **kwargs) -> np.ndarray:
        """
        동시추정용 초기값 설정
        
        Returns:
            전체 파라미터 벡터 (측정 + 구조 + 선택)
        """
        # 사용자 정의 초기값이 있으면 사용
        if 'initial_params' in kwargs and kwargs['initial_params'] is not None:
            self.logger.info("사용자 정의 초기값 사용")
            return kwargs['initial_params']
        
        params = []
        
        # 1. 측정모델 초기값
        measurement_params = self._initialize_measurement(measurement_model)
        params.extend(measurement_params)
        
        # 2. 구조모델 초기값
        structural_params = self._initialize_structural(structural_model)
        params.extend(structural_params)
        
        # 3. 선택모델 초기값
        choice_params = self._initialize_choice(choice_model)
        params.extend(choice_params)
        
        params = np.array(params)
        self.logger.info(f"초기값 설정 완료: {len(params)} 파라미터")
        
        return params
    
    def _initialize_measurement(self, measurement_model) -> List[float]:
        """측정모델 초기값"""
        params = []
        
        # 다중 잠재변수인 경우
        if hasattr(measurement_model, 'models'):
            for lv_name, model in measurement_model.models.items():
                # zeta (요인적재량): 1.0으로 초기화
                params.extend([1.0] * model.n_indicators)
                
                # 측정 방법에 따라 다른 초기값
                if hasattr(model.config, 'measurement_method'):
                    if model.config.measurement_method == 'continuous_linear':
                        # sigma_sq (오차분산): 1.0으로 초기화
                        params.extend([1.0] * model.n_indicators)
                    else:
                        # tau (임계값): 등간격으로 초기화
                        for _ in range(model.n_indicators):
                            tau_init = np.linspace(-2, 2, model.n_thresholds)
                            params.extend(tau_init.tolist())
        else:
            # 단일 잠재변수
            params.extend([1.0] * measurement_model.n_indicators)
            
            # tau (임계값)
            for _ in range(measurement_model.n_indicators):
                tau_init = np.linspace(-2, 2, measurement_model.n_thresholds)
                params.extend(tau_init.tolist())
        
        return params
    
    def _initialize_structural(self, structural_model) -> List[float]:
        """구조모델 초기값"""
        params = []
        
        # 계층적 구조인 경우
        if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
            # 각 경로의 gamma 파라미터
            for path in structural_model.hierarchical_paths:
                n_predictors = len(path['predictors'])
                params.extend([0.5] * n_predictors)
        # 병렬 구조인 경우
        elif hasattr(structural_model, 'endogenous_lv'):
            # gamma_lv (잠재변수 효과)
            params.extend([0.5] * structural_model.n_exo)

            # gamma_x (사회인구학적 변수 효과)
            if hasattr(structural_model, 'covariates'):
                params.extend([0.0] * len(structural_model.covariates))
        else:
            # 단일 잠재변수
            if hasattr(structural_model, 'sociodemographics'):
                params.extend([0.0] * len(structural_model.sociodemographics))

        return params

    def _initialize_choice(self, choice_model) -> List[float]:
        """선택모델 초기값"""
        params = []

        # intercept
        params.append(0.0)

        # beta (선택 속성 계수)
        n_attrs = len(choice_model.choice_attributes)
        params.extend([0.0] * n_attrs)

        # ✅ 모든 LV 주효과 모델인 경우
        if hasattr(choice_model, 'all_lvs_as_main') and choice_model.all_lvs_as_main:
            # lambda_i (각 LV별 주효과)
            if hasattr(choice_model, 'main_lvs') and choice_model.main_lvs:
                params.extend([1.0] * len(choice_model.main_lvs))

            # ✅ LV-Attribute 상호작용 초기값
            if hasattr(choice_model, 'lv_attribute_interactions') and choice_model.lv_attribute_interactions:
                # gamma (상호작용 계수): 0.0으로 초기화 (주효과보다 작게)
                params.extend([0.0] * len(choice_model.lv_attribute_interactions))

        # 조절효과 모델인 경우 (하위 호환성)
        elif hasattr(choice_model, 'moderation_enabled') and choice_model.moderation_enabled:
            # lambda_main (주효과)
            params.append(1.0)

            # lambda_mod (조절효과)
            if hasattr(choice_model, 'moderator_lvs'):
                params.extend([0.5] * len(choice_model.moderator_lvs))
        else:
            # lambda (기본 모델)
            params.append(1.0)

        return params

    def get_parameter_names(self, measurement_model, structural_model,
                           choice_model) -> List[str]:
        """파라미터 이름 리스트 생성"""
        names = []

        # 1. 측정모델 파라미터 이름
        if hasattr(measurement_model, 'models'):
            # 다중 잠재변수
            for lv_name, model in measurement_model.models.items():
                # zeta
                for indicator in model.config.indicators:
                    names.append(f'zeta_{lv_name}_{indicator}')

                # sigma_sq or tau
                if hasattr(model.config, 'measurement_method'):
                    if model.config.measurement_method == 'continuous_linear':
                        for indicator in model.config.indicators:
                            names.append(f'sigma_sq_{lv_name}_{indicator}')
                    else:
                        for indicator in model.config.indicators:
                            for k in range(model.n_thresholds):
                                names.append(f'tau_{lv_name}_{indicator}_{k+1}')
        else:
            # 단일 잠재변수
            for i, indicator in enumerate(measurement_model.config.indicators):
                names.append(f'zeta_{indicator}')

            for i, indicator in enumerate(measurement_model.config.indicators):
                for k in range(measurement_model.n_thresholds):
                    names.append(f'tau_{indicator}_{k+1}')

        # 2. 구조모델 파라미터 이름
        if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
            for path in structural_model.hierarchical_paths:
                for pred in path['predictors']:
                    names.append(f'gamma_{pred}_to_{path["target"]}')
        elif hasattr(structural_model, 'endogenous_lv'):
            for lv in structural_model.exogenous_lvs:
                names.append(f'gamma_lv_{lv}')

            if hasattr(structural_model, 'covariates'):
                for cov in structural_model.covariates:
                    names.append(f'gamma_x_{cov}')
        else:
            if hasattr(structural_model, 'sociodemographics'):
                for var in structural_model.sociodemographics:
                    names.append(f'gamma_{var}')

        # 3. 선택모델 파라미터 이름
        names.append('beta_intercept')

        for attr in choice_model.choice_attributes:
            names.append(f'beta_{attr}')

        # ✅ 모든 LV 주효과 모델인 경우
        if hasattr(choice_model, 'all_lvs_as_main') and choice_model.all_lvs_as_main:
            # lambda_i (각 LV별 주효과)
            if hasattr(choice_model, 'main_lvs') and choice_model.main_lvs:
                for lv_name in choice_model.main_lvs:
                    names.append(f'lambda_{lv_name}')

            # ✅ LV-Attribute 상호작용 파라미터 이름
            if hasattr(choice_model, 'lv_attribute_interactions') and choice_model.lv_attribute_interactions:
                for interaction in choice_model.lv_attribute_interactions:
                    lv_name = interaction['lv']
                    attr_name = interaction['attribute']
                    names.append(f'gamma_{lv_name}_{attr_name}')

        # 조절효과 모델인 경우 (하위 호환성)
        elif hasattr(choice_model, 'moderation_enabled') and choice_model.moderation_enabled:
            names.append('lambda_main')
            if hasattr(choice_model, 'moderator_lvs'):
                for mod_lv in choice_model.moderator_lvs:
                    names.append(f'lambda_mod_{mod_lv}')
        else:
            names.append('lambda')

        return names


class SequentialInitializer(BaseInitializer):
    """
    순차추정용 초기값 설정

    각 단계별로 독립적인 초기값을 설정합니다.
    - 1단계: 측정모델만
    - 2단계: 구조모델만
    - 3단계: 선택모델만
    """

    def __init__(self, config):
        """
        Args:
            config: ICLVConfig 또는 MultiLatentConfig
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def initialize(self, measurement_model, structural_model,
                  choice_model, **kwargs) -> Dict[str, np.ndarray]:
        """
        순차추정용 초기값 설정

        Returns:
            각 모델별 초기값 딕셔너리
            {
                'measurement': np.ndarray,
                'structural': np.ndarray,
                'choice': np.ndarray
            }
        """
        # SimultaneousInitializer 재사용
        sim_initializer = SimultaneousInitializer(self.config)

        measurement_params = sim_initializer._initialize_measurement(measurement_model)
        structural_params = sim_initializer._initialize_structural(structural_model)
        choice_params = sim_initializer._initialize_choice(choice_model)

        self.logger.info(
            f"순차추정 초기값 설정 완료: "
            f"측정({len(measurement_params)}), "
            f"구조({len(structural_params)}), "
            f"선택({len(choice_params)})"
        )

        return {
            'measurement': np.array(measurement_params),
            'structural': np.array(structural_params),
            'choice': np.array(choice_params)
        }

    def get_parameter_names(self, measurement_model, structural_model,
                           choice_model) -> Dict[str, List[str]]:
        """
        각 모델별 파라미터 이름 리스트 생성

        Returns:
            각 모델별 파라미터 이름 딕셔너리
        """
        # SimultaneousInitializer 재사용하여 전체 이름 생성 후 분할
        sim_initializer = SimultaneousInitializer(self.config)
        all_names = sim_initializer.get_parameter_names(
            measurement_model, structural_model, choice_model
        )

        # 파라미터 개수 계산
        n_measurement = len(sim_initializer._initialize_measurement(measurement_model))
        n_structural = len(sim_initializer._initialize_structural(structural_model))
        n_choice = len(sim_initializer._initialize_choice(choice_model))

        # 이름 분할
        measurement_names = all_names[:n_measurement]
        structural_names = all_names[n_measurement:n_measurement+n_structural]
        choice_names = all_names[n_measurement+n_structural:]

        return {
            'measurement': measurement_names,
            'structural': structural_names,
            'choice': choice_names
        }

