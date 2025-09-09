"""
Interaction Builder Module

semopy 기반 상호작용항 생성 및 조절효과 모델 구축을 위한 모듈입니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from .config import ModerationAnalysisConfig, get_factor_items_mapping

logger = logging.getLogger(__name__)


class InteractionBuilder:
    """상호작용항 생성 및 조절효과 모델 구축 클래스"""
    
    def __init__(self, config: Optional[ModerationAnalysisConfig] = None):
        """
        상호작용 빌더 초기화
        
        Args:
            config (Optional[ModerationAnalysisConfig]): 분석 설정
        """
        from .config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG
        self.factor_items = get_factor_items_mapping()
        
        logger.info("상호작용 빌더 초기화 완료")
    
    def create_interaction_terms(self, data: pd.DataFrame, 
                               independent_var: str, moderator_var: str,
                               method: str = "product") -> pd.DataFrame:
        """
        상호작용항 생성
        
        Args:
            data (pd.DataFrame): 원본 데이터
            independent_var (str): 독립변수
            moderator_var (str): 조절변수
            method (str): 상호작용항 생성 방법 ('product', 'orthogonal')
            
        Returns:
            pd.DataFrame: 상호작용항이 추가된 데이터
        """
        logger.info(f"상호작용항 생성: {independent_var} × {moderator_var}")
        
        # 데이터 복사
        interaction_data = data.copy()
        
        # 상호작용항 이름
        interaction_name = f"{independent_var}_x_{moderator_var}"
        
        if method == "product":
            # 곱셈 상호작용항
            interaction_data[interaction_name] = (
                interaction_data[independent_var] * interaction_data[moderator_var]
            )
        elif method == "orthogonal":
            # 직교화된 상호작용항
            interaction_data[interaction_name] = self._create_orthogonal_interaction(
                interaction_data, independent_var, moderator_var
            )
        else:
            raise ValueError(f"지원하지 않는 상호작용항 생성 방법: {method}")
        
        logger.info(f"상호작용항 '{interaction_name}' 생성 완료")
        return interaction_data
    
    def _create_orthogonal_interaction(self, data: pd.DataFrame,
                                     independent_var: str, moderator_var: str) -> pd.Series:
        """직교화된 상호작용항 생성"""
        X = data[independent_var]
        Z = data[moderator_var]
        
        # 곱셈항
        XZ = X * Z
        
        # X와 Z에 대해 직교화
        # XZ_orthogonal = XZ - β1*X - β2*Z - β0
        from sklearn.linear_model import LinearRegression
        
        # X, Z를 예측변수로 하여 XZ를 예측
        predictors = np.column_stack([X, Z])
        reg = LinearRegression().fit(predictors, XZ)
        
        # 잔차가 직교화된 상호작용항
        XZ_pred = reg.predict(predictors)
        XZ_orthogonal = XZ - XZ_pred
        
        return XZ_orthogonal
    
    def build_moderation_model_spec(self, independent_var: str, dependent_var: str,
                                  moderator_var: str, control_vars: Optional[List[str]] = None,
                                  include_measurement_model: bool = True) -> str:
        """
        조절효과 분석을 위한 semopy 모델 스펙 생성
        
        Args:
            independent_var (str): 독립변수
            dependent_var (str): 종속변수
            moderator_var (str): 조절변수
            control_vars (Optional[List[str]]): 통제변수들
            include_measurement_model (bool): 측정모델 포함 여부
            
        Returns:
            str: semopy 모델 스펙
        """
        logger.info("조절효과 모델 스펙 생성 시작")
        
        model_parts = []
        
        # 1. 측정모델 (잠재변수가 있는 경우)
        if include_measurement_model:
            measurement_models = self._create_measurement_models([
                independent_var, dependent_var, moderator_var
            ])
            if measurement_models:
                model_parts.extend(measurement_models)
        
        # 2. 구조모델 (조절효과 모델)
        structural_model = self._create_structural_model(
            independent_var, dependent_var, moderator_var, control_vars
        )
        model_parts.append(structural_model)
        
        # 3. 상호작용항 정의
        interaction_definition = self._create_interaction_definition(
            independent_var, moderator_var
        )
        model_parts.append(interaction_definition)
        
        # 모델 스펙 결합
        model_spec = "\n".join(model_parts)
        
        logger.info("조절효과 모델 스펙 생성 완료")
        logger.debug(f"모델 스펙:\n{model_spec}")
        
        return model_spec
    
    def _create_measurement_models(self, variables: List[str]) -> List[str]:
        """측정모델 생성"""
        measurement_models = []
        
        for variable in variables:
            if variable in self.factor_items:
                items = self.factor_items[variable]
                if len(items) > 1:  # 다중 지표가 있는 경우만
                    measurement_model = f"{variable} =~ " + " + ".join(items)
                    measurement_models.append(measurement_model)
        
        return measurement_models
    
    def _create_structural_model(self, independent_var: str, dependent_var: str,
                               moderator_var: str, control_vars: Optional[List[str]] = None) -> str:
        """구조모델 생성"""
        # 기본 예측변수들
        predictors = [independent_var, moderator_var]
        
        # 상호작용항 추가
        interaction_term = f"{independent_var}_x_{moderator_var}"
        predictors.append(interaction_term)
        
        # 통제변수 추가
        if control_vars:
            predictors.extend(control_vars)
        
        # 구조방정식 생성
        structural_equation = f"{dependent_var} ~ " + " + ".join(predictors)
        
        return structural_equation
    
    def _create_interaction_definition(self, independent_var: str, moderator_var: str) -> str:
        """상호작용항 정의 생성"""
        interaction_name = f"{independent_var}_x_{moderator_var}"
        
        # semopy에서 상호작용항은 데이터에서 미리 계산되어야 함
        # 여기서는 주석으로 표시
        interaction_definition = f"# {interaction_name} := {independent_var} * {moderator_var}"
        
        return interaction_definition
    
    def create_simple_moderation_model(self, independent_var: str, dependent_var: str,
                                     moderator_var: str) -> str:
        """단순 조절효과 모델 생성"""
        return self.build_moderation_model_spec(
            independent_var, dependent_var, moderator_var,
            control_vars=None, include_measurement_model=False
        )
    
    def create_complex_moderation_model(self, independent_var: str, dependent_var: str,
                                      moderator_var: str, control_vars: List[str]) -> str:
        """복합 조절효과 모델 생성 (통제변수 포함)"""
        return self.build_moderation_model_spec(
            independent_var, dependent_var, moderator_var,
            control_vars=control_vars, include_measurement_model=True
        )
    
    def validate_interaction_data(self, data: pd.DataFrame, 
                                independent_var: str, moderator_var: str) -> bool:
        """상호작용 데이터 유효성 검증"""
        try:
            # 필요한 변수들이 있는지 확인
            required_vars = [independent_var, moderator_var]
            missing_vars = [var for var in required_vars if var not in data.columns]
            
            if missing_vars:
                logger.error(f"필요한 변수가 없습니다: {missing_vars}")
                return False
            
            # 결측치 확인
            for var in required_vars:
                if data[var].isnull().any():
                    logger.warning(f"{var}에 결측치가 있습니다.")
            
            # 분산 확인 (상수가 아닌지)
            for var in required_vars:
                if data[var].var() == 0:
                    logger.error(f"{var}의 분산이 0입니다 (상수).")
                    return False
            
            logger.info("상호작용 데이터 유효성 검증 통과")
            return True
            
        except Exception as e:
            logger.error(f"데이터 유효성 검증 실패: {e}")
            return False
    
    def get_interaction_summary(self, data: pd.DataFrame, 
                              independent_var: str, moderator_var: str) -> Dict[str, Any]:
        """상호작용 데이터 요약"""
        interaction_name = f"{independent_var}_x_{moderator_var}"
        
        summary = {
            'variables': {
                'independent': independent_var,
                'moderator': moderator_var,
                'interaction': interaction_name
            },
            'descriptive_stats': {},
            'correlations': {}
        }
        
        # 기술통계
        vars_to_analyze = [independent_var, moderator_var]
        if interaction_name in data.columns:
            vars_to_analyze.append(interaction_name)
        
        for var in vars_to_analyze:
            if var in data.columns:
                summary['descriptive_stats'][var] = {
                    'mean': data[var].mean(),
                    'std': data[var].std(),
                    'min': data[var].min(),
                    'max': data[var].max(),
                    'skewness': data[var].skew(),
                    'kurtosis': data[var].kurtosis()
                }
        
        # 상관관계
        corr_data = data[vars_to_analyze]
        summary['correlations'] = corr_data.corr().to_dict()
        
        return summary


# 편의 함수들
def create_interaction_terms(data: pd.DataFrame, independent_var: str, 
                           moderator_var: str, method: str = "product",
                           config: Optional[ModerationAnalysisConfig] = None) -> pd.DataFrame:
    """상호작용항 생성 편의 함수"""
    builder = InteractionBuilder(config)
    return builder.create_interaction_terms(data, independent_var, moderator_var, method)


def build_moderation_model(independent_var: str, dependent_var: str, moderator_var: str,
                          control_vars: Optional[List[str]] = None,
                          config: Optional[ModerationAnalysisConfig] = None) -> str:
    """조절효과 모델 구축 편의 함수"""
    builder = InteractionBuilder(config)
    return builder.build_moderation_model_spec(
        independent_var, dependent_var, moderator_var, control_vars
    )


def create_interaction_model_spec(independent_var: str, dependent_var: str, 
                                moderator_var: str,
                                config: Optional[ModerationAnalysisConfig] = None) -> str:
    """상호작용 모델 스펙 생성 편의 함수"""
    builder = InteractionBuilder(config)
    return builder.create_simple_moderation_model(independent_var, dependent_var, moderator_var)
