"""
Path Analysis Model Builder

경로분석 모델을 구축하는 클래스와 함수들을 제공합니다.
semopy 모델 스펙을 생성하고 관리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PathModelBuilder:
    """경로분석 모델 구축 클래스"""
    
    def __init__(self, data_dir: str = "processed_data/survey_data"):
        """
        초기화
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.available_factors = self._get_available_factors()
        logger.info(f"PathModelBuilder 초기화 완료. 사용 가능한 요인: {self.available_factors}")
    
    def _get_available_factors(self) -> List[str]:
        """사용 가능한 요인들 확인"""
        if not self.data_dir.exists():
            logger.warning(f"데이터 디렉토리가 존재하지 않습니다: {self.data_dir}")
            return []
        
        factors = []
        for csv_file in self.data_dir.glob("*.csv"):
            factor_name = csv_file.stem
            factors.append(factor_name)
        
        return sorted(factors)
    
    def create_simple_mediation_model(self, 
                                    independent_var: str,
                                    mediator_var: str, 
                                    dependent_var: str) -> str:
        """
        단순 매개모델 생성 (X -> M -> Y)
        
        Args:
            independent_var (str): 독립변수 (X)
            mediator_var (str): 매개변수 (M)
            dependent_var (str): 종속변수 (Y)
            
        Returns:
            str: semopy 모델 스펙
        """
        logger.info(f"단순 매개모델 생성: {independent_var} -> {mediator_var} -> {dependent_var}")
        
        # 각 요인의 측정모델 생성
        measurement_models = []
        for factor in [independent_var, mediator_var, dependent_var]:
            items = self._get_factor_items(factor)
            if items:
                measurement_model = f"{factor} =~ " + " + ".join(items)
                measurement_models.append(measurement_model)
        
        # 구조모델 생성 (경로)
        structural_model = [
            f"{mediator_var} ~ {independent_var}",      # a path
            f"{dependent_var} ~ {mediator_var}",        # b path
            f"{dependent_var} ~ {independent_var}"      # c' path (direct effect)
        ]
        
        # 전체 모델 결합
        full_model = measurement_models + structural_model
        model_spec = "\n".join(full_model)
        
        logger.info("단순 매개모델 생성 완료")
        return model_spec
    
    def create_multiple_mediation_model(self,
                                      independent_var: str,
                                      mediator_vars: List[str],
                                      dependent_var: str,
                                      allow_mediator_correlations: bool = True) -> str:
        """
        다중 매개모델 생성 (X -> M1,M2,... -> Y)
        
        Args:
            independent_var (str): 독립변수
            mediator_vars (List[str]): 매개변수들
            dependent_var (str): 종속변수
            allow_mediator_correlations (bool): 매개변수 간 상관관계 허용 여부
            
        Returns:
            str: semopy 모델 스펙
        """
        logger.info(f"다중 매개모델 생성: {independent_var} -> {mediator_vars} -> {dependent_var}")
        
        # 측정모델 생성
        measurement_models = []
        all_factors = [independent_var] + mediator_vars + [dependent_var]
        
        for factor in all_factors:
            items = self._get_factor_items(factor)
            if items:
                measurement_model = f"{factor} =~ " + " + ".join(items)
                measurement_models.append(measurement_model)
        
        # 구조모델 생성
        structural_model = []
        
        # X -> M paths (a paths)
        for mediator in mediator_vars:
            structural_model.append(f"{mediator} ~ {independent_var}")
        
        # M -> Y paths (b paths)
        for mediator in mediator_vars:
            structural_model.append(f"{dependent_var} ~ {mediator}")
        
        # X -> Y path (c' path - direct effect)
        structural_model.append(f"{dependent_var} ~ {independent_var}")
        
        # 매개변수 간 상관관계 (선택사항)
        if allow_mediator_correlations and len(mediator_vars) > 1:
            for i, med1 in enumerate(mediator_vars):
                for med2 in mediator_vars[i+1:]:
                    structural_model.append(f"{med1} ~~ {med2}")
        
        # 전체 모델 결합
        full_model = measurement_models + structural_model
        model_spec = "\n".join(full_model)
        
        logger.info("다중 매개모델 생성 완료")
        return model_spec
    
    def create_serial_mediation_model(self,
                                    independent_var: str,
                                    mediator_vars: List[str],
                                    dependent_var: str) -> str:
        """
        순차 매개모델 생성 (X -> M1 -> M2 -> ... -> Y)
        
        Args:
            independent_var (str): 독립변수
            mediator_vars (List[str]): 순차적 매개변수들
            dependent_var (str): 종속변수
            
        Returns:
            str: semopy 모델 스펙
        """
        logger.info(f"순차 매개모델 생성: {independent_var} -> {' -> '.join(mediator_vars)} -> {dependent_var}")
        
        # 측정모델 생성
        measurement_models = []
        all_factors = [independent_var] + mediator_vars + [dependent_var]
        
        for factor in all_factors:
            items = self._get_factor_items(factor)
            if items:
                measurement_model = f"{factor} =~ " + " + ".join(items)
                measurement_models.append(measurement_model)
        
        # 구조모델 생성
        structural_model = []
        
        # X -> M1 path
        structural_model.append(f"{mediator_vars[0]} ~ {independent_var}")
        
        # M1 -> M2 -> ... paths (순차적)
        for i in range(len(mediator_vars) - 1):
            structural_model.append(f"{mediator_vars[i+1]} ~ {mediator_vars[i]}")
        
        # 마지막 매개변수 -> Y path
        structural_model.append(f"{dependent_var} ~ {mediator_vars[-1]}")
        
        # X -> Y path (direct effect)
        structural_model.append(f"{dependent_var} ~ {independent_var}")
        
        # 전체 모델 결합
        full_model = measurement_models + structural_model
        model_spec = "\n".join(full_model)
        
        logger.info("순차 매개모델 생성 완료")
        return model_spec
    
    def create_custom_structural_model(self,
                                     variables: List[str],
                                     paths: List[Tuple[str, str]],
                                     correlations: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        사용자 정의 구조모델 생성
        
        Args:
            variables (List[str]): 모델에 포함될 변수들
            paths (List[Tuple[str, str]]): 경로 리스트 [(from, to), ...]
            correlations (Optional[List[Tuple[str, str]]]): 상관관계 리스트
            
        Returns:
            str: semopy 모델 스펙
        """
        logger.info(f"사용자 정의 구조모델 생성: 변수 {len(variables)}개, 경로 {len(paths)}개")
        
        # 측정모델 생성
        measurement_models = []
        for variable in variables:
            items = self._get_factor_items(variable)
            if items:
                measurement_model = f"{variable} =~ " + " + ".join(items)
                measurement_models.append(measurement_model)
        
        # 구조모델 생성 (경로)
        structural_model = []
        for from_var, to_var in paths:
            structural_model.append(f"{to_var} ~ {from_var}")
        
        # 상관관계 추가 (선택사항)
        if correlations:
            for var1, var2 in correlations:
                structural_model.append(f"{var1} ~~ {var2}")
        
        # 전체 모델 결합
        full_model = measurement_models + structural_model
        model_spec = "\n".join(full_model)
        
        logger.info("사용자 정의 구조모델 생성 완료")
        return model_spec

    def create_comprehensive_structural_model(self, variables: List[str],
                                            include_bidirectional: bool = True,
                                            include_feedback: bool = True) -> str:
        """
        포괄적 구조모델 생성 (이론적으로 타당한 모든 경로 포함)

        Args:
            variables (List[str]): 모델에 포함될 변수들
            include_bidirectional (bool): 양방향 경로 포함 여부
            include_feedback (bool): 피드백 경로 포함 여부

        Returns:
            str: semopy 모델 스펙
        """
        logger.info(f"포괄적 구조모델 생성: 변수 {len(variables)}개")

        # 측정모델 생성
        measurement_models = []
        for variable in variables:
            items = self._get_factor_items(variable)
            if items:
                measurement_model = f"{variable} =~ " + " + ".join(items)
                measurement_models.append(measurement_model)

        # 이론적으로 타당한 경로 정의
        theoretical_paths = []

        # 기본 경로 (일반적인 인과관계)
        if 'health_concern' in variables:
            if 'perceived_benefit' in variables:
                theoretical_paths.append(('health_concern', 'perceived_benefit'))
            if 'perceived_price' in variables:
                theoretical_paths.append(('health_concern', 'perceived_price'))
            if 'nutrition_knowledge' in variables:
                theoretical_paths.append(('health_concern', 'nutrition_knowledge'))
            if 'purchase_intention' in variables:
                theoretical_paths.append(('health_concern', 'purchase_intention'))

        if 'nutrition_knowledge' in variables:
            if 'perceived_benefit' in variables:
                theoretical_paths.append(('nutrition_knowledge', 'perceived_benefit'))
            if 'purchase_intention' in variables:
                theoretical_paths.append(('nutrition_knowledge', 'purchase_intention'))
            if 'perceived_price' in variables:
                theoretical_paths.append(('nutrition_knowledge', 'perceived_price'))

        if 'perceived_benefit' in variables and 'purchase_intention' in variables:
            theoretical_paths.append(('perceived_benefit', 'purchase_intention'))

        if 'perceived_price' in variables and 'purchase_intention' in variables:
            theoretical_paths.append(('perceived_price', 'purchase_intention'))

        # 상호작용 경로
        if include_bidirectional:
            if 'perceived_benefit' in variables and 'perceived_price' in variables:
                theoretical_paths.append(('perceived_benefit', 'perceived_price'))
                theoretical_paths.append(('perceived_price', 'perceived_benefit'))

            if 'perceived_benefit' in variables and 'nutrition_knowledge' in variables:
                theoretical_paths.append(('perceived_benefit', 'nutrition_knowledge'))

        # 피드백 경로
        if include_feedback:
            if 'nutrition_knowledge' in variables and 'health_concern' in variables:
                theoretical_paths.append(('nutrition_knowledge', 'health_concern'))

            if 'perceived_benefit' in variables and 'health_concern' in variables:
                theoretical_paths.append(('perceived_benefit', 'health_concern'))

            if 'purchase_intention' in variables and 'health_concern' in variables:
                theoretical_paths.append(('purchase_intention', 'health_concern'))

            # 추가 피드백 경로 (누락된 경로들)
            if 'perceived_price' in variables and 'nutrition_knowledge' in variables:
                theoretical_paths.append(('perceived_price', 'nutrition_knowledge'))

            if 'perceived_price' in variables and 'health_concern' in variables:
                theoretical_paths.append(('perceived_price', 'health_concern'))

            if 'purchase_intention' in variables and 'nutrition_knowledge' in variables:
                theoretical_paths.append(('purchase_intention', 'nutrition_knowledge'))

            if 'purchase_intention' in variables and 'perceived_price' in variables:
                theoretical_paths.append(('purchase_intention', 'perceived_price'))

            if 'purchase_intention' in variables and 'perceived_benefit' in variables:
                theoretical_paths.append(('purchase_intention', 'perceived_benefit'))

        # 구조적 경로 생성
        structural_models = []
        for from_var, to_var in theoretical_paths:
            structural_models.append(f"{to_var} ~ {from_var}")

        # 전체 모델 스펙 조합
        model_spec = "\n".join(measurement_models + structural_models)

        logger.info(f"포괄적 구조모델 생성 완료: {len(theoretical_paths)}개 경로")
        return model_spec

    def create_saturated_structural_model(self, variables: List[str]) -> str:
        """
        포화 구조모델 생성 (모든 가능한 경로 포함)

        Args:
            variables (List[str]): 모델에 포함될 변수들

        Returns:
            str: semopy 모델 스펙
        """
        logger.info(f"포화 구조모델 생성: 변수 {len(variables)}개")

        # 측정모델 생성
        measurement_models = []
        for variable in variables:
            items = self._get_factor_items(variable)
            if items:
                measurement_model = f"{variable} =~ " + " + ".join(items)
                measurement_models.append(measurement_model)

        # 모든 가능한 경로 생성 (자기 자신 제외)
        structural_models = []
        for i, from_var in enumerate(variables):
            for j, to_var in enumerate(variables):
                if i != j:  # 자기 자신으로의 경로는 제외
                    structural_models.append(f"{to_var} ~ {from_var}")

        # 전체 모델 스펙 조합
        model_spec = "\n".join(measurement_models + structural_models)

        total_paths = len(variables) * (len(variables) - 1)
        logger.info(f"포화 구조모델 생성 완료: {total_paths}개 경로 (모든 가능한 경로)")
        return model_spec
    
    def _get_factor_items(self, factor_name: str) -> List[str]:
        """요인의 측정문항들 반환"""
        factor_file = self.data_dir / f"{factor_name}.csv"
        
        if not factor_file.exists():
            logger.warning(f"요인 파일을 찾을 수 없습니다: {factor_file}")
            return []
        
        try:
            data = pd.read_csv(factor_file)
            # 'q'로 시작하는 문항들만 선택
            items = [col for col in data.columns if col.startswith('q')]
            logger.debug(f"{factor_name} 요인의 측정문항: {items}")
            return items
            
        except Exception as e:
            logger.error(f"요인 파일 읽기 오류 ({factor_name}): {e}")
            return []
    
    def validate_model_variables(self, variables: List[str]) -> Dict[str, bool]:
        """모델 변수들의 유효성 검증"""
        validation_results = {}
        
        for variable in variables:
            is_valid = variable in self.available_factors
            validation_results[variable] = is_valid
            
            if not is_valid:
                logger.warning(f"변수 '{variable}'에 대한 데이터를 찾을 수 없습니다.")
        
        return validation_results

    def create_five_factor_comprehensive_model(self,
                                             variables: List[str],
                                             include_all_mediations: bool = True,
                                             mediation_threshold: float = 0.01) -> str:
        """
        5개 요인 간 포괄적 매개모델 생성 (모든 가능한 매개경로 포함)

        Args:
            variables (List[str]): 5개 요인 변수들
            include_all_mediations (bool): 모든 매개경로 포함 여부
            mediation_threshold (float): 매개효과 임계값

        Returns:
            str: semopy 모델 스펙
        """
        if len(variables) != 5:
            raise ValueError("정확히 5개의 변수가 필요합니다.")

        logger.info(f"5개 요인 포괄적 매개모델 생성: {variables}")

        # 측정모델 생성
        measurement_models = []
        for variable in variables:
            items = self._get_factor_items(variable)
            if items:
                measurement_model = f"{variable} =~ " + " + ".join(items)
                measurement_models.append(measurement_model)

        # 구조모델 생성 - 모든 가능한 직접경로
        structural_models = []

        if include_all_mediations:
            # 모든 가능한 X -> Y 직접경로 포함
            for i, from_var in enumerate(variables):
                for j, to_var in enumerate(variables):
                    if i != j:  # 자기 자신 제외
                        structural_models.append(f"{to_var} ~ {from_var}")
        else:
            # 이론적으로 타당한 경로만 포함 (예: 순차적 관계)
            for i in range(len(variables) - 1):
                for j in range(i + 1, len(variables)):
                    # 양방향 경로
                    structural_models.append(f"{variables[j]} ~ {variables[i]}")
                    structural_models.append(f"{variables[i]} ~ {variables[j]}")

        # 공분산 추가 (오차항 간)
        covariance_models = []
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                covariance_models.append(f"{variables[i]} ~~ {variables[j]}")

        # 전체 모델 스펙 조합
        all_models = measurement_models + structural_models + covariance_models
        model_spec = "\n".join(all_models)

        logger.info(f"5개 요인 포괄적 모델 생성 완료: {len(structural_models)}개 구조경로, {len(covariance_models)}개 공분산")
        return model_spec

    def create_mediation_focused_model(self,
                                     variables: List[str],
                                     primary_relationships: List[Tuple[str, str]],
                                     include_all_possible_mediators: bool = True) -> str:
        """
        매개효과 중심의 모델 생성

        Args:
            variables (List[str]): 모든 변수들
            primary_relationships (List[Tuple[str, str]]): 주요 관계 [(X, Y), ...]
            include_all_possible_mediators (bool): 모든 가능한 매개변수 포함 여부

        Returns:
            str: semopy 모델 스펙
        """
        logger.info(f"매개효과 중심 모델 생성: {len(primary_relationships)}개 주요 관계")

        # 측정모델 생성
        measurement_models = []
        for variable in variables:
            items = self._get_factor_items(variable)
            if items:
                measurement_model = f"{variable} =~ " + " + ".join(items)
                measurement_models.append(measurement_model)

        structural_models = []

        # 각 주요 관계에 대해 매개모델 생성
        for independent_var, dependent_var in primary_relationships:
            # 직접경로
            structural_models.append(f"{dependent_var} ~ {independent_var}")

            if include_all_possible_mediators:
                # 모든 다른 변수를 매개변수로 고려
                potential_mediators = [v for v in variables if v not in [independent_var, dependent_var]]

                for mediator in potential_mediators:
                    # X -> M 경로
                    structural_models.append(f"{mediator} ~ {independent_var}")
                    # M -> Y 경로
                    structural_models.append(f"{dependent_var} ~ {mediator}")

        # 중복 제거
        structural_models = list(set(structural_models))

        # 전체 모델 스펙 조합
        model_spec = "\n".join(measurement_models + structural_models)

        logger.info(f"매개효과 중심 모델 생성 완료: {len(structural_models)}개 구조경로")
        return model_spec


# 편의 함수들
def create_mediation_model(independent_var: str,
                          mediator_var: str,
                          dependent_var: str,
                          data_dir: str = "processed_data/survey_data") -> str:
    """단순 매개모델 생성 편의 함수"""
    builder = PathModelBuilder(data_dir)
    return builder.create_simple_mediation_model(independent_var, mediator_var, dependent_var)


def create_structural_model(variables: List[str],
                           paths: List[Tuple[str, str]],
                           correlations: Optional[List[Tuple[str, str]]] = None,
                           data_dir: str = "processed_data/survey_data") -> str:
    """구조모델 생성 편의 함수"""
    builder = PathModelBuilder(data_dir)
    return builder.create_custom_structural_model(variables, paths, correlations)


def create_multiple_mediation_model(independent_var: str,
                                   mediator_vars: List[str],
                                   dependent_var: str,
                                   allow_correlations: bool = True,
                                   data_dir: str = "processed_data/survey_data") -> str:
    """다중 매개모델 생성 편의 함수"""
    builder = PathModelBuilder(data_dir)
    return builder.create_multiple_mediation_model(
        independent_var, mediator_vars, dependent_var, allow_correlations
    )


def create_comprehensive_model(variables: List[str],
                             include_bidirectional: bool = True,
                             include_feedback: bool = True,
                             data_dir: str = "processed_data/survey_data") -> str:
    """포괄적 구조모델 생성 편의 함수"""
    builder = PathModelBuilder(data_dir)
    return builder.create_comprehensive_structural_model(
        variables, include_bidirectional, include_feedback
    )


def create_saturated_model(variables: List[str],
                          data_dir: str = "processed_data/survey_data") -> str:
    """포화 구조모델 생성 편의 함수 (모든 가능한 경로 포함)"""
    builder = PathModelBuilder(data_dir)
    return builder.create_saturated_structural_model(variables)


def create_five_factor_comprehensive_model(variables: List[str],
                                         include_all_mediations: bool = True,
                                         data_dir: str = "processed_data/survey_data") -> str:
    """5개 요인 포괄적 매개모델 생성 편의 함수"""
    builder = PathModelBuilder(data_dir)
    return builder.create_five_factor_comprehensive_model(variables, include_all_mediations)


def create_mediation_focused_model(variables: List[str],
                                 primary_relationships: List[Tuple[str, str]],
                                 include_all_possible_mediators: bool = True,
                                 data_dir: str = "processed_data/survey_data") -> str:
    """매개효과 중심 모델 생성 편의 함수"""
    builder = PathModelBuilder(data_dir)
    return builder.create_mediation_focused_model(
        variables, primary_relationships, include_all_possible_mediators
    )
