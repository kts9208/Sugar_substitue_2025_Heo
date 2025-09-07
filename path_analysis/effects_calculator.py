"""
Effects Calculator Module

경로분석에서 직접효과, 간접효과, 총효과를 계산하는 모듈입니다.
매개효과 분석과 부트스트랩 신뢰구간 계산 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import warnings
from itertools import combinations

# semopy 임포트
try:
    import semopy
    from semopy import Model
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class EffectsCalculator:
    """효과 계산 클래스"""
    
    def __init__(self, model: Optional[Model] = None, 
                 bootstrap_samples: int = 1000,
                 confidence_level: float = 0.95):
        """
        초기화
        
        Args:
            model (Optional[Model]): 적합된 semopy 모델
            bootstrap_samples (int): 부트스트랩 샘플 수
            confidence_level (float): 신뢰수준
        """
        if not SEMOPY_AVAILABLE:
            raise ImportError("semopy 라이브러리가 필요합니다.")
        
        self.model = model
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        logger.info("EffectsCalculator 초기화 완료")
    
    def calculate_all_effects(self, 
                            independent_var: str,
                            dependent_var: str,
                            mediator_vars: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        모든 효과 계산 (직접효과, 간접효과, 총효과)
        
        Args:
            independent_var (str): 독립변수
            dependent_var (str): 종속변수
            mediator_vars (Optional[List[str]]): 매개변수들
            
        Returns:
            Dict[str, Any]: 효과 분석 결과
        """
        if self.model is None:
            raise ValueError("모델이 설정되지 않았습니다.")
        
        logger.info(f"효과 계산 시작: {independent_var} -> {dependent_var}")
        
        results = {
            'variables': {
                'independent': independent_var,
                'dependent': dependent_var,
                'mediators': mediator_vars or []
            },
            'direct_effects': {},
            'indirect_effects': {},
            'total_effects': {},
            'mediation_analysis': {}
        }
        
        try:
            # 직접효과 계산
            results['direct_effects'] = self.calculate_direct_effects(
                independent_var, dependent_var
            )
            
            # 간접효과 계산 (매개변수가 있는 경우)
            if mediator_vars:
                results['indirect_effects'] = self.calculate_indirect_effects(
                    independent_var, dependent_var, mediator_vars
                )
                
                # 매개효과 분석
                results['mediation_analysis'] = self.analyze_mediation_effects(
                    independent_var, dependent_var, mediator_vars
                )
            
            # 총효과 계산
            results['total_effects'] = self.calculate_total_effects(
                results['direct_effects'], results['indirect_effects']
            )
            
            logger.info("효과 계산 완료")
            return results
            
        except Exception as e:
            logger.error(f"효과 계산 중 오류: {e}")
            raise
    
    def calculate_direct_effects(self, 
                               independent_var: str,
                               dependent_var: str) -> Dict[str, Any]:
        """
        직접효과 계산
        
        Args:
            independent_var (str): 독립변수
            dependent_var (str): 종속변수
            
        Returns:
            Dict[str, Any]: 직접효과 결과
        """
        try:
            # 모델 파라미터 가져오기
            params = self.model.inspect()
            
            # 직접 경로 찾기 (independent_var -> dependent_var)
            direct_path = params[
                (params['lval'] == dependent_var) & 
                (params['rval'] == independent_var) &
                (params['op'] == '~')
            ]
            
            if direct_path.empty:
                logger.warning(f"직접 경로를 찾을 수 없습니다: {independent_var} -> {dependent_var}")
                return {
                    'coefficient': 0.0,
                    'standard_error': np.nan,
                    'z_value': np.nan,
                    'p_value': np.nan,
                    'path_exists': False
                }
            
            direct_effect = {
                'coefficient': float(direct_path['Estimate'].iloc[0]),
                'standard_error': float(direct_path['Std. Error'].iloc[0]),
                'z_value': float(direct_path['z-value'].iloc[0]),
                'p_value': float(direct_path['P(>|z|)'].iloc[0]),
                'path_exists': True
            }
            
            # 표준화 계수 추가 (가능한 경우)
            try:
                std_params = self.model.inspect(std_est=True)
                std_direct = std_params[
                    (std_params['lval'] == dependent_var) & 
                    (std_params['rval'] == independent_var) &
                    (std_params['op'] == '~')
                ]
                if not std_direct.empty:
                    direct_effect['standardized_coefficient'] = float(std_direct['Std. Estimate'].iloc[0])
            except:
                pass
            
            logger.info(f"직접효과 계산 완료: {direct_effect['coefficient']:.4f}")
            return direct_effect
            
        except Exception as e:
            logger.error(f"직접효과 계산 오류: {e}")
            return {}
    
    def calculate_indirect_effects(self,
                                 independent_var: str,
                                 dependent_var: str,
                                 mediator_vars: List[str]) -> Dict[str, Any]:
        """
        간접효과 계산
        
        Args:
            independent_var (str): 독립변수
            dependent_var (str): 종속변수
            mediator_vars (List[str]): 매개변수들
            
        Returns:
            Dict[str, Any]: 간접효과 결과
        """
        try:
            params = self.model.inspect()
            
            indirect_effects = {
                'individual_paths': {},
                'total_indirect_effect': 0.0,
                'specific_indirect_effects': {}
            }
            
            # 각 매개변수별 간접효과 계산
            for mediator in mediator_vars:
                # a path: independent -> mediator
                a_path = params[
                    (params['lval'] == mediator) & 
                    (params['rval'] == independent_var) &
                    (params['op'] == '~')
                ]
                
                # b path: mediator -> dependent
                b_path = params[
                    (params['lval'] == dependent_var) & 
                    (params['rval'] == mediator) &
                    (params['op'] == '~')
                ]
                
                if not a_path.empty and not b_path.empty:
                    a_coeff = float(a_path['Estimate'].iloc[0])
                    b_coeff = float(b_path['Estimate'].iloc[0])
                    
                    # 간접효과 = a * b
                    indirect_effect = a_coeff * b_coeff
                    
                    indirect_effects['individual_paths'][mediator] = {
                        'a_path': a_coeff,
                        'b_path': b_coeff,
                        'indirect_effect': indirect_effect,
                        'a_se': float(a_path['Std. Error'].iloc[0]),
                        'b_se': float(b_path['Std. Error'].iloc[0])
                    }
                    
                    indirect_effects['total_indirect_effect'] += indirect_effect
                    indirect_effects['specific_indirect_effects'][f"via_{mediator}"] = indirect_effect
            
            # 다중 매개변수의 경우 순차적 간접효과도 계산
            if len(mediator_vars) > 1:
                indirect_effects['serial_indirect_effects'] = self._calculate_serial_indirect_effects(
                    independent_var, dependent_var, mediator_vars, params
                )
            
            logger.info(f"간접효과 계산 완료: 총 간접효과 = {indirect_effects['total_indirect_effect']:.4f}")
            return indirect_effects
            
        except Exception as e:
            logger.error(f"간접효과 계산 오류: {e}")
            return {}
    
    def _calculate_serial_indirect_effects(self,
                                         independent_var: str,
                                         dependent_var: str,
                                         mediator_vars: List[str],
                                         params: pd.DataFrame) -> Dict[str, float]:
        """순차적 간접효과 계산"""
        serial_effects = {}
        
        # 모든 가능한 순차적 경로 조합 계산
        for i in range(len(mediator_vars)):
            for j in range(i + 1, len(mediator_vars)):
                med1, med2 = mediator_vars[i], mediator_vars[j]
                
                # X -> M1 -> M2 -> Y 경로
                try:
                    # X -> M1
                    path1 = params[
                        (params['lval'] == med1) & 
                        (params['rval'] == independent_var) &
                        (params['op'] == '~')
                    ]
                    
                    # M1 -> M2
                    path2 = params[
                        (params['lval'] == med2) & 
                        (params['rval'] == med1) &
                        (params['op'] == '~')
                    ]
                    
                    # M2 -> Y
                    path3 = params[
                        (params['lval'] == dependent_var) & 
                        (params['rval'] == med2) &
                        (params['op'] == '~')
                    ]
                    
                    if not path1.empty and not path2.empty and not path3.empty:
                        coeff1 = float(path1['Estimate'].iloc[0])
                        coeff2 = float(path2['Estimate'].iloc[0])
                        coeff3 = float(path3['Estimate'].iloc[0])
                        
                        serial_effect = coeff1 * coeff2 * coeff3
                        serial_effects[f"{independent_var}_via_{med1}_and_{med2}"] = serial_effect
                        
                except Exception as e:
                    logger.warning(f"순차적 간접효과 계산 오류 ({med1}->{med2}): {e}")
        
        return serial_effects
    
    def calculate_total_effects(self,
                              direct_effects: Dict[str, Any],
                              indirect_effects: Dict[str, Any]) -> Dict[str, Any]:
        """
        총효과 계산 (직접효과 + 간접효과)
        
        Args:
            direct_effects (Dict[str, Any]): 직접효과 결과
            indirect_effects (Dict[str, Any]): 간접효과 결과
            
        Returns:
            Dict[str, Any]: 총효과 결과
        """
        try:
            direct_coeff = direct_effects.get('coefficient', 0.0)
            indirect_coeff = indirect_effects.get('total_indirect_effect', 0.0)
            
            total_effect = direct_coeff + indirect_coeff
            
            total_effects = {
                'total_effect': total_effect,
                'direct_component': direct_coeff,
                'indirect_component': indirect_coeff,
                'proportion_mediated': indirect_coeff / total_effect if total_effect != 0 else 0.0
            }
            
            logger.info(f"총효과 계산 완료: {total_effect:.4f}")
            return total_effects
            
        except Exception as e:
            logger.error(f"총효과 계산 오류: {e}")
            return {}
    
    def analyze_mediation_effects(self,
                                independent_var: str,
                                dependent_var: str,
                                mediator_vars: List[str]) -> Dict[str, Any]:
        """
        매개효과 분석 (Sobel test, Bootstrap CI 등)
        
        Args:
            independent_var (str): 독립변수
            dependent_var (str): 종속변수
            mediator_vars (List[str]): 매개변수들
            
        Returns:
            Dict[str, Any]: 매개효과 분석 결과
        """
        try:
            mediation_results = {
                'sobel_tests': {},
                'bootstrap_results': {},
                'mediation_type': self._determine_mediation_type(
                    independent_var, dependent_var, mediator_vars
                )
            }
            
            # 각 매개변수별 Sobel test
            for mediator in mediator_vars:
                sobel_result = self._sobel_test(independent_var, mediator, dependent_var)
                mediation_results['sobel_tests'][mediator] = sobel_result
            
            # Bootstrap 신뢰구간 (구현 예정)
            # mediation_results['bootstrap_results'] = self._bootstrap_mediation_ci(
            #     independent_var, dependent_var, mediator_vars
            # )
            
            logger.info("매개효과 분석 완료")
            return mediation_results
            
        except Exception as e:
            logger.error(f"매개효과 분석 오류: {e}")
            return {}
    
    def _sobel_test(self, independent_var: str, mediator_var: str, dependent_var: str) -> Dict[str, float]:
        """Sobel test 수행"""
        try:
            params = self.model.inspect()
            
            # a path (X -> M)
            a_path = params[
                (params['lval'] == mediator_var) & 
                (params['rval'] == independent_var) &
                (params['op'] == '~')
            ]
            
            # b path (M -> Y)
            b_path = params[
                (params['lval'] == dependent_var) & 
                (params['rval'] == mediator_var) &
                (params['op'] == '~')
            ]
            
            if a_path.empty or b_path.empty:
                return {'z_score': np.nan, 'p_value': np.nan}
            
            a = float(a_path['Estimate'].iloc[0])
            se_a = float(a_path['Std. Error'].iloc[0])
            b = float(b_path['Estimate'].iloc[0])
            se_b = float(b_path['Std. Error'].iloc[0])
            
            # Sobel test 통계량
            sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
            z_score = (a * b) / sobel_se
            
            # p-value (양측검정)
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            return {
                'z_score': z_score,
                'p_value': p_value,
                'indirect_effect': a * b,
                'standard_error': sobel_se
            }
            
        except Exception as e:
            logger.warning(f"Sobel test 오류: {e}")
            return {'z_score': np.nan, 'p_value': np.nan}
    
    def _determine_mediation_type(self,
                                independent_var: str,
                                dependent_var: str,
                                mediator_vars: List[str]) -> str:
        """매개효과 유형 판단"""
        try:
            # 직접효과와 간접효과 계산
            direct_effects = self.calculate_direct_effects(independent_var, dependent_var)
            indirect_effects = self.calculate_indirect_effects(independent_var, dependent_var, mediator_vars)
            
            direct_coeff = direct_effects.get('coefficient', 0.0)
            direct_p = direct_effects.get('p_value', 1.0)
            indirect_coeff = indirect_effects.get('total_indirect_effect', 0.0)
            
            # 간접효과의 유의성 (Sobel test 기준)
            indirect_significant = False
            for mediator in mediator_vars:
                sobel_result = self._sobel_test(independent_var, mediator, dependent_var)
                if sobel_result.get('p_value', 1.0) < 0.05:
                    indirect_significant = True
                    break
            
            # 매개효과 유형 판단
            if indirect_significant:
                if direct_p < 0.05:
                    return "partial_mediation"  # 부분매개
                else:
                    return "full_mediation"     # 완전매개
            else:
                return "no_mediation"           # 매개효과 없음
                
        except Exception as e:
            logger.warning(f"매개효과 유형 판단 오류: {e}")
            return "unknown"


# 편의 함수들
def calculate_direct_effects(model: Model, 
                           independent_var: str,
                           dependent_var: str) -> Dict[str, Any]:
    """직접효과 계산 편의 함수"""
    calculator = EffectsCalculator(model)
    return calculator.calculate_direct_effects(independent_var, dependent_var)


def calculate_indirect_effects(model: Model,
                             independent_var: str,
                             dependent_var: str,
                             mediator_vars: List[str]) -> Dict[str, Any]:
    """간접효과 계산 편의 함수"""
    calculator = EffectsCalculator(model)
    return calculator.calculate_indirect_effects(independent_var, dependent_var, mediator_vars)


def calculate_total_effects(model: Model,
                          independent_var: str,
                          dependent_var: str,
                          mediator_vars: Optional[List[str]] = None) -> Dict[str, Any]:
    """총효과 계산 편의 함수"""
    calculator = EffectsCalculator(model)
    return calculator.calculate_all_effects(independent_var, dependent_var, mediator_vars)


def analyze_mediation_effects(model: Model,
                            independent_var: str,
                            dependent_var: str,
                            mediator_vars: List[str]) -> Dict[str, Any]:
    """매개효과 분석 편의 함수"""
    calculator = EffectsCalculator(model)
    return calculator.analyze_mediation_effects(independent_var, dependent_var, mediator_vars)
