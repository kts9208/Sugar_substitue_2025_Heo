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
# 기존 복잡한 병렬 처리 임포트 제거됨 - semopy 내장 기능 사용
import scipy.stats as stats

# tqdm 임포트 (선택적)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable
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
        self.data = None
        self.model_spec = None  # 모델 스펙 저장용
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def set_data(self, data):
        """
        분석에 사용할 데이터 설정

        Args:
            data: pandas DataFrame
        """
        self.data = data

    def set_model(self, model):
        """
        분석에 사용할 모델 설정

        Args:
            model: semopy Model 객체 또는 모델 스펙 문자열
        """
        if isinstance(model, str):
            # 모델 스펙 문자열인 경우 새 모델 생성
            self.model = Model(model)
            if self.data is not None:
                # 데이터가 있으면 즉시 적합
                self.model.fit(self.data)
        else:
            # 이미 적합된 모델 객체인 경우
            self.model = model
        
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

    def calculate_bootstrap_effects(self,
                                  independent_var: str,
                                  dependent_var: str,
                                  mediator_vars: Optional[List[str]] = None,
                                  n_bootstrap: int = 5000,
                                  confidence_level: float = 0.95,
                                  method: str = 'percentile',
                                  parallel: bool = True,
                                  n_jobs: int = -1,
                                  random_seed: Optional[int] = None,
                                  show_progress: bool = True) -> Dict[str, Any]:
        """
        semopy 기반 부트스트래핑을 사용한 효과 분석 및 신뢰구간 계산

        Args:
            independent_var (str): 독립변수
            dependent_var (str): 종속변수
            mediator_vars (Optional[List[str]]): 매개변수들
            n_bootstrap (int): 부트스트래핑 샘플 수
            confidence_level (float): 신뢰수준
            method (str): 신뢰구간 계산 방법 ('percentile', 'bias-corrected')
            parallel (bool): 병렬 처리 사용 여부 (현재 미구현)
            n_jobs (int): 병렬 처리 작업 수 (현재 미구현)
            random_seed (Optional[int]): 랜덤 시드
            show_progress (bool): 진행 상황 표시 여부

        Returns:
            Dict[str, Any]: 부트스트래핑 효과 분석 결과
        """
        if self.model is None or self.model_spec is None:
            raise ValueError("모델과 모델 스펙이 설정되지 않았습니다.")

        if self.data is None:
            raise ValueError("데이터가 설정되지 않았습니다.")

        logger.info(f"semopy 기반 부트스트래핑 효과 분석 시작: {independent_var} -> {dependent_var}")
        logger.info(f"부트스트래핑 샘플 수: {n_bootstrap}, 신뢰수준: {confidence_level}")

        # 랜덤 시드 설정
        if random_seed is not None:
            np.random.seed(random_seed)

        # 원본 효과 계산
        original_effects = self._calculate_path_effects_from_model(
            self.model, independent_var, dependent_var, mediator_vars
        )

        # semopy 내장 부트스트래핑 실행 (우선 시도)
        try:
            bootstrap_results = self._run_semopy_native_bootstrap_sampling(
                independent_var, dependent_var, mediator_vars,
                n_bootstrap, show_progress
            )
            logger.info("semopy 내장 부트스트래핑 사용")
        except Exception as e:
            logger.warning(f"semopy 내장 부트스트래핑 실패: {e}")
            logger.info("수동 semopy 부트스트래핑으로 전환")
            # 대안: 수동 semopy 부트스트래핑
            bootstrap_results = self._run_semopy_bootstrap_sampling(
                independent_var, dependent_var, mediator_vars,
                n_bootstrap, show_progress
            )

        # 신뢰구간 계산
        confidence_intervals = self._calculate_confidence_intervals(
            bootstrap_results, confidence_level, method
        )

        # 결과 통합
        results = {
            'original_effects': original_effects,
            'bootstrap_results': bootstrap_results,
            'confidence_intervals': confidence_intervals,
            'bootstrap_statistics': self._calculate_bootstrap_statistics(bootstrap_results),
            'settings': {
                'n_bootstrap': n_bootstrap,
                'confidence_level': confidence_level,
                'method': method,
                'random_seed': random_seed
            }
        }

        logger.info("부트스트래핑 효과 분석 완료")
        return results
    
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

    # 기존 복잡한 부트스트래핑 메서드 제거됨 - semopy 내장 기능으로 대체

    def _run_bootstrap_sampling_legacy(self,
                               independent_var: str,
                               dependent_var: str,
                               mediator_vars: Optional[List[str]],
                               n_bootstrap: int,
                               parallel: bool,
                               n_jobs: int,
                               show_progress: bool) -> Dict[str, List[float]]:
        """
        부트스트래핑 샘플링 실행

        Returns:
            Dict[str, List[float]]: 부트스트래핑 결과
        """
        logger.info("부트스트래핑 샘플링 시작")

        # 원본 데이터 준비
        original_data = self.data.copy()
        n_obs = len(original_data)

        # 결과 저장용 딕셔너리
        bootstrap_effects = {
            'direct_effects': [],
            'indirect_effects': [],
            'total_effects': []
        }

        if mediator_vars:
            for mediator in mediator_vars:
                bootstrap_effects[f'indirect_via_{mediator}'] = []

        # 부트스트래핑 함수 정의
        def single_bootstrap(seed_offset: int) -> Dict[str, float]:
            """단일 부트스트래핑 샘플 처리"""
            try:
                # 시드 설정 (각 샘플마다 다른 시드)
                np.random.seed(seed_offset)

                # 부트스트래핑 샘플 생성
                bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
                bootstrap_data = original_data.iloc[bootstrap_indices].reset_index(drop=True)

                # 임시 계산기 생성
                temp_calculator = EffectsCalculator()
                temp_calculator.set_data(bootstrap_data)

                # 모델 재추정 (저장된 모델 스펙 사용)
                if self.model_spec is not None:
                    model_spec = self.model_spec
                else:
                    # 대안: 기본 모델 스펙 (오류 방지용)
                    logger.warning("모델 스펙이 없어 부트스트래핑을 건너뜁니다.")
                    return {'direct_effect': 0.0, 'total_effect': 0.0}

                temp_calculator.set_model(model_spec)

                # 효과 계산
                effects = temp_calculator.calculate_all_effects(
                    independent_var, dependent_var, mediator_vars
                )

                # 결과 추출
                result = {
                    'direct_effect': effects.get('direct_effects', {}).get('coefficient', 0.0),
                    'total_effect': effects.get('total_effects', {}).get('total_effect', 0.0)
                }

                # 간접효과 추출
                indirect_effects = effects.get('indirect_effects', {})
                result['indirect_effect'] = indirect_effects.get('total_indirect_effect', 0.0)

                if mediator_vars:
                    individual_paths = indirect_effects.get('individual_paths', {})
                    for mediator in mediator_vars:
                        if mediator in individual_paths:
                            result[f'indirect_via_{mediator}'] = individual_paths[mediator].get('indirect_effect', 0.0)
                        else:
                            result[f'indirect_via_{mediator}'] = 0.0

                return result

            except Exception as e:
                logger.warning(f"부트스트래핑 샘플 {seed_offset} 처리 중 오류: {e}")
                # 오류 발생 시 0으로 반환
                result = {'direct_effect': 0.0, 'indirect_effect': 0.0, 'total_effect': 0.0}
                if mediator_vars:
                    for mediator in mediator_vars:
                        result[f'indirect_via_{mediator}'] = 0.0
                return result

        # 병렬 처리 또는 순차 처리
        if parallel and n_jobs != 1:
            # 병렬 처리
            if n_jobs == -1:
                n_jobs = mp.cpu_count()

            logger.info(f"병렬 처리로 부트스트래핑 실행 (작업 수: {n_jobs})")

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # 진행 상황 표시
                if show_progress:
                    futures = [executor.submit(single_bootstrap, i) for i in range(n_bootstrap)]
                    results = []
                    for future in tqdm(as_completed(futures), total=n_bootstrap, desc="Bootstrap"):
                        results.append(future.result())
                else:
                    futures = [executor.submit(single_bootstrap, i) for i in range(n_bootstrap)]
                    results = [future.result() for future in as_completed(futures)]
        else:
            # 순차 처리
            logger.info("순차 처리로 부트스트래핑 실행")
            results = []
            iterator = tqdm(range(n_bootstrap), desc="Bootstrap") if show_progress else range(n_bootstrap)
            for i in iterator:
                results.append(single_bootstrap(i))

        # 결과 정리
        for result in results:
            bootstrap_effects['direct_effects'].append(result['direct_effect'])
            bootstrap_effects['indirect_effects'].append(result['indirect_effect'])
            bootstrap_effects['total_effects'].append(result['total_effect'])

            if mediator_vars:
                for mediator in mediator_vars:
                    bootstrap_effects[f'indirect_via_{mediator}'].append(
                        result.get(f'indirect_via_{mediator}', 0.0)
                    )

        logger.info(f"부트스트래핑 샘플링 완료: {len(results)}개 샘플")
        return bootstrap_effects

    def _run_semopy_bootstrap_sampling(self,
                                     independent_var: str,
                                     dependent_var: str,
                                     mediator_vars: Optional[List[str]],
                                     n_bootstrap: int,
                                     show_progress: bool) -> Dict[str, List[float]]:
        """
        semopy 기반 효율적인 부트스트래핑 샘플링 실행

        Args:
            independent_var: 독립변수
            dependent_var: 종속변수
            mediator_vars: 매개변수들
            n_bootstrap: 부트스트래핑 샘플 수
            show_progress: 진행 상황 표시 여부

        Returns:
            Dict[str, List[float]]: 부트스트래핑 결과
        """
        from semopy import Model

        logger.info("semopy 기반 부트스트래핑 샘플링 시작")

        # 원본 데이터 준비
        original_data = self.data.copy()
        n_obs = len(original_data)

        # 결과 저장용 딕셔너리
        bootstrap_effects = {
            'direct_effects': [],
            'indirect_effects': [],
            'total_effects': []
        }

        if mediator_vars:
            for mediator in mediator_vars:
                bootstrap_effects[f'indirect_via_{mediator}'] = []

        # 진행 상황 표시
        iterator = tqdm(range(n_bootstrap), desc="semopy Bootstrap") if show_progress else range(n_bootstrap)

        successful_samples = 0

        for i in iterator:
            try:
                # 부트스트래핑 샘플 생성
                bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
                bootstrap_data = original_data.iloc[bootstrap_indices].reset_index(drop=True)

                # semopy 모델 재적합
                bootstrap_model = Model(self.model_spec)
                bootstrap_model.fit(bootstrap_data)

                # 효과 계산
                effects = self._calculate_path_effects_from_model(
                    bootstrap_model, independent_var, dependent_var, mediator_vars
                )

                # 결과 저장
                bootstrap_effects['direct_effects'].append(effects.get('direct_effect', 0.0))
                bootstrap_effects['indirect_effects'].append(effects.get('indirect_effect', 0.0))
                bootstrap_effects['total_effects'].append(effects.get('total_effect', 0.0))

                if mediator_vars:
                    for mediator in mediator_vars:
                        bootstrap_effects[f'indirect_via_{mediator}'].append(
                            effects.get(f'indirect_via_{mediator}', 0.0)
                        )

                successful_samples += 1

            except Exception as e:
                logger.warning(f"부트스트래핑 샘플 {i} 처리 중 오류: {e}")
                # 실패한 샘플에 대해 0 값 추가
                bootstrap_effects['direct_effects'].append(0.0)
                bootstrap_effects['indirect_effects'].append(0.0)
                bootstrap_effects['total_effects'].append(0.0)

                if mediator_vars:
                    for mediator in mediator_vars:
                        bootstrap_effects[f'indirect_via_{mediator}'].append(0.0)

        logger.info(f"semopy 부트스트래핑 샘플링 완료: {successful_samples}/{n_bootstrap}개 성공")
        return bootstrap_effects

    def _run_semopy_native_bootstrap_sampling(self,
                                            independent_var: str,
                                            dependent_var: str,
                                            mediator_vars: Optional[List[str]],
                                            n_bootstrap: int,
                                            show_progress: bool) -> Dict[str, List[float]]:
        """
        semopy 내장 부트스트래핑 기능을 활용한 효율적인 샘플링

        Args:
            independent_var: 독립변수
            dependent_var: 종속변수
            mediator_vars: 매개변수들
            n_bootstrap: 부트스트래핑 샘플 수
            show_progress: 진행 상황 표시 여부

        Returns:
            Dict[str, List[float]]: 부트스트래핑 결과
        """
        from semopy import Model, bias_correction
        from semopy.model_generation import generate_data
        from copy import deepcopy

        logger.info("semopy 내장 부트스트래핑 기능 활용 시작")

        # 결과 저장용 딕셔너리
        bootstrap_effects = {
            'direct_effects': [],
            'indirect_effects': [],
            'total_effects': []
        }

        if mediator_vars:
            for mediator in mediator_vars:
                bootstrap_effects[f'indirect_via_{mediator}'] = []

        # 원본 모델 복사
        original_model = deepcopy(self.model)

        # 진행 상황 표시
        iterator = tqdm(range(n_bootstrap), desc="semopy Native Bootstrap") if show_progress else range(n_bootstrap)

        successful_samples = 0

        for i in iterator:
            try:
                # semopy의 generate_data 함수를 사용하여 부트스트래핑 데이터 생성
                bootstrap_data = generate_data(original_model, n=len(self.data))

                # 새 모델로 재적합
                bootstrap_model = Model(self.model_spec)
                bootstrap_model.fit(bootstrap_data)

                # 효과 계산
                effects = self._calculate_path_effects_from_model(
                    bootstrap_model, independent_var, dependent_var, mediator_vars
                )

                # 결과 저장
                bootstrap_effects['direct_effects'].append(effects.get('direct_effect', 0.0))
                bootstrap_effects['indirect_effects'].append(effects.get('indirect_effect', 0.0))
                bootstrap_effects['total_effects'].append(effects.get('total_effect', 0.0))

                if mediator_vars:
                    for mediator in mediator_vars:
                        bootstrap_effects[f'indirect_via_{mediator}'].append(
                            effects.get(f'indirect_via_{mediator}', 0.0)
                        )

                successful_samples += 1

            except Exception as e:
                logger.warning(f"semopy 내장 부트스트래핑 샘플 {i} 처리 중 오류: {e}")
                # 실패한 샘플에 대해 0 값 추가
                bootstrap_effects['direct_effects'].append(0.0)
                bootstrap_effects['indirect_effects'].append(0.0)
                bootstrap_effects['total_effects'].append(0.0)

                if mediator_vars:
                    for mediator in mediator_vars:
                        bootstrap_effects[f'indirect_via_{mediator}'].append(0.0)

        logger.info(f"semopy 내장 부트스트래핑 완료: {successful_samples}/{n_bootstrap}개 성공")
        return bootstrap_effects

    def _calculate_path_effects_from_model(self,
                                         model,
                                         independent_var: str,
                                         dependent_var: str,
                                         mediator_vars: Optional[List[str]] = None) -> Dict[str, float]:
        """
        semopy 모델에서 경로 효과 추출

        Args:
            model: 적합된 semopy 모델
            independent_var: 독립변수
            dependent_var: 종속변수
            mediator_vars: 매개변수들

        Returns:
            Dict[str, float]: 계산된 효과들
        """
        try:
            # 모델 파라미터 추출
            params = model.inspect()

            # 직접효과 계산 (independent_var -> dependent_var)
            direct_effect = 0.0
            direct_path = params[
                (params['lval'] == dependent_var) &
                (params['op'] == '~') &
                (params['rval'] == independent_var)
            ]

            if len(direct_path) > 0:
                direct_effect = float(direct_path['Estimate'].iloc[0])

            # 간접효과 계산 (매개변수를 통한 효과)
            indirect_effect = 0.0
            indirect_effects_by_mediator = {}

            if mediator_vars:
                for mediator in mediator_vars:
                    # independent_var -> mediator 경로
                    path_a = params[
                        (params['lval'] == mediator) &
                        (params['op'] == '~') &
                        (params['rval'] == independent_var)
                    ]

                    # mediator -> dependent_var 경로
                    path_b = params[
                        (params['lval'] == dependent_var) &
                        (params['op'] == '~') &
                        (params['rval'] == mediator)
                    ]

                    if len(path_a) > 0 and len(path_b) > 0:
                        a_coef = float(path_a['Estimate'].iloc[0])
                        b_coef = float(path_b['Estimate'].iloc[0])
                        mediator_effect = a_coef * b_coef

                        indirect_effects_by_mediator[f'indirect_via_{mediator}'] = mediator_effect
                        indirect_effect += mediator_effect

            # 총효과 계산
            total_effect = direct_effect + indirect_effect

            # 결과 반환
            result = {
                'direct_effect': direct_effect,
                'indirect_effect': indirect_effect,
                'total_effect': total_effect
            }

            # 매개변수별 간접효과 추가
            result.update(indirect_effects_by_mediator)

            return result

        except Exception as e:
            logger.error(f"모델에서 효과 계산 오류: {e}")
            return {
                'direct_effect': 0.0,
                'indirect_effect': 0.0,
                'total_effect': 0.0
            }

    def _calculate_confidence_intervals(self,
                                      bootstrap_results: Dict[str, List[float]],
                                      confidence_level: float,
                                      method: str) -> Dict[str, Dict[str, float]]:
        """
        부트스트래핑 결과로부터 신뢰구간 계산

        Args:
            bootstrap_results: 부트스트래핑 결과
            confidence_level: 신뢰수준
            method: 신뢰구간 계산 방법

        Returns:
            Dict[str, Dict[str, float]]: 신뢰구간 결과
        """
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        confidence_intervals = {}

        for effect_type, values in bootstrap_results.items():
            if not values:
                continue

            values_array = np.array(values)

            if method == 'percentile':
                # 백분위수 방법
                lower_ci = np.percentile(values_array, lower_percentile)
                upper_ci = np.percentile(values_array, upper_percentile)

            elif method == 'bias-corrected' or method == 'bias_corrected':
                # 편향 보정 방법 (BCa의 간단한 버전)
                # 편향 보정 계수 계산
                original_estimate = np.mean(values_array)  # 원본 추정치 대신 평균 사용
                bias_correction = stats.norm.ppf((np.sum(values_array < original_estimate)) / len(values_array))

                # 보정된 백분위수 계산
                z_alpha_2 = stats.norm.ppf(alpha / 2)
                z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

                corrected_lower = stats.norm.cdf(2 * bias_correction + z_alpha_2)
                corrected_upper = stats.norm.cdf(2 * bias_correction + z_1_alpha_2)

                # 백분위수로 변환
                corrected_lower = max(0, min(100, corrected_lower * 100))
                corrected_upper = max(0, min(100, corrected_upper * 100))

                lower_ci = np.percentile(values_array, corrected_lower)
                upper_ci = np.percentile(values_array, corrected_upper)

            else:  # 기본값: percentile
                lower_ci = np.percentile(values_array, lower_percentile)
                upper_ci = np.percentile(values_array, upper_percentile)

            confidence_intervals[effect_type] = {
                'lower': lower_ci,
                'upper': upper_ci,
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'significant': not (lower_ci <= 0 <= upper_ci)  # 0을 포함하지 않으면 유의함
            }

        return confidence_intervals

    def _calculate_bootstrap_statistics(self,
                                      bootstrap_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        부트스트래핑 결과의 기술통계 계산

        Args:
            bootstrap_results: 부트스트래핑 결과

        Returns:
            Dict[str, Dict[str, float]]: 기술통계 결과
        """
        statistics = {}

        for effect_type, values in bootstrap_results.items():
            if not values:
                continue

            values_array = np.array(values)

            statistics[effect_type] = {
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'median': np.median(values_array),
                'q25': np.percentile(values_array, 25),
                'q75': np.percentile(values_array, 75),
                'skewness': stats.skew(values_array),
                'kurtosis': stats.kurtosis(values_array)
            }

        return statistics

    def analyze_all_possible_mediations(self,
                                      variables: List[str],
                                      bootstrap_samples: int = 5000,
                                      confidence_level: float = 0.95,
                                      parallel: bool = True,
                                      show_progress: bool = True) -> Dict[str, Any]:
        """
        5개 요인 간 모든 가능한 매개효과 분석

        Args:
            variables (List[str]): 분석할 변수들 (5개)
            bootstrap_samples (int): 부트스트래핑 샘플 수
            confidence_level (float): 신뢰수준
            parallel (bool): 병렬 처리 사용 여부
            show_progress (bool): 진행 상황 표시 여부

        Returns:
            Dict[str, Any]: 모든 매개효과 분석 결과
        """
        if len(variables) != 5:
            raise ValueError("정확히 5개의 변수가 필요합니다.")

        logger.info(f"5개 요인 간 모든 가능한 매개효과 분석 시작: {variables}")

        all_mediation_results = {}
        total_combinations = 0

        # 모든 가능한 X -> M -> Y 조합 생성
        for i, independent_var in enumerate(variables):
            for j, dependent_var in enumerate(variables):
                if i == j:  # 같은 변수는 제외
                    continue

                # 나머지 변수들을 매개변수로 사용
                potential_mediators = [v for k, v in enumerate(variables) if k != i and k != j]

                for mediator in potential_mediators:
                    total_combinations += 1

                    combination_key = f"{independent_var}_to_{dependent_var}_via_{mediator}"

                    try:
                        # 매개효과 분석
                        mediation_result = self.calculate_bootstrap_effects(
                            independent_var=independent_var,
                            dependent_var=dependent_var,
                            mediator_vars=[mediator],
                            n_bootstrap=bootstrap_samples,
                            confidence_level=confidence_level,
                            parallel=parallel,
                            show_progress=False  # 개별 진행 상황은 표시하지 않음
                        )

                        # 매개효과 유의성 판단
                        indirect_ci = mediation_result['confidence_intervals'].get('indirect_effects', {})
                        is_significant = indirect_ci.get('significant', False)

                        all_mediation_results[combination_key] = {
                            'independent_var': independent_var,
                            'dependent_var': dependent_var,
                            'mediator': mediator,
                            'mediation_result': mediation_result,
                            'is_significant': is_significant,
                            'indirect_effect_mean': indirect_ci.get('mean', 0.0),
                            'indirect_effect_ci': [indirect_ci.get('lower_ci', 0.0), indirect_ci.get('upper_ci', 0.0)]
                        }

                        if show_progress:
                            logger.info(f"완료: {combination_key} (유의함: {is_significant})")

                    except Exception as e:
                        logger.warning(f"매개효과 분석 실패: {combination_key} - {e}")
                        all_mediation_results[combination_key] = {
                            'independent_var': independent_var,
                            'dependent_var': dependent_var,
                            'mediator': mediator,
                            'error': str(e),
                            'is_significant': False
                        }

        # 결과 요약
        significant_mediations = {k: v for k, v in all_mediation_results.items()
                                if v.get('is_significant', False)}

        summary = {
            'total_combinations_tested': total_combinations,
            'significant_mediations_count': len(significant_mediations),
            'significance_rate': len(significant_mediations) / total_combinations if total_combinations > 0 else 0,
            'variables_analyzed': variables,
            'settings': {
                'bootstrap_samples': bootstrap_samples,
                'confidence_level': confidence_level,
                'parallel': parallel
            }
        }

        logger.info(f"모든 매개효과 분석 완료: {total_combinations}개 조합 중 {len(significant_mediations)}개 유의함")

        return {
            'all_results': all_mediation_results,
            'significant_results': significant_mediations,
            'summary': summary
        }
    
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


def calculate_bootstrap_effects(model: Model, data: pd.DataFrame,
                              independent_var: str, dependent_var: str,
                              mediator_vars: Optional[List[str]] = None,
                              n_bootstrap: int = 5000,
                              confidence_level: float = 0.95,
                              method: str = 'bias-corrected',
                              parallel: bool = True,
                              random_seed: Optional[int] = None) -> Dict[str, Any]:
    """부트스트래핑 효과 분석 편의 함수"""
    calculator = EffectsCalculator(model)
    calculator.set_data(data)
    return calculator.calculate_bootstrap_effects(
        independent_var, dependent_var, mediator_vars,
        n_bootstrap, confidence_level, method, parallel,
        n_jobs=-1, random_seed=random_seed, show_progress=True
    )


def analyze_all_possible_mediations(model: Model, data: pd.DataFrame,
                                  variables: List[str],
                                  bootstrap_samples: int = 5000,
                                  confidence_level: float = 0.95,
                                  parallel: bool = True) -> Dict[str, Any]:
    """5개 요인 간 모든 가능한 매개효과 분석 편의 함수"""
    calculator = EffectsCalculator(model)
    calculator.set_data(data)
    return calculator.analyze_all_possible_mediations(
        variables, bootstrap_samples, confidence_level, parallel, show_progress=True
    )
