"""
Sequential Estimation for ICLV Models

ICLV 모델의 순차 추정 엔진입니다.
측정모델, 구조모델, 선택모델을 순차적으로 추정합니다.

단일책임 원칙:
- 각 모델을 독립적으로 추정
- 이전 단계 결과를 다음 단계에 전달
- 최종 결과 통합

참조:
- Ben-Akiva et al. (2002) - Sequential estimation approach
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from scipy import optimize
import logging
import pickle
from pathlib import Path

from .base_estimator import BaseEstimator
from .initializer import SequentialInitializer
from .likelihood_calculator import SequentialLikelihoodCalculator
from .sem_estimator import SEMEstimator

logger = logging.getLogger(__name__)


class SequentialEstimator(BaseEstimator):
    """
    ICLV 모델 순차 추정기
    
    3단계 순차 추정:
    1. 측정모델 추정 → 요인점수 추출
    2. 구조모델 추정 (요인점수 사용)
    3. 선택모델 추정 (요인점수 사용)
    
    장점:
    - 계산 효율성 (시뮬레이션 불필요)
    - 각 단계별 해석 용이
    - 초기값 설정 간단
    
    단점:
    - 측정오차 전파 (measurement error propagation)
    - 동시추정보다 효율성 낮음
    """
    
    def __init__(self, config):
        """
        Args:
            config: ICLVConfig 또는 MultiLatentConfig
        """
        super().__init__(config)
        
        # 순차추정 전용 컴포넌트
        self.initializer = SequentialInitializer(config)
        self.likelihood_calculator = SequentialLikelihoodCalculator(
            config, 
            config.individual_id_column
        )
        
        # 단계별 결과 저장
        self.measurement_results = None
        self.structural_results = None
        self.choice_results = None
        self.factor_scores = None
    
    # ========================================================================
    # 단계별 실행 메서드 (Stage-wise Execution)
    # ========================================================================

    def estimate_stage1_only(self,
                            data: pd.DataFrame,
                            measurement_model,
                            structural_model,
                            save_path: Optional[str] = None,
                            log_file: Optional[str] = None,
                            **kwargs) -> Dict:
        """
        1단계만 실행: 측정모델 + 구조모델 통합 추정 (SEM)

        잠재변수 간 관계를 확인하고 요인점수를 추출합니다.
        결과를 파일로 저장하여 나중에 2단계에서 재사용할 수 있습니다.

        Args:
            data: 통합 데이터
            measurement_model: 측정모델 객체 (MultiLatentMeasurement)
            structural_model: 구조모델 객체 (MultiLatentStructural)
            save_path: 결과 저장 경로 (None이면 저장 안 함)
            log_file: 로그 파일 경로
            **kwargs: 추가 인자

        Returns:
            {
                'sem_results': Dict,  # SEM 전체 결과
                'factor_scores': Dict[str, np.ndarray],  # 요인점수
                'paths': pd.DataFrame,  # 잠재변수 간 경로계수
                'loadings': pd.DataFrame,  # 요인적재량
                'fit_indices': Dict[str, float],  # 적합도 지수
                'log_likelihood': float,
                'save_path': str  # 저장 경로 (저장한 경우)
            }

        Example:
            >>> # 1단계 실행 및 저장
            >>> results = estimator.estimate_stage1_only(
            ...     data, measurement_model, structural_model,
            ...     save_path='results/stage1_results.pkl'
            ... )
            >>> print(results['paths'])  # 잠재변수 간 관계 확인
            >>>
            >>> # 나중에 2단계 실행
            >>> results2 = estimator.estimate_stage2_only(
            ...     data, choice_model,
            ...     factor_scores='results/stage1_results.pkl'
            ... )
        """
        self.logger.info("="*70)
        self.logger.info("1단계 추정 시작 (측정모델 + 구조모델)")
        self.logger.info("="*70)

        # 데이터 검증
        self._validate_data(data)
        self.data = data

        # 로그 설정
        if log_file:
            self._setup_iteration_logger(log_file)

        try:
            # SEM 추정
            self.logger.info("\n[SEM 추정] 측정모델 + 구조모델 통합...")
            sem_results = self._estimate_sem_model(
                data, measurement_model, structural_model
            )
            self.logger.info(
                f"SEM 추정 완료: LL = {sem_results['log_likelihood']:.2f}"
            )

            # 요인점수 추출
            self.factor_scores = sem_results['factor_scores']
            self.logger.info(f"요인점수 추출 완료: {list(self.factor_scores.keys())}")

            # 요인점수 로깅 (표준화 전)
            self._log_factor_scores(self.factor_scores, stage="SEM 추출 직후 (표준화 전)")

            # 요인점수 Z-score 표준화
            self.logger.info("\n요인점수 Z-score 표준화 적용...")
            self.factor_scores = self._standardize_factor_scores(self.factor_scores)
            self.logger.info("요인점수 표준화 완료")

            # 표준화 후 로깅
            self._log_factor_scores(self.factor_scores, stage="SEM 추출 직후 (표준화 후)")

            # 결과 저장
            self.measurement_results = {
                'params': sem_results['loadings'],
                'log_likelihood': sem_results['log_likelihood'],
                'success': True,
                'full_results': sem_results
            }
            self.structural_results = {
                'params': sem_results['paths'],
                'log_likelihood': sem_results['log_likelihood'],
                'success': True,
                'full_results': sem_results
            }

            # 반환 결과 구성
            stage1_results = {
                'sem_results': sem_results,
                'factor_scores': self.factor_scores,
                'paths': sem_results['paths'],
                'loadings': sem_results['loadings'],
                'fit_indices': sem_results['fit_indices'],
                'log_likelihood': sem_results['log_likelihood'],
                'measurement_results': self.measurement_results,
                'structural_results': self.structural_results
            }

            # 파일 저장 (옵션)
            if save_path:
                saved_path = self.save_stage1_results(stage1_results, save_path)
                stage1_results['save_path'] = saved_path
                self.logger.info(f"\n✅ 1단계 결과 저장 완료: {saved_path}")

            self.logger.info("\n" + "="*70)
            self.logger.info("1단계 추정 완료!")
            self.logger.info("="*70)

            return stage1_results

        finally:
            if log_file:
                self._close_iteration_logger()

    def estimate_stage2_only(self,
                            data: pd.DataFrame,
                            choice_model,
                            factor_scores: Union[Dict[str, np.ndarray], str],
                            log_file: Optional[str] = None,
                            **kwargs) -> Dict:
        """
        2단계만 실행: 선택모델 추정

        1단계에서 추출한 요인점수를 사용하여 선택모델을 추정합니다.
        요인점수는 딕셔너리 또는 파일 경로로 제공할 수 있습니다.

        Args:
            data: 통합 데이터
            choice_model: 선택모델 객체 (MultinomialLogitChoice)
            factor_scores: 요인점수 딕셔너리 또는 1단계 결과 파일 경로
            log_file: 로그 파일 경로
            **kwargs: 추가 인자

        Returns:
            {
                'params': Dict,  # 선택모델 파라미터
                'log_likelihood': float,
                'aic': float,
                'bic': float,
                'parameter_statistics': pd.DataFrame,
                'success': bool
            }

        Example:
            >>> # 방법 1: 메모리에서 직접 전달
            >>> results1 = estimator.estimate_stage1_only(...)
            >>> results2 = estimator.estimate_stage2_only(
            ...     data, choice_model, results1['factor_scores']
            ... )
            >>>
            >>> # 방법 2: 파일에서 로드
            >>> results2 = estimator.estimate_stage2_only(
            ...     data, choice_model,
            ...     factor_scores='results/stage1_results.pkl'
            ... )
        """
        self.logger.info("="*70)
        self.logger.info("2단계 추정 시작 (선택모델)")
        self.logger.info("="*70)

        # 데이터 검증
        self._validate_data(data)
        self.data = data

        # 로그 설정
        if log_file:
            self._setup_iteration_logger(log_file)

        try:
            # 요인점수 로드 (파일 경로인 경우)
            if isinstance(factor_scores, str):
                self.logger.info(f"\n요인점수 로드 중: {factor_scores}")
                loaded_results = self.load_stage1_results(factor_scores)
                self.factor_scores = loaded_results['factor_scores']
                self.logger.info(f"요인점수 로드 완료: {list(self.factor_scores.keys())}")
            else:
                self.factor_scores = factor_scores
                self.logger.info(f"요인점수 전달 완료: {list(self.factor_scores.keys())}")

            # 요인점수 로깅
            self._log_factor_scores(self.factor_scores, stage="선택모델 전달 직전")

            # 선택모델 추정
            self.logger.info("\n[선택모델 추정] 시작...")
            self.choice_results = self._estimate_choice(
                data, choice_model, self.factor_scores
            )
            self.logger.info(
                f"선택모델 추정 완료: LL = {self.choice_results['log_likelihood']:.2f}"
            )

            self.logger.info("\n" + "="*70)
            self.logger.info("2단계 추정 완료!")
            self.logger.info("="*70)

            return self.choice_results

        finally:
            if log_file:
                self._close_iteration_logger()

    def estimate(self, data: pd.DataFrame,
                measurement_model,
                structural_model,
                choice_model,
                log_file: Optional[str] = None,
                **kwargs) -> Dict:
        """
        ICLV 모델 순차 추정 (전체: 1단계 + 2단계)

        Args:
            data: 통합 데이터
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            log_file: 로그 파일 경로
            **kwargs: 추가 인자

        Returns:
            추정 결과 딕셔너리
        """
        self.logger.info("="*70)
        self.logger.info("ICLV 모델 순차 추정 시작 (전체)")
        self.logger.info("="*70)

        # 데이터 검증
        self._validate_data(data)
        self.data = data

        # 로그 설정
        if log_file:
            self._setup_iteration_logger(log_file)
        
        try:
            # 1단계: 측정모델 + 구조모델 통합 추정 (SEM 방식)
            self.logger.info("\n[1단계] SEM 추정 시작 (측정모델 + 구조모델 통합)...")
            sem_results = self._estimate_sem_model(
                data, measurement_model, structural_model
            )
            self.logger.info(
                f"SEM 추정 완료: LL = {sem_results['log_likelihood']:.2f}"
            )

            # 요인점수 추출 (SEM 결과에서)
            self.factor_scores = sem_results['factor_scores']
            self.logger.info(f"요인점수 추출 완료: {list(self.factor_scores.keys())}")

            # ✅ 요인점수 상세 로깅 (표준화 전)
            self._log_factor_scores(self.factor_scores, stage="SEM 추출 직후 (표준화 전)")

            # ✅ 요인점수 Z-score 표준화
            self.logger.info("\n요인점수 Z-score 표준화 적용...")
            self.factor_scores = self._standardize_factor_scores(self.factor_scores)
            self.logger.info("요인점수 표준화 완료")

            # ✅ 표준화 후 로깅
            self._log_factor_scores(self.factor_scores, stage="SEM 추출 직후 (표준화 후)")

            # 측정모델 및 구조모델 결과 저장
            self.measurement_results = {
                'params': sem_results['loadings'],
                'log_likelihood': sem_results['log_likelihood'],
                'success': True,
                'full_results': sem_results
            }
            self.structural_results = {
                'params': sem_results['paths'],
                'log_likelihood': sem_results['log_likelihood'],
                'success': True,
                'full_results': sem_results
            }
            
            # 2단계: 선택모델 추정
            self.logger.info("\n[2단계] 선택모델 추정 시작...")

            # ✅ 선택모델로 전달 직전 로깅
            self._log_factor_scores(self.factor_scores, stage="선택모델 전달 직전")

            self.choice_results = self._estimate_choice(
                data, choice_model, self.factor_scores
            )
            self.logger.info(
                f"선택모델 추정 완료: LL = {self.choice_results['log_likelihood']:.2f}"
            )
            
            # 결과 통합
            final_results = self._combine_results(
                measurement_model, structural_model, choice_model
            )
            
            self.logger.info("\n" + "="*70)
            self.logger.info("순차 추정 완료!")
            self.logger.info("="*70)
            
            return final_results
            
        finally:
            if log_file:
                self._close_iteration_logger()
    
    def _initialize_parameters(self, measurement_model, structural_model,
                              choice_model) -> Dict[str, np.ndarray]:
        """
        초기 파라미터 설정 (순차추정용)
        
        Returns:
            각 모델별 초기값 딕셔너리
        """
        return self.initializer.initialize(
            measurement_model, structural_model, choice_model
        )
    
    def _compute_log_likelihood(self, params: np.ndarray, *args) -> float:
        """
        로그우도 계산 (순차추정에서는 단계별로 계산)

        이 메서드는 순차추정에서 직접 사용되지 않습니다.
        각 단계별 메서드에서 개별적으로 우도를 계산합니다.
        """
        raise NotImplementedError(
            "순차추정에서는 _compute_log_likelihood를 직접 사용하지 않습니다. "
            "각 단계별 메서드를 사용하세요."
        )

    # ========================================================================
    # 1단계: SEM 추정 (측정모델 + 구조모델 통합)
    # ========================================================================

    def _estimate_sem_model(self, data: pd.DataFrame,
                           measurement_model,
                           structural_model) -> Dict:
        """
        SEM 방식으로 측정모델 + 구조모델 통합 추정

        semopy를 사용하여 CFA + 경로분석을 동시에 수행합니다.

        Args:
            data: 전체 데이터
            measurement_model: 측정모델 객체 (MultiLatentMeasurement)
            structural_model: 구조모델 객체 (MultiLatentStructural)

        Returns:
            {
                'model': semopy Model 객체,
                'factor_scores': Dict[str, np.ndarray],
                'params': pd.DataFrame,
                'loadings': pd.DataFrame,
                'paths': pd.DataFrame,
                'fit_indices': Dict[str, float],
                'log_likelihood': float
            }
        """
        self.logger.info("SEMEstimator를 사용한 통합 추정 시작")

        # SEMEstimator 생성
        sem_estimator = SEMEstimator()

        # SEM 추정 실행
        results = sem_estimator.fit(data, measurement_model, structural_model)

        # 요약 출력
        summary = sem_estimator.get_model_summary(measurement_model, structural_model)
        self.logger.info(f"\n{summary}")

        return results

    def _estimate_measurement(self, data: pd.DataFrame,
                             measurement_model) -> Dict:
        """
        측정모델 추정

        각 잠재변수별로 독립적으로 CFA 또는 Ordered Probit 추정

        Args:
            data: 전체 데이터
            measurement_model: 측정모델 객체

        Returns:
            측정모델 추정 결과
        """
        # 측정모델의 fit() 메서드 사용
        # (이미 구현되어 있음 - measurement_equations.py)
        results = measurement_model.fit(data)

        return {
            'params': results.get('params', results.get('parameters')),
            'log_likelihood': results.get('log_likelihood', 0.0),
            'success': results.get('success', True),
            'n_iterations': results.get('n_iterations', 0),
            'full_results': results
        }

    def _extract_factor_scores(self, data: pd.DataFrame,
                               measurement_model,
                               params: Dict) -> np.ndarray:
        """
        요인점수 추출

        측정모델 추정 결과로부터 각 개인의 잠재변수 값을 추출합니다.

        Args:
            data: 전체 데이터
            measurement_model: 측정모델 객체
            params: 측정모델 파라미터

        Returns:
            요인점수 배열 (n_individuals,) 또는 (n_individuals, n_latent_vars)
        """
        individual_ids = data[self.config.individual_id_column].unique()

        # 다중 잠재변수인 경우
        if hasattr(measurement_model, 'models'):
            n_lvs = len(measurement_model.models)
            factor_scores = np.zeros((len(individual_ids), n_lvs))

            for i, ind_id in enumerate(individual_ids):
                ind_data = data[data[self.config.individual_id_column] == ind_id]

                # 각 잠재변수별 요인점수 (간단히 지표 평균 사용)
                for j, (lv_name, model) in enumerate(measurement_model.models.items()):
                    indicators = model.config.indicators
                    available_indicators = [ind for ind in indicators if ind in ind_data.columns]

                    if available_indicators:
                        factor_scores[i, j] = ind_data[available_indicators].mean().mean()
                    else:
                        factor_scores[i, j] = 0.0
        else:
            # 단일 잠재변수
            factor_scores = np.zeros(len(individual_ids))

            for i, ind_id in enumerate(individual_ids):
                ind_data = data[data[self.config.individual_id_column] == ind_id]
                indicators = measurement_model.config.indicators
                available_indicators = [ind for ind in indicators if ind in ind_data.columns]

                if available_indicators:
                    factor_scores[i] = ind_data[available_indicators].mean().mean()
                else:
                    factor_scores[i] = 0.0

        return factor_scores

    # ========================================================================
    # 2단계: 구조모델 추정
    # ========================================================================

    def _estimate_structural(self, data: pd.DataFrame,
                            structural_model,
                            factor_scores: np.ndarray) -> Dict:
        """
        구조모델 추정

        OLS 회귀분석: LV = γ*X + ε

        Args:
            data: 전체 데이터
            structural_model: 구조모델 객체
            factor_scores: 요인점수

        Returns:
            구조모델 추정 결과
        """
        # 구조모델의 fit() 메서드 사용
        # (이미 구현되어 있음 - structural_equations.py)

        # 개인별 데이터로 변환
        individual_ids = data[self.config.individual_id_column].unique()
        ind_data_list = []

        for ind_id in individual_ids:
            ind_data = data[data[self.config.individual_id_column] == ind_id].iloc[0]
            ind_data_list.append(ind_data)

        ind_df = pd.DataFrame(ind_data_list)

        # 구조모델 추정
        results = structural_model.fit(ind_df, factor_scores)

        return {
            'params': results.get('gamma', results.get('parameters')),
            'log_likelihood': results.get('log_likelihood', 0.0),
            'success': True,
            'n_iterations': 0,
            'full_results': results
        }

    # ========================================================================
    # 3단계: 선택모델 추정
    # ========================================================================

    def _estimate_choice(self, data: pd.DataFrame,
                        choice_model,
                        factor_scores: Dict[str, np.ndarray]) -> Dict:
        """
        선택모델 추정

        MultinomialLogitChoice.fit() 메서드 사용

        Args:
            data: 전체 데이터
            choice_model: 선택모델 객체 (MultinomialLogitChoice)
            factor_scores: 요인점수 딕셔너리
                {
                    'purchase_intention': np.ndarray (n_individuals,),
                    'perceived_price': np.ndarray (n_individuals,),
                    'nutrition_knowledge': np.ndarray (n_individuals,),
                    ...
                }

        Returns:
            선택모델 추정 결과
        """
        # MultinomialLogitChoice.fit() 메서드 사용
        # 이 메서드는 요인점수를 받아서 선택모델을 추정합니다
        results = choice_model.fit(data, factor_scores)

        return {
            'params': results['params'],
            'log_likelihood': results['log_likelihood'],
            'aic': results.get('aic', None),
            'bic': results.get('bic', None),
            'success': results.get('success', True),
            'n_iterations': 0,  # MultinomialLogitChoice.fit()는 반복 횟수를 반환하지 않음
            'parameter_statistics': results.get('parameter_statistics', None),
            'full_results': results
        }

    # ========================================================================
    # 결과 통합
    # ========================================================================

    def _combine_results(self, measurement_model, structural_model,
                        choice_model) -> Dict:
        """
        각 단계의 결과를 통합

        Args:
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            통합 결과 딕셔너리
        """
        # 전체 파라미터 결합
        # SEM 결과는 DataFrame이므로 처리 방식이 다름
        measurement_params = self.measurement_results['params']
        if isinstance(measurement_params, pd.DataFrame):
            measurement_params = measurement_params['Estimate'].values
        elif isinstance(measurement_params, dict):
            measurement_params = np.array(list(measurement_params.values()))

        structural_params = self.structural_results['params']
        if isinstance(structural_params, pd.DataFrame):
            structural_params = structural_params['Estimate'].values
        elif isinstance(structural_params, dict):
            structural_params = np.array(list(structural_params.values()))

        choice_params = self.choice_results['params']
        if isinstance(choice_params, dict):
            # 딕셔너리의 값들을 flatten하여 결합
            choice_params_list = []
            for v in choice_params.values():
                if isinstance(v, np.ndarray):
                    choice_params_list.extend(v.flatten())
                else:
                    choice_params_list.append(v)
            choice_params = np.array(choice_params_list)

        all_params = np.concatenate([
            measurement_params.flatten(),
            structural_params.flatten(),
            choice_params.flatten()
        ])

        # 전체 로그우도 (각 단계 합산)
        # SEM은 측정모델과 구조모델을 통합 추정하므로 로그우도가 중복됨
        # 선택모델 로그우도만 별도로 계산됨
        total_ll = self.choice_results['log_likelihood']

        # 전체 반복 횟수
        total_iterations = self.choice_results.get('n_iterations', 0)

        # 수렴 여부
        success = (
            self.measurement_results['success'] and
            self.structural_results['success'] and
            self.choice_results['success']
        )

        # 파라미터 이름
        param_names = self.initializer.get_parameter_names(
            measurement_model, structural_model, choice_model
        )
        self.param_names = (
            param_names['measurement'] +
            param_names['structural'] +
            param_names['choice']
        )

        # 결과 딕셔너리 생성
        result = self._create_result_dict(
            params=all_params,
            log_likelihood=total_ll,
            n_iterations=total_iterations,
            success=success,
            parameters={
                'measurement': self.measurement_results['params'],
                'structural': self.structural_results['params'],
                'choice': self.choice_results['params']
            },
            factor_scores=self.factor_scores,
            stage_results={
                'measurement': self.measurement_results,
                'structural': self.structural_results,
                'choice': self.choice_results
            }
        )

        return result

    # ========================================================================
    # 유틸리티 메서드
    # ========================================================================

    def _unpack_choice_params(self, params: np.ndarray, choice_model) -> Dict:
        """
        선택모델 파라미터 언팩

        Args:
            params: 파라미터 벡터
            choice_model: 선택모델

        Returns:
            파라미터 딕셔너리
        """
        param_dict = {}
        idx = 0

        # intercept
        param_dict['intercept'] = params[idx]
        idx += 1

        # beta
        n_attrs = len(choice_model.choice_attributes)
        param_dict['beta'] = params[idx:idx+n_attrs]
        idx += n_attrs

        # lambda (조절효과 또는 기본)
        if hasattr(choice_model, 'moderation_enabled') and choice_model.moderation_enabled:
            param_dict['lambda_main'] = params[idx]
            idx += 1

            if hasattr(choice_model, 'moderator_lvs'):
                n_mods = len(choice_model.moderator_lvs)
                for i, mod_lv in enumerate(choice_model.moderator_lvs):
                    param_dict[f'lambda_mod_{mod_lv}'] = params[idx]
                    idx += 1
        else:
            param_dict['lambda'] = params[idx]
            idx += 1

        return param_dict

    def _standardize_factor_scores(self, factor_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        요인점수 Z-score 표준화

        각 잠재변수의 요인점수를 평균 0, 표준편차 1로 표준화합니다.

        z = (x - mean(x)) / std(x)

        Args:
            factor_scores: 원본 요인점수 딕셔너리
                {
                    'purchase_intention': np.ndarray (n_individuals,),
                    'perceived_price': np.ndarray (n_individuals,),
                    ...
                }

        Returns:
            표준화된 요인점수 딕셔너리 (동일한 구조)
        """
        standardized = {}

        self.logger.info("요인점수 Z-score 표준화:")
        self.logger.info(f"{'변수':30s} {'원본 평균':>12s} {'원본 std':>12s} → {'표준화 평균':>12s} {'표준화 std':>12s}")
        self.logger.info('-' * 90)

        for lv_name, scores in factor_scores.items():
            # 평균과 표준편차 계산
            mean = np.mean(scores)
            std = np.std(scores, ddof=0)  # 모집단 표준편차 (N으로 나눔)

            # Z-score 표준화
            if std > 1e-10:  # 표준편차가 0이 아닌 경우만
                standardized_scores = (scores - mean) / std
            else:
                self.logger.warning(f"  {lv_name}: 표준편차가 0에 가까워 표준화하지 않음")
                standardized_scores = scores - mean  # 평균만 제거

            standardized[lv_name] = standardized_scores

            # 검증
            new_mean = np.mean(standardized_scores)
            new_std = np.std(standardized_scores, ddof=0)

            self.logger.info(
                f'{lv_name:30s} {mean:>12.4f} {std:>12.4f} → {new_mean:>12.6f} {new_std:>12.6f}'
            )

        self.logger.info('-' * 90)
        self.logger.info("✅ 모든 요인점수가 평균 0, 표준편차 1로 표준화됨")

        return standardized

    def _log_factor_scores(self, factor_scores: Dict[str, np.ndarray], stage: str = ""):
        """
        요인점수 상세 로깅 및 파일 저장

        Args:
            factor_scores: 요인점수 딕셔너리
            stage: 로깅 단계 설명
        """
        import os
        from pathlib import Path

        self.logger.info("=" * 70)
        self.logger.info(f"요인점수 상세 정보 [{stage}]")
        self.logger.info("=" * 70)

        # 기본 통계
        for lv_name, scores in factor_scores.items():
            self.logger.info(f"\n{lv_name}:")
            self.logger.info(f"  Shape: {scores.shape}")
            self.logger.info(f"  Mean: {np.mean(scores):.4f}")
            self.logger.info(f"  Std: {np.std(scores):.4f}")
            self.logger.info(f"  Min: {np.min(scores):.4f}")
            self.logger.info(f"  Max: {np.max(scores):.4f}")
            self.logger.info(f"  First 5: {scores[:5]}")

            # NaN/Inf 체크
            n_nan = np.sum(np.isnan(scores))
            n_inf = np.sum(np.isinf(scores))
            if n_nan > 0 or n_inf > 0:
                self.logger.warning(f"  ⚠️ NaN: {n_nan}, Inf: {n_inf}")

        # 로그 파일로 저장 (부트스트랩 중에는 비활성화)
        # self.logger.info("\n파일 저장 시작...")
        # 디스크 공간 절약을 위해 파일 저장 비활성화

        self.logger.info("=" * 70)

    # ========================================================================
    # 결과 저장/로드 메서드 (Save/Load Methods)
    # ========================================================================

    @staticmethod
    def save_stage1_results(results: Dict, path: str) -> str:
        """
        1단계 결과를 파일로 저장

        요인점수와 SEM 결과를 pickle 형식으로 저장하고,
        경로계수, 요인적재량, 적합도 지수를 CSV로도 저장합니다.

        Args:
            results: estimate_stage1_only()의 반환값
            path: 저장 경로 (.pkl 확장자 권장)

        Returns:
            실제 저장된 파일 경로

        Example:
            >>> results = estimator.estimate_stage1_only(...)
            >>> saved_path = SequentialEstimator.save_stage1_results(
            ...     results, 'results/stage1_results.pkl'
            ... )
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 저장할 데이터 구성 (pickle 가능한 데이터만)
        save_data = {
            'factor_scores': results['factor_scores'],
            'paths': results['paths'],
            'loadings': results['loadings'],
            'fit_indices': results['fit_indices'],
            'log_likelihood': results['log_likelihood'],
            'version': '1.0'  # 버전 정보
        }

        # measurement_results와 structural_results에서 pickle 가능한 부분만 추출
        if 'measurement_results' in results and results['measurement_results']:
            save_data['measurement_results'] = {
                'params': results['measurement_results'].get('params'),
                'log_likelihood': results['measurement_results'].get('log_likelihood'),
                'success': results['measurement_results'].get('success')
            }

        if 'structural_results' in results and results['structural_results']:
            save_data['structural_results'] = {
                'params': results['structural_results'].get('params'),
                'log_likelihood': results['structural_results'].get('log_likelihood'),
                'success': results['structural_results'].get('success')
            }

        # 1. pickle로 저장 (요인점수 포함)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"1단계 결과 저장 완료 (pickle): {path}")
        logger.info(f"  - 요인점수: {list(save_data['factor_scores'].keys())}")
        logger.info(f"  - 파일 크기: {path.stat().st_size / 1024:.2f} KB")

        # 2. CSV 파일로도 저장
        base_path = path.with_suffix('')  # 확장자 제거

        # 2-1. 경로계수 저장
        if 'paths' in results and results['paths'] is not None:
            paths_csv = f"{base_path}_paths.csv"
            results['paths'].to_csv(paths_csv, index=False, encoding='utf-8-sig')
            logger.info(f"경로계수 저장: {paths_csv}")

        # 2-2. 요인적재량 저장
        if 'loadings' in results and results['loadings'] is not None:
            loadings_csv = f"{base_path}_loadings.csv"
            results['loadings'].to_csv(loadings_csv, index=False, encoding='utf-8-sig')
            logger.info(f"요인적재량 저장: {loadings_csv}")

        # 2-3. 적합도 지수 저장
        if 'fit_indices' in results and results['fit_indices']:
            fit_csv = f"{base_path}_fit_indices.csv"
            fit_df = pd.DataFrame([results['fit_indices']])
            fit_df.to_csv(fit_csv, index=False, encoding='utf-8-sig')
            logger.info(f"적합도 지수 저장: {fit_csv}")

        # 2-4. 요인점수 통계 저장
        if 'factor_scores' in results and results['factor_scores']:
            factor_stats_csv = f"{base_path}_factor_scores_stats.csv"
            stats_list = []
            for lv_name, scores in results['factor_scores'].items():
                stats_list.append({
                    'latent_variable': lv_name,
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'n_observations': len(scores)
                })
            stats_df = pd.DataFrame(stats_list)
            stats_df.to_csv(factor_stats_csv, index=False, encoding='utf-8-sig')
            logger.info(f"요인점수 통계 저장: {factor_stats_csv}")

        # 2-5. 요인점수 전체 저장 (개인별)
        if 'factor_scores' in results and results['factor_scores']:
            factor_scores_csv = f"{base_path}_factor_scores.csv"
            # 딕셔너리를 DataFrame으로 변환
            factor_scores_df = pd.DataFrame(results['factor_scores'])
            factor_scores_df.to_csv(factor_scores_csv, index=True, index_label='observation_id', encoding='utf-8-sig')
            logger.info(f"요인점수 전체 저장: {factor_scores_csv}")

        return str(path)

    @staticmethod
    def load_stage1_results(path: str) -> Dict:
        """
        1단계 결과를 파일에서 로드

        Args:
            path: 저장된 파일 경로

        Returns:
            {
                'factor_scores': Dict[str, np.ndarray],
                'paths': pd.DataFrame,
                'loadings': pd.DataFrame,
                'fit_indices': Dict,
                'log_likelihood': float,
                'measurement_results': Dict,
                'structural_results': Dict
            }

        Example:
            >>> results = SequentialEstimator.load_stage1_results(
            ...     'results/stage1_results.pkl'
            ... )
            >>> factor_scores = results['factor_scores']
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        # pickle로 로드
        with open(path, 'rb') as f:
            results = pickle.load(f)

        logger.info(f"1단계 결과 로드 완료: {path}")
        logger.info(f"  - 요인점수: {list(results['factor_scores'].keys())}")
        logger.info(f"  - 버전: {results.get('version', 'unknown')}")

        return results

    @staticmethod
    def save_factor_scores(factor_scores: Dict[str, np.ndarray], path: str) -> str:
        """
        요인점수만 별도로 저장 (경량 버전)

        Args:
            factor_scores: 요인점수 딕셔너리
            path: 저장 경로 (.npy 또는 .pkl)

        Returns:
            실제 저장된 파일 경로

        Example:
            >>> SequentialEstimator.save_factor_scores(
            ...     factor_scores, 'results/factor_scores.pkl'
            ... )
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == '.npy':
            # numpy 형식 (단일 배열로 변환 필요)
            # 다중 잠재변수는 pickle 사용 권장
            raise ValueError(
                ".npy 형식은 단일 배열만 지원합니다. "
                ".pkl 확장자를 사용하세요."
            )
        else:
            # pickle 형식
            with open(path, 'wb') as f:
                pickle.dump(factor_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"요인점수 저장 완료: {path}")
        logger.info(f"  - 변수: {list(factor_scores.keys())}")

        return str(path)

    @staticmethod
    def load_factor_scores(path: str) -> Dict[str, np.ndarray]:
        """
        요인점수만 별도로 로드

        Args:
            path: 저장된 파일 경로

        Returns:
            요인점수 딕셔너리

        Example:
            >>> factor_scores = SequentialEstimator.load_factor_scores(
            ...     'results/factor_scores.pkl'
            ... )
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        if path.suffix == '.npy':
            # numpy 형식
            factor_scores = np.load(path, allow_pickle=True).item()
        else:
            # pickle 형식
            with open(path, 'rb') as f:
                factor_scores = pickle.load(f)

        logger.info(f"요인점수 로드 완료: {path}")
        logger.info(f"  - 변수: {list(factor_scores.keys())}")

        return factor_scores

