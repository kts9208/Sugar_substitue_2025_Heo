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
from typing import Dict, Optional
from scipy import optimize
import logging

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
    
    def estimate(self, data: pd.DataFrame,
                measurement_model,
                structural_model,
                choice_model,
                log_file: Optional[str] = None,
                **kwargs) -> Dict:
        """
        ICLV 모델 순차 추정
        
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
        self.logger.info("ICLV 모델 순차 추정 시작")
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
                        factor_scores: np.ndarray) -> Dict:
        """
        선택모델 추정

        Probit 또는 Logit 모델: P(Choice|X, LV)

        Args:
            data: 전체 데이터
            choice_model: 선택모델 객체
            factor_scores: 요인점수

        Returns:
            선택모델 추정 결과
        """
        # 데이터에 요인점수 추가
        individual_ids = data[self.config.individual_id_column].unique()

        # 요인점수를 데이터프레임에 매핑
        factor_score_dict = {
            ind_id: factor_scores[i]
            for i, ind_id in enumerate(individual_ids)
        }

        data_with_lv = data.copy()
        data_with_lv['latent_variable'] = data_with_lv[self.config.individual_id_column].map(
            factor_score_dict
        )

        # 선택모델 추정 (간단한 최적화)
        initial_params = self.initializer._initialize_choice(choice_model)

        def negative_ll(params):
            param_dict = self._unpack_choice_params(params, choice_model)
            ll = 0.0

            for ind_id in individual_ids:
                ind_data = data_with_lv[data_with_lv[self.config.individual_id_column] == ind_id]
                lv = factor_score_dict[ind_id]

                for idx in range(len(ind_data)):
                    ll += choice_model.log_likelihood(
                        ind_data.iloc[idx:idx+1],
                        lv,
                        param_dict
                    )

            return -ll

        result = optimize.minimize(
            negative_ll,
            initial_params,
            method='BFGS',
            options={'maxiter': 1000}
        )

        return {
            'params': result.x,
            'log_likelihood': -result.fun,
            'success': result.success,
            'n_iterations': result.nit,
            'full_results': result
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
        all_params = np.concatenate([
            self.measurement_results['params'].flatten()
            if isinstance(self.measurement_results['params'], np.ndarray)
            else np.array(list(self.measurement_results['params'].values())).flatten(),

            self.structural_results['params'].flatten()
            if isinstance(self.structural_results['params'], np.ndarray)
            else np.array(list(self.structural_results['params'].values())).flatten(),

            self.choice_results['params'].flatten()
        ])

        # 전체 로그우도 (각 단계 합산)
        total_ll = (
            self.measurement_results['log_likelihood'] +
            self.structural_results['log_likelihood'] +
            self.choice_results['log_likelihood']
        )

        # 전체 반복 횟수
        total_iterations = (
            self.measurement_results['n_iterations'] +
            self.structural_results['n_iterations'] +
            self.choice_results['n_iterations']
        )

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

