"""
Multinomial Logit Model 추정 모듈

이 모듈은 StatsModels를 사용하여 Multinomial Logit Model을 추정하고
결과를 처리하는 재사용 가능한 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from scipy import stats
import warnings

# StatsModels 임포트
try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import MNLogit
    from statsmodels.tools import add_constant
except ImportError as e:
    logger.error("StatsModels 라이브러리를 찾을 수 없습니다. pip install statsmodels로 설치해주세요.")
    raise e

from .model_config import ModelConfig, ModelConfigManager

logger = logging.getLogger(__name__)


class MultinomialLogitEstimator:
    """Multinomial Logit Model 추정을 담당하는 클래스"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        모델 추정기 초기화
        
        Args:
            config (Optional[ModelConfig]): 모델 설정 객체
        """
        self.config_manager = ModelConfigManager(config)
        self.model = None
        self.results = None
        self.fitted = False
    
    def prepare_data_for_statsmodels(self, X: np.ndarray, y: np.ndarray, 
                                   choice_sets: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
        """
        StatsModels MNLogit에 적합한 형태로 데이터를 변환
        
        Args:
            X (np.ndarray): 설명변수 행렬
            y (np.ndarray): 선택 결과 벡터
            choice_sets (np.ndarray): 선택 세트 인덱스
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (설명변수 DataFrame, 종속변수 Series)
        """
        logger.info("StatsModels용 데이터 변환 시작")
        
        # 데이터 유효성 검증
        if not self.config_manager.validate_input_data(X, y, choice_sets):
            raise ValueError("입력 데이터가 유효하지 않습니다")
        
        # DataFrame 생성
        feature_names = self.config_manager.config.feature_names
        if len(feature_names) != X.shape[1]:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            logger.warning("특성 이름이 설정되지 않아 기본 이름을 사용합니다")
        
        # 선택 세트 ID 추가
        choice_set_ids = []
        for i, start_idx in enumerate(choice_sets):
            if i < len(choice_sets) - 1:
                end_idx = choice_sets[i + 1]
            else:
                end_idx = len(y)
            
            choice_set_ids.extend([i] * (end_idx - start_idx))
        
        # DataFrame 생성
        df_X = pd.DataFrame(X, columns=feature_names)
        df_X['choice_set_id'] = choice_set_ids
        
        # 대안 ID 생성 (각 선택 세트 내에서 0, 1, 2, ...)
        alternative_ids = []
        for i, start_idx in enumerate(choice_sets):
            if i < len(choice_sets) - 1:
                end_idx = choice_sets[i + 1]
            else:
                end_idx = len(y)
            
            n_alternatives = end_idx - start_idx
            alternative_ids.extend(list(range(n_alternatives)))
        
        df_X['alternative_id'] = alternative_ids
        
        # 종속변수 (선택된 대안의 ID)
        y_choice = []
        for i, start_idx in enumerate(choice_sets):
            if i < len(choice_sets) - 1:
                end_idx = choice_sets[i + 1]
            else:
                end_idx = len(y)
            
            choice_in_set = y[start_idx:end_idx]
            chosen_alternative = np.where(choice_in_set == 1)[0]
            
            if len(chosen_alternative) == 1:
                y_choice.append(chosen_alternative[0])
            else:
                logger.warning(f"선택 세트 {i}에서 선택이 명확하지 않습니다")
                y_choice.append(0)  # 기본값
        
        # 선택 세트별로 데이터 복제 (각 대안마다 하나의 행)
        expanded_data = []
        expanded_y = []
        
        for choice_set_id in range(len(choice_sets)):
            choice_set_data = df_X[df_X['choice_set_id'] == choice_set_id].copy()
            
            for _, row in choice_set_data.iterrows():
                expanded_data.append(row.drop(['choice_set_id', 'alternative_id']))
                expanded_y.append(y_choice[choice_set_id])
        
        final_X = pd.DataFrame(expanded_data).reset_index(drop=True)
        final_y = pd.Series(expanded_y)
        
        logger.info(f"데이터 변환 완료: {len(final_X)} 행, {len(final_X.columns)} 열")
        return final_X, final_y
    
    def fit(self, X: np.ndarray, y: np.ndarray, choice_sets: np.ndarray) -> Dict[str, Any]:
        """
        Multinomial Logit Model을 추정
        
        Args:
            X (np.ndarray): 설명변수 행렬
            y (np.ndarray): 선택 결과 벡터
            choice_sets (np.ndarray): 선택 세트 인덱스
            
        Returns:
            Dict[str, Any]: 추정 결과
        """
        logger.info("Multinomial Logit Model 추정 시작")
        
        try:
            # 데이터 준비
            df_X, df_y = self.prepare_data_for_statsmodels(X, y, choice_sets)
            
            # 상수항 추가
            df_X_with_const = add_constant(df_X)
            
            # 모델 생성 및 추정
            self.model = MNLogit(df_y, df_X_with_const)
            
            # 최적화 파라미터 가져오기
            opt_params = self.config_manager.get_optimization_params()
            
            # 모델 추정
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.results = self.model.fit(**opt_params)
            
            self.fitted = True
            
            # 결과 처리
            estimation_results = self._process_results(X, y, choice_sets)
            
            logger.info("모델 추정 완료")
            return estimation_results
            
        except Exception as e:
            logger.error(f"모델 추정 중 오류 발생: {e}")
            raise
    
    def _process_results(self, X: np.ndarray, y: np.ndarray, 
                        choice_sets: np.ndarray) -> Dict[str, Any]:
        """
        추정 결과를 처리하고 정리
        
        Args:
            X (np.ndarray): 설명변수 행렬
            y (np.ndarray): 선택 결과 벡터
            choice_sets (np.ndarray): 선택 세트 인덱스
            
        Returns:
            Dict[str, Any]: 처리된 결과
        """
        if not self.fitted or self.results is None:
            raise ValueError("모델이 추정되지 않았습니다")
        
        # 결과 템플릿 생성
        results = self.config_manager.create_results_template()
        
        # 기본 추정 결과
        results['estimation_results']['coefficients'] = self.results.params.values
        results['estimation_results']['standard_errors'] = self.results.bse.values
        results['estimation_results']['z_scores'] = self.results.tvalues.values
        results['estimation_results']['p_values'] = self.results.pvalues.values
        
        # 신뢰구간
        alpha = self.config_manager.get_confidence_interval_alpha()
        conf_int = self.results.conf_int(alpha=alpha)
        results['estimation_results']['confidence_intervals'] = conf_int.values
        
        # 모델 통계
        results['model_statistics']['log_likelihood'] = self.results.llf
        results['model_statistics']['aic'] = self.results.aic
        results['model_statistics']['bic'] = self.results.bic
        results['model_statistics']['pseudo_r_squared'] = self.results.prsquared
        results['model_statistics']['n_observations'] = self.results.nobs
        results['model_statistics']['n_choice_sets'] = len(choice_sets)
        
        # 수렴 정보
        results['convergence_info']['converged'] = self.results.mle_retvals['converged']
        results['convergence_info']['iterations'] = self.results.mle_retvals.get('iterations', None)
        
        # 선택적 결과 계산
        if self.config_manager.config.include_marginal_effects:
            results['marginal_effects'] = self._calculate_marginal_effects()
        
        if self.config_manager.config.include_elasticities:
            results['elasticities'] = self._calculate_elasticities(X)
        
        return results
    
    def _calculate_marginal_effects(self) -> Dict[str, np.ndarray]:
        """
        한계효과를 계산
        
        Returns:
            Dict[str, np.ndarray]: 한계효과
        """
        try:
            marginal_effects = self.results.get_margeff()
            return {
                'margeff': marginal_effects.margeff,
                'margeff_se': marginal_effects.margeff_se,
                'margeff_pvalues': marginal_effects.pvalues
            }
        except Exception as e:
            logger.warning(f"한계효과 계산 중 오류: {e}")
            return {}
    
    def _calculate_elasticities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        탄력성을 계산

        Args:
            X (np.ndarray): 설명변수 행렬

        Returns:
            Dict[str, np.ndarray]: 탄력성
        """
        try:
            # 평균값에서의 탄력성 계산
            X_mean = np.mean(X, axis=0)

            # 예측 확률
            df_X_mean = pd.DataFrame([X_mean], columns=self.config_manager.config.feature_names)
            df_X_mean_const = add_constant(df_X_mean)

            probs = self.results.predict(df_X_mean_const)

            # 계수 (상수항 제외)
            params = self.results.params
            if len(params) > len(self.config_manager.config.feature_names):
                coeffs = params.values[1:]  # 상수항 제외
            else:
                coeffs = params.values

            # 탄력성 계산 (연속변수에 대해서만)
            elasticities = {}
            for i, feature_name in enumerate(self.config_manager.config.feature_names):
                if i < len(coeffs) and X_mean[i] != 0:  # 인덱스 범위 확인 및 0으로 나누기 방지
                    if isinstance(probs, np.ndarray) and len(probs) > 0:
                        prob_value = probs[0] if probs.ndim > 0 else probs
                    else:
                        prob_value = probs

                    elasticity = coeffs[i] * X_mean[i] * (1 - prob_value)
                    elasticities[feature_name] = float(elasticity)

            return elasticities

        except Exception as e:
            logger.warning(f"탄력성 계산 중 오류: {e}")
            return {}
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        새로운 데이터에 대한 예측
        
        Args:
            X_new (np.ndarray): 예측할 설명변수 행렬
            
        Returns:
            np.ndarray: 예측 확률
        """
        if not self.fitted:
            raise ValueError("모델이 추정되지 않았습니다")
        
        # DataFrame으로 변환
        df_X_new = pd.DataFrame(X_new, columns=self.config_manager.config.feature_names)
        df_X_new_const = add_constant(df_X_new)
        
        return self.results.predict(df_X_new_const)
    
    def get_summary(self) -> str:
        """
        모델 요약을 반환
        
        Returns:
            str: 모델 요약 문자열
        """
        if not self.fitted:
            raise ValueError("모델이 추정되지 않았습니다")
        
        return str(self.results.summary())


def estimate_multinomial_logit(X: np.ndarray, y: np.ndarray, choice_sets: np.ndarray,
                              config: Optional[ModelConfig] = None) -> Dict[str, Any]:
    """
    Multinomial Logit Model을 추정하는 편의 함수
    
    Args:
        X (np.ndarray): 설명변수 행렬
        y (np.ndarray): 선택 결과 벡터
        choice_sets (np.ndarray): 선택 세트 인덱스
        config (Optional[ModelConfig]): 모델 설정
        
    Returns:
        Dict[str, Any]: 추정 결과
    """
    estimator = MultinomialLogitEstimator(config)
    return estimator.fit(X, y, choice_sets)
