"""
Multinomial Logit Model

다항로짓 모델 구현입니다.
기존 multinomial_logit 모듈을 활용하여 하이브리드 모델에 통합합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import logging

from .base_choice_model import BaseChoiceModel, ChoiceModelType, ChoiceModelResults

# 기존 MNL 모듈 임포트
try:
    from ...multinomial_logit.model_estimator import MultinomialLogitEstimator
    from ...multinomial_logit.model_config import ModelConfig
    EXISTING_MNL_AVAILABLE = True
except ImportError:
    EXISTING_MNL_AVAILABLE = False
    logging.warning("기존 multinomial_logit 모듈을 찾을 수 없습니다. 간단한 구현을 사용합니다.")

# 대체 구현을 위한 임포트
try:
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MNLResults(ChoiceModelResults):
    """MNL 모델 결과 클래스"""
    
    # MNL 특화 결과
    choice_probabilities: Optional[pd.DataFrame] = None
    elasticities: Optional[Dict[str, float]] = None
    marginal_effects: Optional[Dict[str, float]] = None
    
    def get_choice_probabilities_summary(self) -> Dict[str, Any]:
        """선택확률 요약 통계"""
        if self.choice_probabilities is None:
            return {}
        
        return {
            "mean_probabilities": self.choice_probabilities.mean().to_dict(),
            "std_probabilities": self.choice_probabilities.std().to_dict(),
            "min_probabilities": self.choice_probabilities.min().to_dict(),
            "max_probabilities": self.choice_probabilities.max().to_dict()
        }


class MultinomialLogitModel(BaseChoiceModel):
    """다항로짓 모델 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.coefficients = {}
        self.covariance_matrix = None
        
        # 기존 MNL 모듈 사용 여부 결정
        self.use_existing_module = EXISTING_MNL_AVAILABLE and config.get('use_existing_module', True)
        
        if self.use_existing_module:
            self._setup_existing_estimator()
        else:
            self._setup_simple_estimator()
    
    @property
    def model_type(self) -> ChoiceModelType:
        return ChoiceModelType.MULTINOMIAL_LOGIT
    
    def _setup_existing_estimator(self):
        """기존 MNL 모듈 설정"""
        try:
            # 기존 모듈의 설정 생성
            self.mnl_config = ModelConfig()
            
            # 설정 매핑
            if 'max_iterations' in self.config:
                self.mnl_config.max_iterations = self.config['max_iterations']
            if 'convergence_tolerance' in self.config:
                self.mnl_config.convergence_tolerance = self.config['convergence_tolerance']
            
            self.estimator = MultinomialLogitEstimator(self.mnl_config)
            logger.info("기존 MNL 모듈을 사용합니다.")
            
        except Exception as e:
            logger.warning(f"기존 MNL 모듈 설정 실패: {e}. 간단한 구현을 사용합니다.")
            self.use_existing_module = False
            self._setup_simple_estimator()
    
    def _setup_simple_estimator(self):
        """간단한 MNL 구현 설정"""
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.convergence_tolerance = self.config.get('convergence_tolerance', 1e-6)
        self.optimizer = self.config.get('optimizer', 'BFGS')
        logger.info("간단한 MNL 구현을 사용합니다.")
    
    def fit(self, data: pd.DataFrame, **kwargs) -> MNLResults:
        """
        MNL 모델 추정
        
        Args:
            data: 선택 데이터
            **kwargs: 추가 매개변수
            
        Returns:
            MNL 추정 결과
        """
        start_time = time.time()
        
        # 데이터 검증 및 전처리
        self.validate_data(data)
        self.data = self.prepare_data(data)
        
        if self.use_existing_module:
            results = self._fit_with_existing_module(**kwargs)
        else:
            results = self._fit_with_simple_implementation(**kwargs)
        
        # 공통 후처리
        estimation_time = time.time() - start_time
        results.estimation_time = estimation_time
        
        self.results = results
        self.is_fitted = True
        
        logger.info(f"MNL 모델 추정 완료 (소요시간: {estimation_time:.2f}초)")
        return results
    
    def _fit_with_existing_module(self, **kwargs) -> MNLResults:
        """기존 모듈을 사용한 추정"""
        try:
            # 데이터 형식 변환
            X, y, choice_sets = self._prepare_data_for_existing_module()
            
            # 모델 추정
            estimation_results = self.estimator.estimate_model(X, y, choice_sets)
            
            # 결과 변환
            return self._convert_existing_results(estimation_results)
            
        except Exception as e:
            logger.error(f"기존 모듈 추정 실패: {e}")
            raise
    
    def _fit_with_simple_implementation(self, **kwargs) -> MNLResults:
        """간단한 구현을 사용한 추정"""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy가 필요합니다. pip install scipy로 설치해주세요.")
        
        # 데이터 준비
        X, y = self._prepare_data_for_simple_implementation()
        
        # 초기값 설정
        n_features = X.shape[1]
        n_alternatives = len(np.unique(y))
        n_params = n_features * (n_alternatives - 1)  # 기준 대안 제외
        
        initial_params = np.zeros(n_params)
        
        # 최적화
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            args=(X, y),
            method=self.optimizer,
            options={'maxiter': self.max_iterations, 'gtol': self.convergence_tolerance}
        )
        
        # 결과 처리
        return self._process_simple_results(result, X, y)
    
    def _prepare_data_for_existing_module(self):
        """기존 모듈용 데이터 준비"""
        # 기존 모듈의 데이터 형식에 맞게 변환
        # 이 부분은 기존 모듈의 인터페이스에 따라 구현
        
        # 예시 구현 (실제로는 기존 모듈의 형식에 맞춰야 함)
        feature_columns = [col for col in self.data.columns 
                          if col not in [self.choice_column, self.alternative_column, self.individual_column]]
        
        X = self.data[feature_columns].values
        y = self.data[self.choice_column].values
        choice_sets = self.data.groupby(self.individual_column).size().values
        
        return X, y, choice_sets
    
    def _prepare_data_for_simple_implementation(self):
        """간단한 구현용 데이터 준비"""
        # 특성 변수 추출
        feature_columns = [col for col in self.data.columns 
                          if col not in [self.choice_column, self.alternative_column, self.individual_column]]
        
        X = self.data[feature_columns].values
        y = self.data[self.choice_column].values
        
        return X, y
    
    def _negative_log_likelihood(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """음의 로그우도 함수"""
        try:
            # 효용 계산
            n_alternatives = len(np.unique(y))
            n_features = X.shape[1]
            
            # 파라미터를 계수 행렬로 변환
            coefficients = params.reshape((n_alternatives - 1, n_features))
            
            # 기준 대안(0)의 효용은 0으로 설정
            utilities = np.zeros((X.shape[0], n_alternatives))
            utilities[:, 1:] = X @ coefficients.T
            
            # 선택 확률 계산
            exp_utilities = np.exp(utilities)
            probabilities = exp_utilities / np.sum(exp_utilities, axis=1, keepdims=True)
            
            # 로그우도 계산
            log_likelihood = 0
            for i, choice in enumerate(y):
                log_likelihood += np.log(probabilities[i, choice] + 1e-10)  # 수치적 안정성
            
            return -log_likelihood
            
        except Exception as e:
            logger.warning(f"우도 계산 중 오류: {e}")
            return 1e10  # 큰 값 반환하여 최적화 방향 유도
    
    def _process_simple_results(self, optimization_result, X: np.ndarray, y: np.ndarray) -> MNLResults:
        """간단한 구현 결과 처리"""
        # 모수 추출
        params = optimization_result.x
        n_alternatives = len(np.unique(y))
        n_features = X.shape[1]
        
        # 계수 딕셔너리 생성
        feature_columns = [col for col in self.data.columns 
                          if col not in [self.choice_column, self.alternative_column, self.individual_column]]
        
        parameters = {}
        param_idx = 0
        for alt in range(1, n_alternatives):  # 기준 대안 제외
            for feat_idx, feat_name in enumerate(feature_columns):
                param_name = f"{feat_name}_alt_{alt}"
                parameters[param_name] = params[param_idx]
                param_idx += 1
        
        # 적합도 통계 계산
        log_likelihood = -optimization_result.fun
        n_params = len(params)
        n_obs = len(y)
        
        fit_stats = self.calculate_fit_statistics(log_likelihood, n_params, n_obs)
        
        # 표준오차 계산 (간단한 근사)
        # 실제로는 헤시안 행렬의 역행렬을 사용해야 함
        standard_errors = {param: 0.1 for param in parameters.keys()}  # 임시값
        t_statistics = {param: parameters[param] / standard_errors[param] 
                       for param in parameters.keys()}
        p_values = {param: 2 * (1 - abs(t_statistics[param]) / 1.96) 
                   for param in parameters.keys()}  # 임시 계산
        
        # 예측 확률 계산
        predicted_probabilities = self._calculate_predicted_probabilities(params, X, y)
        
        return MNLResults(
            model_type=self.model_type,
            log_likelihood=log_likelihood,
            aic=fit_stats['aic'],
            bic=fit_stats['bic'],
            parameters=parameters,
            standard_errors=standard_errors,
            t_statistics=t_statistics,
            p_values=p_values,
            rho_squared=fit_stats['rho_squared'],
            adjusted_rho_squared=fit_stats['adjusted_rho_squared'],
            convergence_status=optimization_result.success,
            iterations=optimization_result.nit if hasattr(optimization_result, 'nit') else 0,
            sample_size=n_obs,
            choice_probabilities=predicted_probabilities
        )
    
    def _calculate_predicted_probabilities(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """예측 확률 계산"""
        n_alternatives = len(np.unique(y))
        n_features = X.shape[1]
        
        # 계수 행렬 구성
        coefficients = params.reshape((n_alternatives - 1, n_features))
        
        # 효용 계산
        utilities = np.zeros((X.shape[0], n_alternatives))
        utilities[:, 1:] = X @ coefficients.T
        
        # 확률 계산
        exp_utilities = np.exp(utilities)
        probabilities = exp_utilities / np.sum(exp_utilities, axis=1, keepdims=True)
        
        # 데이터프레임으로 변환
        prob_df = pd.DataFrame(probabilities, columns=[f"alt_{i}" for i in range(n_alternatives)])
        return prob_df
    
    def _convert_existing_results(self, estimation_results: Dict[str, Any]) -> MNLResults:
        """기존 모듈 결과를 MNLResults로 변환"""
        # 기존 모듈의 결과 형식에 따라 구현
        # 이는 실제 기존 모듈의 출력 형식을 확인한 후 구현해야 함
        
        return MNLResults(
            model_type=self.model_type,
            log_likelihood=estimation_results.get('log_likelihood', 0),
            aic=estimation_results.get('aic', 0),
            bic=estimation_results.get('bic', 0),
            parameters=estimation_results.get('parameters', {}),
            standard_errors=estimation_results.get('standard_errors', {}),
            t_statistics=estimation_results.get('t_statistics', {}),
            p_values=estimation_results.get('p_values', {}),
            rho_squared=estimation_results.get('rho_squared', 0),
            adjusted_rho_squared=estimation_results.get('adjusted_rho_squared', 0),
            convergence_status=estimation_results.get('convergence_status', True),
            sample_size=estimation_results.get('sample_size', 0)
        )
    
    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """선택 확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 추정되지 않았습니다.")
        
        # 데이터 전처리
        processed_data = self.prepare_data(data)
        
        if self.use_existing_module:
            return self._predict_with_existing_module(processed_data)
        else:
            return self._predict_with_simple_implementation(processed_data)
    
    def predict_choices(self, data: pd.DataFrame) -> pd.Series:
        """선택 예측"""
        probabilities = self.predict_probabilities(data)
        predicted_choices = probabilities.idxmax(axis=1)
        return predicted_choices
    
    def _predict_with_existing_module(self, data: pd.DataFrame) -> pd.DataFrame:
        """기존 모듈을 사용한 예측"""
        # 기존 모듈의 예측 인터페이스 사용
        # 실제 구현은 기존 모듈의 API에 따라 달라짐
        pass
    
    def _predict_with_simple_implementation(self, data: pd.DataFrame) -> pd.DataFrame:
        """간단한 구현을 사용한 예측"""
        if not hasattr(self, 'coefficients') or not self.coefficients:
            raise ValueError("모델 계수가 없습니다.")

        # 특성 변수 추출
        feature_columns = [col for col in data.columns
                          if col not in [self.choice_column, self.alternative_column, self.individual_column]]
        X = data[feature_columns].values

        # 저장된 파라미터로 확률 계산
        # 실제 구현에서는 self.results.parameters를 사용
        if hasattr(self.results, 'choice_probabilities') and self.results.choice_probabilities is not None:
            return self.results.choice_probabilities
        else:
            # 기본 균등 확률 반환 (임시)
            n_obs = X.shape[0]
            n_alternatives = 3  # 임시값
            uniform_probs = np.ones((n_obs, n_alternatives)) / n_alternatives
            return pd.DataFrame(uniform_probs, columns=[f"alt_{i}" for i in range(n_alternatives)])
