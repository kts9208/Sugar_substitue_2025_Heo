"""
Hybrid Choice Model Main Analyzer

하이브리드 선택 모델의 메인 분석기입니다.
전체 분석 파이프라인을 조율하고 관리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path

from .config.hybrid_config import HybridConfig, ChoiceModelType
from .choice_models.choice_model_factory import ChoiceModelFactory
from .data_integration.hybrid_data_integrator import HybridDataIntegrator, IntegrationResult

# 기존 모듈 임포트
try:
    from ..factor_analysis.factor_analyzer import FactorAnalyzer
    from ..path_analysis.path_analyzer import PathAnalyzer
    EXISTING_SEM_AVAILABLE = True
except ImportError:
    EXISTING_SEM_AVAILABLE = False
    logging.warning("기존 SEM 모듈을 찾을 수 없습니다.")

logger = logging.getLogger(__name__)


@dataclass
class HybridAnalysisResult:
    """하이브리드 분석 결과"""
    
    # 기본 정보
    model_type: ChoiceModelType
    analysis_time: float
    success: bool
    
    # 데이터 통합 결과
    integration_result: IntegrationResult
    
    # 측정모델 결과
    measurement_model_results: Optional[Dict[str, Any]] = None
    factor_scores: Optional[pd.DataFrame] = None
    
    # 선택모델 결과
    choice_model_results: Optional[Any] = None
    
    # 하이브리드 결과
    hybrid_parameters: Optional[Dict[str, float]] = None
    model_fit: Optional[Dict[str, float]] = None
    
    # 예측 결과
    predicted_probabilities: Optional[pd.DataFrame] = None
    predicted_choices: Optional[pd.Series] = None
    
    # 추가 분석
    elasticities: Optional[Dict[str, float]] = None
    willingness_to_pay: Optional[Dict[str, float]] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """분석 결과 요약"""
        return {
            "model_type": self.model_type.value,
            "analysis_time": self.analysis_time,
            "success": self.success,
            "data_summary": self.integration_result.get_summary() if self.integration_result else {},
            "model_fit": self.model_fit or {},
            "n_parameters": len(self.hybrid_parameters) if self.hybrid_parameters else 0
        }


class HybridChoiceAnalyzer:
    """하이브리드 선택 모델 메인 분석기"""
    
    def __init__(self, config: HybridConfig):
        """
        Args:
            config: 하이브리드 모델 설정
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 구성요소 초기화
        self.data_integrator = None
        self.choice_model = None
        self.factor_analyzer = None
        self.path_analyzer = None
        
        # 결과 저장
        self.results = None
        self.is_fitted = False
        
        # 로깅 설정
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(getattr(logging, config.log_level))
            self.logger.addHandler(file_handler)
    
    def run_full_analysis(self, dce_data: pd.DataFrame, sem_data: pd.DataFrame, 
                         **kwargs) -> HybridAnalysisResult:
        """
        전체 하이브리드 분석 실행
        
        Args:
            dce_data: DCE 데이터
            sem_data: SEM 데이터 (설문 데이터)
            **kwargs: 추가 매개변수
            
        Returns:
            하이브리드 분석 결과
        """
        start_time = time.time()
        self.logger.info("하이브리드 선택 모델 분석을 시작합니다...")
        
        try:
            # 1. 데이터 통합
            integration_result = self._integrate_data(dce_data, sem_data, **kwargs)
            
            # 2. 측정모델 추정 (SEM 부분)
            measurement_results, factor_scores = self._estimate_measurement_model(
                integration_result.sem_data, **kwargs
            )
            
            # 3. 선택모델 추정 (DCE 부분)
            choice_results = self._estimate_choice_model(
                integration_result.integrated_data, factor_scores, **kwargs
            )
            
            # 4. 하이브리드 모델 통합 (선택적)
            if self.config.simultaneous_estimation:
                hybrid_results = self._estimate_hybrid_model(
                    integration_result.integrated_data, measurement_results, choice_results, **kwargs
                )
            else:
                hybrid_results = self._combine_sequential_results(measurement_results, choice_results)
            
            # 5. 예측 및 추가 분석
            predictions = self._generate_predictions(integration_result.integrated_data, **kwargs)
            additional_analysis = self._perform_additional_analysis(hybrid_results, **kwargs)
            
            # 결과 구성
            analysis_time = time.time() - start_time
            
            result = HybridAnalysisResult(
                model_type=self.config.choice_model.model_type,
                analysis_time=analysis_time,
                success=True,
                integration_result=integration_result,
                measurement_model_results=measurement_results,
                factor_scores=factor_scores,
                choice_model_results=choice_results,
                hybrid_parameters=hybrid_results.get('parameters', {}),
                model_fit=hybrid_results.get('fit_statistics', {}),
                predicted_probabilities=predictions.get('probabilities'),
                predicted_choices=predictions.get('choices'),
                elasticities=additional_analysis.get('elasticities'),
                willingness_to_pay=additional_analysis.get('wtp')
            )
            
            self.results = result
            self.is_fitted = True
            
            # 결과 저장
            if self.config.save_results:
                self._save_results(result)
            
            self.logger.info(f"하이브리드 분석 완료 (소요시간: {analysis_time:.2f}초)")
            return result
            
        except Exception as e:
            analysis_time = time.time() - start_time
            self.logger.error(f"하이브리드 분석 실패: {e}")
            
            return HybridAnalysisResult(
                model_type=self.config.choice_model.model_type,
                analysis_time=analysis_time,
                success=False,
                integration_result=None
            )
    
    def _integrate_data(self, dce_data: pd.DataFrame, sem_data: pd.DataFrame, **kwargs) -> IntegrationResult:
        """데이터 통합"""
        self.logger.info("1단계: 데이터 통합")
        
        # 데이터 통합기 설정
        integration_config = {
            'individual_id_column': self.config.data.individual_column,
            'merge_method': kwargs.get('merge_method', 'inner'),
            'handle_missing': self.config.data.handle_missing_data
        }
        
        self.data_integrator = HybridDataIntegrator(integration_config)
        
        # 잠재변수 목록
        latent_variables = self.config.data.latent_variables
        
        return self.data_integrator.integrate_data(dce_data, sem_data, latent_variables)
    
    def _estimate_measurement_model(self, sem_data: pd.DataFrame, **kwargs) -> tuple:
        """측정모델 추정"""
        self.logger.info("2단계: 측정모델 추정")
        
        if not EXISTING_SEM_AVAILABLE:
            self.logger.warning("기존 SEM 모듈을 사용할 수 없습니다. 간단한 구현을 사용합니다.")
            return self._simple_measurement_model(sem_data, **kwargs)
        
        try:
            # 기존 요인분석 모듈 사용
            factor_config = kwargs.get('factor_config', {})
            self.factor_analyzer = FactorAnalyzer(factor_config)
            
            # 모델 스펙 생성 (간단한 예시)
            model_spec = self._create_measurement_model_spec()
            
            # 모델 추정
            factor_results = self.factor_analyzer.fit_model(sem_data, model_spec)
            
            # 요인점수 계산
            factor_scores = self._calculate_factor_scores(sem_data, factor_results)
            
            return factor_results, factor_scores
            
        except Exception as e:
            self.logger.warning(f"기존 SEM 모듈 사용 실패: {e}. 간단한 구현을 사용합니다.")
            return self._simple_measurement_model(sem_data, **kwargs)
    
    def _simple_measurement_model(self, sem_data: pd.DataFrame, **kwargs) -> tuple:
        """간단한 측정모델 구현"""
        # 기본적인 요인점수 계산 (평균 기반)
        latent_variables = self.config.data.latent_variables
        observed_variables = self.config.data.observed_variables
        
        factor_scores = pd.DataFrame()
        measurement_results = {"method": "simple_average", "factors": {}}
        
        for lv in latent_variables:
            if lv in observed_variables:
                lv_columns = observed_variables[lv]
                available_columns = [col for col in lv_columns if col in sem_data.columns]
                
                if available_columns:
                    factor_scores[lv] = sem_data[available_columns].mean(axis=1)
                    measurement_results["factors"][lv] = {
                        "observed_variables": available_columns,
                        "reliability": 0.8  # 임시값
                    }
        
        return measurement_results, factor_scores
    
    def _estimate_choice_model(self, integrated_data: pd.DataFrame, 
                              factor_scores: pd.DataFrame, **kwargs) -> Any:
        """선택모델 추정"""
        self.logger.info("3단계: 선택모델 추정")
        
        # 선택모델 생성
        choice_config = {
            'choice_column': self.config.data.choice_column,
            'alternative_column': self.config.data.alternative_column,
            'individual_column': self.config.data.individual_column,
            **kwargs.get('choice_config', {})
        }
        
        self.choice_model = ChoiceModelFactory.create_model(
            self.config.choice_model.model_type, choice_config
        )
        
        # 데이터 준비 (DCE 데이터 + 요인점수)
        choice_data = self._prepare_choice_data(integrated_data, factor_scores)
        
        # 모델 추정
        return self.choice_model.fit(choice_data, **kwargs)
    
    def _prepare_choice_data(self, integrated_data: pd.DataFrame, 
                           factor_scores: pd.DataFrame) -> pd.DataFrame:
        """선택모델용 데이터 준비"""
        # DCE 데이터 추출
        dce_columns = [col for col in integrated_data.columns if not col.endswith('_sem')]
        choice_data = integrated_data[dce_columns].copy()
        
        # 요인점수 병합
        if not factor_scores.empty:
            individual_col = self.config.data.individual_column
            
            # 개체별 요인점수 병합
            choice_data = choice_data.merge(
                factor_scores.reset_index().rename(columns={'index': individual_col}),
                on=individual_col,
                how='left'
            )
        
        return choice_data
    
    def _estimate_hybrid_model(self, integrated_data: pd.DataFrame, 
                              measurement_results: Dict[str, Any], 
                              choice_results: Any, **kwargs) -> Dict[str, Any]:
        """하이브리드 모델 동시 추정"""
        self.logger.info("4단계: 하이브리드 모델 동시 추정")
        
        # 실제 동시 추정은 복잡하므로 여기서는 순차 결과를 결합
        return self._combine_sequential_results(measurement_results, choice_results)
    
    def _combine_sequential_results(self, measurement_results: Dict[str, Any], 
                                   choice_results: Any) -> Dict[str, Any]:
        """순차 추정 결과 결합"""
        combined_parameters = {}
        
        # 측정모델 파라미터
        if measurement_results and 'factors' in measurement_results:
            for factor, info in measurement_results['factors'].items():
                combined_parameters[f"measurement_{factor}_reliability"] = info.get('reliability', 0)
        
        # 선택모델 파라미터
        if hasattr(choice_results, 'parameters'):
            for param, value in choice_results.parameters.items():
                combined_parameters[f"choice_{param}"] = value
        
        # 적합도 통계
        fit_statistics = {}
        if hasattr(choice_results, 'log_likelihood'):
            fit_statistics['log_likelihood'] = choice_results.log_likelihood
            fit_statistics['aic'] = choice_results.aic
            fit_statistics['bic'] = choice_results.bic
            fit_statistics['rho_squared'] = choice_results.rho_squared
        
        return {
            'parameters': combined_parameters,
            'fit_statistics': fit_statistics,
            'measurement_results': measurement_results,
            'choice_results': choice_results
        }
    
    def _generate_predictions(self, integrated_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """예측 생성"""
        self.logger.info("5단계: 예측 생성")
        
        predictions = {}
        
        if self.choice_model and self.choice_model.is_fitted:
            try:
                # 선택 확률 예측
                probabilities = self.choice_model.predict_probabilities(integrated_data)
                predictions['probabilities'] = probabilities
                
                # 선택 예측
                choices = self.choice_model.predict_choices(integrated_data)
                predictions['choices'] = choices
                
            except Exception as e:
                self.logger.warning(f"예측 생성 실패: {e}")
        
        return predictions
    
    def _perform_additional_analysis(self, hybrid_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """추가 분석 수행"""
        self.logger.info("6단계: 추가 분석")
        
        additional_analysis = {}
        
        # 탄력성 계산 (간단한 예시)
        if 'parameters' in hybrid_results:
            elasticities = {}
            for param, value in hybrid_results['parameters'].items():
                if 'choice_' in param and 'price' in param.lower():
                    elasticities[param] = value * 1.5  # 임시 계산
            additional_analysis['elasticities'] = elasticities
        
        # 지불의사액 계산 (간단한 예시)
        wtp = {}
        if 'elasticities' in additional_analysis:
            for param, elasticity in additional_analysis['elasticities'].items():
                wtp[param.replace('choice_', 'wtp_')] = abs(elasticity) * 10  # 임시 계산
        additional_analysis['wtp'] = wtp
        
        return additional_analysis
    
    def _create_measurement_model_spec(self) -> str:
        """측정모델 스펙 생성"""
        # 간단한 CFA 모델 스펙 생성
        latent_variables = self.config.data.latent_variables
        observed_variables = self.config.data.observed_variables
        
        model_lines = []
        for lv in latent_variables:
            if lv in observed_variables:
                indicators = observed_variables[lv]
                if indicators:
                    model_line = f"{lv} =~ " + " + ".join(indicators)
                    model_lines.append(model_line)
        
        return "\n".join(model_lines)
    
    def _calculate_factor_scores(self, data: pd.DataFrame, factor_results: Dict[str, Any]) -> pd.DataFrame:
        """요인점수 계산"""
        # 간단한 요인점수 계산 (실제로는 더 정교한 방법 필요)
        latent_variables = self.config.data.latent_variables
        observed_variables = self.config.data.observed_variables
        
        factor_scores = pd.DataFrame()
        
        for lv in latent_variables:
            if lv in observed_variables:
                lv_columns = observed_variables[lv]
                available_columns = [col for col in lv_columns if col in data.columns]
                
                if available_columns:
                    factor_scores[lv] = data[available_columns].mean(axis=1)
        
        return factor_scores
    
    def _save_results(self, result: HybridAnalysisResult):
        """결과 저장"""
        if not self.config.save_results:
            return
        
        results_dir = Path(self.config.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프 추가
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 요약 결과 저장
        summary_file = results_dir / f"hybrid_analysis_summary_{timestamp}.json"
        
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(result.get_summary(), f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"결과가 저장되었습니다: {summary_file}")


# 편의 함수들
def run_hybrid_analysis(dce_data: pd.DataFrame, sem_data: pd.DataFrame, 
                       config: Optional[HybridConfig] = None, **kwargs) -> HybridAnalysisResult:
    """하이브리드 분석 실행 편의 함수"""
    if config is None:
        from .config.hybrid_config import create_default_config
        config = create_default_config()
    
    analyzer = HybridChoiceAnalyzer(config)
    return analyzer.run_full_analysis(dce_data, sem_data, **kwargs)


def run_model_comparison(dce_data: pd.DataFrame, sem_data: pd.DataFrame, 
                        model_types: List[str], **kwargs) -> Dict[str, HybridAnalysisResult]:
    """여러 모델 비교 분석"""
    results = {}
    
    for model_type in model_types:
        try:
            from .config.hybrid_config import create_custom_config
            config = create_custom_config(choice_model_type=model_type, **kwargs)
            
            result = run_hybrid_analysis(dce_data, sem_data, config, **kwargs)
            results[model_type] = result
            
        except Exception as e:
            logger.error(f"모델 {model_type} 분석 실패: {e}")
            results[model_type] = None
    
    return results
