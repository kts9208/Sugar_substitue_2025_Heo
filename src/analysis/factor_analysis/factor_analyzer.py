"""
Factor Analysis Module using semopy

이 모듈은 semopy를 사용하여 확인적 요인분석(CFA)을 수행하고
factor loading 값을 계산하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import warnings

# semopy 임포트
try:
    import semopy
    from semopy import Model
    from semopy.stats import calc_stats
except ImportError as e:
    logging.error("semopy 라이브러리를 찾을 수 없습니다. pip install semopy로 설치해주세요.")
    raise e

from .config import FactorAnalysisConfig, get_default_config
from .data_loader import FactorDataLoader

logger = logging.getLogger(__name__)


class SemopyAnalyzer:
    """semopy를 사용한 요인분석 클래스"""
    
    def __init__(self, config: Optional[FactorAnalysisConfig] = None):
        """
        Semopy Analyzer 초기화
        
        Args:
            config (Optional[FactorAnalysisConfig]): 분석 설정
        """
        self.config = config if config is not None else get_default_config()
        self.model = None
        self.results = None
        self.fitted = False
        
    def fit_model(self, data: pd.DataFrame, model_spec: str) -> Dict[str, Any]:
        """
        모델을 적합하고 결과를 반환
        
        Args:
            data (pd.DataFrame): 분석할 데이터
            model_spec (str): semopy 모델 스펙
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        logger.info("semopy 모델 적합 시작")
        
        try:
            # 데이터 전처리
            clean_data = self._prepare_data(data)
            
            # 모델 생성
            self.model = Model(model_spec)

            # 모델 적합
            print(f"\n[SEM 최적화 시작] solver={self.config.optimizer}")
            logger.info(f"SEM 최적화 시작 (solver={self.config.optimizer})...")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # semopy의 최신 API에 맞게 수정
                self.results = self.model.fit(
                    clean_data,
                    solver=self.config.optimizer
                )

            self.fitted = True

            # 최적화 결과 로깅
            if hasattr(self.model, 'last_result') and self.model.last_result is not None:
                print(f"\n[SEM 최적화 완료]")
                logger.info(f"SEM 최적화 완료:")

                # semopy의 SolverResult 속성 확인
                if hasattr(self.model.last_result, 'n_it'):
                    print(f"  반복 횟수: {self.model.last_result.n_it}")
                    logger.info(f"  반복 횟수: {self.model.last_result.n_it}")
                elif hasattr(self.model.last_result, 'nit'):
                    print(f"  반복 횟수: {self.model.last_result.nit}")
                    logger.info(f"  반복 횟수: {self.model.last_result.nit}")

                if hasattr(self.model.last_result, 'n_fev'):
                    print(f"  함수 평가 횟수: {self.model.last_result.n_fev}")
                    logger.info(f"  함수 평가 횟수: {self.model.last_result.n_fev}")
                elif hasattr(self.model.last_result, 'nfev'):
                    print(f"  함수 평가 횟수: {self.model.last_result.nfev}")
                    logger.info(f"  함수 평가 횟수: {self.model.last_result.nfev}")

                if hasattr(self.model.last_result, 'fun'):
                    print(f"  목적함수 값: {self.model.last_result.fun:.4f}")
                    logger.info(f"  목적함수 값: {self.model.last_result.fun:.4f}")

                if hasattr(self.model.last_result, 'success'):
                    print(f"  수렴 여부: {self.model.last_result.success}")
                    logger.info(f"  수렴 여부: {self.model.last_result.success}")

                if hasattr(self.model.last_result, 'message'):
                    print(f"  메시지: {self.model.last_result.message}")
                    logger.info(f"  메시지: {self.model.last_result.message}")

            # 결과 처리
            analysis_results = self._process_results(clean_data)

            logger.info("모델 적합 완료")
            return analysis_results
            
        except Exception as e:
            logger.error(f"모델 적합 중 오류 발생: {e}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        분석용 데이터 전처리
        
        Args:
            data (pd.DataFrame): 원본 데이터
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        # 'no' 컬럼 제거 (응답자 ID는 분석에 불필요)
        if 'no' in data.columns:
            clean_data = data.drop('no', axis=1)
        else:
            clean_data = data.copy()
        
        # 결측치 처리
        if self.config.missing_data_method == 'listwise':
            clean_data = clean_data.dropna()
            logger.info(f"결측치 제거 후 샘플 수: {len(clean_data)}")
        
        # 수치형 데이터만 선택
        numeric_columns = clean_data.select_dtypes(include=[np.number]).columns
        clean_data = clean_data[numeric_columns]
        
        # 분산이 0인 변수 제거
        zero_var_cols = clean_data.columns[clean_data.var() == 0]
        if len(zero_var_cols) > 0:
            logger.warning(f"분산이 0인 변수들 제거: {list(zero_var_cols)}")
            clean_data = clean_data.drop(zero_var_cols, axis=1)
        
        logger.info(f"전처리 완료: {clean_data.shape}")
        return clean_data
    
    def _process_results(self, clean_data: pd.DataFrame) -> Dict[str, Any]:
        """
        분석 결과를 처리하고 정리
        
        Args:
            data (pd.DataFrame): 분석에 사용된 데이터
            
        Returns:
            Dict[str, Any]: 정리된 분석 결과
        """
        results = {
            'model_info': {
                'n_observations': len(clean_data),
                'n_variables': len(clean_data.columns),
                'estimator': self.config.estimator,
                'optimizer': self.config.optimizer
            },
            'factor_loadings': {},
            'fit_indices': {},
            'parameter_estimates': {},
            'standardized_results': {}
        }
        
        # Factor loadings 추출
        if self.results is not None:
            # 모든 파라미터 추출
            params = self.model.inspect()
            
            # Factor loadings 필터링 (~ 관계 - semopy에서는 ~ 사용)
            loadings = params[params['op'] == '~'].copy()
            if len(loadings) > 0:
                results['factor_loadings'] = self._format_factor_loadings(loadings)
            else:
                results['factor_loadings'] = pd.DataFrame()
            
            # 적합도 지수 계산
            if self.config.calculate_fit_indices:
                try:
                    fit_stats = calc_stats(self.model)
                    results['fit_indices'] = self._format_fit_indices(fit_stats)
                except Exception as e:
                    logger.warning(f"적합도 지수 계산 실패: {e}")
            
            # 표준화 결과
            if self.config.standardized:
                try:
                    std_results = self.model.inspect(std_est=True)
                    std_formatted = self._format_standardized_results(std_results)
                    results['standardized_results'] = std_formatted if std_formatted is not None else {}
                except Exception as e:
                    logger.warning(f"표준화 결과 계산 실패: {e}")
                    results['standardized_results'] = {}

            # 신뢰도 통계 계산
            try:
                reliability_results = self._calculate_reliability_stats(clean_data)
                results['reliability_stats'] = reliability_results
            except Exception as e:
                logger.warning(f"신뢰도 통계 계산 실패: {e}")
                results['reliability_stats'] = {}

            # 모델 객체 포함 (신뢰도 계산 등에 필요)
            results['model'] = self.model

        return results

    def _calculate_reliability_stats(self, clean_data: pd.DataFrame) -> Dict[str, Any]:
        """
        신뢰도 통계 계산

        Args:
            clean_data (pd.DataFrame): 전처리된 데이터

        Returns:
            Dict[str, Any]: 신뢰도 통계 결과
        """
        try:
            # 신뢰도 계산은 별도의 신뢰도 분석 단계에서 수행
            logger.info("신뢰도 통계는 별도의 신뢰도 분석 단계에서 계산됩니다.")

            return {
                'message': '신뢰도 통계는 별도 분석 단계에서 계산됩니다.',
                'note': '요인분석 완료 후 run_independent_reliability_analysis.py를 실행하세요.'
            }

        except Exception as e:
            logger.error(f"신뢰도 통계 계산 중 오류: {e}")
            return {}

    def _format_factor_loadings(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """Factor loadings를 정리된 형태로 포맷"""
        # 사용 가능한 컬럼 확인
        available_cols = loadings.columns.tolist()

        # 필요한 컬럼들의 매핑 (semopy에서는 lval이 item, rval이 factor)
        col_mapping = {
            'lval': 'Item',
            'rval': 'Factor',
            'Estimate': 'Loading',
            'Std. Err': 'SE',
            'z-value': 'Z_value',
            'p-value': 'P_value'
        }

        # 대안 컬럼명들
        alt_cols = {
            'Std. Err': ['SE', 'std_err', 'stderr'],
            'z-value': ['z_value', 'z', 'Z'],
            'p-value': ['p_value', 'pvalue', 'p']
        }

        # 실제 사용할 컬럼들 찾기
        use_cols = []
        new_names = []

        for orig_col, new_name in col_mapping.items():
            if orig_col in available_cols:
                use_cols.append(orig_col)
                new_names.append(new_name)
            elif orig_col in alt_cols:
                for alt_col in alt_cols[orig_col]:
                    if alt_col in available_cols:
                        use_cols.append(alt_col)
                        new_names.append(new_name)
                        break

        if len(use_cols) >= 3:  # 최소한 Factor, Item, Loading은 있어야 함
            formatted = loadings[use_cols].copy()
            formatted.columns = new_names

            # P_value가 있으면 유의성 검정
            if 'P_value' in formatted.columns:
                # P_value를 숫자로 변환 (문자열 '-'는 NaN으로 처리)
                formatted['P_value'] = pd.to_numeric(formatted['P_value'], errors='coerce')
                formatted['Significant'] = formatted['P_value'] < 0.05

            return formatted.round(4)
        else:
            # 기본 컬럼만 사용 (semopy에서는 lval이 item, rval이 factor)
            basic_cols = ['lval', 'rval', 'Estimate']
            if all(col in available_cols for col in basic_cols):
                formatted = loadings[basic_cols].copy()
                formatted.columns = ['Item', 'Factor', 'Loading']
                # 컬럼 순서 조정
                formatted = formatted[['Factor', 'Item', 'Loading']]
                return formatted.round(4)

        return pd.DataFrame()
    
    def _format_fit_indices(self, fit_stats: Dict) -> Dict[str, float]:
        """적합도 지수를 정리된 형태로 포맷"""
        formatted_fit = {}

        # 주요 적합도 지수들
        key_indices = ['CFI', 'TLI', 'RMSEA', 'SRMR', 'GFI', 'AGFI', 'AIC', 'BIC']

        for index in key_indices:
            if index in fit_stats:
                value = fit_stats[index]
                # pandas Series인 경우 첫 번째 값 추출
                if hasattr(value, 'iloc'):
                    value = value.iloc[0] if len(value) > 0 else value
                # numpy array인 경우 스칼라로 변환
                elif hasattr(value, 'item'):
                    value = value.item()
                formatted_fit[index] = round(float(value), 4)

        return formatted_fit
    
    def _format_standardized_results(self, std_results: pd.DataFrame) -> pd.DataFrame:
        """표준화 결과를 정리된 형태로 포맷"""
        # Factor loadings만 필터링 (semopy에서는 ~ 사용)
        std_loadings = std_results[std_results['op'] == '~'].copy()
        if len(std_loadings) > 0:
            # 사용 가능한 컬럼 확인 (semopy에서는 'Est. Std' 사용)
            available_cols = std_results.columns.tolist()
            std_col = None
            for col in ['Est. Std', 'Std. Estimate', 'std_est', 'standardized']:
                if col in available_cols:
                    std_col = col
                    break

            if std_col:
                # 표준화된 loading을 주요 결과로 사용
                formatted = std_loadings[['lval', 'rval', std_col]].copy()
                formatted.columns = ['Item', 'Factor', 'Loading']  # 'Std_Loading' 대신 'Loading' 사용

                # 컬럼 순서 조정
                formatted = formatted[['Factor', 'Item', 'Loading']]

                # 기본 통계 정보 추가 (표준화된 해에서는 SE, Z, P값이 원래 해와 동일)
                original_loadings = std_results[std_results['op'] == '~'].copy()
                if 'Std. Err' in original_loadings.columns:
                    formatted['SE'] = original_loadings['Std. Err'].values
                if 'z-value' in original_loadings.columns:
                    formatted['Z_value'] = original_loadings['z-value'].values
                if 'p-value' in original_loadings.columns:
                    formatted['P_value'] = pd.to_numeric(original_loadings['p-value'], errors='coerce')
                    formatted['Significant'] = formatted['P_value'] < 0.05

                return formatted.round(4)
        return pd.DataFrame()
    
    def get_factor_loadings_table(self) -> pd.DataFrame:
        """Factor loadings 테이블 반환"""
        if not self.fitted:
            raise ValueError("모델이 적합되지 않았습니다")
        
        return self.results.get('factor_loadings', pd.DataFrame())
    
    def get_fit_indices(self) -> Dict[str, float]:
        """적합도 지수 반환"""
        if not self.fitted:
            raise ValueError("모델이 적합되지 않았습니다")
        
        return self.results.get('fit_indices', {})
    
    def get_model_summary(self) -> str:
        """모델 요약 문자열 반환"""
        if not self.fitted:
            raise ValueError("모델이 적합되지 않았습니다")
        
        summary_lines = []
        summary_lines.append("=== Factor Analysis Results Summary ===")
        summary_lines.append(f"Sample size: {self.results['model_info']['n_observations']}")
        summary_lines.append(f"Variables: {self.results['model_info']['n_variables']}")
        summary_lines.append(f"Estimator: {self.results['model_info']['estimator']}")
        summary_lines.append("")
        
        # 적합도 지수
        fit_indices = self.get_fit_indices()
        if fit_indices:
            summary_lines.append("Fit Indices:")
            for index, value in fit_indices.items():
                summary_lines.append(f"  {index}: {value}")
        
        return "\n".join(summary_lines)


class FactorAnalyzer:
    """Factor Analysis 통합 클래스"""
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None,
                 config: Optional[FactorAnalysisConfig] = None):
        """
        Factor Analyzer 초기화
        
        Args:
            data_dir (Optional[Union[str, Path]]): 데이터 디렉토리
            config (Optional[FactorAnalysisConfig]): 분석 설정
        """
        self.data_loader = FactorDataLoader(data_dir)
        self.analyzer = SemopyAnalyzer(config)
        self.config = config if config is not None else get_default_config()
    
    def analyze_single_factor(self, factor_name: str) -> Dict[str, Any]:
        """
        단일 요인 분석
        
        Args:
            factor_name (str): 분석할 요인 이름
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        from .config import create_factor_model_spec
        
        # 데이터 로딩
        data = self.data_loader.load_single_factor(factor_name)
        
        # 모델 스펙 생성
        model_spec = create_factor_model_spec(single_factor=factor_name)
        
        # 분석 실행
        results = self.analyzer.fit_model(data, model_spec)
        results['analysis_type'] = 'single_factor'
        results['factor_name'] = factor_name
        
        return results
    
    def analyze_multiple_factors(self, factor_names: List[str]) -> Dict[str, Any]:
        """
        다중 요인 분석
        
        Args:
            factor_names (List[str]): 분석할 요인 이름들
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        from .config import create_factor_model_spec
        
        # 데이터 로딩 및 병합
        factor_data = self.data_loader.load_multiple_factors(factor_names)
        merged_data = self.data_loader.merge_factors_for_analysis(factor_data)
        
        # 모델 스펙 생성
        model_spec = create_factor_model_spec(factor_names=factor_names)
        
        # 분석 실행
        results = self.analyzer.fit_model(merged_data, model_spec)
        results['analysis_type'] = 'multiple_factors'
        results['factor_names'] = factor_names
        
        return results


def analyze_factor_loading(factor_names: Union[str, List[str]],
                          data_dir: Optional[Union[str, Path]] = None,
                          config: Optional[FactorAnalysisConfig] = None) -> Dict[str, Any]:
    """
    Factor loading 분석을 수행하는 편의 함수
    
    Args:
        factor_names (Union[str, List[str]]): 분석할 요인 이름(들)
        data_dir (Optional[Union[str, Path]]): 데이터 디렉토리
        config (Optional[FactorAnalysisConfig]): 분석 설정
        
    Returns:
        Dict[str, Any]: 분석 결과
    """
    analyzer = FactorAnalyzer(data_dir, config)
    
    if isinstance(factor_names, str):
        return analyzer.analyze_single_factor(factor_names)
    else:
        return analyzer.analyze_multiple_factors(factor_names)
