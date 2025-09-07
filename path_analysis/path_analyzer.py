"""
Path Analysis Core Module

semopy를 사용한 경로분석의 핵심 기능을 제공합니다.
모델 추정, 적합도 평가, 경로계수 추출 등의 기능을 포함합니다.
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
    SEMOPY_AVAILABLE = True
except ImportError as e:
    logging.error("semopy 라이브러리를 찾을 수 없습니다. pip install semopy로 설치해주세요.")
    SEMOPY_AVAILABLE = False

from .config import PathAnalysisConfig, create_default_path_config
from .model_builder import PathModelBuilder

logger = logging.getLogger(__name__)


class PathAnalyzer:
    """경로분석 핵심 클래스"""
    
    def __init__(self, config: Optional[PathAnalysisConfig] = None):
        """
        초기화
        
        Args:
            config (Optional[PathAnalysisConfig]): 분석 설정
        """
        if not SEMOPY_AVAILABLE:
            raise ImportError("semopy 라이브러리가 필요합니다. pip install semopy로 설치하세요.")
        
        self.config = config or create_default_path_config()
        self.model = None
        self.results = None
        self.fitted = False
        self.data = None
        
        logger.info("PathAnalyzer 초기화 완료")
    
    def load_data(self, variables: List[str]) -> pd.DataFrame:
        """
        분석에 필요한 데이터 로드
        
        Args:
            variables (List[str]): 분석할 변수들
            
        Returns:
            pd.DataFrame: 병합된 데이터
        """
        logger.info(f"데이터 로드 중: {variables}")
        
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        
        # 각 변수별 데이터 로드
        dataframes = []
        for variable in variables:
            file_path = data_dir / f"{variable}.csv"
            
            if not file_path.exists():
                raise FileNotFoundError(f"변수 '{variable}'의 데이터 파일을 찾을 수 없습니다: {file_path}")
            
            try:
                df = pd.read_csv(file_path)
                # 'no' 컬럼을 인덱스로 사용
                if 'no' in df.columns:
                    df = df.set_index('no')

                # 컬럼명은 그대로 유지 (q6, q7 등)
                
                dataframes.append(df)
                logger.debug(f"{variable} 데이터 로드 완료: {df.shape}")
                
            except Exception as e:
                logger.error(f"데이터 로드 오류 ({variable}): {e}")
                raise
        
        # 데이터 병합
        if len(dataframes) == 1:
            merged_data = dataframes[0]
        else:
            merged_data = dataframes[0]
            for df in dataframes[1:]:
                merged_data = merged_data.join(df, how='inner')
        
        logger.info(f"데이터 병합 완료: {merged_data.shape}")
        self.data = merged_data
        return merged_data
    
    def fit_model(self, model_spec: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        경로분석 모델 추정
        
        Args:
            model_spec (str): semopy 모델 스펙
            data (Optional[pd.DataFrame]): 분석 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        logger.info("경로분석 모델 추정 시작")
        
        if data is None:
            if self.data is None:
                raise ValueError("분석할 데이터가 없습니다. load_data()를 먼저 실행하거나 data 매개변수를 제공하세요.")
            data = self.data
        
        try:
            # 데이터 전처리
            clean_data = self._prepare_data(data)
            
            # 모델 생성
            self.model = Model(model_spec)
            
            # 모델 추정
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.results = self.model.fit(
                    clean_data,
                    solver=self.config.optimizer
                )
            
            self.fitted = True
            
            # 결과 처리
            analysis_results = self._process_results(clean_data, model_spec)
            
            logger.info("모델 추정 완료")
            return analysis_results
            
        except Exception as e:
            logger.error(f"모델 추정 중 오류 발생: {e}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        logger.info("데이터 전처리 시작")
        
        # 결측치 처리
        if self.config.missing_data_method == 'listwise':
            clean_data = data.dropna()
            logger.info(f"결측치 제거 후 관측치 수: {len(clean_data)}")
        else:
            # FIML은 semopy에서 자동 처리
            clean_data = data.copy()
            logger.info(f"FIML 사용, 전체 관측치 수: {len(clean_data)}")
        
        # 수치형 데이터만 선택
        numeric_columns = clean_data.select_dtypes(include=[np.number]).columns
        clean_data = clean_data[numeric_columns]
        
        logger.info(f"전처리 완료: {clean_data.shape}")
        return clean_data
    
    def _process_results(self, data: pd.DataFrame, model_spec: str) -> Dict[str, Any]:
        """분석 결과 처리"""
        if not self.fitted or self.model is None:
            raise ValueError("모델이 추정되지 않았습니다")
        
        results = {
            'model_info': {
                'n_observations': len(data),
                'n_variables': len(data.columns),
                'estimator': self.config.estimator,
                'optimizer': self.config.optimizer,
                'model_spec': model_spec
            },
            'fit_indices': {},
            'path_coefficients': {},
            'parameter_estimates': {},
            'standardized_results': {},
            'model_object': self.model
        }
        
        try:
            # 모델 적합도 지수 계산
            results['fit_indices'] = self._calculate_fit_indices()

            # 구조적 경로계수 추출 (잠재변수간만)
            results['path_coefficients'] = self._extract_path_coefficients()

            # 모든 가능한 경로 분석
            results['path_analysis'] = self._check_all_possible_paths()
            
            # 파라미터 추정치 추출
            results['parameter_estimates'] = self._extract_parameter_estimates()
            
            # 표준화 결과 (요청시)
            if self.config.standardized:
                results['standardized_results'] = self._extract_standardized_results()
            
        except Exception as e:
            logger.warning(f"결과 처리 중 일부 오류 발생: {e}")
        
        return results
    
    def _calculate_fit_indices(self) -> Dict[str, float]:
        """모델 적합도 지수 계산"""
        try:
            # semopy의 calc_stats 사용
            stats = calc_stats(self.model)
            
            fit_indices = {
                'chi_square': stats.get('chi2', np.nan),
                'df': stats.get('dof', np.nan),
                'p_value': stats.get('chi2_pvalue', np.nan),
                'cfi': stats.get('CFI', np.nan),
                'tli': stats.get('TLI', np.nan),
                'rmsea': stats.get('RMSEA', np.nan),
                'srmr': stats.get('SRMR', np.nan),
                'aic': stats.get('AIC', np.nan),
                'bic': stats.get('BIC', np.nan)
            }
            
            logger.info("적합도 지수 계산 완료")
            return fit_indices
            
        except Exception as e:
            logger.warning(f"적합도 지수 계산 오류: {e}")
            return {}
    
    def _extract_path_coefficients(self) -> Dict[str, Any]:
        """잠재변수간 경로계수 추출 (구조적 경로만)"""
        try:
            # 파라미터 추정치 가져오기
            params = self.model.inspect()

            # 회귀계수만 필터링 (~ 연산자)
            all_path_params = params[params['op'] == '~'].copy()

            # 잠재변수 목록 추출 (관측변수가 아닌 변수들)
            if self.data is not None:
                observed_vars = set(self.data.columns)
                all_vars = set(params['lval'].unique()) | set(params['rval'].unique())
                latent_variables = all_vars - observed_vars
            else:
                # 데이터가 없는 경우 모델 스펙에서 추출
                import re
                model_spec = getattr(self.model, 'model_spec', '')
                latent_pattern = r'(\w+)\s*=~'
                latent_variables = set(re.findall(latent_pattern, str(model_spec)))

            # 잠재변수간 경로만 필터링 (구조적 경로)
            structural_paths = all_path_params[
                all_path_params['lval'].isin(latent_variables) &
                all_path_params['rval'].isin(latent_variables)
            ].copy()

            # 컬럼명 확인 및 안전한 접근
            available_columns = structural_paths.columns.tolist()

            path_coefficients = {
                'coefficients': structural_paths['Estimate'].to_dict() if 'Estimate' in available_columns else {},
                'standard_errors': structural_paths.get('Std. Err', pd.Series()).to_dict() if 'Std. Err' in available_columns else {},
                'z_values': structural_paths.get('z-value', pd.Series()).to_dict() if 'z-value' in available_columns else {},
                'p_values': structural_paths.get('p-value', pd.Series()).to_dict() if 'p-value' in available_columns else {},
                'paths': list(zip(structural_paths['rval'], structural_paths['lval'])) if 'rval' in available_columns and 'lval' in available_columns else [],
                'n_structural_paths': len(structural_paths),
                'latent_variables': list(latent_variables)
            }

            logger.info(f"구조적 경로계수 추출 완료: {len(structural_paths)}개 경로, 잠재변수: {list(latent_variables)}")
            return path_coefficients

        except Exception as e:
            logger.warning(f"경로계수 추출 오류: {e}")
            return {}

    def _check_all_possible_paths(self) -> Dict[str, Any]:
        """모든 가능한 잠재변수간 경로 확인"""
        try:
            # 파라미터 추정치 가져오기
            params = self.model.inspect()

            # 잠재변수 목록 추출 (관측변수가 아닌 변수들)
            if self.data is not None:
                observed_vars = set(self.data.columns)
                all_vars = set(params['lval'].unique()) | set(params['rval'].unique())
                latent_variables = list(all_vars - observed_vars)
            else:
                # 데이터가 없는 경우 모델 스펙에서 추출
                import re
                model_spec = getattr(self.model, 'model_spec', '')
                latent_pattern = r'(\w+)\s*=~'
                latent_variables = list(set(re.findall(latent_pattern, str(model_spec))))

            # 모든 가능한 경로 조합 생성 (자기 자신 제외)
            from itertools import permutations
            all_possible_paths = [(from_var, to_var) for from_var, to_var in permutations(latent_variables, 2)]

            # 현재 모델의 구조적 경로
            structural_params = params[
                (params['op'] == '~') &
                params['lval'].isin(latent_variables) &
                params['rval'].isin(latent_variables)
            ]
            current_paths = [(row['rval'], row['lval']) for _, row in structural_params.iterrows()]

            # 누락된 경로 찾기
            missing_paths = [path for path in all_possible_paths if path not in current_paths]

            path_analysis = {
                'latent_variables': latent_variables,
                'n_latent_variables': len(latent_variables),
                'all_possible_paths': all_possible_paths,
                'n_possible_paths': len(all_possible_paths),
                'current_paths': current_paths,
                'n_current_paths': len(current_paths),
                'missing_paths': missing_paths,
                'n_missing_paths': len(missing_paths),
                'coverage_ratio': len(current_paths) / len(all_possible_paths) if all_possible_paths else 0
            }

            logger.info(f"경로 분석 완료: {len(current_paths)}/{len(all_possible_paths)} 경로 포함 ({path_analysis['coverage_ratio']:.1%})")
            return path_analysis

        except Exception as e:
            logger.warning(f"경로 분석 오류: {e}")
            return {}

    def _extract_parameter_estimates(self) -> Dict[str, Any]:
        """모든 파라미터 추정치 추출"""
        try:
            params = self.model.inspect()
            
            parameter_estimates = {
                'all_parameters': params.to_dict('records'),
                'loadings': params[params['op'] == '=~'].to_dict('records'),
                'regressions': params[params['op'] == '~'].to_dict('records'),
                'covariances': params[params['op'] == '~~'].to_dict('records')
            }
            
            logger.info("파라미터 추정치 추출 완료")
            return parameter_estimates
            
        except Exception as e:
            logger.warning(f"파라미터 추정치 추출 오류: {e}")
            return {}
    
    def _extract_standardized_results(self) -> Dict[str, Any]:
        """표준화 결과 추출"""
        try:
            # 표준화 추정치 가져오기
            std_params = self.model.inspect(std_est=True)

            available_columns = std_params.columns.tolist()

            standardized_results = {
                'standardized_coefficients': std_params.get('Std. Estimate', pd.Series()).to_dict() if 'Std. Estimate' in available_columns else {},
                'standardized_paths': std_params[std_params['op'] == '~'].to_dict('records') if 'op' in available_columns else []
            }

            logger.info("표준화 결과 추출 완료")
            return standardized_results

        except Exception as e:
            logger.warning(f"표준화 결과 추출 오류: {e}")
            return {}


# 편의 함수들
def analyze_path_model(model_spec: str,
                      variables: List[str],
                      config: Optional[PathAnalysisConfig] = None) -> Dict[str, Any]:
    """
    경로분석 실행 편의 함수
    
    Args:
        model_spec (str): semopy 모델 스펙
        variables (List[str]): 분석할 변수들
        config (Optional[PathAnalysisConfig]): 분석 설정
        
    Returns:
        Dict[str, Any]: 분석 결과
    """
    analyzer = PathAnalyzer(config)
    data = analyzer.load_data(variables)
    return analyzer.fit_model(model_spec, data)


def create_path_model(model_type: str, **kwargs) -> str:
    """
    경로모델 생성 편의 함수

    Args:
        model_type (str): 모델 유형
            - 'simple_mediation': 단순 매개모델 (X -> M -> Y)
            - 'multiple_mediation': 다중 매개모델 (X -> M1,M2,... -> Y)
            - 'serial_mediation': 순차 매개모델 (X -> M1 -> M2 -> Y)
            - 'custom': 사용자 정의 경로
            - 'comprehensive': 이론적으로 타당한 모든 경로
            - 'saturated': 모든 가능한 경로 (완전 포화 모델)
        **kwargs: 모델별 매개변수

    Returns:
        str: semopy 모델 스펙
    """
    builder = PathModelBuilder(kwargs.get('data_dir', 'processed_data/survey_data'))
    
    if model_type == 'simple_mediation':
        return builder.create_simple_mediation_model(
            kwargs['independent_var'],
            kwargs['mediator_var'], 
            kwargs['dependent_var']
        )
    elif model_type == 'multiple_mediation':
        return builder.create_multiple_mediation_model(
            kwargs['independent_var'],
            kwargs['mediator_vars'],
            kwargs['dependent_var'],
            kwargs.get('allow_mediator_correlations', True)
        )
    elif model_type == 'serial_mediation':
        return builder.create_serial_mediation_model(
            kwargs['independent_var'],
            kwargs['mediator_vars'],
            kwargs['dependent_var']
        )
    elif model_type == 'custom':
        return builder.create_custom_structural_model(
            kwargs['variables'],
            kwargs['paths'],
            kwargs.get('correlations')
        )
    elif model_type == 'comprehensive':
        return builder.create_comprehensive_structural_model(
            kwargs['variables'],
            kwargs.get('include_bidirectional', True),
            kwargs.get('include_feedback', True)
        )
    elif model_type == 'saturated':
        return builder.create_saturated_structural_model(
            kwargs['variables']
        )
    else:
        raise ValueError(f"지원하지 않는 모델 유형: {model_type}")

    # 지원되는 모델 유형들:
    # - 'simple_mediation': X -> M -> Y
    # - 'multiple_mediation': X -> M1,M2,... -> Y
    # - 'serial_mediation': X -> M1 -> M2 -> Y
    # - 'custom': 사용자 정의 경로
    # - 'comprehensive': 이론적으로 타당한 모든 경로
    # - 'saturated': 모든 가능한 경로 (완전 포화 모델)
