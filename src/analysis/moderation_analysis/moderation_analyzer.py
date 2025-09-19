"""
Moderation Analyzer Module

semopy를 사용한 조절효과 분석 핵심 엔진입니다.
상호작용 효과 검정, 단순기울기 분석, 조건부 효과 계산 등을 수행합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from scipy import stats

# semopy 임포트
try:
    import semopy
    from semopy import Model
    from semopy.stats import calc_stats
except ImportError as e:
    logging.error("semopy 라이브러리를 찾을 수 없습니다. pip install semopy로 설치해주세요.")
    raise e

from .config import ModerationAnalysisConfig
from .data_loader import ModerationDataLoader
from .interaction_builder import InteractionBuilder

logger = logging.getLogger(__name__)


class ModerationAnalyzer:
    """조절효과 분석 핵심 클래스"""
    
    def __init__(self, config: Optional[ModerationAnalysisConfig] = None):
        """
        조절효과 분석기 초기화
        
        Args:
            config (Optional[ModerationAnalysisConfig]): 분석 설정
        """
        from .config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG
        self.data_loader = ModerationDataLoader(config)
        self.interaction_builder = InteractionBuilder(config)
        
        # 분석 결과 저장
        self.model = None
        self.fitted_model = None
        self.data = None
        self.model_spec = None
        
        logger.info("조절효과 분석기 초기화 완료")
    
    def analyze_moderation_effects(self, independent_var: str, dependent_var: str,
                                 moderator_var: str, control_vars: Optional[List[str]] = None,
                                 data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        조절효과 분석 수행
        
        Args:
            independent_var (str): 독립변수
            dependent_var (str): 종속변수
            moderator_var (str): 조절변수
            control_vars (Optional[List[str]]): 통제변수들
            data (Optional[pd.DataFrame]): 분석 데이터 (None이면 자동 로드)
            
        Returns:
            Dict[str, Any]: 조절효과 분석 결과
        """
        logger.info(f"조절효과 분석 시작: {independent_var} × {moderator_var} → {dependent_var}")
        
        try:
            # 1. 데이터 준비
            if data is None:
                self.data = self.data_loader.prepare_moderation_data(
                    independent_var, dependent_var, moderator_var, control_vars
                )
            else:
                self.data = data.copy()
            
            # 2. 상호작용항 생성
            self.data = self.interaction_builder.create_interaction_terms(
                self.data, independent_var, moderator_var
            )
            
            # 3. 모델 스펙 생성
            self.model_spec = self.interaction_builder.build_moderation_model_spec(
                independent_var, dependent_var, moderator_var, control_vars,
                include_measurement_model=False  # 요인점수 사용
            )
            
            # 4. 모델 적합
            self.model = Model(self.model_spec)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_model = self.model.fit(self.data)
            
            # 5. 결과 분석
            results = self._analyze_results(independent_var, dependent_var, moderator_var)
            
            logger.info("조절효과 분석 완료")
            return results
            
        except Exception as e:
            logger.error(f"조절효과 분석 실패: {e}")
            raise
    
    def _analyze_results(self, independent_var: str, dependent_var: str, 
                        moderator_var: str) -> Dict[str, Any]:
        """분석 결과 처리"""
        results = {
            'variables': {
                'independent': independent_var,
                'dependent': dependent_var,
                'moderator': moderator_var,
                'interaction': f"{independent_var}_x_{moderator_var}"
            },
            'model_info': {},
            'coefficients': {},
            'moderation_test': {},
            'simple_slopes': {},
            'conditional_effects': {},
            'fit_indices': {}
        }
        
        # 모델 정보
        results['model_info'] = {
            'n_observations': len(self.data),
            'n_parameters': len(self.model.inspect()),
            'model_specification': self.model_spec
        }
        
        # 계수 추출
        params = self.model.inspect(std_est=True)
        results['coefficients'] = self._extract_coefficients(params, results['variables'])
        
        # 조절효과 유의성 검정
        results['moderation_test'] = self._test_moderation_significance(
            params, results['variables']['interaction']
        )
        
        # 단순기울기 분석
        results['simple_slopes'] = self.calculate_simple_slopes(
            independent_var, dependent_var, moderator_var
        )
        
        # 조건부 효과
        results['conditional_effects'] = self.calculate_conditional_effects(
            independent_var, dependent_var, moderator_var
        )
        
        # 적합도 지수
        try:
            fit_stats = calc_stats(self.model)
            results['fit_indices'] = self._format_fit_indices(fit_stats)
        except Exception as e:
            logger.warning(f"적합도 지수 계산 실패: {e}")
            results['fit_indices'] = {}
        
        return results
    
    def _extract_coefficients(self, params: pd.DataFrame, variables: Dict[str, str]) -> Dict[str, Any]:
        """회귀계수 추출"""
        coefficients = {}
        
        # 구조방정식 파라미터만 필터링
        structural_params = params[params['op'] == '~'].copy()
        
        for _, row in structural_params.iterrows():
            if row['lval'] == variables['dependent']:
                var_name = row['rval']
                coefficients[var_name] = {
                    'estimate': row['Estimate'],
                    'std_error': row['Std. Err'],
                    'z_value': row['z-value'],
                    'p_value': row['p-value'],
                    'std_estimate': row.get('Std. Est', None),
                    'significant': row['p-value'] < 0.05
                }
        
        return coefficients
    
    def _test_moderation_significance(self, params: pd.DataFrame, 
                                    interaction_term: str) -> Dict[str, Any]:
        """조절효과 유의성 검정"""
        # 상호작용항 계수 찾기
        interaction_coeff = params[
            (params['op'] == '~') & (params['rval'] == interaction_term)
        ]
        
        if len(interaction_coeff) == 0:
            return {'significant': False, 'reason': 'interaction_term_not_found'}
        
        coeff_row = interaction_coeff.iloc[0]
        
        moderation_test = {
            'interaction_coefficient': coeff_row['Estimate'],
            'std_error': coeff_row['Std. Err'],
            'z_value': coeff_row['z-value'],
            'p_value': coeff_row['p-value'],
            'significant': coeff_row['p-value'] < 0.05,
            'effect_size': abs(coeff_row['Estimate']),
            'interpretation': self._interpret_moderation_effect(coeff_row['Estimate'], coeff_row['p-value'])
        }
        
        return moderation_test
    
    def _interpret_moderation_effect(self, coefficient: float, p_value: float) -> str:
        """조절효과 해석"""
        if p_value >= 0.05:
            return "조절효과가 통계적으로 유의하지 않음"
        
        if coefficient > 0:
            return "조절변수가 증가할수록 독립변수의 효과가 강화됨"
        else:
            return "조절변수가 증가할수록 독립변수의 효과가 약화됨"
    
    def calculate_simple_slopes(self, independent_var: str, dependent_var: str,
                              moderator_var: str) -> Dict[str, Any]:
        """단순기울기 분석"""
        logger.info("단순기울기 분석 시작")
        
        # 조절변수의 값들 (평균 ± 1SD)
        moderator_values = self._get_moderator_values(moderator_var)
        
        simple_slopes = {}
        
        for level, value in moderator_values.items():
            slope = self._calculate_slope_at_moderator_value(
                independent_var, dependent_var, moderator_var, value
            )
            simple_slopes[level] = slope
        
        logger.info("단순기울기 분석 완료")
        return simple_slopes
    
    def _get_moderator_values(self, moderator_var: str) -> Dict[str, float]:
        """조절변수 값들 계산"""
        moderator_data = self.data[moderator_var]
        mean_val = moderator_data.mean()
        std_val = moderator_data.std()
        
        return {
            'low': mean_val - std_val,
            'mean': mean_val,
            'high': mean_val + std_val
        }
    
    def _calculate_slope_at_moderator_value(self, independent_var: str, dependent_var: str,
                                          moderator_var: str, moderator_value: float) -> Dict[str, Any]:
        """특정 조절변수 값에서의 기울기 계산"""
        # 계수 추출
        params = self.model.inspect()
        structural_params = params[params['op'] == '~']
        
        # 필요한 계수들
        main_effect = 0.0
        interaction_effect = 0.0
        
        for _, row in structural_params.iterrows():
            if row['lval'] == dependent_var:
                if row['rval'] == independent_var:
                    main_effect = row['Estimate']
                elif row['rval'] == f"{independent_var}_x_{moderator_var}":
                    interaction_effect = row['Estimate']
        
        # 단순기울기 = 주효과 + (상호작용효과 × 조절변수값)
        simple_slope = main_effect + (interaction_effect * moderator_value)
        
        # 표준오차 계산 (Delta method)
        std_error = self._calculate_simple_slope_se(
            independent_var, moderator_var, moderator_value
        )
        
        # 유의성 검정
        if std_error > 0:
            t_value = simple_slope / std_error
            p_value = 2 * (1 - stats.norm.cdf(abs(t_value)))
        else:
            t_value = np.nan
            p_value = np.nan
        
        return {
            'moderator_value': moderator_value,
            'simple_slope': simple_slope,
            'std_error': std_error,
            't_value': t_value,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        }
    
    def _calculate_simple_slope_se(self, independent_var: str, moderator_var: str,
                                 moderator_value: float) -> float:
        """단순기울기 표준오차 계산 (Delta method)"""
        try:
            # semopy에서 공분산 행렬 추출이 어려운 경우 근사치 사용
            params = self.model.inspect()
            structural_params = params[params['op'] == '~']

            main_se = 0.0
            interaction_se = 0.0

            for _, row in structural_params.iterrows():
                if row['rval'] == independent_var:
                    main_se = row['Std. Err']
                elif row['rval'] == f"{independent_var}_x_{moderator_var}":
                    interaction_se = row['Std. Err']

            # 근사 표준오차 계산 (보수적 추정)
            variance = (main_se ** 2) + ((moderator_value ** 2) * (interaction_se ** 2))

            return np.sqrt(variance)

        except Exception as e:
            logger.warning(f"단순기울기 표준오차 계산 실패: {e}")
            return 0.0
    
    def calculate_conditional_effects(self, independent_var: str, dependent_var: str,
                                    moderator_var: str) -> Dict[str, Any]:
        """조건부 효과 계산"""
        logger.info("조건부 효과 계산 시작")

        # 조절변수의 다양한 값들에서 효과 계산
        moderator_data = self.data[moderator_var]
        percentiles = [10, 25, 50, 75, 90]
        moderator_values = [np.percentile(moderator_data, p) for p in percentiles]

        conditional_effects = {}

        for i, value in enumerate(moderator_values):
            effect = self._calculate_slope_at_moderator_value(
                independent_var, dependent_var, moderator_var, value
            )
            conditional_effects[f"percentile_{percentiles[i]}"] = effect

        logger.info("조건부 효과 계산 완료")
        return conditional_effects

    def analyze_all_moderation_combinations(self,
                                          variables: Optional[List[str]] = None,
                                          save_results: bool = True,
                                          show_progress: bool = True) -> Dict[str, Any]:
        """
        5개 요인 간 모든 가능한 조절효과 조합 분석

        Args:
            variables (Optional[List[str]]): 분석할 변수들 (기본값: 5개 요인)
            save_results (bool): 결과 저장 여부
            show_progress (bool): 진행 상황 표시 여부

        Returns:
            Dict[str, Any]: 모든 조절효과 분석 결과
        """
        if variables is None:
            from .config import get_factor_items_mapping
            variables = list(get_factor_items_mapping().keys())

        if len(variables) != 5:
            raise ValueError("정확히 5개의 변수가 필요합니다.")

        logger.info(f"5개 요인 간 모든 가능한 조절효과 분석 시작: {variables}")

        all_combinations = []
        results_summary = []

        # 모든 가능한 조합 생성 (60개)
        for dependent in variables:
            for independent in variables:
                if independent != dependent:
                    for moderator in variables:
                        if moderator != dependent and moderator != independent:
                            all_combinations.append({
                                'independent': independent,
                                'dependent': dependent,
                                'moderator': moderator,
                                'name': f"{independent}_x_{moderator}_to_{dependent}"
                            })

        total_combinations = len(all_combinations)
        successful_analyses = 0
        significant_effects = 0

        if show_progress:
            print(f"📊 분석할 조합 수: {total_combinations}개")

        for i, combo in enumerate(all_combinations, 1):
            if show_progress and i % 10 == 0:
                progress = (i / total_combinations) * 100
                print(f"   📈 진행률: {progress:.1f}% ({i}/{total_combinations})")

            try:
                # 조절효과 분석 실행
                results = self.analyze_moderation_effects(
                    independent_var=combo['independent'],
                    dependent_var=combo['dependent'],
                    moderator_var=combo['moderator']
                )

                # 결과 추출
                moderation_test = results.get('moderation_test', {})
                interaction_coef = moderation_test.get('interaction_coefficient', 0)
                p_value = moderation_test.get('p_value', 1)
                significant = moderation_test.get('significant', False)

                # 모델 정보
                model_info = results.get('model_info', {})
                n_obs = model_info.get('n_observations', 0)

                # 적합도 지수
                fit_indices = results.get('fit_indices', {})
                cfi = fit_indices.get('CFI', None)
                rmsea = fit_indices.get('RMSEA', None)

                # 결과 요약
                result_summary = {
                    'combination': combo['name'],
                    'independent': combo['independent'],
                    'dependent': combo['dependent'],
                    'moderator': combo['moderator'],
                    'interaction_coefficient': interaction_coef,
                    'p_value': p_value,
                    'significant': significant,
                    'n_observations': n_obs,
                    'cfi': cfi,
                    'rmsea': rmsea,
                    'status': 'success'
                }

                results_summary.append(result_summary)
                successful_analyses += 1

                if significant:
                    significant_effects += 1

            except Exception as e:
                logger.warning(f"조합 {combo['name']} 분석 실패: {e}")
                result_summary = {
                    'combination': combo['name'],
                    'independent': combo['independent'],
                    'dependent': combo['dependent'],
                    'moderator': combo['moderator'],
                    'error': str(e),
                    'status': 'failed'
                }
                results_summary.append(result_summary)

        # 전체 결과 정리
        comprehensive_results = {
            'summary': {
                'total_combinations': total_combinations,
                'successful_analyses': successful_analyses,
                'significant_effects': significant_effects,
                'success_rate': successful_analyses / total_combinations * 100,
                'significance_rate': significant_effects / successful_analyses * 100 if successful_analyses > 0 else 0
            },
            'detailed_results': results_summary,
            'variables': variables
        }

        if show_progress:
            print(f"✅ 분석 완료: {successful_analyses}/{total_combinations}개 성공")
            print(f"📈 유의한 조절효과: {significant_effects}개")

        # 결과 저장
        if save_results:
            try:
                from .results_exporter import ModerationResultsExporter
                exporter = ModerationResultsExporter(self.config)
                saved_files = exporter.save_comprehensive_results(
                    comprehensive_results,
                    analysis_name="all_combinations"
                )
                comprehensive_results['saved_files'] = saved_files
                logger.info(f"종합 결과 저장 완료: {len(saved_files)}개 파일")
            except Exception as e:
                logger.warning(f"결과 저장 실패: {e}")

        logger.info("모든 조절효과 조합 분석 완료")
        return comprehensive_results
    
    def _format_fit_indices(self, fit_stats: Dict) -> Dict[str, float]:
        """적합도 지수 포맷팅"""
        formatted_indices = {}
        
        # 주요 적합도 지수들
        key_indices = ['CFI', 'TLI', 'RMSEA', 'SRMR', 'AIC', 'BIC', 'Chi-square', 'DoF']
        
        for key in key_indices:
            if key in fit_stats:
                formatted_indices[key] = float(fit_stats[key])
        
        return formatted_indices


# 편의 함수들
def analyze_moderation_effects(independent_var: str, dependent_var: str, moderator_var: str,
                             control_vars: Optional[List[str]] = None,
                             data: Optional[pd.DataFrame] = None,
                             config: Optional[ModerationAnalysisConfig] = None) -> Dict[str, Any]:
    """조절효과 분석 편의 함수"""
    analyzer = ModerationAnalyzer(config)
    return analyzer.analyze_moderation_effects(
        independent_var, dependent_var, moderator_var, control_vars, data
    )


def calculate_simple_slopes(independent_var: str, dependent_var: str, moderator_var: str,
                          data: pd.DataFrame,
                          config: Optional[ModerationAnalysisConfig] = None) -> Dict[str, Any]:
    """단순기울기 분석 편의 함수"""
    analyzer = ModerationAnalyzer(config)
    analyzer.data = data
    return analyzer.calculate_simple_slopes(independent_var, dependent_var, moderator_var)


def calculate_conditional_effects(independent_var: str, dependent_var: str, moderator_var: str,
                                data: pd.DataFrame,
                                config: Optional[ModerationAnalysisConfig] = None) -> Dict[str, Any]:
    """조건부 효과 계산 편의 함수"""
    analyzer = ModerationAnalyzer(config)
    analyzer.data = data
    return analyzer.calculate_conditional_effects(independent_var, dependent_var, moderator_var)


def analyze_all_moderation_combinations(variables: Optional[List[str]] = None,
                                      save_results: bool = True,
                                      show_progress: bool = True,
                                      config: Optional[ModerationAnalysisConfig] = None) -> Dict[str, Any]:
    """5개 요인 간 모든 가능한 조절효과 조합 분석 편의 함수"""
    analyzer = ModerationAnalyzer(config)
    return analyzer.analyze_all_moderation_combinations(variables, save_results, show_progress)


def test_moderation_significance(independent_var: str, dependent_var: str, moderator_var: str,
                               data: pd.DataFrame,
                               config: Optional[ModerationAnalysisConfig] = None) -> Dict[str, Any]:
    """조절효과 유의성 검정 편의 함수"""
    analyzer = ModerationAnalyzer(config)
    results = analyzer.analyze_moderation_effects(
        independent_var, dependent_var, moderator_var, data=data
    )
    return results['moderation_test']
