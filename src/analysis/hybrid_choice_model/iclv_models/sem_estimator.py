"""
SEM Estimator for Sequential ICLV Estimation

이 모듈은 순차추정 1단계(측정모델 + 구조모델 통합)를 위한 SEM 추정기를 제공합니다.
기존 factor_analysis 모듈의 SemopyAnalyzer를 재사용하여 코드 중복을 최소화합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

# semopy 임포트
try:
    from semopy import Model
    from semopy.stats import calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    logging.warning("semopy 라이브러리를 찾을 수 없습니다. pip install semopy로 설치해주세요.")

# 기존 factor_analysis 모듈 임포트
try:
    from ...factor_analysis.factor_analyzer import SemopyAnalyzer
    from ...factor_analysis.config import FactorAnalysisConfig
    FACTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    FACTOR_ANALYSIS_AVAILABLE = False
    logging.warning("factor_analysis 모듈을 찾을 수 없습니다.")

from .multi_latent_measurement import MultiLatentMeasurement
from .multi_latent_structural import MultiLatentStructural

logger = logging.getLogger(__name__)


class SEMEstimator:
    """
    SEM 방식 추정기 (측정모델 + 구조모델 통합)
    
    순차추정 1단계에서 사용됩니다.
    기존 SemopyAnalyzer를 재사용하여 측정모델과 구조모델을 통합 추정합니다.
    
    Features:
        - 측정모델 (CFA): 잠재변수 =~ 관측지표
        - 구조모델 (경로분석): 잠재변수 ~ 잠재변수
        - 요인점수 추출: predict_factors() 또는 수동 계산
        - 적합도 지수: CFI, TLI, RMSEA, SRMR
    
    Example:
        >>> sem_estimator = SEMEstimator()
        >>> results = sem_estimator.fit(data, measurement_model, structural_model)
        >>> factor_scores = results['factor_scores']
        >>> fit_indices = results['fit_indices']
    """
    
    def __init__(self, config: Optional[FactorAnalysisConfig] = None):
        """
        초기화
        
        Args:
            config: Factor Analysis 설정 (선택)
        """
        if not SEMOPY_AVAILABLE:
            raise ImportError("semopy 라이브러리가 필요합니다. pip install semopy로 설치하세요.")
        
        if not FACTOR_ANALYSIS_AVAILABLE:
            raise ImportError("factor_analysis 모듈이 필요합니다.")
        
        # 기존 SemopyAnalyzer 재사용
        self.analyzer = SemopyAnalyzer(config)
        self.model = None
        self.fitted = False
        
        logger.info("SEMEstimator 초기화 완료")
    
    def fit_cfa_only(self, data: pd.DataFrame,
                    measurement_model: MultiLatentMeasurement) -> Dict[str, Any]:
        """
        CFA만 실행하여 잠재변수 간 상관관계 추출

        구조모델 없이 측정모델만 추정하여 잠재변수 간 상관관계를 확인합니다.
        이후 유의한 상관관계를 바탕으로 구조모델을 설정할 수 있습니다.

        Args:
            data: 분석 데이터
            measurement_model: 다중 잠재변수 측정모델

        Returns:
            {
                'model': semopy Model 객체,
                'factor_scores': Dict[str, np.ndarray],  # 요인점수
                'params': pd.DataFrame,  # 모든 파라미터
                'loadings': pd.DataFrame,  # 요인적재량
                'correlations': pd.DataFrame,  # 잠재변수 간 상관관계 (~~)
                'fit_indices': Dict[str, float],  # 적합도 지수
                'log_likelihood': float
            }
        """
        logger.info("CFA 전용 추정 시작 (구조모델 없음)")

        # 0. 개인별 unique 데이터 추출
        individual_col = 'respondent_id'
        if individual_col not in data.columns:
            individual_col = 'id' if 'id' in data.columns else data.columns[0]

        all_indicators = []
        for config in measurement_model.configs.values():
            all_indicators.extend(config.indicators)

        unique_data = data.groupby(individual_col)[all_indicators].first().reset_index()
        logger.info(f"원본 데이터: {len(data)}행 → 개인별 unique 데이터: {len(unique_data)}행")

        # 1. CFA 모델 스펙 생성 (측정모델만)
        model_spec = self._create_cfa_spec(measurement_model)
        logger.info(f"CFA 모델 스펙 생성 완료:\n{model_spec}")

        # 2. SemopyAnalyzer로 CFA 추정
        results = self.analyzer.fit_model(unique_data, model_spec)

        # 3. 모델 객체 저장
        self.model = self.analyzer.model
        self.fitted = True

        # 4. 요인점수 추출
        factor_scores = self._extract_factor_scores(unique_data, measurement_model)

        # 5. 파라미터 추출
        params = self.model.inspect(std_est=True)

        # 잠재변수 목록
        latent_vars = list(measurement_model.configs.keys())

        # 요인적재량
        loadings = params[
            (params['op'] == '~') &
            (params['rval'].isin(latent_vars)) &
            (~params['lval'].isin(latent_vars))
        ].copy()

        # 잠재변수 간 상관관계 (공분산)
        correlations = params[
            (params['op'] == '~~') &
            (params['lval'].isin(latent_vars)) &
            (params['rval'].isin(latent_vars)) &
            (params['lval'] != params['rval'])  # 자기 자신 제외
        ].copy()

        logger.info(f"잠재변수 간 상관관계: {len(correlations)}개")

        # 6. 적합도 지수
        fit_indices = results.get('fit_indices', {})

        # 7. 로그우도
        log_likelihood = self._calculate_log_likelihood()

        return {
            'model': self.model,
            'factor_scores': factor_scores,
            'params': params,
            'loadings': loadings,
            'correlations': correlations,
            'fit_indices': fit_indices,
            'log_likelihood': log_likelihood
        }

    def _create_cfa_spec(self, measurement_model: MultiLatentMeasurement) -> str:
        """
        CFA 전용 모델 스펙 생성 (측정모델만)

        Args:
            measurement_model: 다중 잠재변수 측정모델

        Returns:
            semopy 모델 스펙 문자열

        Example:
            ```
            # Measurement Model (CFA)
            health_concern =~ q6 + q7 + q8 + q9 + q10 + q11
            perceived_benefit =~ q12 + q13 + q14 + q15 + q16 + q17
            purchase_intention =~ q18 + q19 + q20 + q21 + q22 + q23
            ```
        """
        spec_lines = []
        spec_lines.append("# Measurement Model (CFA)")

        for lv_name, config in measurement_model.configs.items():
            indicators = " + ".join(config.indicators)
            spec_lines.append(f"{lv_name} =~ {indicators}")

        model_spec = "\n".join(spec_lines)
        return model_spec

    def fit(self, data: pd.DataFrame,
            measurement_model: MultiLatentMeasurement,
            structural_model: MultiLatentStructural) -> Dict[str, Any]:
        """
        측정모델 + 구조모델 통합 추정 (SEM 방식)

        Args:
            data: 분석 데이터
            measurement_model: 다중 잠재변수 측정모델
            structural_model: 다중 잠재변수 구조모델

        Returns:
            {
                'model': semopy Model 객체,
                'factor_scores': Dict[str, np.ndarray],  # 요인점수
                'params': pd.DataFrame,  # 모든 파라미터
                'loadings': pd.DataFrame,  # 요인적재량
                'measurement_errors': pd.DataFrame,  # 측정 오차분산
                'paths': pd.DataFrame,  # 구조 경로계수
                'structural_errors': pd.DataFrame,  # 구조 오차분산
                'lv_variances': pd.DataFrame,  # 잠재변수 분산
                'fit_indices': Dict[str, float],  # 적합도 지수
                'log_likelihood': float
            }
        """
        logger.info("SEM 방식 추정 시작 (측정모델 + 구조모델 통합)")

        # 0. 개인별 unique 데이터 추출 (중요!)
        # DCE 데이터는 개인×choice_set×alternative 형태이므로
        # SEM 추정을 위해 개인별로 하나의 행만 사용
        individual_col = 'respondent_id'  # 또는 'id'
        if individual_col not in data.columns:
            individual_col = 'id' if 'id' in data.columns else data.columns[0]

        # 잠재변수 지표 컬럼 추출
        all_indicators = []
        for lv_config in measurement_model.configs.values():
            all_indicators.extend(lv_config.indicators)

        # 개인별 첫 번째 행만 선택 (잠재변수 지표는 개인별로 동일)
        unique_data = data.groupby(individual_col)[all_indicators].first().reset_index()

        logger.info(f"원본 데이터: {len(data)}행 → 개인별 unique 데이터: {len(unique_data)}행")

        # 1. 모델 스펙 생성
        model_spec = self._create_sem_spec(measurement_model, structural_model)
        logger.info(f"모델 스펙 생성 완료:\n{model_spec}")

        # 2. 기존 SemopyAnalyzer.fit_model() 재사용 (unique 데이터 사용)
        results = self.analyzer.fit_model(unique_data, model_spec)

        # 3. 모델 객체 저장
        self.model = self.analyzer.model
        self.fitted = True

        # 4. 요인점수 추출 (unique 데이터 사용)
        factor_scores = self._extract_factor_scores(unique_data, measurement_model)

        # 5. 파라미터 추출 (수정됨)
        extracted_params = self._extract_parameters(measurement_model, structural_model)

        # 6. 로그우도 계산
        log_likelihood = self._calculate_log_likelihood()

        logger.info("SEM 추정 완료")

        return {
            'model': self.model,
            'factor_scores': factor_scores,
            'params': extracted_params['all_params'],
            'loadings': extracted_params['loadings'],
            'measurement_errors': extracted_params['measurement_errors'],
            'paths': extracted_params['paths'],
            'structural_errors': extracted_params['structural_errors'],
            'lv_variances': extracted_params['lv_variances'],
            'fit_indices': results.get('fit_indices', {}),
            'log_likelihood': log_likelihood
        }

    def _extract_parameters(self, measurement_model: MultiLatentMeasurement,
                           structural_model: MultiLatentStructural) -> Dict[str, pd.DataFrame]:
        """
        모든 파라미터 추출 (측정모델 + 구조모델)

        semopy의 inspect() 결과에서 파라미터를 올바르게 필터링합니다.

        Args:
            measurement_model: 측정모델
            structural_model: 구조모델

        Returns:
            {
                'all_params': pd.DataFrame,         # 전체 파라미터
                'loadings': pd.DataFrame,           # 요인적재량 (λ)
                'measurement_errors': pd.DataFrame, # 측정 오차분산 (θ)
                'paths': pd.DataFrame,              # 경로계수 (γ)
                'structural_errors': pd.DataFrame,  # 구조 오차분산 (ψ)
                'lv_variances': pd.DataFrame        # 잠재변수 분산 (φ)
            }

        Note:
            semopy는 '=~' 대신 '~'를 사용하므로, lval/rval이 잠재변수인지 확인하여 구분합니다.

            - 요인적재량: op == '~' AND rval이 잠재변수 AND lval이 관측변수
            - 경로계수: op == '~' AND lval, rval 모두 잠재변수
            - 측정 오차분산: op == '~~' AND lval == rval AND 관측변수
            - 구조 오차분산: op == '~~' AND lval == rval AND 내생 잠재변수
            - 외생 LV 분산: op == '~~' AND lval == rval AND 외생 잠재변수
        """
        if not self.fitted:
            raise ValueError("모델이 추정되지 않았습니다. fit()을 먼저 실행하세요.")

        # 전체 파라미터 (표준화 계수 포함)
        params = self.model.inspect(std_est=True)

        # 잠재변수 목록
        latent_vars = list(measurement_model.configs.keys())

        logger.info(f"파라미터 추출 시작 (잠재변수: {latent_vars})")

        # 1. 요인적재량: op == '~' AND rval이 잠재변수 AND lval이 관측변수
        loadings = params[
            (params['op'] == '~') &
            (params['rval'].isin(latent_vars)) &
            (~params['lval'].isin(latent_vars))
        ].copy()
        logger.info(f"  요인적재량: {len(loadings)}개")

        # 2. 경로계수: op == '~' AND lval, rval 모두 잠재변수
        paths = params[
            (params['op'] == '~') &
            (params['lval'].isin(latent_vars)) &
            (params['rval'].isin(latent_vars))
        ].copy()
        logger.info(f"  경로계수: {len(paths)}개")

        # 3. 측정 오차분산: op == '~~' AND lval == rval AND 관측변수
        measurement_errors = params[
            (params['op'] == '~~') &
            (params['lval'] == params['rval']) &
            (~params['lval'].isin(latent_vars))
        ].copy()
        logger.info(f"  측정 오차분산: {len(measurement_errors)}개")

        # 4. 내생 잠재변수 목록 (구조 오차분산 추출용)
        endogenous_lvs = []
        if structural_model.is_hierarchical:
            # 계층적 구조: target들이 내생 LV
            for path in structural_model.hierarchical_paths:
                if path['target'] not in endogenous_lvs:
                    endogenous_lvs.append(path['target'])
        else:
            # 병렬 구조: endogenous_lv가 내생 LV
            endogenous_lvs = [structural_model.endogenous_lv]

        # 5. 구조 오차분산: op == '~~' AND lval == rval AND 내생 잠재변수
        structural_errors = params[
            (params['op'] == '~~') &
            (params['lval'] == params['rval']) &
            (params['lval'].isin(endogenous_lvs))
        ].copy()
        logger.info(f"  구조 오차분산 (내생 LV: {endogenous_lvs}): {len(structural_errors)}개")

        # 6. 외생 잠재변수 분산: op == '~~' AND lval == rval AND 외생 잠재변수
        exogenous_lvs = structural_model.exogenous_lvs
        lv_variances = params[
            (params['op'] == '~~') &
            (params['lval'] == params['rval']) &
            (params['lval'].isin(exogenous_lvs))
        ].copy()
        logger.info(f"  외생 LV 분산 (외생 LV: {exogenous_lvs}): {len(lv_variances)}개")

        return {
            'all_params': params,
            'loadings': loadings,
            'measurement_errors': measurement_errors,
            'paths': paths,
            'structural_errors': structural_errors,
            'lv_variances': lv_variances
        }

    def _create_sem_spec(self, measurement_model: MultiLatentMeasurement,
                        structural_model: MultiLatentStructural) -> str:
        """
        측정모델 + 구조모델 통합 스펙 생성

        Args:
            measurement_model: 다중 잠재변수 측정모델
            structural_model: 다중 잠재변수 구조모델

        Returns:
            semopy 모델 스펙 문자열

        Example:
            ```
            # Measurement Model
            health_concern =~ q6 + q7 + q8 + q9 + q10 + q11
            perceived_benefit =~ q12 + q13 + q14 + q15 + q16 + q17
            purchase_intention =~ q18 + q19 + q20 + q21 + q22 + q23

            # Structural Model
            perceived_benefit ~ health_concern
            purchase_intention ~ perceived_benefit
            ```
        """
        spec_lines = []

        # 1. 측정모델 (CFA)
        spec_lines.append("# Measurement Model (CFA)")
        for lv_name, config in measurement_model.configs.items():
            indicators = " + ".join(config.indicators)
            spec_lines.append(f"{lv_name} =~ {indicators}")

        # 2. 구조모델 (경로분석)
        spec_lines.append("\n# Structural Model (Path Analysis)")

        if structural_model.is_hierarchical:
            # 계층적 구조
            logger.info("계층적 구조모델 스펙 생성")
            for path in structural_model.hierarchical_paths:
                target = path['target']
                predictors = " + ".join(path['predictors'])
                spec_lines.append(f"{target} ~ {predictors}")
        else:
            # 병렬 구조 (하위 호환)
            logger.info("병렬 구조모델 스펙 생성")
            endogenous_lv = structural_model.endogenous_lv
            exogenous_lvs = " + ".join(structural_model.exogenous_lvs)

            # 공변량 추가 (있는 경우)
            if structural_model.covariates:
                covariates = " + ".join(structural_model.covariates)
                spec_lines.append(f"{endogenous_lv} ~ {exogenous_lvs} + {covariates}")
            else:
                spec_lines.append(f"{endogenous_lv} ~ {exogenous_lvs}")

        model_spec = "\n".join(spec_lines)
        return model_spec

    def _extract_factor_scores(self, data: pd.DataFrame,
                               measurement_model: MultiLatentMeasurement) -> Dict[str, np.ndarray]:
        """
        요인점수 추출

        semopy의 predict_factors() 사용 (가능한 경우)
        불가능하면 수동 계산 (Bartlett 방법)

        Args:
            data: 분석 데이터
            measurement_model: 측정모델

        Returns:
            {
                'health_concern': np.ndarray (n_obs,),
                'perceived_benefit': np.ndarray (n_obs,),
                ...
            }
        """
        if not self.fitted:
            raise ValueError("모델이 추정되지 않았습니다. fit()을 먼저 실행하세요.")

        logger.info("요인점수 추출 시작")

        # semopy의 predict_factors() 사용 시도
        try:
            factor_scores_df = self.model.predict_factors(data)

            # Dict 형태로 변환
            factor_scores = {}
            for col in factor_scores_df.columns:
                factor_scores[col] = factor_scores_df[col].values

            logger.info(f"요인점수 추출 완료 (semopy.predict_factors): {list(factor_scores.keys())}")
            return factor_scores

        except (AttributeError, NotImplementedError) as e:
            # semopy 구버전 또는 미지원: 수동 계산
            logger.warning(f"semopy.predict_factors() 미지원: {e}")
            logger.info("수동 요인점수 계산 (Bartlett 방법)")
            return self._manual_factor_scores(data, measurement_model)

    def _manual_factor_scores(self, data: pd.DataFrame,
                              measurement_model: MultiLatentMeasurement) -> Dict[str, np.ndarray]:
        """
        요인점수 수동 계산 (Bartlett 방법)

        Factor Score = (Λ'Λ)^(-1) Λ' X

        Args:
            data: 분석 데이터
            measurement_model: 측정모델

        Returns:
            요인점수 딕셔너리
        """
        # 파라미터 추출
        params = self.model.inspect()
        loadings = params[params['op'] == '=~'].copy()

        factor_scores = {}

        # 각 잠재변수별로 요인점수 계산
        for lv_name in measurement_model.configs.keys():
            # 이 잠재변수의 지표들과 적재량
            lv_loadings = loadings[loadings['rval'] == lv_name]

            if len(lv_loadings) == 0:
                logger.warning(f"잠재변수 '{lv_name}'의 적재량을 찾을 수 없습니다.")
                continue

            indicators = lv_loadings['lval'].values
            lambda_values = lv_loadings['Estimate'].values

            # 데이터 추출
            X = data[indicators].values  # (n_obs, n_indicators)
            Lambda = lambda_values.reshape(-1, 1)  # (n_indicators, 1)

            # Bartlett 방법: (Λ'Λ)^(-1) Λ' X'
            Lambda_T_Lambda = Lambda.T @ Lambda  # (1, 1)
            Lambda_T_Lambda_inv = 1.0 / Lambda_T_Lambda[0, 0]  # 스칼라

            # 요인점수 계산
            factor_score = Lambda_T_Lambda_inv * (Lambda.T @ X.T)  # (1, n_obs)
            factor_scores[lv_name] = factor_score.flatten()  # (n_obs,)

        logger.info(f"수동 요인점수 계산 완료: {list(factor_scores.keys())}")
        return factor_scores

    def _calculate_log_likelihood(self) -> float:
        """
        로그우도 계산

        semopy 모델의 로그우도 추출

        semopy는 MLW (Maximum Likelihood Wishart) objective function을 최소화합니다.
        로그우도 = -0.5 * objective_value

        Returns:
            로그우도 값
        """
        if not self.fitted:
            raise ValueError("모델이 추정되지 않았습니다.")

        try:
            # semopy는 last_result.fun에 objective function 값을 저장
            if hasattr(self.model, 'last_result') and hasattr(self.model.last_result, 'fun'):
                # MLW objective function을 로그우도로 변환
                log_likelihood = -0.5 * self.model.last_result.fun
                logger.info(f"로그우도: {log_likelihood:.4f} (MLW objective: {self.model.last_result.fun:.4f})")
                return float(log_likelihood)
            else:
                logger.warning("로그우도를 추출할 수 없습니다. (last_result.fun 없음)")
                return np.nan

        except Exception as e:
            logger.warning(f"로그우도 계산 오류: {e}")
            return np.nan

    def get_model_summary(self, measurement_model: Optional[MultiLatentMeasurement] = None,
                         structural_model: Optional[MultiLatentStructural] = None) -> str:
        """
        모델 요약 문자열 반환

        Args:
            measurement_model: 측정모델 (파라미터 추출용)
            structural_model: 구조모델 (파라미터 추출용)

        Returns:
            요약 문자열
        """
        if not self.fitted:
            raise ValueError("모델이 추정되지 않았습니다.")

        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("SEM Estimation Results (Measurement + Structural)")
        summary_lines.append("=" * 60)

        # 파라미터 추출
        if measurement_model and structural_model:
            extracted = self._extract_parameters(measurement_model, structural_model)
            loadings = extracted['loadings']
            paths = extracted['paths']
            measurement_errors = extracted['measurement_errors']
            structural_errors = extracted['structural_errors']
            lv_variances = extracted['lv_variances']
        else:
            # Fallback: 전체 파라미터만 사용
            params = self.model.inspect()
            loadings = params[params['op'] == '~']
            paths = pd.DataFrame()
            measurement_errors = pd.DataFrame()
            structural_errors = pd.DataFrame()
            lv_variances = pd.DataFrame()

        # 측정모델
        summary_lines.append("\n[Measurement Model]")
        if len(loadings) > 0:
            summary_lines.append(f"  Number of latent variables: {len(loadings['rval'].unique())}")
            summary_lines.append(f"  Number of indicators: {len(loadings)}")
            summary_lines.append(f"  Number of measurement errors: {len(measurement_errors)}")
        else:
            summary_lines.append("  No measurement model parameters found")

        # 구조모델
        summary_lines.append("\n[Structural Model]")
        if len(paths) > 0:
            summary_lines.append(f"  Number of paths: {len(paths)}")
            summary_lines.append(f"  Number of structural errors: {len(structural_errors)}")
            summary_lines.append(f"  Number of exogenous LV variances: {len(lv_variances)}")
        else:
            summary_lines.append("  No structural model parameters found")

        # 적합도 지수
        try:
            fit_stats = calc_stats(self.model)
            summary_lines.append("\n[Fit Indices]")
            for index in ['CFI', 'TLI', 'RMSEA', 'SRMR']:
                if index in fit_stats:
                    value = fit_stats[index]
                    if hasattr(value, 'iloc'):
                        value = value.iloc[0]
                    summary_lines.append(f"  {index}: {float(value):.4f}")
        except Exception as e:
            logger.warning(f"적합도 지수 추출 오류: {e}")

        # 로그우도
        log_likelihood = self._calculate_log_likelihood()
        summary_lines.append(f"\n[Log-Likelihood]")
        summary_lines.append(f"  LL: {log_likelihood:.4f}")

        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)


def create_sem_estimator(config: Optional[FactorAnalysisConfig] = None) -> SEMEstimator:
    """
    SEMEstimator 생성 헬퍼 함수

    Args:
        config: Factor Analysis 설정 (선택)

    Returns:
        SEMEstimator 인스턴스

    Example:
        >>> sem_estimator = create_sem_estimator()
        >>> results = sem_estimator.fit(data, measurement_model, structural_model)
    """
    return SEMEstimator(config)

