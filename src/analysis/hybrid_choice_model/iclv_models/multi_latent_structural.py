"""
Multi-Latent Variable Structural Model

다중 잠재변수 구조모델입니다.
외생 잠재변수와 내생 잠재변수의 관계를 모델링합니다.

✅ 디폴트: 계층적 구조 (건강관심도 → 건강유익성 → 구매의도)

구조:
1. 계층적 구조 (hierarchical_paths 지정 시):
   - 1차 LV (외생): LV_i = η_i ~ N(0, 1)
   - 2차+ LV (내생): LV_j = Σ(γ_k * LV_k) + η

2. 병렬 구조 (hierarchical_paths=None, 하위 호환):
   - 외생 LV: LV_i = η_i ~ N(0, 1)
   - 내생 LV: LV_endo = Σ(γ_i * LV_i) + Σ(γ_j * X_j) + η

Author: Sugar Substitute Research Team
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.stats import norm
import logging

from .multi_latent_config import MultiLatentStructuralConfig

logger = logging.getLogger(__name__)


class MultiLatentStructural:
    """
    다중 잠재변수 구조모델

    ✅ 디폴트: 계층적 구조

    계층적 구조 예시:
        1차 LV (외생):
            건강관심도 = η1 ~ N(0, 1)
            가격수준 = η2 ~ N(0, 1)
            영양지식 = η3 ~ N(0, 1)

        2차 LV:
            건강유익성 = γ1*건강관심도 + η2

        3차 LV (내생):
            구매의도 = γ2*건강유익성 + η3

    병렬 구조 예시 (하위 호환):
        외생 LV:
            건강관심도 = η1 ~ N(0, 1)
            건강유익성 = η2 ~ N(0, 1)
            가격수준 = η3 ~ N(0, 1)
            영양지식 = η4 ~ N(0, 1)

        내생 LV:
            구매의도 = γ1*건강관심도 + γ2*건강유익성 + γ3*가격수준 + γ4*영양지식
                     + γ5*age + γ6*gender + γ7*income + η
    """

    def __init__(self, config: MultiLatentStructuralConfig):
        """
        초기화

        Args:
            config: 다중 잠재변수 구조모델 설정
        """
        self.config = config
        self.endogenous_lv = config.endogenous_lv
        self.exogenous_lvs = config.exogenous_lvs
        self.covariates = config.covariates
        self.error_variance = config.error_variance

        # 계층적 구조 여부
        self.is_hierarchical = config.is_hierarchical
        self.hierarchical_paths = config.hierarchical_paths if self.is_hierarchical else None

        # 파라미터 수 계산
        if self.is_hierarchical:
            # 계층적 구조: 각 경로마다 파라미터
            self.n_params = 0
            for path in self.hierarchical_paths:
                self.n_params += len(path['predictors'])
        else:
            # 병렬 구조: 외생 LV + 공변량
            self.n_exo = len(self.exogenous_lvs)
            self.n_cov = len(self.covariates)
            self.n_params = self.n_exo + self.n_cov

        # 로깅
        logger.info(f"MultiLatentStructural 초기화")
        logger.info(f"  구조 유형: {'계층적' if self.is_hierarchical else '병렬'}")

        if self.is_hierarchical:
            logger.info(f"  1차 LV ({len(self.exogenous_lvs)}개): {self.exogenous_lvs}")
            logger.info(f"  계층적 경로:")
            for i, path in enumerate(self.hierarchical_paths):
                logger.info(f"    경로 {i+1}: {path['predictors']} → {path['target']}")
            logger.info(f"  총 파라미터: {self.n_params}개")
        else:
            logger.info(f"  외생 LV ({len(self.exogenous_lvs)}개): {self.exogenous_lvs}")
            logger.info(f"  내생 LV: {self.endogenous_lv}")
            logger.info(f"  공변량 ({len(self.covariates)}개): {self.covariates}")
            logger.info(f"  총 파라미터: {self.n_params}개")
    
    def predict(self, data: pd.DataFrame,
                exo_draws: np.ndarray,
                params: Dict[str, Any],
                endo_draw: float = None,
                higher_order_draws: Dict[str, float] = None) -> Dict[str, float]:
        """
        모든 잠재변수 예측

        ✅ 계층적 구조 지원

        Args:
            data: 개인 데이터 (첫 번째 행의 공변량 사용)
            exo_draws: 1차 LV draws (n_exo,) - 표준정규분포
            params: 구조모델 파라미터
                계층적 구조:
                    {
                        'gamma_health_concern_to_perceived_benefit': float,
                        'gamma_perceived_benefit_to_purchase_intention': float
                    }
                병렬 구조 (하위 호환):
                    {
                        'gamma_lv': np.ndarray (n_exo,),
                        'gamma_x': np.ndarray (n_cov,)
                    }
            endo_draw: 내생 LV 오차항 draw (병렬 구조용, 하위 호환)
            higher_order_draws: 2차+ LV 오차항 draws (계층적 구조용)
                {
                    'perceived_benefit': 0.2,
                    'purchase_intention': -0.1
                }

        Returns:
            모든 잠재변수 값
            {
                'health_concern': 0.5,
                'perceived_benefit': 0.3,
                'perceived_price': -0.2,
                'nutrition_knowledge': 0.8,
                'purchase_intention': 0.6
            }
        """
        latent_vars = {}

        # 1. 1차 LV (외생, 표준정규분포)
        for i, lv_name in enumerate(self.exogenous_lvs):
            latent_vars[lv_name] = exo_draws[i]

        # 2. 2차+ LV
        if self.is_hierarchical:
            # ✅ 계층적 구조
            if higher_order_draws is None:
                higher_order_draws = {}

            # 계층적 경로 순서대로 계산
            for path_idx, path in enumerate(self.hierarchical_paths):
                target = path['target']
                predictors = path['predictors']

                # 평균 계산: Σ(γ_k * LV_k)
                lv_mean = 0.0
                gamma_details = []  # 🔍 디버깅용
                for pred in predictors:
                    param_name = f'gamma_{pred}_to_{target}'
                    if param_name not in params:
                        raise KeyError(f"파라미터 '{param_name}'가 없습니다.")

                    gamma = params[param_name]
                    pred_lv = latent_vars[pred]
                    contribution = gamma * pred_lv
                    lv_mean += contribution
                    gamma_details.append(f"{param_name}={gamma:.4f} × {pred}={pred_lv:.4f} = {contribution:.4f}")

                # 오차항 추가
                error_draw = higher_order_draws.get(target, 0.0)
                error_term = np.sqrt(self.error_variance) * error_draw
                latent_vars[target] = lv_mean + error_term

                # ✅ 디버깅: 모든 경로 로깅 (디버깅 플래그가 설정된 경우에만)
                if hasattr(self, '_debug_predict') and self._debug_predict:
                    if hasattr(self, '_iteration_logger'):
                        logger = self._iteration_logger
                    else:
                        import logging
                        logger = logging.getLogger(__name__)

                    logger.info(f"\n[predict() 디버깅] 경로 #{path_idx+1}: {predictors} → {target}")
                    for detail in gamma_details:
                        logger.info(f"  {detail}")
                    logger.info(f"  lv_mean (합계) = {lv_mean:.4f}")
                    logger.info(f"  error_draw = {error_draw:.4f}")
                    logger.info(f"  error_term = √{self.error_variance:.4f} × {error_draw:.4f} = {error_term:.4f}")
                    logger.info(f"  {target} = {lv_mean:.4f} + {error_term:.4f} = {latent_vars[target]:.4f}")

        else:
            # 병렬 구조 (하위 호환)
            gamma_lv = params['gamma_lv']
            gamma_x = params['gamma_x']

            # 외생 LV 효과
            lv_effect = np.sum(gamma_lv * exo_draws)

            # 공변량 효과 (첫 번째 행 사용 - 개인 특성)
            first_row = data.iloc[0]
            x_effect = 0.0
            for i, var in enumerate(self.covariates):
                if var in first_row.index:
                    value = first_row[var]
                    if pd.isna(value):
                        value = 0.0
                    x_effect += gamma_x[i] * value

            # 내생 LV = 외생 LV 효과 + 공변량 효과 + 오차항
            if endo_draw is None:
                endo_draw = 0.0

            latent_vars[self.endogenous_lv] = (
                lv_effect + x_effect + np.sqrt(self.error_variance) * endo_draw
            )

        return latent_vars
    
    def log_likelihood(self, data: pd.DataFrame,
                      latent_vars: Dict[str, float],
                      exo_draws: np.ndarray,
                      params: Dict[str, Any],
                      endo_draw: float = None,
                      higher_order_draws: Dict[str, float] = None) -> float:
        """
        구조모델 로그우도

        ✅ 계층적 구조 지원

        계층적 구조:
            LL = Σ log P(LV_1st) + Σ log P(LV_higher | LV_predictors)
            - 1차 LV: P(LV_i) = N(0, 1)
            - 2차+ LV: P(LV_j | LV_predictors) = N(Σ(γ_k * LV_k), σ²)

        병렬 구조 (하위 호환):
            LL = Σ log P(LV_exo) + log P(LV_endo | LV_exo, X)
            - 외생 LV: P(LV_i) = N(0, 1)
            - 내생 LV: P(LV_endo | LV_exo, X) = N(γ_lv*LV_exo + γ_x*X, σ²)

        Args:
            data: 개인 데이터
            latent_vars: 모든 잠재변수 값
            exo_draws: 1차 LV draws
            params: 구조모델 파라미터
            endo_draw: 내생 LV 오차항 draw (병렬 구조용)
            higher_order_draws: 2차+ LV 오차항 draws (계층적 구조용)

        Returns:
            구조모델 로그우도
        """
        ll = 0.0

        # 1. 1차 LV 로그우도: N(0, 1)
        for lv_name in self.exogenous_lvs:
            lv = latent_vars[lv_name]
            ll += norm.logpdf(lv, loc=0, scale=1)

        # 2. 2차+ LV 로그우도
        if self.is_hierarchical:
            # ✅ 계층적 구조
            for path_idx, path in enumerate(self.hierarchical_paths):
                target = path['target']
                predictors = path['predictors']

                # 평균 계산: Σ(γ_k * LV_k)
                lv_mean = 0.0
                for pred in predictors:
                    param_name = f'gamma_{pred}_to_{target}'
                    gamma = params[param_name]
                    lv_mean += gamma * latent_vars[pred]

                # 로그우도: N(lv_mean, σ²)
                lv_value = latent_vars[target]
                ll_component = norm.logpdf(lv_value, loc=lv_mean, scale=np.sqrt(self.error_variance))
                ll += ll_component

                # 🔍 디버깅: 구조모델 우도 성분
                if hasattr(self, '_debug_ll') and self._debug_ll:
                    if hasattr(self, '_iteration_logger'):
                        logger = self._iteration_logger
                    else:
                        import logging
                        logger = logging.getLogger(__name__)

                    logger.info(f"[log_likelihood() 디버깅] 경로 #{path_idx+1}: {predictors} → {target}")
                    logger.info(f"  lv_mean = {lv_mean:.4f}")
                    logger.info(f"  lv_value = {lv_value:.4f}")
                    logger.info(f"  ll_component = logpdf({lv_value:.4f} | μ={lv_mean:.4f}, σ={np.sqrt(self.error_variance):.4f}) = {ll_component:.4f}")

        else:
            # 병렬 구조 (하위 호환)
            gamma_lv = params['gamma_lv']
            gamma_x = params['gamma_x']

            # 평균 계산
            lv_effect = np.sum(gamma_lv * exo_draws)

            first_row = data.iloc[0]
            x_effect = 0.0
            for i, var in enumerate(self.covariates):
                if var in first_row.index:
                    value = first_row[var]
                    if pd.isna(value):
                        value = 0.0
                    x_effect += gamma_x[i] * value

            lv_endo_mean = lv_effect + x_effect
            lv_endo = latent_vars[self.endogenous_lv]

            ll += norm.logpdf(lv_endo, loc=lv_endo_mean, scale=np.sqrt(self.error_variance))

        return ll
    
    def get_n_parameters(self) -> int:
        """
        파라미터 수 반환
        
        Returns:
            n_exo + n_cov
        """
        return self.n_params
    
    def initialize_parameters(self) -> Dict[str, Any]:
        """
        파라미터 초기화

        ✅ 계층적 구조 지원

        Returns:
            계층적 구조:
                {
                    'gamma_health_concern_to_perceived_benefit': 0.5,
                    'gamma_perceived_benefit_to_purchase_intention': 0.5
                }

            병렬 구조 (하위 호환):
                {
                    'gamma_lv': np.ndarray (n_exo,),
                    'gamma_x': np.ndarray (n_cov,)
                }
        """
        params = {}

        if self.is_hierarchical:
            # ✅ 계층적 구조: 각 경로마다 파라미터
            for path in self.hierarchical_paths:
                target = path['target']
                predictors = path['predictors']

                for pred in predictors:
                    param_name = f'gamma_{pred}_to_{target}'
                    # 초기값: 0.5 (양의 효과 가정)
                    params[param_name] = 0.5

        else:
            # 병렬 구조 (하위 호환)
            params['gamma_lv'] = np.zeros(len(self.exogenous_lvs))
            params['gamma_x'] = np.zeros(len(self.covariates))

        return params
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        파라미터 유효성 검증

        ✅ 계층적 구조 지원

        Args:
            params: 구조모델 파라미터

        Returns:
            유효하면 True
        """
        if self.is_hierarchical:
            # ✅ 계층적 구조: 각 경로의 파라미터 확인
            for path in self.hierarchical_paths:
                target = path['target']
                predictors = path['predictors']

                for pred in predictors:
                    param_name = f'gamma_{pred}_to_{target}'
                    if param_name not in params:
                        logger.error(f"파라미터 '{param_name}'가 없습니다.")
                        return False

            return True

        else:
            # 병렬 구조 (하위 호환)
            # gamma_lv 검증
            if 'gamma_lv' not in params:
                logger.error("gamma_lv가 없습니다.")
                return False

            gamma_lv = params['gamma_lv']
            if len(gamma_lv) != len(self.exogenous_lvs):
                logger.error(
                    f"gamma_lv 크기 불일치: expected {len(self.exogenous_lvs)}, got {len(gamma_lv)}"
                )
                return False

            # gamma_x 검증
            if 'gamma_x' not in params:
                logger.error("gamma_x가 없습니다.")
                return False

            gamma_x = params['gamma_x']
            if len(gamma_x) != len(self.covariates):
                logger.error(
                    f"gamma_x 크기 불일치: expected {len(self.covariates)}, got {len(gamma_x)}"
                )
                return False

            return True
    
    def get_parameter_names(self) -> List[str]:
        """
        파라미터 이름 반환

        ✅ 계층적 구조 지원

        Returns:
            계층적 구조:
                [
                    'gamma_health_concern_to_perceived_benefit',
                    'gamma_perceived_benefit_to_purchase_intention'
                ]

            병렬 구조 (하위 호환):
                [
                    'gamma_health_concern',
                    'gamma_perceived_benefit',
                    'gamma_perceived_price',
                    'gamma_nutrition_knowledge',
                    'gamma_age_std',
                    'gamma_gender',
                    'gamma_income_std'
                ]
        """
        names = []

        if self.is_hierarchical:
            # ✅ 계층적 구조
            for path in self.hierarchical_paths:
                target = path['target']
                predictors = path['predictors']

                for pred in predictors:
                    param_name = f'gamma_{pred}_to_{target}'
                    names.append(param_name)

        else:
            # 병렬 구조 (하위 호환)
            for lv in self.exogenous_lvs:
                names.append(f'gamma_{lv}')

            for var in self.covariates:
                names.append(f'gamma_{var}')

        return names

    def get_higher_order_lvs(self) -> List[str]:
        """
        고차 잠재변수 (2차 이상) 리스트 반환

        계층적 구조에서 hierarchical_paths의 target들이 고차 잠재변수입니다.

        Returns:
            고차 잠재변수 이름 리스트 (순서 유지)
            예: ['perceived_benefit', 'purchase_intention']
        """
        if not self.is_hierarchical:
            # 병렬 구조에서는 endogenous_lv만 고차 변수
            return [self.endogenous_lv]

        # 계층적 구조: hierarchical_paths의 target들을 순서대로 반환
        higher_order_lvs = []
        for path in self.hierarchical_paths:
            target = path['target']
            if target not in higher_order_lvs:
                higher_order_lvs.append(target)

        return higher_order_lvs

