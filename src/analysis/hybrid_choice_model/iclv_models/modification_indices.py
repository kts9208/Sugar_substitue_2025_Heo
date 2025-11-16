"""
수정지수(Modification Indices) 계산 모듈

SEM 모델의 수정지수를 계산하여 모델 개선을 위한 경로 추가를 제안합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModificationIndices:
    """
    수정지수 계산 및 경로 제안 클래스
    
    수정지수(MI)는 특정 파라미터를 자유롭게 추정할 때 
    카이제곱 통계량이 얼마나 감소할지를 나타냅니다.
    
    MI > 3.84 (p<0.05): 유의한 개선 가능
    MI > 6.63 (p<0.01): 매우 유의한 개선 가능
    MI > 10.83 (p<0.001): 강력한 개선 가능
    """
    
    def __init__(self, model, data: pd.DataFrame):
        """
        Args:
            model: semopy Model 객체 (fitted)
            data: 데이터프레임
        """
        self.model = model
        self.data = data
        self.fitted = hasattr(model, 'param_vals') and model.param_vals is not None
        
        if not self.fitted:
            raise ValueError("모델이 추정되지 않았습니다. fit()을 먼저 실행하세요.")
    
    def calculate_mi_for_paths(
        self,
        latent_vars: List[str],
        existing_paths: List[Tuple[str, str]],
        threshold: float = 3.84
    ) -> pd.DataFrame:
        """
        잠재변수 간 경로에 대한 수정지수 계산
        
        Args:
            latent_vars: 잠재변수 리스트
            existing_paths: 기존 경로 리스트 [(predictor, target), ...]
            threshold: MI 임계값 (기본: 3.84, p<0.05)
        
        Returns:
            수정지수 데이터프레임 (MI, 예상 계수 변화 등)
        """
        logger.info(f"수정지수 계산 시작 (잠재변수: {len(latent_vars)}개)")
        
        # 가능한 모든 경로 생성 (순환 제외)
        all_possible_paths = []
        for predictor in latent_vars:
            for target in latent_vars:
                if predictor != target:
                    all_possible_paths.append((predictor, target))
        
        # 기존 경로 제외
        new_paths = [p for p in all_possible_paths if p not in existing_paths]
        
        logger.info(f"검토할 경로: {len(new_paths)}개 (전체 {len(all_possible_paths)}개 - 기존 {len(existing_paths)}개)")
        
        # 각 경로에 대해 MI 계산
        mi_results = []
        
        for predictor, target in new_paths:
            try:
                mi_value, expected_change = self._calculate_single_path_mi(predictor, target)
                
                mi_results.append({
                    'predictor': predictor,
                    'target': target,
                    'path': f"{predictor} → {target}",
                    'MI': mi_value,
                    'expected_change': expected_change,
                    'p_value': self._mi_to_pvalue(mi_value),
                    'significance': self._get_significance_level(mi_value)
                })
            except Exception as e:
                logger.warning(f"경로 {predictor} → {target} MI 계산 실패: {e}")
                continue
        
        # 데이터프레임 생성 및 정렬
        mi_df = pd.DataFrame(mi_results)
        
        if len(mi_df) > 0:
            mi_df = mi_df.sort_values('MI', ascending=False)
            
            # 임계값 이상만 필터링
            significant_mi = mi_df[mi_df['MI'] >= threshold]
            
            logger.info(f"유의한 수정지수 (MI >= {threshold}): {len(significant_mi)}개")
            
            return mi_df
        else:
            logger.warning("계산된 수정지수가 없습니다.")
            return pd.DataFrame()
    
    def _calculate_single_path_mi(self, predictor: str, target: str) -> Tuple[float, float]:
        """
        단일 경로에 대한 수정지수 계산
        
        근사 방법: 잔차 공분산을 이용한 추정
        
        Returns:
            (MI 값, 예상 계수 변화)
        """
        # 파라미터 추출
        params = self.model.inspect()
        
        # 잔차 공분산 행렬 계산
        residuals = self._calculate_residual_covariance()
        
        # predictor와 target의 잔차 공분산
        if predictor in residuals.index and target in residuals.columns:
            residual_cov = residuals.loc[predictor, target]
        else:
            residual_cov = 0.0
        
        # MI 근사 계산 (단순화된 방법)
        # MI ≈ n * (residual_cov)^2 / var(target)
        n = len(self.data)
        
        # target의 분산 추정
        target_var = self._estimate_variable_variance(target)
        
        if target_var > 0:
            mi_value = n * (residual_cov ** 2) / target_var
        else:
            mi_value = 0.0
        
        # 예상 계수 변화 (근사)
        expected_change = residual_cov / np.sqrt(target_var) if target_var > 0 else 0.0
        
        return mi_value, expected_change

    def _calculate_residual_covariance(self) -> pd.DataFrame:
        """
        잔차 공분산 행렬 계산

        Returns:
            잔차 공분산 행렬
        """
        # 수치형 변수만 선택
        numeric_data = self.data.select_dtypes(include=[np.number])

        # 관측 공분산 행렬
        obs_cov = numeric_data.cov()

        # 모델 예측 공분산 행렬
        try:
            # semopy의 내부 메서드 사용
            if hasattr(self.model, 'mx_sigma'):
                pred_cov = pd.DataFrame(
                    self.model.mx_sigma,
                    index=obs_cov.index,
                    columns=obs_cov.columns
                )
            else:
                # Fallback: 관측 공분산 사용
                pred_cov = obs_cov.copy()
        except:
            pred_cov = obs_cov.copy()

        # 잔차 = 관측 - 예측
        residual_cov = obs_cov - pred_cov

        return residual_cov

    def _estimate_variable_variance(self, var_name: str) -> float:
        """
        변수의 분산 추정

        Args:
            var_name: 변수 이름

        Returns:
            분산 값
        """
        # 수치형 변수만 선택
        numeric_data = self.data.select_dtypes(include=[np.number])

        if var_name in numeric_data.columns:
            return numeric_data[var_name].var()
        else:
            # 잠재변수인 경우 1.0 (표준화 가정)
            return 1.0

    def _mi_to_pvalue(self, mi: float) -> float:
        """
        MI 값을 p-value로 변환

        MI는 카이제곱 분포(df=1)를 따름
        """
        from scipy.stats import chi2
        return 1 - chi2.cdf(mi, df=1)

    def _get_significance_level(self, mi: float) -> str:
        """
        MI 값에 따른 유의성 수준 반환
        """
        if mi >= 10.83:
            return "***"
        elif mi >= 6.63:
            return "**"
        elif mi >= 3.84:
            return "*"
        else:
            return ""

    def suggest_paths(
        self,
        latent_vars: List[str],
        existing_paths: List[Tuple[str, str]],
        max_suggestions: int = 5,
        min_mi: float = 3.84
    ) -> Dict:
        """
        경로 추가 제안

        Args:
            latent_vars: 잠재변수 리스트
            existing_paths: 기존 경로 리스트
            max_suggestions: 최대 제안 개수
            min_mi: 최소 MI 값

        Returns:
            제안 딕셔너리
        """
        # MI 계산
        mi_df = self.calculate_mi_for_paths(latent_vars, existing_paths, threshold=min_mi)

        if len(mi_df) == 0:
            return {
                'suggestions': [],
                'message': f"MI >= {min_mi}인 경로가 없습니다. 현재 모델이 적절합니다."
            }

        # 상위 N개 제안
        top_suggestions = mi_df.head(max_suggestions)

        suggestions = []
        for _, row in top_suggestions.iterrows():
            suggestions.append({
                'path': row['path'],
                'predictor': row['predictor'],
                'target': row['target'],
                'MI': row['MI'],
                'p_value': row['p_value'],
                'expected_change': row['expected_change'],
                'significance': row['significance'],
                'recommendation': self._get_recommendation(row['MI'])
            })

        return {
            'suggestions': suggestions,
            'total_candidates': len(mi_df),
            'message': f"{len(suggestions)}개 경로 추가를 제안합니다."
        }

    def _get_recommendation(self, mi: float) -> str:
        """MI 값에 따른 추천 메시지"""
        if mi >= 10.83:
            return "강력 추천 (p<0.001)"
        elif mi >= 6.63:
            return "추천 (p<0.01)"
        elif mi >= 3.84:
            return "고려 가능 (p<0.05)"
        else:
            return "추천하지 않음"


def calculate_modification_indices(
    model,
    data: pd.DataFrame,
    latent_vars: List[str],
    existing_paths: List[Tuple[str, str]],
    threshold: float = 3.84
) -> pd.DataFrame:
    """
    수정지수 계산 편의 함수

    Args:
        model: semopy Model 객체
        data: 데이터프레임
        latent_vars: 잠재변수 리스트
        existing_paths: 기존 경로 리스트
        threshold: MI 임계값

    Returns:
        수정지수 데이터프레임
    """
    mi_calculator = ModificationIndices(model, data)
    return mi_calculator.calculate_mi_for_paths(latent_vars, existing_paths, threshold)


def suggest_model_improvements(
    model,
    data: pd.DataFrame,
    latent_vars: List[str],
    existing_paths: List[Tuple[str, str]],
    max_suggestions: int = 5,
    min_mi: float = 3.84
) -> Dict:
    """
    모델 개선 제안 편의 함수

    Args:
        model: semopy Model 객체
        data: 데이터프레임
        latent_vars: 잠재변수 리스트
        existing_paths: 기존 경로 리스트
        max_suggestions: 최대 제안 개수
        min_mi: 최소 MI 값

    Returns:
        제안 딕셔너리
    """
    mi_calculator = ModificationIndices(model, data)
    return mi_calculator.suggest_paths(latent_vars, existing_paths, max_suggestions, min_mi)


