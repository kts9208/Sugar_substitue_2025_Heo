"""
신뢰도 및 타당도 계산 모듈

이 모듈은 요인분석 결과에서 다음을 계산합니다:
- Cronbach's Alpha (크론바흐 알파)
- Composite Reliability (CR, 합성신뢰도)
- Average Variance Extracted (AVE, 평균분산추출)
- Discriminant Validity (판별타당도)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from semopy import Model

logger = logging.getLogger(__name__)


class ReliabilityCalculator:
    """신뢰도 및 타당도 계산 클래스"""
    
    def __init__(self):
        """신뢰도 계산기 초기화"""
        logger.info("Reliability Calculator 초기화 완료")
    
    def calculate_cronbach_alpha(self, data: pd.DataFrame, items: List[str]) -> float:
        """
        크론바흐 알파 계산
        
        Args:
            data (pd.DataFrame): 원시 데이터
            items (List[str]): 해당 요인의 문항들
            
        Returns:
            float: 크론바흐 알파 값
        """
        try:
            # 해당 문항들만 추출
            item_data = data[items].dropna()
            
            if len(item_data) == 0:
                logger.warning("크론바흐 알파 계산용 데이터가 없습니다.")
                return np.nan
            
            # 문항 수
            k = len(items)
            
            if k < 2:
                logger.warning("크론바흐 알파 계산을 위해서는 최소 2개 문항이 필요합니다.")
                return np.nan
            
            # 각 문항의 분산
            item_variances = item_data.var(ddof=1)
            sum_item_var = item_variances.sum()
            
            # 전체 점수의 분산
            total_scores = item_data.sum(axis=1)
            total_var = total_scores.var(ddof=1)
            
            # 크론바흐 알파 계산
            if total_var == 0:
                return np.nan
            
            alpha = (k / (k - 1)) * (1 - sum_item_var / total_var)
            
            logger.info(f"크론바흐 알파 계산 완료: {alpha:.4f}")
            return alpha
            
        except Exception as e:
            logger.error(f"크론바흐 알파 계산 중 오류: {e}")
            return np.nan
    
    def calculate_composite_reliability(self, loadings: List[float], 
                                      error_variances: List[float]) -> float:
        """
        합성신뢰도 (CR) 계산
        
        Args:
            loadings (List[float]): 표준화된 요인부하량들
            error_variances (List[float]): 오차분산들
            
        Returns:
            float: 합성신뢰도 값
        """
        try:
            loadings = np.array(loadings)
            error_variances = np.array(error_variances)
            
            # 요인부하량의 합
            sum_loadings = np.sum(loadings)
            
            # 요인부하량 제곱의 합
            sum_loadings_squared = np.sum(loadings ** 2)
            
            # 오차분산의 합
            sum_error_var = np.sum(error_variances)
            
            # CR 계산: (Σλ)² / [(Σλ)² + Σδ]
            numerator = sum_loadings ** 2
            denominator = numerator + sum_error_var
            
            if denominator == 0:
                return np.nan
            
            cr = numerator / denominator
            
            logger.info(f"합성신뢰도 계산 완료: {cr:.4f}")
            return cr
            
        except Exception as e:
            logger.error(f"합성신뢰도 계산 중 오류: {e}")
            return np.nan
    
    def calculate_ave(self, loadings: List[float], 
                     error_variances: List[float]) -> float:
        """
        평균분산추출 (AVE) 계산
        
        Args:
            loadings (List[float]): 표준화된 요인부하량들
            error_variances (List[float]): 오차분산들
            
        Returns:
            float: AVE 값
        """
        try:
            loadings = np.array(loadings)
            error_variances = np.array(error_variances)
            
            # 요인부하량 제곱의 합
            sum_loadings_squared = np.sum(loadings ** 2)
            
            # 오차분산의 합
            sum_error_var = np.sum(error_variances)
            
            # AVE 계산: Σλ² / (Σλ² + Σδ)
            denominator = sum_loadings_squared + sum_error_var
            
            if denominator == 0:
                return np.nan
            
            ave = sum_loadings_squared / denominator
            
            logger.info(f"AVE 계산 완료: {ave:.4f}")
            return ave
            
        except Exception as e:
            logger.error(f"AVE 계산 중 오류: {e}")
            return np.nan
    
    def calculate_factor_reliability_stats(self, model: Model, 
                                         data: pd.DataFrame,
                                         factor_name: str,
                                         items: List[str]) -> Dict[str, float]:
        """
        단일 요인의 신뢰도 통계 계산
        
        Args:
            model (Model): 적합된 semopy 모델
            data (pd.DataFrame): 원시 데이터
            factor_name (str): 요인명
            items (List[str]): 해당 요인의 문항들
            
        Returns:
            Dict[str, float]: 신뢰도 통계들
        """
        try:
            # 모델 파라미터 추출
            params = model.inspect(std_est=True)  # 표준화 추정값 포함
            
            # 해당 요인의 factor loadings 추출
            factor_loadings = params[
                (params['op'] == '~') & 
                (params['rval'] == factor_name)
            ]
            
            # 표준화된 요인부하량
            std_loadings = factor_loadings['Est. Std'].values
            
            # 오차분산 추출 (1 - λ²)
            error_variances = 1 - (std_loadings ** 2)
            
            # 1. 크론바흐 알파
            cronbach_alpha = self.calculate_cronbach_alpha(data, items)
            
            # 2. 합성신뢰도 (CR)
            composite_reliability = self.calculate_composite_reliability(
                std_loadings, error_variances
            )
            
            # 3. 평균분산추출 (AVE)
            ave = self.calculate_ave(std_loadings, error_variances)
            
            results = {
                'cronbach_alpha': cronbach_alpha,
                'composite_reliability': composite_reliability,
                'ave': ave,
                'n_items': len(items),
                'mean_loading': np.mean(std_loadings),
                'min_loading': np.min(std_loadings),
                'max_loading': np.max(std_loadings)
            }
            
            logger.info(f"{factor_name} 신뢰도 통계 계산 완료")
            return results
            
        except Exception as e:
            logger.error(f"{factor_name} 신뢰도 통계 계산 중 오류: {e}")
            return {
                'cronbach_alpha': np.nan,
                'composite_reliability': np.nan,
                'ave': np.nan,
                'n_items': len(items),
                'mean_loading': np.nan,
                'min_loading': np.nan,
                'max_loading': np.nan
            }
    
    def calculate_discriminant_validity(self, ave_values: Dict[str, float],
                                      correlations: pd.DataFrame) -> Dict[str, Dict[str, bool]]:
        """
        판별타당도 검증 (Fornell-Larcker 기준)
        
        Args:
            ave_values (Dict[str, float]): 각 요인의 AVE 값들
            correlations (pd.DataFrame): 요인간 상관관계 매트릭스
            
        Returns:
            Dict[str, Dict[str, bool]]: 판별타당도 검증 결과
        """
        try:
            results = {}
            factors = list(ave_values.keys())
            
            for i, factor1 in enumerate(factors):
                results[factor1] = {}
                
                for j, factor2 in enumerate(factors):
                    if i != j:
                        # AVE의 제곱근
                        sqrt_ave1 = np.sqrt(ave_values[factor1])
                        sqrt_ave2 = np.sqrt(ave_values[factor2])
                        
                        # 요인간 상관계수
                        correlation = abs(correlations.loc[factor1, factor2])
                        
                        # 판별타당도: √AVE > 상관계수
                        valid = (sqrt_ave1 > correlation) and (sqrt_ave2 > correlation)
                        results[factor1][factor2] = valid
                    else:
                        results[factor1][factor2] = True  # 자기 자신과는 항상 True
            
            logger.info("판별타당도 검증 완료")
            return results
            
        except Exception as e:
            logger.error(f"판별타당도 검증 중 오류: {e}")
            return {}
    
    def create_reliability_summary_table(self, reliability_stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        신뢰도 요약 테이블 생성
        
        Args:
            reliability_stats (Dict[str, Dict[str, float]]): 요인별 신뢰도 통계
            
        Returns:
            pd.DataFrame: 신뢰도 요약 테이블
        """
        try:
            rows = []
            
            for factor_name, stats in reliability_stats.items():
                row = {
                    'Factor': factor_name,
                    'Items': stats.get('n_items', 0),
                    'Cronbach_Alpha': stats.get('cronbach_alpha', np.nan),
                    'Composite_Reliability': stats.get('composite_reliability', np.nan),
                    'AVE': stats.get('ave', np.nan),
                    'Mean_Loading': stats.get('mean_loading', np.nan),
                    'Min_Loading': stats.get('min_loading', np.nan),
                    'Max_Loading': stats.get('max_loading', np.nan)
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # 신뢰도 기준 추가
            df['Alpha_Acceptable'] = df['Cronbach_Alpha'] >= 0.7
            df['CR_Acceptable'] = df['Composite_Reliability'] >= 0.7
            df['AVE_Acceptable'] = df['AVE'] >= 0.5
            
            logger.info("신뢰도 요약 테이블 생성 완료")
            return df
            
        except Exception as e:
            logger.error(f"신뢰도 요약 테이블 생성 중 오류: {e}")
            return pd.DataFrame()


def calculate_reliability_from_results(analysis_results: Dict[str, Any],
                                     raw_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    분석 결과로부터 신뢰도 통계 계산 편의 함수
    
    Args:
        analysis_results (Dict[str, Any]): factor_analyzer의 분석 결과
        raw_data (Optional[pd.DataFrame]): 원시 데이터 (크론바흐 알파 계산용)
        
    Returns:
        Dict[str, Any]: 신뢰도 분석 결과
    """
    calculator = ReliabilityCalculator()
    
    try:
        # 모델 객체 추출
        model = analysis_results.get('model')
        if model is None:
            logger.error("분석 결과에서 모델 객체를 찾을 수 없습니다.")
            return {'error': '모델 객체 없음'}
        
        # 요인 정보 추출
        if analysis_results.get('analysis_type') == 'single_factor':
            factor_name = analysis_results.get('factor_name')
            factor_names = [factor_name]
        else:
            factor_names = analysis_results.get('factor_names', [])
        
        if not factor_names:
            logger.error("요인 정보를 찾을 수 없습니다.")
            return {'error': '요인 정보 없음'}
        
        # 각 요인별 신뢰도 계산
        reliability_stats = {}
        
        for factor_name in factor_names:
            # 해당 요인의 문항들 추출
            factor_loadings = analysis_results['factor_loadings']
            items = factor_loadings[factor_loadings['Factor'] == factor_name]['Item'].tolist()
            
            # 신뢰도 통계 계산
            stats = calculator.calculate_factor_reliability_stats(
                model, raw_data, factor_name, items
            )
            reliability_stats[factor_name] = stats
        
        # 요약 테이블 생성
        summary_table = calculator.create_reliability_summary_table(reliability_stats)
        
        return {
            'reliability_stats': reliability_stats,
            'summary_table': summary_table,
            'calculator': calculator
        }
        
    except Exception as e:
        logger.error(f"신뢰도 계산 중 오류: {e}")
        return {'error': str(e)}
