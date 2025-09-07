"""
독립적인 신뢰도 및 타당도 계산 모듈

이 모듈은 저장된 요인분석 결과 파일들로부터 다음을 계산합니다:
- Cronbach's Alpha (크론바흐 알파)
- Composite Reliability (CR, 합성신뢰도)
- Average Variance Extracted (AVE, 평균분산추출)
- Discriminant Validity (판별타당도)
- 요인간 상관관계 분석
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import os
import glob
from pathlib import Path

# semopy 상관계수 계산 모듈 임포트 (필요시)
# from .semopy_correlations import SemopyCorrelationExtractor

logger = logging.getLogger(__name__)


class IndependentReliabilityCalculator:
    """독립적인 신뢰도 및 타당도 계산 클래스"""

    def __init__(self, results_dir: str = "factor_analysis_results",
                 survey_data_dir: str = "processed_data/survey_data"):
        """
        신뢰도 계산기 초기화

        Args:
            results_dir (str): 요인분석 결과 파일들이 저장된 디렉토리
            survey_data_dir (str): 원본 설문 데이터가 저장된 디렉토리
        """
        self.results_dir = Path(results_dir)
        self.survey_data_dir = Path(survey_data_dir)

        # 요인간 상관계수 계산기는 별도 모듈로 분리됨
        # self.correlation_calculator = SemopyCorrelationExtractor()

        logger.info("Independent Reliability Calculator 초기화 완료")
    
    def load_latest_analysis_results(self, prefer_post_reverse_coding: bool = True) -> Optional[Dict[str, Any]]:
        """
        가장 최신의 요인분석 결과 파일들을 로드

        Args:
            prefer_post_reverse_coding (bool): 역문항 처리 후 결과를 우선적으로 선택할지 여부

        Returns:
            Dict[str, Any]: 분석 결과 딕셔너리 또는 None
        """
        try:
            # 가장 최신 결과 파일들 찾기
            pattern = str(self.results_dir / "factor_analysis_multiple_factors_*_metadata.json")
            metadata_files = glob.glob(pattern)

            if not metadata_files:
                logger.error("요인분석 결과 파일을 찾을 수 없습니다.")
                return None

            # 역문항 처리 후 결과를 우선적으로 선택
            if prefer_post_reverse_coding:
                latest_metadata_file = self._select_post_reverse_coding_results(metadata_files)
                if latest_metadata_file is None:
                    logger.warning("역문항 처리 후 요인분석 결과를 찾을 수 없습니다. 가장 최신 결과를 사용합니다.")
                    latest_metadata_file = max(metadata_files, key=lambda x: os.path.getmtime(x))
            else:
                # 가장 최신 파일 선택 (수정 시간 기준)
                latest_metadata_file = max(metadata_files, key=lambda x: os.path.getmtime(x))

            base_name = latest_metadata_file.replace('_metadata.json', '')

            # 관련 파일들 로드
            with open(latest_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            loadings_file = f"{base_name}_loadings.csv"
            fit_indices_file = f"{base_name}_fit_indices.csv"

            if not os.path.exists(loadings_file):
                logger.error(f"요인부하량 파일을 찾을 수 없습니다: {loadings_file}")
                return None

            loadings_df = pd.read_csv(loadings_file)
            fit_indices_df = pd.read_csv(fit_indices_file) if os.path.exists(fit_indices_file) else None

            logger.info(f"요인분석 결과 로드 완료: {os.path.basename(latest_metadata_file)}")

            return {
                'metadata': metadata,
                'loadings': loadings_df,
                'fit_indices': fit_indices_df,
                'base_filename': os.path.basename(base_name),
                'file_path': latest_metadata_file,
                'file_timestamp': os.path.getmtime(latest_metadata_file)
            }

        except Exception as e:
            logger.error(f"분석 결과 로드 중 오류: {e}")
            return None

    def _select_post_reverse_coding_results(self, metadata_files: List[str]) -> Optional[str]:
        """
        역문항 처리 후 생성된 요인분석 결과를 선택

        Args:
            metadata_files (List[str]): 메타데이터 파일 목록

        Returns:
            Optional[str]: 선택된 파일 경로 또는 None
        """
        try:
            # 역문항 처리 시점 확인
            reverse_processing_time = self._get_reverse_processing_time()
            if reverse_processing_time is None:
                return None

            # 역문항 처리 이후에 생성된 파일들 필터링
            post_reverse_files = []
            for file_path in metadata_files:
                file_time = os.path.getmtime(file_path)
                if file_time > reverse_processing_time:
                    post_reverse_files.append(file_path)

            if not post_reverse_files:
                return None

            # 가장 최신 파일 선택
            latest_file = max(post_reverse_files, key=lambda x: os.path.getmtime(x))
            logger.info(f"역문항 처리 후 요인분석 결과 선택: {os.path.basename(latest_file)}")
            return latest_file

        except Exception as e:
            logger.warning(f"역문항 처리 후 결과 선택 중 오류: {e}")
            return None

    def _get_reverse_processing_time(self) -> Optional[float]:
        """
        역문항 처리 시점을 확인

        Returns:
            Optional[float]: 역문항 처리 시점 (timestamp) 또는 None
        """
        try:
            # 백업 디렉토리에서 가장 최신 백업 시점 확인
            backup_dir = Path("processed_data/survey_data_backup")
            if not backup_dir.exists():
                return None

            backup_subdirs = list(backup_dir.glob("backup_*"))
            if not backup_subdirs:
                return None

            # 가장 최신 백업 디렉토리의 생성 시간
            latest_backup = max(backup_subdirs, key=lambda x: x.stat().st_mtime)
            return latest_backup.stat().st_mtime

        except Exception as e:
            logger.warning(f"역문항 처리 시점 확인 중 오류: {e}")
            return None

    def load_survey_data(self) -> Dict[str, pd.DataFrame]:
        """
        원본 설문 데이터 로드

        Returns:
            Dict[str, pd.DataFrame]: 요인별 설문 데이터
        """
        try:
            survey_data = {}

            # 각 요인별 데이터 파일 로드
            factor_files = {
                'health_concern': 'health_concern.csv',
                'perceived_benefit': 'perceived_benefit.csv',
                'purchase_intention': 'purchase_intention.csv',
                'perceived_price': 'perceived_price.csv',
                'nutrition_knowledge': 'nutrition_knowledge.csv'
            }

            for factor_name, filename in factor_files.items():
                file_path = self.survey_data_dir / filename
                if file_path.exists():
                    survey_data[factor_name] = pd.read_csv(file_path)
                    logger.info(f"{factor_name} 데이터 로드 완료: {len(survey_data[factor_name])} 행")
                else:
                    logger.warning(f"설문 데이터 파일을 찾을 수 없습니다: {file_path}")

            return survey_data

        except Exception as e:
            logger.error(f"설문 데이터 로드 중 오류: {e}")
            return {}

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
    
    def calculate_factor_reliability_stats_from_loadings(self,
                                                        loadings_df: pd.DataFrame,
                                                        survey_data: Dict[str, pd.DataFrame],
                                                        factor_name: str) -> Dict[str, float]:
        """
        저장된 요인부하량으로부터 단일 요인의 신뢰도 통계 계산

        Args:
            loadings_df (pd.DataFrame): 요인부하량 데이터프레임
            survey_data (Dict[str, pd.DataFrame]): 원본 설문 데이터
            factor_name (str): 요인명

        Returns:
            Dict[str, float]: 신뢰도 통계들
        """
        try:
            # 해당 요인의 요인부하량 추출
            factor_loadings = loadings_df[loadings_df['Factor'] == factor_name]

            if len(factor_loadings) == 0:
                logger.warning(f"요인 '{factor_name}'의 부하량을 찾을 수 없습니다.")
                return self._empty_reliability_stats()

            # 문항 목록과 부하량 추출
            items = factor_loadings['Item'].tolist()
            loadings = factor_loadings['Loading'].values

            # 표준화된 부하량으로 가정 (이미 표준화된 값이라고 가정)
            std_loadings = loadings

            # 오차분산 계산 (1 - λ²)
            error_variances = 1 - (std_loadings ** 2)

            # 1. 크론바흐 알파 계산 (원본 데이터 필요)
            cronbach_alpha = np.nan
            if factor_name in survey_data:
                factor_data = survey_data[factor_name]
                # 'no' 컬럼 제외하고 문항 컬럼들만 사용
                item_columns = [col for col in factor_data.columns if col != 'no']
                cronbach_alpha = self.calculate_cronbach_alpha(factor_data, item_columns)

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
                'sqrt_ave': np.sqrt(ave) if not np.isnan(ave) else np.nan,
                'n_items': len(items),
                'items': items,
                'mean_loading': np.mean(std_loadings),
                'min_loading': np.min(std_loadings),
                'max_loading': np.max(std_loadings),
                'loadings': std_loadings.tolist()
            }

            logger.info(f"{factor_name} 신뢰도 통계 계산 완료")
            return results

        except Exception as e:
            logger.error(f"{factor_name} 신뢰도 통계 계산 중 오류: {e}")
            return self._empty_reliability_stats()

    def _empty_reliability_stats(self) -> Dict[str, Any]:
        """빈 신뢰도 통계 딕셔너리 반환"""
        return {
            'cronbach_alpha': np.nan,
            'composite_reliability': np.nan,
            'ave': np.nan,
            'sqrt_ave': np.nan,
            'n_items': 0,
            'items': [],
            'mean_loading': np.nan,
            'min_loading': np.nan,
            'max_loading': np.nan,
            'loadings': []
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


    def extract_factor_correlations_from_model(self, analysis_results: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        semopy 모델에서 요인간 상관계수 직접 추출 시도

        Args:
            analysis_results (Dict[str, Any]): 요인분석 결과

        Returns:
            Optional[pd.DataFrame]: 요인간 상관계수 매트릭스 또는 None
        """
        try:
            logger.info("semopy 모델에서 요인간 상관계수 추출 시도 중...")

            # 상관계수 추출 기능은 별도 모듈로 분리됨
            # correlations = self.correlation_calculator.extract_from_analysis_results(analysis_results)
            correlations = None

            if correlations is not None and not correlations.empty:
                logger.info("semopy 모델에서 요인간 상관계수 추출 성공")
                return correlations
            else:
                logger.warning("semopy 모델에서 요인간 상관계수 추출 실패")
                return None

        except Exception as e:
            logger.error(f"모델에서 요인간 상관계수 추출 중 오류: {e}")
            return None

    def calculate_factor_correlations(self, loadings_df: pd.DataFrame,
                                     survey_data: Dict[str, pd.DataFrame],
                                     analysis_results: Dict[str, Any] = None) -> pd.DataFrame:
        """
        요인간 상관관계 계산 (semopy 모델 우선, 실패시 원본 데이터 기반)

        Args:
            loadings_df (pd.DataFrame): 요인부하량 데이터프레임
            survey_data (Dict[str, pd.DataFrame]): 원본 설문 데이터
            analysis_results (Dict[str, Any]): 요인분석 결과 (선택사항)

        Returns:
            pd.DataFrame: 요인간 상관관계 매트릭스
        """
        try:
            # 종합적인 상관계수 계산 기능은 별도 모듈로 분리됨
            # results = self.correlation_calculator.calculate_comprehensive(
            #     loadings_df, survey_data, analysis_results
            # )
            results = {'correlations': pd.DataFrame()}

            correlations = results.get('correlations', pd.DataFrame())

            if correlations.empty:
                logger.warning("요인간 상관계수 계산에 실패했습니다.")
                return pd.DataFrame()

            return correlations

        except Exception as e:
            logger.error(f"요인간 상관관계 계산 중 오류: {e}")
            return pd.DataFrame()

    def run_complete_reliability_analysis(self) -> Dict[str, Any]:
        """
        완전한 신뢰도 분석 실행

        Returns:
            Dict[str, Any]: 전체 신뢰도 분석 결과
        """
        try:
            # 1. 분석 결과 로드
            analysis_results = self.load_latest_analysis_results()
            if analysis_results is None:
                return {'error': '분석 결과를 로드할 수 없습니다.'}

            # 2. 설문 데이터 로드
            survey_data = self.load_survey_data()
            if not survey_data:
                logger.warning("설문 데이터를 로드할 수 없습니다. 크론바흐 알파 계산이 제한됩니다.")

            # 3. 각 요인별 신뢰도 계산
            loadings_df = analysis_results['loadings']
            factor_names = analysis_results['metadata']['factor_names']

            reliability_stats = {}
            for factor_name in factor_names:
                stats = self.calculate_factor_reliability_stats_from_loadings(
                    loadings_df, survey_data, factor_name
                )
                reliability_stats[factor_name] = stats

            # 4. 요인간 상관관계 계산
            correlations = self.calculate_factor_correlations(loadings_df, survey_data, analysis_results)

            # 5. 판별타당도 검증
            ave_values = {name: stats['ave'] for name, stats in reliability_stats.items()}
            discriminant_validity = self.calculate_discriminant_validity(ave_values, correlations)

            # 6. 요약 테이블 생성
            summary_table = self.create_reliability_summary_table(reliability_stats)

            return {
                'reliability_stats': reliability_stats,
                'correlations': correlations,
                'discriminant_validity': discriminant_validity,
                'summary_table': summary_table,
                'metadata': analysis_results['metadata'],
                'analysis_timestamp': analysis_results['metadata']['analysis_timestamp']
            }

        except Exception as e:
            logger.error(f"완전한 신뢰도 분석 중 오류: {e}")
            return {'error': str(e)}


def run_independent_reliability_analysis(results_dir: str = "factor_analysis_results",
                                       survey_data_dir: str = "processed_data/survey_data") -> Dict[str, Any]:
    """
    독립적인 신뢰도 분석 실행 편의 함수

    Args:
        results_dir (str): 요인분석 결과 디렉토리
        survey_data_dir (str): 설문 데이터 디렉토리

    Returns:
        Dict[str, Any]: 신뢰도 분석 결과
    """
    calculator = IndependentReliabilityCalculator(results_dir, survey_data_dir)
    return calculator.run_complete_reliability_analysis()
