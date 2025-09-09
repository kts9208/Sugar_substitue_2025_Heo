"""
Moderation Analysis Data Loader Module

processed_data/survey_data의 5개 요인 데이터를 로드하고 조절효과 분석을 위해 준비하는 모듈입니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from .config import ModerationAnalysisConfig, get_factor_items_mapping, get_factor_descriptions

logger = logging.getLogger(__name__)


class ModerationDataLoader:
    """조절효과 분석을 위한 데이터 로더 클래스"""
    
    def __init__(self, config: Optional[ModerationAnalysisConfig] = None):
        """
        데이터 로더 초기화
        
        Args:
            config (Optional[ModerationAnalysisConfig]): 분석 설정
        """
        from .config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG
        self.data_dir = Path(self.config.data_dir)
        self.factor_items = get_factor_items_mapping()
        self.factor_descriptions = get_factor_descriptions()
        
        # 데이터 캐시
        self._factor_data_cache = {}
        self._combined_data_cache = None
        
        logger.info(f"데이터 로더 초기화: {self.data_dir}")
    
    def load_single_factor(self, factor_name: str) -> pd.DataFrame:
        """
        단일 요인 데이터 로드
        
        Args:
            factor_name (str): 요인명 (health_concern, perceived_benefit, etc.)
            
        Returns:
            pd.DataFrame: 요인 데이터
        """
        if factor_name in self._factor_data_cache:
            logger.debug(f"캐시에서 {factor_name} 데이터 반환")
            return self._factor_data_cache[factor_name].copy()
        
        file_path = self.data_dir / f"{factor_name}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"요인 데이터 파일을 찾을 수 없습니다: {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"{factor_name} 데이터 로드 완료: {data.shape}")
            
            # 데이터 검증
            self._validate_factor_data(data, factor_name)
            
            # 캐시에 저장
            self._factor_data_cache[factor_name] = data.copy()
            
            return data
            
        except Exception as e:
            logger.error(f"{factor_name} 데이터 로드 실패: {e}")
            raise
    
    def load_multiple_factors(self, factor_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        여러 요인 데이터 동시 로드
        
        Args:
            factor_names (List[str]): 요인명 리스트
            
        Returns:
            Dict[str, pd.DataFrame]: 요인별 데이터 딕셔너리
        """
        factor_data = {}
        
        for factor_name in factor_names:
            try:
                factor_data[factor_name] = self.load_single_factor(factor_name)
            except Exception as e:
                logger.error(f"{factor_name} 로드 실패: {e}")
                raise
        
        logger.info(f"{len(factor_data)}개 요인 데이터 로드 완료")
        return factor_data
    
    def combine_factor_data(self, factor_names: List[str], 
                           method: str = 'mean') -> pd.DataFrame:
        """
        여러 요인 데이터를 결합하여 조절효과 분석용 데이터셋 생성
        
        Args:
            factor_names (List[str]): 결합할 요인명 리스트
            method (str): 요인 점수 계산 방법 ('mean', 'sum')
            
        Returns:
            pd.DataFrame: 결합된 데이터셋
        """
        cache_key = f"{'-'.join(sorted(factor_names))}_{method}"
        
        if cache_key in self._factor_data_cache:
            logger.debug(f"캐시에서 결합 데이터 반환: {cache_key}")
            return self._factor_data_cache[cache_key].copy()
        
        # 요인 데이터 로드
        factor_data = self.load_multiple_factors(factor_names)
        
        # 공통 인덱스 찾기
        common_index = None
        for factor_name, data in factor_data.items():
            if common_index is None:
                common_index = set(data.index)
            else:
                common_index = common_index.intersection(set(data.index))
        
        if not common_index:
            raise ValueError("요인 데이터 간 공통 인덱스가 없습니다.")
        
        common_index = sorted(list(common_index))
        logger.info(f"공통 관측치 수: {len(common_index)}")
        
        # 결합 데이터셋 생성
        combined_data = pd.DataFrame(index=common_index)
        
        for factor_name in factor_names:
            data = factor_data[factor_name].loc[common_index]
            
            # 요인 점수 계산
            if method == 'mean':
                factor_score = data.mean(axis=1)
            elif method == 'sum':
                factor_score = data.sum(axis=1)
            else:
                raise ValueError(f"지원하지 않는 방법: {method}")
            
            combined_data[factor_name] = factor_score
        
        # 결측치 처리
        combined_data = combined_data.dropna()
        
        logger.info(f"결합 데이터셋 생성 완료: {combined_data.shape}")
        
        # 캐시에 저장
        self._factor_data_cache[cache_key] = combined_data.copy()
        
        return combined_data
    
    def prepare_moderation_data(self, independent_var: str, dependent_var: str,
                               moderator_var: str, control_vars: Optional[List[str]] = None,
                               center_variables: Optional[bool] = None) -> pd.DataFrame:
        """
        조절효과 분석을 위한 데이터 준비
        
        Args:
            independent_var (str): 독립변수
            dependent_var (str): 종속변수
            moderator_var (str): 조절변수
            control_vars (Optional[List[str]]): 통제변수들
            center_variables (Optional[bool]): 변수 중심화 여부
            
        Returns:
            pd.DataFrame: 분석 준비된 데이터
        """
        # 설정에서 중심화 여부 결정
        if center_variables is None:
            center_variables = self.config.center_variables
        
        # 필요한 모든 변수 수집
        all_vars = [independent_var, dependent_var, moderator_var]
        if control_vars:
            all_vars.extend(control_vars)
        
        # 중복 제거
        all_vars = list(set(all_vars))
        
        # 데이터 결합
        data = self.combine_factor_data(all_vars)
        
        # 변수 중심화
        if center_variables:
            data = self._center_variables(data, [independent_var, moderator_var])
            logger.info("변수 중심화 완료")
        
        return data
    
    def _validate_factor_data(self, data: pd.DataFrame, factor_name: str):
        """요인 데이터 유효성 검증"""
        if data.empty:
            raise ValueError(f"{factor_name} 데이터가 비어있습니다.")
        
        # 결측치 비율 확인
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > 0.5:
            logger.warning(f"{factor_name} 데이터의 결측치 비율이 높습니다: {missing_ratio:.2%}")
        
        # 수치형 데이터 확인
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError(f"{factor_name} 데이터에 수치형 변수가 없습니다.")
    
    def _center_variables(self, data: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
        """변수 중심화 (평균 중심화)"""
        centered_data = data.copy()
        
        for var in variables:
            if var in centered_data.columns:
                centered_data[var] = centered_data[var] - centered_data[var].mean()
        
        return centered_data
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 요약 통계"""
        summary = {
            'n_observations': len(data),
            'n_variables': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'descriptive_stats': data.describe().to_dict(),
            'correlations': data.corr().to_dict()
        }
        
        return summary
    
    def get_available_factors(self) -> List[str]:
        """사용 가능한 요인 목록 반환"""
        return list(self.factor_items.keys())


# 편의 함수들
def load_moderation_data(independent_var: str, dependent_var: str, moderator_var: str,
                        control_vars: Optional[List[str]] = None,
                        config: Optional[ModerationAnalysisConfig] = None) -> pd.DataFrame:
    """조절효과 분석 데이터 로드 편의 함수"""
    loader = ModerationDataLoader(config)
    return loader.prepare_moderation_data(
        independent_var, dependent_var, moderator_var, control_vars
    )


def get_available_factors() -> List[str]:
    """사용 가능한 요인 목록 반환 편의 함수"""
    return list(get_factor_items_mapping().keys())


def combine_factor_data(factor_names: List[str], method: str = 'mean',
                       config: Optional[ModerationAnalysisConfig] = None) -> pd.DataFrame:
    """요인 데이터 결합 편의 함수"""
    loader = ModerationDataLoader(config)
    return loader.combine_factor_data(factor_names, method)
