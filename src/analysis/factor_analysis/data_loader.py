"""
Factor Analysis Data Loader Module

이 모듈은 전처리된 요인별 CSV 파일들을 불러오는 기능을 제공합니다.
기존 전처리 모듈과의 중복을 피하고 factor analysis에 특화된 로딩 기능을 구현합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import sys

# 기존 모듈 임포트를 위한 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent

# FactorConfig 대체 클래스 (간단한 버전)
class FactorConfig:
    """요인 설정을 관리하는 간단한 클래스"""

    @staticmethod
    def get_factor_items(factor_name):
        """요인별 문항 목록 반환"""
        factor_items = {
            'health_concern': ['q1', 'q2', 'q3', 'q4', 'q5'],
            'perceived_benefit': ['q6', 'q7', 'q8', 'q9', 'q10'],
            'purchase_intention': ['q11', 'q12', 'q13', 'q14', 'q15'],
            'perceived_price': ['q16', 'q17', 'q18', 'q19', 'q20'],
            'nutrition_knowledge': ['q21', 'q22', 'q23', 'q24', 'q25']
        }
        return factor_items.get(factor_name, [])

logger = logging.getLogger(__name__)


class FactorDataLoader:
    """전처리된 요인별 CSV 파일들을 로딩하는 클래스"""
    
    def __init__(self, data_dir: Union[str, Path] = None):
        """
        Factor Data Loader 초기화
        
        Args:
            data_dir (Union[str, Path]): 전처리된 데이터가 있는 디렉토리 경로
        """
        if data_dir is None:
            # 기본 경로: processed_data/survey_data
            self.data_dir = project_root / "processed_data" / "survey_data"
        else:
            self.data_dir = Path(data_dir)
        
        self.factor_config = FactorConfig()
        self._validate_data_directory()
    
    def _validate_data_directory(self) -> None:
        """데이터 디렉토리 유효성 검증"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {self.data_dir}")
        
        if not self.data_dir.is_dir():
            raise ValueError(f"경로가 디렉토리가 아닙니다: {self.data_dir}")
    
    def get_available_factor_files(self) -> Dict[str, Path]:
        """
        사용 가능한 요인별 CSV 파일들을 찾아 반환
        
        Returns:
            Dict[str, Path]: {요인명: 파일경로} 딕셔너리
        """
        available_files = {}
        
        for factor_name in self.factor_config.get_all_factors():
            file_path = self.data_dir / f"{factor_name}.csv"
            if file_path.exists():
                available_files[factor_name] = file_path
            else:
                logger.warning(f"요인 파일을 찾을 수 없습니다: {file_path}")
        
        return available_files
    
    def load_single_factor(self, factor_name: str) -> pd.DataFrame:
        """
        단일 요인 데이터를 로딩
        
        Args:
            factor_name (str): 요인 이름
            
        Returns:
            pd.DataFrame: 요인 데이터
        """
        if factor_name not in self.factor_config.get_all_factors():
            raise ValueError(f"알 수 없는 요인: {factor_name}")
        
        file_path = self.data_dir / f"{factor_name}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"요인 파일을 찾을 수 없습니다: {file_path}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            logger.info(f"{factor_name} 데이터 로딩 완료: {df.shape}")
            return self._validate_factor_data(df, factor_name)
        except Exception as e:
            logger.error(f"{factor_name} 데이터 로딩 실패: {e}")
            raise
    
    def load_multiple_factors(self, factor_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        여러 요인 데이터를 한번에 로딩
        
        Args:
            factor_names (List[str]): 요인 이름 리스트
            
        Returns:
            Dict[str, pd.DataFrame]: {요인명: 데이터프레임} 딕셔너리
        """
        factor_data = {}
        
        for factor_name in factor_names:
            try:
                factor_data[factor_name] = self.load_single_factor(factor_name)
            except Exception as e:
                logger.error(f"{factor_name} 로딩 실패: {e}")
                continue
        
        logger.info(f"{len(factor_data)}개 요인 데이터 로딩 완료")
        return factor_data
    
    def load_analyzable_factors(self) -> Dict[str, pd.DataFrame]:
        """
        요인분석에 적합한 모든 요인들을 로딩
        
        Returns:
            Dict[str, pd.DataFrame]: 분석 가능한 요인 데이터들
        """
        # 요인분석에 부적합한 요인들 제외
        excluded_factors = ['dce_variables', 'demographics_1', 'demographics_2']
        all_factors = self.factor_config.get_all_factors()
        analyzable_factors = [f for f in all_factors if f not in excluded_factors]
        
        return self.load_multiple_factors(analyzable_factors)
    
    def _validate_factor_data(self, df: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """
        요인 데이터의 유효성 검증
        
        Args:
            df (pd.DataFrame): 검증할 데이터프레임
            factor_name (str): 요인 이름
            
        Returns:
            pd.DataFrame: 검증된 데이터프레임
        """
        # 기본 검증
        if df.empty:
            raise ValueError(f"{factor_name} 데이터가 비어있습니다")
        
        # 'no' 컬럼 (응답자 ID) 확인
        if 'no' not in df.columns:
            raise ValueError(f"{factor_name} 데이터에 'no' 컬럼이 없습니다")
        
        # 예상 문항들 확인
        expected_questions = self.factor_config.get_factor_questions(factor_name)
        missing_questions = [q for q in expected_questions if q not in df.columns]
        
        if missing_questions:
            logger.warning(f"{factor_name}에서 누락된 문항들: {missing_questions}")
        
        # 결측치 확인
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"{factor_name}에서 {missing_count}개의 결측치 발견")
        
        return df
    
    def merge_factors_for_analysis(self, factor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        여러 요인 데이터를 분석용으로 병합

        Args:
            factor_data (Dict[str, pd.DataFrame]): 요인별 데이터

        Returns:
            pd.DataFrame: 병합된 분석용 데이터
        """
        if not factor_data:
            raise ValueError("병합할 요인 데이터가 없습니다")

        # 각 요인 데이터에 고유 ID 부여 (넘버링 문제 해결)
        processed_factor_data = {}
        for factor_name, df in factor_data.items():
            df_copy = df.copy()

            # 'no' 컬럼을 문자열로 변환 (혼합 타입 지원)
            df_copy['no'] = df_copy['no'].astype(str)

            # 고유 ID 생성 (원본 인덱스 기반)
            df_copy['unique_id'] = df_copy.index

            # 중복된 'no' 값이 있는지 확인
            duplicated_nos = df_copy[df_copy['no'].duplicated(keep=False)]['no'].unique()
            if len(duplicated_nos) > 0:
                logger.info(f"{factor_name}에서 넘버링 문제 발견: no={duplicated_nos} (서로 다른 응답자)")

                # 중복된 'no' 값들에 대해 고유 ID로 구분
                for no_val in duplicated_nos:
                    mask = df_copy['no'] == no_val
                    duplicate_indices = df_copy[mask].index.tolist()

                    # 첫 번째는 원래 no 유지, 나머지는 고유 ID 사용
                    for i, idx in enumerate(duplicate_indices[1:], 1):
                        new_no = f"{no_val}_{i}"  # 예: 273_1, 273_2
                        df_copy.loc[idx, 'no'] = new_no
                        logger.info(f"  {factor_name}: index {idx}의 no를 {no_val} → {new_no}로 변경")

            processed_factor_data[factor_name] = df_copy

        # 첫 번째 요인을 기준으로 시작
        factor_names = list(processed_factor_data.keys())
        merged_df = processed_factor_data[factor_names[0]].copy()

        # 나머지 요인들을 순차적으로 병합
        for factor_name in factor_names[1:]:
            factor_df = processed_factor_data[factor_name]

            # 'no' 컬럼을 기준으로 병합 (이제 모든 no가 고유함)
            merged_df = pd.merge(merged_df, factor_df, on='no', how='inner', suffixes=('', f'_{factor_name}'))

        # unique_id 컬럼들 제거 (분석에 불필요)
        id_cols = [col for col in merged_df.columns if 'unique_id' in col]
        if id_cols:
            merged_df = merged_df.drop(columns=id_cols)

        logger.info(f"병합 완료: {merged_df.shape}, 응답자 수: {len(merged_df)}")
        return merged_df
    
    def get_factor_summary(self) -> pd.DataFrame:
        """
        사용 가능한 요인들의 요약 정보 반환
        
        Returns:
            pd.DataFrame: 요인 요약 정보
        """
        available_files = self.get_available_factor_files()
        summary_data = []
        
        for factor_name, file_path in available_files.items():
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                questions = self.factor_config.get_factor_questions(factor_name)
                # FactorConfig에서 설명 가져오기
                factor_info = self.factor_config.FACTOR_DEFINITIONS.get(factor_name, {})
                description = factor_info.get('description', f'Factor: {factor_name}')
                
                summary_data.append({
                    'factor_name': factor_name,
                    'description': description,
                    'file_path': str(file_path),
                    'n_respondents': len(df),
                    'n_questions': len([q for q in questions if q in df.columns]),
                    'missing_questions': len([q for q in questions if q not in df.columns]),
                    'analyzable': factor_name not in ['dce_variables', 'demographics_1', 'demographics_2']
                })
            except Exception as e:
                logger.error(f"{factor_name} 요약 정보 생성 실패: {e}")
                continue
        
        return pd.DataFrame(summary_data)


def load_factor_data(factor_names: Union[str, List[str]], 
                    data_dir: Optional[Union[str, Path]] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    요인 데이터를 로딩하는 편의 함수
    
    Args:
        factor_names (Union[str, List[str]]): 로딩할 요인 이름(들)
        data_dir (Optional[Union[str, Path]]): 데이터 디렉토리 경로
        
    Returns:
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]: 로딩된 데이터
    """
    loader = FactorDataLoader(data_dir)
    
    if isinstance(factor_names, str):
        return loader.load_single_factor(factor_names)
    else:
        return loader.load_multiple_factors(factor_names)


def get_available_factors(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    사용 가능한 요인들의 리스트를 반환하는 편의 함수
    
    Args:
        data_dir (Optional[Union[str, Path]]): 데이터 디렉토리 경로
        
    Returns:
        List[str]: 사용 가능한 요인 이름 리스트
    """
    loader = FactorDataLoader(data_dir)
    return list(loader.get_available_factor_files().keys())
