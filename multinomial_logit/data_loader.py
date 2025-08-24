"""
DCE 데이터 로딩 모듈

이 모듈은 DCE(Discrete Choice Experiment) 데이터를 로딩하고 검증하는 
재사용 가능한 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DCEDataLoader:
    """DCE 데이터 로딩 및 검증을 담당하는 클래스"""
    
    def __init__(self, data_dir: str):
        """
        DCE 데이터 로더 초기화
        
        Args:
            data_dir (str): DCE 데이터 파일들이 위치한 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.required_files = {
            'choice_matrix': 'dce_choice_matrix.csv',
            'attribute_data': 'dce_attribute_data.csv',
            'choice_summary': 'dce_choice_summary.csv',
            'choice_sets_config': 'dce_choice_sets_config.csv',
            'raw_data': 'dce_raw_data.csv',
            'validation_report': 'dce_validation_report.csv'
        }
        self._validate_data_directory()
    
    def _validate_data_directory(self) -> None:
        """데이터 디렉토리와 필수 파일들의 존재를 검증"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {self.data_dir}")
        
        missing_files = []
        for file_key, filename in self.required_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"필수 파일들을 찾을 수 없습니다: {missing_files}")
        
        logger.info(f"데이터 디렉토리 검증 완료: {self.data_dir}")
    
    def load_choice_matrix(self) -> pd.DataFrame:
        """
        선택 매트릭스 데이터를 로딩
        
        Returns:
            pd.DataFrame: 선택 매트릭스 데이터
        """
        file_path = self.data_dir / self.required_files['choice_matrix']
        try:
            df = pd.read_csv(file_path)
            logger.info(f"선택 매트릭스 데이터 로딩 완료: {len(df)} 행")
            return self._validate_choice_matrix(df)
        except Exception as e:
            logger.error(f"선택 매트릭스 데이터 로딩 실패: {e}")
            raise
    
    def load_attribute_data(self) -> pd.DataFrame:
        """
        속성 데이터를 로딩
        
        Returns:
            pd.DataFrame: 속성 데이터
        """
        file_path = self.data_dir / self.required_files['attribute_data']
        try:
            df = pd.read_csv(file_path)
            logger.info(f"속성 데이터 로딩 완료: {len(df)} 행")
            return self._validate_attribute_data(df)
        except Exception as e:
            logger.error(f"속성 데이터 로딩 실패: {e}")
            raise
    
    def load_choice_summary(self) -> pd.DataFrame:
        """
        선택 요약 데이터를 로딩
        
        Returns:
            pd.DataFrame: 선택 요약 데이터
        """
        file_path = self.data_dir / self.required_files['choice_summary']
        try:
            df = pd.read_csv(file_path)
            logger.info(f"선택 요약 데이터 로딩 완료: {len(df)} 행")
            return df
        except Exception as e:
            logger.error(f"선택 요약 데이터 로딩 실패: {e}")
            raise
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        모든 DCE 데이터를 로딩
        
        Returns:
            Dict[str, pd.DataFrame]: 모든 데이터셋을 포함하는 딕셔너리
        """
        data = {}
        try:
            data['choice_matrix'] = self.load_choice_matrix()
            data['attribute_data'] = self.load_attribute_data()
            data['choice_summary'] = self.load_choice_summary()
            logger.info("모든 DCE 데이터 로딩 완료")
            return data
        except Exception as e:
            logger.error(f"데이터 로딩 중 오류 발생: {e}")
            raise
    
    def _validate_choice_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        선택 매트릭스 데이터의 유효성을 검증
        
        Args:
            df (pd.DataFrame): 검증할 데이터프레임
            
        Returns:
            pd.DataFrame: 검증된 데이터프레임
        """
        required_columns = [
            'respondent_id', 'question_id', 'alternative', 'chosen',
            'sugar_type', 'health_label', 'price', 'sugar_free', 'has_health_label'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"선택 매트릭스에 필수 컬럼이 없습니다: {missing_columns}")
        
        # 선택 변수가 0 또는 1인지 확인
        if not df['chosen'].isin([0, 1]).all():
            raise ValueError("'chosen' 컬럼은 0 또는 1의 값만 가져야 합니다")
        
        # 각 질문별로 정확히 하나의 선택이 있는지 확인
        choice_counts = df.groupby(['respondent_id', 'question_id'])['chosen'].sum()
        invalid_choices = choice_counts[choice_counts != 1]
        if len(invalid_choices) > 0:
            logger.warning(f"일부 질문에서 선택이 정확히 하나가 아닙니다: {len(invalid_choices)}개 질문")
        
        return df
    
    def _validate_attribute_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        속성 데이터의 유효성을 검증
        
        Args:
            df (pd.DataFrame): 검증할 데이터프레임
            
        Returns:
            pd.DataFrame: 검증된 데이터프레임
        """
        required_columns = [
            'respondent_id', 'question_id', 'chosen_sugar_type', 
            'chosen_health_label', 'chosen_price', 'chose_sugar_free', 
            'chose_health_label', 'choice_value'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"속성 데이터에 필수 컬럼이 없습니다: {missing_columns}")
        
        # 가격이 양수인지 확인
        if (df['chosen_price'] < 0).any():
            raise ValueError("가격은 음수일 수 없습니다")
        
        return df
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        로딩된 데이터의 요약 정보를 반환
        
        Returns:
            Dict[str, any]: 데이터 요약 정보
        """
        try:
            choice_matrix = self.load_choice_matrix()
            attribute_data = self.load_attribute_data()
            
            summary = {
                'total_respondents': choice_matrix['respondent_id'].nunique(),
                'total_questions': choice_matrix['question_id'].nunique(),
                'total_choices': len(choice_matrix),
                'total_actual_choices': len(attribute_data),
                'alternatives_per_question': choice_matrix.groupby(['respondent_id', 'question_id']).size().iloc[0],
                'sugar_types': choice_matrix['sugar_type'].unique().tolist(),
                'health_labels': choice_matrix['health_label'].unique().tolist(),
                'price_range': {
                    'min': choice_matrix['price'].min(),
                    'max': choice_matrix['price'].max()
                }
            }
            
            return summary
        except Exception as e:
            logger.error(f"데이터 요약 생성 중 오류: {e}")
            raise


def load_dce_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    DCE 데이터를 로딩하는 편의 함수
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        
    Returns:
        Dict[str, pd.DataFrame]: 로딩된 데이터
    """
    loader = DCEDataLoader(data_dir)
    return loader.load_all_data()


def get_dce_summary(data_dir: str) -> Dict[str, any]:
    """
    DCE 데이터 요약을 반환하는 편의 함수
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        
    Returns:
        Dict[str, any]: 데이터 요약 정보
    """
    loader = DCEDataLoader(data_dir)
    return loader.get_data_summary()
