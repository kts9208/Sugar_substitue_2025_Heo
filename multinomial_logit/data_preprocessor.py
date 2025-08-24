"""
DCE 데이터 전처리 모듈

이 모듈은 Multinomial Logit Model에 적합한 형태로 DCE 데이터를 변환하는
재사용 가능한 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DCEDataPreprocessor:
    """DCE 데이터 전처리를 담당하는 클래스"""
    
    def __init__(self):
        """데이터 전처리기 초기화"""
        self.categorical_mappings = {}
        self.fitted = False
    
    def prepare_choice_data(self, choice_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        선택 매트릭스를 Multinomial Logit Model에 적합한 형태로 변환
        
        Args:
            choice_matrix (pd.DataFrame): 원본 선택 매트릭스
            
        Returns:
            pd.DataFrame: 전처리된 선택 데이터
        """
        logger.info("선택 데이터 전처리 시작")
        
        # 복사본 생성
        df = choice_matrix.copy()
        
        # Neither 선택지 제거 (실제 제품 선택만 분석)
        # 먼저 Neither가 선택된 경우를 확인하고 해당 선택 세트 전체를 제거
        neither_chosen = df[(df['alternative'] == 'Neither') & (df['chosen'] == 1)]
        neither_choice_sets = set(zip(neither_chosen['respondent_id'], neither_chosen['question_id']))

        # Neither가 선택된 선택 세트 제거
        if neither_choice_sets:
            logger.info(f"Neither가 선택된 {len(neither_choice_sets)}개 선택 세트를 제거합니다")
            for resp_id, quest_id in neither_choice_sets:
                df = df[~((df['respondent_id'] == resp_id) & (df['question_id'] == quest_id))]

        # 남은 데이터에서 Neither 대안 제거
        df = df[df['alternative'] != 'Neither'].copy()
        
        # 가격을 천원 단위로 변환 (스케일링)
        df['price_scaled'] = df['price'] / 1000
        
        # 범주형 변수를 더미 변수로 변환
        df = self._encode_categorical_variables(df)
        
        # 선택 세트별로 정렬
        df = df.sort_values(['respondent_id', 'question_id', 'alternative'])
        
        logger.info(f"선택 데이터 전처리 완료: {len(df)} 행")
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        범주형 변수를 더미 변수로 인코딩
        
        Args:
            df (pd.DataFrame): 인코딩할 데이터프레임
            
        Returns:
            pd.DataFrame: 인코딩된 데이터프레임
        """
        df_encoded = df.copy()
        
        # 설탕 유형 인코딩 (일반당=0, 무설탕=1로 이미 sugar_free 컬럼에 있음)
        # 추가적인 더미 변수는 필요하지 않음
        
        # 건강라벨 인코딩 (이미 has_health_label 컬럼에 0/1로 있음)
        # 추가적인 더미 변수는 필요하지 않음
        
        # 대안 인코딩 (A=0, B=1)
        df_encoded['alternative_B'] = (df_encoded['alternative'] == 'B').astype(int)
        
        return df_encoded
    
    def create_choice_sets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multinomial Logit Model을 위한 선택 세트 생성
        
        Args:
            df (pd.DataFrame): 전처리된 선택 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (X, y, choice_sets)
                - X: 설명변수 행렬
                - y: 선택 결과 벡터
                - choice_sets: 각 선택 세트의 시작 인덱스
        """
        logger.info("선택 세트 생성 시작")
        
        # 설명변수 선택
        feature_columns = ['sugar_free', 'has_health_label', 'price_scaled', 'alternative_B']
        X = df[feature_columns].values
        
        # 종속변수
        y = df['chosen'].values
        
        # 선택 세트 인덱스 생성
        choice_sets = []
        current_idx = 0
        
        for (respondent_id, question_id), group in df.groupby(['respondent_id', 'question_id']):
            choice_sets.append(current_idx)
            current_idx += len(group)
        
        choice_sets = np.array(choice_sets)
        
        logger.info(f"선택 세트 생성 완료: {len(choice_sets)}개 선택 세트, {len(X)}개 대안")
        
        return X, y, choice_sets
    
    def get_feature_names(self) -> List[str]:
        """
        모델에 사용되는 특성 이름들을 반환
        
        Returns:
            List[str]: 특성 이름 리스트
        """
        return ['sugar_free', 'has_health_label', 'price_scaled', 'alternative_B']
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        각 특성의 설명을 반환
        
        Returns:
            Dict[str, str]: 특성 이름과 설명의 딕셔너리
        """
        return {
            'sugar_free': '무설탕 여부 (1: 무설탕, 0: 일반당)',
            'has_health_label': '건강라벨 유무 (1: 있음, 0: 없음)',
            'price_scaled': '가격 (천원 단위)',
            'alternative_B': '대안 B 여부 (1: 대안 B, 0: 대안 A)'
        }
    
    def validate_choice_sets(self, X: np.ndarray, y: np.ndarray, choice_sets: np.ndarray) -> bool:
        """
        생성된 선택 세트의 유효성을 검증
        
        Args:
            X (np.ndarray): 설명변수 행렬
            y (np.ndarray): 선택 결과 벡터
            choice_sets (np.ndarray): 선택 세트 인덱스
            
        Returns:
            bool: 유효성 검증 결과
        """
        try:
            # 기본 차원 검증
            if len(X) != len(y):
                logger.error("X와 y의 길이가 일치하지 않습니다")
                return False
            
            # 각 선택 세트에서 정확히 하나의 선택이 있는지 확인
            for i, start_idx in enumerate(choice_sets):
                if i < len(choice_sets) - 1:
                    end_idx = choice_sets[i + 1]
                else:
                    end_idx = len(y)
                
                choice_sum = y[start_idx:end_idx].sum()
                if choice_sum != 1:
                    logger.warning(f"선택 세트 {i}에서 선택 개수가 1이 아닙니다: {choice_sum}")
            
            logger.info("선택 세트 유효성 검증 완료")
            return True
            
        except Exception as e:
            logger.error(f"선택 세트 유효성 검증 중 오류: {e}")
            return False
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        전처리된 데이터의 요약 통계를 반환
        
        Args:
            df (pd.DataFrame): 전처리된 데이터
            
        Returns:
            Dict[str, any]: 요약 통계
        """
        feature_columns = ['sugar_free', 'has_health_label', 'price_scaled']
        
        summary = {
            'total_observations': len(df),
            'total_choice_sets': len(df) // 2,  # 각 선택 세트당 2개 대안
            'feature_statistics': {}
        }
        
        for col in feature_columns:
            if col in df.columns:
                summary['feature_statistics'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        # 선택 비율
        summary['choice_rates'] = {
            'sugar_free_chosen': df[df['chosen'] == 1]['sugar_free'].mean(),
            'health_label_chosen': df[df['chosen'] == 1]['has_health_label'].mean(),
            'alternative_B_chosen': df[df['chosen'] == 1]['alternative_B'].mean()
        }
        
        return summary


def preprocess_dce_data(choice_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    DCE 데이터를 전처리하는 편의 함수
    
    Args:
        choice_matrix (pd.DataFrame): 원본 선택 매트릭스
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]: 
            (X, y, choice_sets, feature_names)
    """
    preprocessor = DCEDataPreprocessor()
    
    # 데이터 전처리
    processed_data = preprocessor.prepare_choice_data(choice_matrix)
    
    # 선택 세트 생성
    X, y, choice_sets = preprocessor.create_choice_sets(processed_data)
    
    # 유효성 검증
    if not preprocessor.validate_choice_sets(X, y, choice_sets):
        raise ValueError("생성된 선택 세트가 유효하지 않습니다")
    
    # 특성 이름
    feature_names = preprocessor.get_feature_names()
    
    return X, y, choice_sets, feature_names


def get_preprocessing_summary(choice_matrix: pd.DataFrame) -> Dict[str, any]:
    """
    전처리 요약 정보를 반환하는 편의 함수
    
    Args:
        choice_matrix (pd.DataFrame): 원본 선택 매트릭스
        
    Returns:
        Dict[str, any]: 전처리 요약 정보
    """
    preprocessor = DCEDataPreprocessor()
    processed_data = preprocessor.prepare_choice_data(choice_matrix)
    return preprocessor.get_summary_statistics(processed_data)
