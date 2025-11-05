"""
Sociodemographic Data Loader

사회인구학적 변수를 원본 Excel 파일에서 로드하고 전처리하는 모듈입니다.
기존 BaseDataLoader를 상속받아 일관된 인터페이스를 제공합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Union

# 기존 BaseDataLoader 임포트
try:
    from ...utility_function.data_loader.base_loader import BaseDataLoader
    BASE_LOADER_AVAILABLE = True
except ImportError:
    # BaseDataLoader를 찾을 수 없는 경우 간단한 대체 클래스
    BASE_LOADER_AVAILABLE = False
    logging.warning("BaseDataLoader를 찾을 수 없습니다. 기본 구현을 사용합니다.")
    
    class BaseDataLoader:
        """BaseDataLoader 대체 클래스"""
        def __init__(self, data_dir: Optional[Path] = None):
            self.data_dir = data_dir
            self.data_cache = {}
        
        def _validate_file_exists(self, file_path: Path) -> bool:
            return file_path.exists()
        
        def _validate_required_columns(self, df: pd.DataFrame, required_columns: list) -> bool:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                return False
            return True
        
        def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.dropna(how='all')

logger = logging.getLogger(__name__)


class SociodemographicLoader(BaseDataLoader):
    """
    사회인구학적 변수 로더
    
    원본 Excel 파일에서 사회인구학적 변수를 로드하고 전처리합니다.
    BaseDataLoader를 상속받아 기존 데이터 로더와 일관된 인터페이스를 제공합니다.
    
    주요 기능:
    - 원본 Excel 파일 로드 (DATA, LABEL, CODE 시트)
    - 사회인구학적 변수 선택 및 변수명 변경
    - 데이터 전처리 (표준화, 코딩 변환)
    - 결측치 처리
    """
    
    # 사회인구학적 변수 매핑 (원본 변수명 → 표준 변수명)
    SOCIODEM_VARIABLE_MAPPING = {
        'no': 'respondent_id',
        'q1': 'gender',
        'q2_1': 'age',
        'q3': 'age_category',
        'q4': 'age_group',
        'q5': 'region',
        'q51': 'occupation',
        'q51_14': 'occupation_other',
        'q52': 'income',
        'q53': 'education',
        'q54': 'diabetes',
        'q55': 'family_diabetes',
        'q56': 'sugar_substitute_usage'
    }
    
    # 소득 범주 → 연속형 변환 매핑 (단위: 100만원)
    INCOME_MAPPING = {
        1: 1.5,   # 200만원 미만 → 150만원
        2: 2.5,   # 200-300만원 → 250만원
        3: 3.5,   # 300-400만원 → 350만원
        4: 4.5,   # 400-500만원 → 450만원
        5: 6.0    # 600만원 이상 → 600만원
    }
    
    # 교육수준 매핑
    EDUCATION_MAPPING = {
        1: 1,  # 고졸 미만
        2: 2,  # 고졸
        3: 3,  # 대학 재학
        4: 4,  # 대학 졸업
        5: 5,  # 대학원 재학
        6: 6   # 대학원 졸업
    }
    
    def __init__(self, 
                 raw_data_path: Optional[Union[str, Path]] = None,
                 data_dir: Optional[Path] = None):
        """
        사회인구학적 데이터 로더 초기화
        
        Args:
            raw_data_path: 원본 Excel 파일 경로 (기본값: data/raw/Sugar_substitue_Raw data_250730.xlsx)
            data_dir: 데이터 디렉토리 (BaseDataLoader 호환성)
        """
        super().__init__(data_dir)
        
        # 원본 데이터 경로 설정
        if raw_data_path is None:
            # 프로젝트 루트에서 기본 경로 설정
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.raw_data_path = project_root / "data" / "raw" / "Sugar_substitue_Raw data_250730.xlsx"
        else:
            self.raw_data_path = Path(raw_data_path)
        
        # 파일 존재 확인
        if not self._validate_file_exists(self.raw_data_path):
            raise FileNotFoundError(f"원본 데이터 파일을 찾을 수 없습니다: {self.raw_data_path}")
        
        logger.info(f"SociodemographicLoader 초기화 완료: {self.raw_data_path}")
    
    def load_data(self) -> Dict[str, Any]:
        """
        사회인구학적 데이터 로드 (BaseDataLoader 인터페이스 구현)
        
        Returns:
            Dictionary containing:
            - 'raw_data': 원본 데이터
            - 'processed_data': 전처리된 데이터
            - 'metadata': 메타데이터 (LABEL, CODE 시트)
        """
        logger.info("사회인구학적 데이터 로드 시작...")
        
        data = {}
        
        # 1. 원본 데이터 로드
        data['raw_data'] = self._load_raw_data()
        
        # 2. 메타데이터 로드 (LABEL, CODE 시트)
        data['metadata'] = self._load_metadata()
        
        # 3. 사회인구학적 변수 추출
        data['sociodem_raw'] = self._extract_sociodemographic_variables(data['raw_data'])
        
        # 4. 전처리
        data['processed_data'] = self.preprocess_data(data['sociodem_raw'])
        
        logger.info(f"사회인구학적 데이터 로드 완료: {len(data['processed_data'])}개 관측치")
        return data
    
    def _load_raw_data(self) -> pd.DataFrame:
        """
        원본 Excel 파일의 DATA 시트 로드
        
        Returns:
            원본 데이터 DataFrame
        """
        logger.info(f"원본 데이터 로드 중: {self.raw_data_path}")
        
        try:
            df = pd.read_excel(self.raw_data_path, sheet_name='DATA')
            logger.info(f"원본 데이터 로드 완료: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"원본 데이터 로드 실패: {e}")
            raise
    
    def _load_metadata(self) -> Dict[str, pd.DataFrame]:
        """
        메타데이터 시트 로드 (LABEL, CODE)
        
        Returns:
            메타데이터 딕셔너리
        """
        logger.info("메타데이터 로드 중...")
        
        metadata = {}
        
        try:
            # LABEL 시트 로드 (선택적)
            try:
                metadata['labels'] = pd.read_excel(self.raw_data_path, sheet_name='LABEL')
                logger.info("LABEL 시트 로드 완료")
            except Exception as e:
                logger.warning(f"LABEL 시트 로드 실패: {e}")
                metadata['labels'] = None
            
            # CODE 시트 로드 (선택적)
            try:
                metadata['codes'] = pd.read_excel(self.raw_data_path, sheet_name='CODE')
                logger.info("CODE 시트 로드 완료")
            except Exception as e:
                logger.warning(f"CODE 시트 로드 실패: {e}")
                metadata['codes'] = None
            
            return metadata
        except Exception as e:
            logger.warning(f"메타데이터 로드 중 오류: {e}")
            return {'labels': None, 'codes': None}
    
    def _extract_sociodemographic_variables(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        원본 데이터에서 사회인구학적 변수만 추출
        
        Args:
            raw_data: 원본 데이터
            
        Returns:
            사회인구학적 변수만 포함된 DataFrame
        """
        logger.info("사회인구학적 변수 추출 중...")
        
        # 추출할 변수 목록
        sociodem_vars = list(self.SOCIODEM_VARIABLE_MAPPING.keys())
        
        # 실제 존재하는 변수만 선택
        available_vars = [var for var in sociodem_vars if var in raw_data.columns]
        missing_vars = set(sociodem_vars) - set(available_vars)
        
        if missing_vars:
            logger.warning(f"일부 사회인구학적 변수가 없습니다: {missing_vars}")
        
        # 변수 추출
        sociodem_data = raw_data[available_vars].copy()
        
        # 변수명 변경
        rename_mapping = {k: v for k, v in self.SOCIODEM_VARIABLE_MAPPING.items() 
                         if k in available_vars}
        sociodem_data = sociodem_data.rename(columns=rename_mapping)
        
        logger.info(f"사회인구학적 변수 추출 완료: {len(available_vars)}개 변수")
        return sociodem_data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        사회인구학적 변수 전처리

        Args:
            data: 원본 사회인구학적 데이터

        Returns:
            전처리된 데이터
        """
        logger.info("사회인구학적 변수 전처리 시작...")

        processed = data.copy()

        # 1. 소득 변환 (범주형 → 연속형) - 결측치 처리 전에 수행
        if 'income' in processed.columns:
            processed['income_continuous'] = processed['income'].map(self.INCOME_MAPPING)
            logger.info("소득 변환 완료")

        # 2. 교육수준 매핑 - 결측치 처리 전에 수행
        if 'education' in processed.columns:
            processed['education_level'] = processed['education'].map(self.EDUCATION_MAPPING)
            logger.info("교육수준 매핑 완료")

        # 3. 핵심 변수만 결측치 처리 (occupation_other 등은 제외)
        core_vars = ['respondent_id', 'gender', 'age', 'income', 'education']
        available_core_vars = [var for var in core_vars if var in processed.columns]

        # 핵심 변수에 결측치가 있는 행만 제거
        if available_core_vars:
            before_len = len(processed)
            processed = processed.dropna(subset=available_core_vars)
            after_len = len(processed)
            if before_len != after_len:
                logger.info(f"핵심 변수 결측치 제거: {before_len} → {after_len} 관측치")

        # 4. 나이 표준화 (결측치 제거 후)
        if 'age' in processed.columns:
            processed['age_std'] = self._standardize_variable(processed['age'])
            logger.info("나이 표준화 완료")

        # 5. 소득 표준화 (결측치 제거 후)
        if 'income_continuous' in processed.columns:
            processed['income_std'] = self._standardize_variable(processed['income_continuous'])
            logger.info("소득 표준화 완료")

        # 6. 성별 (0: 남성, 1: 여성) - 그대로 사용
        if 'gender' in processed.columns:
            logger.info("성별 변수 확인 완료")

        logger.info(f"사회인구학적 변수 전처리 완료: {processed.shape}")
        return processed
    
    def _standardize_variable(self, series: pd.Series) -> pd.Series:
        """
        변수 표준화 (평균 0, 표준편차 1)
        
        Args:
            series: 표준화할 시리즈
            
        Returns:
            표준화된 시리즈
        """
        return (series - series.mean()) / series.std()
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        결측치 처리
        
        Args:
            data: 데이터
            
        Returns:
            결측치 처리된 데이터
        """
        if data.isnull().any().any():
            null_counts = data.isnull().sum()
            logger.warning(f"결측치 발견: {null_counts[null_counts > 0].to_dict()}")
            
            # 기본적으로 결측치가 있는 행 제거
            # (향후 더 정교한 대체 방법 구현 가능)
            data_clean = data.dropna()
            logger.info(f"결측치 제거: {len(data)} → {len(data_clean)} 관측치")
            return data_clean
        
        return data
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        데이터 유효성 검증 (BaseDataLoader 인터페이스 구현)
        
        Args:
            data: 검증할 데이터
            
        Returns:
            유효성 여부
        """
        logger.info("데이터 유효성 검증 시작...")
        
        # 필수 키 확인
        required_keys = ['raw_data', 'processed_data']
        for key in required_keys:
            if key not in data:
                logger.error(f"필수 데이터 키가 없습니다: {key}")
                return False
        
        # 처리된 데이터 확인
        processed_data = data['processed_data']
        
        # 최소 관측치 수 확인
        if len(processed_data) < 10:
            logger.error(f"관측치 수가 너무 적습니다: {len(processed_data)}")
            return False
        
        # 필수 변수 확인
        required_vars = ['respondent_id']
        if not self._validate_required_columns(processed_data, required_vars):
            return False
        
        logger.info("데이터 유효성 검증 통과")
        return True
    
    def get_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터 요약 정보 생성
        
        Args:
            data: 요약할 데이터
            
        Returns:
            요약 정보 딕셔너리
        """
        summary = {
            'n_observations': len(data),
            'n_variables': len(data.columns),
            'variables': list(data.columns),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        # 기술통계
        if 'age' in data.columns:
            summary['age_mean'] = data['age'].mean()
            summary['age_std'] = data['age'].std()
        
        if 'gender' in data.columns:
            summary['gender_distribution'] = data['gender'].value_counts().to_dict()
        
        if 'income' in data.columns:
            summary['income_distribution'] = data['income'].value_counts().to_dict()
        
        return summary


# 편의 함수
def load_sociodemographic_data(raw_data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    사회인구학적 데이터 로드 편의 함수
    
    Args:
        raw_data_path: 원본 Excel 파일 경로
        
    Returns:
        전처리된 사회인구학적 데이터
    """
    loader = SociodemographicLoader(raw_data_path)
    data = loader.load_data()
    return data['processed_data']

