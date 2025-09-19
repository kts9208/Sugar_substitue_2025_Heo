"""
Hybrid Data Integrator

DCE와 SEM 데이터를 통합하는 메인 클래스입니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# 기존 모듈 임포트
try:
    from ...multinomial_logit.data_loader import DCEDataLoader
    from ...factor_analysis.data_loader import FactorDataLoader
    EXISTING_LOADERS_AVAILABLE = True
except ImportError:
    EXISTING_LOADERS_AVAILABLE = False
    logging.warning("기존 데이터 로더를 찾을 수 없습니다.")

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """데이터 통합 결과"""
    integrated_data: pd.DataFrame
    dce_data: pd.DataFrame
    sem_data: pd.DataFrame
    integration_summary: Dict[str, Any]
    validation_results: Dict[str, Any]
    
    def get_summary(self) -> Dict[str, Any]:
        """통합 요약 정보"""
        return {
            "total_observations": len(self.integrated_data),
            "dce_observations": len(self.dce_data),
            "sem_observations": len(self.sem_data),
            "common_individuals": self.integration_summary.get("common_individuals", 0),
            "integration_method": self.integration_summary.get("method", "unknown"),
            "validation_passed": self.validation_results.get("overall_valid", False)
        }


class HybridDataIntegrator:
    """하이브리드 데이터 통합기"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 통합 설정
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 설정 추출
        self.individual_id_column = config.get('individual_id_column', 'individual_id')
        self.merge_method = config.get('merge_method', 'inner')  # 'inner', 'outer', 'left', 'right'
        self.handle_missing = config.get('handle_missing', 'drop')  # 'drop', 'impute', 'keep'
        
        # 기존 로더 설정
        self.use_existing_loaders = EXISTING_LOADERS_AVAILABLE and config.get('use_existing_loaders', True)
        
        if self.use_existing_loaders:
            self._setup_existing_loaders()
    
    def _setup_existing_loaders(self):
        """기존 데이터 로더 설정"""
        try:
            self.dce_loader = DCEDataLoader()
            self.factor_loader = FactorDataLoader()
            self.logger.info("기존 데이터 로더를 사용합니다.")
        except Exception as e:
            self.logger.warning(f"기존 로더 설정 실패: {e}")
            self.use_existing_loaders = False
    
    def integrate_data(self, dce_data: pd.DataFrame, sem_data: pd.DataFrame, 
                      latent_variables: Optional[List[str]] = None) -> IntegrationResult:
        """
        DCE와 SEM 데이터 통합
        
        Args:
            dce_data: DCE 데이터
            sem_data: SEM 데이터 (또는 설문 데이터)
            latent_variables: 잠재변수 목록
            
        Returns:
            통합 결과
        """
        self.logger.info("데이터 통합을 시작합니다...")
        
        # 1. 데이터 전처리
        processed_dce = self._process_dce_data(dce_data)
        processed_sem = self._process_sem_data(sem_data, latent_variables)
        
        # 2. 데이터 검증
        validation_results = self._validate_data_compatibility(processed_dce, processed_sem)
        
        # 3. 데이터 병합
        integrated_data, integration_summary = self._merge_data(processed_dce, processed_sem)
        
        # 4. 후처리
        final_data = self._post_process_integrated_data(integrated_data)
        
        result = IntegrationResult(
            integrated_data=final_data,
            dce_data=processed_dce,
            sem_data=processed_sem,
            integration_summary=integration_summary,
            validation_results=validation_results
        )
        
        self.logger.info(f"데이터 통합 완료: {result.get_summary()}")
        return result
    
    def _process_dce_data(self, dce_data: pd.DataFrame) -> pd.DataFrame:
        """DCE 데이터 전처리"""
        self.logger.info("DCE 데이터 전처리 중...")
        
        processed_data = dce_data.copy()
        
        # 필수 컬럼 확인
        required_columns = ['choice', 'alternative', self.individual_id_column]
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        
        if missing_columns:
            raise ValueError(f"DCE 데이터에 필수 컬럼이 없습니다: {missing_columns}")
        
        # 데이터 타입 확인 및 변환
        if processed_data['choice'].dtype not in ['int64', 'bool']:
            processed_data['choice'] = processed_data['choice'].astype(int)
        
        # 개체 ID 정규화
        processed_data[self.individual_id_column] = processed_data[self.individual_id_column].astype(str)
        
        # 결측값 처리
        if processed_data.isnull().any().any():
            if self.handle_missing == 'drop':
                processed_data = processed_data.dropna()
                self.logger.info("DCE 데이터에서 결측값이 있는 행을 제거했습니다.")
            elif self.handle_missing == 'impute':
                # 간단한 대체 (실제로는 더 정교한 방법 필요)
                processed_data = processed_data.fillna(processed_data.mean(numeric_only=True))
                self.logger.info("DCE 데이터의 결측값을 평균으로 대체했습니다.")
        
        self.logger.info(f"DCE 데이터 전처리 완료: {len(processed_data)}개 관측치")
        return processed_data
    
    def _process_sem_data(self, sem_data: pd.DataFrame, latent_variables: Optional[List[str]] = None) -> pd.DataFrame:
        """SEM 데이터 전처리"""
        self.logger.info("SEM 데이터 전처리 중...")
        
        processed_data = sem_data.copy()
        
        # 개체 ID 컬럼 확인
        if self.individual_id_column not in processed_data.columns:
            # 인덱스를 개체 ID로 사용
            processed_data[self.individual_id_column] = processed_data.index.astype(str)
            self.logger.info("SEM 데이터에 개체 ID 컬럼을 생성했습니다.")
        else:
            processed_data[self.individual_id_column] = processed_data[self.individual_id_column].astype(str)
        
        # 잠재변수 관련 처리
        if latent_variables:
            self.logger.info(f"잠재변수 처리: {latent_variables}")
            # 잠재변수별 관측변수 확인
            for lv in latent_variables:
                lv_columns = [col for col in processed_data.columns if col.startswith(lv)]
                if not lv_columns:
                    self.logger.warning(f"잠재변수 '{lv}'에 해당하는 관측변수를 찾을 수 없습니다.")
        
        # 결측값 처리
        if processed_data.isnull().any().any():
            if self.handle_missing == 'drop':
                processed_data = processed_data.dropna()
                self.logger.info("SEM 데이터에서 결측값이 있는 행을 제거했습니다.")
            elif self.handle_missing == 'impute':
                # 수치형 변수만 평균으로 대체
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                processed_data[numeric_columns] = processed_data[numeric_columns].fillna(
                    processed_data[numeric_columns].mean()
                )
                self.logger.info("SEM 데이터의 결측값을 평균으로 대체했습니다.")
        
        self.logger.info(f"SEM 데이터 전처리 완료: {len(processed_data)}개 관측치")
        return processed_data
    
    def _validate_data_compatibility(self, dce_data: pd.DataFrame, sem_data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 호환성 검증"""
        self.logger.info("데이터 호환성 검증 중...")
        
        validation_results = {
            "overall_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # 개체 ID 일치성 확인
        dce_individuals = set(dce_data[self.individual_id_column].unique())
        sem_individuals = set(sem_data[self.individual_id_column].unique())
        
        common_individuals = dce_individuals.intersection(sem_individuals)
        dce_only = dce_individuals - sem_individuals
        sem_only = sem_individuals - dce_individuals
        
        if not common_individuals:
            validation_results["overall_valid"] = False
            validation_results["issues"].append("공통 개체가 없습니다.")
        
        if dce_only:
            validation_results["warnings"].append(f"DCE에만 있는 개체: {len(dce_only)}개")
        
        if sem_only:
            validation_results["warnings"].append(f"SEM에만 있는 개체: {len(sem_only)}개")
        
        # 데이터 크기 확인
        if len(dce_data) == 0:
            validation_results["overall_valid"] = False
            validation_results["issues"].append("DCE 데이터가 비어있습니다.")
        
        if len(sem_data) == 0:
            validation_results["overall_valid"] = False
            validation_results["issues"].append("SEM 데이터가 비어있습니다.")
        
        validation_results["common_individuals"] = len(common_individuals)
        validation_results["dce_individuals"] = len(dce_individuals)
        validation_results["sem_individuals"] = len(sem_individuals)
        
        self.logger.info(f"데이터 호환성 검증 완료: {validation_results}")
        return validation_results
    
    def _merge_data(self, dce_data: pd.DataFrame, sem_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """데이터 병합"""
        self.logger.info(f"데이터 병합 중 (방법: {self.merge_method})...")
        
        # 병합 전 크기 기록
        dce_size_before = len(dce_data)
        sem_size_before = len(sem_data)
        
        # 병합 수행
        merged_data = pd.merge(
            dce_data,
            sem_data,
            on=self.individual_id_column,
            how=self.merge_method,
            suffixes=('_dce', '_sem')
        )
        
        # 병합 후 크기 기록
        merged_size = len(merged_data)
        
        integration_summary = {
            "method": self.merge_method,
            "dce_size_before": dce_size_before,
            "sem_size_before": sem_size_before,
            "merged_size": merged_size,
            "common_individuals": len(merged_data[self.individual_id_column].unique()),
            "merge_ratio": merged_size / max(dce_size_before, sem_size_before) if max(dce_size_before, sem_size_before) > 0 else 0
        }
        
        self.logger.info(f"데이터 병합 완료: {integration_summary}")
        return merged_data, integration_summary
    
    def _post_process_integrated_data(self, integrated_data: pd.DataFrame) -> pd.DataFrame:
        """통합 데이터 후처리"""
        self.logger.info("통합 데이터 후처리 중...")
        
        processed_data = integrated_data.copy()
        
        # 중복 컬럼 처리
        duplicate_columns = []
        for col in processed_data.columns:
            if col.endswith('_dce') and col.replace('_dce', '_sem') in processed_data.columns:
                base_col = col.replace('_dce', '')
                if base_col not in [self.individual_id_column]:  # ID 컬럼은 제외
                    duplicate_columns.append(base_col)
        
        for base_col in duplicate_columns:
            dce_col = f"{base_col}_dce"
            sem_col = f"{base_col}_sem"
            
            # DCE 값 우선 사용 (결측값이면 SEM 값 사용)
            processed_data[base_col] = processed_data[dce_col].fillna(processed_data[sem_col])
            
            # 원본 컬럼 제거
            processed_data = processed_data.drop(columns=[dce_col, sem_col])
        
        # 데이터 정렬
        processed_data = processed_data.sort_values([self.individual_id_column])
        processed_data = processed_data.reset_index(drop=True)
        
        self.logger.info(f"통합 데이터 후처리 완료: {len(processed_data)}개 관측치, {len(processed_data.columns)}개 변수")
        return processed_data


# 편의 함수들
def integrate_dce_sem_data(dce_data: pd.DataFrame, sem_data: pd.DataFrame, 
                          config: Optional[Dict[str, Any]] = None, 
                          latent_variables: Optional[List[str]] = None) -> IntegrationResult:
    """
    DCE와 SEM 데이터 통합 편의 함수
    
    Args:
        dce_data: DCE 데이터
        sem_data: SEM 데이터
        config: 통합 설정
        latent_variables: 잠재변수 목록
        
    Returns:
        통합 결과
    """
    if config is None:
        config = {}
    
    integrator = HybridDataIntegrator(config)
    return integrator.integrate_data(dce_data, sem_data, latent_variables)


def validate_hybrid_data(dce_data: pd.DataFrame, sem_data: pd.DataFrame, 
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    하이브리드 데이터 검증 편의 함수
    
    Args:
        dce_data: DCE 데이터
        sem_data: SEM 데이터
        config: 검증 설정
        
    Returns:
        검증 결과
    """
    if config is None:
        config = {}
    
    integrator = HybridDataIntegrator(config)
    
    # 간단한 전처리 후 검증
    processed_dce = integrator._process_dce_data(dce_data)
    processed_sem = integrator._process_sem_data(sem_data)
    
    return integrator._validate_data_compatibility(processed_dce, processed_sem)
