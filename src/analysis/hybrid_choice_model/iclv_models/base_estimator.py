"""
Base Estimator for ICLV Models

ICLV 모델 추정기의 추상 베이스 클래스입니다.
동시추정(Simultaneous)과 순차추정(Sequential) 모두의 공통 기능을 제공합니다.

단일책임 원칙:
- 데이터 검증
- 로깅 설정
- 결과 저장
- 파라미터 이름 관리
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseEstimator(ABC):
    """
    ICLV 모델 추정기 베이스 클래스
    
    공통 기능:
    - 데이터 검증
    - 로깅 설정
    - 결과 저장
    - 파라미터 이름 관리
    
    하위 클래스에서 구현해야 할 메서드:
    - estimate(): 추정 실행
    - _initialize_parameters(): 초기값 설정
    - _compute_log_likelihood(): 우도 계산
    """
    
    def __init__(self, config):
        """
        Args:
            config: ICLVConfig 또는 MultiLatentConfig 객체
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 공통 속성
        self.data = None
        self.results = None
        self.param_names = []
        
        # 로그 파일 핸들러
        self.log_file_handler = None
        self.iteration_logger = None
        self.csv_log_file = None
        self.csv_writer = None
        self.csv_log_path = None
    
    # ========================================================================
    # 추상 메서드 (하위 클래스에서 반드시 구현)
    # ========================================================================
    
    @abstractmethod
    def estimate(self, data: pd.DataFrame, 
                measurement_model,
                structural_model,
                choice_model,
                **kwargs) -> Dict:
        """
        모델 추정 실행
        
        Args:
            data: 통합 데이터
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
            **kwargs: 추가 인자
        
        Returns:
            추정 결과 딕셔너리
        """
        pass
    
    @abstractmethod
    def _initialize_parameters(self, measurement_model, structural_model, 
                              choice_model) -> np.ndarray:
        """
        초기 파라미터 설정
        
        Args:
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_model: 선택모델 객체
        
        Returns:
            초기 파라미터 벡터
        """
        pass
    
    @abstractmethod
    def _compute_log_likelihood(self, params: np.ndarray, *args) -> float:
        """
        로그우도 계산
        
        Args:
            params: 파라미터 벡터
            *args: 추가 인자 (모델 객체 등)
        
        Returns:
            로그우도 값
        """
        pass
    
    # ========================================================================
    # 공통 메서드 (데이터 검증)
    # ========================================================================
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        데이터 검증
        
        Args:
            data: 검증할 데이터프레임
        
        Raises:
            ValueError: 필수 컬럼이 없거나 데이터가 비어있는 경우
        """
        if data is None or len(data) == 0:
            raise ValueError("데이터가 비어있습니다.")
        
        # 개인 ID 컬럼 확인
        if hasattr(self.config, 'individual_id_column'):
            if self.config.individual_id_column not in data.columns:
                raise ValueError(
                    f"개인 ID 컬럼 '{self.config.individual_id_column}'이 데이터에 없습니다."
                )
        
        # 선택 컬럼 확인
        if hasattr(self.config, 'choice_column'):
            if self.config.choice_column not in data.columns:
                raise ValueError(
                    f"선택 컬럼 '{self.config.choice_column}'이 데이터에 없습니다."
                )
        
        self.logger.info(f"데이터 검증 완료: {len(data)} 행")

    # ========================================================================
    # 공통 메서드 (로깅 설정)
    # ========================================================================

    def _setup_iteration_logger(self, log_file_path: str) -> None:
        """
        반복 과정 로깅을 위한 파일 핸들러 설정

        Args:
            log_file_path: 로그 파일 경로
        """
        # 반복 과정 전용 로거 생성
        self.iteration_logger = logging.getLogger('iclv_iteration')
        self.iteration_logger.setLevel(logging.INFO)

        # 기존 핸들러 제거 (중복 방지)
        self.iteration_logger.handlers.clear()

        # 파일 핸들러 추가
        self.log_file_handler = logging.FileHandler(
            log_file_path, mode='w', encoding='utf-8'
        )
        self.log_file_handler.setLevel(logging.INFO)

        # 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.log_file_handler.setFormatter(formatter)
        self.iteration_logger.addHandler(self.log_file_handler)

        # CSV 로그 파일 설정 (파라미터 및 그래디언트 값 저장용)
        csv_log_path = log_file_path.replace('.txt', '_params_grads.csv')
        self.csv_log_file = open(csv_log_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = None  # 첫 번째 기록 시 헤더와 함께 초기화
        self.csv_log_path = csv_log_path

        self.iteration_logger.info("="*70)
        self.iteration_logger.info(f"{self.__class__.__name__} 추정 시작")
        self.iteration_logger.info("="*70)

    def _close_iteration_logger(self) -> None:
        """반복 과정 로거 종료"""
        if self.log_file_handler:
            self.iteration_logger.removeHandler(self.log_file_handler)
            self.log_file_handler.close()
            self.log_file_handler = None

        if self.csv_log_file:
            self.csv_log_file.close()
            self.csv_log_file = None

    def _log_params_grads_to_csv(self, iteration: int,
                                 params: np.ndarray,
                                 grads: np.ndarray) -> None:
        """
        파라미터와 그래디언트 값을 CSV 파일에 기록

        Args:
            iteration: Major iteration 번호
            params: 파라미터 값 배열 (external scale)
            grads: 그래디언트 값 배열
        """
        import csv

        # 첫 번째 기록 시 헤더 작성
        if self.csv_writer is None:
            fieldnames = ['iteration']

            # 파라미터 이름 추가
            for idx in range(len(params)):
                param_name = (
                    self.param_names[idx]
                    if idx < len(self.param_names)
                    else f"param_{idx}"
                )
                fieldnames.append(f'{param_name}_value')
                fieldnames.append(f'{param_name}_grad')

            self.csv_writer = csv.DictWriter(self.csv_log_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()

        # 데이터 행 작성
        row = {'iteration': iteration}
        for idx in range(len(params)):
            param_name = (
                self.param_names[idx]
                if idx < len(self.param_names)
                else f"param_{idx}"
            )
            row[f'{param_name}_value'] = params[idx]
            row[f'{param_name}_grad'] = grads[idx]

        self.csv_writer.writerow(row)
        self.csv_log_file.flush()  # 즉시 디스크에 기록

    # ========================================================================
    # 공통 메서드 (결과 저장)
    # ========================================================================

    def _create_result_dict(self, params: np.ndarray, log_likelihood: float,
                           n_iterations: int, success: bool,
                           **kwargs) -> Dict:
        """
        결과 딕셔너리 생성

        Args:
            params: 최종 파라미터
            log_likelihood: 최종 로그우도
            n_iterations: 반복 횟수
            success: 수렴 여부
            **kwargs: 추가 정보

        Returns:
            결과 딕셔너리
        """
        n_params = len(params)
        n_obs = len(self.data) if self.data is not None else 0

        # AIC, BIC 계산
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_obs) - 2 * log_likelihood

        result = {
            'success': success,
            'log_likelihood': log_likelihood,
            'n_iterations': n_iterations,
            'n_parameters': n_params,
            'n_observations': n_obs,
            'aic': aic,
            'bic': bic,
            'raw_params': params,
            **kwargs
        }

        return result

