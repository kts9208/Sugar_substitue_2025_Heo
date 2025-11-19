"""
Common Utilities for ICLV Estimators

동시추정과 순차추정에서 공통으로 사용되는 유틸리티 함수들입니다.

단일책임 원칙:
- 로깅 설정
- 데이터 검증
- 결과 저장
- 파라미터 변환

Author: Taeseok Kim
Date: 2025-01-19
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging
from pathlib import Path


def validate_data(data: pd.DataFrame, config, logger: logging.Logger) -> None:
    """
    데이터 검증
    
    Args:
        data: 검증할 데이터프레임
        config: ICLVConfig 또는 MultiLatentConfig
        logger: 로거
    
    Raises:
        ValueError: 필수 컬럼이 없거나 데이터가 비어있는 경우
    """
    if data is None or len(data) == 0:
        raise ValueError("데이터가 비어있습니다.")
    
    # 개인 ID 컬럼 확인
    if hasattr(config, 'individual_id_column'):
        if config.individual_id_column not in data.columns:
            raise ValueError(
                f"개인 ID 컬럼 '{config.individual_id_column}'이 데이터에 없습니다."
            )
    
    # 선택 컬럼 확인
    if hasattr(config, 'choice_column'):
        if config.choice_column not in data.columns:
            raise ValueError(
                f"선택 컬럼 '{config.choice_column}'이 데이터에 없습니다."
            )
    
    logger.info(f"데이터 검증 완료: {len(data)} 행")


def create_result_dict(params: np.ndarray, log_likelihood: float,
                      n_iterations: int, success: bool,
                      n_obs: int, **kwargs) -> Dict:
    """
    결과 딕셔너리 생성
    
    Args:
        params: 최종 파라미터
        log_likelihood: 최종 로그우도
        n_iterations: 반복 횟수
        success: 수렴 여부
        n_obs: 관측치 수
        **kwargs: 추가 정보
    
    Returns:
        결과 딕셔너리
    """
    n_params = len(params)
    
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


def setup_iteration_logger(log_file: str, logger_name: str = 'iteration') -> logging.Logger:
    """
    Iteration 로거 설정
    
    Args:
        log_file: 로그 파일 경로
        logger_name: 로거 이름
    
    Returns:
        설정된 로거
    """
    iteration_logger = logging.getLogger(logger_name)
    iteration_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    iteration_logger.handlers.clear()
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    iteration_logger.addHandler(file_handler)
    iteration_logger.propagate = False
    
    return iteration_logger


def close_iteration_logger(iteration_logger: logging.Logger) -> None:
    """
    Iteration 로거 종료
    
    Args:
        iteration_logger: 종료할 로거
    """
    if iteration_logger:
        for handler in iteration_logger.handlers[:]:
            handler.close()
            iteration_logger.removeHandler(handler)


def save_results(results: Dict, save_path: str, logger: logging.Logger) -> None:
    """
    결과 저장
    
    Args:
        results: 저장할 결과 딕셔너리
        save_path: 저장 경로
        logger: 로거
    """
    import pickle
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"결과 저장 완료: {save_path}")

