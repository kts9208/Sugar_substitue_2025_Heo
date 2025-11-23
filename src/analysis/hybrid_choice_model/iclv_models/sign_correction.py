"""
Sign Correction and Alignment for Bootstrap SEM

이 모듈은 부트스트랩 SEM에서 발생하는 잠재변수 부호 불확정성(sign indeterminacy) 문제를 해결합니다.

주요 기능:
1. Factor Loading Sign Alignment: 요인적재량 부호 정렬
2. Factor Score Sign Alignment: 요인점수 부호 정렬
3. Procrustes Rotation: 다중 잠재변수 모델을 위한 회전 정렬

Author: Augment Agent
Date: 2025-11-23
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def align_factor_loadings_by_dot_product(
    original_loadings: np.ndarray,
    bootstrap_loadings: np.ndarray,
    threshold: float = 0.0
) -> Tuple[np.ndarray, bool]:
    """
    내적(dot product)을 사용한 요인적재량 부호 정렬
    
    원리:
    - 원본과 부트스트랩 요인적재량의 내적이 양수면 같은 방향
    - 내적이 음수면 반대 방향 → 부호 반전 필요
    
    Args:
        original_loadings: 원본 요인적재량 (n_indicators,)
        bootstrap_loadings: 부트스트랩 요인적재량 (n_indicators,)
        threshold: 부호 반전 임계값 (기본값: 0.0)
    
    Returns:
        (정렬된 부트스트랩 요인적재량, 반전 여부)
    
    Example:
        >>> orig = np.array([0.8, 0.6, 0.4])
        >>> boot = np.array([-0.7, -0.5, -0.3])
        >>> aligned, flipped = align_factor_loadings_by_dot_product(orig, boot)
        >>> print(aligned)  # [0.7, 0.5, 0.3]
        >>> print(flipped)  # True
    """
    # 내적 계산
    dot_product = np.dot(original_loadings, bootstrap_loadings)
    
    # 부호 반전 필요 여부 판단
    if dot_product < threshold:
        return -bootstrap_loadings, True
    else:
        return bootstrap_loadings, False


def align_factor_scores_by_correlation(
    original_scores: np.ndarray,
    bootstrap_scores: np.ndarray,
    threshold: float = 0.0
) -> Tuple[np.ndarray, bool]:
    """
    상관계수를 사용한 요인점수 부호 정렬
    
    원리:
    - 원본과 부트스트랩 요인점수의 상관계수가 양수면 같은 방향
    - 상관계수가 음수면 반대 방향 → 부호 반전 필요
    
    Args:
        original_scores: 원본 요인점수 (n_individuals,)
        bootstrap_scores: 부트스트랩 요인점수 (n_individuals,)
        threshold: 부호 반전 임계값 (기본값: 0.0)
    
    Returns:
        (정렬된 부트스트랩 요인점수, 반전 여부)
    
    Example:
        >>> orig = np.array([1.2, 0.5, -0.3, -0.8])
        >>> boot = np.array([-1.1, -0.4, 0.2, 0.9])
        >>> aligned, flipped = align_factor_scores_by_correlation(orig, boot)
        >>> print(aligned)  # [1.1, 0.4, -0.2, -0.9]
        >>> print(flipped)  # True
    """
    # 상관계수 계산
    if len(original_scores) < 2:
        # 샘플 수가 너무 적으면 정렬 불가
        logger.warning("요인점수 샘플 수가 너무 적어 부호 정렬을 수행할 수 없습니다.")
        return bootstrap_scores, False
    
    correlation = np.corrcoef(original_scores, bootstrap_scores)[0, 1]
    
    # NaN 체크
    if np.isnan(correlation):
        logger.warning("상관계수 계산 실패 (NaN). 부호 정렬을 수행하지 않습니다.")
        return bootstrap_scores, False
    
    # 부호 반전 필요 여부 판단
    if correlation < threshold:
        return -bootstrap_scores, True
    else:
        return bootstrap_scores, False


def align_all_factor_scores(
    original_scores_dict: Dict[str, np.ndarray],
    bootstrap_scores_dict: Dict[str, np.ndarray],
    method: str = 'correlation'
) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
    """
    모든 잠재변수의 요인점수 부호 정렬
    
    Args:
        original_scores_dict: 원본 요인점수 딕셔너리
            {'purchase_intention': np.ndarray, 'perceived_benefit': np.ndarray, ...}
        bootstrap_scores_dict: 부트스트랩 요인점수 딕셔너리
        method: 정렬 방법 ('correlation' 또는 'dot_product')
    
    Returns:
        (정렬된 부트스트랩 요인점수 딕셔너리, 반전 여부 딕셔너리)
    
    Example:
        >>> aligned, flipped = align_all_factor_scores(orig_scores, boot_scores)
        >>> print(flipped)  # {'purchase_intention': True, 'perceived_benefit': False}
    """
    aligned_scores = {}
    flip_status = {}
    
    for lv_name in original_scores_dict.keys():
        if lv_name not in bootstrap_scores_dict:
            logger.warning(f"잠재변수 '{lv_name}'가 부트스트랩 결과에 없습니다. 건너뜁니다.")
            continue
        
        orig_scores = original_scores_dict[lv_name]
        boot_scores = bootstrap_scores_dict[lv_name]
        
        # 길이 체크
        if len(orig_scores) != len(boot_scores):
            logger.warning(
                f"잠재변수 '{lv_name}'의 요인점수 길이가 다릅니다. "
                f"(원본: {len(orig_scores)}, 부트스트랩: {len(boot_scores)}). "
                f"부호 정렬을 수행하지 않습니다."
            )
            aligned_scores[lv_name] = boot_scores
            flip_status[lv_name] = False
            continue
        
        # 부호 정렬
        if method == 'correlation':
            aligned, flipped = align_factor_scores_by_correlation(orig_scores, boot_scores)
        elif method == 'dot_product':
            aligned, flipped = align_factor_loadings_by_dot_product(orig_scores, boot_scores)
        else:
            raise ValueError(f"지원하지 않는 정렬 방법: {method}")
        
        aligned_scores[lv_name] = aligned
        flip_status[lv_name] = flipped

    return aligned_scores, flip_status


def align_loadings_dataframe(
    original_loadings: pd.DataFrame,
    bootstrap_loadings: pd.DataFrame,
    lv_column: str = 'lval',
    indicator_column: str = 'rval',
    estimate_column: str = 'Estimate'
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """
    DataFrame 형식의 요인적재량 부호 정렬

    semopy의 loadings DataFrame을 직접 처리합니다.

    Args:
        original_loadings: 원본 요인적재량 DataFrame
        bootstrap_loadings: 부트스트랩 요인적재량 DataFrame
        lv_column: 잠재변수 컬럼명 (기본값: 'lval')
        indicator_column: 관측변수 컬럼명 (기본값: 'rval')
        estimate_column: 추정값 컬럼명 (기본값: 'Estimate')

    Returns:
        (정렬된 부트스트랩 요인적재량 DataFrame, 반전 여부 딕셔너리)

    Example:
        >>> aligned_loadings, flipped = align_loadings_dataframe(orig_loadings, boot_loadings)
        >>> print(flipped)  # {'purchase_intention': True, 'perceived_benefit': False}
    """
    aligned_loadings = bootstrap_loadings.copy()
    flip_status = {}

    # 각 잠재변수별로 처리
    for lv_name in original_loadings[lv_column].unique():
        # 원본 요인적재량 추출
        orig_lv_loadings = original_loadings[original_loadings[lv_column] == lv_name]
        boot_lv_loadings = bootstrap_loadings[bootstrap_loadings[lv_column] == lv_name]

        # 관측변수 순서 정렬 (일치시키기)
        indicators = orig_lv_loadings[indicator_column].values

        orig_values = []
        boot_values = []

        for indicator in indicators:
            orig_val = orig_lv_loadings[orig_lv_loadings[indicator_column] == indicator][estimate_column].values
            boot_val = boot_lv_loadings[boot_lv_loadings[indicator_column] == indicator][estimate_column].values

            if len(orig_val) > 0 and len(boot_val) > 0:
                orig_values.append(orig_val[0])
                boot_values.append(boot_val[0])

        if len(orig_values) == 0:
            logger.warning(f"잠재변수 '{lv_name}'의 요인적재량을 찾을 수 없습니다. 건너뜁니다.")
            flip_status[lv_name] = False
            continue

        # 부호 정렬
        orig_array = np.array(orig_values)
        boot_array = np.array(boot_values)

        aligned_array, flipped = align_factor_loadings_by_dot_product(orig_array, boot_array)

        # DataFrame 업데이트
        if flipped:
            mask = aligned_loadings[lv_column] == lv_name
            aligned_loadings.loc[mask, estimate_column] *= -1

        flip_status[lv_name] = flipped

    return aligned_loadings, flip_status


def procrustes_align_loadings(
    original_loadings: np.ndarray,
    bootstrap_loadings: np.ndarray
) -> np.ndarray:
    """
    Procrustes 회전을 사용한 요인적재량 정렬

    다중 잠재변수 모델에서 요인적재량 행렬을 원본에 최대한 가깝게 회전시킵니다.

    Args:
        original_loadings: 원본 요인적재량 행렬 (n_indicators, n_factors)
        bootstrap_loadings: 부트스트랩 요인적재량 행렬 (n_indicators, n_factors)

    Returns:
        정렬된 부트스트랩 요인적재량 행렬

    Note:
        scipy.linalg.orthogonal_procrustes를 사용합니다.
        이 방법은 다중 잠재변수 모델에서 유용하지만, 단일 잠재변수 모델에서는
        align_factor_loadings_by_dot_product와 동일한 결과를 제공합니다.

    Example:
        >>> orig = np.array([[0.8, 0.3], [0.6, 0.4], [0.4, 0.5]])
        >>> boot = np.array([[-0.7, 0.2], [-0.5, 0.3], [-0.3, 0.6]])
        >>> aligned = procrustes_align_loadings(orig, boot)
    """
    try:
        from scipy.linalg import orthogonal_procrustes
    except ImportError:
        logger.error("scipy가 설치되지 않았습니다. Procrustes 정렬을 사용할 수 없습니다.")
        logger.info("대안으로 align_factor_loadings_by_dot_product를 사용하세요.")
        raise

    # Procrustes 회전 행렬 계산
    # bootstrap_loadings @ R ≈ original_loadings를 만족하는 직교 행렬 R 찾기
    R, _ = orthogonal_procrustes(bootstrap_loadings, original_loadings)

    # 회전 적용
    aligned_loadings = bootstrap_loadings @ R

    return aligned_loadings


def log_sign_correction_summary(flip_status: Dict[str, bool]) -> None:
    """
    부호 정렬 결과 요약 로깅

    Args:
        flip_status: 반전 여부 딕셔너리
            {'purchase_intention': True, 'perceived_benefit': False, ...}
    """
    n_flipped = sum(flip_status.values())
    n_total = len(flip_status)

    logger.info("=" * 70)
    logger.info("부호 정렬 결과 요약")
    logger.info("=" * 70)
    logger.info(f"총 잠재변수 수: {n_total}")
    logger.info(f"부호 반전된 잠재변수 수: {n_flipped}")
    logger.info(f"부호 유지된 잠재변수 수: {n_total - n_flipped}")
    logger.info("")

    if n_flipped > 0:
        logger.info("부호 반전된 잠재변수:")
        for lv_name, flipped in flip_status.items():
            if flipped:
                logger.info(f"  - {lv_name}")
    else:
        logger.info("모든 잠재변수의 부호가 원본과 일치합니다.")

    logger.info("=" * 70)


