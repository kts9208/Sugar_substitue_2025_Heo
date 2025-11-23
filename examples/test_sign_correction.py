"""
Sign Correction 기능 테스트 스크립트

이 스크립트는 Sign Correction 기능을 테스트하고 효과를 비교합니다.

실행 방법:
    python examples/test_sign_correction.py

Author: Augment Agent
Date: 2025-11-23
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.sign_correction import (
    align_factor_loadings_by_dot_product,
    align_factor_scores_by_correlation,
    align_all_factor_scores,
    align_loadings_dataframe,
    log_sign_correction_summary
)


def test_basic_alignment():
    """기본 부호 정렬 테스트"""
    print("=" * 80)
    print("테스트 1: 기본 부호 정렬")
    print("=" * 80)
    
    # 원본 요인적재량
    original_loadings = np.array([0.8, 0.6, 0.4])
    
    # 부트스트랩 요인적재량 (부호 반전됨)
    bootstrap_loadings = np.array([-0.7, -0.5, -0.3])
    
    print(f"\n원본 요인적재량: {original_loadings}")
    print(f"부트스트랩 요인적재량 (반전됨): {bootstrap_loadings}")
    
    # 부호 정렬
    aligned, flipped = align_factor_loadings_by_dot_product(original_loadings, bootstrap_loadings)
    
    print(f"\n정렬 후 요인적재량: {aligned}")
    print(f"부호 반전 여부: {flipped}")
    print(f"✅ 테스트 통과: {np.allclose(aligned, [0.7, 0.5, 0.3])}")


def test_factor_score_alignment():
    """요인점수 부호 정렬 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: 요인점수 부호 정렬")
    print("=" * 80)
    
    # 원본 요인점수
    np.random.seed(42)
    original_scores = np.random.randn(100)
    
    # 부트스트랩 요인점수 (부호 반전 + 노이즈)
    bootstrap_scores = -original_scores + np.random.randn(100) * 0.1
    
    print(f"\n원본 요인점수 평균: {original_scores.mean():.4f}")
    print(f"부트스트랩 요인점수 평균 (반전됨): {bootstrap_scores.mean():.4f}")
    
    # 상관계수 확인
    corr_before = np.corrcoef(original_scores, bootstrap_scores)[0, 1]
    print(f"정렬 전 상관계수: {corr_before:.4f}")
    
    # 부호 정렬
    aligned, flipped = align_factor_scores_by_correlation(original_scores, bootstrap_scores)
    
    corr_after = np.corrcoef(original_scores, aligned)[0, 1]
    print(f"\n정렬 후 상관계수: {corr_after:.4f}")
    print(f"부호 반전 여부: {flipped}")
    print(f"✅ 테스트 통과: {corr_after > 0.9}")


def test_multiple_lv_alignment():
    """다중 잠재변수 부호 정렬 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: 다중 잠재변수 부호 정렬")
    print("=" * 80)
    
    # 원본 요인점수 (3개 잠재변수)
    np.random.seed(42)
    original_scores = {
        'purchase_intention': np.random.randn(100),
        'perceived_benefit': np.random.randn(100),
        'nutrition_knowledge': np.random.randn(100)
    }
    
    # 부트스트랩 요인점수 (일부 반전)
    bootstrap_scores = {
        'purchase_intention': -original_scores['purchase_intention'] + np.random.randn(100) * 0.1,  # 반전
        'perceived_benefit': original_scores['perceived_benefit'] + np.random.randn(100) * 0.1,  # 유지
        'nutrition_knowledge': -original_scores['nutrition_knowledge'] + np.random.randn(100) * 0.1  # 반전
    }
    
    print("\n정렬 전 상관계수:")
    for lv_name in original_scores.keys():
        corr = np.corrcoef(original_scores[lv_name], bootstrap_scores[lv_name])[0, 1]
        print(f"  {lv_name}: {corr:.4f}")
    
    # 부호 정렬
    aligned_scores, flip_status = align_all_factor_scores(original_scores, bootstrap_scores)
    
    print("\n정렬 후 상관계수:")
    for lv_name in original_scores.keys():
        corr = np.corrcoef(original_scores[lv_name], aligned_scores[lv_name])[0, 1]
        print(f"  {lv_name}: {corr:.4f}")
    
    print("\n부호 반전 상태:")
    for lv_name, flipped in flip_status.items():
        print(f"  {lv_name}: {'반전됨' if flipped else '유지됨'}")
    
    # 로깅 테스트
    print()
    log_sign_correction_summary(flip_status)
    
    print(f"✅ 테스트 통과: {all(np.corrcoef(original_scores[lv], aligned_scores[lv])[0, 1] > 0.9 for lv in original_scores.keys())}")


def test_dataframe_alignment():
    """DataFrame 형식 요인적재량 정렬 테스트"""
    print("\n" + "=" * 80)
    print("테스트 4: DataFrame 형식 요인적재량 정렬")
    print("=" * 80)
    
    # 원본 요인적재량 DataFrame (semopy 형식)
    original_loadings = pd.DataFrame({
        'lval': ['purchase_intention', 'purchase_intention', 'purchase_intention',
                 'perceived_benefit', 'perceived_benefit', 'perceived_benefit'],
        'op': ['=~'] * 6,
        'rval': ['pi1', 'pi2', 'pi3', 'pb1', 'pb2', 'pb3'],
        'Estimate': [0.8, 0.6, 0.4, 0.7, 0.5, 0.3]
    })
    
    # 부트스트랩 요인적재량 (purchase_intention만 반전)
    bootstrap_loadings = pd.DataFrame({
        'lval': ['purchase_intention', 'purchase_intention', 'purchase_intention',
                 'perceived_benefit', 'perceived_benefit', 'perceived_benefit'],
        'op': ['=~'] * 6,
        'rval': ['pi1', 'pi2', 'pi3', 'pb1', 'pb2', 'pb3'],
        'Estimate': [-0.75, -0.55, -0.35, 0.65, 0.45, 0.25]
    })
    
    print("\n원본 요인적재량:")
    print(original_loadings[['lval', 'rval', 'Estimate']])
    
    print("\n부트스트랩 요인적재량 (정렬 전):")
    print(bootstrap_loadings[['lval', 'rval', 'Estimate']])
    
    # 부호 정렬
    aligned_loadings, flip_status = align_loadings_dataframe(original_loadings, bootstrap_loadings)
    
    print("\n부트스트랩 요인적재량 (정렬 후):")
    print(aligned_loadings[['lval', 'rval', 'Estimate']])
    
    print("\n부호 반전 상태:")
    for lv_name, flipped in flip_status.items():
        print(f"  {lv_name}: {'반전됨' if flipped else '유지됨'}")
    
    print(f"✅ 테스트 통과: {flip_status['purchase_intention'] and not flip_status['perceived_benefit']}")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Sign Correction 기능 테스트")
    print("=" * 80)
    
    test_basic_alignment()
    test_factor_score_alignment()
    test_multiple_lv_alignment()
    test_dataframe_alignment()
    
    print("\n" + "=" * 80)
    print("모든 테스트 완료!")
    print("=" * 80)

