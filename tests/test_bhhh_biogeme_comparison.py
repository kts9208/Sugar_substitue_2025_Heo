"""
BHHH 모듈 Biogeme 비교 검증 테스트

현재 구현의 BHHH 계산이 Biogeme의 BHHH 계산과
일치하는지 검증합니다.

Biogeme의 BHHH 계산 방식:
- OPG (Outer Product of Gradients) 사용
- H_BHHH = Σ_i (g_i × g_i^T)
- Robust SE = sqrt(diag(inv(H_BHHH)))

Author: Taeseok Kim
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.bhhh_calculator import BHHHCalculator


# ============================================================================
# Biogeme 스타일 BHHH 계산 (참조 구현)
# ============================================================================

def compute_bhhh_biogeme_style(individual_gradients: list) -> np.ndarray:
    """
    Biogeme 스타일 BHHH 행렬 계산
    
    Biogeme 소스코드 참조:
    https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/results.py
    
    BHHH = Σ_i (g_i × g_i^T)
    
    Args:
        individual_gradients: 개인별 gradient 리스트
    
    Returns:
        BHHH 행렬 (n_params, n_params)
    """
    n_params = len(individual_gradients[0])
    bhhh = np.zeros((n_params, n_params))
    
    # Biogeme: BHHH = sum of outer products
    for grad in individual_gradients:
        bhhh += np.outer(grad, grad)
    
    return bhhh


def compute_robust_se_biogeme_style(individual_gradients: list) -> np.ndarray:
    """
    Biogeme 스타일 Robust 표준오차 계산
    
    Robust SE = sqrt(diag(inv(BHHH)))
    
    Args:
        individual_gradients: 개인별 gradient 리스트
    
    Returns:
        Robust 표준오차 벡터 (n_params,)
    """
    bhhh = compute_bhhh_biogeme_style(individual_gradients)
    
    # 역행렬 계산
    try:
        bhhh_inv = np.linalg.inv(bhhh)
    except np.linalg.LinAlgError:
        # 특이행렬인 경우 정규화
        n_params = bhhh.shape[0]
        bhhh_reg = bhhh + 1e-8 * np.eye(n_params)
        bhhh_inv = np.linalg.inv(bhhh_reg)
    
    # 표준오차 = sqrt(diag(inv(BHHH)))
    variances = np.diag(bhhh_inv)
    robust_se = np.sqrt(np.abs(variances))
    
    return robust_se


def compute_sandwich_estimator_biogeme_style(
    individual_gradients: list,
    hessian_numerical: np.ndarray
) -> np.ndarray:
    """
    Biogeme 스타일 Sandwich estimator 계산
    
    Sandwich = H^(-1) @ BHHH @ H^(-1)
    Robust SE = sqrt(diag(Sandwich))
    
    Args:
        individual_gradients: 개인별 gradient 리스트
        hessian_numerical: 수치적 Hessian 행렬
    
    Returns:
        Sandwich 기반 표준오차 벡터 (n_params,)
    """
    bhhh = compute_bhhh_biogeme_style(individual_gradients)
    
    # Hessian 역행렬
    try:
        h_inv = np.linalg.inv(hessian_numerical)
    except np.linalg.LinAlgError:
        n_params = hessian_numerical.shape[0]
        h_reg = hessian_numerical + 1e-8 * np.eye(n_params)
        h_inv = np.linalg.inv(h_reg)
    
    # Sandwich = H^(-1) @ BHHH @ H^(-1)
    sandwich = h_inv @ bhhh @ h_inv
    
    # 표준오차
    variances = np.diag(sandwich)
    robust_se = np.sqrt(np.abs(variances))
    
    return robust_se


# ============================================================================
# 테스트 케이스
# ============================================================================

class TestBHHHBiogemeComparison:
    """BHHH 계산 Biogeme 비교 테스트"""
    
    def test_bhhh_matrix_calculation(self):
        """BHHH 행렬 계산 비교"""
        print("\n" + "="*80)
        print("테스트 1: BHHH 행렬 계산 비교 (Biogeme)")
        print("="*80)
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_individuals = 100
        n_params = 10
        
        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]
        
        # 1. 현재 구현
        bhhh_calc = BHHHCalculator()
        
        # 최대화 문제로 계산 (Biogeme는 최대화)
        hessian_bhhh_ours = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=False  # 최대화 문제
        )
        
        # 2. Biogeme 스타일
        bhhh_biogeme = compute_bhhh_biogeme_style(individual_gradients)
        
        # 3. 비교
        diff = np.abs(hessian_bhhh_ours - bhhh_biogeme)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n현재 구현 BHHH 통계:")
        print(f"  - Shape: {hessian_bhhh_ours.shape}")
        print(f"  - 범위: [{np.min(hessian_bhhh_ours):.6e}, {np.max(hessian_bhhh_ours):.6e}]")
        print(f"  - 평균: {np.mean(hessian_bhhh_ours):.6e}")
        print(f"  - Frobenius norm: {np.linalg.norm(hessian_bhhh_ours, 'fro'):.6e}")
        
        print(f"\nBiogeme 스타일 BHHH 통계:")
        print(f"  - Shape: {bhhh_biogeme.shape}")
        print(f"  - 범위: [{np.min(bhhh_biogeme):.6e}, {np.max(bhhh_biogeme):.6e}]")
        print(f"  - 평균: {np.mean(bhhh_biogeme):.6e}")
        print(f"  - Frobenius norm: {np.linalg.norm(bhhh_biogeme, 'fro'):.6e}")
        
        print(f"\n차이 통계:")
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {mean_diff / np.mean(np.abs(bhhh_biogeme)):.6e}")
        
        # 검증
        assert np.allclose(hessian_bhhh_ours, bhhh_biogeme, rtol=1e-10, atol=1e-12), \
            f"BHHH 행렬 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ BHHH 행렬 계산 Biogeme와 일치!")
        return True
    
    def test_robust_se_calculation(self):
        """Robust 표준오차 계산 비교"""
        print("\n" + "="*80)
        print("테스트 2: Robust 표준오차 계산 비교 (Biogeme)")
        print("="*80)
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_individuals = 100
        n_params = 10
        
        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]
        
        # 1. 현재 구현
        bhhh_calc = BHHHCalculator()
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=False  # 최대화
        )
        hess_inv_ours = bhhh_calc.compute_hessian_inverse(hessian_bhhh)
        se_ours = bhhh_calc.compute_standard_errors(hess_inv_ours)
        
        # 2. Biogeme 스타일
        se_biogeme = compute_robust_se_biogeme_style(individual_gradients)
        
        # 3. 비교
        diff = np.abs(se_ours - se_biogeme)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n현재 구현 Robust SE:")
        print(f"  - 범위: [{np.min(se_ours):.6e}, {np.max(se_ours):.6e}]")
        print(f"  - 평균: {np.mean(se_ours):.6e}")
        print(f"  - 상위 5개: {se_ours[:5]}")
        
        print(f"\nBiogeme 스타일 Robust SE:")
        print(f"  - 범위: [{np.min(se_biogeme):.6e}, {np.max(se_biogeme):.6e}]")
        print(f"  - 평균: {np.mean(se_biogeme):.6e}")
        print(f"  - 상위 5개: {se_biogeme[:5]}")
        
        print(f"\n차이 통계:")
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {mean_diff / np.mean(se_biogeme):.6e}")
        
        # 검증
        assert np.allclose(se_ours, se_biogeme, rtol=1e-8, atol=1e-10), \
            f"Robust SE 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ Robust 표준오차 계산 Biogeme와 일치!")
        return True
    
    def test_sandwich_estimator(self):
        """Sandwich estimator 계산 비교"""
        print("\n" + "="*80)
        print("테스트 3: Sandwich Estimator 계산 비교 (Biogeme)")
        print("="*80)
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_individuals = 100
        n_params = 10
        
        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]
        
        # 수치적 Hessian (임의 생성 - 대칭 행렬)
        hessian_numerical = np.random.randn(n_params, n_params)
        hessian_numerical = (hessian_numerical + hessian_numerical.T) / 2
        
        # 1. 현재 구현
        bhhh_calc = BHHHCalculator()
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=False
        )
        se_sandwich_ours = bhhh_calc.compute_robust_standard_errors(
            hessian_bhhh,
            hessian_numerical
        )
        
        # 2. Biogeme 스타일
        se_sandwich_biogeme = compute_sandwich_estimator_biogeme_style(
            individual_gradients,
            hessian_numerical
        )
        
        # 3. 비교
        diff = np.abs(se_sandwich_ours - se_sandwich_biogeme)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n현재 구현 Sandwich SE:")
        print(f"  - 범위: [{np.min(se_sandwich_ours):.6e}, {np.max(se_sandwich_ours):.6e}]")
        print(f"  - 평균: {np.mean(se_sandwich_ours):.6e}")
        
        print(f"\nBiogeme 스타일 Sandwich SE:")
        print(f"  - 범위: [{np.min(se_sandwich_biogeme):.6e}, {np.max(se_sandwich_biogeme):.6e}]")
        print(f"  - 평균: {np.mean(se_sandwich_biogeme):.6e}")
        
        print(f"\n차이 통계:")
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {mean_diff / np.mean(se_sandwich_biogeme):.6e}")
        
        # 검증 (수치 오차 허용)
        assert np.allclose(se_sandwich_ours, se_sandwich_biogeme, rtol=1e-5, atol=1e-8), \
            f"Sandwich SE 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ Sandwich Estimator 계산 Biogeme와 일치!")
        return True
    
    def test_bhhh_minimization_vs_maximization(self):
        """최소화 vs 최대화 문제 부호 검증"""
        print("\n" + "="*80)
        print("테스트 4: 최소화 vs 최대화 문제 부호 검증")
        print("="*80)
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_individuals = 50
        n_params = 10
        
        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]
        
        # 1. 최소화 문제 (scipy.optimize.minimize)
        bhhh_calc_min = BHHHCalculator()
        hessian_min = bhhh_calc_min.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=True
        )
        
        # 2. 최대화 문제 (Biogeme)
        bhhh_calc_max = BHHHCalculator()
        hessian_max = bhhh_calc_max.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=False
        )
        
        # 3. 부호 검증
        diff = np.abs(hessian_min + hessian_max)
        max_diff = np.max(diff)
        
        print(f"\n최소화 문제 BHHH:")
        print(f"  - 평균: {np.mean(hessian_min):.6e}")
        print(f"  - 부호: {'음수' if np.mean(hessian_min) < 0 else '양수'}")
        
        print(f"\n최대화 문제 BHHH:")
        print(f"  - 평균: {np.mean(hessian_max):.6e}")
        print(f"  - 부호: {'음수' if np.mean(hessian_max) < 0 else '양수'}")
        
        print(f"\n부호 관계:")
        print(f"  - H_min = -H_max 검증")
        print(f"  - 최대 차이: {max_diff:.6e}")
        
        # 검증
        assert np.allclose(hessian_min, -hessian_max, rtol=1e-10, atol=1e-12), \
            f"부호 관계 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ 최소화/최대화 부호 관계 확인!")
        return True


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("BHHH 모듈 Biogeme 비교 검증 테스트")
    print("="*80)
    
    test = TestBHHHBiogemeComparison()
    
    try:
        # 테스트 실행
        test.test_bhhh_matrix_calculation()
        test.test_robust_se_calculation()
        test.test_sandwich_estimator()
        test.test_bhhh_minimization_vs_maximization()
        
        print("\n" + "="*80)
        print("✅ 모든 BHHH 검증 테스트 통과!")
        print("="*80)
        print("\n결론:")
        print("  - 현재 구현의 BHHH 계산이 Biogeme 방식과 완벽히 일치합니다.")
        print("  - OPG (Outer Product of Gradients) 계산이 정확합니다.")
        print("  - Robust 표준오차 계산이 정확합니다.")
        print("  - Sandwich estimator 계산이 정확합니다.")
        print("  - 최소화/최대화 문제 부호 처리가 정확합니다.")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

