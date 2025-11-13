"""
OPG 계산 Statsmodels 비교 검증 테스트

현재 구현의 OPG (Outer Product of Gradients) 계산이
Statsmodels의 OPG 계산과 일치하는지 검증합니다.

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
# Statsmodels 기반 OPG 계산 (참조 구현)
# ============================================================================

def compute_opg_statsmodels_style(individual_gradients: list) -> np.ndarray:
    """
    Statsmodels 스타일 OPG 계산
    
    OPG = Σ_i (g_i × g_i^T)
    
    Args:
        individual_gradients: 개인별 gradient 리스트
    
    Returns:
        OPG 행렬 (n_params, n_params)
    """
    # Statsmodels는 gradient를 (n_obs, n_params) 행렬로 저장
    score_obs = np.array(individual_gradients)  # (n_individuals, n_params)
    
    # OPG = score_obs.T @ score_obs
    # = Σ_i (g_i × g_i^T)
    opg_matrix = score_obs.T @ score_obs
    
    return opg_matrix


def compute_opg_covariance_statsmodels(individual_gradients: list) -> np.ndarray:
    """
    Statsmodels 스타일 OPG 공분산 행렬 계산
    
    Cov = inv(OPG) = inv(Σ_i g_i × g_i^T)
    
    Args:
        individual_gradients: 개인별 gradient 리스트
    
    Returns:
        OPG 공분산 행렬 (n_params, n_params)
    """
    opg = compute_opg_statsmodels_style(individual_gradients)
    
    # 공분산 = OPG의 역행렬
    try:
        cov_opg = np.linalg.inv(opg)
    except np.linalg.LinAlgError:
        # 특이행렬인 경우 pseudo-inverse
        cov_opg = np.linalg.pinv(opg)
    
    return cov_opg


# ============================================================================
# 테스트 케이스
# ============================================================================

class TestOPGComparison:
    """OPG 계산 비교 테스트"""
    
    def test_opg_matrix_calculation(self):
        """OPG 행렬 계산 비교"""
        print("\n" + "="*80)
        print("테스트 1: OPG 행렬 계산 비교")
        print("="*80)
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_individuals = 100
        n_params = 10
        
        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]
        
        # 1. 현재 구현 (BHHH Calculator)
        bhhh_calc = BHHHCalculator()
        
        # BHHH Hessian = -OPG (최소화 문제)
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=True
        )
        opg_ours = -hessian_bhhh  # OPG = -BHHH (최소화 문제)
        
        # 2. Statsmodels 스타일
        opg_statsmodels = compute_opg_statsmodels_style(individual_gradients)
        
        # 3. 비교
        diff = np.abs(opg_ours - opg_statsmodels)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n현재 구현 OPG 통계:")
        print(f"  - Shape: {opg_ours.shape}")
        print(f"  - 범위: [{np.min(opg_ours):.6e}, {np.max(opg_ours):.6e}]")
        print(f"  - 평균: {np.mean(opg_ours):.6e}")
        
        print(f"\nStatsmodels 스타일 OPG 통계:")
        print(f"  - Shape: {opg_statsmodels.shape}")
        print(f"  - 범위: [{np.min(opg_statsmodels):.6e}, {np.max(opg_statsmodels):.6e}]")
        print(f"  - 평균: {np.mean(opg_statsmodels):.6e}")
        
        print(f"\n차이 통계:")
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {mean_diff / np.mean(np.abs(opg_statsmodels)):.6e}")
        
        # 검증
        assert np.allclose(opg_ours, opg_statsmodels, rtol=1e-10, atol=1e-12), \
            f"OPG 행렬 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ OPG 행렬 계산 일치!")
        return True
    
    def test_opg_covariance_calculation(self):
        """OPG 공분산 행렬 계산 비교"""
        print("\n" + "="*80)
        print("테스트 2: OPG 공분산 행렬 계산 비교")
        print("="*80)

        # 샘플 데이터 생성
        np.random.seed(42)
        n_individuals = 100
        n_params = 10

        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]

        # 1. 현재 구현 (최대화 문제로 계산 - Statsmodels와 동일)
        bhhh_calc = BHHHCalculator()
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=False  # 최대화 문제
        )
        cov_ours = bhhh_calc.compute_hessian_inverse(hessian_bhhh)

        # 2. Statsmodels 스타일
        cov_statsmodels = compute_opg_covariance_statsmodels(individual_gradients)
        
        # 3. 비교
        diff = np.abs(cov_ours - cov_statsmodels)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n현재 구현 공분산 행렬 통계:")
        print(f"  - Shape: {cov_ours.shape}")
        print(f"  - 대각 원소 범위: [{np.min(np.diag(cov_ours)):.6e}, {np.max(np.diag(cov_ours)):.6e}]")
        print(f"  - 평균: {np.mean(cov_ours):.6e}")
        
        print(f"\nStatsmodels 스타일 공분산 행렬 통계:")
        print(f"  - Shape: {cov_statsmodels.shape}")
        print(f"  - 대각 원소 범위: [{np.min(np.diag(cov_statsmodels)):.6e}, {np.max(np.diag(cov_statsmodels)):.6e}]")
        print(f"  - 평균: {np.mean(cov_statsmodels):.6e}")
        
        print(f"\n차이 통계:")
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {mean_diff / np.mean(np.abs(cov_statsmodels)):.6e}")
        
        # 검증
        assert np.allclose(cov_ours, cov_statsmodels, rtol=1e-8, atol=1e-10), \
            f"공분산 행렬 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ OPG 공분산 행렬 계산 일치!")
        return True
    
    def test_opg_standard_errors(self):
        """OPG 표준오차 계산 비교"""
        print("\n" + "="*80)
        print("테스트 3: OPG 표준오차 계산 비교")
        print("="*80)

        # 샘플 데이터 생성
        np.random.seed(42)
        n_individuals = 100
        n_params = 10

        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]

        # 1. 현재 구현 (최대화 문제)
        bhhh_calc = BHHHCalculator()
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=False  # 최대화 문제
        )
        cov_ours = bhhh_calc.compute_hessian_inverse(hessian_bhhh)
        se_ours = bhhh_calc.compute_standard_errors(cov_ours)
        
        # 2. Statsmodels 스타일
        cov_statsmodels = compute_opg_covariance_statsmodels(individual_gradients)
        se_statsmodels = np.sqrt(np.diag(cov_statsmodels))
        
        # 3. 비교
        diff = np.abs(se_ours - se_statsmodels)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n현재 구현 표준오차:")
        print(f"  - 범위: [{np.min(se_ours):.6e}, {np.max(se_ours):.6e}]")
        print(f"  - 평균: {np.mean(se_ours):.6e}")
        print(f"  - 상위 5개: {se_ours[:5]}")
        
        print(f"\nStatsmodels 스타일 표준오차:")
        print(f"  - 범위: [{np.min(se_statsmodels):.6e}, {np.max(se_statsmodels):.6e}]")
        print(f"  - 평균: {np.mean(se_statsmodels):.6e}")
        print(f"  - 상위 5개: {se_statsmodels[:5]}")
        
        print(f"\n차이 통계:")
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {mean_diff / np.mean(se_statsmodels):.6e}")
        
        # 검증
        assert np.allclose(se_ours, se_statsmodels, rtol=1e-8, atol=1e-10), \
            f"표준오차 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ OPG 표준오차 계산 일치!")
        return True
    
    def test_opg_symmetry(self):
        """OPG 행렬 대칭성 검증"""
        print("\n" + "="*80)
        print("테스트 4: OPG 행렬 대칭성 검증")
        print("="*80)
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_individuals = 50
        n_params = 20
        
        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]
        
        # 1. 현재 구현
        bhhh_calc = BHHHCalculator()
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=True
        )
        opg_ours = -hessian_bhhh
        
        # 2. 대칭성 검증
        diff_symmetry = np.abs(opg_ours - opg_ours.T)
        max_asymmetry = np.max(diff_symmetry)
        
        print(f"\n대칭성 검증:")
        print(f"  - 최대 비대칭: {max_asymmetry:.6e}")
        print(f"  - 평균 비대칭: {np.mean(diff_symmetry):.6e}")
        
        # 검증
        assert np.allclose(opg_ours, opg_ours.T, rtol=1e-10, atol=1e-12), \
            f"OPG 행렬이 대칭이 아닙니다: 최대 비대칭 {max_asymmetry}"
        
        print("\n✅ OPG 행렬 대칭성 확인!")
        return True
    
    def test_opg_positive_semidefinite(self):
        """OPG 행렬 양반정부호 검증"""
        print("\n" + "="*80)
        print("테스트 5: OPG 행렬 양반정부호 검증")
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
            for_minimization=True
        )
        opg_ours = -hessian_bhhh
        
        # 2. 고유값 계산
        eigenvalues = np.linalg.eigvalsh(opg_ours)
        
        print(f"\n고유값 통계:")
        print(f"  - 최소 고유값: {np.min(eigenvalues):.6e}")
        print(f"  - 최대 고유값: {np.max(eigenvalues):.6e}")
        print(f"  - 음수 고유값 개수: {np.sum(eigenvalues < 0)}/{len(eigenvalues)}")
        
        # 검증 (양반정부호: 모든 고유값 >= 0)
        assert np.all(eigenvalues >= -1e-10), \
            f"OPG 행렬이 양반정부호가 아닙니다: 최소 고유값 {np.min(eigenvalues)}"
        
        print("\n✅ OPG 행렬 양반정부호 확인!")
        return True


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPG 계산 Statsmodels 비교 검증 테스트")
    print("="*80)
    
    test = TestOPGComparison()
    
    try:
        # 테스트 실행
        test.test_opg_matrix_calculation()
        test.test_opg_covariance_calculation()
        test.test_opg_standard_errors()
        test.test_opg_symmetry()
        test.test_opg_positive_semidefinite()
        
        print("\n" + "="*80)
        print("✅ 모든 OPG 검증 테스트 통과!")
        print("="*80)
        print("\n결론:")
        print("  - 현재 구현의 OPG 계산이 Statsmodels 방식과 완벽히 일치합니다.")
        print("  - OPG 행렬은 대칭이며 양반정부호입니다.")
        print("  - 표준오차 계산이 정확합니다.")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

