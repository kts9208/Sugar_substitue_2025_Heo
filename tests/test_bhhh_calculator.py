"""
BHHH Calculator 테스트

BHHH 모듈의 기능을 검증하는 단위 테스트입니다.

Author: Taeseok Kim
Date: 2025-11-13
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.bhhh_calculator import BHHHCalculator


class TestBHHHCalculator:
    """BHHH Calculator 테스트 클래스"""
    
    @pytest.fixture
    def sample_gradients(self):
        """샘플 개인별 gradient 생성"""
        np.random.seed(42)
        n_individuals = 50
        n_params = 10
        
        # 개인별 gradient (정규분포)
        gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]
        
        return gradients
    
    @pytest.fixture
    def bhhh_calc(self):
        """BHHH Calculator 인스턴스"""
        return BHHHCalculator()
    
    def test_compute_bhhh_hessian_shape(self, bhhh_calc, sample_gradients):
        """BHHH Hessian 행렬 크기 테스트"""
        n_params = len(sample_gradients[0])
        
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        assert hessian.shape == (n_params, n_params)
        assert bhhh_calc.hessian_bhhh is not None
    
    def test_compute_bhhh_hessian_symmetry(self, bhhh_calc, sample_gradients):
        """BHHH Hessian 행렬 대칭성 테스트"""
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        # 대칭 행렬인지 확인
        assert np.allclose(hessian, hessian.T)
    
    def test_compute_bhhh_hessian_sign(self, bhhh_calc, sample_gradients):
        """BHHH Hessian 부호 테스트"""
        # 최소화 문제
        hessian_min = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        # 최대화 문제
        hessian_max = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=False
        )
        
        # 부호가 반대인지 확인
        assert np.allclose(hessian_min, -hessian_max)
    
    def test_compute_hessian_inverse_shape(self, bhhh_calc, sample_gradients):
        """Hessian 역행렬 크기 테스트"""
        n_params = len(sample_gradients[0])
        
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian)
        
        assert hess_inv.shape == (n_params, n_params)
        assert bhhh_calc.hessian_inv is not None
    
    def test_compute_hessian_inverse_identity(self, bhhh_calc, sample_gradients):
        """Hessian 역행렬 검증 (H @ H^(-1) = I)"""
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian)
        
        # H @ H^(-1) = I
        identity = hessian @ hess_inv
        n_params = len(sample_gradients[0])
        expected_identity = np.eye(n_params)
        
        assert np.allclose(identity, expected_identity, atol=1e-6)
    
    def test_compute_standard_errors_positive(self, bhhh_calc, sample_gradients):
        """표준오차 양수 테스트"""
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian)
        se = bhhh_calc.compute_standard_errors(hess_inv)
        
        # 모든 표준오차가 양수인지 확인
        assert np.all(se > 0)
        assert bhhh_calc.standard_errors is not None
    
    def test_compute_standard_errors_length(self, bhhh_calc, sample_gradients):
        """표준오차 길이 테스트"""
        n_params = len(sample_gradients[0])
        
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian)
        se = bhhh_calc.compute_standard_errors(hess_inv)
        
        assert len(se) == n_params
    
    def test_compute_t_statistics(self, bhhh_calc, sample_gradients):
        """t-통계량 계산 테스트"""
        n_params = len(sample_gradients[0])
        parameters = np.random.randn(n_params)
        
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian)
        se = bhhh_calc.compute_standard_errors(hess_inv)
        
        t_stats = bhhh_calc.compute_t_statistics(parameters, se)
        
        # t = θ / SE
        expected_t = parameters / se
        assert np.allclose(t_stats, expected_t)
    
    def test_compute_p_values_range(self, bhhh_calc, sample_gradients):
        """p-값 범위 테스트 (0 ~ 1)"""
        n_params = len(sample_gradients[0])
        parameters = np.random.randn(n_params)
        
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian)
        se = bhhh_calc.compute_standard_errors(hess_inv)
        t_stats = bhhh_calc.compute_t_statistics(parameters, se)
        
        p_values = bhhh_calc.compute_p_values(t_stats)
        
        # p-값이 0~1 범위인지 확인
        assert np.all(p_values >= 0)
        assert np.all(p_values <= 1)
    
    def test_get_results_summary(self, bhhh_calc, sample_gradients):
        """결과 요약 DataFrame 테스트"""
        n_params = len(sample_gradients[0])
        parameters = np.random.randn(n_params)
        param_names = [f"param_{i}" for i in range(n_params)]
        
        hessian = bhhh_calc.compute_bhhh_hessian(
            sample_gradients,
            for_minimization=True
        )
        
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian)
        se = bhhh_calc.compute_standard_errors(hess_inv)
        
        summary_df = bhhh_calc.get_results_summary(parameters, param_names)
        
        # DataFrame 구조 확인
        assert len(summary_df) == n_params
        assert 'Parameter' in summary_df.columns
        assert 'Estimate' in summary_df.columns
        assert 'Std.Error' in summary_df.columns
        assert 't-statistic' in summary_df.columns
        assert 'p-value' in summary_df.columns
        assert 'Significant' in summary_df.columns
    
    def test_empty_gradients_error(self, bhhh_calc):
        """빈 gradient 리스트 오류 테스트"""
        with pytest.raises(ValueError, match="개인별 gradient가 비어있습니다"):
            bhhh_calc.compute_bhhh_hessian([])
    
    def test_inconsistent_gradient_dimensions_error(self, bhhh_calc):
        """불일치 gradient 차원 오류 테스트"""
        gradients = [
            np.random.randn(10),
            np.random.randn(5),  # 다른 차원
        ]
        
        with pytest.raises(ValueError, match="gradient 차원 불일치"):
            bhhh_calc.compute_bhhh_hessian(gradients)
    
    def test_regularization_effect(self, bhhh_calc):
        """정규화 효과 테스트"""
        # 특이 행렬 생성 (rank-deficient)
        n_params = 10
        gradients = [np.ones(n_params) for _ in range(5)]
        
        hessian = bhhh_calc.compute_bhhh_hessian(
            gradients,
            for_minimization=True
        )
        
        # 정규화 없이는 역행렬 계산 실패할 수 있음
        # 정규화로 안정적으로 계산
        hess_inv = bhhh_calc.compute_hessian_inverse(
            hessian,
            regularization=1e-6
        )
        
        assert hess_inv is not None
        assert not np.any(np.isnan(hess_inv))
        assert not np.any(np.isinf(hess_inv))


class TestBHHHIntegration:
    """BHHH 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우 테스트"""
        np.random.seed(42)
        
        # 1. 샘플 데이터 생성
        n_individuals = 100
        n_params = 20
        individual_gradients = [
            np.random.randn(n_params) for _ in range(n_individuals)
        ]
        parameters = np.random.randn(n_params)
        param_names = [f"beta_{i}" for i in range(n_params)]
        
        # 2. BHHH 계산기 초기화
        bhhh_calc = BHHHCalculator()
        
        # 3. BHHH Hessian 계산
        hessian = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=True
        )
        
        # 4. Hessian 역행렬 계산
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian)
        
        # 5. 표준오차 계산
        se = bhhh_calc.compute_standard_errors(hess_inv)
        
        # 6. 결과 요약
        summary_df = bhhh_calc.get_results_summary(parameters, param_names)
        
        # 검증
        assert hessian.shape == (n_params, n_params)
        assert hess_inv.shape == (n_params, n_params)
        assert len(se) == n_params
        assert len(summary_df) == n_params
        assert np.all(se > 0)
        
        print("\n=== BHHH 전체 워크플로우 테스트 성공 ===")
        print(f"파라미터 수: {n_params}")
        print(f"개인 수: {n_individuals}")
        print(f"표준오차 범위: [{np.min(se):.6f}, {np.max(se):.6f}]")
        print("\n결과 요약 (상위 5개):")
        print(summary_df.head())


if __name__ == "__main__":
    # 직접 실행 시 통합 테스트만 실행
    print("BHHH Calculator 통합 테스트 실행 중...\n")
    test = TestBHHHIntegration()
    test.test_full_workflow()
    print("\n✅ 모든 테스트 통과!")

