"""
그래디언트 계산 유닛 테스트

수치적 그래디언트와 해석적 그래디언트를 비교하여 정확성을 검증합니다.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.optimize import approx_fprime

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.gradient_calculator import (
    MeasurementGradient,
    StructuralGradient,
    ChoiceGradient
)
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig
)


class TestMeasurementGradient:
    """측정모델 그래디언트 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        # 간단한 측정모델 설정
        self.config = MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8'],
            n_categories=5
        )
        
        # 테스트 데이터
        self.data = pd.DataFrame({
            'q6': [3],
            'q7': [4],
            'q8': [2]
        })
        
        # 잠재변수 값
        self.lv = 0.5
        
        # 파라미터
        self.params = {
            'zeta': np.array([1.0, 1.2, 0.8]),
            'tau': np.array([
                [-2.0, -1.0, 1.0, 2.0],
                [-2.0, -1.0, 1.0, 2.0],
                [-2.0, -1.0, 1.0, 2.0]
            ])
        }
        
        # 모델 및 그래디언트 계산기
        self.model = OrderedProbitMeasurement(self.config)
        self.grad_calc = MeasurementGradient(
            n_indicators=3,
            n_categories=5
        )
    
    def test_gradient_shape(self):
        """그래디언트 shape 확인"""
        grad = self.grad_calc.compute_gradient(
            self.data, self.lv, self.params, self.config.indicators
        )
        
        assert grad['grad_zeta'].shape == (3,)
        assert grad['grad_tau'].shape == (3, 4)
    
    def test_gradient_vs_numerical(self):
        """해석적 그래디언트 vs 수치적 그래디언트"""
        # 해석적 그래디언트
        analytic_grad = self.grad_calc.compute_gradient(
            self.data, self.lv, self.params, self.config.indicators
        )
        
        # 로그우도 함수
        def log_likelihood(params_flat):
            n_indicators = 3
            n_thresholds = 4
            
            zeta = params_flat[:n_indicators]
            tau = params_flat[n_indicators:].reshape(n_indicators, n_thresholds)
            
            params = {'zeta': zeta, 'tau': tau}
            return self.model.log_likelihood(self.data, self.lv, params)
        
        # 파라미터를 1D 배열로 변환
        params_flat = np.concatenate([
            self.params['zeta'],
            self.params['tau'].flatten()
        ])
        
        # 수치적 그래디언트 (epsilon 테스트)
        epsilons = [1e-4, 1e-5, 1e-6]
        
        print("\n=== 측정모델 그래디언트 비교 ===")
        print(f"해석적 grad_zeta: {analytic_grad['grad_zeta']}")
        
        for eps in epsilons:
            numerical_grad = approx_fprime(params_flat, log_likelihood, epsilon=eps)
            numerical_grad_zeta = numerical_grad[:3]
            
            diff = np.abs(analytic_grad['grad_zeta'] - numerical_grad_zeta)
            rel_error = diff / (np.abs(analytic_grad['grad_zeta']) + 1e-10)
            
            print(f"\nEpsilon = {eps}")
            print(f"수치적 grad_zeta: {numerical_grad_zeta}")
            print(f"절대 오차: {diff}")
            print(f"상대 오차: {rel_error}")
            
            # 상대 오차가 1% 이내인지 확인
            assert np.all(rel_error < 0.01), f"Epsilon {eps}: 상대 오차가 너무 큼"


class TestStructuralGradient:
    """구조모델 그래디언트 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.config = StructuralConfig(
            sociodemographics=['age_std', 'gender', 'income_std'],
            error_variance=1.0
        )
        
        self.data = pd.DataFrame({
            'age_std': [0.5],
            'gender': [1.0],
            'income_std': [-0.3]
        })
        
        self.lv = 0.8
        self.draw = 0.2  # 표준정규 draw
        
        self.params = {
            'gamma': np.array([0.3, 0.5, 0.2])
        }
        
        self.model = LatentVariableRegression(self.config)
        self.grad_calc = StructuralGradient(
            n_sociodem=3,
            error_variance=1.0
        )
    
    def test_gradient_vs_numerical(self):
        """해석적 그래디언트 vs 수치적 그래디언트"""
        # 해석적 그래디언트
        analytic_grad = self.grad_calc.compute_gradient(
            self.data, self.lv, self.params, self.config.sociodemographics
        )
        
        # 로그우도 함수
        def log_likelihood(gamma):
            params = {'gamma': gamma}
            return self.model.log_likelihood(self.data, self.lv, params, self.draw)
        
        # 수치적 그래디언트
        epsilons = [1e-4, 1e-5, 1e-6]
        
        print("\n=== 구조모델 그래디언트 비교 ===")
        print(f"해석적 grad_gamma: {analytic_grad['grad_gamma']}")
        
        for eps in epsilons:
            numerical_grad = approx_fprime(self.params['gamma'], log_likelihood, epsilon=eps)
            
            diff = np.abs(analytic_grad['grad_gamma'] - numerical_grad)
            rel_error = diff / (np.abs(analytic_grad['grad_gamma']) + 1e-10)
            
            print(f"\nEpsilon = {eps}")
            print(f"수치적 grad_gamma: {numerical_grad}")
            print(f"절대 오차: {diff}")
            print(f"상대 오차: {rel_error}")
            
            # 상대 오차가 1% 이내인지 확인
            assert np.all(rel_error < 0.01), f"Epsilon {eps}: 상대 오차가 너무 큼"


class TestChoiceGradient:
    """선택모델 그래디언트 테스트"""
    
    def setup_method(self):
        """테스트 데이터 준비"""
        self.config = ChoiceConfig(
            choice_attributes=['sugar_free', 'health_label', 'price']
        )
        
        # 3개 선택 상황
        self.data = pd.DataFrame({
            'sugar_free': [1, 0, 1],
            'health_label': [1, 1, 0],
            'price': [-0.5, -0.3, -0.8],
            'choice': [1, 0, 1]
        })
        
        self.lv = 0.6
        
        self.params = {
            'intercept': 0.1,
            'beta': np.array([0.5, 0.3, -0.2]),
            'lambda': 1.0
        }
        
        self.model = BinaryProbitChoice(self.config)
        self.grad_calc = ChoiceGradient(n_attributes=3)
    
    def test_gradient_vs_numerical(self):
        """해석적 그래디언트 vs 수치적 그래디언트"""
        # 해석적 그래디언트
        analytic_grad = self.grad_calc.compute_gradient(
            self.data, self.lv, self.params, self.config.choice_attributes
        )
        
        # 로그우도 함수
        def log_likelihood(params_flat):
            params = {
                'intercept': params_flat[0],
                'beta': params_flat[1:4],
                'lambda': params_flat[4]
            }
            return self.model.log_likelihood(self.data, self.lv, params)
        
        # 파라미터를 1D 배열로 변환
        params_flat = np.concatenate([
            [self.params['intercept']],
            self.params['beta'],
            [self.params['lambda']]
        ])
        
        # 수치적 그래디언트
        epsilons = [1e-4, 1e-5, 1e-6]
        
        print("\n=== 선택모델 그래디언트 비교 ===")
        print(f"해석적 grad_intercept: {analytic_grad['grad_intercept']}")
        print(f"해석적 grad_beta: {analytic_grad['grad_beta']}")
        print(f"해석적 grad_lambda: {analytic_grad['grad_lambda']}")
        
        for eps in epsilons:
            numerical_grad = approx_fprime(params_flat, log_likelihood, epsilon=eps)
            
            print(f"\nEpsilon = {eps}")
            print(f"수치적 그래디언트: {numerical_grad}")
            
            # intercept
            diff_intercept = np.abs(analytic_grad['grad_intercept'] - numerical_grad[0])
            rel_error_intercept = diff_intercept / (np.abs(analytic_grad['grad_intercept']) + 1e-10)
            print(f"Intercept 상대 오차: {rel_error_intercept}")
            
            # beta
            diff_beta = np.abs(analytic_grad['grad_beta'] - numerical_grad[1:4])
            rel_error_beta = diff_beta / (np.abs(analytic_grad['grad_beta']) + 1e-10)
            print(f"Beta 상대 오차: {rel_error_beta}")
            
            # lambda
            diff_lambda = np.abs(analytic_grad['grad_lambda'] - numerical_grad[4])
            rel_error_lambda = diff_lambda / (np.abs(analytic_grad['grad_lambda']) + 1e-10)
            print(f"Lambda 상대 오차: {rel_error_lambda}")
            
            # 상대 오차가 1% 이내인지 확인
            assert rel_error_intercept < 0.01, f"Epsilon {eps}: Intercept 상대 오차가 너무 큼"
            assert np.all(rel_error_beta < 0.01), f"Epsilon {eps}: Beta 상대 오차가 너무 큼"
            assert rel_error_lambda < 0.01, f"Epsilon {eps}: Lambda 상대 오차가 너무 큼"


class TestGradientNorm:
    """그래디언트 크기 테스트"""
    
    def test_gradient_magnitude(self):
        """그래디언트가 너무 작거나 크지 않은지 확인"""
        # 측정모델
        config = MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7'],
            n_categories=5
        )
        
        data = pd.DataFrame({'q6': [3], 'q7': [4]})
        lv = 0.5
        params = {
            'zeta': np.array([1.0, 1.0]),
            'tau': np.array([[-2.0, -1.0, 1.0, 2.0], [-2.0, -1.0, 1.0, 2.0]])
        }
        
        grad_calc = MeasurementGradient(n_indicators=2, n_categories=5)
        grad = grad_calc.compute_gradient(data, lv, params, config.indicators)
        
        grad_norm = np.linalg.norm(grad['grad_zeta'])
        
        print(f"\n=== 그래디언트 크기 ===")
        print(f"grad_zeta: {grad['grad_zeta']}")
        print(f"grad_zeta norm: {grad_norm}")
        
        # 그래디언트가 너무 작지 않은지 확인 (수렴 문제)
        assert grad_norm > 1e-6, "그래디언트가 너무 작음 (수렴 어려움)"
        
        # 그래디언트가 너무 크지 않은지 확인 (불안정)
        assert grad_norm < 1e6, "그래디언트가 너무 큼 (불안정)"


if __name__ == '__main__':
    # 개별 테스트 실행
    print("=" * 70)
    print("그래디언트 계산 유닛 테스트")
    print("=" * 70)
    
    # 측정모델 그래디언트
    test_meas = TestMeasurementGradient()
    test_meas.setup_method()
    test_meas.test_gradient_shape()
    print("✓ 측정모델 그래디언트 shape 테스트 통과")
    
    test_meas.test_gradient_vs_numerical()
    print("✓ 측정모델 그래디언트 정확성 테스트 통과")
    
    # 구조모델 그래디언트
    test_struct = TestStructuralGradient()
    test_struct.setup_method()
    test_struct.test_gradient_vs_numerical()
    print("✓ 구조모델 그래디언트 정확성 테스트 통과")
    
    # 선택모델 그래디언트
    test_choice = TestChoiceGradient()
    test_choice.setup_method()
    test_choice.test_gradient_vs_numerical()
    print("✓ 선택모델 그래디언트 정확성 테스트 통과")
    
    # 그래디언트 크기
    test_norm = TestGradientNorm()
    test_norm.test_gradient_magnitude()
    print("✓ 그래디언트 크기 테스트 통과")
    
    print("\n" + "=" * 70)
    print("모든 테스트 통과!")
    print("=" * 70)

