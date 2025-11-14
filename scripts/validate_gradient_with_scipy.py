"""
Scipy.optimize.check_grad를 이용한 Analytic Gradient 검증

compute_score_gradient() 함수로 계산한 analytic gradient와
scipy.optimize.check_grad()로 계산한 numerical gradient를 비교합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-14
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from typing import Dict

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    ChoiceConfig,
    EstimationConfig
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    MultiLatentStructuralConfig,
    MultiLatentConfig
)
from src.analysis.hybrid_choice_model.iclv_models.gpu_batch_estimator import GPUBatchEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice


class GradientValidator:
    """
    ICLV 모델의 analytic gradient를 scipy.optimize.check_grad로 검증하는 클래스
    """

    def __init__(self, estimator, data, measurement_model, structural_model, choice_model):
        """
        Args:
            estimator: GPUBatchEstimator 인스턴스
            data: 데이터프레임
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델
        """
        self.estimator = estimator
        self.data = data
        self.measurement_model = measurement_model
        self.structural_model = structural_model
        self.choice_model = choice_model

        # 모델 저장 (gradient 계산에 필요)
        self.estimator.measurement_model = measurement_model
        self.estimator.structural_model = structural_model
        self.estimator.choice_model = choice_model
        self.estimator.data = data

        # Joint gradient calculator 초기화
        from src.analysis.hybrid_choice_model.iclv_models.multi_latent_gradient import (
            MultiLatentJointGradient,
            MultiLatentMeasurementGradient,
            MultiLatentStructuralGradient
        )
        from src.analysis.hybrid_choice_model.iclv_models.gradient_calculator import ChoiceGradient

        # 각 모델의 gradient calculator 생성
        measurement_grad = MultiLatentMeasurementGradient(estimator.config.measurement_configs)
        structural_grad = MultiLatentStructuralGradient(
            n_exo=estimator.config.structural.n_exo,
            n_cov=estimator.config.structural.n_cov,
            error_variance=estimator.config.structural.error_variance
        )
        choice_grad = ChoiceGradient(n_attributes=len(estimator.config.choice.choice_attributes))

        self.estimator.joint_grad = MultiLatentJointGradient(
            measurement_grad,
            structural_grad,
            choice_grad
        )
    
    def objective_function(self, params_flat: np.ndarray) -> float:
        """
        목적함수: -log-likelihood

        scipy.optimize는 최소화를 수행하므로 -LL을 반환합니다.

        Args:
            params_flat: 1D 파라미터 벡터

        Returns:
            -log_likelihood (scalar)
        """
        # _joint_log_likelihood는 flat array를 받음
        ll = self.estimator._joint_log_likelihood(
            params_flat,
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )

        # -LL 반환 (최소화 문제로 변환)
        return -ll
    
    def gradient_function(self, params_flat: np.ndarray) -> np.ndarray:
        """
        Analytic gradient 함수: -∇log-likelihood

        Args:
            params_flat: 1D 파라미터 벡터

        Returns:
            -gradient (1D vector)
        """
        # _compute_gradient는 flat array를 받아서 flat array를 반환
        grad_flat = self.estimator._compute_gradient(
            params_flat,
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )

        # -gradient 반환 (최소화 문제)
        return -grad_flat
    
    def validate(self, params_flat: np.ndarray, epsilon: float = 1.4901161193847656e-08) -> Dict:
        """
        scipy.optimize.check_grad를 사용하여 gradient 검증
        
        Args:
            params_flat: 검증할 파라미터 값
            epsilon: numerical gradient 계산 시 사용할 step size
        
        Returns:
            검증 결과 딕셔너리
        """
        print("\n" + "="*70)
        print("Scipy.optimize.check_grad를 이용한 Gradient 검증")
        print("="*70)
        
        # check_grad 실행
        print(f"\n파라미터 개수: {len(params_flat)}")
        print(f"Epsilon (step size): {epsilon}")
        
        error = check_grad(
            self.objective_function,
            self.gradient_function,
            params_flat,
            epsilon=epsilon
        )
        
        print(f"\n✅ Gradient Error (L2 norm): {error:.10f}")
        
        # 상세 비교를 위해 개별 gradient 계산
        print("\n" + "-"*70)
        print("개별 파라미터 Gradient 비교")
        print("-"*70)
        
        # Analytic gradient
        analytic_grad = self.gradient_function(params_flat)
        
        # Numerical gradient (finite difference)
        numerical_grad = np.zeros_like(params_flat)
        f0 = self.objective_function(params_flat)
        
        for i in range(len(params_flat)):
            params_plus = params_flat.copy()
            params_plus[i] += epsilon
            f_plus = self.objective_function(params_plus)
            numerical_grad[i] = (f_plus - f0) / epsilon
        
        # 파라미터 이름 가져오기
        param_names = self.estimator._get_parameter_names(
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )
        
        # 상위 10개 차이가 큰 파라미터 출력
        abs_diff = np.abs(analytic_grad - numerical_grad)
        top_indices = np.argsort(abs_diff)[-10:][::-1]
        
        print(f"\n{'Parameter':<40} {'Analytic':>15} {'Numerical':>15} {'Abs Diff':>15}")
        print("-"*85)
        
        for idx in top_indices:
            param_name = param_names[idx] if idx < len(param_names) else f"param_{idx}"
            print(f"{param_name:<40} {analytic_grad[idx]:>15.6f} {numerical_grad[idx]:>15.6f} {abs_diff[idx]:>15.6e}")
        
        # 통계 요약
        print("\n" + "-"*70)
        print("통계 요약")
        print("-"*70)
        print(f"Max absolute difference: {np.max(abs_diff):.6e}")
        print(f"Mean absolute difference: {np.mean(abs_diff):.6e}")
        print(f"Median absolute difference: {np.median(abs_diff):.6e}")
        
        # 상대 오차 계산 (analytic gradient가 0이 아닌 경우만)
        nonzero_mask = np.abs(analytic_grad) > 1e-10
        if np.any(nonzero_mask):
            rel_error = abs_diff[nonzero_mask] / np.abs(analytic_grad[nonzero_mask])
            print(f"Max relative error (nonzero): {np.max(rel_error):.6e}")
            print(f"Mean relative error (nonzero): {np.mean(rel_error):.6e}")
        
        # 결과 판정
        print("\n" + "="*70)
        if error < 1e-5:
            print("✅ PASS: Analytic gradient가 정확합니다!")
        elif error < 1e-3:
            print("⚠️  WARNING: Analytic gradient에 작은 오차가 있습니다.")
        else:
            print("❌ FAIL: Analytic gradient에 큰 오차가 있습니다!")
        print("="*70)
        
        return {
            'error': error,
            'analytic_grad': analytic_grad,
            'numerical_grad': numerical_grad,
            'abs_diff': abs_diff,
            'param_names': param_names
        }


def main():
    """메인 실행 함수"""
    
    print("="*70)
    print("Gradient 검증: compute_score_gradient vs scipy.optimize.check_grad")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = Path('data/processed/iclv/integrated_data_cleaned.csv')
    data = pd.read_csv(data_path)

    # 매우 작은 샘플로 테스트 (빠른 검증)
    sample_ids = data['respondent_id'].unique()[:2]  # 2명만
    data = data[data['respondent_id'].isin(sample_ids)].copy()
    print(f"   샘플: {len(sample_ids)}명, {len(data)}개 관측치")
    
    # 2. 모델 설정 (test_gpu_batch_iclv.py와 동일)
    print("\n2. 모델 설정 중...")
    
    # 측정모델 설정 (5개 잠재변수)
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8'],
            n_categories=5,
            measurement_method='continuous_linear',
            fix_first_loading=True
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q9', 'q10', 'q11'],
            n_categories=5,
            measurement_method='continuous_linear',
            fix_first_loading=True
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q12', 'q13', 'q14'],
            n_categories=5,
            measurement_method='continuous_linear',
            fix_first_loading=True
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=['q15', 'q16', 'q17'],
            n_categories=5,
            measurement_method='continuous_linear',
            fix_first_loading=True
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q27', 'q28', 'q29'],
            n_categories=5,
            measurement_method='continuous_linear',
            fix_first_loading=True
        )
    }
    
    # 구조모델 설정 (병렬 - 간단한 테스트를 위해)
    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
        covariates=[],
        hierarchical_paths=[]  # 계층적 구조 제거
    )
    
    # 선택모델 설정 (조절효과)
    choice_config = ChoiceConfig(
        choice_attributes=['price', 'sugar_free'],
        moderation_enabled=True,
        main_lv='purchase_intention',
        moderator_lvs=['perceived_price', 'nutrition_knowledge']
    )
    
    # 추정 설정
    estimation_config = EstimationConfig(
        optimizer='BHHH',
        use_analytic_gradient=True,
        n_draws=10,  # 매우 작게 (빠른 테스트)
        draw_type='halton',
        max_iterations=1,  # gradient만 검증하므로 1회만
        calculate_se=False
    )
    
    # 통합 설정
    config = MultiLatentConfig(
        measurement_configs=measurement_configs,
        structural=structural_config,
        choice=choice_config,
        estimation=estimation_config,
        individual_id_column='respondent_id',
        choice_column='choice'
    )
    
    # 3. 모델 객체 생성
    print("\n3. 모델 객체 생성 중...")
    measurement_model = MultiLatentMeasurement(measurement_configs)
    structural_model = MultiLatentStructural(structural_config)
    choice_model = BinaryProbitChoice(choice_config)
    
    # 4. Estimator 생성
    print("\n4. Estimator 생성 중...")
    estimator = GPUBatchEstimator(config, use_gpu=False)  # CPU 모드로 검증

    # Halton generator 초기화 (다차원)
    from src.analysis.hybrid_choice_model.iclv_models.gpu_batch_estimator import MultiDimensionalHaltonDrawGenerator
    n_individuals = data['respondent_id'].nunique()
    n_dimensions = len(measurement_configs)  # 잠재변수 개수
    estimator.halton_generator = MultiDimensionalHaltonDrawGenerator(
        n_draws=config.estimation.n_draws,
        n_individuals=n_individuals,
        n_dimensions=n_dimensions,
        scramble=False
    )
    estimator.data = data

    # Memory monitor 초기화 (더미 객체)
    class DummyMemoryMonitor:
        def check_and_cleanup(self, *args, **kwargs):
            return None
    estimator.memory_monitor = DummyMemoryMonitor()

    # 5. 초기 파라미터 설정
    print("\n5. 초기 파라미터 설정 중...")
    initial_params = estimator._get_initial_parameters(
        measurement_model,
        structural_model,
        choice_model
    )
    print(f"   초기 파라미터 개수: {len(initial_params)}")
    
    # 6. Gradient 검증
    print("\n6. Gradient 검증 시작...")
    validator = GradientValidator(
        estimator,
        data,
        measurement_model,
        structural_model,
        choice_model
    )
    
    result = validator.validate(initial_params, epsilon=1e-7)
    
    print("\n검증 완료!")


if __name__ == '__main__':
    main()

