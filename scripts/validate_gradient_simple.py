"""
Analytic Gradient 검증 스크립트 (리팩토링 버전)

리팩토링된 _compute_gradient() 메서드를 scipy.optimize.check_grad로 검증합니다.

검증 대상: SimultaneousEstimator._compute_gradient() 메서드
- 이 메서드는 estimate() 내부의 gradient_function()이 사용하는 순수한 gradient 계산 로직입니다.
- 상태 의존성이 제거되어 단위테스트가 가능합니다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy.optimize import check_grad, approx_fprime

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice


class SimpleGradientValidator:
    """
    리팩토링된 _compute_gradient() 메서드를 검증하는 검증기

    scipy.optimize.check_grad를 사용하여 analytic gradient와 numerical gradient를 비교합니다.
    """

    def __init__(self, estimator, data, measurement_model, structural_model, choice_model):
        self.estimator = estimator
        self.data = data
        self.measurement_model = measurement_model
        self.structural_model = structural_model
        self.choice_model = choice_model
        
    def objective_function(self, params_flat):
        """Negative log-likelihood"""
        ll = self.estimator._joint_log_likelihood(
            params_flat,
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )
        return -ll
    
    def gradient_function(self, params_flat):
        """
        Analytic gradient 계산

        ✅ 리팩토링된 _compute_gradient() 메서드를 직접 호출
        이 메서드는 SimultaneousEstimator.estimate() 내부의 gradient_function()이 사용하는
        순수한 gradient 계산 로직입니다.
        """
        # ✅ 리팩토링된 메서드 직접 호출
        return self.estimator._compute_gradient(
            params_flat,
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )
    
    def validate(self, params_flat, epsilon=1e-5):
        """Gradient 검증"""
        print(f"\n검증 중... (epsilon={epsilon:.2e})")
        
        # 1. check_grad로 전체 오차 확인
        error = check_grad(
            self.objective_function,
            self.gradient_function,
            params_flat,
            epsilon=epsilon
        )
        
        print(f"  전체 Gradient Error (norm): {error:.6e}")
        
        # 2. 파라미터별 상세 비교
        print(f"\n  파라미터별 상세 비교:")
        analytic_grad = self.gradient_function(params_flat)
        numerical_grad = approx_fprime(
            params_flat,
            self.objective_function,
            epsilon=epsilon
        )
        
        # 차이 계산
        abs_diff = np.abs(analytic_grad - numerical_grad)
        rel_diff = abs_diff / (np.abs(numerical_grad) + 1e-10)
        
        # 가장 큰 오차 5개 출력
        worst_indices = np.argsort(rel_diff)[-5:][::-1]
        
        for idx in worst_indices:
            print(f"    Param[{idx:2d}]: Analytic={analytic_grad[idx]:10.6f}, "
                  f"Numerical={numerical_grad[idx]:10.6f}, "
                  f"RelDiff={rel_diff[idx]:.2e}")
        
        # 통과 기준: 상대 오차 1% 이내
        n_pass = np.sum(rel_diff < 0.01)
        n_total = len(params_flat)
        
        print(f"\n  통과: {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%)")
        print(f"  평균 상대 오차: {np.mean(rel_diff):.6e}")
        print(f"  최대 상대 오차: {np.max(rel_diff):.6e}")
        
        return error, rel_diff


def main():
    print("="*70)
    print("Analytic Gradient 검증 (리팩토링된 _compute_gradient 메서드)")
    print("="*70)

    # 1. 데이터 로드
    print("\n1. 데이터 로드...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data_cleaned.csv'
    data = pd.read_csv(data_path)

    # 빠른 검증을 위해 5명만 사용
    sample_ids = data['respondent_id'].unique()[:5]
    data = data[data['respondent_id'].isin(sample_ids)].copy()
    print(f"   샘플: {len(sample_ids)}명, {len(data)}개 관측치")
    
    # 2. ICLV 설정 (test_gpu_batch_iclv.py와 동일)
    print("\n2. ICLV 설정...")

    # 측정모델 설정 (5개 잠재변수) - test_gpu_batch_iclv.py와 동일
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            n_categories=5,
            measurement_method='continuous_linear'
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            n_categories=5,
            measurement_method='continuous_linear'
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],
            n_categories=5,
            measurement_method='continuous_linear'
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=[f'q{i}' for i in range(30, 50)],
            n_categories=5,
            measurement_method='continuous_linear'
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            n_categories=5,
            measurement_method='continuous_linear'
        )
    }

    # 구조모델 설정 (병렬 구조로 변경 - CPU gradient가 계층적 구조를 아직 지원하지 않음)
    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
        covariates=[],
        hierarchical_paths=None,  # ✅ 병렬 구조 사용
        error_variance=1.0
    )

    # 선택모델 설정 (조절효과 비활성화 - 병렬 구조와 함께 사용)
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        moderation_enabled=False  # ✅ 조절효과 비활성화
    )

    # 추정 설정 (CPU 모드, analytic gradient)
    estimation_config = EstimationConfig(
        optimizer='BFGS',
        use_analytic_gradient=True,
        n_draws=50,  # 빠른 검증을 위해 적은 draws
        draw_type='halton',
        max_iterations=0,  # 검증만 하므로 최적화 안 함
        gradient_log_level='MINIMAL'
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
    
    # 3. 모델 생성 (test_gpu_batch_iclv.py와 동일)
    print("\n3. 모델 생성...")
    measurement_model = MultiLatentMeasurement(measurement_configs)
    structural_model = MultiLatentStructural(structural_config)
    choice_model = MultinomialLogitChoice(choice_config)
    print("   - 측정모델, 구조모델, 선택모델 생성 완료")

    # 4. GPUBatchEstimator 생성 (CPU 모드로 사용)
    print("\n4. GPUBatchEstimator 생성 (CPU 모드)...")
    estimator = GPUBatchEstimator(
        config,
        use_gpu=False  # ✅ CPU 모드로 설정
    )
    estimator.data = data

    # Halton draws 생성
    from src.analysis.hybrid_choice_model.iclv_models.gpu_batch_estimator import MultiDimensionalHaltonDrawGenerator
    n_individuals = len(sample_ids)
    n_first_order = len(structural_model.exogenous_lvs)
    n_higher_order = len(structural_model.get_higher_order_lvs())
    n_dimensions = n_first_order + n_higher_order

    print(f"   - Halton draws 생성 중... (n_individuals={n_individuals}, n_dimensions={n_dimensions})")

    estimator.halton_generator = MultiDimensionalHaltonDrawGenerator(
        n_draws=config.estimation.n_draws,
        n_individuals=n_individuals,
        n_dimensions=n_dimensions,
        scramble=config.estimation.scramble_halton
    )

    print("   - GPUBatchEstimator 생성 완료 (CPU 모드)")

    # Gradient 계산기 초기화를 위해 estimate() 시작 부분만 실행
    # (estimate() 메서드 내부에서 _initialize_gradient_calculators 호출됨)
    print("\n5. Gradient 계산기 초기화...")

    # estimate() 메서드의 초기화 부분만 실행하기 위해
    # 직접 초기화 로직을 복사
    estimator.data = data

    # Memory monitor 초기화
    from src.analysis.hybrid_choice_model.iclv_models.memory_monitor import MemoryMonitor
    estimator.memory_monitor = MemoryMonitor(
        cpu_threshold_mb=2000,
        gpu_threshold_mb=5000
    )

    # Iteration logger 초기화
    import logging
    from pathlib import Path
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    log_file = results_dir / 'gradient_validation_log.txt'

    estimator.iteration_logger = logging.getLogger('iclv_iteration')
    estimator.iteration_logger.setLevel(logging.INFO)
    estimator.iteration_logger.handlers = []
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    estimator.iteration_logger.addHandler(file_handler)

    # Gradient 계산기 초기화 (SimultaneousEstimator.estimate() 내부 로직)
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_gradient import (
        MultiLatentMeasurementGradient,
        MultiLatentStructuralGradient,
        MultiLatentJointGradient
    )
    from src.analysis.hybrid_choice_model.iclv_models.gradient_calculator import ChoiceGradient

    estimator.use_analytic_gradient = config.estimation.use_analytic_gradient

    if estimator.use_analytic_gradient:
        estimator.measurement_grad = MultiLatentMeasurementGradient(measurement_configs)

        # MultiLatentStructuralGradient 초기화
        n_exo = len(structural_model.exogenous_lvs)
        n_cov = len(structural_config.covariates) if structural_config.covariates else 0
        estimator.structural_grad = MultiLatentStructuralGradient(
            n_exo=n_exo,
            n_cov=n_cov,
            error_variance=structural_config.error_variance
        )

        estimator.choice_grad = ChoiceGradient(len(choice_config.choice_attributes))

        # GPU 사용 여부 확인
        use_gpu_gradient = False
        gpu_measurement_model = None

        if hasattr(estimator, 'use_gpu') and estimator.use_gpu:
            if hasattr(estimator, 'gpu_measurement_model') and estimator.gpu_measurement_model is not None:
                use_gpu_gradient = True
                gpu_measurement_model = estimator.gpu_measurement_model

        estimator.joint_grad = MultiLatentJointGradient(
            estimator.measurement_grad,
            estimator.structural_grad,
            estimator.choice_grad,
            use_gpu=use_gpu_gradient,
            gpu_measurement_model=gpu_measurement_model
        )
        estimator.joint_grad.iteration_logger = estimator.iteration_logger
        estimator.joint_grad.config = estimator.config

    print("   - Gradient 계산기 초기화 완료")
    
    # 6. 초기 파라미터
    print("\n6. 초기 파라미터 설정...")
    params_flat = estimator._get_initial_parameters(
        measurement_model, structural_model, choice_model
    )
    print(f"   파라미터 수: {len(params_flat)}")
    
    # 7. 검증
    print("\n7. Gradient 검증")
    print("="*70)
    
    validator = SimpleGradientValidator(
        estimator, data, measurement_model, structural_model, choice_model
    )
    
    # 여러 epsilon 값으로 테스트
    for eps in [1e-5, 1e-6]:
        error, rel_diff = validator.validate(params_flat, epsilon=eps)
        
        if error < 1e-3:
            print(f"\n  ✓ 검증 통과! (epsilon={eps:.2e})")
        else:
            print(f"\n  ✗ 검증 실패 (epsilon={eps:.2e})")
    
    print("\n" + "="*70)
    print("검증 완료!")
    print("="*70)


if __name__ == '__main__':
    main()

