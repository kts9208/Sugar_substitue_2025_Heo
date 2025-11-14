"""
Analytic Gradient 검증 스크립트

scipy.optimize.check_grad를 사용하여 test_gpu_batch_iclv.py에서 사용하는
analytic gradient의 정확성을 검증합니다.

검증 대상:
- BFGS/BHHH 최적화에서 사용하는 gradient_function
- 다중 잠재변수 ICLV 모델의 analytic gradient
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import check_grad, approx_fprime
import logging

# 프로젝트 루트를 경로에 추가
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
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_gradient import MultiLatentJointGradient

# DataConfig 정의
from dataclasses import dataclass

@dataclass
class DataConfig:
    """데이터 설정"""
    individual_id: str = 'respondent_id'
    choice_id: str = 'choice_set'


class GradientValidator:
    """
    Analytic Gradient 검증 클래스
    
    test_gpu_batch_iclv.py에서 사용하는 실제 gradient 함수를 검증합니다.
    """
    
    def __init__(self, estimator, data, measurement_model, structural_model, choice_model):
        """
        Args:
            estimator: GPUBatchEstimator 인스턴스
            data: 전체 데이터
            measurement_model: 측정모델 인스턴스
            structural_model: 구조모델 인스턴스
            choice_model: 선택모델 인스턴스
        """
        self.estimator = estimator
        self.data = data
        self.measurement_model = measurement_model
        self.structural_model = structural_model
        self.choice_model = choice_model
        
        # Joint gradient 계산기 초기화
        from src.analysis.hybrid_choice_model.iclv_models.multi_latent_gradient import (
            MultiLatentMeasurementGradient,
            MultiLatentStructuralGradient,
            MultiLatentJointGradient
        )
        from src.analysis.hybrid_choice_model.iclv_models.gradient_calculator import ChoiceGradient

        measurement_grad = MultiLatentMeasurementGradient(estimator.config.measurement_configs)
        structural_grad = MultiLatentStructuralGradient(
            n_exo=estimator.config.structural.n_exo,
            n_cov=estimator.config.structural.n_cov,
            error_variance=estimator.config.structural.error_variance
        )
        choice_grad = ChoiceGradient(
            n_attributes=len(estimator.config.choice.choice_attributes)
        )

        # GPU 사용 여부 확인
        use_gpu_gradient = False
        gpu_measurement_model = None

        if hasattr(estimator, 'use_gpu') and estimator.use_gpu:
            if hasattr(estimator, 'gpu_measurement_model') and estimator.gpu_measurement_model is not None:
                use_gpu_gradient = True
                gpu_measurement_model = estimator.gpu_measurement_model
                print(f"[OK] GPU 배치 그래디언트 활성화")

        self.joint_grad = MultiLatentJointGradient(
            measurement_grad,
            structural_grad,
            choice_grad,
            use_gpu=use_gpu_gradient,
            gpu_measurement_model=gpu_measurement_model
        )

        # iteration_logger와 config 전달
        self.joint_grad.iteration_logger = estimator.iteration_logger
        self.joint_grad.config = estimator.config

        print(f"[OK] Gradient 계산기 초기화 완료")
    
    def objective_function(self, params_flat):
        """
        목적 함수: Negative Log-Likelihood

        Args:
            params_flat: 1D 파라미터 벡터

        Returns:
            -LL (스칼라)
        """
        # Log-likelihood 계산
        ll = self.estimator._joint_log_likelihood(
            params_flat,
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )

        return -ll  # 최소화 문제로 변환
    
    def gradient_function(self, params_flat):
        """
        Analytic Gradient 계산

        test_gpu_batch_iclv.py에서 사용하는 실제 gradient 함수와 동일한 로직

        Args:
            params_flat: 1D 파라미터 벡터

        Returns:
            gradient (1D array)
        """
        # 파라미터 언팩
        params_dict = self.estimator._unpack_parameters(
            params_flat,
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )

        # 개인별 그래디언트 계산 및 합산
        individual_ids = self.data[self.estimator.config.individual_id_column].unique()
        total_grad_dict = None

        for ind_id in individual_ids:
            ind_data = self.data[self.data[self.estimator.config.individual_id_column] == ind_id]
            ind_idx = np.where(individual_ids == ind_id)[0][0]
            ind_draws = self.estimator.halton_generator.get_draws()[ind_idx]

            # 개인별 gradient 계산
            ind_grad = self.joint_grad.compute_individual_gradient(
                ind_data=ind_data,
                ind_draws=ind_draws,
                params_dict=params_dict,
                measurement_model=self.measurement_model,
                structural_model=self.structural_model,
                choice_model=self.choice_model,
                ind_id=ind_id
            )

            # 그래디언트 합산
            if total_grad_dict is None:
                import copy
                total_grad_dict = copy.deepcopy(ind_grad)
            else:
                def add_gradients(total, ind):
                    for key in total:
                        if isinstance(total[key], dict):
                            add_gradients(total[key], ind[key])
                        elif isinstance(total[key], np.ndarray):
                            total[key] += ind[key]
                        else:
                            total[key] += ind[key]

                add_gradients(total_grad_dict, ind_grad)

        # 그래디언트 벡터로 변환
        grad_vector = self.estimator._pack_gradient(
            total_grad_dict,
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )

        return -grad_vector  # 최소화 문제로 변환
    
    def validate_with_check_grad(self, params_flat, epsilon=1.4901161193847656e-08):
        """
        scipy.optimize.check_grad를 사용한 검증
        
        Args:
            params_flat: 검증할 파라미터 벡터
            epsilon: 유한차분 step size (scipy 기본값)
            
        Returns:
            gradient_error: 두 gradient 간의 norm 차이
        """
        print(f"\n검증 중... (epsilon={epsilon:.2e})")
        
        error = check_grad(
            self.objective_function,
            self.gradient_function,
            params_flat,
            epsilon=epsilon
        )
        
        return error
    
    def validate_detailed(self, params_flat, epsilon=1e-5, max_params_to_show=20):
        """
        상세 검증: 파라미터별 비교
        
        Args:
            params_flat: 검증할 파라미터 벡터
            epsilon: 유한차분 step size
            max_params_to_show: 출력할 최대 파라미터 수
            
        Returns:
            comparison_df: 파라미터별 비교 결과 DataFrame
        """
        print(f"\n상세 검증 중... (epsilon={epsilon:.2e})")
        
        # Analytic gradient
        print("  - Analytic gradient 계산 중...")
        analytic_grad = self.gradient_function(params_flat)
        
        # Numerical gradient (유한차분법)
        print("  - Numerical gradient 계산 중...")
        numerical_grad = approx_fprime(
            params_flat,
            self.objective_function,
            epsilon=epsilon
        )
        
        # 파라미터 이름 가져오기
        param_names = self.estimator._get_parameter_names(
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )
        
        # 비교 결과 생성
        results = []
        for i, name in enumerate(param_names):
            abs_diff = np.abs(analytic_grad[i] - numerical_grad[i])
            rel_diff = abs_diff / (np.abs(analytic_grad[i]) + 1e-10)
            
            results.append({
                'Parameter': name,
                'Analytic_Grad': analytic_grad[i],
                'Numerical_Grad': numerical_grad[i],
                'Abs_Diff': abs_diff,
                'Rel_Diff': rel_diff,
                'Pass': rel_diff < 0.01  # 1% 이내
            })
        
        df = pd.DataFrame(results)
        
        # 요약 통계
        print(f"\n  [OK] 검증 완료:")
        print(f"    - 전체 파라미터 수: {len(param_names)}")
        print(f"    - 통과: {df['Pass'].sum()}개")
        print(f"    - 실패: {(~df['Pass']).sum()}개")
        print(f"    - 평균 상대 오차: {df['Rel_Diff'].mean():.6e}")
        print(f"    - 최대 상대 오차: {df['Rel_Diff'].max():.6e}")
        
        return df


def main():
    """메인 검증 함수"""
    
    print("="*70)
    print("Analytic Gradient 검증 - scipy.optimize.check_grad")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data_cleaned.csv'
    data = pd.read_csv(data_path)

    # 빠른 검증을 위해 10명만 사용
    sample_ids = data['respondent_id'].unique()[:10]
    data = data[data['respondent_id'].isin(sample_ids)].copy()

    print(f"   데이터 shape: {data.shape} (샘플: {len(sample_ids)}명)")
    
    # 2. 설정 (test_gpu_batch_iclv.py와 동일)
    print("\n2. ICLV 설정...")
    
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

    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
        covariates=[],
        hierarchical_paths=[
            {'target': 'perceived_benefit', 'predictors': ['health_concern']},
            {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
        ],
        error_variance=1.0
    )

    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price']
    )

    # ✅ Analytic gradient 활성화
    estimation_config = EstimationConfig(
        optimizer='BHHH',
        use_analytic_gradient=True,  # ✅ 검증 대상
        n_draws=100,
        draw_type='halton',
        max_iterations=1000,
        calculate_se=True,
        use_parallel=False,
        n_cores=None,
        early_stopping=False,
        gradient_log_level='MINIMAL',  # 로깅 최소화
        use_parameter_scaling=False
    )

    config = MultiLatentConfig(
        measurement_configs=measurement_configs,
        structural=structural_config,
        choice=choice_config,
        estimation=estimation_config,
        individual_id_column='respondent_id',
        choice_column='choice'
    )

    config.data = DataConfig(
        individual_id='respondent_id',
        choice_id='choice_set'
    )

    print("   설정 완료")

    # 3. 모델 생성
    print("\n3. 모델 생성...")

    try:
        measurement_model = MultiLatentMeasurement(measurement_configs)
        structural_model = MultiLatentStructural(structural_config)
        choice_model = MultinomialLogitChoice(choice_config)
        print("   - 측정모델, 구조모델, 선택모델 생성 완료")
    except Exception as e:
        print(f"   [ERROR] 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Estimator 생성
    print("\n4. GPUBatchEstimator 생성...")

    try:
        estimator = GPUBatchEstimator(
            config,
            use_gpu=True,
            memory_monitor_cpu_threshold_mb=2000,
            memory_monitor_gpu_threshold_mb=5000
        )

        # GPU 측정모델 생성 (estimate() 내부 로직 일부 실행)
        from src.analysis.hybrid_choice_model.iclv_models.gpu_measurement_equations import GPUMultiLatentMeasurement
        estimator.gpu_measurement_model = GPUMultiLatentMeasurement(
            config.measurement_configs,
            use_gpu=True
        )
        estimator.structural_model_ref = structural_model
        estimator.use_multi_dimensional_draws = True
        estimator.data = data

        # 다차원 Halton draws 생성
        n_individuals = data['respondent_id'].nunique()
        n_first_order = len(structural_model.exogenous_lvs)
        n_higher_order = len(structural_model.get_higher_order_lvs())
        n_dimensions = n_first_order + n_higher_order

        print(f"   - Halton draws 생성 중... (n_individuals={n_individuals}, n_dimensions={n_dimensions})")

        from src.analysis.hybrid_choice_model.iclv_models.gpu_batch_estimator import MultiDimensionalHaltonDrawGenerator
        estimator.halton_generator = MultiDimensionalHaltonDrawGenerator(
            n_draws=config.estimation.n_draws,
            n_individuals=n_individuals,
            n_dimensions=n_dimensions,
            scramble=config.estimation.scramble_halton
        )

        # Memory monitor 초기화 (estimate() 메서드에서 하는 것과 동일)
        from src.analysis.hybrid_choice_model.iclv_models.memory_monitor import MemoryMonitor
        estimator.memory_monitor = MemoryMonitor(
            cpu_threshold_mb=estimator.memory_monitor_cpu_threshold_mb,
            gpu_threshold_mb=estimator.memory_monitor_gpu_threshold_mb
        )

        # Iteration logger 초기화 (estimate() 메서드에서 하는 것과 동일)
        import logging
        estimator.iteration_logger = logging.getLogger('validation_logger')
        estimator.iteration_logger.setLevel(logging.WARNING)  # 최소 로깅

        print("   - GPUBatchEstimator 생성 완료")
    except Exception as e:
        print(f"   [ERROR] Estimator 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 초기 파라미터 설정
    print("\n5. 초기 파라미터 설정...")

    try:
        params_flat = estimator._get_initial_parameters(
            measurement_model, structural_model, choice_model
        )
        print(f"   - 파라미터 수: {len(params_flat)}")
    except Exception as e:
        print(f"   [ERROR] 초기 파라미터 설정 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Validator 생성
    print("\n6. GradientValidator 생성...")

    try:
        validator = GradientValidator(
            estimator, data, measurement_model, structural_model, choice_model
        )
    except Exception as e:
        print(f"   [ERROR] Validator 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. check_grad 검증
    print("\n" + "="*70)
    print("7. scipy.optimize.check_grad 검증")
    print("="*70)

    # 빠른 검증을 위해 epsilon 값을 줄임
    epsilons = [1e-5, 1e-6]
    check_grad_results = []

    for eps in epsilons:
        try:
            error = validator.validate_with_check_grad(params_flat, epsilon=eps)
            check_grad_results.append({
                'Epsilon': eps,
                'Gradient_Error': error,
                'Pass': error < 1e-3  # 0.1% 이내
            })
            print(f"  Epsilon = {eps:.2e}: Gradient Error = {error:.6e} {'✓' if error < 1e-3 else '✗'}")
        except Exception as e:
            print(f"  Epsilon = {eps:.2e}: 실패 - {e}")
            check_grad_results.append({
                'Epsilon': eps,
                'Gradient_Error': np.nan,
                'Pass': False
            })

    # 8. 상세 검증 (파라미터별)
    print("\n" + "="*70)
    print("8. 파라미터별 상세 검증")
    print("="*70)

    try:
        comparison_df = validator.validate_detailed(params_flat, epsilon=1e-5)

        # 실패한 파라미터만 출력
        failed = comparison_df[~comparison_df['Pass']]

        if len(failed) > 0:
            print(f"\n  ⚠️  검증 실패: {len(failed)}개 파라미터")
            print("\n  실패한 파라미터 (상위 20개):")
            print(failed.nlargest(20, 'Rel_Diff')[['Parameter', 'Analytic_Grad', 'Numerical_Grad', 'Rel_Diff']].to_string(index=False))
        else:
            print(f"\n  ✓ 모든 파라미터 검증 통과!")

        # 9. 결과 저장
        print("\n" + "="*70)
        print("9. 결과 저장")
        print("="*70)

        output_dir = project_root / 'results'
        output_dir.mkdir(exist_ok=True)

        # check_grad 결과 저장
        df_check_grad = pd.DataFrame(check_grad_results)
        check_grad_file = output_dir / 'gradient_validation_check_grad.csv'
        df_check_grad.to_csv(check_grad_file, index=False, encoding='utf-8-sig')
        print(f"  - check_grad 결과: {check_grad_file}")

        # 상세 비교 결과 저장
        comparison_file = output_dir / 'gradient_validation_detailed.csv'
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        print(f"  - 상세 비교 결과: {comparison_file}")

        # 요약 통계 저장
        summary_data = {
            'Metric': [
                'Total_Parameters',
                'Passed_Parameters',
                'Failed_Parameters',
                'Mean_Relative_Error',
                'Max_Relative_Error',
                'Min_Check_Grad_Error',
                'Best_Epsilon'
            ],
            'Value': [
                len(comparison_df),
                comparison_df['Pass'].sum(),
                (~comparison_df['Pass']).sum(),
                f"{comparison_df['Rel_Diff'].mean():.6e}",
                f"{comparison_df['Rel_Diff'].max():.6e}",
                f"{df_check_grad['Gradient_Error'].min():.6e}",
                f"{df_check_grad.loc[df_check_grad['Gradient_Error'].idxmin(), 'Epsilon']:.2e}"
            ]
        }

        df_summary = pd.DataFrame(summary_data)
        summary_file = output_dir / 'gradient_validation_summary.csv'
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"  - 요약 통계: {summary_file}")

        print("\n" + "="*70)
        print("검증 완료!")
        print("="*70)

        # 최종 결과 요약
        print(f"\n최종 결과:")
        print(f"  - 전체 파라미터: {len(comparison_df)}개")
        print(f"  - 통과: {comparison_df['Pass'].sum()}개 ({comparison_df['Pass'].sum()/len(comparison_df)*100:.1f}%)")
        print(f"  - 실패: {(~comparison_df['Pass']).sum()}개 ({(~comparison_df['Pass']).sum()/len(comparison_df)*100:.1f}%)")
        print(f"  - 평균 상대 오차: {comparison_df['Rel_Diff'].mean():.6e}")
        print(f"  - 최적 epsilon: {df_check_grad.loc[df_check_grad['Gradient_Error'].idxmin(), 'Epsilon']:.2e}")
        print(f"  - 최소 check_grad 오차: {df_check_grad['Gradient_Error'].min():.6e}")

    except Exception as e:
        print(f"   [ERROR] 상세 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()


