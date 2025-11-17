"""
Multinomial Logit Gradient 검증 스크립트

scipy.optimize.check_grad를 이용하여 multinomial logit analytic gradient와
numerical gradient를 비교 검증합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-17
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy.optimize import check_grad, approx_fprime
from typing import Dict

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_gpu_batch_estimator import SimultaneousGPUBatchEstimator


class MNLGradientValidator:
    """
    Multinomial Logit Gradient 검증 클래스
    """

    def __init__(self, estimator, data, measurement_model, structural_model, choice_model):
        """
        Args:
            estimator: SimultaneousGPUBatchEstimator 인스턴스
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

        # 모델 저장
        self.estimator.measurement_model = measurement_model
        self.estimator.structural_model = structural_model
        self.estimator.choice_model = choice_model
        self.estimator.data = data
    
    def objective_function(self, params_flat: np.ndarray) -> float:
        """
        목적함수: -log-likelihood
        
        Args:
            params_flat: 1D 파라미터 벡터
        
        Returns:
            -log_likelihood (scalar)
        """
        ll = self.estimator._joint_log_likelihood(
            params_flat,
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )
        return -ll
    
    def gradient_function(self, params_flat: np.ndarray) -> np.ndarray:
        """
        Analytic gradient 함수
        
        Args:
            params_flat: 1D 파라미터 벡터
        
        Returns:
            gradient 벡터 (1D)
        """
        grad = self.estimator.gradient_function(params_flat)
        return -grad  # 최소화 문제이므로 부호 반전
    
    def validate(self, params_flat: np.ndarray, epsilon: float = 1e-7) -> Dict:
        """
        scipy.optimize.check_grad를 사용하여 gradient 검증
        
        Args:
            params_flat: 검증할 파라미터 값
            epsilon: numerical gradient 계산 시 사용할 step size
        
        Returns:
            검증 결과 딕셔너리
        """
        print("\n" + "="*80)
        print("Multinomial Logit Gradient 검증: Analytic vs Numerical")
        print("="*80)
        
        # check_grad 실행
        print(f"\n파라미터 개수: {len(params_flat)}")
        print(f"Epsilon (step size): {epsilon}")
        
        print("\n⏳ check_grad 실행 중... (시간이 걸릴 수 있습니다)")
        error = check_grad(
            self.objective_function,
            self.gradient_function,
            params_flat,
            epsilon=epsilon
        )
        
        print(f"\n✅ Gradient Error (L2 norm): {error:.10f}")
        
        # 상세 비교를 위해 개별 gradient 계산
        print("\n" + "-"*80)
        print("개별 파라미터 Gradient 비교")
        print("-"*80)
        
        print("\n⏳ Analytic gradient 계산 중...")
        analytic_grad = self.gradient_function(params_flat)
        
        print("⏳ Numerical gradient 계산 중...")
        numerical_grad = approx_fprime(
            params_flat,
            self.objective_function,
            epsilon=epsilon
        )
        
        # 파라미터 이름 가져오기
        param_names = self.estimator.param_manager.get_parameter_names(
            self.measurement_model,
            self.structural_model,
            self.choice_model
        )
        
        # 상위 10개 차이가 큰 파라미터 출력
        abs_diff = np.abs(analytic_grad - numerical_grad)
        top_indices = np.argsort(abs_diff)[-10:][::-1]
        
        print(f"\n{'Parameter':<50} {'Analytic':>15} {'Numerical':>15} {'Abs Diff':>15}")
        print("-"*95)
        
        for idx in top_indices:
            param_name = param_names[idx] if idx < len(param_names) else f"param_{idx}"
            print(f"{param_name:<50} {analytic_grad[idx]:>15.6f} {numerical_grad[idx]:>15.6f} {abs_diff[idx]:>15.6e}")
        
        # 통계 요약
        print("\n" + "-"*80)
        print("통계 요약")
        print("-"*80)
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
        print("\n" + "="*80)
        if error < 1e-5:
            print("✅ PASS: Analytic gradient가 정확합니다!")
        elif error < 1e-3:
            print("⚠️  WARNING: Analytic gradient에 작은 오차가 있습니다.")
        else:
            print("❌ FAIL: Analytic gradient에 큰 오차가 있습니다!")
        print("="*80)

        return {
            'error': error,
            'analytic_grad': analytic_grad,
            'numerical_grad': numerical_grad,
            'abs_diff': abs_diff,
            'param_names': param_names
        }


def main():
    """메인 실행 함수"""

    print("="*80)
    print("Multinomial Logit Gradient 검증")
    print("="*80)

    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = Path('data/processed/iclv/integrated_data.csv')
    data = pd.read_csv(data_path)

    # 매우 작은 샘플로 테스트 (빠른 검증)
    sample_ids = data['respondent_id'].unique()[:3]  # 3명만
    data = data[data['respondent_id'].isin(sample_ids)].copy()
    print(f"   샘플: {len(sample_ids)}명, {len(data)}개 관측치")

    # 2. 모델 설정 (test_gpu_batch_iclv.py와 동일)
    print("\n2. 모델 설정 중...")
    config = create_sugar_substitute_multi_lv_config(
        n_main_lvs=2,
        n_interactions=2,
        use_gpu=True,
        n_draws=50,  # 빠른 검증을 위해 50개로 감소
        use_halton=True
    )
    print(f"   Main LVs: {config.choice.main_lvs}")
    print(f"   Interactions: {config.choice.lv_attribute_interactions}")
    print(f"   Choice type: {config.choice.choice_type}")
    print(f"   N alternatives: {config.choice.n_alternatives}")

    # 3. Estimator 생성
    print("\n3. Estimator 생성 중...")
    estimator = SimultaneousGPUBatchEstimator(config)

    # 4. 모델 객체 생성
    print("\n4. 모델 객체 생성 중...")
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
    from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice

    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    choice_model = BinaryProbitChoice(config.choice)

    # 5. 초기 파라미터 설정
    print("\n5. 초기 파라미터 설정 중...")
    initial_params = estimator.param_manager.get_initial_values(
        measurement_model,
        structural_model,
        choice_model
    )
    print(f"   초기 파라미터 개수: {len(initial_params)}")

    # 파라미터 이름 확인
    param_names = estimator.param_manager.get_parameter_names(
        measurement_model,
        structural_model,
        choice_model
    )

    # 선택모델 파라미터만 출력
    print("\n   선택모델 파라미터:")
    choice_param_start = len(param_names) - 12  # 마지막 12개
    for i in range(choice_param_start, len(param_names)):
        print(f"     [{i}] {param_names[i]}: {initial_params[i]:.6f}")

    # 6. Gradient 검증
    print("\n6. Gradient 검증 시작...")
    validator = MNLGradientValidator(
        estimator,
        data,
        measurement_model,
        structural_model,
        choice_model
    )

    # 여러 epsilon 값으로 테스트
    epsilons = [1e-5, 1e-6, 1e-7]

    for eps in epsilons:
        print(f"\n{'='*80}")
        print(f"Epsilon = {eps:.2e} 테스트")
        print(f"{'='*80}")

        result = validator.validate(initial_params, epsilon=eps)

        # 결과 저장 (선택적)
        # np.savez(f'gradient_validation_eps_{eps:.0e}.npz', **result)

    print("\n검증 완료!")


if __name__ == '__main__':
    main()


