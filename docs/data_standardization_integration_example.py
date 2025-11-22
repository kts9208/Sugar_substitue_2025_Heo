"""
데이터 표준화 통합 예시 코드

이 파일은 DataStandardizer를 simultaneous_estimator_fixed.py에
통합하는 방법을 보여줍니다.

Author: Sugar Substitute Research Team
Date: 2025-01-22
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.analysis.hybrid_choice_model.iclv_models.data_standardizer import DataStandardizer
from src.analysis.hybrid_choice_model.iclv_models.parameter_scaler import ParameterScaler


# ============================================================================
# 예시 1: DataStandardizer 단독 사용
# ============================================================================

def example_1_basic_usage():
    """기본 사용법"""
    print("=" * 80)
    print("예시 1: DataStandardizer 기본 사용법")
    print("=" * 80)
    
    # 샘플 데이터 생성
    data = pd.DataFrame({
        'respondent_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'price': [2000, 4000, 6000, 2000, 4000, 6000, 2000, 4000, 6000],
        'health_label': [0, 1, 0, 1, 0, 1, 0, 1, 0],
        'chosen': [1, 0, 0, 0, 1, 0, 0, 0, 1]
    })
    
    print("\n[원본 데이터]")
    print(data.head())
    print(f"\nprice 통계: mean={data['price'].mean():.2f}, std={data['price'].std():.2f}")
    print(f"health_label 통계: mean={data['health_label'].mean():.2f}, std={data['health_label'].std():.2f}")
    
    # DataStandardizer 생성
    standardizer = DataStandardizer(
        variables_to_standardize=['price', 'health_label']
    )
    
    # Fit & Transform
    data_standardized = standardizer.fit_transform(data)
    
    print("\n[표준화된 데이터]")
    print(data_standardized.head())
    print(f"\nprice 통계: mean={data_standardized['price'].mean():.6f}, std={data_standardized['price'].std():.6f}")
    print(f"health_label 통계: mean={data_standardized['health_label'].mean():.6f}, std={data_standardized['health_label'].std():.6f}")
    
    # 표준화 파라미터 확인
    print("\n[표준화 파라미터]")
    params = standardizer.get_standardization_params()
    for var, stats in params.items():
        print(f"  {var}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    # 역변환
    data_inverse = standardizer.inverse_transform(data_standardized)
    print("\n[역변환 데이터 (원본과 동일해야 함)]")
    print(data_inverse.head())
    print(f"\nprice 통계: mean={data_inverse['price'].mean():.2f}, std={data_inverse['price'].std():.2f}")


# ============================================================================
# 예시 2: Estimator에 통합 (의사 코드)
# ============================================================================

def example_2_estimator_integration_pseudocode():
    """
    simultaneous_estimator_fixed.py의 estimate() 메서드에 통합하는 방법
    
    이것은 의사 코드입니다. 실제 코드는 simultaneous_estimator_fixed.py를 수정해야 합니다.
    """
    print("\n" + "=" * 80)
    print("예시 2: Estimator 통합 (의사 코드)")
    print("=" * 80)
    
    code = '''
def estimate(self, data, measurement_model, structural_model, choice_model):
    """ICLV 모델 동시 추정"""
    
    # ========================================================================
    # 1. 데이터 표준화 (새로 추가)
    # ========================================================================
    if self.config.estimation.standardize_choice_attributes:
        self.iteration_logger.info("=" * 80)
        self.iteration_logger.info("선택 속성 Z-score 표준화")
        self.iteration_logger.info("=" * 80)
        
        # DataStandardizer 생성
        self.data_standardizer = DataStandardizer(
            variables_to_standardize=self.config.choice.choice_attributes,
            logger=self.iteration_logger
        )
        
        # 원본 데이터 백업 (나중에 역변환용)
        data_original = data.copy()
        
        # Fit & Transform
        data = self.data_standardizer.fit_transform(data)
        
        # 비교 로깅
        self.data_standardizer.log_standardization_comparison(
            data_original, data
        )
        
        self.iteration_logger.info("✅ 선택 속성 z-score 표준화 완료")
        self.iteration_logger.info("=" * 80)
    else:
        self.data_standardizer = None
        self.iteration_logger.info("선택 속성 표준화 비활성화 (원본 데이터 사용)")
    
    # 표준화된 데이터 저장
    self.data = data
    
    # ========================================================================
    # 2. 파라미터 스케일링 (기존 유지)
    # ========================================================================
    if use_parameter_scaling:
        custom_scales = self._get_custom_scales(param_names)
        self.param_scaler = ParameterScaler(
            initial_params=initial_params,
            param_names=param_names,
            custom_scales=custom_scales,
            logger=self.iteration_logger
        )
        initial_params_scaled = self.param_scaler.scale_parameters(initial_params)
    else:
        self.param_scaler = None
        initial_params_scaled = initial_params
    
    # ========================================================================
    # 3. 최적화 (기존과 동일)
    # ========================================================================
    result = minimize(
        fun=negative_log_likelihood_func,
        x0=initial_params_scaled,
        jac=gradient_func,
        method='L-BFGS-B',
        ...
    )
    
    # ========================================================================
    # 4. 결과 처리 (기존과 동일)
    # ========================================================================
    # 파라미터 언스케일링
    if self.param_scaler is not None:
        final_params = self.param_scaler.unscale_parameters(result.x)
    else:
        final_params = result.x
    
    # 주의: 데이터는 역변환하지 않음 (표준화된 상태로 추정)
    # beta 파라미터는 표준화된 스케일로 해석
    
    return results
    '''
    
    print(code)


# ============================================================================
# 예시 3: 파라미터 해석
# ============================================================================

def example_3_parameter_interpretation():
    """표준화 후 파라미터 해석"""
    print("\n" + "=" * 80)
    print("예시 3: 표준화 후 파라미터 해석")
    print("=" * 80)
    
    # 표준화 파라미터
    price_mean = 4000.0
    price_std = 1000.0
    
    # 추정된 beta (표준화된 스케일)
    beta_price_standardized = -560.0
    
    print(f"\n[표준화 파라미터]")
    print(f"  price_mean = {price_mean:.2f}")
    print(f"  price_std = {price_std:.2f}")
    
    print(f"\n[추정된 파라미터 (표준화된 스케일)]")
    print(f"  beta_price = {beta_price_standardized:.2f}")
    
    print(f"\n[해석 1: 표준화된 스케일]")
    print(f"  가격이 1 표준편차 ({price_std:.0f}원) 증가하면")
    print(f"  효용이 {beta_price_standardized:.2f} 감소")
    
    # 원본 스케일로 변환
    beta_price_original = beta_price_standardized / price_std
    
    print(f"\n[원본 스케일로 변환]")
    print(f"  beta_price_original = beta_price_standardized / price_std")
    print(f"                      = {beta_price_standardized:.2f} / {price_std:.2f}")
    print(f"                      = {beta_price_original:.6f}")
    
    print(f"\n[해석 2: 원본 스케일]")
    print(f"  가격이 1원 증가하면")
    print(f"  효용이 {beta_price_original:.6f} 감소")
    
    print(f"\n[검증: 효용 계산]")
    price_original = 5000.0
    price_standardized = (price_original - price_mean) / price_std
    
    utility_original = beta_price_original * price_original
    utility_standardized = beta_price_standardized * price_standardized
    
    print(f"  원본 스케일: U = {beta_price_original:.6f} × {price_original:.0f} = {utility_original:.2f}")
    print(f"  표준화 스케일: U = {beta_price_standardized:.2f} × {price_standardized:.4f} = {utility_standardized:.2f}")
    print(f"  ✅ 두 값이 동일: {abs(utility_original - utility_standardized) < 1e-6}")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == '__main__':
    example_1_basic_usage()
    example_2_estimator_integration_pseudocode()
    example_3_parameter_interpretation()
    
    print("\n" + "=" * 80)
    print("모든 예시 완료")
    print("=" * 80)

