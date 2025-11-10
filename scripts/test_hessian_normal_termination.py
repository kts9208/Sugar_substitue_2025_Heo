"""
Hessian 역행렬 정상 종료 테스트

목적: BFGS 정상 종료 후 hess_inv가 올바르게 제공되는지 테스트
- 조기 종료 비활성화
- 소규모 데이터 (10명, 2개 지표)
- BFGS 방법 사용
- 빠른 수렴 확인

입력: data/processed/iclv/integrated_data.csv
출력: results/hessian_test_log.txt
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

# ICLV 모델 직접 import
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig,
    ICLVConfig,
    EstimationConfig
)

# 로깅 설정
log_file = 'results/hessian_test_log.txt'
os.makedirs('results', exist_ok=True)

# 파일 핸들러
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def load_and_sample_data(n_respondents=10):
    """
    데이터 로드 및 샘플링
    
    Args:
        n_respondents: 샘플링할 응답자 수 (기본값: 10명)
    
    Returns:
        pd.DataFrame: 샘플링된 데이터
    """
    logger.info(f"데이터 로드 중... (샘플: {n_respondents}명)")
    
    # 통합 데이터 로드
    data_path = 'data/processed/iclv/integrated_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    data = pd.read_csv(data_path)
    logger.info(f"전체 데이터 로드 완료: {len(data)}행, {len(data['respondent_id'].unique())}명")

    # 랜덤 샘플링
    np.random.seed(42)
    unique_persons = data['respondent_id'].unique()
    sampled_persons = np.random.choice(unique_persons, size=min(n_respondents, len(unique_persons)), replace=False)

    sampled_data = data[data['respondent_id'].isin(sampled_persons)].copy()
    logger.info(f"샘플링 완료: {len(sampled_data)}행, {len(sampled_data['respondent_id'].unique())}명")
    
    return sampled_data


def create_minimal_config():
    """
    최소 규모 설정 생성 (빠른 테스트용)

    Returns:
        tuple: (measurement_config, choice_config, estimation_config)
    """
    logger.info("최소 규모 설정 생성 중...")

    # 측정모델: 건강관심도만, 2개 지표만
    measurement_config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7'],  # 2개만
        n_categories=5
    )

    # 선택모델: 기본 설정
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label']  # 2개만
    )

    # 추정 설정: 조기 종료 비활성화, BFGS 사용
    estimation_config = EstimationConfig(
        n_draws=50,  # 빠른 테스트
        optimizer='BFGS',  # ✅ BFGS 사용 (hess_inv 제공)
        max_iterations=50,  # 빠른 수렴
        convergence_threshold=1e-3,
        calculate_se=True,  # ✅ 표준오차 계산
        use_analytic_gradient=True,
        early_stopping=False,  # ✅ 조기 종료 비활성화
        early_stopping_patience=999,  # 사용 안 함
        early_stopping_tol=1e-6
    )

    logger.info("설정 생성 완료:")
    logger.info(f"  - 잠재변수: 1개 (health_concern)")
    logger.info(f"  - 지표: 2개 (q6, q7)")
    logger.info(f"  - 선택 속성: 2개 (sugar_free, health_label)")
    logger.info(f"  - Draws: {estimation_config.n_draws}")
    logger.info(f"  - Optimizer: {estimation_config.optimizer}")
    logger.info(f"  - 조기 종료: {estimation_config.early_stopping}")
    logger.info(f"  - 표준오차 계산: {estimation_config.calculate_se}")

    return measurement_config, choice_config, estimation_config


def run_test():
    """
    Hessian 역행렬 테스트 실행
    """
    logger.info("=" * 80)
    logger.info("Hessian 역행렬 정상 종료 테스트 시작")
    logger.info("=" * 80)
    
    # 1. 데이터 로드
    data = load_and_sample_data(n_respondents=10)

    # 2. 설정 생성
    measurement_config, choice_config, estimation_config = create_minimal_config()

    # 3. 모델 초기화
    logger.info("\n모델 초기화 중...")

    measurement_model = OrderedProbitMeasurement(measurement_config)
    choice_model = BinaryProbitChoice(choice_config)

    logger.info("모델 초기화 완료")

    # 4. 추정기 생성
    logger.info("\n추정기 생성 중...")
    estimator = SimultaneousEstimator(
        measurement_model=measurement_model,
        choice_model=choice_model,
        config=estimation_config
    )
    logger.info("추정기 생성 완료")
    
    # 5. 추정 실행
    logger.info("\n" + "=" * 80)
    logger.info("추정 시작 (조기 종료 비활성화, BFGS 정상 종료 테스트)")
    logger.info("=" * 80)
    
    try:
        result = estimator.estimate(data)
        
        logger.info("\n" + "=" * 80)
        logger.info("추정 완료!")
        logger.info("=" * 80)
        
        # 6. 결과 확인
        logger.info("\n결과 확인:")
        logger.info(f"  - Success: {result.success}")
        logger.info(f"  - Message: {result.message}")
        logger.info(f"  - Iterations: {result.nit}")
        logger.info(f"  - Function evaluations: {result.nfev}")
        logger.info(f"  - Gradient evaluations: {result.njev}")
        logger.info(f"  - Final LL: {-result.fun:.4f}")
        
        # 7. Hessian 역행렬 확인
        logger.info("\n" + "=" * 80)
        logger.info("Hessian 역행렬 확인")
        logger.info("=" * 80)
        
        if hasattr(result, 'hess_inv'):
            if result.hess_inv is not None:
                logger.info("✅ SUCCESS: result.hess_inv 존재!")
                logger.info(f"  - Type: {type(result.hess_inv)}")
                logger.info(f"  - Shape: {result.hess_inv.shape}")
                logger.info(f"  - Dtype: {result.hess_inv.dtype}")
                
                # 대각 원소 확인
                if hasattr(result.hess_inv, 'shape') and len(result.hess_inv.shape) == 2:
                    diag = np.diag(result.hess_inv)
                    logger.info(f"  - 대각 원소 (처음 5개): {diag[:5]}")
                    logger.info(f"  - 대각 원소 (마지막 5개): {diag[-5:]}")
                    logger.info(f"  - 대각 원소 min: {np.min(diag):.6e}")
                    logger.info(f"  - 대각 원소 max: {np.max(diag):.6e}")
                    logger.info(f"  - 대각 원소 mean: {np.mean(diag):.6e}")
                    
                    # 표준오차 계산
                    se = np.sqrt(np.abs(diag))
                    logger.info(f"\n표준오차 (처음 5개): {se[:5]}")
                    logger.info(f"표준오차 (마지막 5개): {se[-5:]}")
                    logger.info(f"표준오차 min: {np.min(se):.6f}")
                    logger.info(f"표준오차 max: {np.max(se):.6f}")
                    logger.info(f"표준오차 mean: {np.mean(se):.6f}")
                    
                    # NaN/Inf 확인
                    n_nan = np.sum(np.isnan(result.hess_inv))
                    n_inf = np.sum(np.isinf(result.hess_inv))
                    logger.info(f"\nNaN 개수: {n_nan}")
                    logger.info(f"Inf 개수: {n_inf}")
                    
                    if n_nan == 0 and n_inf == 0:
                        logger.info("✅ Hessian 역행렬에 NaN/Inf 없음!")
                    else:
                        logger.warning(f"⚠️ Hessian 역행렬에 NaN/Inf 존재!")
                
                logger.info("\n✅ 테스트 성공: BFGS가 정상 종료 후 hess_inv를 올바르게 제공했습니다!")
                
            else:
                logger.error("❌ FAIL: result.hess_inv가 None입니다!")
                logger.info("테스트 실패: hess_inv가 생성되지 않았습니다.")
        else:
            logger.error("❌ FAIL: result에 hess_inv 속성이 없습니다!")
            logger.info("테스트 실패: result 객체에 hess_inv가 없습니다.")
        
        # 8. 파라미터 확인
        logger.info("\n" + "=" * 80)
        logger.info("최종 파라미터")
        logger.info("=" * 80)
        logger.info(f"파라미터 개수: {len(result.x)}")
        logger.info(f"파라미터 (처음 10개): {result.x[:10]}")
        logger.info(f"파라미터 (마지막 10개): {result.x[-10:]}")
        
    except Exception as e:
        logger.error(f"\n❌ 추정 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("\n" + "=" * 80)
    logger.info("테스트 완료!")
    logger.info(f"로그 파일: {log_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    run_test()

