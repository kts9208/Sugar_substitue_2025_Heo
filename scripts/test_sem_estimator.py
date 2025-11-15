"""
SEMEstimator 테스트 스크립트

기존 factor_analysis 모듈을 재사용한 SEM 추정기를 테스트합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-15
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.sem_estimator import SEMEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    MultiLatentConfig,
    MeasurementConfig,
    MultiLatentStructuralConfig
)

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_config():
    """테스트용 설정 생성"""
    
    # 측정모델 설정
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            measurement_method='continuous_linear',
            n_categories=5
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            measurement_method='continuous_linear',
            n_categories=5
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            measurement_method='continuous_linear',
            n_categories=5
        )
    }
    
    # 구조모델 설정 (계층적)
    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern'],
        covariates=[],
        hierarchical_paths=[
            {'target': 'perceived_benefit', 'predictors': ['health_concern']},
            {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
        ]
    )
    
    return measurement_configs, structural_config


def load_test_data():
    """테스트 데이터 로드"""
    logger.info("테스트 데이터 로드 중...")

    # 실제 데이터 파일 경로 (우선순위: data/processed/survey > processed_data/survey_data)
    data_paths = [
        project_root / "data" / "processed" / "survey",
        project_root / "processed_data" / "survey_data"
    ]

    data_dir = None
    for path in data_paths:
        if path.exists():
            data_dir = path
            logger.info(f"데이터 디렉토리 발견: {data_dir}")
            break

    if data_dir is None:
        raise FileNotFoundError("데이터 디렉토리를 찾을 수 없습니다.")

    # 각 요인별 데이터 로드
    dfs = []
    for factor in ['health_concern', 'perceived_benefit', 'purchase_intention']:
        file_path = data_dir / f"{factor}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'no' in df.columns:
                df = df.set_index('no')
            dfs.append(df)
            logger.info(f"  {factor} 로드 완료: {df.shape}")
        else:
            logger.warning(f"파일을 찾을 수 없습니다: {file_path}")

    # 데이터 병합
    if dfs:
        data = pd.concat(dfs, axis=1)
        # 중복 컬럼 제거
        data = data.loc[:, ~data.columns.duplicated()]
        logger.info(f"데이터 로드 완료: {data.shape}")
        return data
    else:
        raise FileNotFoundError("테스트 데이터를 찾을 수 없습니다.")


def test_sem_estimator():
    """SEMEstimator 테스트"""
    logger.info("=" * 70)
    logger.info("SEMEstimator 테스트 시작")
    logger.info("=" * 70)
    
    try:
        # 1. 설정 생성
        measurement_configs, structural_config = create_test_config()
        
        # 2. 모델 객체 생성
        measurement_model = MultiLatentMeasurement(measurement_configs)
        structural_model = MultiLatentStructural(structural_config)
        
        # 3. 데이터 로드
        data = load_test_data()
        
        # 4. SEMEstimator 생성 및 추정
        logger.info("\nSEMEstimator 생성...")
        sem_estimator = SEMEstimator()
        
        logger.info("\nSEM 추정 실행...")
        results = sem_estimator.fit(data, measurement_model, structural_model)

        # 5. 결과 출력
        logger.info("\n" + "=" * 70)
        logger.info("추정 결과")
        logger.info("=" * 70)

        logger.info(f"\n로그우도: {results['log_likelihood']:.4f}")

        logger.info("\n적합도 지수:")
        for index, value in results['fit_indices'].items():
            logger.info(f"  {index}: {value:.4f}")

        logger.info("\n요인점수:")
        for lv_name, scores in results['factor_scores'].items():
            logger.info(f"  {lv_name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

        logger.info(f"\n[측정모델 파라미터]")
        logger.info(f"  요인적재량: {len(results['loadings'])}개")
        if len(results['loadings']) > 0:
            logger.info(f"\n{results['loadings'][['lval', 'rval', 'Estimate']].to_string()}")

        logger.info(f"\n  측정 오차분산: {len(results['measurement_errors'])}개")
        if len(results['measurement_errors']) > 0:
            logger.info(f"\n{results['measurement_errors'][['lval', 'Estimate']].head(5).to_string()}")

        logger.info(f"\n[구조모델 파라미터]")
        logger.info(f"  경로계수: {len(results['paths'])}개")
        if len(results['paths']) > 0:
            logger.info(f"\n{results['paths'][['lval', 'rval', 'Estimate']].to_string()}")

        logger.info(f"\n  구조 오차분산: {len(results['structural_errors'])}개")
        if len(results['structural_errors']) > 0:
            logger.info(f"\n{results['structural_errors'][['lval', 'Estimate']].to_string()}")

        logger.info(f"\n  외생 LV 분산: {len(results['lv_variances'])}개")
        if len(results['lv_variances']) > 0:
            logger.info(f"\n{results['lv_variances'][['lval', 'Estimate']].to_string()}")

        logger.info("\n모델 요약:")
        summary = sem_estimator.get_model_summary(measurement_model, structural_model)
        logger.info(summary)
        
        logger.info("\n" + "=" * 70)
        logger.info("테스트 성공!")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_sem_estimator()
    sys.exit(0 if success else 1)

