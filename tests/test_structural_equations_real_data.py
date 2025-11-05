"""
Structural Equations Real Data Test

역코딩된 실제 데이터로 구조모델 테스트

Author: Sugar Substitute Research Team
Date: 2025-11-04
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 직접 import (모듈 경로 사용)
import importlib.util

# iclv_config 직접 로드
config_path = project_root / "src/analysis/hybrid_choice_model/iclv_models/iclv_config.py"
spec = importlib.util.spec_from_file_location("iclv_config", config_path)
iclv_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iclv_config)
StructuralConfig = iclv_config.StructuralConfig

# structural_equations 직접 로드
struct_path = project_root / "src/analysis/hybrid_choice_model/iclv_models/structural_equations.py"
spec = importlib.util.spec_from_file_location("structural_equations", struct_path)
structural_equations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(structural_equations)
LatentVariableRegression = structural_equations.LatentVariableRegression
estimate_structural_model = structural_equations.estimate_structural_model

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_sociodemographics(n_obs: int = 300) -> pd.DataFrame:
    """
    합성 사회인구학적 변수 생성
    
    실제 데이터에 사회인구학적 변수가 없으므로 합성 데이터 생성
    
    Args:
        n_obs: 관측치 수
    
    Returns:
        사회인구학적 변수 데이터프레임
    """
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.normal(40, 15, n_obs),  # 평균 40세, 표준편차 15
        'gender': np.random.binomial(1, 0.5, n_obs),  # 0: 남성, 1: 여성
        'income': np.random.normal(5, 2, n_obs),  # 평균 500만원 (단위: 100만원)
        'education': np.random.choice([1, 2, 3, 4], n_obs)  # 1: 고졸, 2: 전문대, 3: 대졸, 4: 대학원
    })
    
    # 표준화
    data['age_std'] = (data['age'] - data['age'].mean()) / data['age'].std()
    data['income_std'] = (data['income'] - data['income'].mean()) / data['income'].std()
    
    return data


def create_synthetic_latent_variable(sociodem_data: pd.DataFrame,
                                     true_gamma: np.ndarray,
                                     error_std: float = 1.0) -> np.ndarray:
    """
    합성 잠재변수 생성
    
    LV = γ*X + ε
    
    Args:
        sociodem_data: 사회인구학적 변수
        true_gamma: 실제 회귀계수
        error_std: 오차 표준편차
    
    Returns:
        잠재변수 값
    """
    X = sociodem_data[['age_std', 'gender', 'income_std']].values
    lv_mean = X @ true_gamma
    lv = lv_mean + np.random.normal(0, error_std, len(sociodem_data))
    
    return lv


def test_basic_functionality():
    """기본 기능 테스트"""
    logger.info("=" * 80)
    logger.info("테스트 1: 기본 기능 테스트")
    logger.info("=" * 80)
    
    # 설정
    config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std'],
        error_variance=1.0,
        fix_error_variance=True
    )
    
    # 모델 생성
    model = LatentVariableRegression(config)
    
    # 합성 데이터
    sociodem_data = create_synthetic_sociodemographics(300)
    true_gamma = np.array([0.5, -0.3, 0.2])
    latent_var = create_synthetic_latent_variable(sociodem_data, true_gamma)
    
    # Sequential 추정
    results = model.fit(sociodem_data, latent_var)
    
    logger.info("\n추정 결과:")
    logger.info(f"  실제 γ: {true_gamma}")
    logger.info(f"  추정 γ: {results['gamma']}")
    logger.info(f"  차이: {results['gamma'] - true_gamma}")
    logger.info(f"  R²: {results['r_squared']:.4f}")
    logger.info(f"  잔차 표준편차: {results['sigma']:.4f}")
    
    # 검증
    assert results['r_squared'] > 0.1, "R²이 너무 낮습니다"
    assert np.allclose(results['gamma'], true_gamma, atol=0.2), "회귀계수 추정이 부정확합니다"
    
    logger.info("\n✅ 기본 기능 테스트 통과!")
    
    return results


def test_predict_method():
    """predict 메서드 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("테스트 2: predict 메서드 테스트")
    logger.info("=" * 80)
    
    # 설정
    config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std'],
        error_variance=1.0
    )
    
    model = LatentVariableRegression(config)
    
    # 데이터
    sociodem_data = create_synthetic_sociodemographics(100)
    params = {'gamma': np.array([0.5, -0.3, 0.2])}
    
    # 예측 (스칼라 draw)
    draw_scalar = 0.5
    lv_scalar = model.predict(sociodem_data, params, draw_scalar)
    
    logger.info(f"\n스칼라 draw 예측:")
    logger.info(f"  draw: {draw_scalar}")
    logger.info(f"  LV 평균: {lv_scalar.mean():.4f}")
    logger.info(f"  LV 표준편차: {lv_scalar.std():.4f}")
    
    # 예측 (배열 draw)
    draw_array = np.random.normal(0, 1, 100)
    lv_array = model.predict(sociodem_data, params, draw_array)
    
    logger.info(f"\n배열 draw 예측:")
    logger.info(f"  draw 평균: {draw_array.mean():.4f}")
    logger.info(f"  LV 평균: {lv_array.mean():.4f}")
    logger.info(f"  LV 표준편차: {lv_array.std():.4f}")
    
    # 검증
    assert len(lv_scalar) == 100, "예측 길이가 잘못되었습니다"
    assert len(lv_array) == 100, "예측 길이가 잘못되었습니다"
    
    logger.info("\n✅ predict 메서드 테스트 통과!")
    
    return lv_scalar, lv_array


def test_log_likelihood():
    """log_likelihood 메서드 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("테스트 3: log_likelihood 메서드 테스트")
    logger.info("=" * 80)
    
    # 설정
    config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std'],
        error_variance=1.0
    )
    
    model = LatentVariableRegression(config)
    
    # 데이터
    sociodem_data = create_synthetic_sociodemographics(100)
    true_gamma = np.array([0.5, -0.3, 0.2])
    latent_var = create_synthetic_latent_variable(sociodem_data, true_gamma, error_std=1.0)
    
    # 로그우도 계산
    params = {'gamma': true_gamma}
    ll = model.log_likelihood(sociodem_data, latent_var, params, draw=0)
    
    logger.info(f"\n로그우도:")
    logger.info(f"  총 로그우도: {ll:.2f}")
    logger.info(f"  관측치당 평균: {ll / 100:.2f}")
    
    # 잘못된 파라미터로 로그우도 계산
    wrong_params = {'gamma': np.array([0.0, 0.0, 0.0])}
    ll_wrong = model.log_likelihood(sociodem_data, latent_var, wrong_params, draw=0)
    
    logger.info(f"\n잘못된 파라미터 로그우도:")
    logger.info(f"  총 로그우도: {ll_wrong:.2f}")
    logger.info(f"  관측치당 평균: {ll_wrong / 100:.2f}")
    
    # 검증
    assert ll > ll_wrong, "실제 파라미터의 로그우도가 더 높아야 합니다"
    
    logger.info("\n✅ log_likelihood 메서드 테스트 통과!")
    
    return ll, ll_wrong


def test_with_reversed_data():
    """역코딩된 실제 데이터로 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("테스트 4: 역코딩된 실제 데이터 테스트")
    logger.info("=" * 80)
    
    # 역코딩된 데이터 로드
    try:
        perceived_benefit = pd.read_csv("data/processed/survey/perceived_benefit_reversed.csv")
        logger.info(f"✅ 역코딩 데이터 로드 성공: {perceived_benefit.shape}")
    except FileNotFoundError:
        logger.warning("역코딩 데이터가 없습니다. 원본 데이터 사용")
        perceived_benefit = pd.read_csv("data/processed/survey/perceived_benefit.csv")
    
    # 잠재변수 생성 (지표 평균)
    indicator_cols = [col for col in perceived_benefit.columns if col.startswith('q')]
    latent_var = perceived_benefit[indicator_cols].mean(axis=1).values
    
    logger.info(f"\n잠재변수 통계:")
    logger.info(f"  평균: {latent_var.mean():.4f}")
    logger.info(f"  표준편차: {latent_var.std():.4f}")
    logger.info(f"  최소: {latent_var.min():.4f}")
    logger.info(f"  최대: {latent_var.max():.4f}")
    
    # 합성 사회인구학적 변수
    sociodem_data = create_synthetic_sociodemographics(len(perceived_benefit))
    
    # 구조모델 추정
    results = estimate_structural_model(
        sociodem_data,
        latent_var,
        sociodemographics=['age_std', 'gender', 'income_std']
    )
    
    logger.info(f"\n구조모델 추정 결과:")
    logger.info(f"  회귀계수 (γ):")
    for i, var in enumerate(['age_std', 'gender', 'income_std']):
        logger.info(f"    {var}: {results['gamma'][i]:.4f}")
    logger.info(f"  R²: {results['r_squared']:.4f}")
    logger.info(f"  잔차 표준편차: {results['sigma']:.4f}")
    
    logger.info("\n✅ 역코딩 데이터 테스트 통과!")
    
    return results


def test_all_factors():
    """5개 요인 모두 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("테스트 5: 5개 요인 구조모델 추정")
    logger.info("=" * 80)
    
    factors = {
        'health_concern': 'health_concern.csv',
        'perceived_benefit': 'perceived_benefit_reversed.csv',
        'purchase_intention': 'purchase_intention.csv',
        'perceived_price': 'perceived_price_reversed.csv',
        'nutrition_knowledge': 'nutrition_knowledge_reversed.csv'
    }
    
    results_all = {}
    
    for factor_name, filename in factors.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"요인: {factor_name}")
        logger.info(f"{'=' * 60}")
        
        # 데이터 로드
        try:
            data = pd.read_csv(f"data/processed/survey/{filename}")
        except FileNotFoundError:
            # 역코딩 파일이 없으면 원본 사용
            filename_original = filename.replace('_reversed', '')
            data = pd.read_csv(f"data/processed/survey/{filename_original}")
            logger.info(f"  (원본 데이터 사용: {filename_original})")
        
        # 잠재변수 생성
        indicator_cols = [col for col in data.columns if col.startswith('q')]
        latent_var = data[indicator_cols].mean(axis=1).values
        
        # 합성 사회인구학적 변수
        sociodem_data = create_synthetic_sociodemographics(len(data))
        
        # 구조모델 추정
        results = estimate_structural_model(
            sociodem_data,
            latent_var,
            sociodemographics=['age_std', 'gender', 'income_std']
        )
        
        results_all[factor_name] = results
        
        logger.info(f"\n  회귀계수:")
        logger.info(f"    age_std: {results['gamma'][0]:7.4f}")
        logger.info(f"    gender:  {results['gamma'][1]:7.4f}")
        logger.info(f"    income:  {results['gamma'][2]:7.4f}")
        logger.info(f"  R²: {results['r_squared']:.4f}")
        logger.info(f"  σ:  {results['sigma']:.4f}")
    
    # 요약 테이블
    logger.info("\n" + "=" * 80)
    logger.info("전체 요약")
    logger.info("=" * 80)
    logger.info(f"\n{'요인':<25} {'age_std':>10} {'gender':>10} {'income':>10} {'R²':>8} {'σ':>8}")
    logger.info("-" * 80)
    
    for factor_name, results in results_all.items():
        logger.info(
            f"{factor_name:<25} "
            f"{results['gamma'][0]:10.4f} "
            f"{results['gamma'][1]:10.4f} "
            f"{results['gamma'][2]:10.4f} "
            f"{results['r_squared']:8.4f} "
            f"{results['sigma']:8.4f}"
        )
    
    logger.info("\n✅ 5개 요인 테스트 통과!")
    
    return results_all


if __name__ == "__main__":
    logger.info("구조모델 실제 데이터 테스트 시작\n")
    
    # 테스트 실행
    test_basic_functionality()
    test_predict_method()
    test_log_likelihood()
    test_with_reversed_data()
    test_all_factors()
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 모든 테스트 통과!")
    logger.info("=" * 80)

