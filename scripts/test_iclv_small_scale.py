"""
ICLV 소규모 테스트 실행 스크립트

목적: 동시추정 수렴 확인 (소규모 테스트)
- n_draws = 100 (빠른 테스트)
- 1개 잠재변수만 사용 (건강관심도)
- 소수 응답자 샘플링

입력: data/processed/iclv/integrated_data.csv
출력: 수렴 확인 및 초기 결과
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

# ICLV 모델 직접 import (hybrid_choice_model __init__ 우회)
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator import SimultaneousEstimator
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig,
    ICLVConfig,
    EstimationConfig
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_sample_data(n_respondents=30):
    """
    데이터 로드 및 샘플링
    
    Args:
        n_respondents: 샘플링할 응답자 수 (기본값: 30명)
    
    Returns:
        pd.DataFrame: 샘플링된 데이터
    """
    print("\n[1] 데이터 로드 및 샘플링...")
    
    # 전체 데이터 로드
    df = pd.read_csv('data/processed/iclv/integrated_data.csv')
    print(f"   - 전체 데이터: {len(df):,}행, {df['respondent_id'].nunique()}명")
    
    # 응답자 샘플링
    all_respondents = df['respondent_id'].unique()
    sampled_respondents = np.random.choice(all_respondents, size=n_respondents, replace=False)
    
    df_sample = df[df['respondent_id'].isin(sampled_respondents)].copy()
    
    # "구매안함" 대안 제외
    df_sample = df_sample[df_sample['alternative'] != 3].copy()
    
    print(f"   - 샘플 데이터: {len(df_sample):,}행, {df_sample['respondent_id'].nunique()}명")
    print(f"   - 선택 세트: {df_sample['choice_set'].nunique()}개")
    print(f"   - 대안: {df_sample['alternative'].nunique()}개")
    
    # 결측치 처리
    if df_sample['income_std'].isnull().sum() > 0:
        mean_income = df_sample['income_std'].mean()
        df_sample['income_std'] = df_sample['income_std'].fillna(mean_income)
        print(f"   - income_std 결측치 처리 완료")
    
    return df_sample


def create_simple_config():
    """
    간단한 ICLV 설정 생성 (1개 잠재변수)
    
    Returns:
        ICLVConfig: ICLV 설정
    """
    print("\n[2] ICLV 설정 생성 (소규모 테스트)...")
    
    # 측정모델 설정 (건강관심도만)
    measurement_config = MeasurementConfig(
        latent_variable='health_concern',
        indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        indicator_type='ordered',
        n_categories=5
    )
    print(f"   - 측정모델: {measurement_config.latent_variable} (6개 지표)")
    
    # 구조모델 설정
    structural_config = StructuralConfig(
        sociodemographics=['age_std', 'gender', 'income_std'],
        include_in_choice=True,
        error_variance=1.0,
        fix_error_variance=True
    )
    print(f"   - 구조모델: {structural_config.sociodemographics}")
    
    # 선택모델 설정
    choice_config = ChoiceConfig(
        choice_attributes=['price', 'health_label'],
        choice_type='binary',
        price_variable='price'
    )
    print(f"   - 선택모델: {choice_config.choice_attributes}")
    
    # 추정 설정 (소규모 테스트)
    estimation_config = EstimationConfig(
        method='simultaneous',
        n_draws=100,  # 빠른 테스트
        optimizer='BFGS',
        max_iterations=100,  # 제한된 반복
        scramble_halton=True
    )
    print(f"   - 추정 설정: {estimation_config.n_draws} draws, {estimation_config.max_iterations} iterations")
    
    # ICLV 통합 설정
    config = ICLVConfig(
        measurement=measurement_config,
        structural=structural_config,
        choice=choice_config,
        estimation=estimation_config,
        individual_id_column='respondent_id'
    )
    
    print("   ✓ 설정 생성 완료")
    
    return config


def run_test_estimation(data, config):
    """
    테스트 추정 실행
    
    Args:
        data: 샘플 데이터
        config: ICLV 설정
    
    Returns:
        dict: 추정 결과
    """
    print("\n[3] ICLV 동시추정 실행 (테스트)...")
    print("   ⚠ 소규모 테스트: 수렴 확인용")
    
    try:
        # 모델 생성
        print("\n   [3-1] 모델 초기화...")
        measurement_model = OrderedProbitMeasurement(config.measurement)
        structural_model = LatentVariableRegression(config.structural)
        choice_model = BinaryProbitChoice(config.choice)
        print("   ✓ 모델 초기화 완료")
        
        # 동시추정기 생성
        print("\n   [3-2] 동시추정기 초기화...")
        estimator = SimultaneousEstimator(config)
        print("   ✓ 동시추정기 초기화 완료")
        
        # 추정 실행
        print("\n   [3-3] 동시추정 실행...")
        print("   (시간이 소요될 수 있습니다...)")
        
        results = estimator.estimate(
            data,
            measurement_model,
            structural_model,
            choice_model
        )
        
        print("\n   ✓ 추정 완료!")
        
        return results
        
    except Exception as e:
        print(f"\n   ✗ 추정 실패: {e}")
        print("\n   상세 오류:")
        import traceback
        traceback.print_exc()
        return None


def display_test_results(results):
    """
    테스트 결과 출력
    
    Args:
        results: 추정 결과
    """
    if results is None:
        print("\n[4] 추정 결과 없음")
        return
    
    print("\n[4] 추정 결과:")
    print("=" * 80)
    
    # 수렴 상태
    if 'convergence' in results:
        conv = results['convergence']
        print(f"\n수렴 상태:")
        print(f"  - 성공 여부: {conv.get('success', 'Unknown')}")
        print(f"  - 메시지: {conv.get('message', 'N/A')}")
        print(f"  - 반복 횟수: {conv.get('nit', 'N/A')}")
        print(f"  - 함수 평가: {conv.get('nfev', 'N/A')}")
    
    # 모델 적합도
    if 'model_fit' in results:
        fit = results['model_fit']
        print(f"\n모델 적합도:")
        print(f"  - Log-Likelihood: {fit.get('log_likelihood', 'N/A'):.2f}")
        print(f"  - AIC: {fit.get('aic', 'N/A'):.2f}")
        print(f"  - BIC: {fit.get('bic', 'N/A'):.2f}")
    
    # 파라미터 추정치
    if 'parameters' in results:
        params = results['parameters']
        
        # 측정모델
        if 'measurement' in params:
            print(f"\n측정모델 파라미터:")
            meas = params['measurement']
            if 'zeta' in meas:
                print(f"  - 요인적재량 (zeta): {len(meas['zeta'])}개")
                for i, z in enumerate(meas['zeta'][:3]):  # 처음 3개만
                    print(f"    zeta[{i}] = {z:.3f}")
                if len(meas['zeta']) > 3:
                    print(f"    ... (총 {len(meas['zeta'])}개)")
        
        # 구조모델
        if 'structural' in params:
            print(f"\n구조모델 파라미터:")
            struct = params['structural']
            if 'gamma' in struct:
                print(f"  - 회귀계수 (gamma): {len(struct['gamma'])}개")
                for i, g in enumerate(struct['gamma']):
                    print(f"    gamma[{i}] = {g:.3f}")
        
        # 선택모델
        if 'choice' in params:
            print(f"\n선택모델 파라미터:")
            choice = params['choice']
            if 'beta' in choice:
                print(f"  - 속성계수 (beta): {len(choice['beta'])}개")
                for i, b in enumerate(choice['beta']):
                    print(f"    beta[{i}] = {b:.3f}")
            if 'lambda' in choice:
                print(f"  - 잠재변수계수 (lambda): {choice['lambda']:.3f}")
    
    print("\n" + "=" * 80)


def save_test_results(results):
    """
    테스트 결과 저장
    
    Args:
        results: 추정 결과
    """
    if results is None:
        return
    
    print("\n[5] 결과 저장...")
    
    output_dir = 'results/iclv/test'
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과를 JSON으로 저장
    import json
    
    # NumPy 배열을 리스트로 변환
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    output_path = os.path.join(output_dir, 'test_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"   - 결과 저장: {output_path}")
    print("   ✓ 저장 완료!")


def main():
    """메인 실행 함수"""
    
    print("=" * 80)
    print("ICLV 소규모 테스트 실행")
    print("목적: 동시추정 수렴 확인")
    print("=" * 80)
    
    # 1. 데이터 로드 및 샘플링
    data = load_and_sample_data(n_respondents=30)
    
    # 2. 설정 생성
    config = create_simple_config()
    
    # 3. 추정 실행
    results = run_test_estimation(data, config)
    
    # 4. 결과 출력
    display_test_results(results)
    
    # 5. 결과 저장
    save_test_results(results)
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # 랜덤 시드 설정 (재현성)
    np.random.seed(42)
    
    results = main()

