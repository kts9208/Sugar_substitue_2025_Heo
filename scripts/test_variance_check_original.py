"""
요인점수 분산 체크 기능 테스트 스크립트 (표준화 전 원본 요인점수)

표준화 전 요인점수의 분산을 체크하는 기능을 테스트합니다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator


def create_test_factor_scores():
    """
    테스트용 요인점수 생성 (원본, 표준화 전)
    
    - health_consciousness: 분산이 매우 작음 (0.003)
    - perceived_price: 분산이 작음 (0.006)
    - purchase_intention: 분산이 정상 (1.367)
    """
    np.random.seed(42)
    n_obs = 326
    
    # 1. 분산이 매우 작은 변수 (평균 주변에 밀집)
    health_consciousness = np.random.normal(5.5, 0.05, n_obs)  # std=0.05 → var≈0.0025
    
    # 2. 분산이 작은 변수
    perceived_price = np.random.normal(3.8, 0.08, n_obs)  # std=0.08 → var≈0.0064
    
    # 3. 분산이 정상인 변수
    purchase_intention = np.random.normal(4.0, 1.2, n_obs)  # std=1.2 → var≈1.44
    
    return {
        'health_consciousness': health_consciousness,
        'perceived_price': perceived_price,
        'purchase_intention': purchase_intention
    }


def main():
    print("=" * 100)
    print("요인점수 분산 체크 기능 테스트 (표준화 전 원본 요인점수)")
    print("=" * 100)
    
    # 1. 테스트 데이터 생성
    print("\n[1] 테스트 데이터 생성 중...")
    original_factor_scores = create_test_factor_scores()
    
    print("✅ 생성 완료")
    print(f"   - 변수 개수: {len(original_factor_scores)}")
    print(f"   - 관측 수: {len(original_factor_scores['health_consciousness'])}")
    
    # 2. 원본 요인점수 통계 출력
    print("\n[2] 원본 요인점수 통계 (표준화 전)")
    print("-" * 100)
    print(f"{'변수':30s} {'평균':>12s} {'분산':>12s} {'표준편차':>12s} {'최소값':>12s} {'최대값':>12s}")
    print("-" * 100)
    
    for lv_name, scores in original_factor_scores.items():
        mean = np.mean(scores)
        variance = np.var(scores, ddof=0)
        std = np.std(scores, ddof=0)
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        print(f"{lv_name:30s} {mean:>12.4f} {variance:>12.6f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")
    
    print("-" * 100)
    
    # 3. 분산 체크 (직접 구현)
    print("\n[3] 분산 체크 (표준화 전 원본 요인점수)")
    print("=" * 100)
    print(f"{'변수':30s} {'평균':>12s} {'분산':>12s} {'표준편차':>12s} {'최소값':>12s} {'최대값':>12s}")
    print('-' * 100)

    low_variance_vars = []
    variance_threshold = 0.01

    for lv_name, scores in original_factor_scores.items():
        mean = np.mean(scores)
        variance = np.var(scores, ddof=0)
        std = np.std(scores, ddof=0)
        min_val = np.min(scores)
        max_val = np.max(scores)

        if variance < variance_threshold:
            low_variance_vars.append((lv_name, variance))

        print(f'{lv_name:30s} {mean:>12.4f} {variance:>12.6f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f}')

    print('-' * 100)

    if low_variance_vars:
        print(f"\n⚠️  분산이 {variance_threshold} 미만인 변수: {len(low_variance_vars)}개")
        for var_name, var_value in low_variance_vars:
            print(f"   - {var_name}: 분산 = {var_value:.6f}")
        print("   → 선택모델에서 비유의할 가능성이 높습니다.")
        print("   → 측정모델 재검토 또는 지표 추가를 고려하세요.")
    else:
        print(f"\n✅ 모든 변수의 분산이 임계값({variance_threshold}) 이상입니다.")

    print("=" * 100)

    # 4. 표준화 수행
    print("\n[4] Z-score 표준화 수행")
    print("=" * 100)

    standardized_scores = {}
    for lv_name, scores in original_factor_scores.items():
        mean = np.mean(scores)
        std = np.std(scores, ddof=0)
        if std > 1e-10:
            standardized_scores[lv_name] = (scores - mean) / std
        else:
            standardized_scores[lv_name] = scores - mean
    
    # 5. 표준화 후 통계 출력
    print("\n[5] 표준화 후 통계")
    print("-" * 100)
    print(f"{'변수':30s} {'평균':>12s} {'분산':>12s} {'표준편차':>12s} {'최소값':>12s} {'최대값':>12s}")
    print("-" * 100)
    
    for lv_name, scores in standardized_scores.items():
        mean = np.mean(scores)
        variance = np.var(scores, ddof=0)
        std = np.std(scores, ddof=0)
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        print(f"{lv_name:30s} {mean:>12.6f} {variance:>12.6f} {std:>12.6f} {min_val:>12.4f} {max_val:>12.4f}")
    
    print("-" * 100)
    
    # 6. 저장 테스트
    print("\n[6] 저장 기능 테스트 (원본 요인점수 분산 저장)")
    print("=" * 100)
    
    # 테스트 결과 구성
    test_results = {
        'factor_scores': standardized_scores,  # 표준화된 요인점수
        'original_factor_scores': original_factor_scores,  # ✅ 원본 요인점수 (분산 체크용)
        'paths': pd.DataFrame(),  # 빈 DataFrame
        'loadings': pd.DataFrame(),  # 빈 DataFrame
        'fit_indices': {},
        'log_likelihood': -100.0
    }
    
    # 저장 경로
    save_path = project_root / "results" / "test_variance_check" / "test_stage1_results.pkl"
    
    # 저장 실행
    saved_path = SequentialEstimator.save_stage1_results(test_results, str(save_path))
    
    print(f"\n✅ 저장 완료: {saved_path}")
    
    print("\n" + "=" * 100)
    print("✅ 테스트 완료 - 다음 단계에서 저장된 파일을 확인하세요")
    print("=" * 100)


if __name__ == '__main__':
    main()

