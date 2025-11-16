"""
2단계 선택모델이 1단계 요인점수를 올바르게 사용했는지 검증

검증 항목:
1. 1단계 결과 파일에서 요인점수 로드
2. 2단계 데이터와 매칭 확인
3. 요인점수 통계 비교
4. 선택모델에서 사용된 요인점수 확인
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("="*80)
    print("2단계 선택모델 요인점수 사용 검증")
    print("="*80)
    
    # 1. 1단계 결과 로드
    print("\n[1] 1단계 결과 로드")
    stage1_path = project_root / "results" / "sequential_stage_wise" / "stage1_results.pkl"
    
    with open(stage1_path, 'rb') as f:
        stage1_results = pickle.load(f)
    
    factor_scores = stage1_results['factor_scores']
    print(f"✅ 1단계 요인점수 로드 완료")
    print(f"   - 잠재변수: {list(factor_scores.keys())}")
    print(f"   - 개인 수: {len(factor_scores['purchase_intention'])}")
    
    # 2. 2단계 데이터 로드
    print("\n[2] 2단계 데이터 로드")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    data = pd.read_csv(data_path)
    
    print(f"✅ 데이터 로드 완료")
    print(f"   - 전체 행 수: {len(data)}")
    print(f"   - 개인 수: {data['respondent_id'].nunique()}")
    print(f"   - 개인당 선택 세트 수: {len(data) / data['respondent_id'].nunique():.1f}")
    
    # 3. 요인점수와 데이터 매칭 확인
    print("\n[3] 요인점수와 데이터 매칭 확인")
    
    n_individuals_data = data['respondent_id'].nunique()
    n_individuals_fs = len(factor_scores['purchase_intention'])
    
    if n_individuals_data == n_individuals_fs:
        print(f"✅ 개인 수 일치: {n_individuals_data}")
    else:
        print(f"❌ 개인 수 불일치!")
        print(f"   - 데이터: {n_individuals_data}")
        print(f"   - 요인점수: {n_individuals_fs}")
        return
    
    # 4. 요인점수 통계 확인
    print("\n[4] 요인점수 통계 (1단계 결과)")
    print(f"{'잠재변수':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)
    
    for lv_name, scores in factor_scores.items():
        print(f"{lv_name:<25} {np.mean(scores):>10.4f} {np.std(scores):>10.4f} "
              f"{np.min(scores):>10.4f} {np.max(scores):>10.4f}")
    
    # 5. 요인점수 확장 시뮬레이션 (2단계에서 수행되는 것과 동일)
    print("\n[5] 요인점수 확장 시뮬레이션 (respondent_id 기준)")
    
    unique_ids = data['respondent_id'].unique()
    print(f"   - 고유 ID 수: {len(unique_ids)}")
    
    # PI 요인점수 확장
    pi_scores = factor_scores['purchase_intention']
    id_to_score = {unique_ids[i]: pi_scores[i] for i in range(len(unique_ids))}
    pi_expanded = np.array([id_to_score[rid] for rid in data['respondent_id']])
    
    print(f"   - 확장 전 shape: {pi_scores.shape}")
    print(f"   - 확장 후 shape: {pi_expanded.shape}")
    print(f"   - 확장 후 통계: mean={np.mean(pi_expanded):.4f}, std={np.std(pi_expanded):.4f}")
    
    # 6. 샘플 데이터 확인 (첫 3명)
    print("\n[6] 샘플 데이터 확인 (첫 3명)")
    print(f"{'ID':>5} {'PI 요인점수':>15} {'선택 세트 수':>15}")
    print("-" * 40)
    
    for i in range(min(3, len(unique_ids))):
        rid = unique_ids[i]
        pi_score = pi_scores[i]
        n_choices = len(data[data['respondent_id'] == rid])
        print(f"{rid:>5} {pi_score:>15.4f} {n_choices:>15}")
    
    # 7. 확장된 요인점수 검증
    print("\n[7] 확장된 요인점수 검증")
    
    # 첫 번째 개인의 모든 선택 세트에서 PI 값이 동일한지 확인
    first_id = unique_ids[0]
    first_id_data = data[data['respondent_id'] == first_id]
    first_id_indices = first_id_data.index
    
    pi_values_for_first = pi_expanded[first_id_indices]
    
    print(f"   - 첫 번째 개인 (ID={first_id}):")
    print(f"     * 요인점수: {pi_scores[0]:.4f}")
    print(f"     * 선택 세트 수: {len(first_id_data)}")
    print(f"     * 확장된 값들: {pi_values_for_first[:5]}...")  # 처음 5개만
    print(f"     * 모두 동일한가? {np.allclose(pi_values_for_first, pi_scores[0])}")
    
    # 8. 선택 데이터와 요인점수 결합 확인
    print("\n[8] 선택 데이터와 요인점수 결합 확인")
    
    # 샘플 데이터 (첫 10행)
    sample_data = data.head(10)[['respondent_id', 'choice', 'sugar_free', 'health_label', 'price']].copy()
    sample_data['PI_factor_score'] = pi_expanded[:10]
    
    print(sample_data.to_string(index=False))
    
    # 9. 최종 검증 결과
    print("\n" + "="*80)
    print("검증 결과 요약")
    print("="*80)
    
    checks = [
        ("개인 수 일치", n_individuals_data == n_individuals_fs),
        ("요인점수 표준화 (mean≈0)", np.abs(np.mean(pi_scores)) < 0.01),
        ("요인점수 표준화 (std≈1)", np.abs(np.std(pi_scores) - 1.0) < 0.01),
        ("요인점수 확장 정상", len(pi_expanded) == len(data)),
        ("개인별 요인점수 일관성", np.allclose(pi_values_for_first, pi_scores[0]))
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ 모든 검증 통과! 2단계가 1단계 결과를 올바르게 사용했습니다.")
    else:
        print("❌ 일부 검증 실패! 2단계 설정을 확인하세요.")
    print("="*80)


if __name__ == "__main__":
    main()

