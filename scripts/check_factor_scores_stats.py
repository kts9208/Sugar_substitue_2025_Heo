"""
요인점수 통계 파일 확인 스크립트

1단계 추정 결과로 생성된 요인점수 통계 파일을 읽어서 보기 좋게 출력합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 100)
    print("요인점수 통계 파일 확인")
    print("=" * 100)
    
    # 통계 파일 경로
    stats_file = project_root / "results" / "sequential_stage_wise" / "stage1_results_factor_scores_stats.csv"
    
    if not stats_file.exists():
        print(f"❌ 통계 파일을 찾을 수 없습니다: {stats_file}")
        return
    
    # CSV 파일 읽기
    stats_df = pd.read_csv(stats_file)
    
    print(f"\n✅ 파일 로드 완료: {stats_file}")
    print(f"   총 {len(stats_df)}개 잠재변수\n")
    
    # 보기 좋게 출력
    print("=" * 100)
    print("요인점수 통계 (표준화 후)")
    print("=" * 100)
    print(f"{'잠재변수':25s} {'평균':>12s} {'분산':>12s} {'표준편차':>12s} {'최소값':>12s} {'최대값':>12s} {'관측수':>8s} {'경고':>6s}")
    print("-" * 100)
    
    for _, row in stats_df.iterrows():
        warning_flag = "⚠️ YES" if row['low_variance_warning'] == 'YES' else "NO"
        print(f"{row['latent_variable']:25s} "
              f"{row['mean']:>12.6f} "
              f"{row['variance']:>12.6f} "
              f"{row['std']:>12.6f} "
              f"{row['min']:>12.4f} "
              f"{row['max']:>12.4f} "
              f"{int(row['n_observations']):>8d} "
              f"{warning_flag:>6s}")
    
    print("-" * 100)
    
    # 경고 요약
    low_var_count = (stats_df['low_variance_warning'] == 'YES').sum()
    
    if low_var_count > 0:
        print(f"\n⚠️  분산이 0.01 미만인 변수: {low_var_count}개")
        print("\n[분산이 작은 변수 목록]")
        low_var_df = stats_df[stats_df['low_variance_warning'] == 'YES']
        for _, row in low_var_df.iterrows():
            print(f"   - {row['latent_variable']:25s}: 분산 = {row['variance']:.6f}")
        print("\n   → 선택모델에서 비유의할 가능성이 높습니다.")
        print("   → 측정모델 재검토 또는 지표 추가를 고려하세요.")
    else:
        print(f"\n✅ 모든 변수의 분산이 임계값(0.01) 이상입니다.")
    
    # 요인점수 전체 파일도 확인
    print("\n" + "=" * 100)
    print("요인점수 전체 데이터 확인")
    print("=" * 100)
    
    factor_scores_file = project_root / "results" / "sequential_stage_wise" / "stage1_results_factor_scores.csv"
    
    if factor_scores_file.exists():
        fs_df = pd.read_csv(factor_scores_file)
        print(f"\n✅ 요인점수 파일 로드 완료: {factor_scores_file}")
        print(f"   총 {len(fs_df)}명의 응답자")
        print(f"   변수: {list(fs_df.columns[1:])}")  # observation_id 제외
        
        print("\n[요인점수 샘플 (처음 10명)]")
        print(fs_df.head(10).to_string(index=False))
        
        print("\n[요인점수 기술통계]")
        print(fs_df.describe().to_string())
    else:
        print(f"\n❌ 요인점수 파일을 찾을 수 없습니다: {factor_scores_file}")
    
    print("\n" + "=" * 100)
    print("✅ 확인 완료")
    print("=" * 100)


if __name__ == '__main__':
    main()

