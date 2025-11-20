"""
개인별 구조모델 그래디언트 확인

Author: Sugar Substitute Research Team
Date: 2025-11-20
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 80)
    print("개인별 구조모델 그래디언트 확인")
    print("=" * 80)
    
    # 최신 로그 파일 찾기
    results_dir = project_root / 'results'
    log_files = sorted(results_dir.glob('simultaneous_estimation_log_*_params_grads.csv'), reverse=True)
    
    if not log_files:
        print("❌ 로그 파일을 찾을 수 없습니다.")
        return
    
    log_file = log_files[0]
    print(f"\n[1] 로그 파일: {log_file.name}")
    
    # CSV 로드
    df = pd.read_csv(log_file)
    
    # Iteration #1만 필터링
    df_iter1 = df[df['iteration'] == 1].copy()
    
    print(f"\n[2] Iteration #1 데이터")
    print(f"  개인 수: {len(df_iter1)}")
    print(f"  컬럼: {list(df_iter1.columns)}")
    
    # 구조모델 그래디언트 컬럼 찾기
    struct_grad_cols = [col for col in df_iter1.columns if col.startswith('grad_gamma_')]
    
    print(f"\n[3] 구조모델 그래디언트 컬럼: {struct_grad_cols}")
    
    # 각 파라미터별 통계
    print("\n[4] 개인별 그래디언트 통계 (Iteration #1)")
    print("=" * 80)
    
    for col in struct_grad_cols:
        values = df_iter1[col].values
        
        print(f"\n{col}:")
        print(f"  평균: {values.mean():.6f}")
        print(f"  표준편차: {values.std():.6f}")
        print(f"  최소: {values.min():.6f}")
        print(f"  최대: {values.max():.6f}")
        print(f"  중앙값: {np.median(values):.6f}")
        print(f"  25% 분위: {np.percentile(values, 25):.6f}")
        print(f"  75% 분위: {np.percentile(values, 75):.6f}")
        
        # 0에 가까운 값 비율
        near_zero = np.abs(values) < 0.001
        print(f"  |gradient| < 0.001인 개인 비율: {near_zero.sum() / len(values) * 100:.1f}%")
        
        # 양수/음수 비율
        positive = values > 0
        negative = values < 0
        print(f"  양수: {positive.sum()}명 ({positive.sum() / len(values) * 100:.1f}%)")
        print(f"  음수: {negative.sum()}명 ({negative.sum() / len(values) * 100:.1f}%)")
    
    # 선택모델 그래디언트와 비교
    choice_grad_cols = [col for col in df_iter1.columns if col.startswith('grad_asc_') or col.startswith('grad_beta_')]
    
    if choice_grad_cols:
        print("\n[5] 선택모델 그래디언트 통계 (비교용)")
        print("=" * 80)
        
        for col in choice_grad_cols[:2]:  # 처음 2개만
            values = df_iter1[col].values
            
            print(f"\n{col}:")
            print(f"  평균: {values.mean():.6f}")
            print(f"  표준편차: {values.std():.6f}")
            print(f"  최소: {values.min():.6f}")
            print(f"  최대: {values.max():.6f}")
    
    # 비율 계산
    print("\n[6] 그래디언트 크기 비율")
    print("=" * 80)
    
    if struct_grad_cols and choice_grad_cols:
        struct_mean = df_iter1[struct_grad_cols[0]].abs().mean()
        choice_mean = df_iter1[choice_grad_cols[0]].abs().mean()
        
        print(f"\n구조모델 평균 |gradient|: {struct_mean:.6f}")
        print(f"선택모델 평균 |gradient|: {choice_mean:.6f}")
        print(f"비율 (선택/구조): {choice_mean / struct_mean:.1f}x")
    
    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()

