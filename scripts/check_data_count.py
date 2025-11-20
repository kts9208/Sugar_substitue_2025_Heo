"""
데이터 개수 확인

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
    print("데이터 개수 확인")
    print("=" * 80)
    
    # 데이터 로드
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    df = pd.read_csv(data_path)
    
    print(f"\n[1] 전체 데이터")
    print(f"  전체 행 수: {len(df)}")
    print(f"  고유 respondent_id 수: {df['respondent_id'].nunique()}")
    
    print(f"\n[2] respondent_id 범위")
    print(f"  최소: {df['respondent_id'].min()}")
    print(f"  최대: {df['respondent_id'].max()}")
    
    print(f"\n[3] respondent_id 값 확인")
    ids = sorted(df['respondent_id'].unique())
    print(f"  처음 10개: {ids[:10]}")
    print(f"  마지막 10개: {ids[-10:]}")
    
    print(f"\n[4] 누락된 ID 확인")
    all_ids = set(range(1, 329))
    actual_ids = set(df['respondent_id'].unique())
    missing = sorted(all_ids - actual_ids)
    
    if missing:
        print(f"  ⚠️ 누락된 ID ({len(missing)}개): {missing}")
    else:
        print(f"  ✅ 누락된 ID 없음")
    
    print(f"\n[5] 개인별 선택 상황 수")
    choice_counts = df.groupby('respondent_id').size()
    print(f"  평균: {choice_counts.mean():.2f}")
    print(f"  최소: {choice_counts.min()}")
    print(f"  최대: {choice_counts.max()}")
    print(f"  표준편차: {choice_counts.std():.2f}")
    
    # 18개가 아닌 개인 찾기
    not_18 = choice_counts[choice_counts != 18]
    if len(not_18) > 0:
        print(f"\n  ⚠️ 선택 상황이 18개가 아닌 개인 ({len(not_18)}명):")
        for ind_id, count in not_18.items():
            print(f"    ID {ind_id}: {count}개")
    else:
        print(f"\n  ✅ 모든 개인이 18개 선택 상황을 가짐")
    
    print(f"\n[6] 결측치 확인")
    # 주요 컬럼 결측치 확인
    key_cols = ['choice', 'price', 'sugar_free', 'health_label']
    for col in key_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                print(f"  ⚠️ {col}: {missing_count}개 결측")
                # 결측이 있는 개인 ID
                missing_ids = df[df[col].isna()]['respondent_id'].unique()
                print(f"    결측이 있는 개인 ID: {sorted(missing_ids)}")
            else:
                print(f"  ✅ {col}: 결측 없음")
    
    print("\n" + "=" * 80)
    print("확인 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()

