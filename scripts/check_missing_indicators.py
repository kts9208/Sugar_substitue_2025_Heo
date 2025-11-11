"""
결측 지표 확인 스크립트

모든 측정 지표가 결측인 개인을 찾습니다.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def main():
    print("=" * 70)
    print("결측 지표 확인")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    print(f"   전체 데이터 shape: {data.shape}")
    
    # individual_id 컬럼 확인
    if 'respondent_id' in data.columns:
        data['individual_id'] = data['respondent_id']
    
    n_individuals = data['individual_id'].nunique()
    print(f"   개인 수: {n_individuals}")
    
    # 2. 모든 측정 지표 리스트
    all_indicators = (
        ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'] +  # Health Concern
        ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'] +  # Perceived Benefit
        ['q27', 'q28', 'q29'] +  # Perceived Price
        [f'q{i}' for i in range(30, 50)] +  # Nutrition Knowledge
        ['q18', 'q19', 'q20']  # Purchase Intention
    )
    
    print(f"\n2. 측정 지표 개수: {len(all_indicators)}")
    
    # 3. 각 개인별로 결측 지표 확인
    print(f"\n3. 개인별 결측 지표 확인...")
    
    problematic_individuals = []
    
    for ind_id in data['individual_id'].unique():
        ind_data = data[data['individual_id'] == ind_id]
        
        # 첫 번째 행의 지표 값 확인 (모든 행이 동일해야 함)
        first_row = ind_data.iloc[0]
        
        # 모든 지표가 결측인지 확인
        all_missing = all(pd.isna(first_row[ind]) for ind in all_indicators)
        
        if all_missing:
            problematic_individuals.append(ind_id)
            print(f"   ⚠️ 개인 {ind_id}: 모든 지표 결측")
    
    # 4. 요약
    print(f"\n4. 요약:")
    print(f"   전체 개인 수: {n_individuals}")
    print(f"   문제 있는 개인 수: {len(problematic_individuals)}")
    print(f"   정상 개인 수: {n_individuals - len(problematic_individuals)}")
    
    if problematic_individuals:
        print(f"\n   문제 있는 개인 ID: {problematic_individuals}")
    
    # 5. 각 지표별 결측률 확인
    print(f"\n5. 각 지표별 결측률:")
    
    # 개인별로 첫 번째 행만 추출 (지표는 개인별로 동일)
    first_rows = data.groupby('individual_id').first()
    
    for ind in all_indicators:
        missing_count = first_rows[ind].isna().sum()
        missing_rate = missing_count / len(first_rows) * 100
        print(f"   {ind}: {missing_count}/{len(first_rows)} ({missing_rate:.1f}%)")
    
    # 6. 권장사항
    print(f"\n6. 권장사항:")
    if len(problematic_individuals) > 0:
        print(f"   ⚠️ {len(problematic_individuals)}명의 개인은 모든 지표가 결측입니다.")
        print(f"   이들을 데이터에서 제거하거나 추정 시 제외해야 합니다.")
        print(f"\n   제거 후 개인 수: {n_individuals - len(problematic_individuals)}")
    else:
        print(f"   ✅ 모든 개인이 최소 1개 이상의 지표를 가지고 있습니다.")
    
    print("\n" + "=" * 70)
    print("분석 완료")
    print("=" * 70)

if __name__ == '__main__':
    main()

