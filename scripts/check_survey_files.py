"""
Survey 파일 확인 스크립트

Survey 파일들에 개인 7번이 있는지 확인합니다.
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
    print("Survey 파일 확인")
    print("=" * 70)
    
    survey_path = project_root / 'data' / 'processed' / 'survey'
    
    survey_files = {
        'health_concern.csv': 'Health Concern',
        'perceived_benefit.csv': 'Perceived Benefit',
        'perceived_price.csv': 'Perceived Price',
        'nutrition_knowledge.csv': 'Nutrition Knowledge',
        'purchase_intention.csv': 'Purchase Intention'
    }
    
    for filename, name in survey_files.items():
        filepath = survey_path / filename
        
        if not filepath.exists():
            print(f"\n⚠️ {name}: 파일 없음")
            continue
        
        print(f"\n{name} ({filename}):")
        print("-" * 70)
        
        data = pd.read_csv(filepath)
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        
        # ID 컬럼 확인
        if 'no' in data.columns:
            id_col = 'no'
        elif 'respondent_id' in data.columns:
            id_col = 'respondent_id'
        else:
            print(f"  ⚠️ ID 컬럼 없음!")
            continue
        
        print(f"  ID 컬럼: {id_col}")
        print(f"  개인 수: {data[id_col].nunique()}")
        print(f"  ID 범위: {data[id_col].min()} ~ {data[id_col].max()}")
        
        # 개인 7번 확인
        ind_7 = data[data[id_col] == 7]
        
        if len(ind_7) > 0:
            print(f"  ✅ 개인 7번 존재: {len(ind_7)}행")
            
            # 첫 번째 행의 값 확인
            first_row = ind_7.iloc[0]
            q_cols = [c for c in data.columns if c.startswith('q')]
            
            print(f"  측정 지표 ({len(q_cols)}개):")
            for col in q_cols[:10]:  # 처음 10개만
                value = first_row[col]
                print(f"    {col}: {value} (NaN: {pd.isna(value)})")
        else:
            print(f"  ❌ 개인 7번 없음!")
            
            # 어떤 ID들이 있는지 확인
            all_ids = sorted(data[id_col].unique())
            print(f"  존재하는 ID (처음 20개): {all_ids[:20]}")
            print(f"  존재하는 ID (마지막 20개): {all_ids[-20:]}")
    
    print("\n" + "=" * 70)
    print("분석 완료")
    print("=" * 70)

if __name__ == '__main__':
    main()

