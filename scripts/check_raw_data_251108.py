"""
원본 데이터 확인 - Sugar_substitue_Raw data_251108.xlsx
"""
import pandas as pd
from pathlib import Path

# 원본 데이터 로드
raw_data_path = Path('data/raw/Sugar_substitue_Raw data_251108.xlsx')

print("="*80)
print("원본 데이터 확인: Sugar_substitue_Raw data_251108.xlsx")
print("="*80)

# Excel 파일의 시트 확인
xl_file = pd.ExcelFile(raw_data_path)
print(f"\n시트 목록: {xl_file.sheet_names}")

# 각 시트 확인
for sheet_name in xl_file.sheet_names:
    print(f"\n{'='*80}")
    print(f"시트: {sheet_name}")
    print(f"{'='*80}")
    
    df = pd.read_excel(raw_data_path, sheet_name=sheet_name)
    print(f"Shape: {df.shape}")
    print(f"컬럼 수: {len(df.columns)}")
    print(f"행 수: {len(df)}")
    
    # 컬럼 이름 출력 (처음 20개)
    print(f"\n컬럼 (처음 20개):")
    for i, col in enumerate(df.columns[:20], 1):
        print(f"  {i:2d}. {col}")
    
    if len(df.columns) > 20:
        print(f"  ... (총 {len(df.columns)}개)")
    
    # respondent_id 또는 ID 컬럼 찾기
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'respondent' in col.lower()]
    if id_cols:
        print(f"\nID 관련 컬럼: {id_cols}")
        
        for id_col in id_cols:
            unique_ids = df[id_col].nunique()
            total_rows = len(df)
            print(f"\n{id_col}:")
            print(f"  고유 ID 수: {unique_ids}")
            print(f"  전체 행 수: {total_rows}")
            print(f"  ID당 평균 행 수: {total_rows / unique_ids:.2f}")
            
            # ID별 행 수 분포
            id_counts = df.groupby(id_col).size()
            print(f"\n  ID별 행 수 분포:")
            print(f"    {id_counts.value_counts().sort_index().to_dict()}")
            
            # 중복된 ID 확인 (행 수 > 18)
            dup_ids = id_counts[id_counts > 18].index.tolist()
            if dup_ids:
                print(f"\n  ⚠️ 중복된 ID (행 수 > 18): {len(dup_ids)}개")
                print(f"  중복 ID 목록: {dup_ids[:10]}")  # 처음 10개만
                
                # 257, 273 확인
                for check_id in [257, 273]:
                    if check_id in id_counts.index:
                        print(f"\n  ID {check_id}: {id_counts[check_id]}행")
                        
                        # 해당 ID의 데이터 확인
                        subset = df[df[id_col] == check_id]
                        
                        # choice_set 관련 컬럼 찾기
                        cs_cols = [col for col in subset.columns if 'choice' in col.lower() or 'set' in col.lower()]
                        if cs_cols:
                            print(f"    choice_set 관련 컬럼: {cs_cols[:5]}")
                            
                            # 첫 번째 choice_set 컬럼 사용
                            if cs_cols:
                                cs_col = cs_cols[0]
                                if cs_col in subset.columns:
                                    cs_counts = subset.groupby(cs_col).size()
                                    print(f"    {cs_col}별 행 수:")
                                    print(f"      {cs_counts.to_dict()}")
    
    # 첫 5행 출력
    print(f"\n첫 5행 (주요 컬럼만):")
    display_cols = [col for col in df.columns[:10]]
    print(df[display_cols].head())

