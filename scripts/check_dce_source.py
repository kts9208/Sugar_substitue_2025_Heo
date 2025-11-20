"""
DCE 데이터 소스 확인
"""
import pandas as pd
from pathlib import Path

print("="*80)
print("DCE 데이터 소스 확인")
print("="*80)

# 1. DCE long format 확인
dce_path = Path('data/processed/dce/dce_long_format.csv')
if dce_path.exists():
    print(f"\n[1] DCE Long Format: {dce_path}")
    dce = pd.read_csv(dce_path)
    print(f"  Shape: {dce.shape}")
    print(f"  컬럼: {list(dce.columns)}")
    
    # respondent_id 확인
    if 'respondent_id' in dce.columns:
        print(f"\n  respondent_id 분석:")
        print(f"    고유 ID 수: {dce['respondent_id'].nunique()}")
        print(f"    전체 행 수: {len(dce)}")
        
        # ID별 행 수
        id_counts = dce.groupby('respondent_id').size()
        print(f"\n    ID별 행 수 분포:")
        print(f"      {id_counts.value_counts().sort_index().to_dict()}")
        
        # 중복 확인
        dup_ids = id_counts[id_counts > 18].index.tolist()
        if dup_ids:
            print(f"\n    ⚠️ 중복된 ID (행 수 > 18): {len(dup_ids)}개")
            print(f"    중복 ID: {dup_ids}")
            
            # 257, 273 확인
            for rid in [257, 273]:
                if rid in id_counts.index:
                    print(f"\n    ID {rid}: {id_counts[rid]}행")
                    subset = dce[dce['respondent_id'] == rid]
                    
                    # choice_set 확인
                    if 'choice_set' in subset.columns:
                        cs_counts = subset.groupby('choice_set').size()
                        print(f"      choice_set별 행 수: {cs_counts.to_dict()}")
                    
                    # 첫 20행
                    print(f"\n      첫 20행:")
                    print(subset[['respondent_id', 'choice_set', 'alternative', 'choice']].head(20))

# 2. integrated_data 확인
integrated_path = Path('data/processed/iclv/integrated_data.csv')
if integrated_path.exists():
    print(f"\n{'='*80}")
    print(f"[2] Integrated Data: {integrated_path}")
    integrated = pd.read_csv(integrated_path)
    print(f"  Shape: {integrated.shape}")
    
    if 'respondent_id' in integrated.columns:
        print(f"\n  respondent_id 분석:")
        print(f"    고유 ID 수: {integrated['respondent_id'].nunique()}")
        print(f"    전체 행 수: {len(integrated)}")
        
        # ID별 행 수
        id_counts = integrated.groupby('respondent_id').size()
        print(f"\n    ID별 행 수 분포:")
        print(f"      {id_counts.value_counts().sort_index().to_dict()}")
        
        # 중복 확인
        dup_ids = id_counts[id_counts > 18].index.tolist()
        if dup_ids:
            print(f"\n    ⚠️ 중복된 ID (행 수 > 18): {len(dup_ids)}개")
            print(f"    중복 ID: {dup_ids}")

# 3. 데이터 생성 스크립트 확인
print(f"\n{'='*80}")
print(f"[3] 데이터 생성 스크립트 확인")
print(f"{'='*80}")

scripts_to_check = [
    'scripts/integrate_iclv_data.py',
    'scripts/preprocess_dce_data.py',
    'scripts/create_dce_design_matrix.py'
]

for script_path in scripts_to_check:
    p = Path(script_path)
    if p.exists():
        print(f"\n✓ {p.name} 존재")
    else:
        print(f"\n✗ {p.name} 없음")

