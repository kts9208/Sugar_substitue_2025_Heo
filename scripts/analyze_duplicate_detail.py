"""
중복 데이터 상세 분석 - 선택 응답이 다른지 확인
"""
import pandas as pd
from pathlib import Path

# 데이터 로드
data_path = Path('data/processed/iclv/integrated_data.csv')
data = pd.read_csv(data_path)

print("="*80)
print("중복 데이터 상세 분석")
print("="*80)

# 중복된 개인 ID
dup_ids = [257, 273]

for rid in dup_ids:
    subset = data[data['respondent_id'] == rid].copy()
    
    print(f"\n{'='*80}")
    print(f"respondent_id: {rid}")
    print(f"{'='*80}")
    
    # 각 choice_set별로 중복 확인
    for cs in sorted(subset['choice_set'].unique()):
        cs_data = subset[subset['choice_set'] == cs]
        
        print(f"\n--- choice_set {cs} ---")
        print(f"행 수: {len(cs_data)}")
        
        if len(cs_data) > 3:
            # 중복이 있음
            print("⚠️ 중복 발견!")
            
            # 첫 3행과 다음 3행 비교
            first_3 = cs_data.iloc[:3]
            second_3 = cs_data.iloc[3:6] if len(cs_data) >= 6 else cs_data.iloc[3:]
            
            print("\n[첫 번째 세트]")
            print(first_3[['alternative', 'alternative_name', 'product_type', 'sugar_content', 
                          'health_label', 'price', 'choice']])
            
            print("\n[두 번째 세트]")
            print(second_3[['alternative', 'alternative_name', 'product_type', 'sugar_content', 
                           'health_label', 'price', 'choice']])
            
            # 속성 값 비교
            attrs_to_check = ['product_type', 'sugar_content', 'health_label', 'price']
            print("\n[속성 값 비교]")
            for attr in attrs_to_check:
                first_vals = first_3[attr].tolist()
                second_vals = second_3[attr].tolist()
                if first_vals == second_vals:
                    print(f"  {attr}: 동일 ✓")
                else:
                    print(f"  {attr}: 다름 ✗")
                    print(f"    첫 번째: {first_vals}")
                    print(f"    두 번째: {second_vals}")
            
            # 선택 응답 비교
            first_choice = first_3['choice'].tolist()
            second_choice = second_3['choice'].tolist()
            print(f"\n[선택 응답 비교]")
            print(f"  첫 번째: {first_choice}")
            print(f"  두 번째: {second_choice}")
            if first_choice == second_choice:
                print(f"  → 선택 응답 동일 (완전 중복)")
            else:
                print(f"  → 선택 응답 다름 (재응답?)")

print("\n" + "="*80)
print("원본 데이터 소스 확인")
print("="*80)

# 원본 데이터 파일들 확인
raw_data_dir = Path('data/raw')
if raw_data_dir.exists():
    print(f"\n원본 데이터 디렉토리: {raw_data_dir}")
    for file in raw_data_dir.glob('*.csv'):
        print(f"  - {file.name}")
        
        # 각 파일에서 respondent_id 257, 273 확인
        try:
            df = pd.read_csv(file)
            if 'respondent_id' in df.columns or 'id' in df.columns or 'ID' in df.columns:
                id_col = 'respondent_id' if 'respondent_id' in df.columns else ('id' if 'id' in df.columns else 'ID')
                for rid in dup_ids:
                    count = (df[id_col] == rid).sum()
                    if count > 0:
                        print(f"    respondent_id {rid}: {count}행")
        except Exception as e:
            print(f"    (읽기 실패: {e})")

