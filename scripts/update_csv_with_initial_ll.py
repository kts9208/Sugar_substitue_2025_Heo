"""
기존 CSV 파일에 초기 LL 값 추가
"""

import pandas as pd
from pathlib import Path

# 프로젝트 루트
project_root = Path(__file__).parent.parent

# 로그 파일에서 초기 LL 읽기
log_file = project_root / 'results' / 'iclv_full_data_estimation_log.txt'
initial_ll = None

print("="*80)
print("CSV 파일에 초기 LL 값 추가")
print("="*80)

print("\n1. 로그 파일에서 초기 LL 읽기...")
print(f"   로그 파일: {log_file}")

try:
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Iter    1:' in line and 'LL =' in line:
                # "Iter    1: LL =   -7581.2098 (Best:   -7581.2098) [NEW BEST]"
                ll_str = line.split('LL =')[1].split('(')[0].strip()
                initial_ll = float(ll_str)
                print(f"   ✅ 초기 LL 발견: {initial_ll:.4f}")
                break
    
    if initial_ll is None:
        print("   ⚠️  초기 LL을 찾을 수 없습니다.")
        exit(1)
        
except Exception as e:
    print(f"   ❌ 로그 파일 읽기 실패: {e}")
    exit(1)

# CSV 파일 로드
csv_file = project_root / 'results' / 'iclv_full_data_results.csv'
print(f"\n2. CSV 파일 로드...")
print(f"   CSV 파일: {csv_file}")

df = pd.read_csv(csv_file)
print(f"   ✅ CSV 로드 완료 (총 {len(df)}행)")

# N/A가 있는 행 찾기
print("\n3. N/A 값 찾기...")
iterations_row_idx = df[df['Coefficient'] == 'Iterations'].index
if len(iterations_row_idx) == 0:
    print("   ❌ 'Iterations' 행을 찾을 수 없습니다.")
    exit(1)

iterations_row_idx = iterations_row_idx[0]
print(f"   ✅ 'Iterations' 행 발견: 행 {iterations_row_idx}")

# 현재 값 확인
current_value = df.loc[iterations_row_idx, 'P. Value']
print(f"   현재 P. Value 값: {current_value}")

# 초기 LL 값으로 업데이트
print("\n4. 초기 LL 값으로 업데이트...")
df.loc[iterations_row_idx, 'P. Value'] = f"{initial_ll:.2f}"
print(f"   ✅ 업데이트 완료: {current_value} → {initial_ll:.2f}")

# CSV 저장
print("\n5. CSV 파일 저장...")
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
print(f"   ✅ CSV 저장 완료: {csv_file}")

# 결과 확인
print("\n6. 업데이트 결과 확인...")
print("\n   Estimation statistics 섹션:")
stats_start = df[df['Coefficient'] == 'Estimation statistics'].index[0]
stats_section = df.iloc[stats_start:]
print(stats_section.to_string(index=False))

print("\n" + "="*80)
print("✅ 초기 LL 값 추가 완료!")
print("="*80)

# LL 개선 정도 계산
final_ll_row = df[df['Std. Err.'] == 'LL (final, whole model)']
if len(final_ll_row) > 0:
    final_ll = float(final_ll_row['P. Value'].values[0])
    improvement = final_ll - initial_ll
    improvement_pct = (improvement / abs(initial_ll)) * 100
    
    print(f"\n📊 LL 개선 정도:")
    print(f"   초기 LL:  {initial_ll:10.2f}")
    print(f"   최종 LL:  {final_ll:10.2f}")
    print(f"   개선:     {improvement:10.2f} ({improvement_pct:.1f}%)")

print("\n" + "="*80)

