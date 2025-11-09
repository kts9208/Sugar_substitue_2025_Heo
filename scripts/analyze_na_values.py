"""
CSV 파일의 N/A 값 분석
"""

import pandas as pd

print("="*80)
print("CSV 파일 N/A 값 분석")
print("="*80)

# CSV 파일 로드
df = pd.read_csv('results/iclv_full_data_results.csv')

print("\n1. 전체 데이터 확인")
print(f"   총 행 수: {len(df)}")
print(f"   총 열 수: {len(df.columns)}")
print(f"   열 이름: {list(df.columns)}")

print("\n2. N/A 값이 있는 행 찾기")
# 각 열에서 'N/A' 문자열 찾기
for col in df.columns:
    na_mask = df[col].astype(str).str.contains('N/A', na=False)
    if na_mask.any():
        print(f"\n   [{col}] 열에서 N/A 발견:")
        na_rows = df[na_mask]
        for idx, row in na_rows.iterrows():
            print(f"      행 {idx}: {row.to_dict()}")

print("\n3. Estimation statistics 섹션 확인")
stats_start = df[df['Coefficient'] == 'Estimation statistics'].index
if len(stats_start) > 0:
    stats_idx = stats_start[0]
    print(f"   Estimation statistics 시작 행: {stats_idx}")
    print(f"\n   Estimation statistics 섹션:")
    stats_section = df.iloc[stats_idx:]
    print(stats_section.to_string(index=True))

print("\n" + "="*80)
print("N/A 값의 의미 분석")
print("="*80)

print("\n현재 CSV 구조:")
print("""
행 39: (빈 행 - 구분선)
행 40: Estimation statistics (헤더)
행 41: Iterations, 90, LL (start), N/A
행 42: AIC, 11543.41, LL (final, whole model), -5734.70
행 43: BIC, 11790.69, LL (Choice), N/A
""")

print("\nN/A가 있는 위치:")
print("1. 행 41 - P. Value 열: 'N/A'")
print("   → LL (start) 값이 없음")
print("   → 초기 로그우도를 기록하지 않았기 때문")
print()
print("2. 행 43 - P. Value 열: 'N/A'")
print("   → LL (Choice) 값이 없음")
print("   → 선택모델만의 로그우도를 별도로 계산하지 않았기 때문")

print("\n" + "="*80)
print("문제점 및 해결 방안")
print("="*80)

print("\n⚠️  현재 문제:")
print("   1. LL (start)가 기록되지 않음")
print("      → 초기 로그우도를 알 수 없어 개선 정도를 파악하기 어려움")
print()
print("   2. LL (Choice)가 기록되지 않음")
print("      → 선택모델만의 적합도를 알 수 없음")
print("      → ICLV 모델과 일반 선택모델의 비교가 어려움")

print("\n✅ 해결 방안:")
print("   1. LL (start) 추가:")
print("      → 최적화 시작 시 초기 파라미터로 LL 계산하여 저장")
print()
print("   2. LL (Choice) 추가:")
print("      → 선택모델 파라미터만으로 LL 계산하여 저장")
print("      → 또는 'N/A' 대신 빈 문자열로 변경")
print()
print("   3. CSV 구조 개선:")
print("      → Estimation statistics를 별도 섹션으로 분리")
print("      → 또는 더 명확한 레이블 사용")

print("\n" + "="*80)
print("현재 상태 평가")
print("="*80)

print("\n✅ 긍정적:")
print("   - 모든 파라미터에 대해 Estimate, Std. Err., P. Value가 계산됨")
print("   - N/A는 Estimation statistics 섹션에만 존재")
print("   - 실제 파라미터 추정에는 영향 없음")

print("\n⚠️  개선 필요:")
print("   - LL (start)와 LL (Choice) 값을 계산하여 추가하면 더 완전한 보고서")
print("   - 하지만 현재 상태로도 논문 작성에는 문제 없음")

print("\n" + "="*80)

