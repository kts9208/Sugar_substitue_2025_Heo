"""
측정지표 간 상관관계 확인

목적: 모든 지표가 같은 방향으로 코딩되었는지 확인
"""

import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('data/processed/iclv/integrated_data.csv')

# 측정지표 리스트
indicators = {
    'health_concern': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    'perceived_benefit': ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
    'purchase_intention': ['q18', 'q19', 'q20'],
    'perceived_price': ['q27', 'q28', 'q29'],
}

print("=" * 80)
print("측정지표 간 상관관계 분석")
print("=" * 80)

for lv_name, inds in indicators.items():
    print(f"\n[{lv_name}]")
    print("-" * 80)
    
    # 상관계수 행렬
    corr_matrix = df[inds].corr()
    
    # 하삼각 행렬만 추출 (대각선 제외)
    correlations = []
    for i in range(len(inds)):
        for j in range(i):
            correlations.append(corr_matrix.iloc[i, j])
    
    # 통계
    min_corr = np.min(correlations)
    max_corr = np.max(correlations)
    mean_corr = np.mean(correlations)
    
    print(f"  상관계수 범위: [{min_corr:.3f}, {max_corr:.3f}]")
    print(f"  상관계수 평균: {mean_corr:.3f}")
    
    # 음수 상관계수 확인
    negative_corrs = [c for c in correlations if c < 0]
    if negative_corrs:
        print(f"  ⚠️  음수 상관계수 발견: {len(negative_corrs)}개")
        print(f"     → 일부 지표가 역코딩되었을 가능성")
    else:
        print(f"  ✅ 모든 상관계수가 양수")
        print(f"     → 모든 지표가 같은 방향으로 코딩됨")
        print(f"     → 모든 ζ가 양수일 것으로 예상")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

all_negative = False
for lv_name, inds in indicators.items():
    corr_matrix = df[inds].corr()
    correlations = []
    for i in range(len(inds)):
        for j in range(i):
            correlations.append(corr_matrix.iloc[i, j])
    
    if any(c < 0 for c in correlations):
        all_negative = True
        print(f"\n⚠️  {lv_name}: 음수 상관계수 존재")

if not all_negative:
    print("\n✅ 모든 잠재변수의 지표들이 양의 상관관계를 가짐")
    print("   → 모든 요인적재량(ζ)이 양수일 것으로 예상")
    print("   → 음수 ζ가 나온다면 모델 문제일 가능성")
else:
    print("\n⚠️  일부 잠재변수에 음수 상관계수 존재")
    print("   → 역코딩 지표가 있을 수 있음")
    print("   → 음수 ζ가 나올 수 있음")

