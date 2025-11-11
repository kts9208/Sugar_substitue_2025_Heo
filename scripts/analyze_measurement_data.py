"""
측정모델 데이터 분석 스크립트

목적: 측정모델 우도가 높은 원인 분석
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
    'nutrition_knowledge': [f'q{i}' for i in range(30, 50)]
}

print("=" * 80)
print("측정모델 데이터 분석")
print("=" * 80)

# 1. 각 잠재변수별 지표 통계
for lv_name, inds in indicators.items():
    print(f"\n[{lv_name}] ({len(inds)}개 지표)")
    print("-" * 80)
    
    for ind in inds:
        if ind in df.columns:
            mean = df[ind].mean()
            std = df[ind].std()
            var = df[ind].var()
            min_val = df[ind].min()
            max_val = df[ind].max()
            
            print(f"  {ind:5s}: mean={mean:5.2f}, std={std:5.2f}, var={var:5.2f}, range=[{min_val:.0f}, {max_val:.0f}]")

# 2. 전체 통계
print("\n" + "=" * 80)
print("전체 측정지표 통계")
print("=" * 80)

all_indicators = []
for inds in indicators.values():
    all_indicators.extend(inds)

all_means = []
all_stds = []
all_vars = []

for ind in all_indicators:
    if ind in df.columns:
        all_means.append(df[ind].mean())
        all_stds.append(df[ind].std())
        all_vars.append(df[ind].var())

print(f"평균의 평균: {np.mean(all_means):.2f}")
print(f"평균의 범위: [{np.min(all_means):.2f}, {np.max(all_means):.2f}]")
print(f"표준편차의 평균: {np.mean(all_stds):.2f}")
print(f"표준편차의 범위: [{np.min(all_stds):.2f}, {np.max(all_stds):.2f}]")
print(f"분산의 평균: {np.mean(all_vars):.2f}")
print(f"분산의 범위: [{np.min(all_vars):.2f}, {np.max(all_vars):.2f}]")

# 3. 개인별 측정모델 우도 추정 (간단한 버전)
print("\n" + "=" * 80)
print("측정모델 우도 크기 추정")
print("=" * 80)

# 첫 번째 개인 데이터
first_person = df[df['respondent_id'] == 1].iloc[0]

# 가정: LV ~ N(0, 1), zeta = 1.0, sigma_sq = 0.3 (이전 값)
lv = 0.0  # 평균값
zeta = 1.0
sigma_sq_old = 0.3
sigma_sq_new = 0.8

total_ll_old = 0.0
total_ll_new = 0.0

print("\n개인 1의 측정모델 우도 계산 (LV=0.0 가정):")
print("-" * 80)

for lv_name, inds in indicators.items():
    ll_old = 0.0
    ll_new = 0.0
    
    for ind in inds:
        if ind in first_person.index and not pd.isna(first_person[ind]):
            y_obs = first_person[ind]
            y_pred = zeta * lv
            residual = y_obs - y_pred
            
            # 정규분포 로그우도: log N(y | μ, σ²) = -0.5 * log(2π * σ²) - 0.5 * (y - μ)² / σ²
            ll_i_old = -0.5 * np.log(2 * np.pi * sigma_sq_old) - 0.5 * (residual ** 2) / sigma_sq_old
            ll_i_new = -0.5 * np.log(2 * np.pi * sigma_sq_new) - 0.5 * (residual ** 2) / sigma_sq_new
            
            ll_old += ll_i_old
            ll_new += ll_i_new
    
    total_ll_old += ll_old
    total_ll_new += ll_new
    
    print(f"  {lv_name:20s}: LL(σ²=0.3)={ll_old:8.2f}, LL(σ²=0.8)={ll_new:8.2f}, 차이={ll_new - ll_old:+8.2f}")

print("-" * 80)
print(f"  {'TOTAL':20s}: LL(σ²=0.3)={total_ll_old:8.2f}, LL(σ²=0.8)={total_ll_new:8.2f}, 차이={total_ll_new - total_ll_old:+8.2f}")

# 4. 326명 전체 추정
print("\n" + "=" * 80)
print("전체 326명 측정모델 우도 추정")
print("=" * 80)

total_ll_all_old = 0.0
total_ll_all_new = 0.0

for person_id in df['respondent_id'].unique():
    person_data = df[df['respondent_id'] == person_id].iloc[0]
    
    for lv_name, inds in indicators.items():
        for ind in inds:
            if ind in person_data.index and not pd.isna(person_data[ind]):
                y_obs = person_data[ind]
                y_pred = zeta * lv
                residual = y_obs - y_pred
                
                ll_i_old = -0.5 * np.log(2 * np.pi * sigma_sq_old) - 0.5 * (residual ** 2) / sigma_sq_old
                ll_i_new = -0.5 * np.log(2 * np.pi * sigma_sq_new) - 0.5 * (residual ** 2) / sigma_sq_new
                
                total_ll_all_old += ll_i_old
                total_ll_all_new += ll_i_new

print(f"전체 측정모델 LL (σ²=0.3): {total_ll_all_old:,.2f}")
print(f"전체 측정모델 LL (σ²=0.8): {total_ll_all_new:,.2f}")
print(f"차이: {total_ll_all_new - total_ll_all_old:+,.2f}")
print(f"개선율: {(total_ll_all_new - total_ll_all_old) / abs(total_ll_all_old) * 100:+.2f}%")

# 5. 측정모델 우도가 높은 이유 분석
print("\n" + "=" * 80)
print("측정모델 우도가 높은 이유 분석")
print("=" * 80)

print("\n1. 지표 개수:")
print(f"   - 총 지표 수: {len(all_indicators)}개")
print(f"   - 개인 수: {len(df['respondent_id'].unique())}명")
print(f"   - 총 관측치: {len(all_indicators) * len(df['respondent_id'].unique())} = {len(all_indicators)} × {len(df['respondent_id'].unique())}")

print("\n2. 선택모델 관측치:")
print(f"   - 선택 상황 수: 6개/인")
print(f"   - 개인 수: {len(df['respondent_id'].unique())}명")
print(f"   - 총 선택 관측치: {6 * len(df['respondent_id'].unique())} = 6 × {len(df['respondent_id'].unique())}")

print("\n3. 관측치 비율:")
print(f"   - 측정모델 / 선택모델 = {len(all_indicators) * len(df['respondent_id'].unique())} / {6 * len(df['respondent_id'].unique())} = {len(all_indicators) / 6:.1f}배")

print("\n4. 로그우도 크기 비교 (개인당 평균):")
print(f"   - 측정모델 LL/인 (σ²=0.3): {total_ll_all_old / len(df['respondent_id'].unique()):.2f}")
print(f"   - 측정모델 LL/인 (σ²=0.8): {total_ll_all_new / len(df['respondent_id'].unique()):.2f}")
print(f"   - 선택모델 LL/인 (로그 기준): 약 -30 (6개 선택 상황)")

print("\n5. 결론:")
print("   ✅ 측정모델 우도가 높은 주요 원인:")
print("   1) 지표 개수가 많음 (38개 vs 선택 상황 6개)")
print("   2) 각 지표마다 로그우도가 누적됨")
print("   3) σ²가 작을수록 로그우도 절대값이 커짐 (분모 효과)")
print("   4) 잔차가 클수록 로그우도 절대값이 커짐 (분자 효과)")

