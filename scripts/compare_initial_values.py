"""
초기값 비교 분석 스크립트

비교 대상:
1. ζ = 1.0, σ² = 0.8 (현재)
2. ζ = 0.5, σ² = 0.29 (제안)
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

all_indicators = []
for inds in indicators.values():
    all_indicators.extend(inds)

print("=" * 80)
print("초기값 비교 분석")
print("=" * 80)

# 설정 1: ζ = 1.0, σ² = 0.8
zeta_1 = 1.0
sigma_sq_1 = 0.8

# 설정 2: ζ = 0.5, σ² = 0.29
zeta_2 = 0.5
sigma_sq_2 = 0.29

# 설정 3: ζ = 0.7, σ² = 0.5 (중간값)
zeta_3 = 0.7
sigma_sq_3 = 0.5

print("\n[설정 비교]")
print("-" * 80)
print(f"설정 1: ζ = {zeta_1:.1f}, σ² = {sigma_sq_1:.2f}, Var(Y) = {zeta_1**2 + sigma_sq_1:.2f}")
print(f"설정 2: ζ = {zeta_2:.1f}, σ² = {sigma_sq_2:.2f}, Var(Y) = {zeta_2**2 + sigma_sq_2:.2f}")
print(f"설정 3: ζ = {zeta_3:.1f}, σ² = {sigma_sq_3:.2f}, Var(Y) = {zeta_3**2 + sigma_sq_3:.2f}")
print(f"실제 데이터: Var(Y) 평균 = 0.79")

# LV 초기값 (0.0 가정)
lv = 0.0

# 전체 326명 우도 계산
total_ll_1 = 0.0
total_ll_2 = 0.0
total_ll_3 = 0.0

for person_id in df['respondent_id'].unique():
    person_data = df[df['respondent_id'] == person_id].iloc[0]
    
    for ind in all_indicators:
        if ind in person_data.index and not pd.isna(person_data[ind]):
            y_obs = person_data[ind]
            
            # 설정 1
            y_pred_1 = zeta_1 * lv
            residual_1 = y_obs - y_pred_1
            ll_i_1 = -0.5 * np.log(2 * np.pi * sigma_sq_1) - 0.5 * (residual_1 ** 2) / sigma_sq_1
            total_ll_1 += ll_i_1
            
            # 설정 2
            y_pred_2 = zeta_2 * lv
            residual_2 = y_obs - y_pred_2
            ll_i_2 = -0.5 * np.log(2 * np.pi * sigma_sq_2) - 0.5 * (residual_2 ** 2) / sigma_sq_2
            total_ll_2 += ll_i_2
            
            # 설정 3
            y_pred_3 = zeta_3 * lv
            residual_3 = y_obs - y_pred_3
            ll_i_3 = -0.5 * np.log(2 * np.pi * sigma_sq_3) - 0.5 * (residual_3 ** 2) / sigma_sq_3
            total_ll_3 += ll_i_3

print("\n[전체 측정모델 우도 (LV=0 가정)]")
print("-" * 80)
print(f"설정 1 (ζ=1.0, σ²=0.8):  LL = {total_ll_1:,.2f}")
print(f"설정 2 (ζ=0.5, σ²=0.29): LL = {total_ll_2:,.2f}")
print(f"설정 3 (ζ=0.7, σ²=0.5):  LL = {total_ll_3:,.2f}")

print("\n[개인당 평균 우도]")
print("-" * 80)
n_persons = len(df['respondent_id'].unique())
print(f"설정 1: LL/인 = {total_ll_1 / n_persons:.2f}")
print(f"설정 2: LL/인 = {total_ll_2 / n_persons:.2f}")
print(f"설정 3: LL/인 = {total_ll_3 / n_persons:.2f}")

print("\n[설정 2 vs 설정 1 비교]")
print("-" * 80)
diff_2_1 = total_ll_2 - total_ll_1
print(f"LL 차이: {diff_2_1:+,.2f}")
print(f"개선율: {diff_2_1 / abs(total_ll_1) * 100:+.2f}%")
if diff_2_1 < 0:
    print("⚠️  설정 2가 설정 1보다 나쁩니다!")
else:
    print("✅ 설정 2가 설정 1보다 좋습니다!")

print("\n[설정 3 vs 설정 1 비교]")
print("-" * 80)
diff_3_1 = total_ll_3 - total_ll_1
print(f"LL 차이: {diff_3_1:+,.2f}")
print(f"개선율: {diff_3_1 / abs(total_ll_1) * 100:+.2f}%")
if diff_3_1 < 0:
    print("⚠️  설정 3이 설정 1보다 나쁩니다!")
else:
    print("✅ 설정 3이 설정 1보다 좋습니다!")

# Gradient 추정
print("\n" + "=" * 80)
print("Gradient 크기 추정 (개인 1, 지표 1개)")
print("=" * 80)

first_person = df[df['respondent_id'] == 1].iloc[0]
y_obs = first_person['q6']

print(f"\n관측값: y = {y_obs:.2f}")
print(f"잠재변수: LV = {lv:.2f}")

for i, (zeta, sigma_sq, name) in enumerate([
    (zeta_1, sigma_sq_1, "설정 1"),
    (zeta_2, sigma_sq_2, "설정 2"),
    (zeta_3, sigma_sq_3, "설정 3")
], 1):
    y_pred = zeta * lv
    residual = y_obs - y_pred
    
    # ∂LL/∂σ² = -0.5 / σ² + 0.5 × residual² / σ²²
    grad_sigma = -0.5 / sigma_sq + 0.5 * (residual ** 2) / (sigma_sq ** 2)
    
    # ∂LL/∂ζ = residual × LV / σ²
    grad_zeta = residual * lv / sigma_sq
    
    print(f"\n{name} (ζ={zeta:.1f}, σ²={sigma_sq:.2f}):")
    print(f"  예측값: {y_pred:.2f}")
    print(f"  잔차: {residual:.2f}")
    print(f"  ∂LL/∂σ² = {grad_sigma:+.2f}")
    print(f"  ∂LL/∂ζ = {grad_zeta:+.2f}")

# 전체 gradient 추정
print("\n" + "=" * 80)
print("전체 Gradient 크기 추정 (326명 × 38개 지표)")
print("=" * 80)

total_grad_sigma_1 = 0.0
total_grad_sigma_2 = 0.0
total_grad_sigma_3 = 0.0

for person_id in df['respondent_id'].unique():
    person_data = df[df['respondent_id'] == person_id].iloc[0]
    
    for ind in all_indicators:
        if ind in person_data.index and not pd.isna(person_data[ind]):
            y_obs = person_data[ind]
            
            # 설정 1
            residual_1 = y_obs - zeta_1 * lv
            grad_1 = -0.5 / sigma_sq_1 + 0.5 * (residual_1 ** 2) / (sigma_sq_1 ** 2)
            total_grad_sigma_1 += grad_1
            
            # 설정 2
            residual_2 = y_obs - zeta_2 * lv
            grad_2 = -0.5 / sigma_sq_2 + 0.5 * (residual_2 ** 2) / (sigma_sq_2 ** 2)
            total_grad_sigma_2 += grad_2
            
            # 설정 3
            residual_3 = y_obs - zeta_3 * lv
            grad_3 = -0.5 / sigma_sq_3 + 0.5 * (residual_3 ** 2) / (sigma_sq_3 ** 2)
            total_grad_sigma_3 += grad_3

print(f"\n설정 1 (ζ=1.0, σ²=0.8):  ∂LL/∂σ² (total) = {total_grad_sigma_1:+,.2f}")
print(f"설정 2 (ζ=0.5, σ²=0.29): ∂LL/∂σ² (total) = {total_grad_sigma_2:+,.2f}")
print(f"설정 3 (ζ=0.7, σ²=0.5):  ∂LL/∂σ² (total) = {total_grad_sigma_3:+,.2f}")

print("\n[Gradient 비교]")
print("-" * 80)
print(f"설정 2 / 설정 1 = {abs(total_grad_sigma_2) / abs(total_grad_sigma_1):.2f}배")
print(f"설정 3 / 설정 1 = {abs(total_grad_sigma_3) / abs(total_grad_sigma_1):.2f}배")

if abs(total_grad_sigma_2) > abs(total_grad_sigma_1):
    print("\n⚠️  설정 2의 gradient가 설정 1보다 큽니다!")
    print("    → 수렴이 더 어려울 수 있습니다.")
else:
    print("\n✅ 설정 2의 gradient가 설정 1보다 작습니다!")
    print("    → 수렴이 더 쉬울 수 있습니다.")

print("\n" + "=" * 80)
print("결론 및 권장사항")
print("=" * 80)

print("\n1. 초기 우도 (높을수록 좋음):")
if total_ll_2 > total_ll_1:
    print("   ✅ 설정 2가 우수")
else:
    print("   ❌ 설정 1이 우수")

print("\n2. Gradient 크기 (작을수록 좋음):")
if abs(total_grad_sigma_2) < abs(total_grad_sigma_1):
    print("   ✅ 설정 2가 우수")
else:
    print("   ❌ 설정 1이 우수")

print("\n3. 분산 일치도 (실제 0.79에 가까울수록 좋음):")
var_1 = zeta_1**2 + sigma_sq_1
var_2 = zeta_2**2 + sigma_sq_2
var_3 = zeta_3**2 + sigma_sq_3
diff_1 = abs(var_1 - 0.79)
diff_2 = abs(var_2 - 0.79)
diff_3 = abs(var_3 - 0.79)

print(f"   설정 1: |{var_1:.2f} - 0.79| = {diff_1:.2f}")
print(f"   설정 2: |{var_2:.2f} - 0.79| = {diff_2:.2f}")
print(f"   설정 3: |{var_3:.2f} - 0.79| = {diff_3:.2f}")

if diff_2 < diff_1:
    print("   ✅ 설정 2가 우수")
else:
    print("   ❌ 설정 1이 우수")

print("\n4. 종합 평가:")
score_1 = 0
score_2 = 0
score_3 = 0

if total_ll_1 >= total_ll_2:
    score_1 += 1
else:
    score_2 += 1

if abs(total_grad_sigma_1) <= abs(total_grad_sigma_2):
    score_1 += 1
else:
    score_2 += 1

if diff_1 <= diff_2:
    score_1 += 1
else:
    score_2 += 1

print(f"   설정 1 점수: {score_1}/3")
print(f"   설정 2 점수: {score_2}/3")

if score_1 > score_2:
    print("\n   ✅ 권장: 설정 1 (ζ=1.0, σ²=0.8)")
elif score_2 > score_1:
    print("\n   ✅ 권장: 설정 2 (ζ=0.5, σ²=0.29)")
else:
    print("\n   ⚖️  두 설정 모두 장단점이 있음")
    print("   💡 설정 3 (ζ=0.7, σ²=0.5)을 고려해보세요")

