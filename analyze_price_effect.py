"""
가격 효과 분석
"""
import pandas as pd
import numpy as np

data = pd.read_csv('data/processed/iclv/integrated_data_cleaned.csv')
products = data[data['price'].notna()]

print('=' * 80)
print('가격 속성 분석')
print('=' * 80)
print()

print('1. 원본 가격 (effects coding 전):')
prices_original = [2000, 2500, 3000]
print(f'   - 최소: ₩{min(prices_original):,}')
print(f'   - 최대: ₩{max(prices_original):,}')
print(f'   - 차이: ₩{max(prices_original) - min(prices_original):,} ({(max(prices_original) - min(prices_original))/min(prices_original)*100:.1f}%)')
print()

print('2. Effects coding 후 가격:')
print(f'   - 최소: {products["price"].min():.6f}')
print(f'   - 최대: {products["price"].max():.6f}')
print(f'   - 차이: {products["price"].max() - products["price"].min():.6f}')
print(f'   - 표준편차: {products["price"].std():.6f}')
print()

print('3. 가격 분포:')
print(products['price'].value_counts().sort_index())
print()

print('4. 선택된 대안의 가격 분포:')
chosen = data[data['choice'] == 1]
chosen_products = chosen[chosen['price'].notna()]
print(f'   총 선택: {len(chosen_products)}개')
for price in sorted(chosen_products['price'].unique()):
    count = (chosen_products['price'] == price).sum()
    pct = count / len(chosen_products) * 100
    print(f'   - {price:8.6f}: {count:4d}개 ({pct:5.1f}%)')
print()

print('5. 가격별 선택 확률:')
for price in sorted(products['price'].unique()):
    price_data = data[data['price'] == price]
    chosen_count = (price_data['choice'] == 1).sum()
    total_count = len(price_data)
    choice_prob = chosen_count / total_count * 100
    print(f'   - {price:8.6f}: {chosen_count}/{total_count} = {choice_prob:.1f}%')
print()

print('6. 가격 효과 검증 (카이제곱 검정):')
from scipy.stats import chi2_contingency

# 가격별 선택/비선택 교차표
price_choice_table = []
for price in sorted(products['price'].unique()):
    price_data = data[data['price'] == price]
    chosen = (price_data['choice'] == 1).sum()
    not_chosen = (price_data['choice'] == 0).sum()
    price_choice_table.append([chosen, not_chosen])

chi2, p_value, dof, expected = chi2_contingency(price_choice_table)
print(f'   - 카이제곱 통계량: {chi2:.4f}')
print(f'   - p-value: {p_value:.4f}')
print(f'   - 자유도: {dof}')
if p_value < 0.05:
    print(f'   - 결론: 가격이 선택에 유의한 영향을 미침 (p<0.05)')
else:
    print(f'   - 결론: 가격이 선택에 유의한 영향을 미치지 않음 (p≥0.05)')
print()

print('7. 추정된 가격 계수 해석:')
beta_price = -0.5677
print(f'   - 추정 계수: {beta_price:.4f}')
print(f'   - 표준오차: 0.3435')
print(f'   - p-value: 0.0984')
print()
print('   해석:')
print(f'   - 가격이 1 표준편차 증가하면 효용이 {abs(beta_price):.4f} 감소')
print(f'   - 가격 범위: -1.225 ~ 1.225 (2.45 표준편차)')
print(f'   - 최저가(₩2,000) vs 최고가(₩3,000) 효용 차이: {abs(beta_price) * 2.45:.4f}')
print()

print('8. 가격 민감도가 낮은 이유 (가설):')
print('   ① 가격 차이가 작음 (₩1,000, 50%)')
print('   ② 절대 가격이 낮음 (₩2,000~3,000)')
print('   ③ 건강 관심도가 높은 표본 → 가격보다 건강 속성 중시')
print('   ④ 무설탕/건강라벨 효과가 가격 효과를 압도')
print()

print('=' * 80)

