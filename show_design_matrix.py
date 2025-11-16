"""
Design Matrix 출력 스크립트
"""
import pandas as pd

design = pd.read_csv('data/processed/dce/design_matrix.csv')

print('=' * 100)
print('선택실험 Design Matrix (6개 선택 세트)')
print('=' * 100)
print()

for cs in range(1, 7):
    print(f'【선택 세트 {cs}】')
    cs_data = design[design['choice_set'] == cs]
    
    for _, row in cs_data.iterrows():
        if pd.notna(row['sugar_content']):
            health = '있음' if row['health_label'] == 1.0 else '없음'
            print(f"  {row['alternative_name']:8s}: {row['sugar_content']:4s}, 건강라벨={health:4s}, ₩{row['price']:,.0f}")
        else:
            print(f"  {row['alternative_name']:8s}: (구매하지 않음)")
    print()

print('=' * 100)
print('속성 균형 확인')
print('=' * 100)
print()

# 제품만 (구매안함 제외)
products = design[design['sugar_content'].notna()]

print('1. Sugar Content (설탕 함량):')
print(f"   - 무설탕: {(products['sugar_content'] == '무설탕').sum()}개")
print(f"   - 일반당: {(products['sugar_content'] == '알반당').sum()}개")
print()

print('2. Health Label (건강 라벨):')
print(f"   - 있음 (1): {(products['health_label'] == 1.0).sum()}개")
print(f"   - 없음 (0): {(products['health_label'] == 0.0).sum()}개")
print()

print('3. Price (가격):')
for price in sorted(products['price'].unique()):
    count = (products['price'] == price).sum()
    print(f"   - ₩{price:,.0f}: {count}개")
print()

print('4. Sugar Content × Health Label:')
crosstab = pd.crosstab(products['sugar_content'], products['health_label'])
print(crosstab)
print()

print('5. 건강 라벨이 있는 제품의 설탕 함량:')
hl_products = products[products['health_label'] == 1.0]
print(f"   - 무설탕: {(hl_products['sugar_content'] == '무설탕').sum()}개 ({(hl_products['sugar_content'] == '무설탕').sum()/len(hl_products)*100:.1f}%)")
print(f"   - 일반당: {(hl_products['sugar_content'] == '알반당').sum()}개 ({(hl_products['sugar_content'] == '알반당').sum()/len(hl_products)*100:.1f}%)")
print()

print('=' * 100)

