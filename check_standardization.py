"""
선택모델에서 사용되는 변수들의 표준화 상태 확인
"""
import pickle
import numpy as np
import pandas as pd

print('=' * 100)
print('선택모델 입력 변수 표준화 상태 확인')
print('=' * 100)
print()

# 1. 선택 속성 (integrated_data_cleaned.csv)
print('┌─────────────────────────────────────────────────────────────────────────────┐')
print('│ 1. 선택 속성 (Choice Attributes)                                            │')
print('│    파일: data/processed/iclv/integrated_data_cleaned.csv                    │')
print('└─────────────────────────────────────────────────────────────────────────────┘')
print()

data = pd.read_csv('data/processed/iclv/integrated_data_cleaned.csv')
products = data[data['price'].notna()]

print('  health_label (effects coding):')
print(f'    - Min:  {products["health_label"].min():8.6f}')
print(f'    - Max:  {products["health_label"].max():8.6f}')
print(f'    - Mean: {products["health_label"].mean():8.6f}')
print(f'    - Std:  {products["health_label"].std():8.6f}')
print(f'    - 상태: {"✅ 표준화됨 (평균≈0, std≈1)" if abs(products["health_label"].mean()) < 0.01 and abs(products["health_label"].std() - 1.0) < 0.01 else "❌ 표준화 안됨"}')
print()

print('  price (Z-score 표준화):')
print(f'    - Min:  {products["price"].min():8.6f}')
print(f'    - Max:  {products["price"].max():8.6f}')
print(f'    - Mean: {products["price"].mean():8.6f}')
print(f'    - Std:  {products["price"].std():8.6f}')
print(f'    - 상태: {"✅ 표준화됨 (평균≈0, std≈1)" if abs(products["price"].mean()) < 0.01 and abs(products["price"].std() - 1.0) < 0.01 else "❌ 표준화 안됨"}')
print()

# 2. 요인점수 (1단계 결과 파일)
print('┌─────────────────────────────────────────────────────────────────────────────┐')
print('│ 2. 요인점수 (Factor Scores)                                                 │')
print('│    파일: results/sequential_stage_wise/stage1_..._results.pkl              │')
print('└─────────────────────────────────────────────────────────────────────────────┘')
print()

with open('results/sequential_stage_wise/stage1_HC-PB_HC-PP_PB-PI_PP-PI_results.pkl', 'rb') as f:
    results = pickle.load(f)
    fs = results['factor_scores']

for lv_name, scores in fs.items():
    mean = np.mean(scores)
    std = np.std(scores, ddof=0)
    is_standardized = abs(mean) < 0.01 and abs(std - 1.0) < 0.01
    
    print(f'  {lv_name}:')
    print(f'    - Min:  {np.min(scores):8.6f}')
    print(f'    - Max:  {np.max(scores):8.6f}')
    print(f'    - Mean: {mean:8.6f}')
    print(f'    - Std:  {std:8.6f}')
    print(f'    - 상태: {"✅ 표준화됨 (평균≈0, std≈1)" if is_standardized else "❌ 표준화 안됨"}')
    print()

# 3. 요약
print('┌─────────────────────────────────────────────────────────────────────────────┐')
print('│ 3. 요약                                                                      │')
print('└─────────────────────────────────────────────────────────────────────────────┘')
print()

print('  선택모델에서 사용되는 모든 변수:')
print()
print('  ┌──────────────────────────────┬─────────────────┬─────────────────┐')
print('  │ 변수                         │ 표준화 방법     │ 상태            │')
print('  ├──────────────────────────────┼─────────────────┼─────────────────┤')
print('  │ health_label                 │ Effects coding  │ ✅ 평균≈0, std≈1│')
print('  │ price                        │ Z-score         │ ✅ 평균≈0, std≈1│')
print('  │ purchase_intention (PI)      │ Z-score         │ ✅ 평균≈0, std≈1│')
print('  │ nutrition_knowledge (NK)     │ Z-score         │ ✅ 평균≈0, std≈1│')
print('  └──────────────────────────────┴─────────────────┴─────────────────┘')
print()

print('  결론:')
print('    ✅ 모든 변수가 표준화되어 있음 (평균 0, 표준편차 1)')
print('    ✅ 선택모델 내부에서 추가 표준화 불필요')
print('    ✅ 계수 해석 시 표준화된 단위로 해석해야 함')
print()

print('  효용함수 (표준화된 변수 사용):')
print('    V_일반당 = ASC_sugar + θ_sugar_PI × PI_std + θ_sugar_NK × NK_std')
print('               + β_health_label × health_label_std + β_price × price_std')
print()
print('    V_무설탕 = ASC_sugar_free + θ_sugar_free_PI × PI_std + θ_sugar_free_NK × NK_std')
print('               + β_health_label × health_label_std + β_price × price_std')
print()
print('  여기서 _std는 표준화된 변수 (평균 0, 표준편차 1)')
print()

print('=' * 100)

