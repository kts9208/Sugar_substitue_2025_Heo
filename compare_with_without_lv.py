"""
잠재변수 효과 유무에 따른 선택모델 비교

1. 잠재변수 효과 없음 (선택모델만)
2. 잠재변수 효과 포함 (ICLV 모델)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 100)
print("선택모델 비교: 잠재변수 효과 유무")
print("=" * 100)
print()

# 1. 잠재변수 효과 없음 (선택모델만)
print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ 모델 1: 선택모델만 (잠재변수 효과 제외)                                     │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()

stats_path = Path("results/choice_model_only/parameter_statistics.csv")
summary_path = Path("results/choice_model_only/model_summary.txt")

if stats_path.exists() and summary_path.exists():
    # CSV에서 파라미터 로드
    stats_df = pd.read_csv(stats_path)

    # 요약 파일에서 적합도 로드
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print("효용함수:")
    print("  V_일반당 = ASC_sugar + β_health_label × health_label + β_price × price")
    print("  V_무설탕 = ASC_sugar_free + β_health_label × health_label + β_price × price")
    print()

    print("파라미터 추정치:")
    print()

    # 파라미터 출력
    for _, row in stats_df.iterrows():
        param = row['parameter']
        est = row['estimate']
        se = row['se']
        t = row['t']
        p = row['p']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {param:30s}: {est:8.4f} (SE={se:6.4f}, t={t:6.3f}, p={p:7.5f}) {sig}")

    # 모델 적합도
    print()
    if 'LL:' in content:
        ll_line = [line for line in content.split('\n') if 'LL:' in line][0]
        ll = float(ll_line.split(':')[1].strip())
        print(f"  로그우도 (LL): {ll:.2f}")
    if 'AIC:' in content:
        aic_line = [line for line in content.split('\n') if 'AIC:' in line][0]
        aic = float(aic_line.split(':')[1].strip())
        print(f"  AIC:           {aic:.2f}")
    if 'BIC:' in content:
        bic_line = [line for line in content.split('\n') if 'BIC:' in line][0]
        bic = float(bic_line.split(':')[1].strip())
        print(f"  BIC:           {bic:.2f}")
else:
    print("  ❌ 결과 파일이 없습니다. 먼저 examples/choice_model_only.py를 실행하세요.")

print()
print()

# 2. 잠재변수 효과 포함 (ICLV 모델)
print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ 모델 2: ICLV 모델 (잠재변수 효과 포함)                                      │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()

stats2_path = Path("results/sequential_stage_wise/stage2_parameter_statistics.csv")
if stats2_path.exists():
    stats2 = pd.read_csv(stats2_path, index_col=0)
    
    print("효용함수:")
    print("  V_일반당 = ASC_sugar + θ_sugar_PI × PI + θ_sugar_NK × NK")
    print("             + β_health_label × health_label + β_price × price")
    print("  V_무설탕 = ASC_sugar_free + θ_sugar_free_PI × PI + θ_sugar_free_NK × NK")
    print("             + β_health_label × health_label + β_price × price")
    print()
    
    print("파라미터 추정치:")
    print()
    
    # ASC
    for param in ['asc_sugar', 'asc_sugar_free']:
        if param in stats2.index:
            row = stats2.loc[param]
            est = row['Estimate']
            se = row['Std.Error']
            t = row['t-value']
            p = row['p-value']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {param:30s}: {est:8.4f} (SE={se:6.4f}, t={t:6.3f}, p={p:7.5f}) {sig}")
    
    # 선택 속성
    for param in ['beta_health_label', 'beta_price']:
        if param in stats2.index:
            row = stats2.loc[param]
            est = row['Estimate']
            se = row['Std.Error']
            t = row['t-value']
            p = row['p-value']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {param:30s}: {est:8.4f} (SE={se:6.4f}, t={t:6.3f}, p={p:7.5f}) {sig}")
    
    # 잠재변수 효과
    print()
    for param in ['theta_sugar_purchase_intention', 'theta_sugar_nutrition_knowledge',
                  'theta_sugar_free_purchase_intention', 'theta_sugar_free_nutrition_knowledge']:
        if param in stats2.index:
            row = stats2.loc[param]
            est = row['Estimate']
            se = row['Std.Error']
            t = row['t-value']
            p = row['p-value']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {param:30s}: {est:8.4f} (SE={se:6.4f}, t={t:6.3f}, p={p:7.5f}) {sig}")
    
    # 모델 적합도 (로그 파일에서 추출)
    # 간단히 하기 위해 생략
else:
    print("  ❌ 결과 파일이 없습니다.")

print()
print("=" * 100)

