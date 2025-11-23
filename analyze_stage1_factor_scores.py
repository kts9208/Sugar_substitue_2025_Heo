"""
1단계 요인점수 결과 분석 스크립트
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# 1단계 결과 파일 로드
stage1_path = Path("results/sequential_stage_wise/stage1_HC-PB_PB-PI_results.pkl")

print("=" * 70)
print("1단계 요인점수 결과 분석")
print("=" * 70)

with open(stage1_path, 'rb') as f:
    results = pickle.load(f)

print("\n[1] 결과 파일 구조")
print("-" * 70)
print("포함된 키:", list(results.keys()))

print("\n[2] 요인점수 정보")
print("-" * 70)
print("요인점수 변수:", list(results['factor_scores'].keys()))
print("\n각 변수별 shape:")
for k, v in results['factor_scores'].items():
    print(f"  {k}: {v.shape}")

print("\n[3] 원본 요인점수 존재 여부")
print("-" * 70)
has_original = 'original_factor_scores' in results
print(f"original_factor_scores 존재: {has_original}")

if has_original:
    print("\n원본 요인점수 변수:", list(results['original_factor_scores'].keys()))
    print("\n각 변수별 shape:")
    for k, v in results['original_factor_scores'].items():
        print(f"  {k}: {v.shape}")

print("\n[4] 요인점수 통계 (변환된 요인점수)")
print("-" * 70)
print(f"{'변수명':25s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
print("-" * 70)
for lv_name, scores in results['factor_scores'].items():
    print(f"{lv_name:25s} {np.mean(scores):12.6f} {np.std(scores):12.6f} {np.min(scores):12.6f} {np.max(scores):12.6f}")

if has_original:
    print("\n[5] 원본 요인점수 통계 (변환 전)")
    print("-" * 70)
    print(f"{'변수명':25s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 70)
    for lv_name, scores in results['original_factor_scores'].items():
        print(f"{lv_name:25s} {np.mean(scores):12.6f} {np.std(scores):12.6f} {np.min(scores):12.6f} {np.max(scores):12.6f}")

print("\n[6] 샘플 데이터 (첫 5개 관측치)")
print("-" * 70)
first_lv = list(results['factor_scores'].keys())[0]
scores = results['factor_scores'][first_lv]
print(f"{first_lv} (변환된 요인점수):")
print(f"  {scores[:5]}")

if has_original:
    orig_scores = results['original_factor_scores'][first_lv]
    print(f"\n{first_lv} (원본 요인점수):")
    print(f"  {orig_scores[:5]}")

print("\n[7] 변환 방법 확인")
print("-" * 70)
if has_original:
    # Z-score 변환 확인
    first_lv = list(results['factor_scores'].keys())[0]
    orig = results['original_factor_scores'][first_lv]
    transformed = results['factor_scores'][first_lv]
    
    # 수동 Z-score 계산
    manual_zscore = (orig - np.mean(orig)) / np.std(orig)
    
    print(f"변수: {first_lv}")
    print(f"  원본 평균: {np.mean(orig):.6f}")
    print(f"  원본 표준편차: {np.std(orig):.6f}")
    print(f"  변환 후 평균: {np.mean(transformed):.6f}")
    print(f"  변환 후 표준편차: {np.std(transformed):.6f}")
    print(f"  수동 Z-score와 일치: {np.allclose(manual_zscore, transformed)}")
else:
    print("원본 요인점수가 없어 변환 방법을 확인할 수 없습니다.")

print("\n[8] 2단계에서 사용되는 방식")
print("-" * 70)
print("✅ 2단계 추정 시:")
print("  1. 1단계 PKL 파일 로드")
print("  2. 'original_factor_scores' 존재 확인")
print("  3-A. 존재하면: 현재 설정(STANDARDIZATION_METHOD)에 맞게 재변환")
print("  3-B. 없으면: 저장된 'factor_scores'를 그대로 사용 (경고 메시지)")
print("\n✅ 장점:")
print("  - 1단계와 2단계에서 다른 변환 방법 사용 가능")
print("  - 원본 요인점수 보존으로 유연성 확보")

print("\n[9] CSV 파일 확인")
print("-" * 70)
csv_path = Path("results/sequential_stage_wise/stage1_HC-PB_PB-PI_results_factor_scores.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"CSV 파일 shape: {df.shape}")
    print(f"CSV 파일 컬럼: {list(df.columns)}")
    print(f"\n첫 3행:")
    print(df.head(3))
    
    # CSV와 PKL 일치 확인
    first_lv = 'health_concern'
    csv_scores = df[first_lv].values
    pkl_scores = results['factor_scores'][first_lv]
    print(f"\nCSV와 PKL 일치 확인 ({first_lv}):")
    print(f"  일치: {np.allclose(csv_scores, pkl_scores)}")
else:
    print("CSV 파일이 없습니다.")

print("\n" + "=" * 70)
print("분석 완료")
print("=" * 70)

