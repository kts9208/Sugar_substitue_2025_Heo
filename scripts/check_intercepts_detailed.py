"""
절편 상세 확인
"""
import pandas as pd
import pickle
from pathlib import Path

print("="*80)
print("절편 상세 확인")
print("="*80)

# 1. 데이터 로드
data_path = Path('data/processed/iclv/integrated_data_cleaned.csv')
data = pd.read_csv(data_path)

# 2. 지표 목록
indicators = [f'q{i}' for i in range(6, 50)]
indicators = [q for q in indicators if q in data.columns]

# 3. 개인별 unique 데이터
unique_data = data.groupby('respondent_id')[indicators].first().reset_index()

print(f"\n데이터:")
print(f"  원본: {len(data)}행")
print(f"  Unique: {len(unique_data)}행")
print(f"  지표: {len(indicators)}개")

# 4. 지표 평균 계산
indicator_means = unique_data[indicators].mean()

print(f"\n지표 평균 (처음 20개):")
print(indicator_means.head(20))

# 5. CFA 결과에서 절편 로드
pkl_path = Path('results/sequential_stage_wise/cfa_results.pkl')
with open(pkl_path, 'rb') as f:
    cfa_results = pickle.load(f)

intercepts = cfa_results.get('intercepts')

if intercepts is not None and len(intercepts) > 0:
    print(f"\n{'='*80}")
    print(f"CFA 추정 절편 (처음 20개):")
    print(f"{'='*80}")
    print(intercepts[['lval', 'Estimate']].head(20))
    
    # 6. 비교
    print(f"\n{'='*80}")
    print(f"지표 평균 vs CFA 절편 비교")
    print(f"{'='*80}")
    
    print(f"\n{'지표':10s} {'데이터 평균':>12s} {'CFA 절편':>12s} {'차이':>12s}")
    print(f"{'-'*50}")
    
    for _, row in intercepts.head(20).iterrows():
        indicator = row['lval']
        cfa_intercept = row['Estimate']
        data_mean = indicator_means.get(indicator, None)
        
        if data_mean is not None:
            diff = abs(cfa_intercept - data_mean)
            print(f"{indicator:10s} {data_mean:12.6f} {cfa_intercept:12.6f} {diff:12.6f}")
    
    # 7. 통계
    print(f"\n{'='*80}")
    print(f"절편 통계")
    print(f"{'='*80}")
    
    print(f"\nCFA 절편:")
    print(f"  평균: {intercepts['Estimate'].mean():.6f}")
    print(f"  표준편차: {intercepts['Estimate'].std():.6f}")
    print(f"  최소: {intercepts['Estimate'].min():.6f}")
    print(f"  최대: {intercepts['Estimate'].max():.6f}")
    
    print(f"\n지표 평균:")
    print(f"  평균: {indicator_means.mean():.6f}")
    print(f"  표준편차: {indicator_means.std():.6f}")
    print(f"  최소: {indicator_means.min():.6f}")
    print(f"  최대: {indicator_means.max():.6f}")
    
    # 8. 절편이 너무 작은 문제
    print(f"\n{'='*80}")
    print(f"문제 진단")
    print(f"{'='*80}")
    
    if intercepts['Estimate'].mean() < 2.5:
        print(f"\n❌ 절편이 너무 작습니다!")
        print(f"  CFA 절편 평균: {intercepts['Estimate'].mean():.2f}")
        print(f"  지표 평균: {indicator_means.mean():.2f}")
        print(f"  차이: {abs(intercepts['Estimate'].mean() - indicator_means.mean()):.2f}")
        print(f"\n원인:")
        print(f"  ModelMeans가 잠재변수 평균을 추정하면서")
        print(f"  절편이 조정되었을 가능성이 있습니다.")
        print(f"\n해결 방안:")
        print(f"  1. 지표 평균을 절편으로 직접 사용")
        print(f"  2. ModelMeans 대신 Model + 수동 절편 계산")
    else:
        print(f"\n✅ 절편이 합리적입니다!")

else:
    print(f"\n❌ 절편이 없습니다!")

print(f"\n{'='*80}")

