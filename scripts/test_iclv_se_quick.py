"""
ICLV 모델 표준오차 계산 빠른 테스트
- 소규모 샘플 (50명)로 빠르게 표준오차 계산 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import create_iclv_config, EstimationConfig
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice

print("=" * 70)
print("ICLV 표준오차 계산 빠른 테스트 (50명 샘플)")
print("=" * 70)

# 1. 데이터 로드
print("\n1. 데이터 로드 중...")
data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
df = pd.read_csv(data_path)
print(f"   전체 데이터 shape: {df.shape}")

# 2. 소규모 샘플 추출 (처음 50명)
print("\n2. 소규모 샘플 추출 중...")
unique_ids = df['respondent_id'].unique()
sample_ids = unique_ids[:50]  # 처음 50명만
df_sample = df[df['respondent_id'].isin(sample_ids)].copy()
print(f"   샘플 데이터 shape: {df_sample.shape}")
print(f"   샘플 개인 수: {df_sample['respondent_id'].nunique()}")

# 3. ICLV 설정
print("\n3. ICLV 설정...")
config = create_iclv_config(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    sociodemographics=['age_std', 'gender', 'income_std'],
    choice_attributes=['price', 'health_label'],
    optimizer='BFGS',
    max_iterations=100,  # 빠른 테스트를 위해 100회로 제한
    n_draws=50,  # Halton draws도 50개로 줄임
    individual_id_column='respondent_id'
)

# 표준오차 계산 활성화
config.estimation.calculate_se = True
config.estimation.use_parallel = False  # Windows에서 multiprocessing 문제 방지
config.estimation.use_analytic_gradient = True
print("   설정 완료")
print(f"   - 잠재변수: {config.measurement.latent_variable}")
print(f"   - 지표 수: {len(config.measurement.indicators)}")
print(f"   - 사회인구학적 변수: {len(config.structural.sociodemographics)}")
print(f"   - 선택 속성: {len(config.choice.choice_attributes)}")
print(f"   - Halton draws: {config.estimation.n_draws}")
print(f"   - 최대 반복: {config.estimation.max_iterations}")
print(f"   - 샘플 개인 수: {df_sample['respondent_id'].nunique()}")

# 4. 모델 생성
print("\n4. 모델 생성...")

# 측정모델
measurement_model = OrderedProbitMeasurement(config.measurement)
print("   - 측정모델 생성 완료")

# 구조모델
structural_model = LatentVariableRegression(config.structural)
print("   - 구조모델 생성 완료")

# 선택모델
choice_model = BinaryProbitChoice(config.choice)
print("   - 선택모델 생성 완료")

# 5. ICLV 모델 추정 실행
print("\n5. ICLV 모델 추정 실행...")
print("   (소규모 샘플 + BFGS + Analytic Gradient + 병렬처리)")
print("   ⏱️  예상 소요 시간: 5-10분")

estimator = SimultaneousEstimator(config)
results = estimator.estimate(
    data=df_sample,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_model=choice_model
)

# 6. 결과 확인
print("\n" + "=" * 70)
print("추정 결과")
print("=" * 70)

print(f"\n수렴 여부: {results.get('success', 'N/A')}")
print(f"메시지: {results.get('message', 'N/A')}")
print(f"반복 횟수: {results.get('n_iterations', 'N/A')}")
print(f"최종 로그우도: {results.get('log_likelihood', 'N/A'):.4f}")
print(f"AIC: {results.get('aic', 'N/A'):.2f}")
print(f"BIC: {results.get('bic', 'N/A'):.2f}")

# 7. 표준오차 확인
print("\n" + "=" * 70)
print("표준오차 계산 결과 확인")
print("=" * 70)

if 'standard_errors' in results:
    print("\n✅ 표준오차 계산 성공!")
    print(f"   표준오차 벡터 길이: {len(results['standard_errors'])}")
    print(f"   표준오차 범위: [{results['standard_errors'].min():.4f}, {results['standard_errors'].max():.4f}]")
else:
    print("\n❌ 표준오차 계산 실패!")

if 't_statistics' in results:
    print("\n✅ t-통계량 계산 성공!")
    print(f"   t-통계량 벡터 길이: {len(results['t_statistics'])}")
else:
    print("\n❌ t-통계량 계산 실패!")

if 'p_values' in results:
    print("\n✅ p-값 계산 성공!")
    print(f"   p-값 벡터 길이: {len(results['p_values'])}")
    print(f"   유의한 파라미터 (p<0.05): {(results['p_values'] < 0.05).sum()}개")
else:
    print("\n❌ p-값 계산 실패!")

if 'parameter_statistics' in results:
    print("\n✅ 파라미터별 통계량 구조화 성공!")
    print(f"   구조: {list(results['parameter_statistics'].keys())}")
else:
    print("\n❌ 파라미터별 통계량 구조화 실패!")

# 8. 샘플 파라미터 출력 (선택모델만)
if 'parameter_statistics' in results and 'choice' in results['parameter_statistics']:
    print("\n" + "=" * 70)
    print("선택모델 파라미터 (샘플)")
    print("=" * 70)
    
    choice_stats = results['parameter_statistics']['choice']
    
    # Intercept
    if 'intercept' in choice_stats:
        ic = choice_stats['intercept']
        print(f"\nβ_Intercept:")
        print(f"  Estimate: {ic['estimate']:.4f}")
        print(f"  Std. Err.: {ic['std_error']:.4f}")
        print(f"  t-stat: {ic['t_statistic']:.4f}")
        print(f"  p-value: {ic['p_value']:.4f}")
    
    # Beta
    if 'beta' in choice_stats:
        print(f"\nβ (속성 효과):")
        for attr, stats in choice_stats['beta'].items():
            print(f"  β_{attr}:")
            print(f"    Estimate: {stats['estimate']:.4f}")
            print(f"    Std. Err.: {stats['std_error']:.4f}")
            print(f"    t-stat: {stats['t_statistic']:.4f}")
            print(f"    p-value: {stats['p_value']:.4f}")
    
    # Lambda
    if 'lambda' in choice_stats:
        lam = choice_stats['lambda']
        print(f"\nλ (잠재변수 효과):")
        print(f"  Estimate: {lam['estimate']:.4f}")
        print(f"  Std. Err.: {lam['std_error']:.4f}")
        print(f"  t-stat: {lam['t_statistic']:.4f}")
        print(f"  p-value: {lam['p_value']:.4f}")

# 9. CSV 저장 테스트
print("\n" + "=" * 70)
print("CSV 저장 테스트")
print("=" * 70)

output_path = project_root / "results" / "iclv_se_quick_test.csv"

if 'parameter_statistics' in results:
    print("\n표준오차 포함하여 저장 중...")
    
    rows = []
    param_stats = results['parameter_statistics']
    
    # 측정모델
    if 'measurement' in param_stats:
        # Zeta
        if 'zeta' in param_stats['measurement']:
            for i, stats in param_stats['measurement']['zeta'].items():
                rows.append({
                    'Coefficient': f'ζ_{i+1}',
                    'Estimate': f"{stats['estimate']:.4f}",
                    'Std. Err.': f"{stats['std_error']:.4f}",
                    'P. Value': f"{stats['p_value']:.4f}" if stats['p_value'] >= 0.001 else "<0.001"
                })
    
    # 구조모델
    if 'structural' in param_stats:
        if 'gamma' in param_stats['structural']:
            for var, stats in param_stats['structural']['gamma'].items():
                rows.append({
                    'Coefficient': f'γ_{var}',
                    'Estimate': f"{stats['estimate']:.4f}",
                    'Std. Err.': f"{stats['std_error']:.4f}",
                    'P. Value': f"{stats['p_value']:.4f}" if stats['p_value'] >= 0.001 else "<0.001"
                })
    
    # 선택모델
    if 'choice' in param_stats:
        # Intercept
        if 'intercept' in param_stats['choice']:
            stats = param_stats['choice']['intercept']
            rows.append({
                'Coefficient': 'β_Intercept',
                'Estimate': f"{stats['estimate']:.4f}",
                'Std. Err.': f"{stats['std_error']:.4f}",
                'P. Value': f"{stats['p_value']:.4f}" if stats['p_value'] >= 0.001 else "<0.001"
            })
        
        # Beta
        if 'beta' in param_stats['choice']:
            for attr, stats in param_stats['choice']['beta'].items():
                rows.append({
                    'Coefficient': f'β_{attr}',
                    'Estimate': f"{stats['estimate']:.4f}",
                    'Std. Err.': f"{stats['std_error']:.4f}",
                    'P. Value': f"{stats['p_value']:.4f}" if stats['p_value'] >= 0.001 else "<0.001"
                })
        
        # Lambda
        if 'lambda' in param_stats['choice']:
            stats = param_stats['choice']['lambda']
            rows.append({
                'Coefficient': 'λ',
                'Estimate': f"{stats['estimate']:.4f}",
                'Std. Err.': f"{stats['std_error']:.4f}",
                'P. Value': f"{stats['p_value']:.4f}" if stats['p_value'] >= 0.001 else "<0.001"
            })
    
    # DataFrame 생성 및 저장
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 결과 저장 완료: {output_path}")
    print(f"   총 {len(rows)}개 파라미터")
    
else:
    print("\n❌ parameter_statistics가 없어 저장 불가")

print("\n" + "=" * 70)
print("빠른 테스트 완료!")
print("=" * 70)

