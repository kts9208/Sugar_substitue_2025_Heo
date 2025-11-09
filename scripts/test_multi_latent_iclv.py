"""
다중 잠재변수 ICLV 모델 테스트 스크립트

5개 잠재변수 완전 동시추정:
1. 건강관심도 (health_concern)
2. 건강유익성 (perceived_benefit)
3. 가격수준 (perceived_price)
4. 영양지식 (nutrition_knowledge)
5. 구매의도 (purchase_intention) - 내생변수

Author: Sugar Substitute Research Team
Date: 2025-11-09
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import time
import multiprocessing

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    create_default_multi_lv_config
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_estimator import (
    MultiLatentSimultaneousEstimator
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import (
    MultiLatentMeasurement
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import (
    MultiLatentStructural
)
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import (
    BinaryProbitChoice
)


def main():
    print("="*70)
    print("다중 잠재변수 ICLV 동시추정 (5개 잠재변수)")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    print(f"   데이터 shape: {data.shape}")
    n_individuals = data['respondent_id'].nunique()
    print(f"   전체 개인 수: {n_individuals}")
    
    # 2. 설정
    print("\n2. ICLV 설정...")
    
    # CPU 정보
    n_cpus = multiprocessing.cpu_count()
    print(f"   사용 가능한 CPU 코어: {n_cpus}개")
    
    use_parallel = True
    n_cores = max(1, n_cpus - 1)
    
    config = create_default_multi_lv_config(
        n_draws=100,
        max_iterations=1000,
        use_parallel=use_parallel,
        n_cores=n_cores
    )
    
    print("   설정 완료")
    print(f"   - 잠재변수: 5개 (건강관심도, 건강유익성, 가격수준, 영양지식, 구매의도)")
    print(f"   - 측정모델 지표: 38개")
    print(f"   - 구조모델 공변량: 4개 (age, gender, income, education)")
    print(f"   - 선택모델 속성: 3개 (sugar_free, health_label, price)")
    print(f"   - Halton draws: {config.estimation.n_draws}")
    print(f"   - 최대 반복: {config.estimation.max_iterations}")
    print(f"   - 병렬처리: {'✅ 활성화' if use_parallel else '❌ 비활성화'}")
    if use_parallel:
        print(f"   - 사용 코어: {n_cores}/{n_cpus}개 ({n_cores/n_cpus*100:.1f}%)")
        print(f"   - 예상 속도 향상: ~{n_cores}배")
    
    # 3. 모델 생성
    print("\n3. 모델 생성...")
    
    # 측정모델
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    print("   - 측정모델 생성 완료 (5개 잠재변수)")
    
    # 구조모델
    structural_model = MultiLatentStructural(config.structural)
    print("   - 구조모델 생성 완료 (4 외생 → 1 내생)")
    
    # 선택모델
    choice_model = BinaryProbitChoice(config.choice)
    print("   - 선택모델 생성 완료")
    
    # 4. ICLV 동시추정 실행
    print("\n4. ICLV 동시추정 실행...")
    print("   (5개 잠재변수 + BFGS + 병렬처리)")
    print("\n   ⚠️  다중 잠재변수 추정은 시간이 오래 걸릴 수 있습니다...")
    
    # 로그 파일 경로 설정
    log_file = project_root / 'results' / 'multi_lv_iclv_estimation_log.txt'
    print(f"   로그 파일: {log_file}")
    
    start_time = time.time()
    
    estimator = MultiLatentSimultaneousEstimator(config, data)
    results = estimator.estimate()
    
    elapsed_time = time.time() - start_time
    
    # 5. 결과 출력
    print("\n" + "="*70)
    print("추정 결과 (다중 잠재변수)")
    print("="*70)
    print(f"\n추정 시간: {elapsed_time/60:.2f}분 ({elapsed_time:.1f}초)")
    
    if 'convergence' in results:
        print(f"수렴 여부: {results['convergence']['success']}")
        print(f"반복 횟수: {results['convergence']['n_iterations']}")
    else:
        print(f"수렴 여부: 미확인")
        print(f"반복 횟수: 미확인")
    
    print(f"최종 로그우도: {results['log_likelihood']:.4f}")
    
    print("\n파라미터 추정값:")
    print("\n[측정모델]")
    for lv_name in ['health_concern', 'perceived_benefit', 'perceived_price', 
                    'nutrition_knowledge', 'purchase_intention']:
        if lv_name in results['parameters']['measurement']:
            params = results['parameters']['measurement'][lv_name]
            print(f"  {lv_name}:")
            print(f"    zeta: {params['zeta'][:3]}... (총 {len(params['zeta'])}개)")
    
    print("\n[구조모델]")
    print(f"  gamma_lv (잠재변수 계수): {results['parameters']['structural']['gamma_lv']}")
    print(f"  gamma_x (공변량 계수): {results['parameters']['structural']['gamma_x']}")
    
    print("\n[선택모델]")
    print(f"  intercept: {results['parameters']['choice']['intercept']:.4f}")
    print(f"  beta: {results['parameters']['choice']['beta']}")
    print(f"  lambda: {results['parameters']['choice']['lambda']:.4f}")
    
    # 6. 결과 저장
    output_dir = project_root / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # 파라미터를 DataFrame으로 변환
    param_list = []
    
    # 측정모델 파라미터
    for lv_name in ['health_concern', 'perceived_benefit', 'perceived_price', 
                    'nutrition_knowledge', 'purchase_intention']:
        if lv_name in results['parameters']['measurement']:
            params = results['parameters']['measurement'][lv_name]
            
            # zeta
            for i, val in enumerate(params['zeta']):
                param_list.append({
                    'Model': 'Measurement',
                    'Latent_Variable': lv_name,
                    'Parameter': f'ζ_{i+1}',
                    'Estimate': val
                })
            
            # tau
            tau = params['tau']
            for i in range(tau.shape[0]):
                for j in range(tau.shape[1]):
                    param_list.append({
                        'Model': 'Measurement',
                        'Latent_Variable': lv_name,
                        'Parameter': f'τ_{i+1},{j+1}',
                        'Estimate': tau[i, j]
                    })
    
    # 구조모델 파라미터
    gamma_lv = results['parameters']['structural']['gamma_lv']
    lv_names = ['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge']
    for i, lv_name in enumerate(lv_names):
        param_list.append({
            'Model': 'Structural',
            'Latent_Variable': 'purchase_intention',
            'Parameter': f'γ_{lv_name}',
            'Estimate': gamma_lv[i]
        })
    
    gamma_x = results['parameters']['structural']['gamma_x']
    covariate_names = ['age_std', 'gender', 'income_std', 'education_level']
    for i, cov_name in enumerate(covariate_names):
        param_list.append({
            'Model': 'Structural',
            'Latent_Variable': 'purchase_intention',
            'Parameter': f'γ_{cov_name}',
            'Estimate': gamma_x[i]
        })
    
    # 선택모델 파라미터
    param_list.append({
        'Model': 'Choice',
        'Latent_Variable': 'N/A',
        'Parameter': 'β_Intercept',
        'Estimate': results['parameters']['choice']['intercept']
    })
    
    beta = results['parameters']['choice']['beta']
    choice_attrs = ['sugar_free', 'health_label', 'price']
    for i, attr in enumerate(choice_attrs):
        param_list.append({
            'Model': 'Choice',
            'Latent_Variable': 'N/A',
            'Parameter': f'β_{attr}',
            'Estimate': beta[i]
        })
    
    param_list.append({
        'Model': 'Choice',
        'Latent_Variable': 'N/A',
        'Parameter': 'λ',
        'Estimate': results['parameters']['choice']['lambda']
    })
    
    # DataFrame 생성
    df_params = pd.DataFrame(param_list)
    
    # CSV 저장
    csv_file = output_dir / 'multi_lv_iclv_results.csv'
    df_params.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # 요약 정보도 별도 CSV로 저장
    summary_data = {
        'Metric': ['Estimation_Time_Minutes', 'N_Individuals', 'N_Observations',
                   'N_Latent_Variables', 'Halton_Draws', 'Optimizer', 
                   'Log_Likelihood', 'AIC', 'BIC'],
        'Value': [f"{elapsed_time/60:.2f}", str(n_individuals), str(data.shape[0]),
                  '5', str(config.estimation.n_draws), config.estimation.optimizer,
                  f"{results['log_likelihood']:.4f}", 
                  f"{results.get('aic', 'N/A')}", 
                  f"{results.get('bic', 'N/A')}"]
    }
    
    df_summary = pd.DataFrame(summary_data)
    summary_file = output_dir / 'multi_lv_iclv_summary.csv'
    df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    print(f"\n결과 저장:")
    print(f"  - 파라미터: {csv_file}")
    print(f"  - 요약정보: {summary_file}")
    
    print("\n" + "="*70)
    print("다중 잠재변수 추정 완료!")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()

