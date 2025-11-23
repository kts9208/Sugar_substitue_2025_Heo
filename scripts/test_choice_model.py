"""
선택모델 fit() 메서드 테스트

순차추정 Step 2: 요인점수를 사용한 선택모델 추정

Author: Sugar Substitute Research Team
Date: 2025-01-15
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig


def create_sample_data(n_individuals=100, seed=42):
    """
    샘플 선택 데이터 생성
    
    각 개인은 3개 대안 (제품A, 제품B, 구매안함) 중 선택
    """
    np.random.seed(seed)
    
    data_list = []
    
    for i in range(n_individuals):
        # 제품A
        data_list.append({
            'individual_id': i,
            'alternative': 1,
            'sugar_free': np.random.choice([0, 1]),
            'health_label': np.random.choice([0, 1]),
            'price': np.random.uniform(2.0, 4.0),
            'choice': 0  # 나중에 설정
        })
        
        # 제품B
        data_list.append({
            'individual_id': i,
            'alternative': 2,
            'sugar_free': np.random.choice([0, 1]),
            'health_label': np.random.choice([0, 1]),
            'price': np.random.uniform(2.0, 4.0),
            'choice': 0  # 나중에 설정
        })
        
        # 구매안함 (opt-out)
        data_list.append({
            'individual_id': i,
            'alternative': 3,
            'sugar_free': np.nan,
            'health_label': np.nan,
            'price': np.nan,
            'choice': 0  # 나중에 설정
        })
    
    data = pd.DataFrame(data_list)
    
    # 선택 결과 생성 (랜덤)
    for i in range(n_individuals):
        chosen_alt = np.random.choice([0, 1, 2])  # 0=제품A, 1=제품B, 2=구매안함
        data.loc[i*3 + chosen_alt, 'choice'] = 1
    
    return data


def create_sample_factor_scores(n_individuals=100, seed=42):
    """
    샘플 요인점수 생성
    """
    np.random.seed(seed)
    
    factor_scores = {
        'purchase_intention': np.random.randn(n_individuals),
        'perceived_price': np.random.randn(n_individuals),
        'nutrition_knowledge': np.random.randn(n_individuals)
    }
    
    return factor_scores


def main():
    print("=" * 70)
    print("선택모델 fit() 메서드 테스트")
    print("=" * 70)
    print()
    
    # 1. 샘플 데이터 생성
    print("[1] 샘플 데이터 생성...")
    n_individuals = 100
    choice_data = create_sample_data(n_individuals)
    factor_scores = create_sample_factor_scores(n_individuals)
    
    print(f"  선택 데이터: {choice_data.shape}")
    print(f"  개인 수: {n_individuals}")
    print(f"  요인점수:")
    for lv_name, scores in factor_scores.items():
        print(f"    - {lv_name}: {scores.shape}")
    print()
    
    # 2. 모델 설정
    print("[2] 모델 설정...")
    config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
        choice_type='multinomial',
        price_variable='price',
        moderation_enabled=True,
        moderator_lvs=['perceived_price', 'nutrition_knowledge'],
        main_lv='purchase_intention'
    )
    
    model = MultinomialLogitChoice(config)
    print("✅ 모델 설정 완료")
    print()
    
    # 3. 모델 추정
    print("[3] 모델 추정...")
    results = model.fit(choice_data, factor_scores)
    print()
    
    # 4. 결과 출력
    print("=" * 70)
    print("추정 결과")
    print("=" * 70)
    print()
    
    print("[파라미터]")
    params = results['params']
    print(f"  intercept: {params['intercept']:.4f}")
    print(f"  beta:")
    for i, attr in enumerate(config.choice_attributes):
        print(f"    - {attr}: {params['beta'][i]:.4f}")
    print(f"  lambda_main (구매의도 주효과): {params['lambda_main']:.4f}")
    print(f"  lambda_mod_perceived_price (가격 조절효과): {params['lambda_mod_perceived_price']:.4f}")
    print(f"  lambda_mod_nutrition_knowledge (지식 조절효과): {params['lambda_mod_nutrition_knowledge']:.4f}")
    print()
    
    print("[모델 적합도]")
    print(f"  로그우도: {results['log_likelihood']:.2f}")
    print(f"  AIC: {results['aic']:.2f}")
    print(f"  BIC: {results['bic']:.2f}")
    print(f"  파라미터 수: {results['n_params']}")
    print(f"  관측치 수: {results['n_obs']}")
    print()
    
    print("[최적화 정보]")
    print(f"  성공: {results['success']}")
    print(f"  메시지: {results['message']}")
    print(f"  반복 횟수: {results['n_iterations']}")
    print()

    # 표준오차 및 p-value 출력
    if results.get('parameter_statistics') is not None:
        print("[통계적 유의성]")
        print(f"{'파라미터':<40} {'추정값':>10} {'표준오차':>10} {'t-값':>10} {'p-값':>10}")
        print("-" * 80)

        stats = results['parameter_statistics']

        # Intercept
        if 'intercept' in stats:
            s = stats['intercept']
            sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
            print(f"{'intercept':<40} {s['estimate']:>10.4f} {s['se']:>10.4f} {s['t']:>10.4f} {s['p']:>10.4f} {sig}")

        # Beta
        if 'beta' in stats:
            for attr, s in stats['beta'].items():
                sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
                print(f"{'beta_' + attr:<40} {s['estimate']:>10.4f} {s['se']:>10.4f} {s['t']:>10.4f} {s['p']:>10.4f} {sig}")

        # Lambda main
        if 'lambda_main' in stats:
            s = stats['lambda_main']
            sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
            print(f"{'lambda_main':<40} {s['estimate']:>10.4f} {s['se']:>10.4f} {s['t']:>10.4f} {s['p']:>10.4f} {sig}")

        # Lambda moderators
        for key in ['lambda_mod_perceived_price', 'lambda_mod_nutrition_knowledge']:
            if key in stats:
                s = stats[key]
                sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
                print(f"{key:<40} {s['estimate']:>10.4f} {s['se']:>10.4f} {s['t']:>10.4f} {s['p']:>10.4f} {sig}")

        print()
        print("유의수준: *** p<0.001, ** p<0.01, * p<0.05")
    else:
        print("[경고] 표준오차 및 p-value를 계산할 수 없습니다.")

    # 결과 저장
    print("\n[결과 저장]")
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    save_dir = project_root / "results" / "final" / "choice_only"
    save_dir.mkdir(parents=True, exist_ok=True)

    # CSV 파일로 저장
    save_file = save_dir / "choice_model_results.csv"

    # 결과 데이터 준비
    result_data = []

    # 모델 적합도
    result_data.append({
        'section': 'Model_Fit',
        'parameter': 'log_likelihood',
        'estimate': results['log_likelihood'],
        'std_error': '',
        't_statistic': '',
        'p_value': '',
        'significance': ''
    })
    result_data.append({
        'section': 'Model_Fit',
        'parameter': 'AIC',
        'estimate': results['aic'],
        'std_error': '',
        't_statistic': '',
        'p_value': '',
        'significance': ''
    })
    result_data.append({
        'section': 'Model_Fit',
        'parameter': 'BIC',
        'estimate': results['bic'],
        'std_error': '',
        't_statistic': '',
        'p_value': '',
        'significance': ''
    })
    result_data.append({
        'section': 'Model_Fit',
        'parameter': 'n_params',
        'estimate': results['n_params'],
        'std_error': '',
        't_statistic': '',
        'p_value': '',
        'significance': ''
    })
    result_data.append({
        'section': 'Model_Fit',
        'parameter': 'n_obs',
        'estimate': results['n_obs'],
        'std_error': '',
        't_statistic': '',
        'p_value': '',
        'significance': ''
    })

    # 파라미터
    params = results['params']
    stats = results.get('parameter_statistics', {})

    # Intercept
    if 'intercept' in stats:
        s = stats['intercept']
        sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
        result_data.append({
            'section': 'Parameters',
            'parameter': 'intercept',
            'estimate': s['estimate'],
            'std_error': s['se'],
            't_statistic': s['t'],
            'p_value': s['p'],
            'significance': sig
        })
    else:
        result_data.append({
            'section': 'Parameters',
            'parameter': 'intercept',
            'estimate': params['intercept'],
            'std_error': '',
            't_statistic': '',
            'p_value': '',
            'significance': ''
        })

    # Beta
    for i, attr in enumerate(config.choice_attributes):
        param_name = f'beta_{attr}'
        if param_name in stats:
            s = stats[param_name]
            sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
            result_data.append({
                'section': 'Parameters',
                'parameter': param_name,
                'estimate': s['estimate'],
                'std_error': s['se'],
                't_statistic': s['t'],
                'p_value': s['p'],
                'significance': sig
            })
        else:
            result_data.append({
                'section': 'Parameters',
                'parameter': param_name,
                'estimate': params['beta'][i],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': ''
            })

    # Lambda
    if 'lambda_main' in params:
        if 'lambda_main' in stats:
            s = stats['lambda_main']
            sig = "***" if s['p'] < 0.001 else "**" if s['p'] < 0.01 else "*" if s['p'] < 0.05 else ""
            result_data.append({
                'section': 'Parameters',
                'parameter': 'lambda_main',
                'estimate': s['estimate'],
                'std_error': s['se'],
                't_statistic': s['t'],
                'p_value': s['p'],
                'significance': sig
            })
        else:
            result_data.append({
                'section': 'Parameters',
                'parameter': 'lambda_main',
                'estimate': params['lambda_main'],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': ''
            })

    import pandas as pd
    df_results = pd.DataFrame(result_data)
    df_results.to_csv(save_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ 결과 저장 완료: {save_file.name}")

    print()
    print("=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)


if __name__ == '__main__':
    main()

