"""
2단계 추정: 확장 모델(HC→PB→PI, HC→PP→PI)의 요인점수를 사용한 선택모델 추정

선택모델에 포함되는 잠재변수:
- purchase_intention (PI): 구매의도 - 주 효과
- nutrition_knowledge (NK): 영양지식 - 주 효과
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


def _get_significance(p_value: float) -> str:
    """p-value에서 유의성 기호 반환"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    create_sugar_substitute_multi_lv_config
)
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice


def main():
    print("=" * 70)
    print("2단계 추정: 선택모델 (PI + NK 주 효과)")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data_cleaned.csv"
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")
    
    # 2. 1단계 결과 로드
    print("\n[2] 1단계 결과 로드 중...")
    stage1_path = project_root / "results" / "sequential_stage_wise" / "stage1_HC-PB_HC-PP_PB-PI_PP-PI_results.pkl"
    
    if not stage1_path.exists():
        raise FileNotFoundError(f"1단계 결과 파일이 없습니다: {stage1_path}")
    
    print(f"✅ 1단계 결과 파일: {stage1_path.name}")
    
    # 3. 모델 설정 생성
    print("\n[3] 선택모델 설정 중...")

    # 확장 모델 경로 설정 (HC→PB→PI, HC→PP→PI)
    custom_paths = [
        {'target': 'perceived_benefit', 'predictors': ['health_concern']},
        {'target': 'perceived_price', 'predictors': ['health_concern']},
        {'target': 'purchase_intention', 'predictors': ['perceived_benefit', 'perceived_price']}
    ]

    config = create_sugar_substitute_multi_lv_config(
        n_draws=100,
        max_iterations=1000,
        use_hierarchical=False,
        all_lvs_as_main=False,  # 모든 LV 주효과 사용 안 함
        custom_paths=custom_paths
    )

    # 선택모델 설정 수정: PI와 NK만 주 효과로 사용
    from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

    config.choice = ChoiceConfig(
        choice_attributes=['health_label', 'price'],  # sugar_free 제거 (대안 A/B로 구분됨)
        choice_type='binary',
        price_variable='price',
        all_lvs_as_main=True,
        main_lvs=['purchase_intention', 'nutrition_knowledge'],  # PI와 NK만
        moderation_enabled=False
    )

    # 선택모델에 사용할 잠재변수 확인
    print(f"✅ 선택모델 주 효과:")
    print(f"   - purchase_intention (PI): 구매의도")
    print(f"   - nutrition_knowledge (NK): 영양지식")

    # 4. 선택모델 생성
    print("\n[4] 선택모델 생성 중...")
    choice_model = MultinomialLogitChoice(config.choice)
    print("✅ 선택모델 생성 완료")

    # 5. Estimator 생성
    print("\n[5] Estimator 생성 중...")
    estimator = SequentialEstimator(config)
    print("✅ Estimator 생성 완료")

    # 6. 2단계 추정 실행
    print("\n[6] 2단계 추정 실행 중...")
    print("   (1단계 요인점수를 사용하여 선택모델 추정)")

    results = estimator.estimate_stage2_only(
        data=data,
        choice_model=choice_model,
        factor_scores=str(stage1_path)  # 1단계 결과 파일 경로
    )
    
    print("\n✅ 2단계 추정 완료!")

    # 7. 결과 출력
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    
    print(f"\n[로그우도] {results['log_likelihood']:.2f}")
    print(f"[AIC] {results['aic']:.2f}")
    print(f"[BIC] {results['bic']:.2f}")

    # 파라미터 출력 (통계량 포함)
    if 'parameter_statistics' in results and results['parameter_statistics'] is not None:
        param_stats = results['parameter_statistics']

        print(f"\n[선택모델 파라미터]\n")
        print("-" * 100)
        print(f"{'파라미터':40s} {'계수':>12s} {'표준오차':>12s} {'t-통계량':>12s} {'p-value':>12s} {'유의성':>10s}")
        print("-" * 100)

        # ASC (대안별 상수)
        for key in ['asc_sugar', 'ASC_sugar', 'asc_sugar_free', 'ASC_sugar_free',
                   'asc_A', 'ASC_A', 'asc_B', 'ASC_B']:
            if key in param_stats:
                stat = param_stats[key]
                sig = _get_significance(stat['p'])
                print(f"{key:40s} {stat['estimate']:12.4f} {stat['se']:12.4f} {stat['t']:12.4f} {stat['p']:12.4f} {sig:>10s}")

        # intercept (대안별 모델이 아닌 경우)
        if 'intercept' in param_stats:
            stat = param_stats['intercept']
            sig = _get_significance(stat['p'])
            print(f"{'intercept':40s} {stat['estimate']:12.4f} {stat['se']:12.4f} {stat['t']:12.4f} {stat['p']:12.4f} {sig:>10s}")

        # beta (속성 계수)
        if 'beta' in param_stats:
            for attr_name, stat in param_stats['beta'].items():
                sig = _get_significance(stat['p'])
                print(f"{f'beta_{attr_name}':40s} {stat['estimate']:12.4f} {stat['se']:12.4f} {stat['t']:12.4f} {stat['p']:12.4f} {sig:>10s}")

        # theta (대안별 잠재변수 계수)
        for key in sorted([k for k in param_stats.keys() if k.startswith('theta_')]):
            stat = param_stats[key]
            sig = _get_significance(stat['p'])
            print(f"{key:40s} {stat['estimate']:12.4f} {stat['se']:12.4f} {stat['t']:12.4f} {stat['p']:12.4f} {sig:>10s}")

        # lambda (잠재변수 주 효과 - 대안별 모델이 아닌 경우)
        for key in ['lambda_purchase_intention', 'lambda_nutrition_knowledge',
                    'lambda_main', 'lambda_mod_perceived_price', 'lambda_mod_nutrition_knowledge']:
            if key in param_stats:
                stat = param_stats[key]
                sig = _get_significance(stat['p'])
                print(f"{key:40s} {stat['estimate']:12.4f} {stat['se']:12.4f} {stat['t']:12.4f} {stat['p']:12.4f} {sig:>10s}")

        print("-" * 100)

        # 유의한 파라미터 개수
        all_p_values = []
        for key in ['asc_sugar', 'ASC_sugar', 'asc_sugar_free', 'ASC_sugar_free',
                   'asc_A', 'ASC_A', 'asc_B', 'ASC_B', 'intercept']:
            if key in param_stats:
                all_p_values.append(param_stats[key]['p'])
        if 'beta' in param_stats:
            all_p_values.extend([stat['p'] for stat in param_stats['beta'].values()])
        for key in param_stats.keys():
            if key.startswith('theta_') or key.startswith('lambda_'):
                all_p_values.append(param_stats[key]['p'])

        sig_count = sum(1 for p in all_p_values if p < 0.05)
        print(f"\n유의한 파라미터 (p<0.05): {sig_count}/{len(all_p_values)}개")

    elif 'params' in results:
        # 통계량이 없는 경우 파라미터만 출력
        params = results['params']

        print(f"\n[선택모델 파라미터]\n")
        print("-" * 80)
        print(f"{'파라미터':40s} {'값':>15s} {'설명':>20s}")
        print("-" * 80)

        # intercept
        if 'intercept' in params:
            print(f"{'intercept':40s} {params['intercept']:15.4f} {'절편':>20s}")

        # beta (속성 계수)
        if 'beta' in params:
            beta = params['beta']
            beta_names = ['sugar_free', 'health_label', 'price']
            if isinstance(beta, np.ndarray):
                for i, val in enumerate(beta):
                    name = beta_names[i] if i < len(beta_names) else f'beta_{i}'
                    print(f"{f'beta_{name}':40s} {val:15.4f} {name:>20s}")
            else:
                print(f"{'beta':40s} {beta:15.4f} {'속성계수':>20s}")

        # lambda (잠재변수 주 효과)
        if 'lambda_purchase_intention' in params:
            print(f"{'lambda_purchase_intention':40s} {params['lambda_purchase_intention']:15.4f} {'구매의도 (PI)':>20s}")

        if 'lambda_nutrition_knowledge' in params:
            print(f"{'lambda_nutrition_knowledge':40s} {params['lambda_nutrition_knowledge']:15.4f} {'영양지식 (NK)':>20s}")

        # 기타 lambda (하위 호환)
        if 'lambda_main' in params:
            print(f"{'lambda_main':40s} {params['lambda_main']:15.4f} {'주 효과':>20s}")

        if 'lambda_mod_perceived_price' in params:
            print(f"{'lambda_mod_perceived_price':40s} {params['lambda_mod_perceived_price']:15.4f} {'가격 조절':>20s}")

        if 'lambda_mod_nutrition_knowledge' in params:
            print(f"{'lambda_mod_nutrition_knowledge':40s} {params['lambda_mod_nutrition_knowledge']:15.4f} {'지식 조절':>20s}")

        print("-" * 80)
    
    # 8. 결과 저장
    print("\n" + "=" * 70)
    print("결과 저장")
    print("=" * 70)
    
    save_dir = project_root / "results" / "sequential_stage_wise"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 파라미터 저장 (통계량 포함)
    if 'parameter_statistics' in results and results['parameter_statistics'] is not None:
        param_stats = results['parameter_statistics']
        param_data = []

        # ASC (대안별 상수)
        asc_descriptions = {
            'asc_sugar': '일반당 상수',
            'ASC_sugar': '일반당 상수',
            'asc_sugar_free': '무설탕 상수',
            'ASC_sugar_free': '무설탕 상수',
            'asc_A': '대안 A 상수',
            'ASC_A': '대안 A 상수',
            'asc_B': '대안 B 상수',
            'ASC_B': '대안 B 상수'
        }

        for key, desc in asc_descriptions.items():
            if key in param_stats:
                stat = param_stats[key]
                param_data.append({
                    'parameter': key,
                    'estimate': stat['estimate'],
                    'std_error': stat['se'],
                    't_statistic': stat['t'],
                    'p_value': stat['p'],
                    'significance': _get_significance(stat['p']),
                    'description': desc
                })

        # intercept (대안별 모델이 아닌 경우)
        if 'intercept' in param_stats:
            stat = param_stats['intercept']
            param_data.append({
                'parameter': 'intercept',
                'estimate': stat['estimate'],
                'std_error': stat['se'],
                't_statistic': stat['t'],
                'p_value': stat['p'],
                'significance': _get_significance(stat['p']),
                'description': '절편'
            })

        # beta (속성 계수)
        if 'beta' in param_stats:
            for attr_name, stat in param_stats['beta'].items():
                param_data.append({
                    'parameter': f'beta_{attr_name}',
                    'estimate': stat['estimate'],
                    'std_error': stat['se'],
                    't_statistic': stat['t'],
                    'p_value': stat['p'],
                    'significance': _get_significance(stat['p']),
                    'description': attr_name
                })

        # theta (대안별 잠재변수 계수)
        theta_descriptions = {
            'theta_sugar_purchase_intention': '일반당 × 구매의도',
            'theta_sugar_nutrition_knowledge': '일반당 × 영양지식',
            'theta_sugar_free_purchase_intention': '무설탕 × 구매의도',
            'theta_sugar_free_nutrition_knowledge': '무설탕 × 영양지식',
            'theta_A_purchase_intention': '대안 A × 구매의도',
            'theta_A_nutrition_knowledge': '대안 A × 영양지식',
            'theta_B_purchase_intention': '대안 B × 구매의도',
            'theta_B_nutrition_knowledge': '대안 B × 영양지식'
        }

        for key in sorted([k for k in param_stats.keys() if k.startswith('theta_')]):
            stat = param_stats[key]
            desc = theta_descriptions.get(key, key)
            param_data.append({
                'parameter': key,
                'estimate': stat['estimate'],
                'std_error': stat['se'],
                't_statistic': stat['t'],
                'p_value': stat['p'],
                'significance': _get_significance(stat['p']),
                'description': desc
            })

        # lambda (잠재변수 주 효과 - 대안별 모델이 아닌 경우)
        lambda_descriptions = {
            'lambda_purchase_intention': '구매의도 (PI)',
            'lambda_nutrition_knowledge': '영양지식 (NK)',
            'lambda_main': '주 효과',
            'lambda_mod_perceived_price': '가격 조절',
            'lambda_mod_nutrition_knowledge': '지식 조절'
        }

        for key, desc in lambda_descriptions.items():
            if key in param_stats:
                stat = param_stats[key]
                param_data.append({
                    'parameter': key,
                    'estimate': stat['estimate'],
                    'std_error': stat['se'],
                    't_statistic': stat['t'],
                    'p_value': stat['p'],
                    'significance': _get_significance(stat['p']),
                    'description': desc
                })

        param_df = pd.DataFrame(param_data)
        param_path = save_dir / "stage2_extended_model_parameters.csv"
        param_df.to_csv(param_path, index=False, encoding='utf-8-sig')
        print(f"\n  📁 {param_path}")

    elif 'params' in results:
        # 통계량이 없는 경우 파라미터만 저장
        params = results['params']
        param_data = []
        beta_names = ['sugar_free', 'health_label', 'price']

        # intercept
        if 'intercept' in params:
            param_data.append({'parameter': 'intercept', 'value': params['intercept'], 'description': '절편'})

        # beta
        if 'beta' in params:
            beta = params['beta']
            if isinstance(beta, np.ndarray):
                for i, val in enumerate(beta):
                    name = beta_names[i] if i < len(beta_names) else f'beta_{i}'
                    param_data.append({'parameter': f'beta_{name}', 'value': val, 'description': name})
            else:
                param_data.append({'parameter': 'beta', 'value': beta, 'description': '속성계수'})

        # lambda (잠재변수 주 효과)
        if 'lambda_purchase_intention' in params:
            param_data.append({'parameter': 'lambda_purchase_intention', 'value': params['lambda_purchase_intention'], 'description': '구매의도 (PI)'})

        if 'lambda_nutrition_knowledge' in params:
            param_data.append({'parameter': 'lambda_nutrition_knowledge', 'value': params['lambda_nutrition_knowledge'], 'description': '영양지식 (NK)'})

        # 기타 lambda (하위 호환)
        if 'lambda_main' in params:
            param_data.append({'parameter': 'lambda_main', 'value': params['lambda_main'], 'description': '주 효과'})
        if 'lambda_mod_perceived_price' in params:
            param_data.append({'parameter': 'lambda_mod_perceived_price', 'value': params['lambda_mod_perceived_price'], 'description': '가격 조절'})
        if 'lambda_mod_nutrition_knowledge' in params:
            param_data.append({'parameter': 'lambda_mod_nutrition_knowledge', 'value': params['lambda_mod_nutrition_knowledge'], 'description': '지식 조절'})

        param_df = pd.DataFrame(param_data)
        param_path = save_dir / "stage2_extended_model_parameters.csv"
        param_df.to_csv(param_path, index=False, encoding='utf-8-sig')
        print(f"\n  📁 {param_path}")

    # 적합도 저장
    fit_path = save_dir / "stage2_extended_model_fit.csv"
    fit_df = pd.DataFrame([{
        'log_likelihood': results['log_likelihood'],
        'AIC': results['aic'],
        'BIC': results['bic']
    }])
    fit_df.to_csv(fit_path, index=False, encoding='utf-8-sig')
    print(f"  📁 {fit_path}")
    
    print("\n" + "=" * 70)
    print("2단계 추정 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()

