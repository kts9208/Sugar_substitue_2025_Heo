"""
2단계 추정: 1단계 요인점수를 사용한 선택모델 추정

🎯 사용법:
    main() 함수 내 상단의 설정 변수만 수정하면 됩니다!

📌 설정 예시:

1. Base Model (잠재변수 없음):
    MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']  # Auto-generated
    MODERATION_LVS = []  # Auto-generated
    LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]  # Auto-generated

2. Base + PI 주효과:
    MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']  # Auto-generated
    MODERATION_LVS = []  # Auto-generated
    LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]  # Auto-generated

3. Base + PI + NK 주효과:
    MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']  # Auto-generated
    MODERATION_LVS = []  # Auto-generated
    LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]  # Auto-generated

4. Base + PI 주효과 + PI×price 상호작용:
    MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']  # Auto-generated
    MODERATION_LVS = []  # Auto-generated
    LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]  # Auto-generated

5. Base + PI + NK 주효과 + 조절효과 + 상호작용:
    MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']  # Auto-generated
    MODERATION_LVS = []  # Auto-generated
    LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]  # Auto-generated
        ('purchase_intention', 'price'),
        ('nutrition_knowledge', 'health_label')
    ]

💡 사용 가능한 잠재변수:
    - 'purchase_intention' (PI): 구매의도
    - 'nutrition_knowledge' (NK): 영양지식
    - 'perceived_benefit' (PB): 건강유익성
    - 'perceived_price' (PP): 가격수준
    - 'health_concern' (HC): 건강관심도

💡 사용 가능한 속성:
    - 'health_label': 건강 라벨
    - 'price': 가격

Author: ICLV Team
Date: 2025-01-16
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# 공통 유틸리티 import
from model_config_utils import (
    build_choice_config_dict,
    extract_stage1_model_name,
    generate_stage2_filename
)


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
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import (
    MultinomialLogitChoice,
    BinaryProbitChoice
)


def main():
    # ═══════════════════════════════════════════════════════════════════
    # 🎯 사용자 설정 영역 - 여기만 수정하세요!
    # ═══════════════════════════════════════════════════════════════════

    # 📌 1단계 결과 파일명 (1단계에서 생성된 파일명)
    STAGE1_RESULT_FILE = "stage1_HC-PB_PB-PI_results.pkl"

    # 📌 요인점수 변환 방법
    # 'zscore': Z-score 표준화 (평균 0, 표준편차 1) - 기본값
    # 'center': 중심화 (평균 0, 표준편차는 원본 유지)
    STANDARDIZATION_METHOD = 'zscore'  # Z-score 표준화 사용

    # 📌 선택모델 설정
    CHOICE_ATTRIBUTES = ['health_label', 'price']  # 선택 속성
    CHOICE_TYPE = 'multinomial'  # 'binary' 또는 'multinomial' - 3개 대안이므로 multinomial 사용
    PRICE_VARIABLE = 'price'  # 가격 변수명

    # 📌 잠재변수 주효과 (원하는 잠재변수만 추가)
    # 예시: [] = Base Model (잠재변수 없음)
    #      ['purchase_intention'] = Base + PI 주효과
    #      ['purchase_intention', 'nutrition_knowledge'] = Base + PI + NK 주효과
    MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']  # Auto-generated

    # 📌 조절효과 (잠재변수 2개 세트)
    # 예시: [('perceived_price', 'nutrition_knowledge')] = PP와 NK의 조절효과
    MODERATION_LVS = []  # Auto-generated

    # 📌 LV-Attribute 상호작용 (잠재변수-속성 2개 세트)
    # 예시: [('purchase_intention', 'price')] = PI × price 상호작용
    #      [('purchase_intention', 'price'), ('nutrition_knowledge', 'health_label')]
    LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]  # Auto-generated

    # ═══════════════════════════════════════════════════════════════════
    # 🤖 자동 처리 영역 - 수정 불필요
    # ═══════════════════════════════════════════════════════════════════

    # 모델 유형 자동 판단
    model_type_parts = ["Base Model"]
    if MAIN_LVS:
        lv_abbr = {'purchase_intention': 'PI', 'nutrition_knowledge': 'NK',
                   'perceived_benefit': 'PB', 'perceived_price': 'PP', 'health_concern': 'HC'}
        lv_names = [lv_abbr.get(lv, lv.upper()) for lv in MAIN_LVS]
        model_type_parts.append(f"+ {' + '.join(lv_names)} 주효과")
    if MODERATION_LVS:
        model_type_parts.append(f"+ 조절효과 {len(MODERATION_LVS)}개")
    if LV_ATTRIBUTE_INTERACTIONS:
        model_type_parts.append(f"+ LV-Attr 상호작용 {len(LV_ATTRIBUTE_INTERACTIONS)}개")

    model_type_str = " ".join(model_type_parts)

    print("=" * 70)
    print(f"2단계 추정: 선택모델 ({model_type_str})")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    data = pd.read_csv(data_path)
    print(f"[OK] 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")

    # 2. 1단계 결과 로드
    print("\n[2] 1단계 결과 로드 중...")
    # 최종 결과 폴더에서 로드
    stage1_path = project_root / "results" / "final" / "sequential" / "stage1" / STAGE1_RESULT_FILE

    if not stage1_path.exists():
        raise FileNotFoundError(f"1단계 결과 파일이 없습니다: {stage1_path}")

    print(f"[OK] 1단계 결과 파일: {stage1_path.name}")

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
        all_lvs_as_main=False,
        custom_paths=custom_paths
    )

    # 선택모델 설정 자동 생성 (공통 유틸리티 사용)
    from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

    choice_config_dict = build_choice_config_dict(
        main_lvs=MAIN_LVS,
        moderation_lvs=MODERATION_LVS,
        lv_attribute_interactions=LV_ATTRIBUTE_INTERACTIONS
    )

    config.choice = ChoiceConfig(
        choice_attributes=CHOICE_ATTRIBUTES,
        choice_type=CHOICE_TYPE,
        price_variable=PRICE_VARIABLE,
        **choice_config_dict  # 공통 유틸리티에서 생성한 설정 사용
    )

    # 선택모델 설정 자동 출력
    print(f"[OK] 선택모델 설정:")
    print(f"   - 모델 유형: {model_type_str}")
    print(f"   - 선택 속성: {', '.join(CHOICE_ATTRIBUTES)}")

    if MAIN_LVS:
        lv_full_names = {'purchase_intention': '구매의도(PI)', 'nutrition_knowledge': '영양지식(NK)',
                        'perceived_benefit': '건강유익성(PB)', 'perceived_price': '가격수준(PP)',
                        'health_concern': '건강관심도(HC)'}
        lv_display = [lv_full_names.get(lv, lv) for lv in MAIN_LVS]
        print(f"   - 잠재변수 주효과: {', '.join(lv_display)}")
    else:
        print(f"   - 잠재변수 주효과: 없음")

    if MODERATION_LVS:
        print(f"   - 조절효과: {len(MODERATION_LVS)}개")
        for mod, moderated in MODERATION_LVS:
            print(f"      * {mod} × {moderated}")
    else:
        print(f"   - 조절효과: 없음")

    if LV_ATTRIBUTE_INTERACTIONS:
        print(f"   - LV-Attribute 상호작용: {len(LV_ATTRIBUTE_INTERACTIONS)}개")
        for lv, attr in LV_ATTRIBUTE_INTERACTIONS:
            print(f"      * {lv} × {attr}")
    else:
        print(f"   - LV-Attribute 상호작용: 없음")

    # 4. 선택모델 생성
    print("\n[4] 선택모델 생성 중...")
    if CHOICE_TYPE == 'multinomial':
        choice_model = MultinomialLogitChoice(config.choice)
        print("[OK] 선택모델 생성 완료 (Multinomial Logit)")
    elif CHOICE_TYPE == 'binary':
        choice_model = BinaryProbitChoice(config.choice)
        print("[OK] 선택모델 생성 완료 (Binary Probit)")
    else:
        raise ValueError(f"지원하지 않는 CHOICE_TYPE: {CHOICE_TYPE}")

    # 5. Estimator 생성
    print("\n[5] Estimator 생성 중...")
    estimator = SequentialEstimator(config, standardization_method=STANDARDIZATION_METHOD)
    print("[OK] Estimator 생성 완료")
    print(f"   - 요인점수 변환 방법: {STANDARDIZATION_METHOD}")

    # 6. 2단계 추정 실행
    print("\n[6] 2단계 추정 실행 중...")
    print("   (1단계 요인점수를 사용하여 선택모델 추정)")

    results = estimator.estimate_stage2_only(
        data=data,
        choice_model=choice_model,
        factor_scores=str(stage1_path)  # 1단계 결과 파일 경로
    )
    
    print("\n[OK] 2단계 추정 완료!")

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

        # gamma (LV-Attribute 상호작용)
        for key in sorted([k for k in param_stats.keys() if k.startswith('gamma_')]):
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
            if key.startswith('theta_') or key.startswith('lambda_') or key.startswith('gamma_'):
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

        # gamma (LV-Attribute 상호작용, 대안별)
        gamma_descriptions = {
            'gamma_sugar_purchase_intention_price': '일반당: PI × price',
            'gamma_sugar_purchase_intention_health_label': '일반당: PI × health_label',
            'gamma_sugar_nutrition_knowledge_health_label': '일반당: NK × health_label',
            'gamma_sugar_free_purchase_intention_price': '무설탕: PI × price',
            'gamma_sugar_free_purchase_intention_health_label': '무설탕: PI × health_label',
            'gamma_sugar_free_nutrition_knowledge_health_label': '무설탕: NK × health_label'
        }

        for key, desc in gamma_descriptions.items():
            if key in params:
                print(f"{key:40s} {params[key]:15.4f} {desc:>20s}")

        print("-" * 80)
    
    # 8. 결과 저장
    print("\n" + "=" * 70)
    print("결과 저장")
    print("=" * 70)

    # 최종 결과 폴더에 저장
    save_dir = project_root / "results" / "final" / "sequential" / "stage2"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1단계 모델 이름 추출
    stage1_model_name = extract_stage1_model_name(STAGE1_RESULT_FILE)

    # 동적 파일명 생성 (1단계 + 2단계 정보 포함)
    filename_prefix = generate_stage2_filename(config, stage1_model_name)
    print(f"\n파일명 접두사: {filename_prefix}")
    print(f"  - 1단계 모델: {stage1_model_name}")
    print(f"  - 2단계 모델: {filename_prefix.split('1_')[1].replace('2', '')}")

    # 통합 결과 저장 (적합도 + 파라미터)
    combined_data = []

    # 1. 적합도 지수 추가 (섹션: Model_Fit)
    combined_data.append({
        'section': 'Model_Fit',
        'parameter': 'log_likelihood',
        'estimate': results['log_likelihood'],
        'std_error': '',
        't_statistic': '',
        'p_value': '',
        'significance': '',
        'description': 'Log-Likelihood'
    })
    combined_data.append({
        'section': 'Model_Fit',
        'parameter': 'AIC',
        'estimate': results['aic'],
        'std_error': '',
        't_statistic': '',
        'p_value': '',
        'significance': '',
        'description': 'Akaike Information Criterion'
    })
    combined_data.append({
        'section': 'Model_Fit',
        'parameter': 'BIC',
        'estimate': results['bic'],
        'std_error': '',
        't_statistic': '',
        'p_value': '',
        'significance': '',
        'description': 'Bayesian Information Criterion'
    })

    # 2. 파라미터 추가 (섹션: Parameters)
    if 'parameter_statistics' in results and results['parameter_statistics'] is not None:
        param_stats = results['parameter_statistics']

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
                combined_data.append({
                    'section': 'Parameters',
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
            combined_data.append({
                'section': 'Parameters',
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
                combined_data.append({
                    'section': 'Parameters',
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
            combined_data.append({
                'section': 'Parameters',
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
                combined_data.append({
                    'section': 'Parameters',
                    'parameter': key,
                    'estimate': stat['estimate'],
                    'std_error': stat['se'],
                    't_statistic': stat['t'],
                    'p_value': stat['p'],
                    'significance': _get_significance(stat['p']),
                    'description': desc
                })

        # gamma (LV-Attribute 상호작용, 대안별)
        gamma_descriptions = {
            'gamma_sugar_purchase_intention_price': '일반당: PI × price',
            'gamma_sugar_purchase_intention_health_label': '일반당: PI × health_label',
            'gamma_sugar_nutrition_knowledge_health_label': '일반당: NK × health_label',
            'gamma_sugar_free_purchase_intention_price': '무설탕: PI × price',
            'gamma_sugar_free_purchase_intention_health_label': '무설탕: PI × health_label',
            'gamma_sugar_free_nutrition_knowledge_health_label': '무설탕: NK × health_label'
        }

        for key in sorted([k for k in param_stats.keys() if k.startswith('gamma_')]):
            stat = param_stats[key]
            desc = gamma_descriptions.get(key, key)
            combined_data.append({
                'section': 'Parameters',
                'parameter': key,
                'estimate': stat['estimate'],
                'std_error': stat['se'],
                't_statistic': stat['t'],
                'p_value': stat['p'],
                'significance': _get_significance(stat['p']),
                'description': desc
            })

    elif 'params' in results:
        # 통계량이 없는 경우 파라미터만 저장 (간소화된 형식)
        params = results['params']
        beta_names = ['sugar_free', 'health_label', 'price']

        # intercept
        if 'intercept' in params:
            combined_data.append({
                'section': 'Parameters',
                'parameter': 'intercept',
                'estimate': params['intercept'],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
                'description': '절편'
            })

        # beta
        if 'beta' in params:
            beta = params['beta']
            if isinstance(beta, np.ndarray):
                for i, val in enumerate(beta):
                    name = beta_names[i] if i < len(beta_names) else f'beta_{i}'
                    combined_data.append({
                        'section': 'Parameters',
                        'parameter': f'beta_{name}',
                        'estimate': val,
                        'std_error': '',
                        't_statistic': '',
                        'p_value': '',
                        'significance': '',
                        'description': name
                    })
            else:
                combined_data.append({
                    'section': 'Parameters',
                    'parameter': 'beta',
                    'estimate': beta,
                    'std_error': '',
                    't_statistic': '',
                    'p_value': '',
                    'significance': '',
                    'description': '속성계수'
                })

        # lambda (잠재변수 주 효과)
        if 'lambda_purchase_intention' in params:
            combined_data.append({
                'section': 'Parameters',
                'parameter': 'lambda_purchase_intention',
                'estimate': params['lambda_purchase_intention'],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
                'description': '구매의도 (PI)'
            })

        if 'lambda_nutrition_knowledge' in params:
            combined_data.append({
                'section': 'Parameters',
                'parameter': 'lambda_nutrition_knowledge',
                'estimate': params['lambda_nutrition_knowledge'],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
                'description': '영양지식 (NK)'
            })

        # 기타 lambda (하위 호환)
        if 'lambda_main' in params:
            combined_data.append({
                'section': 'Parameters',
                'parameter': 'lambda_main',
                'estimate': params['lambda_main'],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
                'description': '주 효과'
            })
        if 'lambda_mod_perceived_price' in params:
            combined_data.append({
                'section': 'Parameters',
                'parameter': 'lambda_mod_perceived_price',
                'estimate': params['lambda_mod_perceived_price'],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
                'description': '가격 조절'
            })
        if 'lambda_mod_nutrition_knowledge' in params:
            combined_data.append({
                'section': 'Parameters',
                'parameter': 'lambda_mod_nutrition_knowledge',
                'estimate': params['lambda_mod_nutrition_knowledge'],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
                'description': '지식 조절'
            })

        # gamma (LV-Attribute 상호작용, 대안별)
        gamma_descriptions = {
            'gamma_sugar_purchase_intention_price': '일반당: PI × price',
            'gamma_sugar_purchase_intention_health_label': '일반당: PI × health_label',
            'gamma_sugar_nutrition_knowledge_health_label': '일반당: NK × health_label',
            'gamma_sugar_free_purchase_intention_price': '무설탕: PI × price',
            'gamma_sugar_free_purchase_intention_health_label': '무설탕: PI × health_label',
            'gamma_sugar_free_nutrition_knowledge_health_label': '무설탕: NK × health_label'
        }

        for key, desc in gamma_descriptions.items():
            if key in params:
                combined_data.append({
                    'section': 'Parameters',
                    'parameter': key,
                    'estimate': params[key],
                    'std_error': '',
                    't_statistic': '',
                    'p_value': '',
                    'significance': '',
                    'description': desc
                })

    # 통합 결과 저장 (하나의 CSV 파일)
    combined_df = pd.DataFrame(combined_data)
    combined_path = save_dir / f"{filename_prefix}_results.csv"
    combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
    print(f"\n  [SAVED] {combined_path}")
    
    print("\n" + "=" * 70)
    print("2단계 추정 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()

