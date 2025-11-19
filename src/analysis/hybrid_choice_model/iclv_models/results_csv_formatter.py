"""
ICLV 모델 결과를 표준 CSV 형식으로 저장하는 모듈

순차추정과 동시추정 모두에서 사용 가능한 공통 CSV 저장 로직
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def get_significance(p_value: float) -> str:
    """p-value에서 유의성 기호 반환"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def format_iclv_results_to_csv(
    results: Dict[str, Any],
    output_path: Path,
    include_measurement: bool = False,
    include_structural: bool = False
) -> None:
    """
    ICLV 모델 결과를 표준 CSV 형식으로 저장
    
    Args:
        results: 추정 결과 딕셔너리
            - log_likelihood: 로그우도
            - aic: AIC
            - bic: BIC
            - parameter_statistics: 파라미터 통계량 (estimate, se, t, p)
            - params: 파라미터 값 (통계량이 없는 경우)
        output_path: 출력 CSV 파일 경로
        include_measurement: 측정모델 파라미터 포함 여부 (동시추정용)
        include_structural: 구조모델 파라미터 포함 여부 (동시추정용)
    
    CSV 형식:
        section | parameter | estimate | std_error | t_statistic | p_value | significance | description
    """
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
    
    # 2. 파라미터 추가
    if 'parameter_statistics' in results and results['parameter_statistics'] is not None:
        param_stats = results['parameter_statistics']

        # 동시추정 (계층적 구조): {'measurement': {...}, 'structural': {...}, 'choice': {...}}
        if 'measurement' in param_stats or 'structural' in param_stats:
            # 측정모델 파라미터 (동시추정용)
            if include_measurement and 'measurement' in param_stats:
                combined_data.extend(_format_measurement_params_hierarchical(param_stats['measurement']))

            # 구조모델 파라미터 (동시추정용)
            if include_structural and 'structural' in param_stats:
                combined_data.extend(_format_structural_params_hierarchical(param_stats['structural']))

            # 선택모델 파라미터 (동시추정용)
            if 'choice' in param_stats:
                combined_data.extend(_format_choice_params_hierarchical(param_stats['choice']))
        else:
            # 순차추정 (flat 구조): {'asc_sugar': {...}, 'beta_health_label': {...}, ...}
            combined_data.extend(_format_choice_params_flat(param_stats))

    elif 'params' in results:
        # 통계량이 없는 경우 파라미터만 저장
        combined_data.extend(_format_params_without_stats(results['params']))
    
    # DataFrame 생성 및 저장
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')


def _format_measurement_params_hierarchical(measurement_stats: Dict) -> list:
    """측정모델 파라미터 포맷팅 (동시추정용 - 계층적 구조)

    Args:
        measurement_stats: {'lv_name': {'zeta': {...}, 'sigma_sq': {...}}, ...}
    """
    data = []

    for lv_name, lv_stats in measurement_stats.items():
        # zeta (요인적재량)
        if 'zeta' in lv_stats:
            zeta_stats = lv_stats['zeta']
            estimates = zeta_stats['estimate']
            std_errors = zeta_stats['std_error']
            t_stats = zeta_stats['t_statistic']
            p_values = zeta_stats['p_value']

            for i in range(len(estimates)):
                # NaN 값을 빈 문자열로 변환 (고정 파라미터)
                se = '' if np.isnan(std_errors[i]) else std_errors[i]
                t = '' if np.isnan(t_stats[i]) else t_stats[i]
                p = '' if np.isnan(p_values[i]) else p_values[i]
                sig = '' if np.isnan(p_values[i]) else get_significance(p_values[i])

                data.append({
                    'section': 'Measurement_Model',
                    'parameter': f'zeta_{lv_name}_{i}',
                    'estimate': estimates[i],
                    'std_error': se,
                    't_statistic': t,
                    'p_value': p,
                    'significance': sig,
                    'description': f'요인적재량: {lv_name} 지표 {i+1}'
                })

        # sigma_sq (측정오차 분산)
        if 'sigma_sq' in lv_stats:
            sigma_sq_stats = lv_stats['sigma_sq']
            estimates = sigma_sq_stats['estimate']
            std_errors = sigma_sq_stats['std_error']
            t_stats = sigma_sq_stats['t_statistic']
            p_values = sigma_sq_stats['p_value']

            for i in range(len(estimates)):
                # NaN 값을 빈 문자열로 변환 (고정 파라미터)
                se = '' if np.isnan(std_errors[i]) else std_errors[i]
                t = '' if np.isnan(t_stats[i]) else t_stats[i]
                p = '' if np.isnan(p_values[i]) else p_values[i]
                sig = '' if np.isnan(p_values[i]) else get_significance(p_values[i])

                data.append({
                    'section': 'Measurement_Model',
                    'parameter': f'sigma_sq_{lv_name}_{i}',
                    'estimate': estimates[i],
                    'std_error': se,
                    't_statistic': t,
                    'p_value': p,
                    'significance': sig,
                    'description': f'측정오차 분산: {lv_name} 지표 {i+1}'
                })

        # tau (절편)
        if 'tau' in lv_stats:
            tau_stats = lv_stats['tau']
            estimates = tau_stats['estimate']
            std_errors = tau_stats['std_error']
            t_stats = tau_stats['t_statistic']
            p_values = tau_stats['p_value']

            for i in range(len(estimates)):
                # NaN 값을 빈 문자열로 변환 (고정 파라미터)
                se = '' if np.isnan(std_errors[i]) else std_errors[i]
                t = '' if np.isnan(t_stats[i]) else t_stats[i]
                p = '' if np.isnan(p_values[i]) else p_values[i]
                sig = '' if np.isnan(p_values[i]) else get_significance(p_values[i])

                data.append({
                    'section': 'Measurement_Model',
                    'parameter': f'tau_{lv_name}_{i}',
                    'estimate': estimates[i],
                    'std_error': se,
                    't_statistic': t,
                    'p_value': p,
                    'significance': sig,
                    'description': f'절편: {lv_name} 지표 {i+1}'
                })

    return data


def _format_structural_params_hierarchical(structural_stats: Dict) -> list:
    """구조모델 파라미터 포맷팅 (동시추정용 - 계층적 구조)

    Args:
        structural_stats: {'gamma_pred_to_target': {...}, ...}
    """
    data = []

    # gamma (구조계수)
    gamma_descriptions = {
        'gamma_health_concern_to_perceived_benefit': 'HC → PB',
        'gamma_perceived_benefit_to_purchase_intention': 'PB → PI',
        'gamma_health_concern_to_purchase_intention': 'HC → PI',
        'gamma_nutrition_knowledge_to_purchase_intention': 'NK → PI'
    }

    for key, stat in structural_stats.items():
        if key.startswith('gamma_'):
            desc = gamma_descriptions.get(key, key)

            # NaN 값을 빈 문자열로 변환 (고정 파라미터)
            se = '' if np.isnan(stat['std_error']) else stat['std_error']
            t = '' if np.isnan(stat['t_statistic']) else stat['t_statistic']
            p = '' if np.isnan(stat['p_value']) else stat['p_value']
            sig = '' if np.isnan(stat['p_value']) else get_significance(stat['p_value'])

            data.append({
                'section': 'Structural_Model',
                'parameter': key,
                'estimate': stat['estimate'],
                'std_error': se,
                't_statistic': t,
                'p_value': p,
                'significance': sig,
                'description': desc
            })

    return data


def _format_choice_params_hierarchical(choice_stats: Dict) -> list:
    """선택모델 파라미터 포맷팅 (동시추정용 - 계층적 구조)

    Args:
        choice_stats: {'asc': {...}, 'beta': {...}, 'theta': {...}, 'gamma': {...}}
    """
    data = []

    # ASC (대안별 상수) - Multinomial Logit
    if 'asc' in choice_stats:
        asc_descriptions = {
            'sugar': '일반당 상수',
            'sugar_free': '무설탕 상수',
            'A': '대안 A 상수',
            'B': '대안 B 상수'
        }

        for alt_name, stat in choice_stats['asc'].items():
            desc = asc_descriptions.get(alt_name, f'{alt_name} 상수')

            # NaN 값을 빈 문자열로 변환 (고정 파라미터)
            se = '' if np.isnan(stat['std_error']) else stat['std_error']
            t = '' if np.isnan(stat['t_statistic']) else stat['t_statistic']
            p = '' if np.isnan(stat['p_value']) else stat['p_value']
            sig = '' if np.isnan(stat['p_value']) else get_significance(stat['p_value'])

            data.append({
                'section': 'Choice_Model',
                'parameter': f'asc_{alt_name}',
                'estimate': stat['estimate'],
                'std_error': se,
                't_statistic': t,
                'p_value': p,
                'significance': sig,
                'description': desc
            })

    # intercept (Binary Logit - 하위 호환성)
    if 'intercept' in choice_stats:
        stat = choice_stats['intercept']

        # NaN 값을 빈 문자열로 변환 (고정 파라미터)
        se = '' if np.isnan(stat['std_error']) else stat['std_error']
        t = '' if np.isnan(stat['t_statistic']) else stat['t_statistic']
        p = '' if np.isnan(stat['p_value']) else stat['p_value']
        sig = '' if np.isnan(stat['p_value']) else get_significance(stat['p_value'])

        data.append({
            'section': 'Choice_Model',
            'parameter': 'intercept',
            'estimate': stat['estimate'],
            'std_error': se,
            't_statistic': t,
            'p_value': p,
            'significance': sig,
            'description': '절편'
        })

    # beta (속성 계수)
    if 'beta' in choice_stats:
        beta_stats = choice_stats['beta']
        estimates = beta_stats['estimate']
        std_errors = beta_stats['std_error']
        t_stats = beta_stats['t_statistic']
        p_values = beta_stats['p_value']

        for i in range(len(estimates)):
            # NaN 값을 빈 문자열로 변환 (고정 파라미터)
            se = '' if np.isnan(std_errors[i]) else std_errors[i]
            t = '' if np.isnan(t_stats[i]) else t_stats[i]
            p = '' if np.isnan(p_values[i]) else p_values[i]
            sig = '' if np.isnan(p_values[i]) else get_significance(p_values[i])

            data.append({
                'section': 'Choice_Model',
                'parameter': f'beta_{i}',
                'estimate': estimates[i],
                'std_error': se,
                't_statistic': t,
                'p_value': p,
                'significance': sig,
                'description': f'속성 {i+1}'
            })

    # theta (대안별 잠재변수 계수) - Multinomial Logit
    if 'theta' in choice_stats:
        theta_descriptions = {
            'sugar_purchase_intention': '일반당 × 구매의도',
            'sugar_nutrition_knowledge': '일반당 × 영양지식',
            'sugar_free_purchase_intention': '무설탕 × 구매의도',
            'sugar_free_nutrition_knowledge': '무설탕 × 영양지식'
        }

        for theta_name, stat in choice_stats['theta'].items():
            desc = theta_descriptions.get(theta_name, theta_name)

            # NaN 값을 빈 문자열로 변환 (고정 파라미터)
            se = '' if np.isnan(stat['std_error']) else stat['std_error']
            t = '' if np.isnan(stat['t_statistic']) else stat['t_statistic']
            p = '' if np.isnan(stat['p_value']) else stat['p_value']
            sig = '' if np.isnan(stat['p_value']) else get_significance(stat['p_value'])

            data.append({
                'section': 'Choice_Model',
                'parameter': f'theta_{theta_name}',
                'estimate': stat['estimate'],
                'std_error': se,
                't_statistic': t,
                'p_value': p,
                'significance': sig,
                'description': desc
            })

    # lambda (잠재변수 주 효과 - Binary Logit, 하위 호환성)
    if 'lambda' in choice_stats:
        lambda_descriptions = {
            'purchase_intention': '구매의도 (PI)',
            'nutrition_knowledge': '영양지식 (NK)',
            'main': '주 효과'
        }

        for lv_name, stat in choice_stats['lambda'].items():
            desc = lambda_descriptions.get(lv_name, lv_name)

            # NaN 값을 빈 문자열로 변환 (고정 파라미터)
            se = '' if np.isnan(stat['std_error']) else stat['std_error']
            t = '' if np.isnan(stat['t_statistic']) else stat['t_statistic']
            p = '' if np.isnan(stat['p_value']) else stat['p_value']
            sig = '' if np.isnan(stat['p_value']) else get_significance(stat['p_value'])

            data.append({
                'section': 'Choice_Model',
                'parameter': f'lambda_{lv_name}',
                'estimate': stat['estimate'],
                'std_error': se,
                't_statistic': t,
                'p_value': p,
                'significance': sig,
                'description': desc
            })

    # gamma (LV-Attribute 상호작용) - Multinomial Logit
    if 'gamma' in choice_stats:
        gamma_descriptions = {
            'sugar_purchase_intention_price': '일반당: PI × price',
            'sugar_purchase_intention_health_label': '일반당: PI × health_label',
            'sugar_nutrition_knowledge_health_label': '일반당: NK × health_label',
            'sugar_free_purchase_intention_price': '무설탕: PI × price',
            'sugar_free_purchase_intention_health_label': '무설탕: PI × health_label',
            'sugar_free_nutrition_knowledge_health_label': '무설탕: NK × health_label'
        }

        for gamma_name, stat in choice_stats['gamma'].items():
            desc = gamma_descriptions.get(gamma_name, gamma_name)

            # NaN 값을 빈 문자열로 변환 (고정 파라미터)
            se = '' if np.isnan(stat['std_error']) else stat['std_error']
            t = '' if np.isnan(stat['t_statistic']) else stat['t_statistic']
            p = '' if np.isnan(stat['p_value']) else stat['p_value']
            sig = '' if np.isnan(stat['p_value']) else get_significance(stat['p_value'])

            data.append({
                'section': 'Choice_Model',
                'parameter': f'gamma_{gamma_name}',
                'estimate': stat['estimate'],
                'std_error': se,
                't_statistic': t,
                'p_value': p,
                'significance': sig,
                'description': desc
            })

    return data


def _format_choice_params_flat(param_stats: Dict) -> list:
    """선택모델 파라미터 포맷팅 (순차추정용 - flat 구조)

    Args:
        param_stats: {'asc_sugar': {...}, 'beta_health_label': {...}, ...}
    """
    data = []

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
            data.append({
                'section': 'Choice_Model',
                'parameter': key,
                'estimate': stat['estimate'],
                'std_error': stat['se'],
                't_statistic': stat['t'],
                'p_value': stat['p'],
                'significance': get_significance(stat['p']),
                'description': desc
            })

    # intercept (대안별 모델이 아닌 경우)
    if 'intercept' in param_stats:
        stat = param_stats['intercept']
        data.append({
            'section': 'Choice_Model',
            'parameter': 'intercept',
            'estimate': stat['estimate'],
            'std_error': stat['se'],
            't_statistic': stat['t'],
            'p_value': stat['p'],
            'significance': get_significance(stat['p']),
            'description': '절편'
        })

    # beta (속성 계수)
    if 'beta' in param_stats:
        for attr_name, stat in param_stats['beta'].items():
            data.append({
                'section': 'Choice_Model',
                'parameter': f'beta_{attr_name}',
                'estimate': stat['estimate'],
                'std_error': stat['se'],
                't_statistic': stat['t'],
                'p_value': stat['p'],
                'significance': get_significance(stat['p']),
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
        data.append({
            'section': 'Choice_Model',
            'parameter': key,
            'estimate': stat['estimate'],
            'std_error': stat['se'],
            't_statistic': stat['t'],
            'p_value': stat['p'],
            'significance': get_significance(stat['p']),
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
            data.append({
                'section': 'Choice_Model',
                'parameter': key,
                'estimate': stat['estimate'],
                'std_error': stat['se'],
                't_statistic': stat['t'],
                'p_value': stat['p'],
                'significance': get_significance(stat['p']),
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

    for key in sorted([k for k in param_stats.keys() if k.startswith('gamma_') and '_to_' not in k]):
        stat = param_stats[key]
        desc = gamma_descriptions.get(key, key)
        data.append({
            'section': 'Choice_Model',
            'parameter': key,
            'estimate': stat['estimate'],
            'std_error': stat['se'],
            't_statistic': stat['t'],
            'p_value': stat['p'],
            'significance': get_significance(stat['p']),
            'description': desc
        })

    return data


def _format_params_without_stats(params: Dict) -> list:
    """통계량이 없는 경우 파라미터만 포맷팅"""
    data = []
    beta_names = ['sugar_free', 'health_label', 'price']

    # intercept
    if 'intercept' in params:
        data.append({
            'section': 'Choice_Model',
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
                data.append({
                    'section': 'Choice_Model',
                    'parameter': f'beta_{name}',
                    'estimate': val,
                    'std_error': '',
                    't_statistic': '',
                    'p_value': '',
                    'significance': '',
                    'description': name
                })
        else:
            data.append({
                'section': 'Choice_Model',
                'parameter': 'beta',
                'estimate': beta,
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
                'description': '속성계수'
            })

    # lambda (잠재변수 주 효과)
    lambda_descriptions = {
        'lambda_purchase_intention': '구매의도 (PI)',
        'lambda_nutrition_knowledge': '영양지식 (NK)',
        'lambda_main': '주 효과',
        'lambda_mod_perceived_price': '가격 조절',
        'lambda_mod_nutrition_knowledge': '지식 조절'
    }

    for key, desc in lambda_descriptions.items():
        if key in params:
            data.append({
                'section': 'Choice_Model',
                'parameter': key,
                'estimate': params[key],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
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

    for key, desc in gamma_descriptions.items():
        if key in params:
            data.append({
                'section': 'Choice_Model',
                'parameter': key,
                'estimate': params[key],
                'std_error': '',
                't_statistic': '',
                'p_value': '',
                'significance': '',
                'description': desc
            })

    return data

