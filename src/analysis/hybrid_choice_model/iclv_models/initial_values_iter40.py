"""
Iteration 40 기반 예상 수렴값 (초기값으로 사용)

이전 분석에서 예측한 파라미터 수렴값을 정리
"""

# 측정모델 파라미터 - 요인적재량 (zeta)
# 첫 번째 지표는 1.0으로 고정되므로 파라미터 벡터에 포함되지 않음
ZETA_INITIAL_VALUES = {
    'health_concern': {
        # 예상 수렴값: 1.5~2.0 (현재 1.068에서 큰 변화 예상)
        # 보수적 추정: 1.5
        'values': [1.5, 1.5, 1.5, 1.5, 1.5]  # 첫 번째 제외 5개
    },
    'perceived_benefit': {
        # 예상 수렴값: 0.7~0.9 (현재 0.461에서 증가 중)
        # 보수적 추정: 0.8
        'values': [0.8, 0.8, 0.8, 0.8, 0.8]  # 첫 번째 제외 5개
    },
    'perceived_price': {
        # 예상 수렴값: 1.3~1.6 (현재 0.915에서 증가 중)
        # 보수적 추정: 1.4
        'values': [1.4, 1.4]  # 첫 번째 제외 2개
    },
    'nutrition_knowledge': {
        # 예상 수렴값: 0.40~0.42 (거의 수렴)
        # 현재값 사용: 0.41
        'values': [0.41] * 19  # 첫 번째 제외 19개
    },
    'purchase_intention': {
        # 예상 수렴값: 0.7~0.9 (현재 0.443에서 증가 중)
        # 보수적 추정: 0.8
        'values': [0.8, 0.8]  # 첫 번째 제외 2개
    }
}

# 측정모델 파라미터 - 오차분산 (sigma_sq)
# 모든 지표에 대해 설정
SIGMA_SQ_INITIAL_VALUES = {
    'health_concern': {
        # 예상 수렴값: 0.10~0.12 (현재 0.149에서 감소 중)
        # 보수적 추정: 0.11
        'values': [0.11] * 6  # 6개 지표
    },
    'perceived_benefit': {
        # 예상 수렴값: 0.16~0.18 (현재 0.180에서 안정화)
        # 현재값 사용: 0.17
        'values': [0.17] * 6  # 6개 지표
    },
    'perceived_price': {
        # 예상 수렴값: 0.28~0.30 (현재 0.290에서 안정화)
        # 현재값 사용: 0.29
        'values': [0.29] * 3  # 3개 지표
    },
    'nutrition_knowledge': {
        # 예상 수렴값: 0.18~0.20 (안정화)
        # 보수적 추정: 0.19
        'values': [0.19] * 20  # 20개 지표
    },
    'purchase_intention': {
        # 예상 수렴값: 0.48~0.50 (현재 0.477에서 안정화)
        # 현재값 사용: 0.49
        'values': [0.49] * 3  # 3개 지표
    }
}

# 구조모델 파라미터 (gamma)
# 계층적 경로: HC → PB, PB → PI
GAMMA_INITIAL_VALUES = {
    'health_concern_to_perceived_benefit': 0.5,  # 안정화된 값
    'perceived_benefit_to_purchase_intention': 0.5  # 안정화된 값
}

# 선택모델 파라미터
CHOICE_INITIAL_VALUES = {
    # 절편
    'intercept': 0.22,  # 예상 수렴값: 0.20~0.23 (현재 0.251에서 감소 중)
    
    # 속성 계수 (beta)
    'beta_sugar_free': 0.23,  # 예상 수렴값: 0.22~0.23 (거의 수렴)
    'beta_health_label': 0.22,  # 초기값 유지 (로그에 정보 없음)
    'beta_price': -0.13,  # 예상 수렴값: -0.12~-0.15 (현재 -0.103에서 감소 중)
    
    # 잠재변수 계수 (lambda)
    'lambda_main': 0.87,  # 예상 수렴값: 0.86~0.87 (거의 수렴)
    'lambda_mod_perceived_price': -0.51,  # 예상 수렴값: -0.50~-0.52 (거의 수렴)
    'lambda_mod_nutrition_knowledge': 1.18,  # 예상 수렴값: 1.17~1.18 (거의 수렴)
}


def get_initial_parameters_from_iter40():
    """
    Iteration 40 기반 초기값을 파라미터 벡터로 변환
    
    Returns:
        dict: 각 파라미터 유형별 초기값
    """
    return {
        'zeta': ZETA_INITIAL_VALUES,
        'sigma_sq': SIGMA_SQ_INITIAL_VALUES,
        'gamma': GAMMA_INITIAL_VALUES,
        'choice': CHOICE_INITIAL_VALUES
    }


def get_zeta_initial_value(lv_name, default=0.05):
    """
    특정 잠재변수의 zeta 초기값 반환
    
    Args:
        lv_name: 잠재변수 이름
        default: 기본값 (해당 LV가 없을 경우)
    
    Returns:
        float: 초기값
    """
    if lv_name in ZETA_INITIAL_VALUES:
        # 평균값 반환 (모든 지표가 같은 값이므로)
        return ZETA_INITIAL_VALUES[lv_name]['values'][0]
    return default


def get_sigma_sq_initial_value(lv_name, default=0.03):
    """
    특정 잠재변수의 sigma_sq 초기값 반환
    
    Args:
        lv_name: 잠재변수 이름
        default: 기본값 (해당 LV가 없을 경우)
    
    Returns:
        float: 초기값
    """
    if lv_name in SIGMA_SQ_INITIAL_VALUES:
        # 평균값 반환 (모든 지표가 같은 값이므로)
        return SIGMA_SQ_INITIAL_VALUES[lv_name]['values'][0]
    return default


def get_gamma_initial_value(path_name, default=0.5):
    """
    특정 구조 경로의 gamma 초기값 반환
    
    Args:
        path_name: 경로 이름 (예: 'health_concern_to_perceived_benefit')
        default: 기본값
    
    Returns:
        float: 초기값
    """
    return GAMMA_INITIAL_VALUES.get(path_name, default)


def get_choice_initial_value(param_name, default=0.0):
    """
    선택모델 파라미터 초기값 반환
    
    Args:
        param_name: 파라미터 이름
        default: 기본값
    
    Returns:
        float: 초기값
    """
    return CHOICE_INITIAL_VALUES.get(param_name, default)

