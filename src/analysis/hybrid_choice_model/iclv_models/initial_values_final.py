"""
최종 수렴값 기반 초기값 (Iteration 8 - 2025-01-12 업데이트)

최근 로그 파일의 마지막 iteration 파라미터 값을 초기값으로 사용
측정모델 gradient 18배 과대평가 버그 수정 전 마지막 값
"""

# 측정모델 파라미터 - 요인적재량 (zeta)
# 첫 번째 지표는 1.0으로 고정되므로 파라미터 벡터에 포함되지 않음
ZETA_INITIAL_VALUES = {
    'health_concern': {
        'values': [1.9579, 1.9587, 2.1215, 2.1476, 1.9166]
    },
    'nutrition_knowledge': {
        'values': [1.3002, 1.4784, 1.2481, 1.4896, 1.5051, 1.4682, 1.3015, 1.3920, 1.4595, 1.2007, 1.4586, 1.4932, 1.5176, 1.4917, 1.4774, 1.3542, 0.9745, 1.3598, 1.2325]
    },
    'perceived_benefit': {
        'values': [1.0277, 1.2337, 1.3693, 1.2767, 1.1046]
    },
    'perceived_price': {
        'values': [1.6652, 1.7911]
    },
    'purchase_intention': {
        'values': [1.1472, 1.1700]
    },
}

# 측정모델 파라미터 - 오차분산 (sigma_sq)
# 모든 지표에 대해 설정
SIGMA_SQ_INITIAL_VALUES = {
    'health_concern': {
        'values': [0.6084, 0.5040, 0.4953, 0.5315, 0.6096, 0.4910]
    },
    'nutrition_knowledge': {
        'values': [0.4524, 0.3326, 0.3238, 0.3606, 0.3407, 0.3112, 0.3464, 0.3501, 0.3501, 0.3317, 0.3566, 0.3338, 0.3744, 0.3687, 0.3195, 0.3408, 0.3481, 0.3631, 0.3202, 0.3476]
    },
    'perceived_benefit': {
        'values': [0.6098, 0.4769, 0.5276, 0.5704, 0.5720, 0.5039]
    },
    'perceived_price': {
        'values': [0.7980, 0.7872, 0.7770]
    },
    'purchase_intention': {
        'values': [1.5412, 1.4769, 1.4672]
    },
}

# 구조모델 파라미터 (gamma)
# 계층적 경로: HC → PB, PB → PI
GAMMA_INITIAL_VALUES = {
    'health_concern_to_perceived_benefit': 0.518035,
    'perceived_benefit_to_purchase_intention': 0.498547,
}

# 선택모델 파라미터
CHOICE_INITIAL_VALUES = {
    'intercept': -0.113815,
    'beta_health_label': 0.231053,
    'beta_price': -0.295845,
    'beta_sugar_free': 0.236200,
    'lambda_main': 0.320569,
    'lambda_mod_nutrition_knowledge': 1.068264,
    'lambda_mod_perceived_price': -1.739702,
}


def get_initial_parameters_from_final():
    """
    최종 수렴값 기반 초기값을 파라미터 벡터로 변환
    
    Returns:
        dict: 각 파라미터 유형별 초기값
    """
    return {
        'zeta': ZETA_INITIAL_VALUES,
        'sigma_sq': SIGMA_SQ_INITIAL_VALUES,
        'gamma': GAMMA_INITIAL_VALUES,
        'choice': CHOICE_INITIAL_VALUES
    }


def get_zeta_initial_value(lv_name, default=1.0):
    """
    특정 잠재변수의 zeta 초기값 반환
    
    Args:
        lv_name: 잠재변수 이름
        default: 기본값 (해당 LV가 없을 경우)
    
    Returns:
        list: 초기값 리스트
    """
    if lv_name in ZETA_INITIAL_VALUES:
        return ZETA_INITIAL_VALUES[lv_name]['values']
    return [default]


def get_sigma_sq_initial_value(lv_name, default=1.0):
    """
    특정 잠재변수의 sigma_sq 초기값 반환
    
    Args:
        lv_name: 잠재변수 이름
        default: 기본값 (해당 LV가 없을 경우)
    
    Returns:
        list: 초기값 리스트
    """
    if lv_name in SIGMA_SQ_INITIAL_VALUES:
        return SIGMA_SQ_INITIAL_VALUES[lv_name]['values']
    return [default]


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
