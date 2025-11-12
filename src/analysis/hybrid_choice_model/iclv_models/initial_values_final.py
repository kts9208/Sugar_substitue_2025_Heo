"""
최종 수렴값 기반 초기값 (Iteration 24)

최근 로그 파일의 마지막 iteration 파라미터 값을 초기값으로 사용
"""

# 측정모델 파라미터 - 요인적재량 (zeta)
# 첫 번째 지표는 1.0으로 고정되므로 파라미터 벡터에 포함되지 않음
ZETA_INITIAL_VALUES = {
    'health_concern': {
        'values': [1.821545, 1.817093, 2.004141, 2.042785, 1.79169]
    },
    'nutrition_knowledge': {
        'values': [1.127794, 1.260891, 1.087865, 1.24954, 1.303558, 1.305639, 1.106603, 1.183989, 1.255195, 1.028991, 1.262609, 1.261023, 1.274052, 1.28539, 1.275248, 1.191841, 0.859048, 1.160671, 1.045778]
    },
    'perceived_benefit': {
        'values': [0.912519, 1.028369, 1.180854, 1.089028, 0.936935]
    },
    'perceived_price': {
        'values': [1.626653, 1.744699]
    },
    'purchase_intention': {
        'values': [1.033757, 1.056102]
    },
}

# 측정모델 파라미터 - 오차분산 (sigma_sq)
# 모든 지표에 대해 설정
SIGMA_SQ_INITIAL_VALUES = {
    'health_concern': {
        'values': [0.242753, 0.307487, 0.318766, 0.374413, 0.497122, 0.280641]
    },
    'nutrition_knowledge': {
        'values': [0.251034, 0.176283, 0.207327, 0.19267, 0.292034, 0.240615, 0.274186, 0.194213, 0.268437, 0.238901, 0.202389, 0.276854, 0.326, 0.331827, 0.230682, 0.283563, 0.207114, 0.187191, 0.184124, 0.191217]
    },
    'perceived_benefit': {
        'values': [0.400324, 0.347557, 0.343263, 0.391935, 0.384655, 0.338511]
    },
    'perceived_price': {
        'values': [0.625037, 0.627247, 0.614678]
    },
    'purchase_intention': {
        'values': [1.3463, 1.305794, 1.290706]
    },
}

# 구조모델 파라미터 (gamma)
# 계층적 경로: HC → PB, PB → PI
GAMMA_INITIAL_VALUES = {
    'health_concern_to_perceived_benefit': 0.510461,
    'perceived_benefit_to_purchase_intention': 0.498259,
}

# 선택모델 파라미터
CHOICE_INITIAL_VALUES = {
    'intercept': -0.020155,
    'beta_health_label': 0.226071,
    'beta_price': -0.260845,
    'beta_sugar_free': 0.232747,
    'lambda_main': 0.445158,
    'lambda_mod_nutrition_knowledge': 1.051328,
    'lambda_mod_perceived_price': -1.496759,
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
