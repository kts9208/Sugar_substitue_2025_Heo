"""
공통 모델 설정 유틸리티

순차추정과 동시추정에서 공통으로 사용하는 설정 함수들을 제공합니다.

주요 기능:
1. 경로 설정 (PATHS 딕셔너리 → hierarchical_paths 변환)
2. 선택모델 설정 (MAIN_LVS, MODERATION_LVS, LV_ATTRIBUTE_INTERACTIONS → ChoiceConfig)
3. 파일명 생성 (모델 구조 기반 자동 파일명 생성)

Author: Sugar Substitute Research Team
Date: 2025-11-17
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ============================================================================
# 잠재변수 매핑
# ============================================================================

# 약어 → 전체 이름
LV_NAMES = {
    'HC': 'health_concern',
    'PB': 'perceived_benefit',
    'PP': 'perceived_price',
    'NK': 'nutrition_knowledge',
    'PI': 'purchase_intention'
}

# 약어 → 한글 이름
LV_KOREAN = {
    'HC': '건강관심도',
    'PB': '건강유익성',
    'PP': '가격수준',
    'NK': '영양지식',
    'PI': '구매의도'
}

# 전체 이름 → 약어 (역매핑)
LV_ABBR = {v: k for k, v in LV_NAMES.items()}

# 속성 약어 매핑
ATTR_NAMES = {
    'hl': 'health_label',
    'pr': 'price'
}

# 속성 전체 이름 → 약어 (역매핑)
ATTR_ABBR = {v: k for k, v in ATTR_NAMES.items()}


# ============================================================================
# CSV 파일명 파싱 함수
# ============================================================================

def parse_csv_filename(csv_filename: str) -> Dict:
    """
    순차추정 2단계 CSV 파일명에서 모델 설정 추출

    파일명 형식:
    - st2_{stage1_paths}1_{stage2_config}2_results.csv

    예시:
    1. st2_HC-PB_PB-PI1_NK_PI2_results.csv
       → stage1: HC->PB, PB->PI
       → stage2: NK, PI 주효과

    2. st2_HC-PB_PB-PI1_PI_int_PIxhl_NKxpr2_results.csv
       → stage1: HC->PB, PB->PI
       → stage2: PI 주효과 + PI×health_label + NK×price 상호작용

    3. st2_HC-PB_PB-PI1_NK_PI_int_PIxhl2_results.csv
       → stage1: HC->PB, PB->PI
       → stage2: NK, PI 주효과 + PI×health_label 상호작용

    Args:
        csv_filename: CSV 파일명 (예: "st2_HC-PB_PB-PI1_NK_PI2_results.csv")

    Returns:
        {
            'stage1_paths': {'HC->PB': True, 'PB->PI': True, ...},
            'main_lvs': ['nutrition_knowledge', 'purchase_intention'],
            'lv_attribute_interactions': [('purchase_intention', 'health_label'), ...]
        }
    """
    import re

    # 파일명에서 확장자 제거
    filename = csv_filename.replace('_results.csv', '').replace('.csv', '')

    # st2_{stage1}1_{stage2}2 형식 파싱
    match = re.match(r'st2_(.+?)1_(.+)2', filename)
    if not match:
        raise ValueError(f"파일명 형식이 올바르지 않습니다: {csv_filename}\n"
                        f"예상 형식: st2_{{stage1}}1_{{stage2}}2_results.csv")

    stage1_str = match.group(1)
    stage2_str = match.group(2)

    # ========================================
    # 1. Stage 1 경로 파싱
    # ========================================
    paths_config = {
        'HC->PB': False,
        'HC->PP': False,
        'HC->PI': False,
        'PB->PI': False,
        'PP->PI': False,
        'NK->PI': False,
    }

    # "HC-PB_PB-PI" → ["HC-PB", "PB-PI"]
    path_parts = stage1_str.split('_')
    for part in path_parts:
        if '-' in part:
            # "HC-PB" → "HC->PB"
            path_key = part.replace('-', '->')
            if path_key in paths_config:
                paths_config[path_key] = True

    # ========================================
    # 2. Stage 2 선택모델 설정 파싱
    # ========================================
    main_lvs = []
    lv_attribute_interactions = []

    # "_int_" 구분자로 주효과와 상호작용 분리
    if '_int_' in stage2_str:
        parts = stage2_str.split('_int_')
        main_part = parts[0]  # 주효과 부분
        interaction_part = parts[1] if len(parts) > 1 else ""  # 상호작용 부분
    else:
        main_part = stage2_str
        interaction_part = ""

    # 2-1. 주효과 LV 파싱
    # "NK_PI" → ["NK", "PI"]
    if main_part and main_part != "base":
        lv_abbrs = main_part.split('_')
        for abbr in lv_abbrs:
            if abbr in LV_NAMES:
                main_lvs.append(LV_NAMES[abbr])

    # 2-2. LV-Attribute 상호작용 파싱
    # "PIxhl_NKxpr" → [("PI", "hl"), ("NK", "pr")]
    if interaction_part:
        # 언더스코어로 분리된 각 상호작용 파싱
        interaction_items = interaction_part.split('_')
        for item in interaction_items:
            # "PIxhl" → ("PI", "hl")
            if 'x' in item:
                lv_abbr, attr_abbr = item.split('x', 1)

                # 약어를 전체 이름으로 변환
                if lv_abbr in LV_NAMES and attr_abbr in ATTR_NAMES:
                    lv_name = LV_NAMES[lv_abbr]
                    attr_name = ATTR_NAMES[attr_abbr]
                    lv_attribute_interactions.append((lv_name, attr_name))

    return {
        'stage1_paths': paths_config,
        'main_lvs': main_lvs,
        'lv_attribute_interactions': lv_attribute_interactions,
        'moderation_lvs': []  # 현재는 조절효과 미지원
    }


def parse_csv_content(csv_filepath: str) -> Dict:
    """
    CSV 파일 내용을 파싱하여 모델 설정 추출

    파일명만으로는 상호작용 정보를 알 수 없는 경우,
    CSV 파일 내용(파라미터 이름)을 분석하여 설정 추출

    Args:
        csv_filepath: CSV 파일 전체 경로

    Returns:
        {
            'main_lvs': [...],
            'lv_attribute_interactions': [...]
        }
    """
    import pandas as pd
    import re

    # CSV 파일 읽기
    df = pd.read_csv(csv_filepath)

    # Parameters 섹션만 필터링
    params_df = df[df['section'] == 'Parameters']

    main_lvs = set()
    lv_attribute_interactions = set()

    # 파라미터 이름 분석
    for param_name in params_df['parameter']:
        # theta_sugar_purchase_intention → PI 주효과
        # theta_sugar_free_nutrition_knowledge → NK 주효과
        if param_name.startswith('theta_'):
            # "theta_sugar_purchase_intention" → "purchase_intention"
            # "theta_sugar_free_nutrition_knowledge" → "nutrition_knowledge"

            # sugar_free 먼저 제거
            if 'sugar_free' in param_name:
                lv_name = param_name.replace('theta_sugar_free_', '')
            elif 'sugar' in param_name:
                lv_name = param_name.replace('theta_sugar_', '')
            else:
                lv_name = param_name.replace('theta_', '')

            main_lvs.add(lv_name)

        # gamma_sugar_purchase_intention_health_label → (purchase_intention, health_label) 상호작용
        # gamma_sugar_free_nutrition_knowledge_price → (nutrition_knowledge, price) 상호작용
        elif param_name.startswith('gamma_'):
            # sugar_free 먼저 제거
            if 'sugar_free' in param_name:
                remaining = param_name.replace('gamma_sugar_free_', '')
            elif 'sugar' in param_name:
                remaining = param_name.replace('gamma_sugar_', '')
            else:
                remaining = param_name.replace('gamma_', '')

            # "purchase_intention_health_label" → ("purchase_intention", "health_label")
            # "nutrition_knowledge_price" → ("nutrition_knowledge", "price")

            # 알려진 속성 이름으로 분리
            if remaining.endswith('_health_label'):
                lv_name = remaining.replace('_health_label', '')
                attr_name = 'health_label'
                lv_attribute_interactions.add((lv_name, attr_name))
            elif remaining.endswith('_price'):
                lv_name = remaining.replace('_price', '')
                attr_name = 'price'
                lv_attribute_interactions.add((lv_name, attr_name))

    return {
        'main_lvs': sorted(list(main_lvs)),
        'lv_attribute_interactions': sorted(list(lv_attribute_interactions))
    }


def validate_csv_config_match(
    csv_filename: str,
    csv_filepath: str,
    paths_config: Dict[str, bool],
    main_lvs: List[str],
    lv_attribute_interactions: List[Tuple[str, str]],
    moderation_lvs: List[Tuple[str, str]] = None
) -> bool:
    """
    CSV 파일명/내용과 현재 설정이 일치하는지 검증

    Args:
        csv_filename: CSV 파일명
        csv_filepath: CSV 파일 전체 경로
        paths_config: 경로 설정 딕셔너리
        main_lvs: 주효과 LV 리스트
        lv_attribute_interactions: LV-Attribute 상호작용 리스트
        moderation_lvs: 조절효과 리스트 (선택)

    Returns:
        True if match, raises ValueError if mismatch
    """
    # 1. CSV 파일명에서 설정 파싱
    parsed_filename = parse_csv_filename(csv_filename)

    # 2. CSV 파일 내용에서 설정 파싱 (더 정확함)
    parsed_content = parse_csv_content(csv_filepath)

    # 3. Stage 1 경로 검증 (파일명 기반)
    if parsed_filename['stage1_paths'] != paths_config:
        raise ValueError(
            f"[ERROR] CSV 파일명과 경로 설정이 불일치합니다!\n"
            f"  CSV 파일명: {csv_filename}\n"
            f"  CSV 경로: {[k for k, v in parsed_filename['stage1_paths'].items() if v]}\n"
            f"  현재 설정: {[k for k, v in paths_config.items() if v]}"
        )

    # 4. 주효과 LV 검증 (파일 내용 기반)
    parsed_main = set(parsed_content['main_lvs'])
    config_main = set(main_lvs)
    if parsed_main != config_main:
        raise ValueError(
            f"[ERROR] CSV 파일 내용과 주효과 LV 설정이 불일치합니다!\n"
            f"  CSV 파일: {csv_filename}\n"
            f"  CSV 주효과 (파일 내용): {parsed_content['main_lvs']}\n"
            f"  현재 설정: {main_lvs}"
        )

    # 5. LV-Attribute 상호작용 검증 (파일 내용 기반)
    parsed_interactions = set(parsed_content['lv_attribute_interactions'])
    config_interactions = set(lv_attribute_interactions)
    if parsed_interactions != config_interactions:
        raise ValueError(
            f"[ERROR] CSV 파일 내용과 LV-Attribute 상호작용 설정이 불일치합니다!\n"
            f"  CSV 파일: {csv_filename}\n"
            f"  CSV 상호작용 (파일 내용): {parsed_content['lv_attribute_interactions']}\n"
            f"  현재 설정: {lv_attribute_interactions}"
        )

    return True


# ============================================================================
# 경로 설정 함수
# ============================================================================

def build_paths_from_config(paths_config: Dict[str, bool]) -> Tuple[Optional[List[Dict]], str, str, int]:
    """
    경로 설정에서 hierarchical_paths 생성

    Args:
        paths_config: {'HC->PB': True, ...} 형태의 딕셔너리

    Returns:
        hierarchical_paths: [{'target': ..., 'predictors': [...]}, ...] or None
        path_name: 파일명용 경로 이름 (예: '2path' 또는 '3path')
        model_description: 모델 설명 (예: 'HC→PB + PB→PI' 또는 'Base Model (경로 없음)')
        n_paths: 경로 개수
    """
    # 활성화된 경로만 필터링
    active_paths = {k: v for k, v in paths_config.items() if v}

    # 경로 개수
    n_paths = len(active_paths)

    # 경로가 없으면 base_model
    if not active_paths:
        return None, "base_model", "Base Model (경로 없음)", 0

    # 경로를 target별로 그룹화
    target_predictors = {}

    for path_str in active_paths.keys():
        # 'HC->PB' 형태를 파싱
        parts = path_str.split('->')
        if len(parts) != 2:
            raise ValueError(f"잘못된 경로 형식: {path_str}. 'LV1->LV2' 형태여야 합니다.")

        predictor_abbr, target_abbr = parts
        predictor = LV_NAMES.get(predictor_abbr)
        target = LV_NAMES.get(target_abbr)

        if predictor is None or target is None:
            raise ValueError(f"알 수 없는 잠재변수: {path_str}")

        if target not in target_predictors:
            target_predictors[target] = []
        target_predictors[target].append(predictor)

    # hierarchical_paths 생성
    hierarchical_paths = []
    for target, predictors in target_predictors.items():
        hierarchical_paths.append({
            'target': target,
            'predictors': predictors
        })

    # 파일명용 경로 이름 생성 (예: '2path', '3path')
    path_name = f"{n_paths}path"

    # 모델 설명 생성 (예: 'HC→PB + PB→PI')
    model_description = ' + '.join(sorted(active_paths.keys())).replace('->', '→')

    return hierarchical_paths, path_name, model_description, n_paths


# ============================================================================
# 선택모델 설정 함수
# ============================================================================

def build_choice_config_dict(
    main_lvs: List[str] = None,
    lv_attribute_interactions: List[Tuple[str, str]] = None
) -> Dict:
    """
    선택모델 설정 딕셔너리 생성 (유연한 리스트 기반)

    ✅ 핵심 원칙: 플래그 없이 리스트만으로 모든 모델 표현

    Args:
        main_lvs: 주효과 잠재변수 리스트 (빈 리스트 = Base Model)
        lv_attribute_interactions: LV-Attribute 상호작용 리스트 [(lv_name, attr_name), ...]

    Returns:
        ChoiceConfig에 전달할 딕셔너리

    Examples:
        >>> # Base Model (잠재변수 없음)
        >>> build_choice_config_dict(main_lvs=[], lv_attribute_interactions=[])

        >>> # 주효과 모델
        >>> build_choice_config_dict(main_lvs=['purchase_intention', 'nutrition_knowledge'])

        >>> # LV-Attribute 상호작용 모델
        >>> build_choice_config_dict(
        ...     main_lvs=['purchase_intention', 'nutrition_knowledge'],
        ...     lv_attribute_interactions=[('purchase_intention', 'price'), ...]
        ... )
    """
    # 기본값 설정
    main_lvs = main_lvs if main_lvs is not None else []
    lv_attribute_interactions = lv_attribute_interactions or []

    # LV-속성 상호작용을 딕셔너리 형태로 변환
    lv_attr_config = [{'lv': pair[0], 'attribute': pair[1]} for pair in lv_attribute_interactions]

    # ✅ 유연한 리스트 기반: 플래그 없이 리스트만 반환
    config_dict = {
        'main_lvs': main_lvs,
        'lv_attribute_interactions': lv_attr_config
    }

    return config_dict


# ============================================================================
# 결과 저장 함수
# ============================================================================

def _get_significance(p_value) -> str:
    """p-value에서 유의성 기호 반환"""
    import math

    # p_value가 문자열인 경우 float로 변환
    if isinstance(p_value, str):
        try:
            p_value = float(p_value)
        except (ValueError, TypeError):
            return ''

    # None이거나 NaN인 경우
    if p_value is None:
        return ''

    # float인 경우 NaN/Inf 체크
    if isinstance(p_value, (float, int)):
        if math.isnan(float(p_value)) or math.isinf(float(p_value)):
            return ''

    # 유의성 판단
    try:
        p_value = float(p_value)
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    except (ValueError, TypeError):
        return ''


def save_iclv_results(
    results: Dict,
    save_path: Path,
    estimation_type: str = 'sequential',
    cfa_results: Dict = None,
    config = None
) -> None:
    """
    ICLV 모델 결과를 CSV 파일로 저장 (순차추정 및 동시추정 통합)

    Args:
        results: 추정 결과 딕셔너리
            - estimate_stage2_only() 또는 estimate_stage2()의 반환값 (순차추정)
            - 동시추정 결과 딕셔너리 (동시추정)
        save_path: 저장할 CSV 파일 경로
        estimation_type: 'sequential' (순차추정) 또는 'simultaneous' (동시추정)
        cfa_results: CFA 결과 딕셔너리 (동시추정에서 측정모델 파라미터 추출용)
        config: MultiLatentConfig 객체 (동시추정에서 측정모델 파라미터 추출용)

    Note:
        - 순차추정: 선택모델 파라미터만 저장
        - 동시추정: 측정모델 + 구조모델 + 선택모델 파라미터 모두 저장
    """
    import pandas as pd
    import numpy as np

    # 통합 결과 저장 (적합도 + 파라미터)
    combined_data = []

    # ========================================================================
    # 동시추정: 측정모델 파라미터 추가
    # ========================================================================
    if estimation_type == 'simultaneous' and cfa_results and config:
        # CFA 결과에서 측정모델 파라미터 추출
        if 'loadings' in cfa_results and 'measurement_errors' in cfa_results:
            loadings_df = cfa_results['loadings']
            errors_df = cfa_results['measurement_errors']

            # 요인적재량 (loading)
            for _, row in loadings_df.iterrows():
                indicator = row['lval']
                lv_name = row['rval']
                combined_data.append({
                    'section': 'Measurement_Model',
                    'parameter': f'zeta_{lv_name}_{indicator}',
                    'estimate': row['Estimate'],
                    'std_error': row['Std. Err'] if pd.notna(row['Std. Err']) else '',
                    't_statistic': '',
                    'p_value': row['p-value'] if pd.notna(row['p-value']) else '',
                    'significance': _get_significance(row['p-value']) if pd.notna(row['p-value']) else '',
                    'description': f'{lv_name} → {indicator} (요인적재량)'
                })

            # 오차분산 (error_variance)
            for _, row in errors_df.iterrows():
                indicator = row['lval']
                # lval 형식: "q10~~q10" -> "q10"으로 변환
                indicator_clean = indicator.split('~~')[0]

                # 해당 지표가 어느 잠재변수에 속하는지 찾기
                lv_name = None
                for lv, lv_config in config.measurement_configs.items():
                    if indicator_clean in lv_config.indicators:
                        lv_name = lv
                        break

                if lv_name:
                    combined_data.append({
                        'section': 'Measurement_Model',
                        'parameter': f'sigma_sq_{lv_name}_{indicator_clean}',
                        'estimate': row['Estimate'],
                        'std_error': row['Std. Err'] if pd.notna(row['Std. Err']) else '',
                        't_statistic': '',
                        'p_value': row['p-value'] if pd.notna(row['p-value']) else '',
                        'significance': _get_significance(row['p-value']) if pd.notna(row['p-value']) else '',
                        'description': f'{lv_name}_{indicator_clean} 오차분산'
                    })

        # 구조모델 파라미터 추가 (동시추정)
        if 'parameter_statistics' in results and 'structural' in results['parameter_statistics']:
            struct = results['parameter_statistics']['structural']
            for key, value in struct.items():
                # ✅ 모든 구조모델 파라미터 저장 (gamma로 시작하는 것만 아니라 모두)
                combined_data.append({
                    'section': 'Structural_Model',
                    'parameter': key,
                    'estimate': value['estimate'],
                    'std_error': value.get('std_error', ''),
                    't_statistic': value.get('t', value.get('t_statistic', '')),
                    'p_value': value.get('p_value', ''),
                    'significance': _get_significance(value.get('p_value', 1.0)) if 'p_value' in value else '',
                    'description': key.replace('gamma_', '').replace('_', ' → ')
                })

        # ✅ 선택모델 파라미터 추가 (동시추정 - 계층 구조)
        if 'parameter_statistics' in results and 'choice' in results['parameter_statistics']:
            choice = results['parameter_statistics']['choice']

            # ASC 파라미터
            asc_descriptions = {
                'asc_sugar': '일반당 상수',
                'asc_sugar_free': '무설탕 상수',
                'asc_A': '대안 A 상수',
                'asc_B': '대안 B 상수'
            }

            for key in sorted([k for k in choice.keys() if k.startswith('asc_')]):
                value = choice[key]
                combined_data.append({
                    'section': 'Choice_Model',
                    'parameter': key,
                    'estimate': value['estimate'],
                    'std_error': value.get('std_error', ''),
                    't_statistic': value.get('t', value.get('t_statistic', '')),
                    'p_value': value.get('p_value', ''),
                    'significance': _get_significance(value.get('p_value', 1.0)) if 'p_value' in value else '',
                    'description': asc_descriptions.get(key, key)
                })

            # Beta 파라미터 (속성 계수)
            for key in sorted([k for k in choice.keys() if k.startswith('beta_')]):
                value = choice[key]
                attr_name = key.replace('beta_', '')
                combined_data.append({
                    'section': 'Choice_Model',
                    'parameter': key,
                    'estimate': value['estimate'],
                    'std_error': value.get('std_error', ''),
                    't_statistic': value.get('t', value.get('t_statistic', '')),
                    'p_value': value.get('p_value', ''),
                    'significance': _get_significance(value.get('p_value', 1.0)) if 'p_value' in value else '',
                    'description': attr_name
                })

            # Theta 파라미터 (대안별 잠재변수 계수)
            theta_descriptions = {
                'theta_sugar_purchase_intention': '일반당 × 구매의도',
                'theta_sugar_nutrition_knowledge': '일반당 × 영양지식',
                'theta_sugar_perceived_price': '일반당 × 가격수준',
                'theta_sugar_perceived_benefit': '일반당 × 건강유익성',
                'theta_sugar_free_purchase_intention': '무설탕 × 구매의도',
                'theta_sugar_free_nutrition_knowledge': '무설탕 × 영양지식',
                'theta_sugar_free_perceived_price': '무설탕 × 가격수준',
                'theta_sugar_free_perceived_benefit': '무설탕 × 건강유익성',
            }

            for key in sorted([k for k in choice.keys() if k.startswith('theta_')]):
                value = choice[key]
                combined_data.append({
                    'section': 'Choice_Model',
                    'parameter': key,
                    'estimate': value['estimate'],
                    'std_error': value.get('std_error', ''),
                    't_statistic': value.get('t', value.get('t_statistic', '')),
                    'p_value': value.get('p_value', ''),
                    'significance': _get_significance(value.get('p_value', 1.0)) if 'p_value' in value else '',
                    'description': theta_descriptions.get(key, key)
                })

            # Gamma 파라미터 (LV-Attribute 상호작용)
            gamma_descriptions = {
                'gamma_sugar_purchase_intention_price': '일반당: PI × price',
                'gamma_sugar_purchase_intention_health_label': '일반당: PI × health_label',
                'gamma_sugar_nutrition_knowledge_price': '일반당: NK × price',
                'gamma_sugar_nutrition_knowledge_health_label': '일반당: NK × health_label',
                'gamma_sugar_perceived_price_price': '일반당: PP × price',
                'gamma_sugar_perceived_price_health_label': '일반당: PP × health_label',
                'gamma_sugar_free_purchase_intention_price': '무설탕: PI × price',
                'gamma_sugar_free_purchase_intention_health_label': '무설탕: PI × health_label',
                'gamma_sugar_free_nutrition_knowledge_price': '무설탕: NK × price',
                'gamma_sugar_free_nutrition_knowledge_health_label': '무설탕: NK × health_label',
                'gamma_sugar_free_perceived_price_price': '무설탕: PP × price',
                'gamma_sugar_free_perceived_price_health_label': '무설탕: PP × health_label'
            }

            for key in sorted([k for k in choice.keys() if k.startswith('gamma_')]):
                value = choice[key]
                combined_data.append({
                    'section': 'Choice_Model',
                    'parameter': key,
                    'estimate': value['estimate'],
                    'std_error': value.get('std_error', ''),
                    't_statistic': value.get('t', value.get('t_statistic', '')),
                    'p_value': value.get('p_value', ''),
                    'significance': _get_significance(value.get('p_value', 1.0)) if 'p_value' in value else '',
                    'description': gamma_descriptions.get(key, key)
                })

    # ========================================================================
    # 적합도 지수 (공통)
    # ========================================================================

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

    # ========================================================================
    # 선택모델 파라미터 (공통)
    # ========================================================================
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
            'theta_sugar_perceived_price': '일반당 × 가격수준',
            'theta_sugar_perceived_benefit': '일반당 × 건강유익성',
            'theta_sugar_free_purchase_intention': '무설탕 × 구매의도',
            'theta_sugar_free_nutrition_knowledge': '무설탕 × 영양지식',
            'theta_sugar_free_perceived_price': '무설탕 × 가격수준',
            'theta_sugar_free_perceived_benefit': '무설탕 × 건강유익성',
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
            'gamma_sugar_nutrition_knowledge_price': '일반당: NK × price',
            'gamma_sugar_nutrition_knowledge_health_label': '일반당: NK × health_label',
            'gamma_sugar_perceived_price_price': '일반당: PP × price',
            'gamma_sugar_perceived_price_health_label': '일반당: PP × health_label',
            'gamma_sugar_free_purchase_intention_price': '무설탕: PI × price',
            'gamma_sugar_free_purchase_intention_health_label': '무설탕: PI × health_label',
            'gamma_sugar_free_nutrition_knowledge_price': '무설탕: NK × price',
            'gamma_sugar_free_nutrition_knowledge_health_label': '무설탕: NK × health_label',
            'gamma_sugar_free_perceived_price_price': '무설탕: PP × price',
            'gamma_sugar_free_perceived_price_health_label': '무설탕: PP × health_label'
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
            'gamma_sugar_nutrition_knowledge_price': '일반당: NK × price',
            'gamma_sugar_nutrition_knowledge_health_label': '일반당: NK × health_label',
            'gamma_sugar_perceived_price_price': '일반당: PP × price',
            'gamma_sugar_perceived_price_health_label': '일반당: PP × health_label',
            'gamma_sugar_free_purchase_intention_price': '무설탕: PI × price',
            'gamma_sugar_free_purchase_intention_health_label': '무설탕: PI × health_label',
            'gamma_sugar_free_nutrition_knowledge_price': '무설탕: NK × price',
            'gamma_sugar_free_nutrition_knowledge_health_label': '무설탕: NK × health_label',
            'gamma_sugar_free_perceived_price_price': '무설탕: PP × price',
            'gamma_sugar_free_perceived_price_health_label': '무설탕: PP × health_label'
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
    combined_df.to_csv(save_path, index=False, encoding='utf-8-sig')


def save_stage2_results(results: Dict, save_path: Path) -> None:
    """
    순차추정 2단계 결과를 CSV 파일로 저장 (하위 호환용)

    Args:
        results: estimate_stage2_only() 또는 estimate_stage2()의 반환값
        save_path: 저장할 CSV 파일 경로

    Note:
        이 함수는 하위 호환성을 위해 유지됩니다.
        새로운 코드에서는 save_iclv_results()를 사용하세요.
    """
    save_iclv_results(results, save_path, estimation_type='sequential')


def save_simultaneous_results(
    results: Dict,
    save_path: Path,
    cfa_results: Dict,
    config
) -> None:
    """
    동시추정 결과를 CSV 파일로 저장

    Args:
        results: 동시추정 결과 딕셔너리
        save_path: 저장할 CSV 파일 경로
        cfa_results: CFA 결과 딕셔너리 (측정모델 파라미터 추출용)
        config: MultiLatentConfig 객체

    Note:
        내부적으로 save_iclv_results()를 호출합니다.
    """
    save_iclv_results(
        results,
        save_path,
        estimation_type='simultaneous',
        cfa_results=cfa_results,
        config=config
    )


# ============================================================================
# 파일명 생성 함수
# ============================================================================

def extract_stage1_model_name(stage1_filename: str) -> str:
    """
    1단계 결과 파일명에서 모델 이름 추출

    Args:
        stage1_filename: 1단계 결과 파일명 (예: "stage1_HC-PB_PB-PI_results.pkl")

    Returns:
        모델 이름 (예: "HC-PB_PB-PI" 또는 "base")
    """
    # 파일명에서 확장자 제거
    name = stage1_filename.replace('.pkl', '')

    # "stage1_" 제거
    if name.startswith('stage1_'):
        name = name[7:]  # len('stage1_') = 7

    # "_results" 제거
    if name.endswith('_results'):
        name = name[:-8]  # len('_results') = 8

    # 빈 문자열이거나 "base_model"이면 "base"로 변환
    if not name or name == 'base_model':
        return 'base'

    return name


def generate_iclv_filename(
    choice_config,
    stage1_model_name: str = None,
    estimation_type: str = 'sequential'
) -> str:
    """
    ICLV 모델 결과 파일명 생성 (순차추정 및 동시추정 통합)

    Args:
        choice_config: ChoiceConfig 또는 MultiLatentConfig 객체
        stage1_model_name: 1단계 모델 이름 (예: "2path", "3path", "HC-PB_PB-PI")
        estimation_type: 'sequential' (순차추정) 또는 'simultaneous' (동시추정)

    Returns:
        파일명 접두사
        - 순차추정: "st2_2path_PI_NK_PP"
        - 동시추정: "simul_2path_PI_NK_PP"

    Examples:
        >>> # 순차추정
        >>> generate_iclv_filename(config, "2path", "sequential")
        'st2_2path_NK_PI'

        >>> # 동시추정
        >>> generate_iclv_filename(config, "2path", "simultaneous")
        'simul_2path_NK_PI'
    """
    # config가 MultiLatentConfig인 경우 choice 속성 추출
    if hasattr(choice_config, 'choice'):
        choice_config = choice_config.choice

    # 1단계 모델 이름 (기본값: "base")
    stage1_name = stage1_model_name if stage1_model_name else "base"

    # 2단계 모델 이름 생성
    parts = []

    # 주효과 LV 추가
    main_lvs = getattr(choice_config, 'main_lvs', None)
    if main_lvs and len(main_lvs) > 0:
        main_abbrs = []
        for lv in main_lvs:
            abbr = LV_ABBR.get(lv)
            if abbr:
                main_abbrs.append(abbr)
        if main_abbrs:
            parts.extend(sorted(main_abbrs))

    # LV-Attribute 상호작용 추가
    lv_attr_interactions = getattr(choice_config, 'lv_attribute_interactions', None)
    if lv_attr_interactions:
        for interaction in lv_attr_interactions:
            lv = interaction.get('lv') if isinstance(interaction, dict) else interaction[0]
            attr = interaction.get('attribute') if isinstance(interaction, dict) else interaction[1]

            lv_abbr = LV_ABBR.get(lv)
            attr_abbr = ATTR_ABBR.get(attr)

            if lv_abbr and attr_abbr:
                parts.append(f"{lv_abbr}x{attr_abbr}")

    # 조절효과 추가 (필요시)
    if getattr(choice_config, 'moderation_enabled', False):
        moderator_lvs = getattr(choice_config, 'moderator_lvs', None)
        if moderator_lvs:
            for lv in moderator_lvs:
                abbr = LV_ABBR.get(lv)
                if abbr:
                    parts.append(f"{abbr}_mod")

    # 추정 방법에 따른 접두사 선택
    prefix = 'st2' if estimation_type == 'sequential' else 'simul'

    # 최종 파일명: {prefix}_{stage1_name}_{parts}
    if parts:
        stage2_name = '_'.join(parts)
        filename = f"{prefix}_{stage1_name}_{stage2_name}"
    else:
        # 잠재변수가 전혀 없으면 base만
        filename = f"{prefix}_{stage1_name}_base"

    return filename


def generate_stage2_filename(
    choice_config,
    stage1_model_name: str = None
) -> str:
    """
    순차추정 2단계 결과 파일명 생성 (하위 호환용)

    Args:
        choice_config: ChoiceConfig 또는 MultiLatentConfig 객체
        stage1_model_name: 1단계 모델 이름 (예: "2path", "3path")

    Returns:
        파일명 접두사 (예: "st2_2path_PI_NK_PP", "st2_3path_PI_NKxlabel_PPxprice")

    Note:
        이 함수는 하위 호환성을 위해 유지됩니다.
        새로운 코드에서는 generate_iclv_filename()을 사용하세요.
    """
    return generate_iclv_filename(choice_config, stage1_model_name, 'sequential')


def generate_simultaneous_filename(
    choice_config,
    stage1_model_name: str = None,
    include_timestamp: bool = False,
    timestamp: str = None
) -> str:
    """
    동시추정 결과 파일명 생성 (하위 호환용)

    Args:
        choice_config: ChoiceConfig 또는 MultiLatentConfig 객체
        stage1_model_name: 1단계 모델 이름 (예: "2path", "HC-PB_PB-PI")
        include_timestamp: 타임스탬프 포함 여부 (기본값: False)
        timestamp: 타임스탬프 (예: "20251117_123456")

    Returns:
        파일명 (예: "simul_2path_NK_PI_results.csv")

    Note:
        이 함수는 하위 호환성을 위해 유지됩니다.
        새로운 코드에서는 generate_iclv_filename()을 사용하세요.

        기존 시그니처와의 호환성:
        - path_name → stage1_model_name으로 변경
        - 타임스탬프는 기본적으로 제거 (include_timestamp=False)
    """
    filename_prefix = generate_iclv_filename(choice_config, stage1_model_name, 'simultaneous')

    # 타임스탬프 추가 (선택적)
    if include_timestamp and timestamp:
        filename = f"{filename_prefix}_results_{timestamp}.csv"
    else:
        filename = f"{filename_prefix}_results.csv"

    return filename

