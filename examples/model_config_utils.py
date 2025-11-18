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

def build_paths_from_config(paths_config: Dict[str, bool]) -> Tuple[Optional[List[Dict]], str, str]:
    """
    경로 설정에서 hierarchical_paths 생성
    
    Args:
        paths_config: {'HC->PB': True, ...} 형태의 딕셔너리
    
    Returns:
        hierarchical_paths: [{'target': ..., 'predictors': [...]}, ...] or None
        path_name: 파일명용 경로 이름 (예: 'HC-PB_PB-PI' 또는 'base_model')
        model_description: 모델 설명 (예: 'HC→PB + PB→PI' 또는 'Base Model (경로 없음)')
    """
    # 활성화된 경로만 필터링
    active_paths = {k: v for k, v in paths_config.items() if v}
    
    # 경로가 없으면 base_model
    if not active_paths:
        return None, "base_model", "Base Model (경로 없음)"
    
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
    
    # 파일명용 경로 이름 생성 (예: 'HC-PB_PB-PI')
    path_name = '_'.join(sorted(active_paths.keys())).replace('->', '-')
    
    # 모델 설명 생성 (예: 'HC→PB + PB→PI')
    model_description = ' + '.join(sorted(active_paths.keys())).replace('->', '→')
    
    return hierarchical_paths, path_name, model_description


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


def generate_stage2_filename(
    choice_config,
    stage1_model_name: str = None
) -> str:
    """
    선택모델 설정을 기반으로 2단계 결과 파일명 생성

    Args:
        choice_config: ChoiceConfig 또는 MultiLatentConfig 객체
        stage1_model_name: 1단계 모델 이름 (예: "HC-PB_PB-PI" 또는 "base")

    Returns:
        파일명 접두사 (예: "st2_base1_base2", "st2_HC-PB_PB-PI1_NK2")
    """
    # config가 MultiLatentConfig인 경우 choice 속성 추출
    if hasattr(choice_config, 'choice'):
        choice_config = choice_config.choice

    # 1단계 모델 이름 (기본값: "base")
    stage1_name = stage1_model_name if stage1_model_name else "base"

    # 2단계 모델 이름 생성
    # 1. 잠재변수가 없는 경우 -> base
    has_lvs = False

    # 주효과 LV 확인
    if getattr(choice_config, 'all_lvs_as_main', False):
        main_lvs = getattr(choice_config, 'main_lvs', None)
        if main_lvs and len(main_lvs) > 0:
            has_lvs = True

    # 조절효과 확인
    if getattr(choice_config, 'moderation_enabled', False):
        has_lvs = True

    # LV-Attribute 상호작용 확인 (주효과 없이 상호작용만 있을 수도 있음)
    lv_attr_interactions = getattr(choice_config, 'lv_attribute_interactions', None)
    if lv_attr_interactions and len(lv_attr_interactions) > 0:
        has_lvs = True

    # 잠재변수가 전혀 없으면 base
    if not has_lvs:
        stage2_name = "base"
    else:
        # 2. 잠재변수가 있는 경우 -> 약어 조합
        lv_abbrs = set()

        # 주효과 LV 추가
        if getattr(choice_config, 'all_lvs_as_main', False):
            main_lvs = getattr(choice_config, 'main_lvs', None)
            if main_lvs:
                for lv in main_lvs:
                    abbr = LV_ABBR.get(lv)
                    if abbr:
                        lv_abbrs.add(abbr)

        # 조절효과 LV 추가
        if getattr(choice_config, 'moderation_enabled', False):
            moderator_lvs = getattr(choice_config, 'moderator_lvs', None)
            if moderator_lvs:
                for lv in moderator_lvs:
                    abbr = LV_ABBR.get(lv)
                    if abbr:
                        lv_abbrs.add(abbr)

        # LV-Attribute 상호작용 LV 추가
        if lv_attr_interactions:
            for interaction in lv_attr_interactions:
                lv = interaction.get('lv') if isinstance(interaction, dict) else interaction[0]
                abbr = LV_ABBR.get(lv)
                if abbr:
                    lv_abbrs.add(abbr)

        # 약어를 알파벳 순으로 정렬하여 조합
        stage2_name = '_'.join(sorted(lv_abbrs))

        # 약어가 없으면 base
        if not stage2_name:
            stage2_name = "base"

    # 최종 파일명: st2_{stage1_name}1_{stage2_name}2
    filename = f"st2_{stage1_name}1_{stage2_name}2"

    return filename


def generate_simultaneous_filename(
    path_name: str,
    choice_config,
    timestamp: str = None
) -> str:
    """
    동시추정 결과 파일명 생성

    Args:
        path_name: 경로 이름 (예: "HC-PB_PB-PI" 또는 "base_model")
        choice_config: ChoiceConfig 또는 MultiLatentConfig 객체
        timestamp: 타임스탬프 (예: "20251117_123456"), None이면 타임스탬프 없음

    Returns:
        파일명 (예: "simultaneous_HC-PB_PB-PI_NK_results_20251117_123456.csv")
    """
    # config가 MultiLatentConfig인 경우 choice 속성 추출
    if hasattr(choice_config, 'choice'):
        choice_config = choice_config.choice

    # 선택모델 LV 약어 추출
    lv_abbrs = set()

    # 주효과 LV 추가
    if getattr(choice_config, 'all_lvs_as_main', False):
        main_lvs = getattr(choice_config, 'main_lvs', None)
        if main_lvs:
            for lv in main_lvs:
                abbr = LV_ABBR.get(lv)
                if abbr:
                    lv_abbrs.add(abbr)

    # 조절효과 LV 추가
    if getattr(choice_config, 'moderation_enabled', False):
        moderator_lvs = getattr(choice_config, 'moderator_lvs', None)
        if moderator_lvs:
            for lv in moderator_lvs:
                abbr = LV_ABBR.get(lv)
                if abbr:
                    lv_abbrs.add(abbr)

    # LV-Attribute 상호작용 LV 추가
    lv_attr_interactions = getattr(choice_config, 'lv_attribute_interactions', None)
    if lv_attr_interactions:
        for interaction in lv_attr_interactions:
            lv = interaction.get('lv') if isinstance(interaction, dict) else interaction[0]
            abbr = LV_ABBR.get(lv)
            if abbr:
                lv_abbrs.add(abbr)

    # 파일명 구성 요소
    parts = ['simultaneous']

    # 경로 이름 추가
    if path_name and path_name != 'base_model':
        parts.append(path_name)

    # 선택모델 LV 추가
    if lv_abbrs:
        lv_part = '_'.join(sorted(lv_abbrs))
        parts.append(lv_part)

    # 'results' 추가
    parts.append('results')

    # 타임스탬프 추가
    if timestamp:
        parts.append(timestamp)

    # 최종 파일명
    filename = '_'.join(parts) + '.csv'

    return filename

