"""
모든 가능한 선택모델 케이스 생성

2경로 및 3경로 모델에 대해 가능한 모든 선택모델 조합을 생성합니다.

사용법:
    python examples/generate_all_model_cases.py

Author: ICLV Team
Date: 2025-11-23
"""

from itertools import combinations, chain
from typing import List, Dict, Tuple


# ============================================================================
# 설정
# ============================================================================

# 사용 가능한 잠재변수 (선택모델에서)
AVAILABLE_LVS = ['purchase_intention', 'nutrition_knowledge', 'perceived_price']
LV_ABBR = {
    'purchase_intention': 'PI',
    'nutrition_knowledge': 'NK',
    'perceived_price': 'PP'
}

# 사용 가능한 속성
AVAILABLE_ATTRS = ['health_label', 'price']
ATTR_ABBR = {
    'health_label': 'label',
    'price': 'price'
}


# ============================================================================
# 조합 생성 함수
# ============================================================================

def powerset(iterable):
    """
    멱집합 생성 (공집합 포함)
    
    예: powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def generate_all_main_effect_cases() -> List[List[str]]:
    """
    모든 주효과 조합 생성
    
    Returns:
        주효과 조합 리스트 (공집합 포함)
    """
    return [list(combo) for combo in powerset(AVAILABLE_LVS)]


def generate_all_interaction_cases() -> List[List[Tuple[str, str]]]:
    """
    모든 LV-Attribute 상호작용 조합 생성
    
    Returns:
        상호작용 조합 리스트 (공집합 포함)
    """
    # 가능한 모든 상호작용
    all_interactions = [
        (lv, attr) 
        for lv in AVAILABLE_LVS 
        for attr in AVAILABLE_ATTRS
    ]
    
    return [list(combo) for combo in powerset(all_interactions)]


def filter_valid_cases(
    main_lvs: List[str],
    interactions: List[Tuple[str, str]]
) -> bool:
    """
    유효한 모델 조합인지 검증
    
    규칙:
    1. 상호작용이 있으면 해당 LV의 주효과도 있어야 함
    2. 예: PI×label이 있으면 PI 주효과 필수
    
    Args:
        main_lvs: 주효과 LV 리스트
        interactions: 상호작용 리스트
    
    Returns:
        유효하면 True, 아니면 False
    """
    if not interactions:
        return True  # 상호작용이 없으면 항상 유효
    
    # 상호작용에 사용된 LV 추출
    interaction_lvs = set(lv for lv, _ in interactions)
    
    # 모든 상호작용 LV가 주효과에 포함되어야 함
    return interaction_lvs.issubset(set(main_lvs))


def generate_all_valid_cases() -> List[Dict]:
    """
    모든 유효한 선택모델 케이스 생성
    
    Returns:
        케이스 리스트 [{'main_lvs': [...], 'interactions': [...]}, ...]
    """
    all_cases = []
    
    # 모든 주효과 조합
    main_effect_cases = generate_all_main_effect_cases()
    
    # 모든 상호작용 조합
    interaction_cases = generate_all_interaction_cases()
    
    # 모든 조합 생성 및 필터링
    for main_lvs in main_effect_cases:
        for interactions in interaction_cases:
            if filter_valid_cases(main_lvs, interactions):
                all_cases.append({
                    'main_lvs': main_lvs,
                    'interactions': interactions
                })
    
    return all_cases


def generate_theory_driven_cases() -> List[Dict]:
    """
    이론적으로 타당한 케이스만 생성 (추천)
    
    Returns:
        케이스 리스트
    """
    cases = []
    
    # 1. Base Model
    cases.append({
        'name': 'Base',
        'main_lvs': [],
        'interactions': []
    })
    
    # 2. 주효과만 (단일)
    for lv in AVAILABLE_LVS:
        cases.append({
            'name': f'{LV_ABBR[lv]}_main',
            'main_lvs': [lv],
            'interactions': []
        })
    
    # 3. 주효과만 (2개 조합)
    for lv1, lv2 in combinations(AVAILABLE_LVS, 2):
        cases.append({
            'name': f'{LV_ABBR[lv1]}_{LV_ABBR[lv2]}_main',
            'main_lvs': [lv1, lv2],
            'interactions': []
        })
    
    # 4. 주효과만 (3개 전체)
    cases.append({
        'name': 'PI_NK_PP_main',
        'main_lvs': AVAILABLE_LVS.copy(),
        'interactions': []
    })
    
    # 5. 주효과 + 상호작용 (이론적으로 타당한 조합)
    # PI 주효과 + PI×label
    cases.append({
        'name': 'PI_PIxlabel',
        'main_lvs': ['purchase_intention'],
        'interactions': [('purchase_intention', 'health_label')]
    })
    
    # NK 주효과 + NK×label
    cases.append({
        'name': 'NK_NKxlabel',
        'main_lvs': ['nutrition_knowledge'],
        'interactions': [('nutrition_knowledge', 'health_label')]
    })
    
    # PP 주효과 + PP×price
    cases.append({
        'name': 'PP_PPxprice',
        'main_lvs': ['perceived_price'],
        'interactions': [('perceived_price', 'price')]
    })
    
    # PI, NK 주효과 + PI×label + NK×label
    cases.append({
        'name': 'PI_NK_PIxlabel_NKxlabel',
        'main_lvs': ['purchase_intention', 'nutrition_knowledge'],
        'interactions': [
            ('purchase_intention', 'health_label'),
            ('nutrition_knowledge', 'health_label')
        ]
    })
    
    # PI, PP 주효과 + PI×label + PP×price
    cases.append({
        'name': 'PI_PP_PIxlabel_PPxprice',
        'main_lvs': ['purchase_intention', 'perceived_price'],
        'interactions': [
            ('purchase_intention', 'health_label'),
            ('perceived_price', 'price')
        ]
    })
    
    # NK, PP 주효과 + NK×label + PP×price
    cases.append({
        'name': 'NK_PP_NKxlabel_PPxprice',
        'main_lvs': ['nutrition_knowledge', 'perceived_price'],
        'interactions': [
            ('nutrition_knowledge', 'health_label'),
            ('perceived_price', 'price')
        ]
    })
    
    return cases


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    print("=" * 70)
    print("선택모델 케이스 생성")
    print("=" * 70)
    
    # 방안 1: 이론 기반 케이스 (추천)
    print("\n[방안 1] 이론적으로 타당한 케이스")
    theory_cases = generate_theory_driven_cases()
    print(f"총 케이스 수: {len(theory_cases)}")
    
    for i, case in enumerate(theory_cases, 1):
        print(f"\n{i}. {case.get('name', 'Unnamed')}")
        print(f"   주효과: {case['main_lvs']}")
        print(f"   상호작용: {case['interactions']}")
    
    # 방안 2: 전체 유효 케이스
    print("\n" + "=" * 70)
    print("[방안 2] 모든 유효한 케이스 (필터링 적용)")
    all_valid_cases = generate_all_valid_cases()
    print(f"총 케이스 수: {len(all_valid_cases)}")
    print(f"(상호작용이 있으면 해당 LV 주효과 필수 규칙 적용)")
    
    # 통계
    print("\n" + "=" * 70)
    print("통계")
    print("=" * 70)
    print(f"이론 기반 케이스: {len(theory_cases)}개")
    print(f"전체 유효 케이스: {len(all_valid_cases)}개")
    print(f"필터링된 케이스: {512 - len(all_valid_cases)}개")


if __name__ == "__main__":
    main()

