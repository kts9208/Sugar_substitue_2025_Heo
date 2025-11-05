"""
DCE 설계 매트릭스 생성 스크립트

목적: 설문지 정보를 기반으로 6개 선택 세트의 속성 조합을 담은 설계 매트릭스 생성
출력: data/processed/dce/design_matrix.csv

설문지 정보:
- 선택 세트: 6개 (q21-q26)
- 대안: 3개 (제품 A, 제품 B, 아무것도 구매하지 않음)
- 속성: 4개 (항목, 설탕 함량, 건강 라벨, 가격)
"""

import pandas as pd
import os

def create_design_matrix():
    """
    설문지 정보 기반 DCE 설계 매트릭스 생성
    
    Returns:
        pd.DataFrame: 설계 매트릭스 (18행 × 6컬럼)
    """
    
    # 설계 매트릭스 데이터
    design_data = []
    
    # ========================================
    # 선택 세트 1 (q21)
    # ========================================
    # 제품 A: 알반당, 알반당, 건강 강조 표시 있음, ₩2,500
    design_data.append({
        'choice_set': 1,
        'alternative': 1,
        'alternative_name': '제품 A',
        'product_type': '알반당',
        'sugar_content': '알반당',
        'health_label': 1,  # 1 = 있음
        'price': 2500
    })
    
    # 제품 B: 무설탕, 무설탕, 없음, ₩2,000
    design_data.append({
        'choice_set': 1,
        'alternative': 2,
        'alternative_name': '제품 B',
        'product_type': '무설탕',
        'sugar_content': '무설탕',
        'health_label': 0,  # 0 = 없음
        'price': 2000
    })
    
    # 아무것도 구매하지 않음
    design_data.append({
        'choice_set': 1,
        'alternative': 3,
        'alternative_name': '구매안함',
        'product_type': None,
        'sugar_content': None,
        'health_label': None,
        'price': None
    })
    
    # ========================================
    # 선택 세트 2 (q22)
    # ========================================
    # 제품 A: 알반당, 알반당, 없음, ₩3,000
    design_data.append({
        'choice_set': 2,
        'alternative': 1,
        'alternative_name': '제품 A',
        'product_type': '알반당',
        'sugar_content': '알반당',
        'health_label': 0,
        'price': 3000
    })
    
    # 제품 B: 무설탕, 무설탕, 건강 강조 표시 있음, ₩2,500
    design_data.append({
        'choice_set': 2,
        'alternative': 2,
        'alternative_name': '제품 B',
        'product_type': '무설탕',
        'sugar_content': '무설탕',
        'health_label': 1,
        'price': 2500
    })
    
    # 아무것도 구매하지 않음
    design_data.append({
        'choice_set': 2,
        'alternative': 3,
        'alternative_name': '구매안함',
        'product_type': None,
        'sugar_content': None,
        'health_label': None,
        'price': None
    })
    
    # ========================================
    # 선택 세트 3 (q23)
    # ========================================
    # 제품 A: 알반당, 무설탕, 없음, ₩2,000
    design_data.append({
        'choice_set': 3,
        'alternative': 1,
        'alternative_name': '제품 A',
        'product_type': '알반당',
        'sugar_content': '무설탕',
        'health_label': 0,
        'price': 2000
    })
    
    # 제품 B: 무설탕, 무설탕, 건강 강조 표시 있음, ₩3,000
    design_data.append({
        'choice_set': 3,
        'alternative': 2,
        'alternative_name': '제품 B',
        'product_type': '무설탕',
        'sugar_content': '무설탕',
        'health_label': 1,
        'price': 3000
    })
    
    # 아무것도 구매하지 않음
    design_data.append({
        'choice_set': 3,
        'alternative': 3,
        'alternative_name': '구매안함',
        'product_type': None,
        'sugar_content': None,
        'health_label': None,
        'price': None
    })
    
    # ========================================
    # 선택 세트 4 (q24)
    # ========================================
    # 제품 A: 알반당, 알반당, 건강 강조 표시 있음, ₩2,000
    design_data.append({
        'choice_set': 4,
        'alternative': 1,
        'alternative_name': '제품 A',
        'product_type': '알반당',
        'sugar_content': '알반당',
        'health_label': 1,
        'price': 2000
    })
    
    # 제품 B: 무설탕, 알반당, 없음, ₩2,500
    design_data.append({
        'choice_set': 4,
        'alternative': 2,
        'alternative_name': '제품 B',
        'product_type': '무설탕',
        'sugar_content': '알반당',
        'health_label': 0,
        'price': 2500
    })
    
    # 아무것도 구매하지 않음
    design_data.append({
        'choice_set': 4,
        'alternative': 3,
        'alternative_name': '구매안함',
        'product_type': None,
        'sugar_content': None,
        'health_label': None,
        'price': None
    })
    
    # ========================================
    # 선택 세트 5 (q25)
    # ========================================
    # 제품 A: 알반당, 무설탕, 없음, ₩2,500
    design_data.append({
        'choice_set': 5,
        'alternative': 1,
        'alternative_name': '제품 A',
        'product_type': '알반당',
        'sugar_content': '무설탕',
        'health_label': 0,
        'price': 2500
    })
    
    # 제품 B: 무설탕, 알반당, 건강 강조 표시 있음, ₩3,000
    design_data.append({
        'choice_set': 5,
        'alternative': 2,
        'alternative_name': '제품 B',
        'product_type': '무설탕',
        'sugar_content': '알반당',
        'health_label': 1,
        'price': 3000
    })
    
    # 아무것도 구매하지 않음
    design_data.append({
        'choice_set': 5,
        'alternative': 3,
        'alternative_name': '구매안함',
        'product_type': None,
        'sugar_content': None,
        'health_label': None,
        'price': None
    })
    
    # ========================================
    # 선택 세트 6 (q26)
    # ========================================
    # 제품 A: 알반당, 알반당, 건강 강조 표시 있음, ₩3,000
    design_data.append({
        'choice_set': 6,
        'alternative': 1,
        'alternative_name': '제품 A',
        'product_type': '알반당',
        'sugar_content': '알반당',
        'health_label': 1,
        'price': 3000
    })
    
    # 제품 B: 무설탕, 무설탕, 없음, ₩2,000
    design_data.append({
        'choice_set': 6,
        'alternative': 2,
        'alternative_name': '제품 B',
        'product_type': '무설탕',
        'sugar_content': '무설탕',
        'health_label': 0,
        'price': 2000
    })
    
    # 아무것도 구매하지 않음
    design_data.append({
        'choice_set': 6,
        'alternative': 3,
        'alternative_name': '구매안함',
        'product_type': None,
        'sugar_content': None,
        'health_label': None,
        'price': None
    })
    
    # DataFrame 생성
    df_design = pd.DataFrame(design_data)
    
    return df_design


def main():
    """메인 실행 함수"""
    
    print("=" * 80)
    print("DCE 설계 매트릭스 생성")
    print("=" * 80)
    
    # 설계 매트릭스 생성
    print("\n[1] 설계 매트릭스 생성 중...")
    df_design = create_design_matrix()
    
    print(f"   - 생성 완료: {len(df_design)}행 × {len(df_design.columns)}컬럼")
    print(f"   - 선택 세트: {df_design['choice_set'].nunique()}개")
    print(f"   - 대안: {df_design['alternative'].nunique()}개")
    
    # 출력 디렉토리 생성
    output_dir = 'data/processed/dce'
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 저장
    output_path = os.path.join(output_dir, 'design_matrix.csv')
    df_design.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[2] 저장 완료: {output_path}")
    
    # 미리보기
    print("\n[3] 설계 매트릭스 미리보기:")
    print("-" * 80)
    print(df_design.to_string())
    
    # 요약 통계
    print("\n[4] 요약 통계:")
    print("-" * 80)
    print(f"   - 총 행 수: {len(df_design)}")
    print(f"   - 선택 세트: {df_design['choice_set'].unique()}")
    print(f"   - 대안: {df_design['alternative'].unique()}")
    print(f"   - 가격 범위: {df_design['price'].dropna().min():.0f} ~ {df_design['price'].dropna().max():.0f}원")
    print(f"   - 건강 라벨 있음: {df_design['health_label'].sum():.0f}개")
    print(f"   - 건강 라벨 없음: {(df_design['health_label'] == 0).sum()}개")
    
    print("\n" + "=" * 80)
    print("설계 매트릭스 생성 완료!")
    print("=" * 80)
    
    return df_design


if __name__ == "__main__":
    df_design = main()

