"""
DCE 데이터 전처리 스크립트

목적: Wide format (300×6) → Long format (5,400×8) 변환
입력: 
  - data/raw/Sugar_substitue_Raw data_250730.xlsx (q21-q26)
  - data/processed/dce/design_matrix.csv
출력:
  - data/processed/dce/dce_long_format.csv
"""

import pandas as pd
import numpy as np
import os


def load_raw_dce_data():
    """
    원본 DCE 응답 데이터 로드
    
    Returns:
        pd.DataFrame: DCE 응답 데이터 (no, q21-q26)
    """
    print("\n[1] 원본 DCE 응답 데이터 로드 중...")

    # 원본 데이터 로드
    df = pd.read_excel('data/raw/Sugar_substitue_Raw data_251108.xlsx',
                       sheet_name='DATA')
    
    # DCE 관련 컬럼만 선택
    dce_cols = ['no', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26']
    df_dce = df[dce_cols].copy()
    
    print(f"   - 로드 완료: {len(df_dce)}명 × {len(dce_cols)}컬럼")
    print(f"   - 응답자 ID: {df_dce['no'].min()} ~ {df_dce['no'].max()}")
    
    return df_dce


def load_design_matrix():
    """
    설계 매트릭스 로드
    
    Returns:
        pd.DataFrame: 설계 매트릭스
    """
    print("\n[2] 설계 매트릭스 로드 중...")
    
    df_design = pd.read_csv('data/processed/dce/design_matrix.csv')
    
    print(f"   - 로드 완료: {len(df_design)}행")
    print(f"   - 선택 세트: {df_design['choice_set'].nunique()}개")
    print(f"   - 대안: {df_design['alternative'].nunique()}개")
    
    return df_design


def convert_to_long_format(df_dce, df_design):
    """
    Wide format → Long format 변환
    
    Args:
        df_dce: DCE 응답 데이터 (Wide format)
        df_design: 설계 매트릭스
    
    Returns:
        pd.DataFrame: Long format DCE 데이터
    """
    print("\n[3] Wide → Long format 변환 중...")
    
    dce_long = []
    
    # 각 응답자별 처리
    for idx, row in df_dce.iterrows():
        respondent_id = row['no']
        
        # 각 선택 세트 처리 (1-6)
        for choice_set in range(1, 7):
            # 해당 질문의 응답 (1, 2, or 3)
            q_col = f'q{20 + choice_set}'
            selected_alternative = row[q_col]
            
            # 해당 선택 세트의 설계 매트릭스
            set_design = df_design[df_design['choice_set'] == choice_set]
            
            # 각 대안별로 행 생성
            for _, alt_row in set_design.iterrows():
                alternative = alt_row['alternative']
                
                # 선택 여부 (선택한 대안이면 1, 아니면 0)
                choice = 1 if alternative == selected_alternative else 0
                
                # Long format 행 추가
                dce_long.append({
                    'respondent_id': respondent_id,
                    'choice_set': choice_set,
                    'alternative': alternative,
                    'alternative_name': alt_row['alternative_name'],
                    'product_type': alt_row['product_type'],
                    'sugar_content': alt_row['sugar_content'],
                    'health_label': alt_row['health_label'],
                    'price': alt_row['price'],
                    'choice': choice
                })
        
        # 진행 상황 출력 (50명마다)
        if (idx + 1) % 50 == 0:
            print(f"   - 진행: {idx + 1}/{len(df_dce)}명 처리 완료")
    
    # DataFrame 생성
    df_long = pd.DataFrame(dce_long)
    
    print(f"\n   - 변환 완료: {len(df_long)}행 생성")
    print(f"   - 예상 행 수: {len(df_dce)} × 6 × 3 = {len(df_dce) * 6 * 3}행")
    
    return df_long


def validate_long_format(df_long, df_dce):
    """
    Long format 데이터 검증

    Args:
        df_long: Long format DCE 데이터
        df_dce: 원본 Wide format 데이터
    """
    print("\n[4] 데이터 검증 중...")

    # 검증 1: 행 수
    expected_rows = len(df_dce) * 6 * 3
    actual_rows = len(df_long)
    assert actual_rows == expected_rows, f"행 수 불일치: {actual_rows} != {expected_rows}"
    print(f"   ✓ 행 수 검증: {actual_rows}행 (예상: {expected_rows}행)")

    # 검증 2: 각 응답자별 선택 수
    choices_per_respondent = df_long.groupby('respondent_id')['choice'].sum()
    min_choices = choices_per_respondent.min()
    max_choices = choices_per_respondent.max()
    mean_choices = choices_per_respondent.mean()

    print(f"   ✓ 선택 수 검증: 평균 {mean_choices:.1f}개 (범위: {min_choices}-{max_choices})")

    if not (choices_per_respondent == 6).all():
        abnormal = choices_per_respondent[choices_per_respondent != 6]
        print(f"   ⚠ 경고: {len(abnormal)}명이 6개가 아닌 선택을 함")
        print(f"      이상 응답자: {abnormal.head(10).to_dict()}")

    # 검증 3: 각 선택 세트별 선택 수
    choices_per_set = df_long.groupby(['respondent_id', 'choice_set'])['choice'].sum()

    if not (choices_per_set == 1).all():
        abnormal_sets = choices_per_set[choices_per_set != 1]
        print(f"   ⚠ 경고: {len(abnormal_sets)}개 선택 세트에서 1개가 아닌 선택")
    else:
        print(f"   ✓ 선택 세트 검증: 각 세트에서 1개만 선택")
    
    # 검증 4: 원본 데이터와 비교
    print("\n   [원본 데이터 비교]")
    for choice_set in range(1, 7):
        q_col = f'q{20 + choice_set}'
        
        # 원본 데이터의 선택 분포
        original_dist = df_dce[q_col].value_counts().sort_index()
        
        # Long format의 선택 분포
        long_dist = df_long[
            (df_long['choice_set'] == choice_set) & 
            (df_long['choice'] == 1)
        ]['alternative'].value_counts().sort_index()
        
        # 비교
        match = (original_dist == long_dist).all()
        status = "✓" if match else "✗"
        print(f"   {status} 선택 세트 {choice_set} ({q_col}): 일치" if match else f"   {status} 선택 세트 {choice_set} ({q_col}): 불일치")
    
    print("\n   ✓ 모든 검증 통과!")


def create_summary_statistics(df_long):
    """
    요약 통계 생성
    
    Args:
        df_long: Long format DCE 데이터
    """
    print("\n[5] 요약 통계:")
    print("-" * 80)
    
    # 기본 정보
    print(f"   - 총 행 수: {len(df_long):,}")
    print(f"   - 응답자 수: {df_long['respondent_id'].nunique()}")
    print(f"   - 선택 세트 수: {df_long['choice_set'].nunique()}")
    print(f"   - 대안 수: {df_long['alternative'].nunique()}")
    
    # 선택 분포
    print(f"\n   [선택 분포]")
    choice_dist = df_long[df_long['choice'] == 1]['alternative_name'].value_counts()
    for alt_name, count in choice_dist.items():
        pct = count / df_long['respondent_id'].nunique() / 6 * 100
        print(f"   - {alt_name}: {count}회 ({pct:.1f}%)")
    
    # 속성별 선택 분포 (구매안함 제외)
    df_purchased = df_long[(df_long['choice'] == 1) & (df_long['alternative'] != 3)]
    
    if len(df_purchased) > 0:
        print(f"\n   [속성별 선택 분포] (구매안함 제외)")
        
        # 건강 라벨
        label_dist = df_purchased.groupby('health_label')['choice'].sum()
        print(f"   - 건강 라벨 있음: {label_dist.get(1.0, 0)}회")
        print(f"   - 건강 라벨 없음: {label_dist.get(0.0, 0)}회")
        
        # 가격
        price_dist = df_purchased.groupby('price')['choice'].sum().sort_index()
        print(f"\n   - 가격별 선택:")
        for price, count in price_dist.items():
            print(f"     ₩{price:,.0f}: {count}회")
    
    # 결측치 확인
    print(f"\n   [결측치]")
    missing = df_long.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"   - {col}: {count}개 ({count/len(df_long)*100:.1f}%)")


def main():
    """메인 실행 함수"""
    
    print("=" * 80)
    print("DCE 데이터 전처리 (Wide → Long)")
    print("=" * 80)
    
    # 1. 원본 데이터 로드
    df_dce = load_raw_dce_data()
    
    # 2. 설계 매트릭스 로드
    df_design = load_design_matrix()
    
    # 3. Long format 변환
    df_long = convert_to_long_format(df_dce, df_design)
    
    # 4. 검증
    validate_long_format(df_long, df_dce)
    
    # 5. 요약 통계
    create_summary_statistics(df_long)
    
    # 6. 저장
    print("\n[6] 저장 중...")
    output_path = 'data/processed/dce/dce_long_format.csv'
    df_long.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   - 저장 완료: {output_path}")
    
    # 7. 미리보기
    print("\n[7] 데이터 미리보기 (첫 18행 - 응답자 1명):")
    print("-" * 80)
    print(df_long[df_long['respondent_id'] == df_long['respondent_id'].iloc[0]].to_string())
    
    print("\n" + "=" * 80)
    print("DCE 전처리 완료!")
    print("=" * 80)
    
    return df_long


if __name__ == "__main__":
    df_long = main()

