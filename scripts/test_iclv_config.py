"""
ICLV 설정 테스트 스크립트

목적: 5개 잠재변수 ICLV 설정 검증
"""

import pandas as pd
import os


def test_integrated_data():
    """통합 데이터 검증"""
    
    print("=" * 80)
    print("ICLV 통합 데이터 검증 (5개 잠재변수)")
    print("=" * 80)
    
    # 데이터 로드
    print("\n[1] 데이터 로드...")
    df = pd.read_csv('data/processed/iclv/integrated_data.csv')
    
    print(f"   - 총 행 수: {len(df):,}")
    print(f"   - 총 컬럼 수: {len(df.columns)}")
    print(f"   - 응답자 수: {df['respondent_id'].nunique()}")
    
    # 5개 잠재변수 지표 확인
    print("\n[2] 잠재변수 지표 확인...")
    
    latent_variables = {
        '건강관심도': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        '건강유익성': ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
        '구매의도': ['q18', 'q19', 'q20'],
        '가격수준': ['q27', 'q28', 'q29'],
        '영양지식': [f'q{i}' for i in range(30, 50)]
    }
    
    total_indicators = 0
    for lv_name, indicators in latent_variables.items():
        # 모든 지표가 존재하는지 확인
        missing = [ind for ind in indicators if ind not in df.columns]
        
        if missing:
            print(f"   ✗ {lv_name}: {len(missing)}개 지표 누락 - {missing}")
        else:
            print(f"   ✓ {lv_name}: {len(indicators)}개 지표 모두 존재")
            total_indicators += len(indicators)
    
    print(f"\n   총 {total_indicators}개 지표 확인 완료")
    
    # DCE 변수 확인
    print("\n[3] DCE 변수 확인...")
    dce_vars = ['choice_set', 'alternative', 'health_label', 'price', 'choice']
    
    for var in dce_vars:
        if var in df.columns:
            print(f"   ✓ {var}")
        else:
            print(f"   ✗ {var} 누락")
    
    # 구조모델 변수 확인
    print("\n[4] 구조모델 변수 확인...")
    structural_vars = ['age_std', 'gender', 'income_std', 'education_level']
    
    for var in structural_vars:
        if var in df.columns:
            missing = df[var].isnull().sum()
            print(f"   ✓ {var} (결측치: {missing}개)")
        else:
            print(f"   ✗ {var} 누락")
    
    # 결측치 요약
    print("\n[5] 결측치 요약...")
    
    # 잠재변수 지표 결측치
    all_indicators = []
    for indicators in latent_variables.values():
        all_indicators.extend(indicators)
    
    indicator_missing = df[all_indicators].isnull().sum().sum()
    print(f"   - 잠재변수 지표 ({len(all_indicators)}개): {indicator_missing}개 결측치")
    
    # DCE 변수 결측치 (구매안함 제외)
    df_choice = df[df['alternative'] != 3].copy()
    dce_missing = df_choice[['health_label', 'price']].isnull().sum().sum()
    print(f"   - DCE 변수 (구매안함 제외): {dce_missing}개 결측치")
    
    # 구조모델 변수 결측치
    structural_missing = df[structural_vars].isnull().sum().sum()
    print(f"   - 구조모델 변수: {structural_missing}개 결측치")
    
    # ICLV 추정용 데이터 준비
    print("\n[6] ICLV 추정용 데이터 준비...")
    
    # 구매안함 제외
    df_iclv = df[df['alternative'] != 3].copy()
    print(f"   - 구매안함 제외: {len(df):,}행 → {len(df_iclv):,}행")
    
    # 결측치 처리
    if df_iclv['income_std'].isnull().sum() > 0:
        mean_income = df_iclv['income_std'].mean()
        df_iclv['income_std'] = df_iclv['income_std'].fillna(mean_income)
        print(f"   - income_std 결측치를 평균({mean_income:.3f})으로 대체")
    
    # 최종 데이터 확인
    print(f"\n   최종 ICLV 데이터:")
    print(f"   - 행 수: {len(df_iclv):,}")
    print(f"   - 응답자 수: {df_iclv['respondent_id'].nunique()}")
    print(f"   - 선택 세트: {df_iclv['choice_set'].nunique()}개")
    print(f"   - 대안 수: {df_iclv['alternative'].nunique()}개")
    
    # 선택 분포
    print(f"\n   선택 분포:")
    choice_dist = df_iclv[df_iclv['choice'] == 1]['alternative_name'].value_counts()
    for alt_name, count in choice_dist.items():
        pct = count / choice_dist.sum() * 100
        print(f"   - {alt_name}: {count}회 ({pct:.1f}%)")
    
    # ICLV 모델 설정 정보
    print("\n[7] ICLV 모델 설정 정보...")
    
    print("\n   측정모델 (5개):")
    for lv_name, indicators in latent_variables.items():
        print(f"   - {lv_name}: {len(indicators)}개 지표")
    
    print("\n   구조모델:")
    print(f"   - 사회인구학적 변수: {structural_vars}")
    
    print("\n   선택모델:")
    print(f"   - 속성: ['health_label', 'price']")
    print(f"   - 가격 변수: 'price'")
    print(f"   - 선택 유형: binary")
    
    print("\n" + "=" * 80)
    print("ICLV 설정 검증 완료!")
    print(f"5개 잠재변수, {total_indicators}개 지표")
    print("=" * 80)
    
    return df_iclv


if __name__ == "__main__":
    df_iclv = test_integrated_data()

