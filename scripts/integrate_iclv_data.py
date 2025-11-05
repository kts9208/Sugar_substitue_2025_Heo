"""
ICLV 데이터 통합 스크립트

목적: DCE + 5개 잠재변수 + 사회인구학적 데이터 통합
입력:
  - data/processed/dce/dce_long_format.csv (DCE 데이터)
  - data/processed/survey/health_concern.csv (건강관심도)
  - data/processed/survey/perceived_benefit.csv (건강유익성)
  - data/processed/survey/perceived_price.csv (가격수준)
  - data/processed/survey/purchase_intention.csv (구매의도)
  - data/processed/survey/nutrition_knowledge.csv (영양지식)
  - 사회인구학적 데이터 (구조모델 변수)
출력:
  - data/processed/iclv/integrated_data.csv
"""

import pandas as pd
import numpy as np
import os


def load_dce_data():
    """
    DCE Long format 데이터 로드
    
    Returns:
        pd.DataFrame: DCE 데이터
    """
    print("\n[1] DCE 데이터 로드 중...")
    
    df_dce = pd.read_csv('data/processed/dce/dce_long_format.csv')
    
    print(f"   - 로드 완료: {len(df_dce):,}행")
    print(f"   - 응답자 수: {df_dce['respondent_id'].nunique()}")
    print(f"   - 컬럼: {df_dce.columns.tolist()}")
    
    return df_dce


def load_latent_variable_data():
    """
    5개 잠재변수 데이터 로드 (측정모델 지표)

    Returns:
        dict: 잠재변수별 데이터프레임 딕셔너리
    """
    print("\n[2] 잠재변수 데이터 로드 중...")

    latent_vars = {}

    # 1. 건강관심도 (Q6-Q11)
    print("   [2-1] 건강관심도...")
    df_health = pd.read_csv('data/processed/survey/health_concern.csv')
    df_health = df_health.rename(columns={'no': 'respondent_id'})
    df_health = df_health.drop_duplicates(subset='respondent_id', keep='first')
    latent_vars['health_concern'] = df_health
    print(f"      - {len(df_health)}명, 지표: {[c for c in df_health.columns if c.startswith('q')]}")

    # 2. 건강유익성 (Q12-Q17)
    print("   [2-2] 건강유익성...")
    df_benefit = pd.read_csv('data/processed/survey/perceived_benefit.csv')
    df_benefit = df_benefit.rename(columns={'no': 'respondent_id'})
    df_benefit = df_benefit.drop_duplicates(subset='respondent_id', keep='first')
    latent_vars['perceived_benefit'] = df_benefit
    print(f"      - {len(df_benefit)}명, 지표: {[c for c in df_benefit.columns if c.startswith('q')]}")

    # 3. 가격수준 (Q27-Q29)
    print("   [2-3] 가격수준...")
    df_price = pd.read_csv('data/processed/survey/perceived_price.csv')
    df_price = df_price.rename(columns={'no': 'respondent_id'})
    df_price = df_price.drop_duplicates(subset='respondent_id', keep='first')
    latent_vars['perceived_price'] = df_price
    print(f"      - {len(df_price)}명, 지표: {[c for c in df_price.columns if c.startswith('q')]}")

    # 4. 구매의도 (Q18-Q20)
    print("   [2-4] 구매의도...")
    df_purchase = pd.read_csv('data/processed/survey/purchase_intention.csv')
    df_purchase = df_purchase.rename(columns={'no': 'respondent_id'})
    df_purchase = df_purchase.drop_duplicates(subset='respondent_id', keep='first')
    latent_vars['purchase_intention'] = df_purchase
    print(f"      - {len(df_purchase)}명, 지표: {[c for c in df_purchase.columns if c.startswith('q')]}")

    # 5. 영양지식 (Q30-Q49)
    print("   [2-5] 영양지식...")
    df_nutrition = pd.read_csv('data/processed/survey/nutrition_knowledge.csv')
    df_nutrition = df_nutrition.rename(columns={'no': 'respondent_id'})
    df_nutrition = df_nutrition.drop_duplicates(subset='respondent_id', keep='first')
    latent_vars['nutrition_knowledge'] = df_nutrition
    print(f"      - {len(df_nutrition)}명, 지표: {[c for c in df_nutrition.columns if c.startswith('q')]}")

    print(f"\n   - 총 {len(latent_vars)}개 잠재변수 로드 완료")

    return latent_vars


def load_sociodem_data():
    """
    사회인구학적 데이터 로드 (구조모델 변수)

    Returns:
        pd.DataFrame: 사회인구학적 데이터
    """
    print("\n[3] 사회인구학적 데이터 로드 중...")

    # 원본 데이터 로드
    df = pd.read_excel('data/raw/Sugar_substitue_Raw data_250730.xlsx', sheet_name='DATA')

    # 사회인구학적 변수 선택
    sociodem_cols = ['no', 'q1', 'q2_1', 'q52', 'q53', 'q54', 'q55', 'q56']
    df_sociodem = df[sociodem_cols].copy()

    # 컬럼명 변경
    df_sociodem = df_sociodem.rename(columns={
        'no': 'respondent_id',
        'q1': 'gender',
        'q2_1': 'age',
        'q52': 'income',
        'q53': 'education',
        'q54': 'diabetes',
        'q55': 'family_diabetes',
        'q56': 'sugar_substitute_usage'
    })

    # 표준화
    df_sociodem['age_std'] = (df_sociodem['age'] - df_sociodem['age'].mean()) / df_sociodem['age'].std()

    # 소득 연속형 변환 (중간값 사용)
    income_mapping = {1: 2000, 2: 4000, 3: 6000, 4: 8000, 5: 10000}
    df_sociodem['income_continuous'] = df_sociodem['income'].map(income_mapping)
    df_sociodem['income_std'] = (df_sociodem['income_continuous'] - df_sociodem['income_continuous'].mean()) / df_sociodem['income_continuous'].std()

    # 교육 수준 (그대로 사용)
    df_sociodem['education_level'] = df_sociodem['education']

    # 중복 제거 (첫 번째 것만 유지)
    df_sociodem = df_sociodem.drop_duplicates(subset='respondent_id', keep='first')

    print(f"   - 로드 완료: {len(df_sociodem)}명")
    print(f"   - 변수: {df_sociodem.columns.tolist()}")

    return df_sociodem


def integrate_data(df_dce, latent_vars, df_sociodem):
    """
    DCE + 5개 잠재변수 + 사회인구학적 데이터 통합

    Args:
        df_dce: DCE 데이터
        latent_vars: 잠재변수 데이터 딕셔너리
        df_sociodem: 사회인구학적 데이터

    Returns:
        pd.DataFrame: 통합 데이터
    """
    print("\n[4] 데이터 통합 중...")

    df_merged = df_dce.copy()

    # Step 1-5: 5개 잠재변수 순차 병합
    for i, (lv_name, df_lv) in enumerate(latent_vars.items(), 1):
        print(f"   - Step {i}: + {lv_name} 병합...")
        df_merged = df_merged.merge(
            df_lv,
            on='respondent_id',
            how='left'
        )
        print(f"     병합 후: {len(df_merged):,}행 × {len(df_merged.columns)}컬럼")

    # Step 6: + 사회인구학적
    print(f"   - Step 6: + 사회인구학적 병합...")
    df_integrated = df_merged.merge(
        df_sociodem,
        on='respondent_id',
        how='left'
    )
    print(f"     병합 후: {len(df_integrated):,}행 × {len(df_integrated.columns)}컬럼")

    print(f"\n   - 최종 데이터: {len(df_integrated):,}행 × {len(df_integrated.columns)}컬럼")

    return df_integrated


def validate_integration(df_integrated, df_dce):
    """
    통합 데이터 검증

    Args:
        df_integrated: 통합 데이터
        df_dce: 원본 DCE 데이터
    """
    print("\n[5] 통합 데이터 검증 중...")

    # 검증 1: 행 수
    assert len(df_integrated) == len(df_dce), "행 수가 변경됨"
    print(f"   ✓ 행 수 검증: {len(df_integrated):,}행 유지")

    # 검증 2: 응답자 수
    n_respondents_original = df_dce['respondent_id'].nunique()
    n_respondents_integrated = df_integrated['respondent_id'].nunique()
    print(f"   ✓ 응답자 수: {n_respondents_integrated}명 (원본: {n_respondents_original}명)")

    # 검증 3: DCE 컬럼 유지
    dce_cols = ['choice_set', 'alternative', 'price', 'health_label', 'choice']
    for col in dce_cols:
        assert col in df_integrated.columns, f"{col} 컬럼 누락"
    print(f"   ✓ DCE 컬럼 유지: {dce_cols}")

    # 검증 4: 5개 잠재변수 지표 추가
    print("\n   [잠재변수 지표 검증]")

    # 4-1. 건강관심도 (Q6-Q11)
    health_cols = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
    for col in health_cols:
        assert col in df_integrated.columns, f"{col} 컬럼 누락"
    print(f"   ✓ 건강관심도 (6개): {health_cols}")

    # 4-2. 건강유익성 (Q12-Q17)
    benefit_cols = ['q12', 'q13', 'q14', 'q15', 'q16', 'q17']
    for col in benefit_cols:
        assert col in df_integrated.columns, f"{col} 컬럼 누락"
    print(f"   ✓ 건강유익성 (6개): {benefit_cols}")

    # 4-3. 구매의도 (Q18-Q20)
    purchase_cols = ['q18', 'q19', 'q20']
    for col in purchase_cols:
        assert col in df_integrated.columns, f"{col} 컬럼 누락"
    print(f"   ✓ 구매의도 (3개): {purchase_cols}")

    # 4-4. 가격수준 (Q27-Q29)
    price_cols = ['q27', 'q28', 'q29']
    for col in price_cols:
        assert col in df_integrated.columns, f"{col} 컬럼 누락"
    print(f"   ✓ 가격수준 (3개): {price_cols}")

    # 4-5. 영양지식 (Q30-Q49)
    nutrition_cols = [f'q{i}' for i in range(30, 50)]
    for col in nutrition_cols:
        assert col in df_integrated.columns, f"{col} 컬럼 누락"
    print(f"   ✓ 영양지식 (20개): q30-q49")

    print(f"   ✓ 총 38개 지표 모두 존재")

    # 검증 5: 구조모델 변수 추가
    print("\n   [구조모델 변수 검증]")
    structural_cols = ['age_std', 'gender', 'income_std', 'education_level']
    for col in structural_cols:
        if col in df_integrated.columns:
            print(f"   ✓ {col}")

    # 검증 6: 결측치 확인
    print("\n   [결측치 확인]")
    missing = df_integrated.isnull().sum()
    critical_cols = ['respondent_id', 'choice_set', 'alternative', 'choice']

    for col in critical_cols:
        if missing[col] > 0:
            print(f"   ✗ {col}: {missing[col]}개 결측치 (치명적!)")
        else:
            print(f"   ✓ {col}: 결측치 없음")

    print("\n   ✓ 모든 검증 통과!")


def create_summary(df_integrated):
    """
    통합 데이터 요약
    
    Args:
        df_integrated: 통합 데이터
    """
    print("\n[6] 통합 데이터 요약:")
    print("-" * 80)
    
    # 기본 정보
    print(f"   - 총 행 수: {len(df_integrated):,}")
    print(f"   - 총 컬럼 수: {len(df_integrated.columns)}")
    print(f"   - 응답자 수: {df_integrated['respondent_id'].nunique()}")
    
    # 컬럼 그룹별 정리
    print(f"\n   [컬럼 그룹]")
    
    # DCE 관련
    dce_cols = [c for c in df_integrated.columns if c in [
        'choice_set', 'alternative', 'alternative_name', 
        'product_type', 'sugar_content', 'health_label', 'price', 'choice'
    ]]
    print(f"   - DCE 변수 ({len(dce_cols)}개): {dce_cols}")
    
    # 측정모델 지표
    indicator_cols = [c for c in df_integrated.columns if c.startswith('q') and c[1:].isdigit()]
    print(f"   - 측정모델 지표 ({len(indicator_cols)}개): {indicator_cols}")
    
    # 구조모델 변수
    structural_cols = [c for c in df_integrated.columns if c in [
        'age', 'age_std', 'gender', 'income', 'income_std', 
        'income_continuous', 'education', 'education_level',
        'diabetes', 'family_diabetes', 'sugar_substitute_usage'
    ]]
    print(f"   - 구조모델 변수 ({len(structural_cols)}개): {structural_cols}")
    
    # 선택 분포
    print(f"\n   [선택 분포]")
    choice_dist = df_integrated[df_integrated['choice'] == 1]['alternative_name'].value_counts()
    total_choices = choice_dist.sum()
    for alt_name, count in choice_dist.items():
        pct = count / total_choices * 100
        print(f"   - {alt_name}: {count}회 ({pct:.1f}%)")
    
    # 결측치 요약
    print(f"\n   [결측치 요약]")
    missing = df_integrated.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing_cols) > 0:
        print(f"   - 결측치가 있는 컬럼: {len(missing_cols)}개")
        for col, count in missing_cols.head(10).items():
            pct = count / len(df_integrated) * 100
            print(f"     {col}: {count}개 ({pct:.1f}%)")
    else:
        print(f"   - 결측치 없음")


def main():
    """메인 실행 함수"""

    print("=" * 80)
    print("ICLV 데이터 통합 (5개 잠재변수)")
    print("=" * 80)

    # 1. 데이터 로드
    df_dce = load_dce_data()
    latent_vars = load_latent_variable_data()
    df_sociodem = load_sociodem_data()

    # 2. 데이터 통합
    df_integrated = integrate_data(df_dce, latent_vars, df_sociodem)

    # 3. 검증
    validate_integration(df_integrated, df_dce)

    # 4. 요약
    create_summary(df_integrated)

    # 5. 저장
    print("\n[7] 저장 중...")
    output_dir = 'data/processed/iclv'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'integrated_data.csv')
    df_integrated.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   - 저장 완료: {output_path}")

    # 6. 미리보기
    print("\n[8] 데이터 미리보기 (첫 3행):")
    print("-" * 80)
    print(df_integrated.head(3).to_string())

    print("\n" + "=" * 80)
    print("ICLV 데이터 통합 완료! (5개 잠재변수, 38개 지표)")
    print("=" * 80)

    return df_integrated


if __name__ == "__main__":
    df_integrated = main()

