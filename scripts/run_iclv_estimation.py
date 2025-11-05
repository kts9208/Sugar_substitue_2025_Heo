"""
ICLV 동시추정 실행 스크립트

목적: 5개 잠재변수를 포함한 측정모델 + 구조모델 + 선택모델 동시추정
입력: data/processed/iclv/integrated_data.csv (5개 잠재변수, 38개 지표)
출력: ICLV 추정 결과
"""

import pandas as pd
import numpy as np
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

from src.analysis.hybrid_choice_model.iclv_models import (
    OrderedProbitMeasurement,
    LatentVariableRegression,
    BinaryProbitChoice,
    SimultaneousEstimator
)
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    StructuralConfig,
    ChoiceConfig,
    ICLVConfig
)


def load_integrated_data():
    """
    통합 데이터 로드
    
    Returns:
        pd.DataFrame: 통합 데이터
    """
    print("\n[1] 통합 데이터 로드 중...")
    
    df = pd.read_csv('data/processed/iclv/integrated_data.csv')
    
    print(f"   - 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼")
    print(f"   - 응답자 수: {df['respondent_id'].nunique()}")
    
    return df


def create_iclv_config():
    """
    ICLV 모델 설정 생성 (5개 잠재변수)

    Returns:
        dict: 5개 잠재변수별 ICLV 설정 딕셔너리
    """
    print("\n[2] ICLV 모델 설정 생성 중...")
    print("   (5개 잠재변수 모두 포함)")

    configs = {}

    # 1. 건강관심도 (Q6-Q11)
    print("\n   [2-1] 건강관심도 설정...")
    configs['health_concern'] = {
        'measurement': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            indicator_type='ordered',
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['age_std', 'gender', 'income_std', 'education_level'],
            include_in_choice=True
        )
    }
    print(f"      - 지표: 6개 (q6-q11)")

    # 2. 건강유익성 (Q12-Q17)
    print("   [2-2] 건강유익성 설정...")
    configs['perceived_benefit'] = {
        'measurement': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            indicator_type='ordered',
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['health_concern'],  # 경로 1: 건강관심도 → 건강유익성
            include_in_choice=True
        )
    }
    print(f"      - 지표: 6개 (q12-q17)")
    print(f"      - 구조경로: 건강관심도 → 건강유익성")

    # 3. 구매의도 (Q18-Q20)
    print("   [2-3] 구매의도 설정...")
    configs['purchase_intention'] = {
        'measurement': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            indicator_type='ordered',
            n_categories=5
        ),
        'structural': StructuralConfig(
            # 경로 2: 건강유익성 → 구매의도
            # 경로 3: 인지된 가격수준 → 구매의도
            # 경로 4: 영양지식수준 → 구매의도
            sociodemographics=['perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
            include_in_choice=True
        )
    }
    print(f"      - 지표: 3개 (q18-q20)")
    print(f"      - 구조경로: 건강유익성 → 구매의도")
    print(f"      - 구조경로: 인지된 가격수준 → 구매의도")
    print(f"      - 구조경로: 영양지식수준 → 구매의도")

    # 4. 가격수준 (Q27-Q29)
    print("   [2-4] 가격수준 설정...")
    configs['perceived_price'] = {
        'measurement': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],
            indicator_type='ordered',
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['income_std'],  # 소득의 영향
            include_in_choice=True
        )
    }
    print(f"      - 지표: 3개 (q27-q29)")
    print(f"      - 구조경로: 소득 → 인지된 가격수준")

    # 5. 영양지식 (Q30-Q49)
    print("   [2-5] 영양지식 설정...")
    nutrition_indicators = [f'q{i}' for i in range(30, 50)]
    configs['nutrition_knowledge'] = {
        'measurement': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=nutrition_indicators,
            indicator_type='ordered',
            n_categories=5
        ),
        'structural': StructuralConfig(
            sociodemographics=['age_std', 'education_level'],  # 연령, 교육의 영향
            include_in_choice=True
        )
    }
    print(f"      - 지표: 20개 (q30-q49)")
    print(f"      - 구조경로: 연령, 교육 → 영양지식수준")

    # 선택모델 설정 (공통)
    print("\n   [2-6] 선택모델 설정...")
    choice_config = ChoiceConfig(
        choice_attributes=['health_label', 'price'],
        price_variable='price',
        choice_type='binary',
        lv_in_choice=True  # 잠재변수를 선택모델에 포함
    )
    print(f"      - 속성: {choice_config.choice_attributes}")
    print(f"      - 가격 변수: {choice_config.price_variable}")

    # 추정 설정 (공통)
    print("\n   [2-7] 추정 설정...")
    estimation_config = {
        'n_draws': 500,  # Halton draws
        'seed': 42,
        'method': 'simultaneous'
    }
    print(f"      - Halton draws: {estimation_config['n_draws']}")
    print(f"      - 추정 방법: {estimation_config['method']}")

    # 전체 설정 반환
    result = {
        'latent_variables': configs,
        'choice': choice_config,
        'estimation': estimation_config
    }

    print(f"\n   ✓ 총 5개 잠재변수, 38개 지표 설정 완료")

    return result


def prepare_data_for_iclv(df):
    """
    ICLV 추정을 위한 데이터 준비

    Args:
        df: 통합 데이터

    Returns:
        pd.DataFrame: 준비된 데이터
    """
    print("\n[3] 데이터 준비 중...")

    # "구매안함" 대안 제외 (선택모델에서는 제품 선택만 분석)
    df_prepared = df[df['alternative'] != 3].copy()

    print(f"   - 구매안함 제외: {len(df):,}행 → {len(df_prepared):,}행")

    # 결측치 처리
    print(f"   - health_label 결측치: {df_prepared['health_label'].isnull().sum()}")
    print(f"   - price 결측치: {df_prepared['price'].isnull().sum()}")

    # 5개 잠재변수 지표 결측치 확인
    print("\n   [잠재변수 지표 결측치 확인]")

    # 건강관심도
    health_cols = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
    health_missing = df_prepared[health_cols].isnull().sum().sum()
    print(f"   - 건강관심도 (q6-q11): {health_missing}개")

    # 건강유익성
    benefit_cols = ['q12', 'q13', 'q14', 'q15', 'q16', 'q17']
    benefit_missing = df_prepared[benefit_cols].isnull().sum().sum()
    print(f"   - 건강유익성 (q12-q17): {benefit_missing}개")

    # 구매의도
    purchase_cols = ['q18', 'q19', 'q20']
    purchase_missing = df_prepared[purchase_cols].isnull().sum().sum()
    print(f"   - 구매의도 (q18-q20): {purchase_missing}개")

    # 가격수준
    price_cols = ['q27', 'q28', 'q29']
    price_missing = df_prepared[price_cols].isnull().sum().sum()
    print(f"   - 가격수준 (q27-q29): {price_missing}개")

    # 영양지식
    nutrition_cols = [f'q{i}' for i in range(30, 50)]
    nutrition_missing = df_prepared[nutrition_cols].isnull().sum().sum()
    print(f"   - 영양지식 (q30-q49): {nutrition_missing}개")

    # 구조모델 변수 결측치 확인
    print("\n   [구조모델 변수 결측치 확인]")
    structural_cols = ['age_std', 'gender', 'income_std', 'education_level']
    for col in structural_cols:
        missing = df_prepared[col].isnull().sum()
        if missing > 0:
            print(f"   ⚠ {col}: {missing}개 결측치")

    # income_std 결측치 처리 (평균으로 대체)
    if df_prepared['income_std'].isnull().sum() > 0:
        mean_income = df_prepared['income_std'].mean()
        df_prepared['income_std'] = df_prepared['income_std'].fillna(mean_income)
        print(f"   - income_std 결측치를 평균({mean_income:.3f})으로 대체")

    print(f"\n   - 최종 데이터: {len(df_prepared):,}행 × {len(df_prepared.columns)}컬럼")

    return df_prepared


def run_iclv_estimation(df, config):
    """
    ICLV 동시추정 실행 (5개 잠재변수)

    Args:
        df: 준비된 데이터
        config: ICLV 설정 딕셔너리

    Returns:
        dict: 추정 결과
    """
    print("\n[4] ICLV 동시추정 실행 중...")
    print("   (5개 잠재변수 동시추정 - 시간이 소요될 수 있습니다...)")

    try:
        # 현재는 간단한 검증만 수행
        print("\n   [4-1] 설정 검증...")

        # 잠재변수 설정 확인
        lv_configs = config['latent_variables']
        print(f"   - 잠재변수 수: {len(lv_configs)}개")
        for lv_name, lv_config in lv_configs.items():
            n_indicators = len(lv_config['measurement'].indicators)
            print(f"     {lv_name}: {n_indicators}개 지표")

        # 선택모델 설정 확인
        choice_config = config['choice']
        print(f"   - 선택모델 속성: {choice_config.choice_attributes}")

        # 추정 설정 확인
        est_config = config['estimation']
        print(f"   - 추정 방법: {est_config['method']}")
        print(f"   - Halton draws: {est_config['n_draws']}")

        print("\n   [4-2] 데이터 검증...")

        # 모든 지표가 데이터에 존재하는지 확인
        all_indicators = []
        for lv_config in lv_configs.values():
            all_indicators.extend(lv_config['measurement'].indicators)

        missing_indicators = [ind for ind in all_indicators if ind not in df.columns]
        if missing_indicators:
            print(f"   ✗ 누락된 지표: {missing_indicators}")
            return None
        else:
            print(f"   ✓ 모든 지표 존재 ({len(all_indicators)}개)")

        # 선택모델 속성 확인
        for attr in choice_config.choice_attributes:
            if attr not in df.columns:
                print(f"   ✗ 누락된 속성: {attr}")
                return None
        print(f"   ✓ 모든 선택 속성 존재")

        print("\n   [4-3] 추정 준비 완료")
        print("   ⚠ 실제 추정은 SimultaneousEstimator 클래스가 완전히 구현된 후 실행됩니다.")
        print("   ⚠ 현재는 설정 검증 및 데이터 준비 단계까지만 완료되었습니다.")

        # 결과 구조 반환 (실제 추정은 아직 미구현)
        results = {
            'status': 'prepared',
            'message': '5개 잠재변수 ICLV 모델 설정 및 데이터 준비 완료',
            'n_latent_variables': len(lv_configs),
            'n_indicators': len(all_indicators),
            'n_observations': len(df),
            'n_respondents': df['respondent_id'].nunique(),
            'config': config
        }

        print("\n   ✓ 설정 및 데이터 검증 완료!")

        return results

    except Exception as e:
        print(f"\n   ✗ 실패: {e}")
        print(f"\n   상세 오류:")
        import traceback
        traceback.print_exc()
        return None


def display_results(results):
    """
    추정 결과 출력

    Args:
        results: 추정 결과
    """
    if results is None:
        print("\n[5] 추정 결과 없음")
        return

    print("\n[5] 추정 결과:")
    print("=" * 80)

    # 현재 상태 출력
    if 'status' in results:
        print(f"\n상태: {results['status']}")
        print(f"메시지: {results['message']}")
        print(f"\n모델 정보:")
        print(f"  - 잠재변수 수: {results['n_latent_variables']}개")
        print(f"  - 총 지표 수: {results['n_indicators']}개")
        print(f"  - 관측치 수: {results['n_observations']:,}행")
        print(f"  - 응답자 수: {results['n_respondents']}명")

        # 잠재변수별 정보
        if 'config' in results:
            print(f"\n잠재변수별 지표:")
            for lv_name, lv_config in results['config']['latent_variables'].items():
                n_ind = len(lv_config['measurement'].indicators)
                print(f"  - {lv_name}: {n_ind}개")

        return

    # 실제 추정 결과 (미래 구현)
    # 측정모델 결과
    if 'measurement' in results:
        print("\n[측정모델 결과]")
        print("-" * 80)
        print(results['measurement'])

    # 구조모델 결과
    if 'structural' in results:
        print("\n[구조모델 결과]")
        print("-" * 80)
        print(results['structural'])

    # 선택모델 결과
    if 'choice' in results:
        print("\n[선택모델 결과]")
        print("-" * 80)
        print(results['choice'])

    # WTP
    if 'wtp' in results:
        print("\n[WTP (지불의사액)]")
        print("-" * 80)
        print(results['wtp'])

    # 모델 적합도
    if 'fit' in results:
        print("\n[모델 적합도]")
        print("-" * 80)
        print(results['fit'])


def save_results(results):
    """
    추정 결과 저장
    
    Args:
        results: 추정 결과
    """
    if results is None:
        return
    
    print("\n[6] 결과 저장 중...")
    
    output_dir = 'results/iclv'
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 결과를 CSV로 저장
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            output_path = os.path.join(output_dir, f'{key}_results.csv')
            value.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"   - {key}: {output_path}")
    
    print("   ✓ 저장 완료!")


def main():
    """메인 실행 함수"""

    print("=" * 80)
    print("ICLV 동시추정 실행 (5개 잠재변수)")
    print("=" * 80)

    # 1. 데이터 로드
    df = load_integrated_data()

    # 2. 설정 생성
    config = create_iclv_config()

    # 3. 데이터 준비
    df_prepared = prepare_data_for_iclv(df)

    # 4. 추정 실행
    results = run_iclv_estimation(df_prepared, config)

    # 5. 결과 출력
    display_results(results)

    # 6. 결과 저장
    save_results(results)

    print("\n" + "=" * 80)
    print("ICLV 설정 및 데이터 준비 완료!")
    print("5개 잠재변수, 38개 지표")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()

