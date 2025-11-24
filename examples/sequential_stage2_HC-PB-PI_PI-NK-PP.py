"""
2단계 추정: Base 선택모델 + PI, NK, PP 주효과

1단계 경로: HC→PB→PI
2단계 변수: Base Model + PI + NK + PP 주효과

Author: ICLV Team
Date: 2025-11-23
"""

import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_config_utils import (
    build_choice_config_dict,
    create_sugar_substitute_multi_lv_config
)
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice


# ============================================================================
# 🎯 사용자 설정 영역
# ============================================================================

# 📌 1단계 결과 파일명
STAGE1_RESULT_FILE = "stage1_HC-PB_PB-PI_results.pkl"

# 📌 요인점수 변환 방법
STANDARDIZATION_METHOD = 'zscore'

# 📌 선택모델 설정
CHOICE_ATTRIBUTES = ['health_label', 'price']
CHOICE_TYPE = 'multinomial'
PRICE_VARIABLE = 'price'

# 📌 잠재변수 설정
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge', 'perceived_price']  # PI, NK, PP 주효과
LV_ATTRIBUTE_INTERACTIONS = []  # 상호작용 없음


def main():
    print("=" * 70)
    print("2단계 추정: Base Model + PI + NK + PP 주효과")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "sugar_substitute_choice_data.csv"
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {data['respondent_id'].nunique()}명")

    # 2. 1단계 결과 파일 확인
    print("\n[2] 1단계 결과 파일 확인 중...")
    stage1_dir = project_root / "results" / "final" / "sequential" / "stage1"
    stage1_path = stage1_dir / STAGE1_RESULT_FILE

    if not stage1_path.exists():
        raise FileNotFoundError(
            f"1단계 결과 파일을 찾을 수 없습니다: {stage1_path}\n"
            f"먼저 sequential_stage1_HC-PB-PI.py를 실행하세요."
        )

    print(f"[OK] 1단계 결과 파일: {stage1_path.name}")

    # 3. 모델 설정 생성
    print("\n[3] 선택모델 설정 중...")

    # 1단계 경로 설정 (HC→PB→PI)
    custom_paths = [
        {'target': 'perceived_benefit', 'predictors': ['health_concern']},
        {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
    ]

    config = create_sugar_substitute_multi_lv_config(
        n_draws=100,
        max_iterations=1000,
        use_hierarchical=False,
        all_lvs_as_main=False,
        custom_paths=custom_paths
    )

    # 선택모델 설정
    from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

    choice_config_dict = build_choice_config_dict(
        main_lvs=MAIN_LVS,
        lv_attribute_interactions=LV_ATTRIBUTE_INTERACTIONS
    )

    choice_config = ChoiceConfig(
        choice_attributes=CHOICE_ATTRIBUTES,
        **choice_config_dict
    )

    print(f"[OK] 선택모델 설정 완료")
    print(f"   - 선택 속성: {CHOICE_ATTRIBUTES}")
    print(f"   - 주효과 LV: {MAIN_LVS}")
    print(f"   - 상호작용: {len(LV_ATTRIBUTE_INTERACTIONS)}개")

    # 4. 선택모델 생성
    print("\n[4] 선택모델 생성 중...")
    choice_model = MultinomialLogitChoice(
        choice_config=choice_config,
        alternatives=['sugar', 'sugar_free', 'allulose'],
        choice_column='choice',
        availability_column='availability',
        price_variable=PRICE_VARIABLE
    )
    print("[OK] 선택모델 생성 완료")

    # 5. Estimator 생성
    print("\n[5] Estimator 생성 중...")
    estimator = SequentialEstimator(config, standardization_method=STANDARDIZATION_METHOD)
    print("[OK] Estimator 생성 완료")
    print(f"   - 요인점수 변환 방법: {STANDARDIZATION_METHOD}")

    # 6. 2단계 추정 실행
    print("\n[6] 2단계 추정 실행 중...")
    print("   (1단계 요인점수를 사용하여 선택모델 추정)")

    results = estimator.estimate_stage2_only(
        data=data,
        choice_model=choice_model,
        factor_scores=str(stage1_path)
    )

    print("\n[OK] 2단계 추정 완료!")

    # 7. 결과 출력
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    print(f"\n[로그우도] {results['log_likelihood']:.2f}")
    print(f"[AIC] {results['aic']:.2f}")
    print(f"[BIC] {results['bic']:.2f}")

    if 'params' in results:
        params_df = results['params']
        print(f"\n[파라미터] ({len(params_df)}개)")
        print(params_df.to_string(index=False))

    # 8. 결과 저장
    print("\n[8] 결과 저장 중...")
    save_dir = project_root / "results" / "final" / "sequential" / "stage2"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 생성
    from model_config_utils import extract_stage1_model_name, generate_stage2_filename
    stage1_model_name = extract_stage1_model_name(STAGE1_RESULT_FILE)
    stage2_filename = generate_stage2_filename(stage1_model_name, MAIN_LVS, LV_ATTRIBUTE_INTERACTIONS)

    save_path = save_dir / stage2_filename
    params_df.to_csv(save_path, index=False)

    print(f"✅ 결과 저장 완료: {save_path.name}")


if __name__ == "__main__":
    main()

