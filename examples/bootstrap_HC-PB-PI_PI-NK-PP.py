"""
부트스트래핑: HC→PB→PI + Base Model + PI + NK + PP

1단계 경로: HC→PB→PI
2단계 변수: Base Model + PI + NK + PP 주효과

Author: ICLV Team
Date: 2025-11-23
"""

import sys
from pathlib import Path
import pandas as pd
import pickle
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_config_utils import (
    build_paths_from_config,
    build_choice_config_dict,
    extract_stage1_model_name,
    generate_stage2_filename,
    create_sugar_substitute_multi_lv_config
)
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_both_stages
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig


# ============================================================================
# 🎯 사용자 설정 영역
# ============================================================================

# 📌 1단계 경로 설정 (HC→PB→PI)
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': False,  # 건강관심도 → 가격수준
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': False,  # 가격수준 → 구매의도
    'NK->PI': False,  # 영양지식 → 구매의도
}

# 📌 2단계 선택모델 설정
CHOICE_ATTRIBUTES = ['health_label', 'price']
PRICE_VARIABLE = 'price'
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge', 'perceived_price']  # PI, NK, PP 주효과
LV_ATTRIBUTE_INTERACTIONS = []  # 상호작용 없음

# 📌 부트스트랩 설정
N_BOOTSTRAP = 10  # 테스트용 10개 (실제 분석: 1000)
N_WORKERS = 4
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def main():
    print("=" * 80)
    print("순차추정 부트스트래핑: HC→PB→PI + Base Model + PI + NK + PP")
    print("=" * 80)

    # 1. 경로 구성
    hierarchical_paths, path_name, model_description = build_paths_from_config(PATHS)

    print(f"\n[1단계 설정] {model_description}")
    if hierarchical_paths:
        for i, path_dict in enumerate(hierarchical_paths, 1):
            print(f"   {i}. {path_dict}")

    print(f"\n[2단계 설정] Base Model + PI + NK + PP 주효과")
    print(f"   - 선택 속성: {CHOICE_ATTRIBUTES}")
    print(f"   - 주효과 LV: {MAIN_LVS}")
    print(f"   - 상호작용: {len(LV_ATTRIBUTE_INTERACTIONS)}개")

    # 2. 데이터 로드
    print(f"\n[데이터 로드]")
    data_path = project_root / "data" / "processed" / "sugar_substitute_choice_data.csv"
    data = pd.read_csv(data_path)
    print(f"   ✅ {len(data)}행, {data['respondent_id'].nunique()}명")

    # 3. 모델 설정 생성
    print(f"\n[모델 설정 생성]")

    # 1단계 설정
    config = create_sugar_substitute_multi_lv_config(
        n_draws=100,
        max_iterations=1000,
        use_hierarchical=True,
        all_lvs_as_main=False,
        custom_paths=hierarchical_paths
    )
    print(f"   ✅ 1단계 설정 완료")

    # 2단계 선택모델 설정
    choice_config_dict = build_choice_config_dict(
        main_lvs=MAIN_LVS,
        lv_attribute_interactions=LV_ATTRIBUTE_INTERACTIONS
    )

    choice_config = ChoiceConfig(
        choice_attributes=CHOICE_ATTRIBUTES,
        **choice_config_dict
    )

    choice_model = MultinomialLogitChoice(
        choice_config=choice_config,
        alternatives=['sugar', 'sugar_free', 'allulose'],
        choice_column='choice',
        availability_column='availability',
        price_variable=PRICE_VARIABLE
    )
    print(f"   ✅ 2단계 설정 완료")

    # 4. 부트스트래핑 실행
    print(f"\n[부트스트래핑 실행]")
    print(f"   - 샘플 수: {N_BOOTSTRAP}회")
    print(f"   - 워커 수: {N_WORKERS}개")
    print(f"   - 신뢰수준: {CONFIDENCE_LEVEL*100}%")
    print(f"   - 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if N_BOOTSTRAP < 100:
        print(f"\n⏳ 부트스트래핑 진행 중... (예상 소요 시간: 1~2분)")
    else:
        print(f"\n⏳ 부트스트래핑 진행 중... (예상 소요 시간: 30~60분)")

    results = bootstrap_both_stages(
        data=data,
        measurement_model=config.measurement_configs,
        structural_model=config.structural,
        choice_model=choice_model,
        n_bootstrap=N_BOOTSTRAP,
        n_workers=N_WORKERS,
        confidence_level=CONFIDENCE_LEVEL,
        random_seed=RANDOM_SEED
    )

    print(f"\n✅ 부트스트래핑 완료!")
    print(f"   - 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    elapsed_min = results.get('elapsed_time', 0) / 60
    print(f"   - 총 소요 시간: {elapsed_min:.1f}분 ({results.get('elapsed_time', 0):.0f}초)")
    print(f"   - 성공: {results['n_successful']}/{N_BOOTSTRAP}")
    print(f"   - 실패: {results['n_failed']}/{N_BOOTSTRAP}")
    print(f"   - 성공률: {results['n_successful']/N_BOOTSTRAP*100:.1f}%")

    # 5. 결과 저장
    print(f"\n[부트스트래핑 결과]")

    ci_df = results['confidence_intervals']
    stats_df = results['bootstrap_statistics']

    print(f"\n신뢰구간 (상위 20개):")
    print(ci_df.head(20).to_string(index=False))

    print(f"\n부트스트랩 통계량 (상위 20개):")
    print(stats_df.head(20).to_string(index=False))

    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / "results" / "bootstrap" / "sequential"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 생성
    main_lvs_str = '_'.join(MAIN_LVS) if MAIN_LVS else 'base'
    filename_base = f"bootstrap_{path_name}_{main_lvs_str}_{timestamp}"

    ci_file = save_dir / f"{filename_base}_ci.csv"
    stats_file = save_dir / f"{filename_base}_stats.csv"
    full_file = save_dir / f"{filename_base}_full.pkl"

    ci_df.to_csv(ci_file, index=False)
    stats_df.to_csv(stats_file, index=False)

    with open(full_file, 'wb') as f:
        pickle.dump(results, f)

    # Sign Flip 통계 저장
    if 'sign_flip_statistics' in results and results['sign_flip_statistics'] is not None:
        sign_flip_file = save_dir / f"{filename_base}_sign_flip.csv"
        results['sign_flip_statistics'].to_csv(sign_flip_file, index=False)

    print(f"\n[결과 저장]")
    print(f"   📁 저장 위치: {save_dir}")
    print(f"   ✅ {ci_file.name}")
    print(f"   ✅ {stats_file.name}")
    if 'sign_flip_statistics' in results and results['sign_flip_statistics'] is not None:
        print(f"   ✅ {sign_flip_file.name} (Sign Correction 통계)")
    print(f"   ✅ {full_file.name}")

    print("\n" + "=" * 80)
    print("부트스트래핑 완료! 🎉")
    print("=" * 80)


if __name__ == "__main__":
    main()

