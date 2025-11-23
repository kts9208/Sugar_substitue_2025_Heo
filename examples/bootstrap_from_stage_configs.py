"""
순차추정 부트스트래핑 - Stage1/Stage2 설정 자동 불러오기

✅ 이 스크립트는 sequential_stage1.py와 sequential_stage2_with_extended_model.py의
   설정을 그대로 불러와서 부트스트래핑을 수행합니다.

사용법:
    python examples/bootstrap_from_stage_configs.py

주요 기능:
    - sequential_stage1.py의 PATHS 설정 자동 불러오기
    - sequential_stage2_with_extended_model.py의 선택모델 설정 자동 불러오기
    - 1+2단계 통합 부트스트래핑 (Both Stages)
    - 결과 자동 저장

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

# 공통 유틸리티 import
from model_config_utils import (
    build_paths_from_config,
    build_choice_config_dict,
    extract_stage1_model_name,
    generate_stage2_filename
)

from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_both_stages
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig


# ============================================================================
# 🎯 설정 불러오기 - sequential_stage1.py와 동일한 설정
# ============================================================================

# sequential_stage1.py의 PATHS 설정
PATHS = {
    'HC->PB': True,   # 건강관심도 → 건강유익성
    'HC->PP': True,   # 건강관심도 → 가격수준
    'HC->PI': False,  # 건강관심도 → 구매의도
    'PB->PI': True,   # 건강유익성 → 구매의도
    'PP->PI': True,   # 가격수준 → 구매의도
    'NK->PI': False,  # 영양지식 → 구매의도
}

# sequential_stage2_with_extended_model.py의 선택모델 설정
CHOICE_ATTRIBUTES = ['health_label', 'price']
CHOICE_TYPE = 'multinomial'
PRICE_VARIABLE = 'price'
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
MODERATION_LVS = []
LV_ATTRIBUTE_INTERACTIONS = [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]

# 부트스트래핑 설정
N_BOOTSTRAP = 10    # 부트스트랩 샘플 수 (테스트용: 10개, 실제: 1000개)
N_WORKERS = 4       # 병렬 처리 워커 수
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def main():
    print("=" * 80)
    print("순차추정 부트스트래핑 - Stage1/Stage2 설정 자동 불러오기")
    print("=" * 80)
    
    # 1. 경로 구성 (sequential_stage1.py와 동일)
    hierarchical_paths, path_name, model_description = build_paths_from_config(PATHS)
    
    print(f"\n[1단계 설정] {model_description}")
    if hierarchical_paths:
        for i, path_dict in enumerate(hierarchical_paths, 1):
            print(f"   {i}. {path_dict}")
    
    # 2. 선택모델 설정 (sequential_stage2_with_extended_model.py와 동일)
    model_type_parts = ["Base Model"]
    if MAIN_LVS:
        lv_abbr = {'purchase_intention': 'PI', 'nutrition_knowledge': 'NK',
                   'perceived_benefit': 'PB', 'perceived_price': 'PP', 'health_concern': 'HC'}
        lv_names = [lv_abbr.get(lv, lv.upper()) for lv in MAIN_LVS]
        model_type_parts.append(f"+ {' + '.join(lv_names)} 주효과")
    if MODERATION_LVS:
        model_type_parts.append(f"+ 조절효과 {len(MODERATION_LVS)}개")
    if LV_ATTRIBUTE_INTERACTIONS:
        model_type_parts.append(f"+ LV-Attr 상호작용 {len(LV_ATTRIBUTE_INTERACTIONS)}개")
    
    model_type_str = " ".join(model_type_parts)
    print(f"\n[2단계 설정] {model_type_str}")
    print(f"   - 선택 속성: {', '.join(CHOICE_ATTRIBUTES)}")
    print(f"   - 주효과 LV: {', '.join(MAIN_LVS) if MAIN_LVS else '없음'}")
    print(f"   - 상호작용: {len(LV_ATTRIBUTE_INTERACTIONS)}개")
    
    # 3. 데이터 로드
    print(f"\n[데이터 로드]")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    data = pd.read_csv(data_path)
    print(f"   ✅ {len(data)}행, {data['respondent_id'].nunique()}명")
    
    # 4. 모델 설정 생성
    print(f"\n[모델 설정 생성]")
    config = create_sugar_substitute_multi_lv_config(
        use_hierarchical=False,
        custom_paths=hierarchical_paths
    )
    
    choice_config_dict = build_choice_config_dict(
        main_lvs=MAIN_LVS,
        lv_attribute_interactions=LV_ATTRIBUTE_INTERACTIONS
    )
    
    choice_config = ChoiceConfig(
        choice_attributes=CHOICE_ATTRIBUTES,
        choice_type=CHOICE_TYPE,
        price_variable=PRICE_VARIABLE,
        **choice_config_dict  # main_lvs와 lv_attribute_interactions 포함
    )
    
    print(f"   ✅ 1단계 설정 완료")
    print(f"   ✅ 2단계 설정 완료")
    
    # 5. 부트스트래핑 실행
    print(f"\n[부트스트래핑 실행]")
    print(f"   - 샘플 수: {N_BOOTSTRAP}회")
    print(f"   - 워커 수: {N_WORKERS}개")
    print(f"   - 신뢰수준: {CONFIDENCE_LEVEL*100}%")

    start_time = datetime.now()
    print(f"   - 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n⏳ 부트스트래핑 진행 중... (예상 소요 시간: 30~60분)")

    results = bootstrap_both_stages(
        data=data,
        measurement_model=config.measurement_configs,
        structural_model=config.structural,
        choice_model=choice_config,
        n_bootstrap=N_BOOTSTRAP,
        n_workers=N_WORKERS,
        confidence_level=CONFIDENCE_LEVEL,
        random_seed=RANDOM_SEED,
        show_progress=True
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n✅ 부트스트래핑 완료!")
    print(f"   - 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   - 총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.0f}초)")
    print(f"   - 성공: {results['n_successful']}/{N_BOOTSTRAP}")
    print(f"   - 실패: {results['n_failed']}/{N_BOOTSTRAP}")
    print(f"   - 성공률: {results['n_successful']/N_BOOTSTRAP*100:.1f}%")

    # 6. 결과 출력
    print(f"\n[부트스트래핑 결과]")
    print(f"\n신뢰구간 (상위 20개):")
    print(results['confidence_intervals'].head(20).to_string(index=False))

    print(f"\n부트스트랩 통계량 (상위 20개):")
    print(results['bootstrap_statistics'].head(20).to_string(index=False))

    # 7. 결과 저장
    save_dir = project_root / "results" / "bootstrap" / "sequential"
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = start_time.strftime('%Y%m%d_%H%M%S')

    # 파일명 생성 (1단계 경로명 + 2단계 모델 타입)
    stage1_name = path_name
    stage2_abbr = "_".join([lv_abbr.get(lv, lv[:2].upper()) for lv in MAIN_LVS]) if MAIN_LVS else "base"
    filename_base = f"bootstrap_{stage1_name}_{stage2_abbr}_{timestamp}"

    ci_file = save_dir / f"{filename_base}_ci.csv"
    stats_file = save_dir / f"{filename_base}_stats.csv"
    full_file = save_dir / f"{filename_base}_full.pkl"

    results['confidence_intervals'].to_csv(ci_file, index=False)
    results['bootstrap_statistics'].to_csv(stats_file, index=False)

    # ✅ Sign Flip 통계 저장 (있는 경우)
    if 'sign_flip_statistics' in results and results['sign_flip_statistics'] is not None:
        sign_flip_file = save_dir / f"{filename_base}_sign_flip.csv"
        results['sign_flip_statistics'].to_csv(sign_flip_file, index=False)

    with open(full_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n[결과 저장]")
    print(f"   📁 저장 위치: {save_dir}")
    print(f"   ✅ {ci_file.name}")
    print(f"   ✅ {stats_file.name}")
    if 'sign_flip_statistics' in results and results['sign_flip_statistics'] is not None:
        print(f"   ✅ {sign_flip_file.name} (Sign Correction 통계)")
    print(f"   ✅ {full_file.name}")

    print(f"\n{'='*80}")
    print(f"부트스트래핑 완료! 🎉")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


