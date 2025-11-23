"""
Sign Correction이 적용된 부트스트랩 테스트

10개 샘플로 빠르게 테스트하여 Sign Correction이 제대로 작동하는지 확인합니다.

실행 방법:
    python examples/test_sign_correction_bootstrap.py

Author: Augment Agent
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
from src.utils.common_utils import setup_project_paths
setup_project_paths()

from src.data.data_loader import DataLoader
from src.analysis.hybrid_choice_model.iclv_models.bootstrap_sequential import bootstrap_both_stages
from examples.sequential_stage1 import config
from examples.sequential_stage2_with_extended_model import build_choice_config_dict

# 설정
N_BOOTSTRAP = 10  # 테스트용 10개 샘플
N_WORKERS = 4
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# 2단계 설정 (sequential_stage2_with_extended_model.py와 동일)
MAIN_LVS = ['purchase_intention', 'nutrition_knowledge']
MODERATION_LVS = []
INTERACTIONS = [
    ('purchase_intention', 'health_label'),
    ('nutrition_knowledge', 'price')
]


def main():
    print("=" * 80)
    print("Sign Correction 부트스트랩 테스트 (10개 샘플)")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    loader = DataLoader()
    data = loader.load_choice_data()
    print(f"   ✅ 데이터 로드 완료: {len(data)}행, {len(data['respondent_id'].unique())}명")
    
    # 2. 1단계 설정 (sequential_stage1.py와 동일)
    print("\n[2] 1단계 모델 설정...")
    print(f"   - 측정모델: {len(config.measurement_configs)}개 잠재변수")
    print(f"   - 구조모델: {len(config.structural.paths)}개 경로")
    for path in config.structural.paths:
        print(f"      {path.from_lv} → {path.to_lv}")
    
    # 3. 2단계 설정
    print("\n[3] 2단계 모델 설정...")
    choice_config = build_choice_config_dict(
        main_lvs=MAIN_LVS,
        moderation_lvs=MODERATION_LVS,
        interactions=INTERACTIONS
    )
    print(f"   - 주효과 LV: {MAIN_LVS}")
    print(f"   - 상호작용: {len(INTERACTIONS)}개")
    for lv, attr in INTERACTIONS:
        print(f"      {lv} × {attr}")
    
    # 4. 부트스트래핑 실행 (Sign Correction 자동 적용)
    print(f"\n[4] 부트스트래핑 실행 (Sign Correction 활성화)")
    print(f"   - 샘플 수: {N_BOOTSTRAP}회")
    print(f"   - 워커 수: {N_WORKERS}개")
    print(f"   - 신뢰수준: {CONFIDENCE_LEVEL*100}%")
    
    start_time = datetime.now()
    print(f"   - 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    print(f"   - 총 소요 시간: {elapsed:.1f}초")
    print(f"   - 성공: {results['n_successful']}/{N_BOOTSTRAP}")
    print(f"   - 실패: {results['n_failed']}/{N_BOOTSTRAP}")
    
    # 5. Sign Flip 통계 확인
    if 'sign_flip_statistics' in results and results['sign_flip_statistics'] is not None:
        print(f"\n[5] Sign Correction 통계")
        print("=" * 80)
        sign_flip_stats = results['sign_flip_statistics']
        print(sign_flip_stats.to_string(index=False))
        
        print(f"\n요약:")
        print(f"   - 총 잠재변수 수: {len(sign_flip_stats)}")
        print(f"   - 평균 부호 반전율: {sign_flip_stats['flip_rate'].mean()*100:.1f}%")
        
        # 반전율이 높은 변수
        high_flip = sign_flip_stats[sign_flip_stats['flip_rate'] > 0.3]
        if len(high_flip) > 0:
            print(f"   - 부호 반전율 > 30%인 변수: {len(high_flip)}개")
            for _, row in high_flip.iterrows():
                print(f"      {row['lv_name']}: {row['flip_rate']*100:.1f}%")
    else:
        print(f"\n⚠️  Sign Flip 통계가 없습니다.")
    
    # 6. 결과 저장
    save_dir = project_root / "results" / "bootstrap" / "test_sign_correction"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    filename_base = f"test_sign_correction_{timestamp}"
    
    ci_file = save_dir / f"{filename_base}_ci.csv"
    stats_file = save_dir / f"{filename_base}_stats.csv"
    sign_flip_file = save_dir / f"{filename_base}_sign_flip.csv"
    full_file = save_dir / f"{filename_base}_full.pkl"
    
    results['confidence_intervals'].to_csv(ci_file, index=False)
    results['bootstrap_statistics'].to_csv(stats_file, index=False)
    
    if 'sign_flip_statistics' in results and results['sign_flip_statistics'] is not None:
        results['sign_flip_statistics'].to_csv(sign_flip_file, index=False)
    
    with open(full_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n[6] 결과 저장")
    print(f"   📁 저장 위치: {save_dir}")
    print(f"   ✅ {ci_file.name}")
    print(f"   ✅ {stats_file.name}")
    if 'sign_flip_statistics' in results and results['sign_flip_statistics'] is not None:
        print(f"   ✅ {sign_flip_file.name}")
    print(f"   ✅ {full_file.name}")
    
    print(f"\n{'='*80}")
    print(f"테스트 완료! 🎉")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

