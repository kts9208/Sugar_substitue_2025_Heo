"""
동시추정 중 잠재변수 값 확인

실제 동시추정에서 생성되는 잠재변수 값의 분포를 확인합니다.

Author: Sugar Substitute Research Team
Date: 2025-11-20
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    create_sugar_substitute_multi_lv_config
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_gpu_batch_estimator import MultiDimensionalHaltonDrawGenerator as HaltonGenerator


def main():
    print("=" * 80)
    print("동시추정 중 잠재변수 값 확인")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    n_individuals = data['respondent_id'].nunique()
    print(f"  개인 수: {n_individuals}")
    
    # 2. Config 생성
    print("\n[2] Config 생성")
    PATHS = {
        'health_concern_to_perceived_benefit': True,
        'perceived_benefit_to_purchase_intention': True
    }
    
    config = create_sugar_substitute_multi_lv_config(
        paths=PATHS,
        main_lvs=['purchase_intention'],
        n_draws=100,
        max_iterations=1000
    )
    
    # 3. 측정모델 및 구조모델 생성
    print("\n[3] 모델 생성")
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    
    # CFA 결과 로드
    pkl_path = project_root / 'results' / 'sequential_stage_wise' / 'cfa_results.pkl'
    with open(pkl_path, 'rb') as f:
        cfa_results = pickle.load(f)
    
    loadings_df = cfa_results['loadings']
    errors_df = cfa_results['measurement_errors']
    intercepts_df = cfa_results.get('intercepts', None)
    
    # 측정모델에 CFA 결과 설정
    for lv_name, model in measurement_model.models.items():
        lv_config = config.measurement_configs[lv_name]
        indicators = lv_config.indicators
        
        zeta_values = []
        for indicator in indicators:
            row = loadings_df[(loadings_df['lval'] == indicator) &
                             (loadings_df['op'] == '~') &
                             (loadings_df['rval'] == lv_name)]
            if not row.empty:
                zeta_values.append(float(row['Estimate'].iloc[0]))
            else:
                zeta_values.append(1.0)
        
        sigma_sq_values = []
        for indicator in indicators:
            row = errors_df[(errors_df['lval'] == indicator) &
                           (errors_df['op'] == '~~') &
                           (errors_df['rval'] == indicator)]
            if not row.empty:
                sigma_sq_values.append(float(row['Estimate'].iloc[0]))
            else:
                sigma_sq_values.append(0.5)
        
        alpha_values = []
        if intercepts_df is not None:
            for indicator in indicators:
                row = intercepts_df[(intercepts_df['lval'] == indicator) &
                                   (intercepts_df['op'] == '~') &
                                   (intercepts_df['rval'] == '1')]
                if not row.empty:
                    alpha_values.append(float(row['Estimate'].iloc[0]))
                else:
                    alpha_values.append(0.0)
        else:
            alpha_values = [0.0] * len(indicators)
        
        model.config.zeta = np.array(zeta_values)
        model.config.sigma_sq = np.array(sigma_sq_values)
        model.config.alpha = np.array(alpha_values)
    
    print(f"  ✅ 측정모델 설정 완료")
    
    # 4. Halton draws 생성
    print("\n[4] Halton draws 생성")
    n_dimensions = 5  # 5개 잠재변수
    halton_gen = HaltonGenerator(
        n_individuals=n_individuals,
        n_draws=10,  # 10개 draws만
        n_dimensions=n_dimensions,
        scramble=True,
        seed=42
    )
    draws = halton_gen.get_draws()
    print(f"  Draws shape: {draws.shape}")
    
    # 5. 첫 번째 개인의 잠재변수 값 생성
    print("\n[5] 첫 번째 개인의 잠재변수 값 생성")
    individual_ids = data['respondent_id'].unique()
    ind_id = individual_ids[0]
    ind_data = data[data['respondent_id'] == ind_id]
    ind_draws = draws[0]  # (10, 5)
    
    print(f"  개인 ID: {ind_id}")
    print(f"  Draws shape: {ind_draws.shape}")
    
    # 구조모델 파라미터 (초기값)
    params = {
        'gamma_health_concern_to_perceived_benefit': 0.5,
        'gamma_perceived_benefit_to_purchase_intention': 0.5
    }
    
    # 각 draw에 대해 잠재변수 값 생성
    lv_names = list(config.measurement_configs.keys())
    all_lvs = {lv: [] for lv in lv_names}
    
    for draw_idx in range(10):
        draw = ind_draws[draw_idx]  # (5,)
        
        # 구조모델로 잠재변수 예측
        lvs_dict = structural_model.predict(
            ind_data,
            draw,
            params,
            higher_order_draws=None
        )
        
        for lv_name in lv_names:
            all_lvs[lv_name].append(lvs_dict[lv_name])
    
    # 6. 통계 출력
    print("\n[6] 잠재변수 값 통계 (10 draws)")
    print(f"{'잠재변수':30s} {'평균':>12s} {'표준편차':>12s} {'최소':>12s} {'최대':>12s}")
    print("-" * 80)
    
    for lv_name in lv_names:
        values = np.array(all_lvs[lv_name])
        print(f"{lv_name:30s} {values.mean():>12.6f} {values.std():>12.6f} "
              f"{values.min():>12.6f} {values.max():>12.6f}")
    
    print("\n" + "=" * 80)
    print("조사 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()

