"""
구조모델 그래디언트 고정 버그 조사

이 스크립트는 구조모델 그래디언트가 왜 고정되는지 조사합니다.

조사 항목:
1. 잠재변수 값 분포 (health_concern, perceived_benefit, purchase_intention)
2. 구조모델 경로별 predictor 값 통계
3. 그래디언트 계산에 사용되는 중간값 확인

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


def main():
    print("=" * 80)
    print("구조모델 그래디언트 고정 버그 조사")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    n_individuals = data['respondent_id'].nunique()
    print(f"  개인 수: {n_individuals}")
    print(f"  전체 관측치: {len(data)}")
    
    # 2. CFA 결과 로드
    print("\n[2] CFA 결과 로드")
    pkl_path = project_root / 'results' / 'sequential_stage_wise' / 'cfa_results.pkl'
    
    if not pkl_path.exists():
        print(f"  ❌ CFA 결과 파일이 없습니다: {pkl_path}")
        return
    
    with open(pkl_path, 'rb') as f:
        cfa_results = pickle.load(f)
    
    print(f"  ✅ CFA 결과 로드 완료")
    
    # 3. Config 생성
    print("\n[3] Config 생성")
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
    
    # 4. 측정모델 생성 및 CFA 결과 로드
    print("\n[4] 측정모델 생성")
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    
    # CFA 결과 설정
    loadings_df = cfa_results['loadings']
    errors_df = cfa_results['measurement_errors']
    intercepts_df = cfa_results.get('intercepts', None)
    
    for lv_name, model in measurement_model.models.items():
        lv_config = config.measurement_configs[lv_name]
        indicators = lv_config.indicators
        
        # zeta
        zeta_values = []
        for indicator in indicators:
            row = loadings_df[(loadings_df['lval'] == indicator) &
                             (loadings_df['op'] == '~') &
                             (loadings_df['rval'] == lv_name)]
            if not row.empty:
                zeta_values.append(float(row['Estimate'].iloc[0]))
            else:
                zeta_values.append(1.0)
        
        # sigma_sq
        sigma_sq_values = []
        for indicator in indicators:
            row = errors_df[(errors_df['lval'] == indicator) &
                           (errors_df['op'] == '~~') &
                           (errors_df['rval'] == indicator)]
            if not row.empty:
                sigma_sq_values.append(float(row['Estimate'].iloc[0]))
            else:
                sigma_sq_values.append(0.5)
        
        # alpha
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
    
    # 5. 개인별 잠재변수 값 생성 (간단한 방법: 지표 평균)
    print("\n[5] 잠재변수 값 생성 (지표 평균 방식)")
    print("  각 개인의 지표 평균으로 잠재변수 값 근사")

    # config에 있는 잠재변수만 사용
    lv_names = list(config.measurement_configs.keys())
    print(f"  잠재변수: {lv_names}")

    # 개인별 데이터 추출
    individual_ids = data['respondent_id'].unique()[:100]  # 처음 100명

    lv_values = {lv: [] for lv in lv_names}

    for ind_id in individual_ids:
        ind_data = data[data['respondent_id'] == ind_id].iloc[0]  # 첫 번째 행만

        # 각 잠재변수 값 계산 (지표 평균)
        for lv_name in lv_names:
            lv_config = config.measurement_configs[lv_name]
            indicators = lv_config.indicators

            # 지표 값 추출
            indicator_values = []
            for ind in indicators:
                if ind in ind_data.index and not pd.isna(ind_data[ind]):
                    indicator_values.append(ind_data[ind])

            # 평균 계산
            if indicator_values:
                lv_val = np.mean(indicator_values)
            else:
                lv_val = 0.0

            lv_values[lv_name].append(lv_val)

    # 6. 잠재변수 통계
    print("\n[6] 잠재변수 값 통계 (100명 샘플)")
    print(f"{'잠재변수':30s} {'평균':>12s} {'표준편차':>12s} {'최소':>12s} {'최대':>12s}")
    print("-" * 80)

    for lv_name in lv_names:
        values = np.array(lv_values[lv_name])
        print(f"{lv_name:30s} {values.mean():>12.6f} {values.std():>12.6f} "
              f"{values.min():>12.6f} {values.max():>12.6f}")

    # 7. 구조모델 경로별 상관관계 확인
    print("\n[7] 구조모델 경로별 상관관계")
    print("-" * 80)

    # 경로 1: health_concern → perceived_benefit
    hc_values = np.array(lv_values['health_concern'])
    pb_values = np.array(lv_values['perceived_benefit'])
    corr_hc_pb = np.corrcoef(hc_values, pb_values)[0, 1]

    print(f"\n경로 1: health_concern → perceived_benefit")
    print(f"  health_concern 평균: {hc_values.mean():.4f}, 표준편차: {hc_values.std():.4f}")
    print(f"  perceived_benefit 평균: {pb_values.mean():.4f}, 표준편차: {pb_values.std():.4f}")
    print(f"  상관계수: {corr_hc_pb:.4f}")

    # 경로 2: perceived_benefit → purchase_intention
    pi_values = np.array(lv_values['purchase_intention'])
    corr_pb_pi = np.corrcoef(pb_values, pi_values)[0, 1]

    print(f"\n경로 2: perceived_benefit → purchase_intention")
    print(f"  perceived_benefit 평균: {pb_values.mean():.4f}, 표준편차: {pb_values.std():.4f}")
    print(f"  purchase_intention 평균: {pi_values.mean():.4f}, 표준편차: {pi_values.std():.4f}")
    print(f"  상관계수: {corr_pb_pi:.4f}")

    # 8. 그래디언트 계산 시뮬레이션
    print("\n[8] 그래디언트 계산 시뮬레이션")
    print("-" * 80)

    # 경로 1: gamma_health_concern_to_perceived_benefit
    gamma_hc_pb = 0.5  # 초기값
    residual_hc_pb = pb_values - gamma_hc_pb * hc_values
    grad_hc_pb = np.mean(residual_hc_pb * hc_values)

    print(f"\n경로 1: gamma_health_concern_to_perceived_benefit")
    print(f"  gamma 초기값: {gamma_hc_pb:.4f}")
    print(f"  residual 평균: {residual_hc_pb.mean():.4f}, 표준편차: {residual_hc_pb.std():.4f}")
    print(f"  predictor (health_concern) 평균: {hc_values.mean():.4f}")
    print(f"  gradient = mean(residual * predictor) = {grad_hc_pb:.6f}")

    # 경로 2: gamma_perceived_benefit_to_purchase_intention
    gamma_pb_pi = 0.5  # 초기값
    residual_pb_pi = pi_values - gamma_pb_pi * pb_values
    grad_pb_pi = np.mean(residual_pb_pi * pb_values)

    print(f"\n경로 2: gamma_perceived_benefit_to_purchase_intention")
    print(f"  gamma 초기값: {gamma_pb_pi:.4f}")
    print(f"  residual 평균: {residual_pb_pi.mean():.4f}, 표준편차: {residual_pb_pi.std():.4f}")
    print(f"  predictor (perceived_benefit) 평균: {pb_values.mean():.4f}")
    print(f"  gradient = mean(residual * predictor) = {grad_pb_pi:.6f}")
    
    print("\n" + "=" * 80)
    print("조사 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()

