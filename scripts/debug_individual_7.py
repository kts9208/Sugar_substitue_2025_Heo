"""
개인 7번 데이터 분석 스크립트

NaN이 발생하는 개인 7번의 데이터를 상세히 분석합니다.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.iclv_config import ICLVConfig, MeasurementConfig

def main():
    print("=" * 70)
    print("개인 7번 데이터 분석")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)
    print(f"   전체 데이터 shape: {data.shape}")

    # individual_id 컬럼 확인
    if 'respondent_id' in data.columns:
        data['individual_id'] = data['respondent_id']
    print(f"   개인 수: {data['individual_id'].nunique()}")
    
    # 2. 개인 7번 데이터 추출
    ind_7_data = data[data['individual_id'] == 7]
    print(f"\n2. 개인 7번 데이터:")
    print(f"   선택 상황 수: {len(ind_7_data)}")
    print(f"   선택 (choice): {ind_7_data['choice'].values}")
    
    # 3. 측정 지표 분석
    print(f"\n3. 측정 지표 분석:")
    
    # Health Concern (q6-q11)
    hc_indicators = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
    print(f"\n   [Health Concern] 지표: {hc_indicators}")
    for ind in hc_indicators:
        values = ind_7_data[ind].values
        print(f"     {ind}: {values[0]:.2f} (결측: {pd.isna(values[0])})")
    
    # Perceived Benefit (q12-q17)
    pb_indicators = ['q12', 'q13', 'q14', 'q15', 'q16', 'q17']
    print(f"\n   [Perceived Benefit] 지표: {pb_indicators}")
    for ind in pb_indicators:
        values = ind_7_data[ind].values
        print(f"     {ind}: {values[0]:.2f} (결측: {pd.isna(values[0])})")
    
    # Perceived Price (q27-q29)
    pp_indicators = ['q27', 'q28', 'q29']
    print(f"\n   [Perceived Price] 지표: {pp_indicators}")
    for ind in pp_indicators:
        values = ind_7_data[ind].values
        print(f"     {ind}: {values[0]:.2f} (결측: {pd.isna(values[0])})")
    
    # Nutrition Knowledge (q30-q49)
    nk_indicators = [f'q{i}' for i in range(30, 50)]
    print(f"\n   [Nutrition Knowledge] 지표: q30-q49")
    for ind in nk_indicators:
        values = ind_7_data[ind].values
        print(f"     {ind}: {values[0]:.2f} (결측: {pd.isna(values[0])})")
    
    # Purchase Intention (q18-q20)
    pi_indicators = ['q18', 'q19', 'q20']
    print(f"\n   [Purchase Intention] 지표: {pi_indicators}")
    for ind in pi_indicators:
        values = ind_7_data[ind].values
        print(f"     {ind}: {values[0]:.2f} (결측: {pd.isna(values[0])})")
    
    # 4. 선택 속성 분석
    print(f"\n4. 선택 속성 분석:")
    choice_attrs = ['sugar_free', 'health_label', 'price']
    for attr in choice_attrs:
        values = ind_7_data[attr].values
        print(f"   {attr}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")
        print(f"            값: {values}")
    
    # 5. 극단값 확인
    print(f"\n5. 극단값 확인:")
    all_indicators = hc_indicators + pb_indicators + pp_indicators + nk_indicators + pi_indicators
    for ind in all_indicators:
        values = ind_7_data[ind].values
        if not pd.isna(values[0]):
            if abs(values[0]) > 10:
                print(f"   ⚠️ {ind}: {values[0]:.2f} (극단값!)")
    
    # 6. 다른 개인과 비교
    print(f"\n6. 다른 개인과 비교:")
    
    # 개인 1번 데이터
    ind_1_data = data[data['individual_id'] == 1]
    print(f"\n   개인 1번 (정상):")
    print(f"     선택 상황 수: {len(ind_1_data)}")
    print(f"     Health Concern q6: {ind_1_data['q6'].values[0]:.2f}")
    print(f"     Perceived Benefit q12: {ind_1_data['q12'].values[0]:.2f}")
    print(f"     Purchase Intention q18: {ind_1_data['q18'].values[0]:.2f}")
    
    print(f"\n   개인 7번 (NaN 발생):")
    print(f"     선택 상황 수: {len(ind_7_data)}")
    print(f"     Health Concern q6: {ind_7_data['q6'].values[0]:.2f}")
    print(f"     Perceived Benefit q12: {ind_7_data['q12'].values[0]:.2f}")
    print(f"     Purchase Intention q18: {ind_7_data['q18'].values[0]:.2f}")
    
    # 7. 전체 통계
    print(f"\n7. 전체 데이터 통계:")
    for ind in all_indicators:
        all_values = data[ind].dropna()
        print(f"   {ind}: min={all_values.min():.2f}, max={all_values.max():.2f}, mean={all_values.mean():.2f}, std={all_values.std():.2f}")
    
    print("\n" + "=" * 70)
    print("분석 완료")
    print("=" * 70)

if __name__ == '__main__':
    main()

