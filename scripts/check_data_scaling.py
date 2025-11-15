"""
데이터 스케일링 상태 확인 스크립트

test_gpu_batch_iclv.py에서 사용하는 데이터의 스케일링 상태를 확인합니다.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_data_scaling():
    """데이터 스케일링 상태 분석"""
    
    print("="*80)
    print("데이터 스케일링 상태 분석")
    print("="*80)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data_cleaned.csv'
    data = pd.read_csv(data_path)
    print(f"   데이터 shape: {data.shape}")
    print(f"   전체 개인 수: {data['respondent_id'].nunique()}")
    
    # 2. 측정 지표 변수들 정의 (test_gpu_batch_iclv.py와 동일)
    indicator_groups = {
        'health_concern': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
        'perceived_benefit': ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
        'perceived_price': ['q27', 'q28', 'q29'],
        'nutrition_knowledge': [f'q{i}' for i in range(30, 50)],  # q30-q49
        'purchase_intention': ['q18', 'q19', 'q20']
    }
    
    # 3. 선택 속성 변수들
    choice_attributes = ['sugar_free', 'health_label', 'price']
    
    # 4. 각 변수 그룹별 통계 분석
    print("\n" + "="*80)
    print("측정 지표 변수 (Indicators) 분석")
    print("="*80)
    
    all_indicators = []
    for lv_name, indicators in indicator_groups.items():
        print(f"\n[{lv_name}]")
        print(f"  지표 개수: {len(indicators)}")
        
        for ind in indicators:
            if ind in data.columns:
                all_indicators.append(ind)
                values = data[ind].dropna()
                
                print(f"\n  {ind}:")
                print(f"    - 범위: [{values.min():.2f}, {values.max():.2f}]")
                print(f"    - 평균: {values.mean():.4f}")
                print(f"    - 표준편차: {values.std():.4f}")
                print(f"    - 중앙값: {values.median():.2f}")
                print(f"    - 고유값 개수: {values.nunique()}")
                print(f"    - 고유값: {sorted(values.unique())}")
                
                # 스케일링 필요성 판단
                if values.std() > 10:
                    print(f"    ⚠️  표준편차가 큼 (>10) - 스케일링 권장")
                elif values.max() - values.min() > 100:
                    print(f"    ⚠️  범위가 큼 (>100) - 스케일링 권장")
                elif values.min() >= 1 and values.max() <= 5:
                    print(f"    ✅ Likert 척도 (1-5) - 스케일링 불필요")
                elif values.min() >= 0 and values.max() <= 1:
                    print(f"    ✅ 이진 변수 (0-1) - 스케일링 불필요")
                else:
                    print(f"    ℹ️  일반 범위 - 스케일링 검토 필요")
            else:
                print(f"\n  {ind}: ❌ 데이터에 없음")
    
    # 5. 선택 속성 변수 분석
    print("\n" + "="*80)
    print("선택 속성 변수 (Choice Attributes) 분석")
    print("="*80)
    
    for attr in choice_attributes:
        if attr in data.columns:
            values = data[attr].dropna()
            
            print(f"\n[{attr}]")
            print(f"  - 범위: [{values.min():.2f}, {values.max():.2f}]")
            print(f"  - 평균: {values.mean():.4f}")
            print(f"  - 표준편차: {values.std():.4f}")
            print(f"  - 중앙값: {values.median():.2f}")
            print(f"  - 고유값 개수: {values.nunique()}")
            
            if values.nunique() <= 20:
                print(f"  - 고유값: {sorted(values.unique())}")
            
            # 스케일링 필요성 판단
            if values.std() > 10:
                print(f"  ⚠️  표준편차가 큼 (>10) - 스케일링 권장")
            elif values.max() - values.min() > 100:
                print(f"  ⚠️  범위가 큼 (>100) - 스케일링 권장")
            elif values.min() >= 0 and values.max() <= 1:
                print(f"  ✅ 이진 변수 (0-1) - 스케일링 불필요")
            else:
                print(f"  ℹ️  일반 범위 - 스케일링 검토 필요")
        else:
            print(f"\n[{attr}]: ❌ 데이터에 없음")
    
    # 6. 전체 요약
    print("\n" + "="*80)
    print("전체 요약 및 권장사항")
    print("="*80)
    
    # 모든 지표 변수의 통계
    all_indicator_data = data[all_indicators].values.flatten()
    all_indicator_data = all_indicator_data[~np.isnan(all_indicator_data)]
    
    print(f"\n[모든 측정 지표 변수 통합 통계]")
    print(f"  - 전체 범위: [{all_indicator_data.min():.2f}, {all_indicator_data.max():.2f}]")
    print(f"  - 전체 평균: {all_indicator_data.mean():.4f}")
    print(f"  - 전체 표준편차: {all_indicator_data.std():.4f}")
    
    # 선택 속성 변수의 통계
    available_attrs = [attr for attr in choice_attributes if attr in data.columns]
    if available_attrs:
        all_attr_data = data[available_attrs].values.flatten()
        all_attr_data = all_attr_data[~np.isnan(all_attr_data)]
        
        print(f"\n[모든 선택 속성 변수 통합 통계]")
        print(f"  - 전체 범위: [{all_attr_data.min():.2f}, {all_attr_data.max():.2f}]")
        print(f"  - 전체 평균: {all_attr_data.mean():.4f}")
        print(f"  - 전체 표준편차: {all_attr_data.std():.4f}")
    
    # 권장사항
    print("\n" + "="*80)
    print("스케일링 권장사항")
    print("="*80)
    
    needs_scaling = []
    
    # 측정 지표 변수 체크
    for lv_name, indicators in indicator_groups.items():
        for ind in indicators:
            if ind in data.columns:
                values = data[ind].dropna()
                if values.std() > 10 or (values.max() - values.min()) > 100:
                    needs_scaling.append(ind)
    
    # 선택 속성 변수 체크
    for attr in choice_attributes:
        if attr in data.columns:
            values = data[attr].dropna()
            if values.std() > 10 or (values.max() - values.min()) > 100:
                needs_scaling.append(attr)
    
    if needs_scaling:
        print(f"\n⚠️  스케일링이 필요한 변수: {len(needs_scaling)}개")
        for var in needs_scaling:
            print(f"    - {var}")
        print("\n권장 스케일링 방법:")
        print("  1. StandardScaler: (X - mean) / std")
        print("  2. MinMaxScaler: (X - min) / (max - min)")
        print("  3. RobustScaler: (X - median) / IQR (이상치에 강건)")
    else:
        print("\n✅ 모든 변수가 적절한 스케일 범위에 있습니다.")
        print("   대부분 Likert 척도 (1-5) 또는 이진 변수 (0-1)로 보입니다.")
        print("   추가 스케일링이 필요하지 않을 수 있습니다.")
    
    # 7. 변수 간 스케일 차이 분석
    print("\n" + "="*80)
    print("변수 간 스케일 차이 분석")
    print("="*80)
    
    # 각 잠재변수별 지표들의 평균 표준편차
    print("\n[잠재변수별 지표 표준편차]")
    for lv_name, indicators in indicator_groups.items():
        available_inds = [ind for ind in indicators if ind in data.columns]
        if available_inds:
            stds = [data[ind].std() for ind in available_inds]
            print(f"  {lv_name}:")
            print(f"    - 평균 표준편차: {np.mean(stds):.4f}")
            print(f"    - 표준편차 범위: [{np.min(stds):.4f}, {np.max(stds):.4f}]")
            
            # 표준편차 차이가 큰 경우
            if np.max(stds) / np.min(stds) > 2:
                print(f"    ⚠️  지표 간 표준편차 차이가 큼 (비율: {np.max(stds)/np.min(stds):.2f})")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    analyze_data_scaling()

