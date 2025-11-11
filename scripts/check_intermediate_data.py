"""
중간 전처리 단계 데이터 확인

개인 7번이 어느 단계에서 NaN으로 변환되는지 추적합니다.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def check_individual_7(data, data_name, id_col='respondent_id'):
    """개인 7번 데이터 확인"""
    if id_col not in data.columns:
        print(f"   ⚠️ {data_name}: ID 컬럼 '{id_col}' 없음")
        return
    
    ind_7 = data[data[id_col] == 7]
    
    if len(ind_7) == 0:
        print(f"   ⚠️ {data_name}: 개인 7번 없음")
        return
    
    print(f"\n   {data_name}:")
    print(f"     행 수: {len(ind_7)}")
    
    # 측정 지표 확인
    indicators = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15']
    
    for ind in indicators:
        if ind in data.columns:
            value = ind_7[ind].iloc[0]
            is_nan = pd.isna(value)
            status = "❌ NaN" if is_nan else f"✅ {value}"
            print(f"       {ind}: {status}")
        else:
            print(f"       {ind}: (컬럼 없음)")

def main():
    print("=" * 70)
    print("중간 전처리 단계 데이터 확인")
    print("=" * 70)
    
    # 1. Survey 데이터 확인
    print("\n1. Survey 데이터 확인...")
    survey_path = project_root / 'data' / 'processed' / 'survey'
    
    survey_files = {
        'health_concern.csv': 'Health Concern',
        'perceived_benefit.csv': 'Perceived Benefit',
        'perceived_price.csv': 'Perceived Price',
        'nutrition_knowledge.csv': 'Nutrition Knowledge',
        'purchase_intention.csv': 'Purchase Intention'
    }
    
    for filename, name in survey_files.items():
        filepath = survey_path / filename
        if filepath.exists():
            try:
                data = pd.read_csv(filepath)
                check_individual_7(data, name)
            except Exception as e:
                print(f"   ⚠️ {name}: 읽기 실패 - {e}")
    
    # 2. DCE 데이터 확인
    print("\n2. DCE 데이터 확인...")
    dce_path = project_root / 'data' / 'processed' / 'dce'
    
    dce_files = {
        'dce_long_format.csv': 'DCE Long Format'
    }
    
    for filename, name in dce_files.items():
        filepath = dce_path / filename
        if filepath.exists():
            try:
                data = pd.read_csv(filepath)
                check_individual_7(data, name)
            except Exception as e:
                print(f"   ⚠️ {name}: 읽기 실패 - {e}")
    
    # 3. ICLV 통합 데이터 확인
    print("\n3. ICLV 통합 데이터 확인...")
    iclv_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    
    if iclv_path.exists():
        try:
            data = pd.read_csv(iclv_path)
            check_individual_7(data, 'ICLV Integrated')
        except Exception as e:
            print(f"   ⚠️ ICLV Integrated: 읽기 실패 - {e}")
    
    # 4. 전처리 스크립트 찾기
    print("\n4. 전처리 스크립트 확인...")
    scripts_path = project_root / 'scripts'
    
    # ICLV 관련 전처리 스크립트 찾기
    iclv_scripts = list(scripts_path.glob('*iclv*.py')) + list(scripts_path.glob('*integrate*.py'))
    
    print(f"   ICLV 관련 스크립트: {len(iclv_scripts)}개")
    for script in iclv_scripts:
        print(f"     - {script.name}")
    
    print("\n" + "=" * 70)
    print("분석 완료")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("   1. Survey 데이터에서 개인 7번이 정상인지 확인")
    print("   2. DCE 데이터에서 개인 7번이 정상인지 확인")
    print("   3. ICLV 통합 스크립트에서 merge 로직 확인")

if __name__ == '__main__':
    main()

