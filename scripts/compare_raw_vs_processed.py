"""
원본 데이터와 전처리된 데이터 비교

개인 7번의 데이터가 어느 단계에서 결측치로 바뀌는지 추적합니다.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def main():
    print("=" * 70)
    print("원본 데이터 vs 전처리 데이터 비교")
    print("=" * 70)
    
    # 1. 원본 데이터 로드
    print("\n1. 원본 데이터 로드 중...")
    raw_path = project_root / 'data' / 'raw' / 'Sugar_substitue_Raw data_251108.xlsx'
    
    if not raw_path.exists():
        raw_path = project_root / 'data' / 'raw' / 'Sugar_substitue_Raw data_250730.xlsx'
    
    print(f"   파일: {raw_path.name}")
    raw_data = pd.read_excel(raw_path)
    print(f"   원본 데이터 shape: {raw_data.shape}")
    print(f"   원본 데이터 컬럼: {list(raw_data.columns[:20])}")
    
    # 2. 전처리된 데이터 로드
    print("\n2. 전처리된 데이터 로드 중...")
    processed_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    processed_data = pd.read_csv(processed_path)
    print(f"   전처리 데이터 shape: {processed_data.shape}")
    
    # 3. 개인 ID 매칭 확인
    print("\n3. 개인 ID 확인...")
    
    # 원본 데이터의 개인 ID 컬럼 찾기
    possible_id_cols = ['no', 'respondent_id', 'ID', 'id', 'participant_id', 'subject_id']
    raw_id_col = None

    for col in possible_id_cols:
        if col in raw_data.columns:
            raw_id_col = col
            break

    if raw_id_col is None:
        print("   원본 데이터의 ID 컬럼을 찾을 수 없습니다.")
        print(f"   사용 가능한 컬럼: {list(raw_data.columns[:30])}")
        return
    
    print(f"   원본 ID 컬럼: {raw_id_col}")
    print(f"   원본 개인 수: {raw_data[raw_id_col].nunique()}")
    print(f"   전처리 개인 수: {processed_data['respondent_id'].nunique()}")
    
    # 4. 개인 7번 데이터 비교
    print("\n4. 개인 7번 데이터 비교...")
    
    # 원본에서 개인 7번 찾기
    raw_ind_7 = raw_data[raw_data[raw_id_col] == 7]
    processed_ind_7 = processed_data[processed_data['respondent_id'] == 7]
    
    print(f"\n   원본 데이터 (개인 7):")
    print(f"     행 수: {len(raw_ind_7)}")
    
    if len(raw_ind_7) > 0:
        # 측정 지표 확인
        indicators = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15']
        
        print(f"\n     측정 지표 (원본):")
        for ind in indicators:
            if ind in raw_ind_7.columns:
                value = raw_ind_7[ind].iloc[0]
                print(f"       {ind}: {value} (결측: {pd.isna(value)})")
            else:
                print(f"       {ind}: 컬럼 없음")
    else:
        print(f"     ⚠️ 원본 데이터에 개인 7번이 없습니다!")
    
    print(f"\n   전처리 데이터 (개인 7):")
    print(f"     행 수: {len(processed_ind_7)}")
    
    if len(processed_ind_7) > 0:
        print(f"\n     측정 지표 (전처리):")
        for ind in indicators:
            if ind in processed_ind_7.columns:
                value = processed_ind_7[ind].iloc[0]
                print(f"       {ind}: {value} (결측: {pd.isna(value)})")
            else:
                print(f"       {ind}: 컬럼 없음")
    
    # 5. 중간 단계 데이터 확인
    print("\n5. 중간 전처리 단계 데이터 확인...")
    
    # survey 데이터
    survey_path = project_root / 'data' / 'processed' / 'survey'
    if survey_path.exists():
        survey_files = list(survey_path.glob('*.csv'))
        print(f"\n   Survey 데이터 파일: {len(survey_files)}개")
        for f in survey_files:
            print(f"     - {f.name}")
            try:
                survey_data = pd.read_csv(f)
                if 'respondent_id' in survey_data.columns:
                    survey_ind_7 = survey_data[survey_data['respondent_id'] == 7]
                    if len(survey_ind_7) > 0:
                        print(f"       개인 7 존재: {len(survey_ind_7)}행")
                        if 'q6' in survey_data.columns:
                            print(f"       q6 값: {survey_ind_7['q6'].iloc[0]}")
            except Exception as e:
                print(f"       읽기 실패: {e}")
    
    # DCE 데이터
    dce_path = project_root / 'data' / 'processed' / 'dce'
    if dce_path.exists():
        dce_files = list(dce_path.glob('*.csv'))
        print(f"\n   DCE 데이터 파일: {len(dce_files)}개")
        for f in dce_files:
            print(f"     - {f.name}")
    
    # 6. 전처리 스크립트 확인
    print("\n6. 전처리 스크립트 확인...")
    scripts_path = project_root / 'scripts'
    preprocessing_scripts = list(scripts_path.glob('*preprocess*.py')) + list(scripts_path.glob('*prepare*.py'))
    
    print(f"   전처리 관련 스크립트: {len(preprocessing_scripts)}개")
    for script in preprocessing_scripts:
        print(f"     - {script.name}")
    
    # 7. 원본 데이터의 전체 결측 패턴 확인
    print("\n7. 원본 데이터의 결측 패턴 확인...")
    
    if len(raw_ind_7) > 0:
        print(f"\n   개인 7번 원본 데이터 전체 컬럼 (처음 50개):")
        for col in list(raw_ind_7.columns[:50]):
            value = raw_ind_7[col].iloc[0]
            if pd.isna(value):
                print(f"     {col}: NaN ⚠️")
            else:
                print(f"     {col}: {value}")
    
    print("\n" + "=" * 70)
    print("분석 완료")
    print("=" * 70)

if __name__ == '__main__':
    main()

