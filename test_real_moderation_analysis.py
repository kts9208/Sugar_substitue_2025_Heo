"""
실제 데이터를 사용한 조절효과 분석 테스트

이 스크립트는 processed_data/survey_data의 실제 데이터를 사용하여
조절효과 분석 모듈을 테스트하고 결과를 검토합니다.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    """데이터 로딩 테스트"""
    print("=" * 60)
    print("1. 데이터 로딩 테스트")
    print("=" * 60)
    
    try:
        from moderation_analysis import load_moderation_data, get_available_factors
        
        # 사용 가능한 요인 확인
        available_factors = get_available_factors()
        print(f"사용 가능한 요인들: {available_factors}")
        
        # 데이터 로딩 테스트
        data = load_moderation_data(
            independent_var='health_concern',
            dependent_var='purchase_intention', 
            moderator_var='nutrition_knowledge'
        )
        
        print(f"✅ 데이터 로딩 성공")
        print(f"   데이터 크기: {data.shape}")
        print(f"   컬럼: {list(data.columns)}")
        print(f"   기술통계:")
        print(data.describe())
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None

def test_interaction_terms(data):
    """상호작용항 생성 테스트"""
    print("\n" + "=" * 60)
    print("2. 상호작용항 생성 테스트")
    print("=" * 60)
    
    if data is None:
        print("❌ 데이터가 없어 테스트를 건너뜁니다.")
        return None
    
    try:
        from moderation_analysis import create_interaction_terms
        
        interaction_data = create_interaction_terms(
            data=data,
            independent_var='health_concern',
            moderator_var='nutrition_knowledge',
            method='product'
        )
        
        print(f"✅ 상호작용항 생성 성공")
        print(f"   데이터 크기: {interaction_data.shape}")
        print(f"   새로운 컬럼: {[col for col in interaction_data.columns if col not in data.columns]}")
        
        # 상호작용항 통계
        interaction_col = 'health_concern_x_nutrition_knowledge'
        if interaction_col in interaction_data.columns:
            stats = interaction_data[interaction_col].describe()
            print(f"   상호작용항 통계:")
            print(f"   {stats}")
            
            # 상관관계 확인
            corr_matrix = interaction_data[['health_concern', 'nutrition_knowledge', interaction_col]].corr()
            print(f"   상관관계 매트릭스:")
            print(corr_matrix)
        
        return interaction_data
        
    except Exception as e:
        print(f"❌ 상호작용항 생성 실패: {e}")
        return None

def test_moderation_analysis(data):
    """조절효과 분석 테스트"""
    print("\n" + "=" * 60)
    print("3. 조절효과 분석 테스트")
    print("=" * 60)
    
    if data is None:
        print("❌ 데이터가 없어 테스트를 건너뜁니다.")
        return None
    
    try:
        from moderation_analysis import analyze_moderation_effects
        
        results = analyze_moderation_effects(
            independent_var='health_concern',
            dependent_var='purchase_intention',
            moderator_var='nutrition_knowledge',
            data=data
        )
        
        print(f"✅ 조절효과 분석 성공")
        print(f"   분석 결과 키: {list(results.keys())}")
        
        # 변수 정보
        if 'variables' in results:
            variables = results['variables']
            print(f"   변수 정보:")
            for key, value in variables.items():
                print(f"     {key}: {value}")
        
        # 조절효과 검정 결과
        if 'moderation_test' in results:
            moderation_test = results['moderation_test']
            print(f"   조절효과 검정:")
            print(f"     상호작용 계수: {moderation_test.get('interaction_coefficient', 'N/A')}")
            print(f"     p-value: {moderation_test.get('p_value', 'N/A')}")
            print(f"     유의성: {'유의함' if moderation_test.get('significant', False) else '유의하지 않음'}")
            print(f"     해석: {moderation_test.get('interpretation', 'N/A')}")
        
        # 단순기울기 분석
        if 'simple_slopes' in results:
            simple_slopes = results['simple_slopes']
            print(f"   단순기울기 분석:")
            for level, slope_info in simple_slopes.items():
                if isinstance(slope_info, dict):
                    coeff = slope_info.get('coefficient', 'N/A')
                    p_val = slope_info.get('p_value', 'N/A')
                    if isinstance(coeff, (int, float)):
                        print(f"     {level}: 계수={coeff:.4f}, p-value={p_val}")
                    else:
                        print(f"     {level}: 계수={coeff}, p-value={p_val}")
        
        # 적합도 지수
        if 'fit_indices' in results:
            fit_indices = results['fit_indices']
            print(f"   모델 적합도:")
            for index, value in fit_indices.items():
                print(f"     {index}: {value}")
        
        return results
        
    except Exception as e:
        print(f"❌ 조절효과 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_results_export(results):
    """결과 저장 테스트"""
    print("\n" + "=" * 60)
    print("4. 결과 저장 테스트")
    print("=" * 60)
    
    if results is None:
        print("❌ 분석 결과가 없어 테스트를 건너뜁니다.")
        return None
    
    try:
        from moderation_analysis import export_moderation_results
        
        saved_files = export_moderation_results(
            results,
            analysis_name="real_data_test"
        )
        
        print(f"✅ 결과 저장 성공")
        print(f"   저장된 파일들:")
        for file_type, file_path in saved_files.items():
            print(f"     {file_type}: {file_path}")
            
            # 파일 존재 확인
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"       크기: {file_size} bytes")
            else:
                print(f"       ⚠️ 파일이 존재하지 않습니다")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return None

def test_visualization(results):
    """시각화 테스트"""
    print("\n" + "=" * 60)
    print("5. 시각화 테스트")
    print("=" * 60)
    
    if results is None:
        print("❌ 분석 결과가 없어 테스트를 건너뜁니다.")
        return
    
    try:
        from moderation_analysis import visualize_moderation_analysis
        
        plot_files = visualize_moderation_analysis(
            results,
            save_plots=True,
            analysis_name="real_data_test"
        )
        
        print(f"✅ 시각화 성공")
        print(f"   생성된 플롯 파일들:")
        for plot_type, plot_path in plot_files.items():
            print(f"     {plot_type}: {plot_path}")
        
    except Exception as e:
        print(f"❌ 시각화 실패: {e}")

def main():
    """메인 테스트 함수"""
    print("실제 데이터를 사용한 조절효과 분석 테스트 시작")
    print("=" * 80)
    
    # 1. 데이터 로딩
    data = test_data_loading()
    
    # 2. 상호작용항 생성
    interaction_data = test_interaction_terms(data)
    
    # 3. 조절효과 분석
    results = test_moderation_analysis(data)
    
    # 4. 결과 저장
    saved_files = test_results_export(results)
    
    # 5. 시각화
    test_visualization(results)
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)

if __name__ == "__main__":
    main()
