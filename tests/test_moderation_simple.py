#!/usr/bin/env python3
"""
간단한 조절효과 분석 테스트 스크립트
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_data_loading():
    """데이터 로딩 테스트"""
    print("=== 1. 데이터 로딩 테스트 ===")
    try:
        from moderation_analysis import load_moderation_data, get_factor_items_mapping
        
        # 요인 매핑 확인
        mapping = get_factor_items_mapping()
        print("✅ 요인별 문항 매핑:")
        for factor, items in mapping.items():
            print(f"   {factor}: {items}")
        
        # 데이터 로드
        data = load_moderation_data(
            independent_var='health_concern',
            dependent_var='perceived_benefit',
            moderator_var='nutrition_knowledge'
        )
        print(f"✅ 데이터 로드 성공: {data.shape}")
        print(f"   컬럼: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None


def test_interaction_terms(data):
    """상호작용항 생성 테스트"""
    print("\n=== 2. 상호작용항 생성 테스트 ===")
    try:
        from moderation_analysis import create_interaction_terms
        
        interaction_data = create_interaction_terms(
            data=data,
            independent_var='health_concern',
            moderator_var='nutrition_knowledge'
        )
        print(f"✅ 상호작용항 생성 성공: {interaction_data.shape}")
        print(f"   컬럼: {list(interaction_data.columns)}")
        
        # 상호작용항 통계
        interaction_col = 'health_concern_x_nutrition_knowledge'
        if interaction_col in interaction_data.columns:
            stats = interaction_data[interaction_col].describe()
            print(f"   상호작용항 통계:\n{stats}")
        
        return interaction_data
        
    except Exception as e:
        print(f"❌ 상호작용항 생성 실패: {e}")
        return None


def test_simple_regression(data):
    """간단한 회귀분석 테스트 (semopy 대신)"""
    print("\n=== 3. 간단한 회귀분석 테스트 ===")
    try:
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        # 변수 준비
        X = data[['health_concern', 'nutrition_knowledge', 'health_concern_x_nutrition_knowledge']]
        y = data['perceived_benefit']
        
        # 회귀분석
        model = LinearRegression()
        model.fit(X, y)
        
        # 계수 확인
        coefficients = {
            'health_concern': model.coef_[0],
            'nutrition_knowledge': model.coef_[1], 
            'health_concern_x_nutrition_knowledge': model.coef_[2]
        }
        
        print("✅ 회귀분석 성공")
        print("📊 회귀계수:")
        for var, coef in coefficients.items():
            print(f"   {var}: {coef:.6f}")
        
        # 상호작용 계수 특별 표시
        interaction_coef = coefficients['health_concern_x_nutrition_knowledge']
        print(f"\n🎯 상호작용 계수: {interaction_coef:.6f}")
        
        # R-squared
        r2 = model.score(X, y)
        print(f"   R-squared: {r2:.4f}")
        
        return coefficients
        
    except Exception as e:
        print(f"❌ 회귀분석 실패: {e}")
        return None


def test_results_export():
    """결과 저장 테스트"""
    print("\n=== 4. 결과 저장 테스트 ===")
    try:
        from moderation_analysis import export_moderation_results
        
        # 테스트용 결과 생성
        test_results = {
            'variables': {
                'independent': 'health_concern',
                'dependent': 'perceived_benefit',
                'moderator': 'nutrition_knowledge',
                'interaction': 'health_concern_x_nutrition_knowledge'
            },
            'model_info': {
                'n_observations': 300,
                'n_parameters': 4
            },
            'coefficients': {
                'health_concern': {
                    'estimate': 0.5234,
                    'std_error': 0.1123,
                    'z_value': 4.6612,
                    'p_value': 0.0001,
                    'significant': True
                },
                'health_concern_x_nutrition_knowledge': {
                    'estimate': 0.1789,
                    'std_error': 0.0654,
                    'z_value': 2.7345,
                    'p_value': 0.0062,
                    'significant': True
                }
            },
            'moderation_test': {
                'interaction_coefficient': 0.1789,
                'std_error': 0.0654,
                'z_value': 2.7345,
                'p_value': 0.0062,
                'significant': True,
                'effect_size': 0.1789,
                'interpretation': '조절변수가 증가할수록 독립변수의 효과가 강화됨'
            },
            'simple_slopes': {
                'low': {'simple_slope': 0.3445, 'p_value': 0.0052, 'significant': True},
                'mean': {'simple_slope': 0.5234, 'p_value': 0.0001, 'significant': True},
                'high': {'simple_slope': 0.7023, 'p_value': 0.0001, 'significant': True}
            },
            'fit_indices': {
                'CFI': 0.956,
                'RMSEA': 0.067
            }
        }
        
        # 결과 저장
        saved_files = export_moderation_results(test_results, analysis_name='simple_test')
        
        print(f"✅ 결과 저장 성공: {len(saved_files)}개 파일")
        for file_type, file_path in saved_files.items():
            print(f"   - {file_type}: {file_path}")
            
        # 상호작용 계수 확인
        interaction_coef = test_results['moderation_test']['interaction_coefficient']
        p_value = test_results['moderation_test']['p_value']
        significant = test_results['moderation_test']['significant']
        
        print(f"\n🎯 저장된 상호작용 계수: {interaction_coef:.6f}")
        print(f"   P값: {p_value:.6f}")
        print(f"   유의성: {'✅ 유의함' if significant else '❌ 유의하지 않음'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🔍 조절효과 분석 모듈 간단 테스트")
    print("=" * 60)
    
    # 1. 데이터 로딩
    data = test_data_loading()
    if data is None:
        print("❌ 데이터 로딩 실패로 테스트 중단")
        return
    
    # 2. 상호작용항 생성
    interaction_data = test_interaction_terms(data)
    if interaction_data is None:
        print("❌ 상호작용항 생성 실패로 테스트 중단")
        return
    
    # 3. 간단한 회귀분석
    coefficients = test_simple_regression(interaction_data)
    if coefficients is None:
        print("❌ 회귀분석 실패")
    
    # 4. 결과 저장
    export_success = test_results_export()
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("🎉 테스트 완료 요약")
    print("=" * 60)
    print("✅ 데이터 로딩: 성공")
    print("✅ 상호작용항 생성: 성공")
    print(f"{'✅' if coefficients else '❌'} 회귀분석: {'성공' if coefficients else '실패'}")
    print(f"{'✅' if export_success else '❌'} 결과 저장: {'성공' if export_success else '실패'}")
    
    if coefficients:
        interaction_coef = coefficients['health_concern_x_nutrition_knowledge']
        print(f"\n🎯 최종 상호작용 계수: {interaction_coef:.6f}")
        print("   (sklearn 기반 회귀분석 결과)")


if __name__ == "__main__":
    main()
