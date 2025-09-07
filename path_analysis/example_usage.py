"""
Path Analysis Example Usage

경로분석 모듈의 사용 예제들을 제공합니다.
다양한 매개모델과 구조모델의 예제를 포함합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

# 경로분석 모듈 임포트
from path_analysis import (
    PathAnalyzer,
    PathModelBuilder,
    EffectsCalculator,
    PathResultsExporter,
    PathAnalysisVisualizer,
    analyze_path_model,
    create_path_model,
    export_path_results,
    create_default_path_config,
    create_mediation_config
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_simple_mediation():
    """예제 1: 단순 매개모델 분석"""
    print("\n" + "=" * 60)
    print("예제 1: 단순 매개모델 분석")
    print("건강관심도 -> 지각된혜택 -> 구매의도")
    print("=" * 60)
    
    try:
        # 1. 모델 스펙 생성
        model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        print("생성된 모델 스펙:")
        print(model_spec[:500] + "..." if len(model_spec) > 500 else model_spec)
        
        # 2. 분석 실행
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        config = create_mediation_config(verbose=True)
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 3. 결과 출력
        print("\n=== 분석 결과 ===")
        if 'fit_indices' in results:
            print("적합도 지수:")
            for index, value in results['fit_indices'].items():
                if not pd.isna(value):
                    print(f"  {index}: {value:.4f}")
        
        # 4. 효과 분석
        if 'model_object' in results:
            effects_calc = EffectsCalculator(results['model_object'])
            effects = effects_calc.calculate_all_effects(
                'health_concern', 'purchase_intention', ['perceived_benefit']
            )
            
            print("\n효과 분석:")
            if 'direct_effects' in effects:
                direct = effects['direct_effects']['coefficient']
                print(f"  직접효과: {direct:.4f}")
            
            if 'indirect_effects' in effects:
                indirect = effects['indirect_effects']['total_indirect_effect']
                print(f"  간접효과: {indirect:.4f}")
            
            if 'total_effects' in effects:
                total = effects['total_effects']['total_effect']
                print(f"  총효과: {total:.4f}")
        
        # 5. 결과 저장
        saved_files = export_path_results(results, filename_prefix="simple_mediation")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
    except Exception as e:
        print(f"오류 발생: {e}")


def example_2_multiple_mediation():
    """예제 2: 다중 매개모델 분석"""
    print("\n" + "=" * 60)
    print("예제 2: 다중 매개모델 분석")
    print("건강관심도 -> [지각된혜택, 영양지식] -> 구매의도")
    print("=" * 60)
    
    try:
        # 1. 모델 스펙 생성
        model_spec = create_path_model(
            model_type='multiple_mediation',
            independent_var='health_concern',
            mediator_vars=['perceived_benefit', 'nutrition_knowledge'],
            dependent_var='purchase_intention',
            allow_mediator_correlations=True
        )
        
        print("다중 매개모델 스펙 생성 완료")
        
        # 2. 분석 실행
        variables = ['health_concern', 'perceived_benefit', 'nutrition_knowledge', 'purchase_intention']
        config = create_mediation_config(bootstrap_samples=2000)
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 3. 효과 분석
        if 'model_object' in results:
            effects_calc = EffectsCalculator(results['model_object'])
            effects = effects_calc.calculate_all_effects(
                'health_concern', 'purchase_intention', 
                ['perceived_benefit', 'nutrition_knowledge']
            )
            
            print("\n=== 다중 매개효과 분석 ===")
            if 'indirect_effects' in effects:
                indirect = effects['indirect_effects']
                print(f"총 간접효과: {indirect.get('total_indirect_effect', 0):.4f}")
                
                # 개별 매개효과
                for mediator, path_info in indirect.get('individual_paths', {}).items():
                    effect = path_info.get('indirect_effect', 0)
                    print(f"  {mediator}를 통한 간접효과: {effect:.4f}")
        
        # 4. 결과 저장
        saved_files = export_path_results(results, filename_prefix="multiple_mediation")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
    except Exception as e:
        print(f"오류 발생: {e}")


def example_3_custom_structural_model():
    """예제 3: 사용자 정의 구조모델"""
    print("\n" + "=" * 60)
    print("예제 3: 사용자 정의 구조모델")
    print("복합적인 경로 관계 분석")
    print("=" * 60)
    
    try:
        # 1. 사용자 정의 경로 설정
        variables = ['health_concern', 'perceived_benefit', 'perceived_price', 'purchase_intention']
        paths = [
            ('health_concern', 'perceived_benefit'),    # 건강관심도 -> 지각된혜택
            ('health_concern', 'perceived_price'),      # 건강관심도 -> 지각된가격
            ('perceived_benefit', 'purchase_intention'), # 지각된혜택 -> 구매의도
            ('perceived_price', 'purchase_intention'),   # 지각된가격 -> 구매의도
            ('health_concern', 'purchase_intention')     # 건강관심도 -> 구매의도 (직접효과)
        ]
        
        # 상관관계 설정 (선택사항)
        correlations = [
            ('perceived_benefit', 'perceived_price')    # 지각된혜택 <-> 지각된가격
        ]
        
        # 2. 모델 스펙 생성
        model_spec = create_path_model(
            model_type='custom',
            variables=variables,
            paths=paths,
            correlations=correlations
        )
        
        print("사용자 정의 구조모델 스펙 생성 완료")
        
        # 3. 분석 실행
        config = create_default_path_config(
            standardized=True,
            create_diagrams=True,
            verbose=True
        )
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 4. 경로계수 출력
        print("\n=== 경로계수 ===")
        if 'path_coefficients' in results:
            path_coeffs = results['path_coefficients']
            if 'paths' in path_coeffs and 'coefficients' in path_coeffs:
                for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                    coeff = path_coeffs['coefficients'].get(i, 0)
                    p_val = path_coeffs.get('p_values', {}).get(i, 1)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"  {from_var} -> {to_var}: {coeff:.4f}{sig}")
        
        # 5. 결과 저장 및 시각화
        saved_files = export_path_results(results, filename_prefix="custom_structural")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
    except Exception as e:
        print(f"오류 발생: {e}")


def example_4_effects_comparison():
    """예제 4: 효과 비교 분석"""
    print("\n" + "=" * 60)
    print("예제 4: 효과 비교 분석")
    print("여러 매개변수의 효과 크기 비교")
    print("=" * 60)
    
    try:
        # 1. 다중 매개모델 분석
        variables = ['health_concern', 'perceived_benefit', 'nutrition_knowledge', 'purchase_intention']
        model_spec = create_path_model(
            model_type='multiple_mediation',
            independent_var='health_concern',
            mediator_vars=['perceived_benefit', 'nutrition_knowledge'],
            dependent_var='purchase_intention'
        )
        
        results = analyze_path_model(model_spec, variables)
        
        # 2. 효과 계산
        if 'model_object' in results:
            effects_calc = EffectsCalculator(results['model_object'])
            
            # 각 매개변수별 효과 분석
            mediators = ['perceived_benefit', 'nutrition_knowledge']
            
            print("\n=== 매개변수별 효과 비교 ===")
            for mediator in mediators:
                # 단일 매개효과 계산
                single_effects = effects_calc.calculate_all_effects(
                    'health_concern', 'purchase_intention', [mediator]
                )
                
                if 'indirect_effects' in single_effects:
                    indirect_effect = single_effects['indirect_effects']['total_indirect_effect']
                    print(f"{mediator}: {indirect_effect:.4f}")
                
                # Sobel test
                if 'mediation_analysis' in single_effects:
                    sobel_tests = single_effects['mediation_analysis'].get('sobel_tests', {})
                    if mediator in sobel_tests:
                        z_score = sobel_tests[mediator].get('z_score', 0)
                        p_value = sobel_tests[mediator].get('p_value', 1)
                        print(f"  Sobel test: Z = {z_score:.3f}, p = {p_value:.3f}")
        
        print("\n효과 비교 분석 완료")
        
    except Exception as e:
        print(f"오류 발생: {e}")


def example_5_model_comparison():
    """예제 5: 모델 비교"""
    print("\n" + "=" * 60)
    print("예제 5: 모델 비교")
    print("직접효과 모델 vs 매개모델 비교")
    print("=" * 60)
    
    try:
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        
        # 1. 직접효과 모델 (매개변수 없음)
        direct_model_spec = """
        health_concern =~ q6 + q7 + q8 + q9 + q10
        perceived_benefit =~ q11 + q12 + q13 + q14 + q15
        purchase_intention =~ q1 + q2 + q3 + q4 + q5
        
        purchase_intention ~ health_concern
        """
        
        print("1. 직접효과 모델 분석...")
        direct_results = analyze_path_model(direct_model_spec, variables)
        
        # 2. 매개모델
        mediation_model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        print("2. 매개모델 분석...")
        mediation_results = analyze_path_model(mediation_model_spec, variables)
        
        # 3. 모델 비교
        print("\n=== 모델 비교 ===")
        
        # 적합도 지수 비교
        direct_fit = direct_results.get('fit_indices', {})
        mediation_fit = mediation_results.get('fit_indices', {})
        
        print("적합도 지수 비교:")
        for index in ['chi_square', 'cfi', 'rmsea', 'aic']:
            direct_val = direct_fit.get(index, np.nan)
            mediation_val = mediation_fit.get(index, np.nan)
            print(f"  {index}:")
            print(f"    직접모델: {direct_val:.4f}")
            print(f"    매개모델: {mediation_val:.4f}")
        
        print("\n모델 비교 완료")
        
    except Exception as e:
        print(f"오류 발생: {e}")


def run_all_examples():
    """모든 예제 실행"""
    print("=" * 60)
    print("PATH ANALYSIS EXAMPLES")
    print("=" * 60)
    
    examples = [
        example_1_simple_mediation,
        example_2_multiple_mediation,
        example_3_custom_structural_model,
        example_4_effects_comparison,
        example_5_model_comparison
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"예제 {i} 실행 중 오류: {e}")
        
        print("\n" + "-" * 60)
    
    print("모든 예제 실행 완료!")


if __name__ == "__main__":
    # 개별 예제 실행
    # example_1_simple_mediation()
    
    # 모든 예제 실행
    run_all_examples()
