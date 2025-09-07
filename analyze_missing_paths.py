#!/usr/bin/env python3
"""
누락된 경로 원인 분석 및 해결
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 우리 모듈 임포트
from path_analysis import (
    PathAnalyzer,
    create_default_path_config,
    create_path_model
)

def analyze_missing_paths():
    """누락된 경로 원인 분석"""
    print("🔍 누락된 경로 원인 분석")
    print("=" * 60)
    
    # 5개 요인 모델 설정
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    # 현재 모델의 경로 (8개)
    current_paths = [
        ('health_concern', 'perceived_benefit'),
        ('health_concern', 'perceived_price'),
        ('health_concern', 'nutrition_knowledge'),
        ('nutrition_knowledge', 'perceived_benefit'),
        ('perceived_benefit', 'purchase_intention'),
        ('perceived_price', 'purchase_intention'),
        ('nutrition_knowledge', 'purchase_intention'),
        ('health_concern', 'purchase_intention')
    ]
    
    # 모든 가능한 경로 (20개)
    from itertools import permutations
    all_possible_paths = [(from_var, to_var) for from_var, to_var in permutations(variables, 2)]
    
    # 누락된 경로 (12개)
    missing_paths = [path for path in all_possible_paths if path not in current_paths]
    
    print(f"현재 모델 경로: {len(current_paths)}개")
    print(f"가능한 총 경로: {len(all_possible_paths)}개")
    print(f"누락된 경로: {len(missing_paths)}개")
    
    print(f"\n현재 포함된 경로:")
    for i, (from_var, to_var) in enumerate(current_paths, 1):
        print(f"  {i:2d}. {from_var} → {to_var}")
    
    print(f"\n누락된 경로:")
    for i, (from_var, to_var) in enumerate(missing_paths, 1):
        print(f"  {i:2d}. {from_var} → {to_var}")
    
    return current_paths, missing_paths, all_possible_paths

def test_saturated_model():
    """포화모델 테스트 (모든 경로 포함)"""
    print(f"\n" + "=" * 60)
    print("포화모델 테스트 (모든 20개 경로 포함)")
    print("=" * 60)
    
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    # 모든 가능한 경로 생성
    from itertools import permutations
    all_paths = [(from_var, to_var) for from_var, to_var in permutations(variables, 2)]
    
    print(f"포화모델 경로 수: {len(all_paths)}개")
    
    # 포화모델 생성
    model_spec = create_path_model(
        model_type='custom',
        variables=variables,
        paths=all_paths,
        correlations=None  # 포화모델에서는 상관관계 불필요
    )
    
    print(f"\n생성된 포화모델 스펙:")
    print(model_spec)
    
    config = create_default_path_config(verbose=False)
    analyzer = PathAnalyzer(config)
    data = analyzer.load_data(variables)
    
    try:
        print(f"\n포화모델 추정 시도...")
        results = analyzer.fit_model(model_spec, data)
        
        path_coefficients = results.get('path_coefficients', {})
        path_analysis = results.get('path_analysis', {})
        
        print(f"✅ 포화모델 추정 성공!")
        print(f"추정된 경로 수: {len(path_coefficients.get('paths', []))}개")
        print(f"경로 포함률: {path_analysis.get('coverage_ratio', 0):.1%}")
        print(f"누락된 경로: {path_analysis.get('n_missing_paths', 0)}개")
        
        # 경로계수 출력
        paths = path_coefficients.get('paths', [])
        coefficients = path_coefficients.get('coefficients', {})
        p_values = path_coefficients.get('p_values', {})
        
        print(f"\n포화모델 경로계수:")
        for i, (from_var, to_var) in enumerate(paths):
            coeff = coefficients.get(i, 0)
            p_val = p_values.get(i, 1)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {from_var} → {to_var}: {coeff:8.4f} {sig}")
        
        return results
        
    except Exception as e:
        print(f"❌ 포화모델 추정 실패: {e}")
        print(f"실패 원인 분석:")
        
        # 모델 식별 문제 확인
        print(f"  - 모델 식별 문제: 포화모델은 자유도가 0 또는 음수가 될 수 있음")
        print(f"  - 파라미터 수: {len(all_paths)} (구조적 경로)")
        print(f"  - 관측변수 수: {data.shape[1]}")
        print(f"  - 표본 크기: {data.shape[0]}")
        
        return None

def test_partial_saturated_models():
    """부분 포화모델들 테스트"""
    print(f"\n" + "=" * 60)
    print("부분 포화모델 테스트")
    print("=" * 60)
    
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    # 현재 모델에 누락된 경로를 하나씩 추가해보기
    current_paths = [
        ('health_concern', 'perceived_benefit'),
        ('health_concern', 'perceived_price'),
        ('health_concern', 'nutrition_knowledge'),
        ('nutrition_knowledge', 'perceived_benefit'),
        ('perceived_benefit', 'purchase_intention'),
        ('perceived_price', 'purchase_intention'),
        ('nutrition_knowledge', 'purchase_intention'),
        ('health_concern', 'purchase_intention')
    ]
    
    # 누락된 경로들
    from itertools import permutations
    all_possible_paths = [(from_var, to_var) for from_var, to_var in permutations(variables, 2)]
    missing_paths = [path for path in all_possible_paths if path not in current_paths]
    
    successful_additions = []
    failed_additions = []
    
    print(f"누락된 경로를 하나씩 추가하여 테스트:")
    
    for i, missing_path in enumerate(missing_paths[:5], 1):  # 처음 5개만 테스트
        print(f"\n{i}. {missing_path[0]} → {missing_path[1]} 추가 테스트")
        
        # 현재 경로 + 누락된 경로 1개
        test_paths = current_paths + [missing_path]
        
        try:
            model_spec = create_path_model(
                model_type='custom',
                variables=variables,
                paths=test_paths,
                correlations=None
            )
            
            config = create_default_path_config(verbose=False)
            analyzer = PathAnalyzer(config)
            data = analyzer.load_data(variables)
            results = analyzer.fit_model(model_spec, data)
            
            path_coefficients = results.get('path_coefficients', {})
            paths = path_coefficients.get('paths', [])
            coefficients = path_coefficients.get('coefficients', {})
            
            # 추가된 경로의 계수 찾기
            added_coeff = None
            for j, (from_var, to_var) in enumerate(paths):
                if (from_var, to_var) == missing_path:
                    added_coeff = coefficients.get(j, 0)
                    break
            
            print(f"   ✅ 성공: 계수 = {added_coeff:.4f}")
            successful_additions.append((missing_path, added_coeff))
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
            failed_additions.append((missing_path, str(e)))
    
    print(f"\n부분 포화모델 테스트 결과:")
    print(f"성공한 경로 추가: {len(successful_additions)}개")
    print(f"실패한 경로 추가: {len(failed_additions)}개")
    
    if successful_additions:
        print(f"\n성공한 경로들:")
        for path, coeff in successful_additions:
            print(f"  {path[0]} → {path[1]}: {coeff:.4f}")
    
    if failed_additions:
        print(f"\n실패한 경로들:")
        for path, error in failed_additions:
            print(f"  {path[0]} → {path[1]}: {error}")
    
    return successful_additions, failed_additions

def create_extended_model():
    """확장된 모델 생성 (성공한 경로들 포함)"""
    print(f"\n" + "=" * 60)
    print("확장된 모델 생성")
    print("=" * 60)
    
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    # 기본 경로
    base_paths = [
        ('health_concern', 'perceived_benefit'),
        ('health_concern', 'perceived_price'),
        ('health_concern', 'nutrition_knowledge'),
        ('nutrition_knowledge', 'perceived_benefit'),
        ('perceived_benefit', 'purchase_intention'),
        ('perceived_price', 'purchase_intention'),
        ('nutrition_knowledge', 'purchase_intention'),
        ('health_concern', 'purchase_intention')
    ]
    
    # 추가할 경로들 (이론적으로 타당한 경로들)
    additional_paths = [
        ('perceived_benefit', 'perceived_price'),  # 혜택 인식이 가격 인식에 영향
        ('perceived_benefit', 'nutrition_knowledge'),  # 혜택 인식이 영양 지식에 영향
        ('perceived_price', 'perceived_benefit'),  # 가격 인식이 혜택 인식에 영향
        ('nutrition_knowledge', 'perceived_price'),  # 영양 지식이 가격 인식에 영향
    ]
    
    extended_paths = base_paths + additional_paths
    
    print(f"기본 경로: {len(base_paths)}개")
    print(f"추가 경로: {len(additional_paths)}개")
    print(f"확장 모델 총 경로: {len(extended_paths)}개")
    
    try:
        model_spec = create_path_model(
            model_type='custom',
            variables=variables,
            paths=extended_paths,
            correlations=None
        )
        
        config = create_default_path_config(verbose=False)
        analyzer = PathAnalyzer(config)
        data = analyzer.load_data(variables)
        results = analyzer.fit_model(model_spec, data)
        
        path_coefficients = results.get('path_coefficients', {})
        path_analysis = results.get('path_analysis', {})
        fit_indices = results.get('fit_indices', {})
        
        print(f"✅ 확장 모델 추정 성공!")
        print(f"추정된 경로 수: {len(path_coefficients.get('paths', []))}개")
        print(f"경로 포함률: {path_analysis.get('coverage_ratio', 0):.1%}")
        
        # 적합도 지수
        print(f"\n적합도 지수:")
        for index_name, value in fit_indices.items():
            if hasattr(value, 'iloc'):
                numeric_value = value.iloc[0] if len(value) > 0 else np.nan
            else:
                numeric_value = value
            
            if isinstance(numeric_value, (int, float)) and not pd.isna(numeric_value):
                print(f"  {index_name.upper()}: {numeric_value:.4f}")
        
        # 추가된 경로들의 계수
        paths = path_coefficients.get('paths', [])
        coefficients = path_coefficients.get('coefficients', {})
        p_values = path_coefficients.get('p_values', {})
        
        print(f"\n추가된 경로들의 계수:")
        for from_var, to_var in additional_paths:
            for i, (path_from, path_to) in enumerate(paths):
                if (path_from, path_to) == (from_var, to_var):
                    coeff = coefficients.get(i, 0)
                    p_val = p_values.get(i, 1)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"  {from_var} → {to_var}: {coeff:8.4f} (p={p_val:.4f}) {sig}")
                    break
        
        return results
        
    except Exception as e:
        print(f"❌ 확장 모델 추정 실패: {e}")
        return None

def main():
    """메인 함수"""
    print("🔍 누락된 경로 원인 분석 및 해결")
    print(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 누락된 경로 분석
        current_paths, missing_paths, all_possible_paths = analyze_missing_paths()
        
        # 2. 포화모델 테스트
        saturated_results = test_saturated_model()
        
        # 3. 부분 포화모델 테스트
        successful_additions, failed_additions = test_partial_saturated_models()
        
        # 4. 확장된 모델 생성
        extended_results = create_extended_model()
        
        print(f"\n" + "=" * 60)
        print("📊 누락 경로 분석 결과")
        print("=" * 60)
        
        print(f"🔍 누락 원인:")
        print(f"  - 모델 설계 시 이론적 근거에 따라 일부 경로만 포함")
        print(f"  - 포화모델은 식별 문제로 추정 어려움")
        print(f"  - 역방향 경로나 상호 영향 경로는 이론적 타당성 검토 필요")
        
        print(f"\n✅ 해결 방안:")
        print(f"  - 이론적으로 타당한 경로들을 단계적으로 추가")
        print(f"  - 모델 적합도와 경로 유의성을 고려한 모델 개선")
        print(f"  - 확장 모델을 통해 더 많은 경로 포함 가능")
        
        if extended_results:
            path_analysis = extended_results.get('path_analysis', {})
            print(f"  - 확장 모델 경로 포함률: {path_analysis.get('coverage_ratio', 0):.1%}")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
