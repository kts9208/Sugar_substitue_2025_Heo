#!/usr/bin/env python3
"""
경로 추출 디버깅
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 우리 모듈 임포트
from path_analysis import (
    PathAnalyzer,
    create_default_path_config,
    create_path_model
)

def debug_path_extraction():
    """경로 추출 과정 디버깅"""
    print("🔍 경로 추출 디버깅")
    print("=" * 60)
    
    # 5개 요인 모델 설정
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    paths = [
        ('health_concern', 'perceived_benefit'),
        ('health_concern', 'perceived_price'),
        ('health_concern', 'nutrition_knowledge'),
        ('nutrition_knowledge', 'perceived_benefit'),
        ('perceived_benefit', 'purchase_intention'),
        ('perceived_price', 'purchase_intention'),
        ('nutrition_knowledge', 'purchase_intention'),
        ('health_concern', 'purchase_intention')
    ]
    
    # 모델 생성
    model_spec = create_path_model(
        model_type='custom',
        variables=variables,
        paths=paths,
        correlations=None
    )
    
    print("생성된 모델 스펙:")
    print(model_spec)
    print()
    
    # 분석 실행
    config = create_default_path_config(verbose=True)
    analyzer = PathAnalyzer(config)
    data = analyzer.load_data(variables)
    
    # 모델 추정
    import semopy
    from semopy import Model
    
    model = Model(model_spec)
    model.fit(data)
    
    # 파라미터 직접 확인
    params = model.inspect()
    print("전체 파라미터:")
    print(params)
    print()
    
    # 측정모델 확인
    measurement_params = params[params['op'] == '=~']
    print("측정모델 파라미터:")
    print(measurement_params)
    print()
    
    if not measurement_params.empty:
        latent_variables = list(measurement_params['lval'].unique())
        print(f"잠재변수: {latent_variables}")
        
        # 구조적 경로 확인
        structural_params = params[
            (params['op'] == '~') & 
            params['lval'].isin(latent_variables) & 
            params['rval'].isin(latent_variables)
        ]
        print(f"\n구조적 경로:")
        print(structural_params)
    else:
        print("❌ 측정모델 파라미터가 없습니다!")
    
    # 우리 모듈로 분석
    print(f"\n" + "=" * 40)
    print("우리 모듈 분석 결과:")
    results = analyzer.fit_model(model_spec, data)
    
    path_coefficients = results.get('path_coefficients', {})
    print(f"경로계수 결과: {path_coefficients}")
    
    path_analysis = results.get('path_analysis', {})
    print(f"경로 분석 결과: {path_analysis}")

if __name__ == "__main__":
    debug_path_extraction()
