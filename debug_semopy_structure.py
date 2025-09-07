#!/usr/bin/env python3
"""
semopy 파라미터 구조 디버깅
"""

import pandas as pd
import numpy as np

# semopy 직접 임포트
import semopy
from semopy import Model

def debug_semopy_structure():
    """semopy 파라미터 구조 확인"""
    print("🔍 semopy 파라미터 구조 디버깅")
    print("=" * 60)
    
    # 간단한 모델 스펙
    model_spec = """
    health_concern =~ q6 + q7 + q8
    perceived_benefit =~ q16 + q17
    perceived_benefit ~ health_concern
    """
    
    print("모델 스펙:")
    print(model_spec)
    
    # 데이터 로드
    from path_analysis import PathAnalyzer, create_default_path_config
    
    config = create_default_path_config()
    analyzer = PathAnalyzer(config)
    data = analyzer.load_data(['health_concern', 'perceived_benefit'])
    
    print(f"\n데이터 크기: {data.shape}")
    print(f"컬럼: {list(data.columns)}")
    
    # 모델 생성 및 추정
    model = Model(model_spec)
    model.fit(data)
    
    # 파라미터 확인
    params = model.inspect()
    print(f"\n전체 파라미터 수: {len(params)}")
    print(f"파라미터 컬럼: {list(params.columns)}")
    
    # 연산자별 분류
    print(f"\n연산자별 파라미터:")
    for op in params['op'].unique():
        op_params = params[params['op'] == op]
        print(f"  {op}: {len(op_params)}개")
        
        if op == '=~':
            print("    측정모델:")
            for _, row in op_params.iterrows():
                print(f"      {row['lval']} =~ {row['rval']}: {row['Estimate']:.4f}")
        elif op == '~':
            print("    회귀:")
            for _, row in op_params.iterrows():
                print(f"      {row['lval']} ~ {row['rval']}: {row['Estimate']:.4f}")
        elif op == '~~':
            print("    분산/공분산:")
            for _, row in op_params.head(5).iterrows():  # 처음 5개만
                print(f"      {row['lval']} ~~ {row['rval']}: {row['Estimate']:.4f}")
            if len(op_params) > 5:
                print(f"      ... 및 {len(op_params) - 5}개 더")
    
    # 잠재변수 식별 방법 확인
    print(f"\n잠재변수 식별:")
    
    # 방법 1: 측정모델에서 lval
    measurement_params = params[params['op'] == '=~']
    if not measurement_params.empty:
        latent_vars_method1 = list(measurement_params['lval'].unique())
        print(f"  방법 1 (=~ lval): {latent_vars_method1}")
    else:
        print(f"  방법 1 (=~ lval): 없음")
    
    # 방법 2: 관측변수가 아닌 변수들
    all_vars = set(params['lval'].unique()) | set(params['rval'].unique())
    observed_vars = set(data.columns)
    latent_vars_method2 = list(all_vars - observed_vars)
    print(f"  방법 2 (전체 - 관측): {latent_vars_method2}")
    
    # 방법 3: 모델 스펙에서 직접 추출
    import re
    latent_pattern = r'(\w+)\s*=~'
    latent_vars_method3 = re.findall(latent_pattern, model_spec)
    print(f"  방법 3 (스펙 파싱): {latent_vars_method3}")
    
    return params, latent_vars_method2, latent_vars_method3

if __name__ == "__main__":
    debug_semopy_structure()
