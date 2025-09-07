#!/usr/bin/env python3
"""
간단한 경로분석 테스트
"""

import pandas as pd
import numpy as np
from path_analysis import (
    PathAnalyzer,
    create_default_path_config,
    create_path_model,
    export_path_results
)

def simple_test():
    """간단한 경로분석 테스트"""
    print("=== 간단한 경로분석 테스트 ===")
    
    try:
        # 1. 모델 스펙 생성
        model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        print("모델 스펙:")
        print(model_spec)
        
        # 2. 분석기 초기화
        config = create_default_path_config(verbose=True)
        analyzer = PathAnalyzer(config)
        
        # 3. 데이터 로드
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        data = analyzer.load_data(variables)
        print(f"\n데이터 로드 완료: {data.shape}")
        print(f"컬럼: {list(data.columns)}")
        
        # 4. 모델 적합
        results = analyzer.fit_model(model_spec, data)
        print(f"\n모델 적합 완료!")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        
        # 5. 적합도 지수 출력
        if 'fit_indices' in results and results['fit_indices']:
            print("\n적합도 지수:")
            for index, value in results['fit_indices'].items():
                try:
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        print(f"  {index}: {value:.4f}")
                    elif hasattr(value, '__len__') and len(value) > 0:
                        print(f"  {index}: {value}")
                except:
                    print(f"  {index}: {value}")
        
        # 6. 경로계수 출력
        if 'path_coefficients' in results and results['path_coefficients']:
            print("\n경로계수:")
            path_coeffs = results['path_coefficients']
            if 'paths' in path_coeffs and path_coeffs['paths']:
                for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                    coeff = path_coeffs.get('coefficients', {}).get(i, 0)
                    print(f"  {from_var} -> {to_var}: {coeff:.4f}")
        
        # 7. 결과 저장
        saved_files = export_path_results(results, filename_prefix="simple_test")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    print(f"\n테스트 {'성공' if success else '실패'}!")
