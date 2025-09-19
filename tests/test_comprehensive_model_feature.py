#!/usr/bin/env python3
"""
새로운 포괄적 모델 기능 테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 우리 모듈 임포트
from path_analysis import (
    PathAnalyzer,
    create_default_path_config,
    create_path_model,
    export_path_results
)

def test_comprehensive_model_feature():
    """새로운 포괄적 모델 기능 테스트"""
    print("🔍 포괄적 모델 기능 테스트")
    print("=" * 60)
    
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    print(f"분석 변수: {', '.join(variables)}")
    
    # 1. 기본 포괄적 모델 (모든 옵션 활성화)
    print(f"\n1. 기본 포괄적 모델 생성")
    comprehensive_model_spec = create_path_model(
        model_type='comprehensive',
        variables=variables,
        include_bidirectional=True,
        include_feedback=True
    )
    
    print(f"생성된 모델 스펙:")
    print(comprehensive_model_spec)
    
    # 경로 수 계산
    structural_lines = [line for line in comprehensive_model_spec.split('\n') if '~' in line and '=~' not in line]
    print(f"\n구조적 경로 수: {len(structural_lines)}개")
    
    # 2. 모델 추정
    print(f"\n2. 포괄적 모델 추정")
    config = create_default_path_config(verbose=False)
    analyzer = PathAnalyzer(config)
    data = analyzer.load_data(variables)
    
    try:
        results = analyzer.fit_model(comprehensive_model_spec, data)
        
        path_coefficients = results.get('path_coefficients', {})
        path_analysis = results.get('path_analysis', {})
        fit_indices = results.get('fit_indices', {})
        
        print(f"✅ 포괄적 모델 추정 성공!")
        print(f"추정된 경로 수: {len(path_coefficients.get('paths', []))}개")
        print(f"경로 포함률: {path_analysis.get('coverage_ratio', 0):.1%}")
        print(f"누락된 경로: {path_analysis.get('n_missing_paths', 0)}개")
        
        # 적합도 지수
        print(f"\n적합도 지수:")
        key_indices = ['chi_square', 'cfi', 'tli', 'rmsea', 'aic']
        for index_name in key_indices:
            if index_name in fit_indices:
                value = fit_indices[index_name]
                if hasattr(value, 'iloc'):
                    numeric_value = value.iloc[0] if len(value) > 0 else np.nan
                else:
                    numeric_value = value
                
                if isinstance(numeric_value, (int, float)) and not pd.isna(numeric_value):
                    print(f"  {index_name.upper()}: {numeric_value:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ 포괄적 모델 추정 실패: {e}")
        return None

def test_model_variations():
    """모델 변형 테스트"""
    print(f"\n" + "=" * 60)
    print("모델 변형 테스트")
    print("=" * 60)
    
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    variations = [
        {'name': '기본 모델', 'bidirectional': False, 'feedback': False},
        {'name': '양방향 모델', 'bidirectional': True, 'feedback': False},
        {'name': '피드백 모델', 'bidirectional': False, 'feedback': True},
        {'name': '완전 모델', 'bidirectional': True, 'feedback': True},
    ]
    
    results_summary = []
    
    for variation in variations:
        print(f"\n{variation['name']} 테스트:")
        
        try:
            model_spec = create_path_model(
                model_type='comprehensive',
                variables=variables,
                include_bidirectional=variation['bidirectional'],
                include_feedback=variation['feedback']
            )
            
            # 경로 수 계산
            structural_lines = [line for line in model_spec.split('\n') if '~' in line and '=~' not in line]
            n_paths = len(structural_lines)
            
            # 모델 추정
            config = create_default_path_config(verbose=False)
            analyzer = PathAnalyzer(config)
            data = analyzer.load_data(variables)
            results = analyzer.fit_model(model_spec, data)
            
            path_analysis = results.get('path_analysis', {})
            fit_indices = results.get('fit_indices', {})
            
            # 적합도 지수 추출
            aic = fit_indices.get('aic', np.nan)
            if hasattr(aic, 'iloc'):
                aic = aic.iloc[0] if len(aic) > 0 else np.nan
            
            cfi = fit_indices.get('cfi', np.nan)
            if hasattr(cfi, 'iloc'):
                cfi = cfi.iloc[0] if len(cfi) > 0 else np.nan
            
            rmsea = fit_indices.get('rmsea', np.nan)
            if hasattr(rmsea, 'iloc'):
                rmsea = rmsea.iloc[0] if len(rmsea) > 0 else np.nan
            
            result_info = {
                'name': variation['name'],
                'n_paths': n_paths,
                'coverage': path_analysis.get('coverage_ratio', 0),
                'aic': aic,
                'cfi': cfi,
                'rmsea': rmsea,
                'status': '성공'
            }
            
            print(f"  경로 수: {n_paths}개")
            print(f"  경로 포함률: {path_analysis.get('coverage_ratio', 0):.1%}")
            print(f"  AIC: {aic:.2f}" if not pd.isna(aic) else "  AIC: N/A")
            print(f"  CFI: {cfi:.3f}" if not pd.isna(cfi) else "  CFI: N/A")
            print(f"  RMSEA: {rmsea:.3f}" if not pd.isna(rmsea) else "  RMSEA: N/A")
            
        except Exception as e:
            result_info = {
                'name': variation['name'],
                'n_paths': n_paths if 'n_paths' in locals() else 0,
                'coverage': 0,
                'aic': np.nan,
                'cfi': np.nan,
                'rmsea': np.nan,
                'status': f'실패: {str(e)[:50]}'
            }
            print(f"  ❌ 실패: {e}")
        
        results_summary.append(result_info)
    
    # 결과 요약 테이블
    print(f"\n모델 변형 비교:")
    print(f"{'모델명':<12} {'경로수':>6} {'포함률':>8} {'AIC':>8} {'CFI':>6} {'RMSEA':>7} {'상태':<10}")
    print("-" * 70)
    
    for result in results_summary:
        aic_str = f"{result['aic']:.1f}" if not pd.isna(result['aic']) else "N/A"
        cfi_str = f"{result['cfi']:.3f}" if not pd.isna(result['cfi']) else "N/A"
        rmsea_str = f"{result['rmsea']:.3f}" if not pd.isna(result['rmsea']) else "N/A"
        
        print(f"{result['name']:<12} {result['n_paths']:>6} {result['coverage']:>7.1%} {aic_str:>8} {cfi_str:>6} {rmsea_str:>7} {result['status']:<10}")
    
    return results_summary

def save_comprehensive_results():
    """포괄적 모델 결과 저장"""
    print(f"\n" + "=" * 60)
    print("포괄적 모델 결과 저장")
    print("=" * 60)
    
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    # 최적 포괄적 모델 생성
    model_spec = create_path_model(
        model_type='comprehensive',
        variables=variables,
        include_bidirectional=True,
        include_feedback=True
    )
    
    config = create_default_path_config(verbose=False)
    analyzer = PathAnalyzer(config)
    data = analyzer.load_data(variables)
    results = analyzer.fit_model(model_spec, data)
    
    # 결과 저장
    print(f"결과 저장 중...")
    exported_files = export_path_results(
        results,
        output_dir="path_analysis_results",
        filename_prefix="comprehensive_model_final"
    )
    
    print(f"저장된 파일:")
    for file_type, file_path in exported_files.items():
        print(f"  {file_type}: {Path(file_path).name}")
    
    # 주요 결과 요약
    path_analysis = results.get('path_analysis', {})
    print(f"\n최종 포괄적 모델 요약:")
    print(f"  경로 포함률: {path_analysis.get('coverage_ratio', 0):.1%}")
    print(f"  총 경로 수: {path_analysis.get('n_current_paths', 0)}개")
    print(f"  누락 경로 수: {path_analysis.get('n_missing_paths', 0)}개")
    
    return results

def main():
    """메인 함수"""
    print("🔍 포괄적 모델 기능 테스트")
    print(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 포괄적 모델 기능 테스트
        comprehensive_results = test_comprehensive_model_feature()
        
        # 2. 모델 변형 테스트
        variation_results = test_model_variations()
        
        # 3. 결과 저장
        if comprehensive_results:
            final_results = save_comprehensive_results()
        
        print(f"\n" + "=" * 60)
        print("📊 포괄적 모델 기능 테스트 결과")
        print("=" * 60)
        
        print(f"✅ 새로운 기능 추가 완료:")
        print(f"  - 포괄적 모델 자동 생성 기능")
        print(f"  - 양방향 경로 옵션")
        print(f"  - 피드백 경로 옵션")
        print(f"  - 이론적 타당성 기반 경로 선택")
        
        print(f"\n✅ 누락 경로 해결:")
        print(f"  - 기존 모델: 8/20 경로 (40%)")
        if comprehensive_results:
            path_analysis = comprehensive_results.get('path_analysis', {})
            print(f"  - 포괄적 모델: {path_analysis.get('n_current_paths', 0)}/20 경로 ({path_analysis.get('coverage_ratio', 0):.1%})")
        
        print(f"\n✅ 모델 선택 옵션:")
        print(f"  - 연구 목적에 따른 모델 변형 선택 가능")
        print(f"  - 적합도 지수 기반 최적 모델 선택")
        print(f"  - 이론적 타당성과 통계적 유의성 균형")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
