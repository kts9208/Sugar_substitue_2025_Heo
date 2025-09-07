#!/usr/bin/env python3
"""
완전한 경로분석 실행 스크립트

누락된 경로 없이 모든 가능한 경로를 포함한 경로분석을 수행합니다.
- 포화 모델 (모든 가능한 경로)
- 포괄적 모델 (이론적으로 타당한 경로)
- 기존 모델과의 비교
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# 경로분석 모듈 임포트
from path_analysis import (
    PathAnalyzer,
    analyze_path_model,
    create_path_model,
    export_path_results,
    create_default_path_config,
    create_saturated_model,
    create_comprehensive_model
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_availability():
    """데이터 가용성 확인"""
    print("=" * 60)
    print("데이터 가용성 확인")
    print("=" * 60)
    
    data_dir = Path("processed_data/survey_data")
    factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 
               'perceived_price', 'nutrition_knowledge']
    
    available_factors = []
    for factor in factors:
        file_path = data_dir / f"{factor}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            items = [col for col in df.columns if col.startswith('q')]
            print(f"✅ {factor}: {len(df)}행, {len(items)}개 문항")
            available_factors.append(factor)
        else:
            print(f"❌ {factor}: 파일 없음")
    
    return available_factors


def run_saturated_model_analysis(variables):
    """포화 모델 분석 (모든 가능한 경로 포함)"""
    print("\n" + "=" * 60)
    print("1. 포화 모델 분석 (모든 가능한 경로)")
    print("=" * 60)
    
    try:
        # 1. 포화 모델 스펙 생성
        model_spec = create_path_model(
            model_type='saturated',
            variables=variables
        )
        
        print(f"생성된 포화 모델:")
        print(f"- 변수 수: {len(variables)}")
        print(f"- 예상 경로 수: {len(variables) * (len(variables) - 1)}")
        print(f"- 변수: {', '.join(variables)}")
        
        # 2. 분석 실행
        config = create_default_path_config(verbose=True)
        results = analyze_path_model(model_spec, variables, config)
        
        # 3. 결과 출력
        print("\n=== 분석 결과 ===")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        print(f"변수 수: {results['model_info']['n_variables']}")
        
        if 'fit_indices' in results and results['fit_indices']:
            print("\n적합도 지수:")
            for index, value in results['fit_indices'].items():
                try:
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        print(f"  {index}: {value:.4f}")
                except (TypeError, ValueError):
                    print(f"  {index}: {value}")
        
        # 4. 경로 분석 결과
        if 'path_analysis' in results:
            path_info = results['path_analysis']
            print(f"\n=== 경로 분석 ===")
            print(f"잠재변수 수: {path_info['n_latent_variables']}")
            print(f"가능한 경로 수: {path_info['n_possible_paths']}")
            print(f"현재 경로 수: {path_info['n_current_paths']}")
            print(f"누락된 경로 수: {path_info['n_missing_paths']}")
            print(f"경로 포함률: {path_info['coverage_ratio']:.1%}")
            
            if path_info['missing_paths']:
                print(f"\n누락된 경로들:")
                for i, (from_var, to_var) in enumerate(path_info['missing_paths'], 1):
                    print(f"  {i}. {from_var} → {to_var}")
        
        # 5. 결과 저장
        saved_files = export_path_results(results, filename_prefix="saturated_model")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
        return results
        
    except Exception as e:
        print(f"❌ 포화 모델 분석 실패: {e}")
        logger.error(f"포화 모델 분석 오류: {e}")
        return None


def run_comprehensive_model_analysis(variables):
    """포괄적 모델 분석 (이론적으로 타당한 모든 경로)"""
    print("\n" + "=" * 60)
    print("2. 포괄적 모델 분석 (이론적으로 타당한 경로)")
    print("=" * 60)
    
    try:
        # 1. 포괄적 모델 스펙 생성
        model_spec = create_path_model(
            model_type='comprehensive',
            variables=variables,
            include_bidirectional=True,
            include_feedback=True
        )
        
        print(f"생성된 포괄적 모델:")
        print(f"- 변수 수: {len(variables)}")
        print(f"- 변수: {', '.join(variables)}")
        print(f"- 양방향 경로 포함: 예")
        print(f"- 피드백 경로 포함: 예")
        
        # 2. 분석 실행
        config = create_default_path_config(verbose=True)
        results = analyze_path_model(model_spec, variables, config)
        
        # 3. 결과 출력
        print("\n=== 분석 결과 ===")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        print(f"변수 수: {results['model_info']['n_variables']}")
        
        if 'fit_indices' in results and results['fit_indices']:
            print("\n적합도 지수:")
            for index, value in results['fit_indices'].items():
                try:
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        print(f"  {index}: {value:.4f}")
                except (TypeError, ValueError):
                    print(f"  {index}: {value}")
        
        # 4. 경로 분석 결과
        if 'path_analysis' in results:
            path_info = results['path_analysis']
            print(f"\n=== 경로 분석 ===")
            print(f"잠재변수 수: {path_info['n_latent_variables']}")
            print(f"가능한 경로 수: {path_info['n_possible_paths']}")
            print(f"현재 경로 수: {path_info['n_current_paths']}")
            print(f"누락된 경로 수: {path_info['n_missing_paths']}")
            print(f"경로 포함률: {path_info['coverage_ratio']:.1%}")
            
            if path_info['missing_paths']:
                print(f"\n누락된 경로들:")
                for i, (from_var, to_var) in enumerate(path_info['missing_paths'], 1):
                    print(f"  {i}. {from_var} → {to_var}")
        
        # 5. 결과 저장
        saved_files = export_path_results(results, filename_prefix="comprehensive_model_complete")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
        return results
        
    except Exception as e:
        print(f"❌ 포괄적 모델 분석 실패: {e}")
        logger.error(f"포괄적 모델 분석 오류: {e}")
        return None


def compare_models(saturated_results, comprehensive_results):
    """모델 비교 분석"""
    print("\n" + "=" * 60)
    print("3. 모델 비교 분석")
    print("=" * 60)
    
    if not saturated_results or not comprehensive_results:
        print("❌ 비교할 모델 결과가 없습니다.")
        return
    
    print("모델 비교:")
    print("-" * 40)
    
    # 경로 포함률 비교
    sat_path = saturated_results.get('path_analysis', {})
    comp_path = comprehensive_results.get('path_analysis', {})
    
    print(f"포화 모델:")
    print(f"  - 경로 포함률: {sat_path.get('coverage_ratio', 0):.1%}")
    print(f"  - 누락된 경로: {sat_path.get('n_missing_paths', 'N/A')}개")
    
    print(f"포괄적 모델:")
    print(f"  - 경로 포함률: {comp_path.get('coverage_ratio', 0):.1%}")
    print(f"  - 누락된 경로: {comp_path.get('n_missing_paths', 'N/A')}개")
    
    # 적합도 지수 비교
    sat_fit = saturated_results.get('fit_indices', {})
    comp_fit = comprehensive_results.get('fit_indices', {})
    
    print(f"\n적합도 지수 비교:")
    print("-" * 40)
    fit_indices = ['CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']
    
    for index in fit_indices:
        sat_val = sat_fit.get(index, np.nan)
        comp_val = comp_fit.get(index, np.nan)
        
        if not pd.isna(sat_val) and not pd.isna(comp_val):
            print(f"{index}:")
            print(f"  포화 모델: {sat_val:.4f}")
            print(f"  포괄적 모델: {comp_val:.4f}")


def main():
    """메인 실행 함수"""
    print("🔍 완전한 경로분석 실행")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 데이터 확인
    available_factors = check_data_availability()
    
    if len(available_factors) < 3:
        print("❌ 경로분석을 위한 충분한 데이터가 없습니다. (최소 3개 요인 필요)")
        return
    
    print(f"\n✅ 분석 가능한 요인: {len(available_factors)}개")
    print(f"요인 목록: {', '.join(available_factors)}")
    
    # 2. 포화 모델 분석
    saturated_results = run_saturated_model_analysis(available_factors)
    
    # 3. 포괄적 모델 분석
    comprehensive_results = run_comprehensive_model_analysis(available_factors)
    
    # 4. 모델 비교
    compare_models(saturated_results, comprehensive_results)
    
    print(f"\n🎉 완전한 경로분석 완료! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 60)


if __name__ == "__main__":
    main()
