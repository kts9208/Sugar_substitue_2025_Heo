#!/usr/bin/env python3
"""
통합 경로분석 실행 스크립트

이 스크립트는 다양한 경로분석 모델을 실행합니다:
1. 단순 매개모델 (Simple Mediation)
2. 다중 매개모델 (Multiple Mediation)  
3. 포괄적 구조모델 (Comprehensive Structural Model)
4. 포화 모델 (Saturated Model)

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
import sys
sys.path.append('.')
sys.path.append('..')

try:
    from src.analysis.path_analysis import (
        PathAnalyzer, analyze_path_model, create_path_model,
        export_path_results, create_default_path_config,
        create_mediation_config, create_saturated_model,
        create_comprehensive_model
    )
    from src.utils.results_manager import save_results, archive_previous_results
except ImportError:
    # Fallback to current structure
    from path_analysis import (
        PathAnalyzer, analyze_path_model, create_path_model,
        export_path_results, create_default_path_config,
        create_mediation_config, create_saturated_model,
        create_comprehensive_model
    )

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('path_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_data_availability():
    """데이터 가용성 확인"""
    try:
        # 데이터 경로 확인
        data_paths = [
            "data/processed/survey",
            "processed_data/survey_data"  # Fallback
        ]
        
        available_path = None
        for path in data_paths:
            if Path(path).exists():
                available_path = Path(path)
                break
        
        if not available_path:
            logger.error("설문조사 데이터 디렉토리를 찾을 수 없습니다.")
            return []
        
        # 요인별 데이터 파일 확인
        factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 
                  'perceived_price', 'nutrition_knowledge']
        
        available_factors = []
        for factor in factors:
            factor_file = available_path / f"{factor}.csv"
            if factor_file.exists():
                available_factors.append(factor)
        
        return available_factors
        
    except Exception as e:
        logger.error(f"데이터 가용성 확인 오류: {e}")
        return []


def run_simple_mediation_analysis():
    """단순 매개모델 분석"""
    print("\n" + "=" * 60)
    print("1. 단순 매개모델 분석")
    print("건강관심도 → 지각된유익성 → 구매의도")
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
        print(model_spec)
        
        # 2. 분석 실행
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        config = create_mediation_config(verbose=True)
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 3. 결과 출력
        print("\n=== 분석 결과 ===")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        
        if 'fit_indices' in results and results['fit_indices']:
            print("\n적합도 지수:")
            for index, value in results['fit_indices'].items():
                try:
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        print(f"  {index}: {value:.4f}")
                except (TypeError, ValueError):
                    print(f"  {index}: {value}")
        
        # 4. 결과 저장
        saved_files = export_path_results(results, filename_prefix="simple_mediation")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        logger.error(f"단순 매개모델 분석 오류: {e}")
        return None


def run_multiple_mediation_analysis():
    """다중 매개모델 분석"""
    print("\n" + "=" * 60)
    print("2. 다중 매개모델 분석")
    print("건강관심도 → [지각된유익성, 영양지식] → 구매의도")
    print("=" * 60)
    
    try:
        # 1. 모델 스펙 생성
        model_spec = create_path_model(
            model_type='multiple_mediation',
            independent_var='health_concern',
            mediator_vars=['perceived_benefit', 'nutrition_knowledge'],
            dependent_var='purchase_intention'
        )
        
        print("다중 매개모델 스펙 생성 완료")
        
        # 2. 분석 실행
        variables = ['health_concern', 'perceived_benefit', 'nutrition_knowledge', 'purchase_intention']
        config = create_default_path_config(
            standardized=True,
            calculate_effects=True,
            include_bootstrap_ci=True,
            verbose=True
        )
        
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
        
        # 4. 결과 저장
        saved_files = export_path_results(results, filename_prefix="multiple_mediation")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        logger.error(f"다중 매개모델 분석 오류: {e}")
        return None


def run_comprehensive_structural_model():
    """포괄적 구조모델 분석"""
    print("\n" + "=" * 60)
    print("3. 포괄적 구조모델 분석")
    print("5개 요인 간 이론적으로 타당한 모든 경로 포함")
    print("=" * 60)
    
    try:
        # 1. 5개 요인 모두 포함
        variables = ['health_concern', 'perceived_benefit', 'perceived_price',
                    'nutrition_knowledge', 'purchase_intention']

        print(f"분석 변수: {', '.join(variables)}")

        # 2. 포괄적 구조모델 생성
        model_spec = create_path_model(
            model_type='comprehensive',
            variables=variables,
            include_bidirectional=True,
            include_feedback=True
        )
        
        print("종합적인 구조모델 스펙 생성 완료")
        
        # 3. 분석 실행
        config = create_default_path_config(
            standardized=True,
            create_diagrams=True,
            calculate_effects=True,
            include_bootstrap_ci=True,
            verbose=True
        )
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 4. 결과 출력
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
        
        # 5. 경로 분석 결과
        if 'path_analysis' in results:
            path_info = results['path_analysis']
            print(f"\n=== 경로 분석 ===")
            print(f"잠재변수 수: {path_info['n_latent_variables']}")
            print(f"가능한 경로 수: {path_info['n_possible_paths']}")
            print(f"현재 경로 수: {path_info['n_current_paths']}")
            print(f"경로 포함률: {path_info['coverage_ratio']:.1%}")
        
        # 6. 결과 저장
        saved_files = export_path_results(results, filename_prefix="comprehensive_structural")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        logger.error(f"종합적인 구조모델 분석 오류: {e}")
        return None


def run_saturated_model_analysis():
    """포화 모델 분석"""
    print("\n" + "=" * 60)
    print("4. 포화 모델 분석")
    print("5개 요인 간 모든 가능한 경로 포함")
    print("=" * 60)

    try:
        # 1. 5개 요인 모두 포함
        variables = ['health_concern', 'perceived_benefit', 'perceived_price',
                    'nutrition_knowledge', 'purchase_intention']

        print(f"분석 변수: {', '.join(variables)}")
        print(f"예상 경로 수: {len(variables) * (len(variables) - 1)} (모든 가능한 경로)")

        # 2. 포화 모델 생성
        model_spec = create_path_model(
            model_type='saturated',
            variables=variables
        )

        print("포화 모델 스펙 생성 완료")

        # 3. 분석 실행
        config = create_default_path_config(
            standardized=True,
            create_diagrams=True,
            verbose=True
        )
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 4. 결과 출력
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
        
        # 5. 결과 저장
        saved_files = export_path_results(results, filename_prefix="saturated_model")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        logger.error(f"포화 모델 분석 오류: {e}")
        return None


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='통합 경로분석 실행')
    parser.add_argument('--model', choices=['simple', 'multiple', 'comprehensive', 'saturated', 'all'],
                       default='all', help='실행할 모델 타입')
    parser.add_argument('--output-dir', default='path_analysis_results',
                       help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    print("🔍 통합 경로분석 실행")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"모델 타입: {args.model}")
    
    # 1. 데이터 확인
    available_factors = check_data_availability()
    
    if len(available_factors) < 3:
        print("❌ 경로분석을 위한 충분한 데이터가 없습니다. (최소 3개 요인 필요)")
        return
    
    print(f"\n✅ 분석 가능한 요인: {len(available_factors)}개")
    print(f"요인 목록: {', '.join(available_factors)}")
    
    # 2. 분석 실행
    results = {}
    
    if args.model in ['simple', 'all']:
        simple_results = run_simple_mediation_analysis()
        if simple_results:
            results['simple_mediation'] = simple_results
    
    if args.model in ['multiple', 'all']:
        multiple_results = run_multiple_mediation_analysis()
        if multiple_results:
            results['multiple_mediation'] = multiple_results
    
    if args.model in ['comprehensive', 'all']:
        comprehensive_results = run_comprehensive_structural_model()
        if comprehensive_results:
            results['comprehensive_structural'] = comprehensive_results
    
    if args.model in ['saturated', 'all']:
        saturated_results = run_saturated_model_analysis()
        if saturated_results:
            results['saturated_model'] = saturated_results
    
    # 3. 최종 요약
    print("\n" + "=" * 60)
    print("✅ 경로분석 완료!")
    print("=" * 60)
    print(f"실행된 모델: {len(results)}개")
    for model_name in results.keys():
        print(f"  - {model_name}")
    
    print(f"\n📁 결과 파일 위치: {args.output_dir}/")
    print(f"\n🎯 다음 단계 권장:")
    print(f"  1. 적합도 지수 확인 (CFI ≥ 0.9, RMSEA ≤ 0.08)")
    print(f"  2. 경로계수 유의성 확인 (p < 0.05)")
    print(f"  3. 매개효과 부트스트래핑 결과 확인")


if __name__ == "__main__":
    main()
