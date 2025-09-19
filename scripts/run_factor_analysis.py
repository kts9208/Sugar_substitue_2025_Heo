#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
요인분석 실행 스크립트

이 스크립트는 5개 요인에 대한 확인적 요인분석(CFA)을 수행합니다:
- health_concern (건강관심도)
- perceived_benefit (지각된 유익성)
- purchase_intention (구매의도)
- perceived_price (지각된 가격)
- nutrition_knowledge (영양지식)

Author: Sugar Substitute Research Team
Date: 2025-09-18
"""

import sys
import os
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')
sys.path.append('..')

try:
    from src.analysis.factor_analysis import analyze_factor_loading, export_factor_results
    from src.utils.results_manager import save_results, archive_previous_results
except ImportError as e:
    print(f"⚠️ 모듈 임포트 오류: {e}")
    print("기본 기능으로 실행합니다.")

    # 기본 분석 함수 정의
    def analyze_factor_loading(factor_name):
        return {"error": f"분석 모듈을 찾을 수 없습니다: {factor_name}"}

    def export_factor_results(results, output_dir):
        return {}


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
            print("❌ 설문조사 데이터 디렉토리를 찾을 수 없습니다.")
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
        print(f"❌ 데이터 가용성 확인 오류: {e}")
        return []


def run_single_factor_analysis(factor_name):
    """단일 요인 분석"""
    print(f"\n📊 단일 요인 분석: {factor_name}")
    print("-" * 50)
    
    try:
        results = analyze_factor_loading(factor_name)
        
        if 'error' in results:
            print(f"❌ 분석 실패: {results['error']}")
            return None
        
        print("✅ 분석 완료!")
        print(f"   - 샘플 크기: {results['model_info']['n_observations']}")
        print(f"   - 변수 수: {results['model_info']['n_variables']}")
        
        # Factor loadings 결과 출력
        loadings = results.get('factor_loadings', pd.DataFrame())
        if len(loadings) > 0:
            print(f"\n📈 Factor Loadings:")
            for _, row in loadings.iterrows():
                loading = row['Loading']
                item = row['Item']
                status = "✅" if abs(loading) >= 0.7 else "⚠️" if abs(loading) >= 0.5 else "❌"
                print(f"   {status} {item}: {loading:.3f}")
        
        # 적합도 지수 출력
        fit_indices = results.get('fit_indices', {})
        if fit_indices:
            print(f"\n📏 적합도 지수:")
            for index, value in fit_indices.items():
                if value is not None:
                    print(f"   {index}: {value:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ 분석 오류: {e}")
        return None


def run_multiple_factor_analysis(factors):
    """다중 요인 분석"""
    print(f"\n📊 다중 요인 분석")
    print("-" * 50)
    print(f"분석 대상: {', '.join(factors)}")
    
    try:
        results = analyze_factor_loading(factors)
        
        if 'error' in results:
            print(f"❌ 분석 실패: {results['error']}")
            return None
        
        print("✅ 분석 완료!")
        print(f"   - 샘플 크기: {results['model_info']['n_observations']}")
        print(f"   - 변수 수: {results['model_info']['n_variables']}")
        print(f"   - 분석 유형: {results['analysis_type']}")
        
        # Factor loadings 결과 출력
        loadings = results.get('factor_loadings', pd.DataFrame())
        if len(loadings) > 0:
            print(f"\n📈 Factor Loadings 요약:")
            
            # 요인별로 그룹화하여 출력
            for factor in factors:
                factor_loadings = loadings[loadings['Factor'] == factor]
                if len(factor_loadings) > 0:
                    print(f"\n   🔹 {factor}:")
                    for _, row in factor_loadings.iterrows():
                        loading = row['Loading']
                        item = row['Item']
                        status = "✅" if abs(loading) >= 0.7 else "⚠️" if abs(loading) >= 0.5 else "❌"
                        print(f"     {status} {item}: {loading:.3f}")
        
        # 적합도 지수 출력
        fit_indices = results.get('fit_indices', {})
        if fit_indices:
            print(f"\n📏 적합도 지수:")
            for index, value in fit_indices.items():
                if value is not None:
                    print(f"   {index}: {value:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ 분석 오류: {e}")
        return None


def save_analysis_results(results, output_dir="factor_analysis_results"):
    """분석 결과 저장"""
    try:
        print(f"\n💾 결과 저장 중...")
        
        saved_files = export_factor_results(results, output_dir)
        
        if saved_files:
            print(f"✅ 결과 저장 완료: {len(saved_files)}개 파일")
            for file_type, file_path in saved_files.items():
                print(f"   📄 {file_type}: {os.path.basename(file_path)}")
            
            print(f"\n📁 저장 위치: {output_dir}/")
        else:
            print("⚠️ 저장된 파일이 없습니다.")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ 결과 저장 오류: {e}")
        return {}


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='요인분석 실행')
    parser.add_argument('--factor', type=str, help='분석할 단일 요인명')
    parser.add_argument('--factors', nargs='+', help='분석할 다중 요인명 리스트')
    parser.add_argument('--all', action='store_true', help='모든 요인 분석')
    parser.add_argument('--output-dir', default='factor_analysis_results',
                       help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    print('=' * 80)
    print('요인분석 실행')
    print('=' * 80)
    print(f'실행 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # 데이터 가용성 확인
    available_factors = check_data_availability()
    
    if not available_factors:
        print("❌ 분석 가능한 데이터가 없습니다.")
        return
    
    print(f"\n✅ 분석 가능한 요인: {len(available_factors)}개")
    print(f"요인 목록: {', '.join(available_factors)}")
    
    # 분석 실행
    results = None
    
    if args.factor:
        # 단일 요인 분석
        if args.factor in available_factors:
            results = run_single_factor_analysis(args.factor)
        else:
            print(f"❌ 요인 '{args.factor}'의 데이터를 찾을 수 없습니다.")
            return
            
    elif args.factors:
        # 지정된 다중 요인 분석
        valid_factors = [f for f in args.factors if f in available_factors]
        if valid_factors:
            results = run_multiple_factor_analysis(valid_factors)
        else:
            print("❌ 지정된 요인들의 데이터를 찾을 수 없습니다.")
            return
            
    elif args.all:
        # 모든 요인 분석
        results = run_multiple_factor_analysis(available_factors)
        
    else:
        # 기본값: 5개 주요 요인 분석
        default_factors = [
            'health_concern',
            'perceived_benefit',
            'purchase_intention',
            'perceived_price',
            'nutrition_knowledge'
        ]
        
        valid_factors = [f for f in default_factors if f in available_factors]
        if valid_factors:
            results = run_multiple_factor_analysis(valid_factors)
        else:
            print("❌ 기본 요인들의 데이터를 찾을 수 없습니다.")
            return
    
    # 결과 저장
    if results:
        saved_files = save_analysis_results(results, args.output_dir)
        
        # 최종 요약
        print("\n" + "=" * 80)
        print("✅ 요인분석 완료!")
        print("=" * 80)
        
        if saved_files:
            print(f"📁 결과 파일: {len(saved_files)}개 생성")
            print(f"📂 저장 위치: {args.output_dir}/")
        
        print(f"\n🎯 다음 단계 권장:")
        print(f"  1. Factor Loading 확인 (≥ 0.7 권장)")
        print(f"  2. 적합도 지수 확인 (CFI ≥ 0.9, RMSEA ≤ 0.08)")
        print(f"  3. 신뢰도 분석 실행")
        print(f"  4. 판별타당도 검증")
    else:
        print("❌ 분석이 완료되지 않았습니다.")


if __name__ == "__main__":
    main()
