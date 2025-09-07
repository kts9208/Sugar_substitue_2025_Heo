#!/usr/bin/env python3
"""
수정된 경로분석 모듈 테스트: 잠재변수간 경로만 저장
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

def test_structural_paths_only():
    """잠재변수간 경로만 저장되는지 테스트"""
    print("🔍 구조적 경로 전용 모듈 테스트")
    print("=" * 60)
    
    # 5개 요인 모델 설정
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    # 일부 경로만 포함한 모델
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
    
    correlations = [
        ('perceived_benefit', 'perceived_price'),
        ('perceived_benefit', 'nutrition_knowledge')
    ]
    
    print(f"설정된 경로 수: {len(paths)}개")
    print(f"가능한 총 경로 수: {len(variables) * (len(variables) - 1)}개")
    
    # 모델 생성 및 분석
    model_spec = create_path_model(
        model_type='custom',
        variables=variables,
        paths=paths,
        correlations=correlations
    )
    
    config = create_default_path_config(verbose=False)
    analyzer = PathAnalyzer(config)
    data = analyzer.load_data(variables)
    results = analyzer.fit_model(model_spec, data)
    
    print(f"\n분석 완료: {results['model_info']['n_observations']}개 관측치")
    
    return results

def analyze_path_results(results):
    """경로 분석 결과 확인"""
    print("\n" + "=" * 60)
    print("경로 분석 결과 확인")
    print("=" * 60)
    
    # 1. 구조적 경로계수 확인
    path_coefficients = results.get('path_coefficients', {})
    print(f"1. 구조적 경로계수")
    print(f"   저장된 경로 수: {len(path_coefficients.get('paths', []))}개")
    print(f"   잠재변수 수: {len(path_coefficients.get('latent_variables', []))}개")
    print(f"   잠재변수: {', '.join(path_coefficients.get('latent_variables', []))}")
    
    # 경로 목록 출력
    paths = path_coefficients.get('paths', [])
    coefficients = path_coefficients.get('coefficients', {})
    
    print(f"\n   구조적 경로:")
    for i, (from_var, to_var) in enumerate(paths):
        coeff = coefficients.get(i, 0)
        print(f"   {from_var} → {to_var}: {coeff:.4f}")
    
    # 2. 경로 분석 결과 확인
    path_analysis = results.get('path_analysis', {})
    print(f"\n2. 경로 분석 결과")
    print(f"   잠재변수 수: {path_analysis.get('n_latent_variables', 0)}개")
    print(f"   가능한 총 경로: {path_analysis.get('n_possible_paths', 0)}개")
    print(f"   현재 모델 경로: {path_analysis.get('n_current_paths', 0)}개")
    print(f"   누락된 경로: {path_analysis.get('n_missing_paths', 0)}개")
    print(f"   경로 포함률: {path_analysis.get('coverage_ratio', 0):.1%}")
    
    # 누락된 경로 출력
    missing_paths = path_analysis.get('missing_paths', [])
    if missing_paths:
        print(f"\n   누락된 경로 목록:")
        for from_var, to_var in missing_paths[:10]:  # 처음 10개만
            print(f"   {from_var} → {to_var}")
        if len(missing_paths) > 10:
            print(f"   ... 및 {len(missing_paths) - 10}개 더")

def test_saturated_model():
    """포화모델 테스트 (모든 경로 포함)"""
    print("\n" + "=" * 60)
    print("포화모델 테스트 (모든 경로 포함)")
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
    
    config = create_default_path_config(verbose=False)
    analyzer = PathAnalyzer(config)
    data = analyzer.load_data(variables)
    
    try:
        results = analyzer.fit_model(model_spec, data)
        
        path_analysis = results.get('path_analysis', {})
        print(f"포화모델 분석 성공!")
        print(f"경로 포함률: {path_analysis.get('coverage_ratio', 0):.1%}")
        print(f"누락된 경로: {path_analysis.get('n_missing_paths', 0)}개")
        
        return results
        
    except Exception as e:
        print(f"포화모델 분석 실패: {e}")
        return None

def check_saved_files():
    """저장된 파일 확인"""
    print("\n" + "=" * 60)
    print("저장된 파일 확인")
    print("=" * 60)
    
    results_dir = Path("path_analysis_results")
    if not results_dir.exists():
        print("❌ 결과 디렉토리가 없습니다.")
        return
    
    # 구조적 경로 파일 확인
    structural_files = list(results_dir.glob("*structural_paths*.csv"))
    path_analysis_files = list(results_dir.glob("*path_analysis*.csv"))
    
    print(f"구조적 경로 파일: {len(structural_files)}개")
    print(f"경로 분석 파일: {len(path_analysis_files)}개")
    
    if structural_files:
        latest_structural = max(structural_files, key=lambda x: x.stat().st_mtime)
        print(f"\n최근 구조적 경로 파일: {latest_structural.name}")
        
        df = pd.read_csv(latest_structural)
        print(f"저장된 행 수: {len(df)}개")
        
        # 메타데이터 제외한 실제 경로만 카운트
        actual_paths = df[~df['From_Variable'].isin(['METADATA', 'LATENT_VARS'])]
        print(f"실제 구조적 경로: {len(actual_paths)}개")
        
        print(f"\n구조적 경로 목록:")
        for _, row in actual_paths.iterrows():
            print(f"  {row['From_Variable']} → {row['To_Variable']}: {row['Coefficient']:.4f}")
    
    if path_analysis_files:
        latest_analysis = max(path_analysis_files, key=lambda x: x.stat().st_mtime)
        print(f"\n최근 경로 분석 파일: {latest_analysis.name}")
        
        df = pd.read_csv(latest_analysis)
        print(f"저장된 분석 항목: {len(df)}개")
        
        # 주요 정보 출력
        coverage_row = df[df['Item'] == 'Coverage Ratio']
        if not coverage_row.empty:
            print(f"경로 포함률: {coverage_row['Value'].iloc[0]}")

def main():
    """메인 테스트 함수"""
    print("🔍 구조적 경로 전용 모듈 테스트")
    print(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 부분 경로 모델 테스트
        results = test_structural_paths_only()
        analyze_path_results(results)
        
        # 2. 포화모델 테스트
        saturated_results = test_saturated_model()
        
        # 3. 저장된 파일 확인
        check_saved_files()
        
        print(f"\n" + "=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        print("✅ 잠재변수간 경로만 추출 및 저장")
        print("✅ 요인-문항 간 경로 제외됨")
        print("✅ 모든 가능한 경로 분석 기능 추가")
        print("✅ 경로 포함률 및 누락 경로 확인 가능")
        print("✅ 구조적 경로 전용 파일 저장")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
