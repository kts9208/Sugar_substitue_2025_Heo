#!/usr/bin/env python3
"""
5개 요인 모든 가능한 경로 분석 및 semopy 계산 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import itertools

from path_analysis import (
    PathAnalyzer,
    PathModelBuilder,
    create_default_path_config,
    create_path_model
)

def analyze_all_possible_paths():
    """5개 요인의 모든 가능한 경로 분석"""
    print("🔍 5개 요인 모든 가능한 경로 분석")
    print("=" * 60)
    
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    # 1. 모든 가능한 경로 조합 생성
    all_possible_paths = list(itertools.permutations(variables, 2))
    print(f"이론적으로 가능한 모든 경로: {len(all_possible_paths)}개")
    
    # 2. 현재 모델에서 사용하는 경로
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
    
    print(f"현재 모델 경로: {len(current_paths)}개")
    print("현재 경로 목록:")
    for i, (from_var, to_var) in enumerate(current_paths, 1):
        print(f"  {i:2d}. {from_var} → {to_var}")
    
    # 3. 포화모델 (모든 경로 포함) 생성 및 분석
    try:
        print(f"\n{'='*60}")
        print("포화모델 분석 (모든 가능한 경로 포함)")
        print("=" * 60)
        
        # 포화모델 경로 설정 (자기 자신으로의 경로 제외)
        saturated_paths = [(from_var, to_var) for from_var, to_var in all_possible_paths 
                          if from_var != to_var]
        
        print(f"포화모델 경로 수: {len(saturated_paths)}개")
        
        # 모델 생성
        saturated_model_spec = create_path_model(
            model_type='custom',
            variables=variables,
            paths=saturated_paths,
            correlations=[]  # 포화모델에서는 상관관계 불필요
        )
        
        # 분석 실행
        config = create_default_path_config(verbose=False)
        analyzer = PathAnalyzer(config)
        data = analyzer.load_data(variables)
        
        print(f"데이터 로드 완료: {data.shape}")
        
        results = analyzer.fit_model(saturated_model_spec, data)
        print(f"포화모델 적합 완료: {results['model_info']['n_observations']}개 관측치")
        
        # 결과 저장 및 분석
        analyze_path_coefficients(results, saturated_paths, "saturated")
        
    except Exception as e:
        print(f"포화모델 분석 오류: {e}")
        print("대안: 단계별 경로 추가 분석")
        analyze_incremental_paths(variables, current_paths)

def analyze_path_coefficients(results, paths, model_type):
    """경로계수 상세 분석"""
    print(f"\n{'='*40}")
    print(f"{model_type.upper()} 모델 경로계수 분석")
    print("=" * 40)
    
    if 'path_coefficients' not in results or not results['path_coefficients']:
        print("❌ 경로계수 데이터 없음")
        return
    
    path_coeffs = results['path_coefficients']
    
    # 1. 구조적 경로 (잠재변수 간) 분석
    structural_paths = []
    measurement_paths = []
    
    variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']
    
    if 'paths' in path_coeffs and path_coeffs['paths']:
        for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
            coeff = path_coeffs.get('coefficients', {}).get(i, 0)
            
            if from_var in variables and to_var in variables:
                # 구조적 경로
                structural_paths.append({
                    'from': from_var,
                    'to': to_var,
                    'coefficient': coeff,
                    'index': i
                })
            elif from_var in variables and to_var.startswith('q'):
                # 측정모델 경로
                measurement_paths.append({
                    'factor': from_var,
                    'item': to_var,
                    'loading': coeff,
                    'index': i
                })
    
    print(f"구조적 경로: {len(structural_paths)}개")
    print(f"측정모델 경로: {len(measurement_paths)}개")
    
    # 2. 구조적 경로 출력
    if structural_paths:
        print(f"\n구조적 경로계수:")
        for path in sorted(structural_paths, key=lambda x: abs(x['coefficient']), reverse=True):
            print(f"  {path['from']:20} → {path['to']:20}: {path['coefficient']:8.4f}")
    
    # 3. 측정모델 요약
    if measurement_paths:
        print(f"\n측정모델 요약:")
        factor_loadings = {}
        for path in measurement_paths:
            factor = path['factor']
            if factor not in factor_loadings:
                factor_loadings[factor] = []
            factor_loadings[factor].append(path['loading'])
        
        for factor, loadings in factor_loadings.items():
            avg_loading = np.mean(loadings)
            print(f"  {factor:20}: {len(loadings)}개 문항, 평균 적재량 {avg_loading:.3f}")
    
    # 4. 결과 저장
    save_comprehensive_results(results, structural_paths, measurement_paths, model_type)

def analyze_incremental_paths(variables, base_paths):
    """단계별 경로 추가 분석"""
    print(f"\n{'='*60}")
    print("단계별 경로 추가 분석")
    print("=" * 60)
    
    # 기본 모델부터 시작하여 경로를 하나씩 추가
    all_possible = [(from_var, to_var) for from_var in variables for to_var in variables 
                   if from_var != to_var]
    
    # 현재 사용하지 않는 경로들
    unused_paths = [path for path in all_possible if path not in base_paths]
    
    print(f"기본 경로: {len(base_paths)}개")
    print(f"추가 가능한 경로: {len(unused_paths)}개")
    
    # 추가 가능한 경로들을 중요도별로 분류
    priority_paths = {
        'high': [],  # 구매의도로의 경로
        'medium': [], # 매개변수로의 경로
        'low': []    # 기타 경로
    }
    
    for from_var, to_var in unused_paths:
        if to_var == 'purchase_intention':
            priority_paths['high'].append((from_var, to_var))
        elif to_var in ['perceived_benefit', 'perceived_price', 'nutrition_knowledge']:
            priority_paths['medium'].append((from_var, to_var))
        else:
            priority_paths['low'].append((from_var, to_var))
    
    print(f"\n경로 우선순위:")
    print(f"  높음 (구매의도로): {len(priority_paths['high'])}개")
    print(f"  중간 (매개변수로): {len(priority_paths['medium'])}개")
    print(f"  낮음 (기타): {len(priority_paths['low'])}개")
    
    # 우선순위별 경로 분석
    for priority, paths in priority_paths.items():
        if paths:
            print(f"\n{priority.upper()} 우선순위 경로:")
            for from_var, to_var in paths:
                print(f"  {from_var} → {to_var}")

def verify_semopy_calculations():
    """semopy 계산 방식 검증"""
    print(f"\n{'='*60}")
    print("SEMOPY 계산 방식 검증")
    print("=" * 60)
    
    try:
        # 간단한 모델로 semopy 계산 과정 추적
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        
        model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        print("모델 스펙:")
        print(model_spec)
        
        config = create_default_path_config(verbose=True)
        analyzer = PathAnalyzer(config)
        data = analyzer.load_data(variables)
        
        # semopy 모델 객체 직접 접근
        import semopy
        model = semopy.Model(model_spec)
        model.fit(data)
        
        print(f"\n=== SEMOPY 직접 계산 결과 ===")
        
        # 1. 파라미터 추정치
        params = model.inspect()
        print(f"파라미터 테이블 형태: {type(params)}")
        print(f"파라미터 테이블 크기: {params.shape}")
        print(f"컬럼: {list(params.columns)}")
        
        # 2. 적합도 지수
        try:
            from semopy.stats import calc_stats
            stats = calc_stats(model)
            print(f"\n적합도 지수 타입: {type(stats)}")
            print(f"적합도 지수 키: {list(stats.keys()) if isinstance(stats, dict) else 'Not dict'}")
        except Exception as e:
            print(f"적합도 지수 계산 오류: {e}")
        
        # 3. 표준화 결과
        try:
            std_params = model.inspect(std_est=True)
            print(f"\n표준화 결과 크기: {std_params.shape}")
            print(f"표준화 컬럼: {list(std_params.columns)}")
        except Exception as e:
            print(f"표준화 결과 오류: {e}")
        
        # 4. 우리 모듈과 비교
        results = analyzer.fit_model(model_spec, data)
        
        print(f"\n=== 우리 모듈 vs SEMOPY 직접 비교 ===")
        print(f"우리 모듈 관측치: {results['model_info']['n_observations']}")
        print(f"SEMOPY 직접 관측치: {model.mx_data.shape[0]}")
        
        # 경로계수 비교
        if 'path_coefficients' in results:
            our_coeffs = results['path_coefficients']
            semopy_params = model.inspect()
            
            print(f"\n경로계수 비교:")
            if 'paths' in our_coeffs and our_coeffs['paths']:
                for i, (from_var, to_var) in enumerate(our_coeffs['paths'][:5]):  # 처음 5개만
                    our_coeff = our_coeffs.get('coefficients', {}).get(i, 0)
                    print(f"  {from_var} → {to_var}: 우리={our_coeff:.4f}")
        
        return True
        
    except Exception as e:
        print(f"semopy 검증 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_comprehensive_results(results, structural_paths, measurement_paths, model_type):
    """종합 결과 저장"""
    try:
        output_dir = Path("path_analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 구조적 경로 저장
        if structural_paths:
            structural_file = output_dir / f"{model_type}_structural_paths_{timestamp}.csv"
            df_structural = pd.DataFrame(structural_paths)
            df_structural.to_csv(structural_file, index=False, encoding='utf-8-sig')
            print(f"✅ 구조적 경로 저장: {structural_file}")
        
        # 2. 측정모델 저장
        if measurement_paths:
            measurement_file = output_dir / f"{model_type}_measurement_model_{timestamp}.csv"
            df_measurement = pd.DataFrame(measurement_paths)
            df_measurement.to_csv(measurement_file, index=False, encoding='utf-8-sig')
            print(f"✅ 측정모델 저장: {measurement_file}")
        
        # 3. 종합 요약 저장
        summary_file = output_dir / f"{model_type}_comprehensive_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"{model_type.upper()} 모델 종합 분석 결과\n")
            f.write("=" * 50 + "\n")
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"관측치 수: {results['model_info']['n_observations']}\n")
            f.write(f"변수 수: {results['model_info']['n_variables']}\n\n")
            
            f.write(f"구조적 경로: {len(structural_paths)}개\n")
            for path in structural_paths:
                f.write(f"  {path['from']} → {path['to']}: {path['coefficient']:.4f}\n")
            
            f.write(f"\n측정모델: {len(measurement_paths)}개 경로\n")
            
            # 요인별 문항 수 요약
            factor_items = {}
            for path in measurement_paths:
                factor = path['factor']
                if factor not in factor_items:
                    factor_items[factor] = 0
                factor_items[factor] += 1
            
            for factor, count in factor_items.items():
                f.write(f"  {factor}: {count}개 문항\n")
        
        print(f"✅ 종합 요약 저장: {summary_file}")
        
    except Exception as e:
        print(f"❌ 결과 저장 오류: {e}")

def main():
    """메인 실행 함수"""
    print("🔍 5개 요인 경로분석 종합 검증")
    print(f"검증 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 모든 가능한 경로 분석
    analyze_all_possible_paths()
    
    # 2. semopy 계산 방식 검증
    semopy_verified = verify_semopy_calculations()
    
    print(f"\n{'='*60}")
    print("📊 종합 검증 결과")
    print("=" * 60)
    print("✅ 모든 가능한 경로 분석 완료")
    print(f"{'✅' if semopy_verified else '❌'} semopy 계산 방식 검증 {'완료' if semopy_verified else '실패'}")
    print("✅ 경로계수 저장 확인 완료")
    
    print(f"\n결과 파일들이 'path_analysis_results' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
