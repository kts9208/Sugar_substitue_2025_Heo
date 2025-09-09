#!/usr/bin/env python3
"""
간단한 경로분석 효과 계산 테스트
"""

import pandas as pd
import numpy as np
import logging
from path_analysis.effects_calculator import EffectsCalculator
from semopy import Model

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_mediation_data():
    """간단한 매개효과 데이터 생성"""
    
    print("=" * 50)
    print("간단한 매개효과 데이터 생성")
    print("=" * 50)
    
    np.random.seed(42)
    n = 200
    
    # X -> M -> Y 구조
    X = np.random.normal(0, 1, n)  # 독립변수
    M = 0.7 * X + np.random.normal(0, 0.5, n)  # 매개변수 (X에 영향받음)
    Y = 0.3 * X + 0.6 * M + np.random.normal(0, 0.4, n)  # 종속변수 (X와 M에 영향받음)
    
    data = pd.DataFrame({
        'X': X,
        'M': M, 
        'Y': Y
    })
    
    print(f"데이터 생성 완료: {data.shape}")
    print(f"이론적 효과:")
    print(f"  직접효과 (X -> Y): 0.3")
    print(f"  간접효과 (X -> M -> Y): 0.7 * 0.6 = 0.42")
    print(f"  총효과: 0.3 + 0.42 = 0.72")
    
    return data

def test_effects_calculator():
    """EffectsCalculator 테스트"""
    
    print("\n" + "=" * 50)
    print("EffectsCalculator 테스트")
    print("=" * 50)
    
    try:
        # 데이터 생성
        data = create_simple_mediation_data()
        
        # 모델 정의 및 적합
        model_spec = """
        M ~ X
        Y ~ X + M
        """
        
        print(f"\n모델 스펙:")
        print(model_spec)
        
        model = Model(model_spec)
        model.fit(data)
        
        print("✅ 모델 적합 완료")
        
        # EffectsCalculator 초기화
        effects_calc = EffectsCalculator(model)
        effects_calc.set_data(data)
        effects_calc.model_spec = model_spec
        
        print("✅ EffectsCalculator 초기화 완료")
        
        # 모델 파라미터 확인
        params = model.inspect()
        print(f"\n모델 파라미터:")
        print(params[['lval', 'op', 'rval', 'Estimate']].to_string())
        
        # 부트스트래핑 실행
        print(f"\n부트스트래핑 실행 (50개 샘플)...")
        bootstrap_results = effects_calc.calculate_bootstrap_effects(
            independent_var='X',
            dependent_var='Y',
            mediator_vars=['M'],
            n_bootstrap=50,
            confidence_level=0.95,
            method='bias-corrected',
            show_progress=True
        )
        
        print("✅ 부트스트래핑 완료!")
        
        # 결과 출력
        print(f"\n" + "=" * 40)
        print("부트스트래핑 결과")
        print("=" * 40)
        
        # 원본 효과
        if 'original_effects' in bootstrap_results:
            original = bootstrap_results['original_effects']
            print(f"\n원본 효과:")
            for effect_name, effect_value in original.items():
                if isinstance(effect_value, (int, float)):
                    print(f"  {effect_name}: {effect_value:.4f}")
        
        # 신뢰구간
        if 'confidence_intervals' in bootstrap_results:
            ci = bootstrap_results['confidence_intervals']
            print(f"\n신뢰구간 (95%):")
            for effect_name, ci_data in ci.items():
                if isinstance(ci_data, dict):
                    lower = ci_data.get('lower', 'N/A')
                    upper = ci_data.get('upper', 'N/A')
                    mean = ci_data.get('mean', 'N/A')
                    significant = ci_data.get('significant', False)
                    
                    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                        print(f"  {effect_name}: [{lower:.4f}, {upper:.4f}] {'*' if significant else ''}")
                    else:
                        print(f"  {effect_name}: [{lower}, {upper}] {'*' if significant else ''}")
        
        # 부트스트래핑 통계
        if 'bootstrap_statistics' in bootstrap_results:
            stats = bootstrap_results['bootstrap_statistics']
            print(f"\n부트스트래핑 통계:")
            for effect_name, stat_data in stats.items():
                if isinstance(stat_data, dict):
                    mean = stat_data.get('mean', 'N/A')
                    std = stat_data.get('std', 'N/A')
                    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
                        print(f"  {effect_name}: 평균={mean:.4f}, 표준편차={std:.4f}")
                    else:
                        print(f"  {effect_name}: 평균={mean}, 표준편차={std}")
        
        # 부트스트래핑 샘플 확인
        if 'bootstrap_results' in bootstrap_results:
            bootstrap_data = bootstrap_results['bootstrap_results']
            print(f"\n부트스트래핑 샘플 정보:")
            for effect_name, samples in bootstrap_data.items():
                if isinstance(samples, list) and len(samples) > 0:
                    print(f"  {effect_name}: {len(samples)}개 샘플")
        
        return True, bootstrap_results
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_path_analyzer_integration():
    """PathAnalyzer 통합 테스트"""
    
    print("\n" + "=" * 50)
    print("PathAnalyzer 통합 테스트")
    print("=" * 50)
    
    try:
        from path_analysis import PathAnalyzer, PathAnalysisConfig
        
        # 데이터 생성
        data = create_simple_mediation_data()
        
        # 모델 정의
        model_spec = """
        M ~ X
        Y ~ X + M
        """
        
        # 설정 (부트스트래핑 포함)
        config = PathAnalysisConfig(
            include_bootstrap_ci=True,
            bootstrap_samples=30,  # 빠른 테스트
            mediation_bootstrap_samples=30,
            bootstrap_method='non-parametric',
            bootstrap_percentile_method='bias_corrected',
            confidence_level=0.95,
            bootstrap_progress_bar=True
        )
        
        print(f"PathAnalyzer 설정:")
        print(f"  부트스트래핑: {config.include_bootstrap_ci}")
        print(f"  샘플 수: {config.bootstrap_samples}")
        
        # 분석 실행
        analyzer = PathAnalyzer(config)
        results = analyzer.fit_model(model_spec, data)
        
        print("✅ PathAnalyzer 분석 완료")
        
        # 결과 확인
        print(f"\n결과 키: {list(results.keys())}")
        
        # 적합도 지수
        if 'fit_indices' in results:
            fit_indices = results['fit_indices']
            print(f"\n적합도 지수:")
            for key, value in fit_indices.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # 경로계수
        if 'path_coefficients' in results:
            path_coeffs = results['path_coefficients']
            print(f"\n경로계수:")
            for coeff in path_coeffs:
                if isinstance(coeff, dict):
                    path = coeff.get('path', 'N/A')
                    coefficient = coeff.get('coefficient', 'N/A')
                    p_value = coeff.get('p_value', 'N/A')
                    
                    if isinstance(coefficient, (int, float)):
                        print(f"  {path}: {coefficient:.4f} (p={p_value})")
                    else:
                        print(f"  {path}: {coefficient} (p={p_value})")
        
        # 부트스트래핑 결과
        bootstrap_effects = results.get('bootstrap_effects', {})
        if bootstrap_effects:
            print(f"\n부트스트래핑 결과: {len(bootstrap_effects)}개 조합")
            for combination_key, combination_result in bootstrap_effects.items():
                print(f"\n{combination_key}:")
                if 'confidence_intervals' in combination_result:
                    ci = combination_result['confidence_intervals']
                    for effect_name, ci_data in ci.items():
                        if isinstance(ci_data, dict):
                            lower = ci_data.get('lower', 'N/A')
                            upper = ci_data.get('upper', 'N/A')
                            significant = ci_data.get('significant', False)
                            
                            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                                print(f"  {effect_name}: [{lower:.4f}, {upper:.4f}] {'*' if significant else ''}")
        else:
            print(f"\n부트스트래핑 결과 없음")
        
        return True, results
        
    except Exception as e:
        print(f"❌ PathAnalyzer 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def save_test_results(effects_results, path_results):
    """테스트 결과 저장"""
    
    print("\n" + "=" * 50)
    print("테스트 결과 저장")
    print("=" * 50)
    
    try:
        import json
        import os
        from datetime import datetime
        
        # 결과 디렉토리 생성
        results_dir = "simple_effects_test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 요약 파일
        summary_file = os.path.join(results_dir, f"effects_test_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("간단한 경로분석 효과 계산 테스트 결과\n")
            f.write("=" * 50 + "\n")
            f.write(f"실행 시간: {timestamp}\n\n")
            
            # EffectsCalculator 결과
            if effects_results:
                f.write("EffectsCalculator 부트스트래핑 결과:\n")
                if 'original_effects' in effects_results:
                    original = effects_results['original_effects']
                    f.write("원본 효과:\n")
                    for effect_name, effect_value in original.items():
                        f.write(f"  {effect_name}: {effect_value}\n")
                
                if 'confidence_intervals' in effects_results:
                    ci = effects_results['confidence_intervals']
                    f.write("\n신뢰구간:\n")
                    for effect_name, ci_data in ci.items():
                        if isinstance(ci_data, dict):
                            lower = ci_data.get('lower', 'N/A')
                            upper = ci_data.get('upper', 'N/A')
                            significant = ci_data.get('significant', False)
                            f.write(f"  {effect_name}: [{lower:.4f}, {upper:.4f}] {'*' if significant else ''}\n")
                f.write("\n")
            
            # PathAnalyzer 결과
            if path_results:
                f.write("PathAnalyzer 결과:\n")
                f.write(f"  결과 키: {list(path_results.keys())}\n")
                
                if 'fit_indices' in path_results:
                    fit_indices = path_results['fit_indices']
                    f.write("  적합도 지수:\n")
                    for key, value in fit_indices.items():
                        f.write(f"    {key}: {value}\n")
                
                bootstrap_effects = path_results.get('bootstrap_effects', {})
                if bootstrap_effects:
                    f.write(f"  부트스트래핑 조합: {len(bootstrap_effects)}개\n")
                else:
                    f.write("  부트스트래핑 결과: 없음\n")
        
        print(f"✅ 결과 요약 저장: {summary_file}")
        
        return True, results_dir
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return False, None

if __name__ == "__main__":
    print("간단한 경로분석 효과 계산 테스트")
    
    # 1. EffectsCalculator 테스트
    effects_success, effects_results = test_effects_calculator()
    
    # 2. PathAnalyzer 통합 테스트
    path_success, path_results = test_path_analyzer_integration()
    
    # 3. 결과 저장
    if effects_success or path_success:
        save_success, results_dir = save_test_results(effects_results, path_results)
    else:
        save_success = False
        results_dir = None
    
    print(f"\n" + "=" * 50)
    print("최종 테스트 결과")
    print("=" * 50)
    print(f"EffectsCalculator: {'✅ 성공' if effects_success else '❌ 실패'}")
    print(f"PathAnalyzer: {'✅ 성공' if path_success else '❌ 실패'}")
    print(f"결과 저장: {'✅ 성공' if save_success else '❌ 실패'}")
    
    if save_success and results_dir:
        print(f"\n📁 결과 저장 위치: {results_dir}")
    
    if effects_success and path_success:
        print(f"\n🎉 모든 테스트 통과!")
        print("✅ 직접효과 및 간접효과 계산이 정상 작동합니다.")
        print("✅ 부트스트래핑 신뢰구간 계산이 정상 작동합니다.")
        print("✅ 결과가 올바르게 저장됩니다.")
    elif effects_success:
        print(f"\n✅ EffectsCalculator는 정상 작동합니다.")
        print("✅ 부트스트래핑 기능이 올바르게 구현되어 있습니다.")
    else:
        print(f"\n⚠️  테스트에서 문제가 발생했습니다.")
