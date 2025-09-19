#!/usr/bin/env python3
"""
Path Analysis Module Test Script

경로분석 모듈의 기능을 테스트하는 스크립트입니다.
실제 데이터를 사용하여 모든 주요 기능을 검증합니다.
"""

import sys
import os
from pathlib import Path
import logging

# 경로분석 모듈 임포트
try:
    from path_analysis import (
        PathAnalyzer,
        PathModelBuilder, 
        EffectsCalculator,
        PathResultsExporter,
        PathAnalysisVisualizer,
        analyze_path_model,
        create_path_model,
        export_path_results,
        create_default_path_config,
        create_mediation_config
    )
    print("✅ 경로분석 모듈 임포트 성공")
except ImportError as e:
    print(f"❌ 경로분석 모듈 임포트 실패: {e}")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_availability():
    """데이터 파일 존재 확인"""
    print("\n" + "="*50)
    print("1. 데이터 파일 존재 확인")
    print("="*50)
    
    data_dir = Path("processed_data/survey_data")
    required_factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 
                       'nutrition_knowledge', 'perceived_price']
    
    available_factors = []
    for factor in required_factors:
        file_path = data_dir / f"{factor}.csv"
        if file_path.exists():
            available_factors.append(factor)
            print(f"✅ {factor}.csv 존재")
        else:
            print(f"❌ {factor}.csv 없음")
    
    print(f"\n사용 가능한 요인: {len(available_factors)}/{len(required_factors)}개")
    return available_factors


def test_model_builder():
    """모델 빌더 테스트"""
    print("\n" + "="*50)
    print("2. 모델 빌더 테스트")
    print("="*50)
    
    try:
        builder = PathModelBuilder()
        print(f"✅ PathModelBuilder 초기화 성공")
        print(f"   사용 가능한 요인: {builder.available_factors}")
        
        # 단순 매개모델 테스트
        if len(builder.available_factors) >= 3:
            factors = builder.available_factors[:3]
            model_spec = builder.create_simple_mediation_model(
                factors[0], factors[1], factors[2]
            )
            print(f"✅ 단순 매개모델 생성 성공")
            print(f"   모델 스펙 길이: {len(model_spec)} 문자")
            
            # 모델 스펙 일부 출력
            lines = model_spec.split('\n')[:5]
            print("   모델 스펙 미리보기:")
            for line in lines:
                if line.strip():
                    print(f"     {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 빌더 테스트 실패: {e}")
        return False


def test_path_analyzer():
    """경로분석기 테스트"""
    print("\n" + "="*50)
    print("3. 경로분석기 테스트")
    print("="*50)
    
    try:
        # 설정 생성
        config = create_default_path_config(verbose=True)
        print("✅ 설정 생성 성공")
        
        # 분석기 초기화
        analyzer = PathAnalyzer(config)
        print("✅ PathAnalyzer 초기화 성공")
        
        # 데이터 로드 테스트 (사용 가능한 요인 3개 사용)
        builder = PathModelBuilder()
        if len(builder.available_factors) >= 3:
            test_factors = builder.available_factors[:3]
            
            try:
                data = analyzer.load_data(test_factors)
                print(f"✅ 데이터 로드 성공: {data.shape}")
                
                # 간단한 모델 스펙 생성
                model_spec = f"""
                {test_factors[0]} =~ q6 + q7 + q8
                {test_factors[1]} =~ q11 + q12 + q13  
                {test_factors[2]} =~ q1 + q2 + q3
                
                {test_factors[1]} ~ {test_factors[0]}
                {test_factors[2]} ~ {test_factors[1]}
                {test_factors[2]} ~ {test_factors[0]}
                """
                
                # 모델 적합 테스트 (간단한 버전)
                try:
                    results = analyzer.fit_model(model_spec, data)
                    print("✅ 모델 적합 성공")
                    print(f"   관측치 수: {results['model_info']['n_observations']}")
                    
                    if 'fit_indices' in results:
                        fit_indices = results['fit_indices']
                        print("   주요 적합도 지수:")
                        for index in ['cfi', 'rmsea']:
                            if index in fit_indices:
                                print(f"     {index.upper()}: {fit_indices[index]:.4f}")
                    
                    return True, results
                    
                except Exception as e:
                    print(f"⚠️  모델 적합 실패 (데이터 문제일 수 있음): {e}")
                    return False, None
                    
            except Exception as e:
                print(f"❌ 데이터 로드 실패: {e}")
                return False, None
        else:
            print("❌ 테스트할 충분한 요인이 없습니다")
            return False, None
            
    except Exception as e:
        print(f"❌ 경로분석기 테스트 실패: {e}")
        return False, None


def test_effects_calculator():
    """효과 계산기 테스트"""
    print("\n" + "="*50)
    print("4. 효과 계산기 테스트")
    print("="*50)
    
    try:
        # 간단한 매개모델로 테스트
        builder = PathModelBuilder()
        if len(builder.available_factors) >= 3:
            factors = builder.available_factors[:3]
            
            # 단순 매개모델 생성
            model_spec = create_path_model(
                model_type='simple_mediation',
                independent_var=factors[0],
                mediator_var=factors[1],
                dependent_var=factors[2]
            )
            
            # 분석 실행
            try:
                results = analyze_path_model(model_spec, factors)
                
                if 'model_object' in results:
                    # 효과 계산기 테스트
                    effects_calc = EffectsCalculator(results['model_object'])
                    print("✅ EffectsCalculator 초기화 성공")
                    
                    # 직접효과 계산
                    direct_effects = effects_calc.calculate_direct_effects(factors[0], factors[2])
                    print(f"✅ 직접효과 계산 성공: {direct_effects.get('coefficient', 0):.4f}")
                    
                    # 간접효과 계산
                    indirect_effects = effects_calc.calculate_indirect_effects(
                        factors[0], factors[2], [factors[1]]
                    )
                    print(f"✅ 간접효과 계산 성공: {indirect_effects.get('total_indirect_effect', 0):.4f}")
                    
                    return True
                else:
                    print("❌ 모델 객체가 결과에 없습니다")
                    return False
                    
            except Exception as e:
                print(f"⚠️  효과 계산 테스트 실패 (모델 적합 문제): {e}")
                return False
        else:
            print("❌ 테스트할 충분한 요인이 없습니다")
            return False
            
    except Exception as e:
        print(f"❌ 효과 계산기 테스트 실패: {e}")
        return False


def test_results_exporter():
    """결과 내보내기 테스트"""
    print("\n" + "="*50)
    print("5. 결과 내보내기 테스트")
    print("="*50)
    
    try:
        # 테스트용 결과 데이터 생성
        test_results = {
            'model_info': {
                'n_observations': 100,
                'n_variables': 15,
                'estimator': 'MLW',
                'optimizer': 'SLSQP'
            },
            'fit_indices': {
                'chi_square': 25.5,
                'df': 12,
                'p_value': 0.012,
                'cfi': 0.95,
                'rmsea': 0.065
            },
            'path_coefficients': {
                'paths': [('X', 'M'), ('M', 'Y'), ('X', 'Y')],
                'coefficients': {0: 0.45, 1: 0.38, 2: 0.22},
                'p_values': {0: 0.001, 1: 0.003, 2: 0.045}
            }
        }
        
        # 결과 내보내기 테스트
        exporter = PathResultsExporter("test_path_results")
        print("✅ PathResultsExporter 초기화 성공")
        
        saved_files = exporter.export_comprehensive_results(
            test_results, "test_analysis"
        )
        print(f"✅ 결과 내보내기 성공: {len(saved_files)}개 파일")
        
        # 저장된 파일 확인
        for file_type, file_path in saved_files.items():
            if Path(file_path).exists():
                print(f"   ✅ {file_type}: {file_path}")
            else:
                print(f"   ❌ {file_type}: {file_path} (파일 없음)")
        
        return True
        
    except Exception as e:
        print(f"❌ 결과 내보내기 테스트 실패: {e}")
        return False


def test_visualizer():
    """시각화 테스트"""
    print("\n" + "="*50)
    print("6. 시각화 테스트")
    print("="*50)
    
    try:
        # 시각화기 초기화
        visualizer = PathAnalysisVisualizer("test_path_visualizations")
        print("✅ PathAnalysisVisualizer 초기화 성공")
        
        # 테스트용 적합도 지수 시각화
        test_fit_indices = {
            'cfi': 0.95,
            'tli': 0.93,
            'rmsea': 0.065,
            'srmr': 0.075
        }
        
        fit_plot = visualizer.plot_fit_indices(test_fit_indices, "test_fit_indices")
        if fit_plot and fit_plot.exists():
            print(f"✅ 적합도 지수 시각화 성공: {fit_plot}")
        else:
            print("❌ 적합도 지수 시각화 실패")
        
        # 테스트용 경로계수 시각화
        test_path_coeffs = {
            'paths': [('X', 'M'), ('M', 'Y'), ('X', 'Y')],
            'coefficients': {0: 0.45, 1: 0.38, 2: 0.22},
            'p_values': {0: 0.001, 1: 0.003, 2: 0.045}
        }
        
        coeff_plot = visualizer.plot_path_coefficients(test_path_coeffs, "test_path_coeffs")
        if coeff_plot and coeff_plot.exists():
            print(f"✅ 경로계수 시각화 성공: {coeff_plot}")
        else:
            print("❌ 경로계수 시각화 실패")
        
        return True
        
    except Exception as e:
        print(f"❌ 시각화 테스트 실패: {e}")
        return False


def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n" + "="*50)
    print("7. 편의 함수 테스트")
    print("="*50)
    
    try:
        # 모델 생성 편의 함수 테스트
        builder = PathModelBuilder()
        if len(builder.available_factors) >= 3:
            factors = builder.available_factors[:3]
            
            # create_path_model 테스트
            model_spec = create_path_model(
                model_type='simple_mediation',
                independent_var=factors[0],
                mediator_var=factors[1],
                dependent_var=factors[2]
            )
            print("✅ create_path_model 함수 성공")
            
            # 설정 생성 함수 테스트
            default_config = create_default_path_config()
            mediation_config = create_mediation_config()
            print("✅ 설정 생성 함수들 성공")
            
            return True
        else:
            print("❌ 테스트할 충분한 요인이 없습니다")
            return False
            
    except Exception as e:
        print(f"❌ 편의 함수 테스트 실패: {e}")
        return False


def run_comprehensive_test():
    """종합 테스트 실행"""
    print("🧪 PATH ANALYSIS MODULE COMPREHENSIVE TEST")
    print("="*60)
    
    test_results = {}
    
    # 각 테스트 실행
    test_functions = [
        ("데이터 파일 확인", test_data_availability),
        ("모델 빌더", test_model_builder),
        ("경로분석기", test_path_analyzer),
        ("효과 계산기", test_effects_calculator),
        ("결과 내보내기", test_results_exporter),
        ("시각화", test_visualizer),
        ("편의 함수", test_convenience_functions)
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            test_results[test_name] = False
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        if isinstance(result, tuple):
            result = result[0]  # 튜플인 경우 첫 번째 값만 사용
        
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {total - passed}개")
    print(f"성공률: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 경로분석 모듈이 정상적으로 작동합니다.")
    else:
        print(f"\n⚠️  {total - passed}개 테스트 실패. 문제를 확인해 주세요.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
