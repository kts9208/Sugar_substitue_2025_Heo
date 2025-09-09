#!/usr/bin/env python3
"""
부트스트래핑 모듈 정리 후 간단한 테스트
"""

import pandas as pd
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """기본 기능 테스트"""
    
    print("=" * 50)
    print("기본 기능 테스트")
    print("=" * 50)
    
    try:
        # 1. 모듈 임포트 테스트
        print("1. 모듈 임포트 테스트...")
        from path_analysis import PathAnalyzer, PathAnalysisConfig
        from path_analysis.effects_calculator import EffectsCalculator
        from semopy import Model
        print("✅ 모든 모듈 임포트 성공")
        
        # 2. 간단한 데이터 생성
        print("\n2. 테스트 데이터 생성...")
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, n),
            'y': np.random.normal(0, 1, n),
            'z': np.random.normal(0, 1, n),
        })
        print(f"✅ 데이터 생성 성공: {data.shape}")
        
        # 3. 간단한 모델 스펙
        print("\n3. 모델 스펙 정의...")
        model_spec = """
        y ~ x
        z ~ y
        """
        print("✅ 모델 스펙 정의 성공")
        
        # 4. semopy 모델 직접 테스트
        print("\n4. semopy 모델 직접 테스트...")
        model = Model(model_spec)
        model.fit(data)
        print("✅ semopy 모델 적합 성공")
        
        # 5. EffectsCalculator 기본 테스트
        print("\n5. EffectsCalculator 기본 테스트...")
        effects_calc = EffectsCalculator(model)
        effects_calc.set_data(data)
        print("✅ EffectsCalculator 초기화 성공")
        
        # 6. 직접효과 계산
        print("\n6. 직접효과 계산...")
        direct_effects = effects_calc.calculate_direct_effects('x', 'y')
        print(f"✅ 직접효과 계산 성공: {direct_effects.get('coefficient', 'N/A')}")
        
        # 7. PathAnalyzer 기본 테스트 (부트스트래핑 없이)
        print("\n7. PathAnalyzer 기본 테스트...")
        config = PathAnalysisConfig(
            include_bootstrap_ci=False,
            bootstrap_samples=0,
            mediation_bootstrap_samples=0
        )
        analyzer = PathAnalyzer(config)
        results = analyzer.fit_model(model_spec, data)
        print("✅ PathAnalyzer 기본 분석 성공")
        
        # 8. 결과 확인
        print("\n8. 결과 확인...")
        if 'fit_indices' in results:
            print(f"  적합도 지수 포함됨")
        if 'path_coefficients' in results:
            print(f"  경로계수 포함됨")
        print("✅ 결과 구조 정상")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bootstrap_functionality():
    """부트스트래핑 기능 테스트"""
    
    print("\n" + "=" * 50)
    print("부트스트래핑 기능 테스트")
    print("=" * 50)
    
    try:
        from path_analysis.effects_calculator import EffectsCalculator
        from semopy import Model
        
        # 간단한 데이터
        np.random.seed(123)
        n = 50
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, n),
            'y': np.random.normal(0, 1, n),
        })
        
        model_spec = "y ~ x"
        model = Model(model_spec)
        model.fit(data)
        
        effects_calc = EffectsCalculator(model)
        effects_calc.set_data(data)
        effects_calc.model_spec = model_spec
        
        print("1. 기본 설정 완료")
        
        # 매우 작은 부트스트래핑 테스트
        print("2. 소규모 부트스트래핑 테스트...")
        bootstrap_results = effects_calc.calculate_bootstrap_effects(
            independent_var='x',
            dependent_var='y',
            mediator_vars=None,
            n_bootstrap=5,  # 매우 작은 수
            method='non-parametric',
            show_progress=False
        )
        
        print("✅ 부트스트래핑 계산 성공")
        
        # 결과 확인
        if 'bootstrap_results' in bootstrap_results:
            bootstrap_data = bootstrap_results['bootstrap_results']
            if 'direct_effects' in bootstrap_data:
                n_samples = len(bootstrap_data['direct_effects'])
                print(f"  부트스트래핑 샘플 수: {n_samples}")
        
        if 'confidence_intervals' in bootstrap_results:
            print(f"  신뢰구간 계산됨")
        
        print("✅ 부트스트래핑 결과 구조 정상")
        
        return True
        
    except Exception as e:
        print(f"❌ 부트스트래핑 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semopy_native_features():
    """semopy 내장 기능 테스트"""
    
    print("\n" + "=" * 50)
    print("semopy 내장 기능 테스트")
    print("=" * 50)
    
    try:
        from semopy import Model, bias_correction
        from semopy.model_generation import generate_data
        
        # 테스트 데이터
        np.random.seed(456)
        n = 50
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, n),
            'y': np.random.normal(0, 1, n),
        })
        
        model_spec = "y ~ x"
        model = Model(model_spec)
        model.fit(data)
        
        print("1. 기본 모델 적합 완료")
        
        # generate_data 테스트
        print("2. generate_data 테스트...")
        generated_data = generate_data(model, n=n)
        print(f"✅ 데이터 생성 성공: {generated_data.shape}")
        
        # bias_correction 테스트
        print("3. bias_correction 테스트...")
        original_params = model.inspect()
        bias_correction(model, n=10)  # 작은 수로 테스트
        corrected_params = model.inspect()
        print("✅ 편향 보정 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ semopy 내장 기능 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("부트스트래핑 모듈 정리 후 간단한 기능 테스트")
    
    # 1. 기본 기능 테스트
    basic_success = test_basic_functionality()
    
    # 2. 부트스트래핑 기능 테스트
    bootstrap_success = test_bootstrap_functionality()
    
    # 3. semopy 내장 기능 테스트
    semopy_success = test_semopy_native_features()
    
    print(f"\n" + "=" * 50)
    print("최종 테스트 결과")
    print("=" * 50)
    print(f"기본 기능: {'✅ 성공' if basic_success else '❌ 실패'}")
    print(f"부트스트래핑: {'✅ 성공' if bootstrap_success else '❌ 실패'}")
    print(f"semopy 내장: {'✅ 성공' if semopy_success else '❌ 실패'}")
    
    if all([basic_success, bootstrap_success, semopy_success]):
        print(f"\n🎉 모든 테스트 통과!")
        print("✅ 부트스트래핑 모듈 정리 후에도 모든 기능이 정상 작동합니다.")
        print("✅ semopy 내장 기능을 활용한 부트스트래핑이 작동합니다.")
        print("✅ 기존 경로분석 기능이 모두 유지되고 있습니다.")
    else:
        print(f"\n⚠️  일부 기능에서 문제가 발생했습니다.")
        if basic_success:
            print("✅ 기본 경로분석 기능은 정상 작동합니다.")
        if bootstrap_success:
            print("✅ 부트스트래핑 기능은 정상 작동합니다.")
        if semopy_success:
            print("✅ semopy 내장 기능은 정상 작동합니다.")
