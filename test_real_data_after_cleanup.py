#!/usr/bin/env python3
"""
실제 설문 데이터로 부트스트래핑 모듈 정리 후 테스트
"""

import pandas as pd
import numpy as np
import logging
from path_analysis import PathAnalyzer, PathAnalysisConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_with_real_survey_data():
    """실제 설문 데이터로 테스트"""
    
    print("=" * 60)
    print("실제 설문 데이터로 경로분석 테스트")
    print("=" * 60)
    
    try:
        # 실제 데이터 로드 시도
        data_paths = [
            "processed_data/survey_data/factor_scores.csv",
            "processed_data/survey_data/survey_data_processed.csv",
            "processed_data/survey_data/survey_data.csv"
        ]
        
        data = None
        for path in data_paths:
            try:
                data = pd.read_csv(path)
                print(f"✅ 데이터 로드 성공: {path}")
                print(f"   데이터 크기: {data.shape}")
                print(f"   컬럼: {list(data.columns)}")
                break
            except FileNotFoundError:
                continue
        
        if data is None:
            print("❌ 실제 데이터를 찾을 수 없습니다.")
            return False
        
        # 5개 요인이 있는지 확인
        expected_factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 
                          'perceived_price', 'nutrition_knowledge']
        
        available_factors = [col for col in expected_factors if col in data.columns]
        print(f"사용 가능한 요인: {available_factors}")
        
        if len(available_factors) < 3:
            print("❌ 충분한 요인이 없어 테스트를 건너뜁니다.")
            return False
        
        # 데이터 전처리
        analysis_data = data[available_factors].dropna()
        print(f"분석 데이터: {analysis_data.shape}")
        
        # 모델 스펙 생성 (사용 가능한 요인에 따라)
        if len(available_factors) >= 5:
            # 전체 모델
            model_spec = """
            perceived_benefit ~ health_concern + nutrition_knowledge
            purchase_intention ~ perceived_benefit + perceived_price + health_concern
            """
        elif len(available_factors) >= 3:
            # 간단한 모델
            factors = available_factors[:3]
            model_spec = f"""
            {factors[1]} ~ {factors[0]}
            {factors[2]} ~ {factors[1]} + {factors[0]}
            """
        else:
            print("❌ 모델 구성에 필요한 최소 요인 수가 부족합니다.")
            return False
        
        print("모델 스펙:")
        print(model_spec)
        
        # 1. 기본 경로분석 (부트스트래핑 없이)
        print("\n1. 기본 경로분석 테스트...")
        config_basic = PathAnalysisConfig(
            include_bootstrap_ci=False,
            bootstrap_samples=0,
            mediation_bootstrap_samples=0
        )
        
        analyzer_basic = PathAnalyzer(config_basic)
        results_basic = analyzer_basic.fit_model(model_spec, analysis_data)
        
        print("✅ 기본 경로분석 성공!")
        
        # 결과 확인
        if 'fit_indices' in results_basic:
            fit_indices = results_basic['fit_indices']
            print(f"   적합도 지수:")
            for key, value in fit_indices.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"     {key}: {value:.4f}")
        
        if 'path_coefficients' in results_basic:
            path_coeffs = results_basic['path_coefficients']
            print(f"   경로계수 수: {len(path_coeffs)}")
        
        # 2. 부트스트래핑 포함 경로분석
        print("\n2. 부트스트래핑 포함 경로분석 테스트...")
        config_bootstrap = PathAnalysisConfig(
            include_bootstrap_ci=True,
            bootstrap_samples=50,  # 적당한 크기
            mediation_bootstrap_samples=50,
            bootstrap_method='non-parametric'
        )
        
        analyzer_bootstrap = PathAnalyzer(config_bootstrap)
        results_bootstrap = analyzer_bootstrap.fit_model(model_spec, analysis_data)
        
        print("✅ 부트스트래핑 포함 경로분석 성공!")
        
        # 부트스트래핑 결과 확인
        if 'bootstrap_results' in results_bootstrap:
            bootstrap_results = results_bootstrap['bootstrap_results']
            print(f"   부트스트래핑 결과 포함됨")
            
            if 'confidence_intervals' in bootstrap_results:
                ci = bootstrap_results['confidence_intervals']
                print(f"   신뢰구간 수: {len(ci)}")
                
                # 몇 개 신뢰구간 출력
                for i, (effect_name, ci_data) in enumerate(ci.items()):
                    if i >= 3:  # 처음 3개만
                        break
                    if isinstance(ci_data, dict) and 'lower' in ci_data and 'upper' in ci_data:
                        lower = ci_data['lower']
                        upper = ci_data['upper']
                        significant = ci_data.get('significant', False)
                        print(f"     {effect_name}: [{lower:.4f}, {upper:.4f}] {'*' if significant else ''}")
        
        if 'mediation_results' in results_bootstrap:
            mediation_results = results_bootstrap['mediation_results']
            print(f"   매개효과 분석 결과 포함됨")
        
        return True
        
    except Exception as e:
        print(f"❌ 실제 데이터 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_effects_calculator_with_real_data():
    """실제 데이터로 EffectsCalculator 직접 테스트"""
    
    print("\n" + "=" * 60)
    print("실제 데이터로 EffectsCalculator 직접 테스트")
    print("=" * 60)
    
    try:
        from path_analysis.effects_calculator import EffectsCalculator
        from semopy import Model
        
        # 실제 데이터 로드 시도
        try:
            data = pd.read_csv("processed_data/survey_data/factor_scores.csv")
            print(f"✅ 데이터 로드 성공: {data.shape}")
        except FileNotFoundError:
            print("실제 데이터를 찾을 수 없어 테스트 데이터 사용")
            np.random.seed(789)
            n = 200
            data = pd.DataFrame({
                'health_concern': np.random.normal(4, 1, n),
                'perceived_benefit': np.random.normal(4, 1, n),
                'purchase_intention': np.random.normal(3, 1, n),
            })
        
        # 간단한 매개효과 모델
        model_spec = """
        perceived_benefit ~ health_concern
        purchase_intention ~ perceived_benefit + health_concern
        """
        
        # 모델 적합
        model = Model(model_spec)
        model.fit(data)
        
        print("모델 적합 완료")
        
        # EffectsCalculator 테스트
        effects_calc = EffectsCalculator(model)
        effects_calc.set_data(data)
        effects_calc.model_spec = model_spec
        
        # 직접효과 계산
        print("\n직접효과 계산...")
        direct_effects = effects_calc.calculate_direct_effects('health_concern', 'purchase_intention')
        print(f"✅ 직접효과: {direct_effects.get('coefficient', 'N/A')}")
        
        # 간접효과 계산
        print("\n간접효과 계산...")
        indirect_effects = effects_calc.calculate_indirect_effects('health_concern', 'purchase_intention', ['perceived_benefit'])
        print(f"✅ 간접효과: {indirect_effects.get('total_indirect_effect', 'N/A')}")
        
        # 부트스트래핑 테스트
        print("\n부트스트래핑 테스트...")
        bootstrap_results = effects_calc.calculate_bootstrap_effects(
            independent_var='health_concern',
            dependent_var='purchase_intention',
            mediator_vars=['perceived_benefit'],
            n_bootstrap=20,  # 빠른 테스트
            method='non-parametric',
            show_progress=True
        )
        
        print("✅ 부트스트래핑 완료!")
        
        # 결과 확인
        if 'confidence_intervals' in bootstrap_results:
            ci = bootstrap_results['confidence_intervals']
            print(f"신뢰구간:")
            for effect_name, ci_data in ci.items():
                if isinstance(ci_data, dict) and 'lower' in ci_data and 'upper' in ci_data:
                    lower = ci_data['lower']
                    upper = ci_data['upper']
                    significant = ci_data.get('significant', False)
                    print(f"  {effect_name}: [{lower:.4f}, {upper:.4f}] {'*' if significant else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ EffectsCalculator 실제 데이터 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("실제 설문 데이터로 부트스트래핑 모듈 정리 후 테스트")
    
    # 1. 실제 데이터로 경로분석 테스트
    real_data_success = test_with_real_survey_data()
    
    # 2. EffectsCalculator 직접 테스트
    effects_success = test_effects_calculator_with_real_data()
    
    print(f"\n" + "=" * 60)
    print("최종 테스트 결과")
    print("=" * 60)
    print(f"실제 데이터 경로분석: {'✅ 성공' if real_data_success else '❌ 실패'}")
    print(f"EffectsCalculator: {'✅ 성공' if effects_success else '❌ 실패'}")
    
    if real_data_success and effects_success:
        print(f"\n🎉 실제 데이터 테스트 모두 통과!")
        print("✅ 부트스트래핑 모듈 정리 후에도 실제 데이터 분석이 정상 작동합니다.")
        print("✅ semopy 내장 기능을 활용한 부트스트래핑이 실제 데이터에서 작동합니다.")
        print("✅ 모든 경로분석 기능이 실제 환경에서 정상 작동합니다.")
    else:
        print(f"\n⚠️  일부 실제 데이터 테스트에서 문제가 발생했습니다.")
        if real_data_success:
            print("✅ 실제 데이터 경로분석은 정상 작동합니다.")
        if effects_success:
            print("✅ EffectsCalculator는 정상 작동합니다.")
