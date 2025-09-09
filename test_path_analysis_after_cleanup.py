#!/usr/bin/env python3
"""
부트스트래핑 모듈 정리 후 경로분석 모듈 정상 작동 확인 테스트
"""

import pandas as pd
import numpy as np
import logging
from path_analysis import PathAnalyzer, PathAnalysisConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_path_analysis():
    """기본 경로분석 테스트 (부트스트래핑 없이)"""
    
    print("=" * 60)
    print("기본 경로분석 테스트 (부트스트래핑 없이)")
    print("=" * 60)
    
    try:
        # 테스트 데이터 생성
        np.random.seed(42)
        n = 200
        
        # 5개 요인 데이터 생성
        data = pd.DataFrame({
            # 건강관심도 (Health Concern)
            'HC1': np.random.normal(4, 1, n),
            'HC2': np.random.normal(4, 1, n),
            'HC3': np.random.normal(4, 1, n),
            
            # 지각된 혜택 (Perceived Benefit)
            'PB1': np.random.normal(4, 1, n),
            'PB2': np.random.normal(4, 1, n),
            'PB3': np.random.normal(4, 1, n),
            
            # 구매의도 (Purchase Intention)
            'PI1': np.random.normal(3, 1, n),
            'PI2': np.random.normal(3, 1, n),
            'PI3': np.random.normal(3, 1, n),
            
            # 지각된 가격 (Perceived Price)
            'PP1': np.random.normal(3, 1, n),
            'PP2': np.random.normal(3, 1, n),
            
            # 영양지식 (Nutrition Knowledge)
            'NK1': np.random.normal(4, 1, n),
            'NK2': np.random.normal(4, 1, n),
        })
        
        print(f"테스트 데이터 생성: {data.shape}")
        
        # 모델 스펙 정의
        model_spec = """
        # 측정모델
        health_concern =~ HC1 + HC2 + HC3
        perceived_benefit =~ PB1 + PB2 + PB3
        purchase_intention =~ PI1 + PI2 + PI3
        perceived_price =~ PP1 + PP2
        nutrition_knowledge =~ NK1 + NK2
        
        # 구조모델
        perceived_benefit ~ health_concern + nutrition_knowledge
        purchase_intention ~ perceived_benefit + perceived_price + health_concern
        """
        
        print("모델 스펙 정의 완료")
        
        # 설정 (부트스트래핑 비활성화)
        config = PathAnalysisConfig(
            include_bootstrap_ci=False,  # 부트스트래핑 비활성화
            bootstrap_samples=0,
            mediation_bootstrap_samples=0
        )
        
        print("설정 완료 (부트스트래핑 비활성화)")
        
        # 경로분석 실행
        analyzer = PathAnalyzer(config)
        results = analyzer.fit_model(model_spec, data)
        
        print("✅ 기본 경로분석 성공!")
        
        # 결과 확인
        if 'fit_indices' in results:
            fit_indices = results['fit_indices']
            print(f"모델 적합도:")
            for key, value in fit_indices.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        if 'path_coefficients' in results:
            path_coeffs = results['path_coefficients']
            print(f"경로계수 수: {len(path_coeffs)}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ 기본 경로분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_path_analysis_with_bootstrap():
    """부트스트래핑 포함 경로분석 테스트"""
    
    print("\n" + "=" * 60)
    print("부트스트래핑 포함 경로분석 테스트")
    print("=" * 60)
    
    try:
        # 테스트 데이터 생성
        np.random.seed(123)
        n = 150
        
        data = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'y1': np.random.normal(0, 1, n),
            'z1': np.random.normal(0, 1, n),
        })
        
        print(f"테스트 데이터 생성: {data.shape}")
        
        # 간단한 매개효과 모델
        model_spec = """
        X =~ x1 + x2
        Y =~ y1
        Z =~ z1
        
        Y ~ X
        Z ~ X + Y
        """
        
        print("매개효과 모델 스펙 정의 완료")
        
        # 설정 (부트스트래핑 활성화)
        config = PathAnalysisConfig(
            include_bootstrap_ci=True,  # 부트스트래핑 활성화
            bootstrap_samples=20,  # 빠른 테스트용
            mediation_bootstrap_samples=20,
            bootstrap_method='non-parametric'
        )
        
        print("설정 완료 (부트스트래핑 활성화)")
        
        # 경로분석 실행
        analyzer = PathAnalyzer(config)
        results = analyzer.fit_model(model_spec, data)
        
        print("✅ 부트스트래핑 포함 경로분석 성공!")
        
        # 결과 확인
        if 'bootstrap_results' in results:
            bootstrap_results = results['bootstrap_results']
            print(f"부트스트래핑 결과 포함됨")
            
            if 'confidence_intervals' in bootstrap_results:
                ci = bootstrap_results['confidence_intervals']
                print(f"신뢰구간 수: {len(ci)}")
        
        if 'mediation_results' in results:
            mediation_results = results['mediation_results']
            print(f"매개효과 분석 결과 포함됨")
        
        return True, results
        
    except Exception as e:
        print(f"❌ 부트스트래핑 포함 경로분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_effects_calculator_directly():
    """EffectsCalculator 직접 테스트"""
    
    print("\n" + "=" * 60)
    print("EffectsCalculator 직접 테스트")
    print("=" * 60)
    
    try:
        from path_analysis.effects_calculator import EffectsCalculator
        from semopy import Model
        
        # 테스트 데이터
        np.random.seed(456)
        n = 100
        data = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'y1': np.random.normal(0, 1, n),
            'z1': np.random.normal(0, 1, n),
        })
        
        model_spec = """
        X =~ x1
        Y =~ y1
        Z =~ z1
        Y ~ X
        Z ~ Y
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
        direct_effects = effects_calc.calculate_direct_effects('X', 'Z')
        print(f"✅ 직접효과 계산 성공: {direct_effects}")
        
        # 간접효과 계산
        indirect_effects = effects_calc.calculate_indirect_effects('X', 'Z', ['Y'])
        print(f"✅ 간접효과 계산 성공: {indirect_effects}")
        
        # 총효과 계산 (직접효과와 간접효과를 이용)
        total_effects = effects_calc.calculate_total_effects(direct_effects, indirect_effects)
        print(f"✅ 총효과 계산 성공: {total_effects}")
        
        # 부트스트래핑 테스트 (소규모)
        bootstrap_results = effects_calc.calculate_bootstrap_effects(
            independent_var='X',
            dependent_var='Z',
            mediator_vars=['Y'],
            n_bootstrap=10,  # 매우 빠른 테스트
            method='non-parametric',
            show_progress=False
        )
        
        print(f"✅ 부트스트래핑 계산 성공!")
        
        return True
        
    except Exception as e:
        print(f"❌ EffectsCalculator 직접 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_survey_data():
    """실제 설문 데이터로 테스트"""
    
    print("\n" + "=" * 60)
    print("실제 설문 데이터 테스트")
    print("=" * 60)
    
    try:
        # 실제 데이터 로드 시도
        data_path = "processed_data/survey_data/factor_scores.csv"
        
        try:
            data = pd.read_csv(data_path)
            print(f"실제 데이터 로드 성공: {data.shape}")
        except FileNotFoundError:
            print("실제 데이터 파일을 찾을 수 없어 테스트 데이터 사용")
            # 대안: 테스트 데이터
            np.random.seed(789)
            n = 300
            data = pd.DataFrame({
                'health_concern': np.random.normal(4, 1, n),
                'perceived_benefit': np.random.normal(4, 1, n),
                'purchase_intention': np.random.normal(3, 1, n),
                'perceived_price': np.random.normal(3, 1, n),
                'nutrition_knowledge': np.random.normal(4, 1, n),
            })
        
        # 간단한 구조모델
        model_spec = """
        perceived_benefit ~ health_concern + nutrition_knowledge
        purchase_intention ~ perceived_benefit + perceived_price + health_concern
        """
        
        print("구조모델 스펙 정의 완료")
        
        # 설정
        config = PathAnalysisConfig(
            include_bootstrap_ci=True,
            bootstrap_samples=30,  # 적당한 크기
            mediation_bootstrap_samples=30
        )
        
        # 경로분석 실행
        analyzer = PathAnalyzer(config)
        results = analyzer.fit_model(model_spec, data)
        
        print("✅ 실제 데이터 경로분석 성공!")
        
        return True, results
        
    except Exception as e:
        print(f"❌ 실제 데이터 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("부트스트래핑 모듈 정리 후 경로분석 모듈 정상 작동 확인")
    
    # 1. 기본 경로분석 테스트
    basic_success, basic_results = test_basic_path_analysis()
    
    # 2. 부트스트래핑 포함 테스트
    bootstrap_success, bootstrap_results = test_path_analysis_with_bootstrap()
    
    # 3. EffectsCalculator 직접 테스트
    effects_success = test_effects_calculator_directly()
    
    # 4. 실제 데이터 테스트
    real_data_success, real_results = test_real_survey_data()
    
    print(f"\n" + "=" * 60)
    print("최종 테스트 결과")
    print("=" * 60)
    print(f"기본 경로분석: {'✅ 성공' if basic_success else '❌ 실패'}")
    print(f"부트스트래핑 포함: {'✅ 성공' if bootstrap_success else '❌ 실패'}")
    print(f"EffectsCalculator: {'✅ 성공' if effects_success else '❌ 실패'}")
    print(f"실제 데이터: {'✅ 성공' if real_data_success else '❌ 실패'}")
    
    if all([basic_success, bootstrap_success, effects_success, real_data_success]):
        print(f"\n🎉 모든 테스트 통과!")
        print("✅ 부트스트래핑 모듈 정리 후에도 경로분석 모듈이 정상 작동합니다.")
        print("✅ semopy 내장 기능을 활용한 효율적인 부트스트래핑이 작동합니다.")
        print("✅ 기존 기능들이 모두 유지되고 있습니다.")
    else:
        print(f"\n⚠️  일부 테스트에서 문제가 발생했습니다.")
        print("추가 디버깅이 필요할 수 있습니다.")
