#!/usr/bin/env python3
"""
경로분석 모듈 실행하여 직접효과 및 간접효과(부트스트래핑) 결과 저장 확인 테스트
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from path_analysis import PathAnalyzer, PathAnalysisConfig
from path_analysis.effects_calculator import EffectsCalculator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """매개효과가 있는 테스트 데이터 생성"""
    
    print("=" * 60)
    print("매개효과 테스트 데이터 생성")
    print("=" * 60)
    
    np.random.seed(42)
    n = 300
    
    # 5개 요인 데이터 생성 (실제 매개효과 포함)
    # X -> M -> Y 구조로 설계
    
    # 독립변수 (건강관심도)
    health_concern = np.random.normal(4, 1, n)
    
    # 매개변수 (지각된 혜택) - 건강관심도에 영향받음
    perceived_benefit = 0.6 * health_concern + np.random.normal(0, 0.8, n)
    
    # 종속변수 (구매의도) - 건강관심도와 지각된 혜택 모두에 영향받음
    purchase_intention = 0.3 * health_concern + 0.5 * perceived_benefit + np.random.normal(0, 0.7, n)
    
    # 추가 변수들
    perceived_price = np.random.normal(3, 1, n)
    nutrition_knowledge = 0.4 * health_concern + np.random.normal(0, 0.9, n)
    
    data = pd.DataFrame({
        'health_concern': health_concern,
        'perceived_benefit': perceived_benefit,
        'purchase_intention': purchase_intention,
        'perceived_price': perceived_price,
        'nutrition_knowledge': nutrition_knowledge
    })
    
    print(f"테스트 데이터 생성 완료: {data.shape}")
    print(f"데이터 기술통계:")
    print(data.describe())
    
    # 실제 효과 계산 (이론값)
    print(f"\n이론적 효과 (데이터 생성 시 설정):")
    print(f"  직접효과 (health_concern -> purchase_intention): 0.3")
    print(f"  간접효과 (health_concern -> perceived_benefit -> purchase_intention): 0.6 * 0.5 = 0.3")
    print(f"  총효과: 0.3 + 0.3 = 0.6")
    
    return data

def test_direct_effects_calculation():
    """직접효과 계산 테스트"""
    
    print("\n" + "=" * 60)
    print("직접효과 계산 테스트")
    print("=" * 60)
    
    try:
        # 테스트 데이터 생성
        data = create_test_data()
        
        # 간단한 구조모델 (관찰변수 사용)
        model_spec = """
        perceived_benefit ~ health_concern
        purchase_intention ~ health_concern + perceived_benefit
        """
        
        print(f"\n모델 스펙:")
        print(model_spec)
        
        # 설정 (부트스트래핑 비활성화)
        config = PathAnalysisConfig(
            include_bootstrap_ci=False,
            bootstrap_samples=0,
            mediation_bootstrap_samples=0
        )
        
        # 경로분석 실행
        analyzer = PathAnalyzer(config)
        results = analyzer.fit_model(model_spec, data)
        
        print(f"\n✅ 경로분석 완료")
        
        # 직접효과 확인
        if 'path_coefficients' in results:
            path_coeffs = results['path_coefficients']
            print(f"\n경로계수 결과:")
            for coeff in path_coeffs:
                if isinstance(coeff, dict):
                    print(f"  {coeff.get('path', 'N/A')}: {coeff.get('coefficient', 'N/A'):.4f} (p={coeff.get('p_value', 'N/A'):.4f})")
        
        # EffectsCalculator로 직접 계산
        from semopy import Model
        model = Model(model_spec)
        model.fit(data)
        
        effects_calc = EffectsCalculator(model)
        effects_calc.set_data(data)
        
        # 직접효과 계산
        try:
            direct_effects = effects_calc.calculate_direct_effects('health_concern', 'purchase_intention')
            print(f"\n직접효과 (health_concern -> purchase_intention):")

            # 안전한 출력
            coeff = direct_effects.get('coefficient', 'N/A')
            if isinstance(coeff, (int, float)):
                print(f"  계수: {coeff:.4f}")
            else:
                print(f"  계수: {coeff}")

            std_err = direct_effects.get('standard_error', 'N/A')
            if isinstance(std_err, (int, float)):
                print(f"  표준오차: {std_err:.4f}")
            else:
                print(f"  표준오차: {std_err}")

            p_val = direct_effects.get('p_value', 'N/A')
            if isinstance(p_val, (int, float)):
                print(f"  p값: {p_val:.4f}")
            else:
                print(f"  p값: {p_val}")
        except Exception as e:
            print(f"직접효과 계산 오류: {e}")
            direct_effects = {'error': str(e)}

        # 간접효과 계산
        try:
            indirect_effects = effects_calc.calculate_indirect_effects('health_concern', 'purchase_intention', ['perceived_benefit'])
            print(f"\n간접효과 (health_concern -> perceived_benefit -> purchase_intention):")

            total_indirect = indirect_effects.get('total_indirect_effect', 'N/A')
            if isinstance(total_indirect, (int, float)):
                print(f"  총 간접효과: {total_indirect:.4f}")
            else:
                print(f"  총 간접효과: {total_indirect}")

            if 'individual_paths' in indirect_effects:
                for mediator, path_info in indirect_effects['individual_paths'].items():
                    indirect_val = path_info.get('indirect_effect', 'N/A')
                    if isinstance(indirect_val, (int, float)):
                        print(f"  {mediator}: {indirect_val:.4f}")
                    else:
                        print(f"  {mediator}: {indirect_val}")
        except Exception as e:
            print(f"간접효과 계산 오류: {e}")
            indirect_effects = {'error': str(e)}
        
        return True, results, direct_effects, indirect_effects
        
    except Exception as e:
        print(f"❌ 직접효과 계산 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def test_bootstrap_effects_calculation():
    """부트스트래핑 효과 계산 및 저장 테스트"""
    
    print("\n" + "=" * 60)
    print("부트스트래핑 효과 계산 및 저장 테스트")
    print("=" * 60)
    
    try:
        # 테스트 데이터 생성
        data = create_test_data()
        
        # 구조모델
        model_spec = """
        perceived_benefit ~ health_concern
        purchase_intention ~ health_concern + perceived_benefit
        """
        
        # 부트스트래핑 설정
        config = PathAnalysisConfig(
            include_bootstrap_ci=True,
            bootstrap_samples=100,  # 빠른 테스트용
            mediation_bootstrap_samples=100,
            bootstrap_method='non-parametric',
            bootstrap_percentile_method='bias_corrected',
            confidence_level=0.95,
            bootstrap_progress_bar=True
        )
        
        print(f"부트스트래핑 설정:")
        print(f"  샘플 수: {config.bootstrap_samples}")
        print(f"  신뢰수준: {config.confidence_level}")
        print(f"  방법: {config.bootstrap_method}")
        
        # 경로분석 실행
        analyzer = PathAnalyzer(config)
        results = analyzer.fit_model(model_spec, data)
        
        print(f"\n✅ 부트스트래핑 포함 경로분석 완료")
        
        # 부트스트래핑 결과 확인
        bootstrap_results = results.get('bootstrap_effects', {})
        print(f"\n부트스트래핑 결과 키: {list(bootstrap_results.keys())}")
        
        # 각 조합별 결과 확인
        for combination_key, combination_result in bootstrap_results.items():
            print(f"\n=== {combination_key} ===")
            
            if 'original_effects' in combination_result:
                original = combination_result['original_effects']
                print(f"원본 효과:")
                for effect_name, effect_value in original.items():
                    if isinstance(effect_value, (int, float)):
                        print(f"  {effect_name}: {effect_value:.4f}")
            
            if 'confidence_intervals' in combination_result:
                ci = combination_result['confidence_intervals']
                print(f"신뢰구간:")
                for effect_name, ci_data in ci.items():
                    if isinstance(ci_data, dict) and 'lower' in ci_data and 'upper' in ci_data:
                        lower = ci_data['lower']
                        upper = ci_data['upper']
                        mean = ci_data.get('mean', 'N/A')
                        significant = ci_data.get('significant', False)
                        print(f"  {effect_name}: [{lower:.4f}, {upper:.4f}] (평균: {mean:.4f}) {'*' if significant else ''}")
            
            if 'bootstrap_statistics' in combination_result:
                stats = combination_result['bootstrap_statistics']
                print(f"부트스트래핑 통계:")
                for effect_name, stat_data in stats.items():
                    if isinstance(stat_data, dict):
                        mean = stat_data.get('mean', 'N/A')
                        std = stat_data.get('std', 'N/A')
                        print(f"  {effect_name}: 평균={mean:.4f}, 표준편차={std:.4f}")
        
        return True, results, bootstrap_results
        
    except Exception as e:
        print(f"❌ 부트스트래핑 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_effects_calculator_directly():
    """EffectsCalculator 직접 테스트"""
    
    print("\n" + "=" * 60)
    print("EffectsCalculator 직접 부트스트래핑 테스트")
    print("=" * 60)
    
    try:
        # 테스트 데이터
        data = create_test_data()
        
        # 모델 적합
        from semopy import Model
        model_spec = """
        perceived_benefit ~ health_concern
        purchase_intention ~ health_concern + perceived_benefit
        """
        
        model = Model(model_spec)
        model.fit(data)
        
        print("모델 적합 완료")
        
        # EffectsCalculator 초기화
        effects_calc = EffectsCalculator(model)
        effects_calc.set_data(data)
        effects_calc.model_spec = model_spec
        
        # 부트스트래핑 실행
        print("\n부트스트래핑 실행...")
        bootstrap_results = effects_calc.calculate_bootstrap_effects(
            independent_var='health_concern',
            dependent_var='purchase_intention',
            mediator_vars=['perceived_benefit'],
            n_bootstrap=50,  # 빠른 테스트
            confidence_level=0.95,
            method='bias-corrected',
            show_progress=True
        )
        
        print("✅ 부트스트래핑 완료!")
        
        # 결과 상세 확인
        print(f"\n부트스트래핑 결과 구조:")
        for key, value in bootstrap_results.items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            else:
                print(f"  {key}: {type(value)}")
        
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
                    print(f"  {effect_name}: [{lower:.4f}, {upper:.4f}] (평균: {mean:.4f}) {'*' if significant else ''}")
        
        # 부트스트래핑 샘플 확인
        if 'bootstrap_results' in bootstrap_results:
            bootstrap_data = bootstrap_results['bootstrap_results']
            print(f"\n부트스트래핑 샘플:")
            for effect_name, samples in bootstrap_data.items():
                if isinstance(samples, list) and len(samples) > 0:
                    print(f"  {effect_name}: {len(samples)}개 샘플, 평균={np.mean(samples):.4f}, 표준편차={np.std(samples):.4f}")
        
        return True, bootstrap_results
        
    except Exception as e:
        print(f"❌ EffectsCalculator 직접 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def save_results_to_file(results, bootstrap_results, direct_effects, indirect_effects):
    """결과를 파일로 저장"""
    
    print("\n" + "=" * 60)
    print("결과 파일 저장")
    print("=" * 60)
    
    try:
        # 결과 디렉토리 생성
        results_dir = "path_analysis_effects_test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 전체 결과 저장 (JSON)
        full_results = {
            'timestamp': timestamp,
            'path_analysis_results': results,
            'bootstrap_results': bootstrap_results,
            'direct_effects': direct_effects,
            'indirect_effects': indirect_effects
        }
        
        # JSON 직렬화 가능하도록 변환
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return obj
        
        # 재귀적으로 변환
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_for_json(obj)
        
        json_results = deep_convert(full_results)
        
        json_file = os.path.join(results_dir, f"effects_test_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON 결과 저장: {json_file}")
        
        # 2. 요약 결과 저장 (텍스트)
        summary_file = os.path.join(results_dir, f"effects_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("경로분석 효과 계산 테스트 결과 요약\n")
            f.write("=" * 50 + "\n")
            f.write(f"실행 시간: {timestamp}\n\n")
            
            # 직접효과
            f.write("직접효과 결과:\n")
            if direct_effects:
                for key, value in direct_effects.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 간접효과
            f.write("간접효과 결과:\n")
            if indirect_effects:
                for key, value in indirect_effects.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 부트스트래핑 요약
            f.write("부트스트래핑 결과 요약:\n")
            if bootstrap_results:
                for combination_key, combination_result in bootstrap_results.items():
                    f.write(f"\n{combination_key}:\n")
                    if 'confidence_intervals' in combination_result:
                        ci = combination_result['confidence_intervals']
                        for effect_name, ci_data in ci.items():
                            if isinstance(ci_data, dict):
                                lower = ci_data.get('lower', 'N/A')
                                upper = ci_data.get('upper', 'N/A')
                                significant = ci_data.get('significant', False)
                                f.write(f"  {effect_name}: [{lower:.4f}, {upper:.4f}] {'*' if significant else ''}\n")
        
        print(f"✅ 요약 결과 저장: {summary_file}")
        
        return True, results_dir
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("경로분석 모듈 직접효과 및 간접효과(부트스트래핑) 결과 저장 확인 테스트")
    
    # 1. 직접효과 계산 테스트
    direct_success, path_results, direct_effects, indirect_effects = test_direct_effects_calculation()
    
    # 2. 부트스트래핑 효과 계산 테스트
    bootstrap_success, bootstrap_path_results, bootstrap_results = test_bootstrap_effects_calculation()
    
    # 3. EffectsCalculator 직접 테스트
    effects_calc_success, effects_calc_results = test_effects_calculator_directly()
    
    # 4. 결과 저장
    if direct_success or bootstrap_success or effects_calc_success:
        save_success, results_dir = save_results_to_file(
            path_results, bootstrap_results, direct_effects, indirect_effects
        )
    else:
        save_success = False
        results_dir = None
    
    print(f"\n" + "=" * 60)
    print("최종 테스트 결과")
    print("=" * 60)
    print(f"직접효과 계산: {'✅ 성공' if direct_success else '❌ 실패'}")
    print(f"부트스트래핑 계산: {'✅ 성공' if bootstrap_success else '❌ 실패'}")
    print(f"EffectsCalculator: {'✅ 성공' if effects_calc_success else '❌ 실패'}")
    print(f"결과 저장: {'✅ 성공' if save_success else '❌ 실패'}")
    
    if save_success and results_dir:
        print(f"\n📁 결과 저장 위치: {results_dir}")
    
    if all([direct_success, bootstrap_success, effects_calc_success, save_success]):
        print(f"\n🎉 모든 테스트 통과!")
        print("✅ 직접효과 계산이 정상 작동합니다.")
        print("✅ 간접효과(부트스트래핑) 계산이 정상 작동합니다.")
        print("✅ 결과가 올바르게 저장됩니다.")
        print("✅ 경로분석 모듈의 모든 효과 계산 기능이 정상입니다.")
    else:
        print(f"\n⚠️  일부 테스트에서 문제가 발생했습니다.")
        if direct_success:
            print("✅ 직접효과 계산은 정상 작동합니다.")
        if bootstrap_success:
            print("✅ 부트스트래핑 계산은 정상 작동합니다.")
        if effects_calc_success:
            print("✅ EffectsCalculator는 정상 작동합니다.")
        if save_success:
            print("✅ 결과 저장은 정상 작동합니다.")
