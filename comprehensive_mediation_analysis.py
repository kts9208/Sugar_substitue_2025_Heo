#!/usr/bin/env python3
"""
모든 요인에 대한 포괄적 매개효과 분석
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from path_analysis import PathAnalyzer, PathAnalysisConfig
from path_analysis.effects_calculator import EffectsCalculator
from semopy import Model

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_comprehensive_five_factor_data():
    """5개 요인 간 현실적인 관계를 가진 데이터 생성"""
    
    print("=" * 60)
    print("5개 요인 포괄적 매개효과 데이터 생성")
    print("=" * 60)
    
    np.random.seed(42)
    n = 300
    
    # 5개 요인: health_concern, perceived_benefit, purchase_intention, perceived_price, nutrition_knowledge
    
    # 1. 건강관심도 (기본 독립변수)
    health_concern = np.random.normal(4.0, 1.0, n)
    
    # 2. 영양지식 (건강관심도에 영향받음)
    nutrition_knowledge = 0.6 * health_concern + np.random.normal(0, 0.8, n)
    
    # 3. 지각된 혜택 (건강관심도와 영양지식에 영향받음)
    perceived_benefit = 0.4 * health_concern + 0.3 * nutrition_knowledge + np.random.normal(0, 0.7, n)
    
    # 4. 지각된 가격 (건강관심도에 약간 부정적 영향)
    perceived_price = -0.2 * health_concern + np.random.normal(3.5, 1.0, n)
    
    # 5. 구매의도 (모든 요인에 영향받음)
    purchase_intention = (0.3 * health_concern + 
                         0.4 * perceived_benefit + 
                         -0.3 * perceived_price + 
                         0.2 * nutrition_knowledge + 
                         np.random.normal(0, 0.6, n))
    
    # 1-7 스케일로 조정
    data = pd.DataFrame({
        'health_concern': np.clip(health_concern, 1, 7),
        'perceived_benefit': np.clip(perceived_benefit, 1, 7),
        'purchase_intention': np.clip(purchase_intention, 1, 7),
        'perceived_price': np.clip(perceived_price, 1, 7),
        'nutrition_knowledge': np.clip(nutrition_knowledge, 1, 7)
    })
    
    print(f"데이터 생성 완료: {data.shape}")
    print(f"5개 요인: {list(data.columns)}")
    print(f"\n기술통계:")
    print(data.describe())
    
    print(f"\n이론적 관계:")
    print(f"  health_concern → nutrition_knowledge (0.6)")
    print(f"  health_concern → perceived_benefit (0.4)")
    print(f"  nutrition_knowledge → perceived_benefit (0.3)")
    print(f"  health_concern → perceived_price (-0.2)")
    print(f"  health_concern → purchase_intention (0.3)")
    print(f"  perceived_benefit → purchase_intention (0.4)")
    print(f"  perceived_price → purchase_intention (-0.3)")
    print(f"  nutrition_knowledge → purchase_intention (0.2)")
    
    return data

def create_comprehensive_model_spec():
    """5개 요인 포괄적 구조모델 스펙 생성"""
    
    model_spec = """
    # 측정모델 (관찰변수 사용)
    
    # 구조모델 (이론적 관계 기반)
    nutrition_knowledge ~ health_concern
    perceived_benefit ~ health_concern + nutrition_knowledge
    perceived_price ~ health_concern
    purchase_intention ~ health_concern + perceived_benefit + perceived_price + nutrition_knowledge
    """
    
    return model_spec

def analyze_all_mediation_combinations():
    """모든 가능한 매개효과 조합 분석"""
    
    print("\n" + "=" * 60)
    print("모든 가능한 매개효과 조합 분석")
    print("=" * 60)
    
    try:
        # 데이터 생성
        data = create_comprehensive_five_factor_data()
        
        # 모델 스펙
        model_spec = create_comprehensive_model_spec()
        
        print(f"\n모델 스펙:")
        print(model_spec)
        
        # 모델 적합
        model = Model(model_spec)
        model.fit(data)
        
        print("✅ 모델 적합 완료")
        
        # 모델 파라미터 확인
        params = model.inspect()
        structural_params = params[params['op'] == '~']
        if len(structural_params) > 0:
            print(f"\n구조적 경로 파라미터:")
            basic_cols = ['lval', 'op', 'rval', 'Estimate']
            available_cols = [col for col in basic_cols if col in params.columns]
            if available_cols:
                print(structural_params[available_cols].to_string())
        
        # EffectsCalculator 초기화
        effects_calc = EffectsCalculator(model)
        effects_calc.set_data(data)
        effects_calc.model_spec = model_spec
        
        # 5개 요인 리스트
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention', 'perceived_price', 'nutrition_knowledge']
        
        print(f"\n모든 가능한 매개효과 분석 시작...")
        print(f"분석 대상 변수: {variables}")
        
        # 모든 가능한 매개효과 분석
        all_mediation_results = effects_calc.analyze_all_possible_mediations(
            variables=variables,
            bootstrap_samples=100,  # 빠른 테스트용
            confidence_level=0.95,
            parallel=False,  # 안정성을 위해 비활성화
            show_progress=True
        )
        
        print("✅ 모든 매개효과 분석 완료!")
        
        return True, all_mediation_results, data, model_spec
        
    except Exception as e:
        print(f"❌ 매개효과 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def analyze_pathanalyzer_comprehensive():
    """PathAnalyzer를 사용한 포괄적 분석"""
    
    print("\n" + "=" * 60)
    print("PathAnalyzer 포괄적 매개효과 분석")
    print("=" * 60)
    
    try:
        # 데이터 생성
        data = create_comprehensive_five_factor_data()
        
        # 모델 스펙
        model_spec = create_comprehensive_model_spec()
        
        # 포괄적 설정
        config = PathAnalysisConfig(
            include_bootstrap_ci=True,
            bootstrap_samples=50,  # 빠른 테스트
            mediation_bootstrap_samples=50,
            bootstrap_method='non-parametric',
            bootstrap_percentile_method='bias_corrected',
            confidence_level=0.95,
            all_possible_mediations=True,
            analyze_all_paths=True,
            bootstrap_progress_bar=True
        )
        
        print(f"PathAnalyzer 설정:")
        print(f"  부트스트래핑: {config.include_bootstrap_ci}")
        print(f"  모든 매개효과 분석: {config.all_possible_mediations}")
        print(f"  모든 경로 분석: {config.analyze_all_paths}")
        
        # 분석 실행
        analyzer = PathAnalyzer(config)
        results = analyzer.fit_model(model_spec, data)
        
        print("✅ PathAnalyzer 포괄적 분석 완료")
        
        # 결과 확인
        print(f"\n결과 키: {list(results.keys())}")
        
        # 부트스트래핑 결과
        bootstrap_effects = results.get('bootstrap_effects', {})
        print(f"\n부트스트래핑 결과: {len(bootstrap_effects)}개 조합")
        
        # 모든 매개효과 결과
        all_mediations = results.get('all_mediations', {})
        print(f"모든 매개효과 결과: {type(all_mediations)}")
        
        if isinstance(all_mediations, dict):
            if 'all_results' in all_mediations:
                all_results = all_mediations['all_results']
                print(f"  전체 매개효과 조합: {len(all_results)}개")
            
            if 'significant_results' in all_mediations:
                significant_results = all_mediations['significant_results']
                print(f"  유의한 매개효과: {len(significant_results)}개")
            
            if 'summary' in all_mediations:
                summary = all_mediations['summary']
                print(f"  요약 정보: {summary}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ PathAnalyzer 포괄적 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def save_comprehensive_results(effects_results, pathanalyzer_results, data, model_spec):
    """포괄적 분석 결과 저장"""
    
    print("\n" + "=" * 60)
    print("포괄적 분석 결과 저장")
    print("=" * 60)
    
    try:
        # 결과 디렉토리 생성
        results_dir = "comprehensive_mediation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 상세 결과 저장 (텍스트)
        detailed_file = os.path.join(results_dir, f"comprehensive_mediation_{timestamp}.txt")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            f.write("5개 요인 포괄적 매개효과 분석 결과\n")
            f.write("=" * 60 + "\n")
            f.write(f"실행 시간: {timestamp}\n")
            f.write(f"데이터 크기: {data.shape}\n")
            f.write(f"분석 변수: {list(data.columns)}\n\n")
            
            f.write("모델 스펙:\n")
            f.write(model_spec + "\n\n")
            
            # EffectsCalculator 결과
            if effects_results:
                f.write("=" * 40 + "\n")
                f.write("EffectsCalculator 모든 매개효과 분석 결과\n")
                f.write("=" * 40 + "\n")
                
                if 'all_results' in effects_results:
                    all_results = effects_results['all_results']
                    f.write(f"전체 매개효과 조합: {len(all_results)}개\n\n")
                    
                    for combination_key, combination_result in all_results.items():
                        f.write(f"--- {combination_key} ---\n")
                        
                        if 'original_effects' in combination_result:
                            original = combination_result['original_effects']
                            f.write("원본 효과:\n")
                            for effect_name, effect_value in original.items():
                                if isinstance(effect_value, (int, float)):
                                    f.write(f"  {effect_name}: {effect_value:.6f}\n")
                        
                        if 'confidence_intervals' in combination_result:
                            ci = combination_result['confidence_intervals']
                            f.write("신뢰구간 (95%):\n")
                            for effect_name, ci_data in ci.items():
                                if isinstance(ci_data, dict):
                                    lower = ci_data.get('lower', 'N/A')
                                    upper = ci_data.get('upper', 'N/A')
                                    significant = ci_data.get('significant', False)
                                    f.write(f"  {effect_name}: [{lower:.6f}, {upper:.6f}] {'*' if significant else ''}\n")
                        
                        f.write("\n")
                
                if 'significant_results' in effects_results:
                    significant_results = effects_results['significant_results']
                    f.write(f"유의한 매개효과: {len(significant_results)}개\n")
                    for sig_key in significant_results.keys():
                        f.write(f"  - {sig_key}\n")
                    f.write("\n")
                
                if 'summary' in effects_results:
                    summary = effects_results['summary']
                    f.write("요약 통계:\n")
                    for key, value in summary.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            # PathAnalyzer 결과
            if pathanalyzer_results:
                f.write("=" * 40 + "\n")
                f.write("PathAnalyzer 포괄적 분석 결과\n")
                f.write("=" * 40 + "\n")
                
                f.write(f"결과 키: {list(pathanalyzer_results.keys())}\n\n")
                
                # 적합도 지수
                if 'fit_indices' in pathanalyzer_results:
                    fit_indices = pathanalyzer_results['fit_indices']
                    f.write("적합도 지수:\n")
                    for key, value in fit_indices.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # 부트스트래핑 결과
                bootstrap_effects = pathanalyzer_results.get('bootstrap_effects', {})
                if bootstrap_effects:
                    f.write(f"부트스트래핑 결과: {len(bootstrap_effects)}개 조합\n")
                    for combination_key, combination_result in bootstrap_effects.items():
                        f.write(f"  - {combination_key}\n")
                    f.write("\n")
                
                # 모든 매개효과 결과
                all_mediations = pathanalyzer_results.get('all_mediations', {})
                if all_mediations:
                    f.write("모든 매개효과 분석:\n")
                    if isinstance(all_mediations, dict):
                        for key, value in all_mediations.items():
                            if isinstance(value, dict):
                                f.write(f"  {key}: {len(value)}개 항목\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                    f.write("\n")
        
        print(f"✅ 상세 결과 저장: {detailed_file}")
        
        # 2. 요약 결과 저장 (JSON)
        summary_file = os.path.join(results_dir, f"mediation_summary_{timestamp}.json")
        
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
        
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_for_json(obj)
        
        summary_data = {
            'timestamp': timestamp,
            'data_shape': data.shape,
            'variables': list(data.columns),
            'model_spec': model_spec,
            'effects_calculator_results': deep_convert(effects_results) if effects_results else None,
            'pathanalyzer_results_summary': {
                'result_keys': list(pathanalyzer_results.keys()) if pathanalyzer_results else [],
                'bootstrap_combinations': len(pathanalyzer_results.get('bootstrap_effects', {})) if pathanalyzer_results else 0,
                'has_all_mediations': 'all_mediations' in pathanalyzer_results if pathanalyzer_results else False
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 요약 결과 저장: {summary_file}")
        
        return True, results_dir
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("5개 요인 포괄적 매개효과 분석")
    
    # 1. EffectsCalculator로 모든 매개효과 조합 분석
    effects_success, effects_results, data, model_spec = analyze_all_mediation_combinations()
    
    # 2. PathAnalyzer로 포괄적 분석
    pathanalyzer_success, pathanalyzer_results = analyze_pathanalyzer_comprehensive()
    
    # 3. 결과 저장
    if effects_success or pathanalyzer_success:
        save_success, results_dir = save_comprehensive_results(
            effects_results, pathanalyzer_results, data, model_spec
        )
    else:
        save_success = False
        results_dir = None
    
    print(f"\n" + "=" * 60)
    print("최종 분석 결과")
    print("=" * 60)
    print(f"EffectsCalculator 모든 매개효과: {'✅ 성공' if effects_success else '❌ 실패'}")
    print(f"PathAnalyzer 포괄적 분석: {'✅ 성공' if pathanalyzer_success else '❌ 실패'}")
    print(f"결과 저장: {'✅ 성공' if save_success else '❌ 실패'}")
    
    if save_success and results_dir:
        print(f"\n📁 결과 저장 위치: {results_dir}")
    
    if effects_success and pathanalyzer_success:
        print(f"\n🎉 모든 분석 성공!")
        print("✅ 5개 요인 간 모든 가능한 매개효과가 분석되었습니다.")
        print("✅ 포괄적 부트스트래핑 신뢰구간이 계산되었습니다.")
        print("✅ 유의한 매개효과가 식별되었습니다.")
        print("✅ 결과가 상세히 저장되었습니다.")
        
        # 간단한 결과 요약 출력
        if effects_results and 'summary' in effects_results:
            summary = effects_results['summary']
            print(f"\n📊 분석 요약:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
    else:
        print(f"\n⚠️  일부 분석에서 문제가 발생했습니다.")
        if effects_success:
            print("✅ EffectsCalculator 모든 매개효과 분석은 성공했습니다.")
        if pathanalyzer_success:
            print("✅ PathAnalyzer 포괄적 분석은 성공했습니다.")
