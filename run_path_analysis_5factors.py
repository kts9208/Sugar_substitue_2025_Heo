#!/usr/bin/env python3
"""
5개 요인 경로분석 실행 스크립트

실제 설문 데이터를 사용하여 5개 요인 간의 경로분석을 수행합니다.
- health_concern (건강관심도): q6~q11
- perceived_benefit (지각된혜택): q16~q17  
- purchase_intention (구매의도): q18~q19
- perceived_price (지각된가격): q20~q21
- nutrition_knowledge (영양지식): q30~q49
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# 경로분석 모듈 임포트
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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_structure():
    """데이터 구조 확인"""
    print("=" * 60)
    print("데이터 구조 확인")
    print("=" * 60)
    
    data_dir = Path("processed_data/survey_data")
    factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 
               'perceived_price', 'nutrition_knowledge']
    
    for factor in factors:
        file_path = data_dir / f"{factor}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            items = [col for col in df.columns if col.startswith('q')]
            print(f"{factor}: {len(df)}행, 문항 {items}")
        else:
            print(f"{factor}: 파일 없음")


def run_simple_mediation_analysis():
    """단순 매개모델 분석: 건강관심도 -> 지각된혜택 -> 구매의도"""
    print("\n" + "=" * 60)
    print("1. 단순 매개모델 분석")
    print("건강관심도 -> 지각된혜택 -> 구매의도")
    print("=" * 60)
    
    try:
        # 1. 모델 스펙 생성
        model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        print("생성된 모델 스펙:")
        print(model_spec)
        
        # 2. 분석 실행
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        config = create_mediation_config(verbose=True)
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 3. 결과 출력
        print("\n=== 분석 결과 ===")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        
        if 'fit_indices' in results and results['fit_indices']:
            print("\n적합도 지수:")
            for index, value in results['fit_indices'].items():
                try:
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        print(f"  {index}: {value:.4f}")
                except (TypeError, ValueError):
                    print(f"  {index}: {value}")
        
        # 4. 효과 분석
        if 'model_object' in results:
            effects_calc = EffectsCalculator(results['model_object'])
            effects = effects_calc.calculate_all_effects(
                'health_concern', 'purchase_intention', ['perceived_benefit']
            )
            
            print("\n=== 효과 분석 ===")
            if 'direct_effects' in effects:
                direct = effects['direct_effects']['coefficient']
                direct_p = effects['direct_effects']['p_value']
                sig = "***" if direct_p < 0.001 else "**" if direct_p < 0.01 else "*" if direct_p < 0.05 else ""
                print(f"직접효과: {direct:.4f}{sig} (p = {direct_p:.3f})")
            
            if 'indirect_effects' in effects:
                indirect = effects['indirect_effects']['total_indirect_effect']
                print(f"간접효과: {indirect:.4f}")
            
            if 'total_effects' in effects:
                total = effects['total_effects']['total_effect']
                proportion = effects['total_effects']['proportion_mediated']
                print(f"총효과: {total:.4f}")
                print(f"매개비율: {proportion:.1%}")
            
            # 매개효과 분석
            if 'mediation_analysis' in effects:
                mediation = effects['mediation_analysis']
                if 'sobel_tests' in mediation:
                    for mediator, sobel_result in mediation['sobel_tests'].items():
                        z_score = sobel_result.get('z_score', 0)
                        p_value = sobel_result.get('p_value', 1)
                        print(f"\nSobel test ({mediator}):")
                        print(f"  Z = {z_score:.3f}, p = {p_value:.3f}")
        
        # 5. 결과 저장
        saved_files = export_path_results(results, filename_prefix="simple_mediation")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        logger.error(f"단순 매개모델 분석 오류: {e}")
        return None


def run_multiple_mediation_analysis():
    """다중 매개모델 분석: 건강관심도 -> [지각된혜택, 지각된가격] -> 구매의도"""
    print("\n" + "=" * 60)
    print("2. 다중 매개모델 분석")
    print("건강관심도 -> [지각된혜택, 지각된가격] -> 구매의도")
    print("=" * 60)
    
    try:
        # 1. 모델 스펙 생성
        model_spec = create_path_model(
            model_type='multiple_mediation',
            independent_var='health_concern',
            mediator_vars=['perceived_benefit', 'perceived_price'],
            dependent_var='purchase_intention',
            allow_mediator_correlations=True
        )
        
        print("다중 매개모델 스펙 생성 완료")
        
        # 2. 분석 실행
        variables = ['health_concern', 'perceived_benefit', 'perceived_price', 'purchase_intention']
        config = create_mediation_config(bootstrap_samples=2000)
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 3. 결과 출력
        print("\n=== 분석 결과 ===")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        
        if 'fit_indices' in results:
            print("\n적합도 지수:")
            for index, value in results['fit_indices'].items():
                try:
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        print(f"  {index}: {value:.4f}")
                except (TypeError, ValueError):
                    print(f"  {index}: {value}")
        
        # 4. 경로계수 출력
        if 'path_coefficients' in results:
            print("\n=== 경로계수 ===")
            path_coeffs = results['path_coefficients']
            if 'paths' in path_coeffs and 'coefficients' in path_coeffs:
                for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                    coeff = path_coeffs['coefficients'].get(i, 0)
                    p_val = path_coeffs.get('p_values', {}).get(i, 1)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"  {from_var} -> {to_var}: {coeff:.4f}{sig}")
        
        # 5. 다중 매개효과 분석
        if 'model_object' in results:
            effects_calc = EffectsCalculator(results['model_object'])
            effects = effects_calc.calculate_all_effects(
                'health_concern', 'purchase_intention', 
                ['perceived_benefit', 'perceived_price']
            )
            
            print("\n=== 다중 매개효과 분석 ===")
            if 'indirect_effects' in effects:
                indirect = effects['indirect_effects']
                print(f"총 간접효과: {indirect.get('total_indirect_effect', 0):.4f}")
                
                # 개별 매개효과
                for mediator, path_info in indirect.get('individual_paths', {}).items():
                    effect = path_info.get('indirect_effect', 0)
                    print(f"  {mediator}를 통한 간접효과: {effect:.4f}")
        
        # 6. 결과 저장
        saved_files = export_path_results(results, filename_prefix="multiple_mediation")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        logger.error(f"다중 매개모델 분석 오류: {e}")
        return None


def run_comprehensive_structural_model():
    """종합적인 구조모델: 5개 요인 모두 포함 (누락된 경로 없음)"""
    print("\n" + "=" * 60)
    print("3. 종합적인 구조모델 분석 (완전한 경로)")
    print("5개 요인 모두 포함한 복합 경로 모델 - 누락된 경로 없음")
    print("=" * 60)
    
    try:
        # 1. 5개 요인 모두 포함
        variables = ['health_concern', 'perceived_benefit', 'perceived_price',
                    'nutrition_knowledge', 'purchase_intention']

        print(f"분석 변수: {', '.join(variables)}")

        # 2. 포괄적 구조모델 생성 (누락된 경로 없음)
        model_spec = create_path_model(
            model_type='comprehensive',
            variables=variables,
            include_bidirectional=True,
            include_feedback=True
        )
        
        print("종합적인 구조모델 스펙 생성 완료")
        
        # 3. 분석 실행
        config = create_default_path_config(
            standardized=True,
            create_diagrams=True,
            verbose=True
        )
        
        results = analyze_path_model(model_spec, variables, config)
        
        # 4. 결과 출력
        print("\n=== 분석 결과 ===")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        print(f"변수 수: {results['model_info']['n_variables']}")
        
        if 'fit_indices' in results:
            print("\n적합도 지수:")
            for index, value in results['fit_indices'].items():
                try:
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        interpretation = interpret_fit_index(index, value)
                        print(f"  {index}: {value:.4f} ({interpretation})")
                except (TypeError, ValueError):
                    print(f"  {index}: {value}")
        
        # 5. 경로 분석 결과
        if 'path_analysis' in results:
            path_info = results['path_analysis']
            print(f"\n=== 경로 분석 ===")
            print(f"잠재변수 수: {path_info['n_latent_variables']}")
            print(f"가능한 경로 수: {path_info['n_possible_paths']}")
            print(f"현재 경로 수: {path_info['n_current_paths']}")
            print(f"누락된 경로 수: {path_info['n_missing_paths']}")
            print(f"경로 포함률: {path_info['coverage_ratio']:.1%}")

            if path_info['missing_paths']:
                print(f"\n누락된 경로들:")
                for i, (from_var, to_var) in enumerate(path_info['missing_paths'], 1):
                    print(f"  {i}. {from_var} → {to_var}")
            else:
                print("\n✅ 모든 가능한 경로가 포함되었습니다!")

        # 6. 주요 경로계수 출력
        print("\n=== 주요 경로계수 ===")
        if 'path_coefficients' in results:
            path_coeffs = results['path_coefficients']
            if 'paths' in path_coeffs and 'coefficients' in path_coeffs:
                # 구매의도에 대한 직접효과만 출력
                for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                    if to_var == 'purchase_intention':
                        coeff = path_coeffs['coefficients'].get(i, 0)
                        p_val = path_coeffs.get('p_values', {}).get(i, 1)
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        print(f"  {from_var} -> {to_var}: {coeff:.4f}{sig}")
        
        # 7. 결과 저장 및 시각화
        saved_files = export_path_results(results, filename_prefix="comprehensive_structural")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")
        
        # 시각화
        try:
            visualizer = PathAnalysisVisualizer()
            viz_files = visualizer.create_comprehensive_visualization(results, "comprehensive_model")
            print(f"시각화 완료: {len(viz_files)}개 파일")
        except Exception as e:
            print(f"시각화 오류: {e}")
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        logger.error(f"종합적인 구조모델 분석 오류: {e}")
        return None


def run_saturated_model_analysis():
    """포화 모델 분석: 모든 가능한 경로 포함"""
    print("\n" + "=" * 60)
    print("4. 포화 모델 분석 (모든 가능한 경로)")
    print("5개 요인 간 모든 가능한 경로를 포함한 완전 포화 모델")
    print("=" * 60)

    try:
        # 1. 5개 요인 모두 포함
        variables = ['health_concern', 'perceived_benefit', 'perceived_price',
                    'nutrition_knowledge', 'purchase_intention']

        print(f"분석 변수: {', '.join(variables)}")
        print(f"예상 경로 수: {len(variables) * (len(variables) - 1)} (모든 가능한 경로)")

        # 2. 포화 모델 생성
        model_spec = create_path_model(
            model_type='saturated',
            variables=variables
        )

        print("포화 모델 스펙 생성 완료")

        # 3. 분석 실행
        config = create_default_path_config(
            standardized=True,
            create_diagrams=True,
            verbose=True
        )

        results = analyze_path_model(model_spec, variables, config)

        # 4. 결과 출력
        print("\n=== 분석 결과 ===")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        print(f"변수 수: {results['model_info']['n_variables']}")

        if 'fit_indices' in results:
            print("\n적합도 지수:")
            for index, value in results['fit_indices'].items():
                try:
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        interpretation = interpret_fit_index(index, value)
                        print(f"  {index}: {value:.4f} ({interpretation})")
                except (TypeError, ValueError):
                    print(f"  {index}: {value}")

        # 5. 경로 분석 결과
        if 'path_analysis' in results:
            path_info = results['path_analysis']
            print(f"\n=== 경로 분석 ===")
            print(f"잠재변수 수: {path_info['n_latent_variables']}")
            print(f"가능한 경로 수: {path_info['n_possible_paths']}")
            print(f"현재 경로 수: {path_info['n_current_paths']}")
            print(f"누락된 경로 수: {path_info['n_missing_paths']}")
            print(f"경로 포함률: {path_info['coverage_ratio']:.1%}")

            if path_info['missing_paths']:
                print(f"\n누락된 경로들:")
                for i, (from_var, to_var) in enumerate(path_info['missing_paths'], 1):
                    print(f"  {i}. {from_var} → {to_var}")
            else:
                print("\n✅ 모든 가능한 경로가 포함되었습니다!")

        # 6. 결과 저장
        saved_files = export_path_results(results, filename_prefix="saturated_model")
        print(f"\n결과 저장 완료: {len(saved_files)}개 파일")

        return results

    except Exception as e:
        print(f"오류 발생: {e}")
        logger.error(f"포화 모델 분석 오류: {e}")
        return None


def interpret_fit_index(index_name: str, value: float) -> str:
    """적합도 지수 해석"""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    
    if index_name == 'cfi' or index_name == 'tli':
        if value > 0.95:
            return "우수"
        elif value > 0.90:
            return "수용가능"
        else:
            return "부족"
    elif index_name == 'rmsea':
        if value < 0.06:
            return "우수"
        elif value < 0.08:
            return "수용가능"
        else:
            return "부족"
    elif index_name == 'srmr':
        if value < 0.08:
            return "우수"
        elif value < 0.10:
            return "수용가능"
        else:
            return "부족"
    elif index_name == 'p_value':
        if value > 0.05:
            return "좋음 (비유의적)"
        else:
            return "나쁨 (유의적)"
    
    return ""


def main():
    """메인 실행 함수"""
    print("🔍 5개 요인 경로분석 실행")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 구조 확인
    check_data_structure()
    
    # 분석 실행
    results = {}
    
    # 1. 단순 매개모델
    simple_results = run_simple_mediation_analysis()
    if simple_results:
        results['simple_mediation'] = simple_results
    
    # 2. 다중 매개모델
    multiple_results = run_multiple_mediation_analysis()
    if multiple_results:
        results['multiple_mediation'] = multiple_results
    
    # 3. 종합적인 구조모델 (누락된 경로 없음)
    comprehensive_results = run_comprehensive_structural_model()
    if comprehensive_results:
        results['comprehensive_structural'] = comprehensive_results

    # 4. 포화 모델 (모든 가능한 경로)
    saturated_results = run_saturated_model_analysis()
    if saturated_results:
        results['saturated_model'] = saturated_results
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 분석 완료 요약")
    print("=" * 60)
    print(f"성공한 분석: {len(results)}개")
    for analysis_type in results.keys():
        print(f"  ✅ {analysis_type}")
    
    print(f"\n결과 파일들이 'path_analysis_results' 디렉토리에 저장되었습니다.")
    print("분석 완료!")


if __name__ == "__main__":
    main()
