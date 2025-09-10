#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
원본 데이터(6개 문항 perceived_benefit, 3개 문항 purchase_intention)로 분석 실행
"""

import sys
sys.path.append('.')

def run_factor_analysis():
    """요인분석 실행"""
    print('=== 요인분석 실행 시작 ===')
    
    try:
        from factor_analysis import analyze_factor_loading, export_factor_results
        import pandas as pd
        from datetime import datetime
        
        # 분석할 5개 요인
        target_factors = [
            'health_concern',        # 건강관심도
            'perceived_benefit',     # 지각된 유익성 (6개 문항)
            'purchase_intention',    # 구매의도 (3개 문항)
            'perceived_price',       # 지각된 가격
            'nutrition_knowledge'    # 영양지식
        ]
        
        print(f'분석 시작 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'분석할 요인들: {target_factors}')
        print()
        
        # 다중 요인 분석 실행
        print('다중 요인 분석 실행 중...')
        results = analyze_factor_loading(target_factors)
        
        print('분석 완료!')
        print(f'   - 샘플 크기: {results["model_info"]["n_observations"]}')
        print(f'   - 변수 수: {results["model_info"]["n_variables"]}')
        print(f'   - 분석 유형: {results["analysis_type"]}')
        print()
        
        # Factor loadings 결과 출력
        loadings = results.get('factor_loadings', pd.DataFrame())
        if len(loadings) > 0:
            print('Factor Loadings 결과:')
            print('-' * 60)
            
            # 요인별로 그룹화하여 출력
            for factor in target_factors:
                factor_loadings = loadings[loadings['Factor'] == factor]
                if len(factor_loadings) > 0:
                    print(f'\n{factor.upper().replace("_", " ")}:')
                    sig_count = factor_loadings['Significant'].sum() if 'Significant' in factor_loadings.columns else 0
                    avg_loading = factor_loadings['Loading'].mean()
                    print(f'   문항 수: {len(factor_loadings)}, 유의한 loadings: {sig_count}, 평균 loading: {avg_loading:.3f}')
                    
                    # 각 문항의 loading 출력
                    for _, row in factor_loadings.iterrows():
                        sig_mark = '***' if row.get('Significant', False) else ''
                        print(f'     {row["Item"]}: {row["Loading"]:.3f} {sig_mark}')
        
        # 모델 적합도 출력
        fit_indices = results.get('fit_indices', {})
        if fit_indices:
            print('\n모델 적합도:')
            print('-' * 40)
            for key, value in fit_indices.items():
                if isinstance(value, (int, float)):
                    print(f'   {key}: {value:.3f}')
                else:
                    print(f'   {key}: {value}')
        
        # 결과 저장
        print('\n결과 저장 중...')
        saved_files = export_factor_results(results, comprehensive=True)
        print(f'{len(saved_files)}개 파일 저장 완료:')
        for file_type, file_path in saved_files.items():
            print(f'   - {file_type}: {file_path.name}')
        
        print(f'\n요인분석 완료! ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
        return True
        
    except Exception as e:
        print(f'요인분석 실행 중 오류: {e}')
        import traceback
        traceback.print_exc()
        return False

def run_reliability_analysis():
    """신뢰성 분석 실행"""
    print('\n=== 신뢰성 분석 실행 시작 ===')
    
    try:
        from factor_analysis import analyze_factor_loading, ReliabilityCalculator
        import pandas as pd
        
        # 분석 대상 요인 정의
        target_factors = [
            'health_concern',      # 건강관심도
            'perceived_benefit',   # 지각된 유익성 (6개 문항)
            'purchase_intention',  # 구매의도 (3개 문항)
            'perceived_price',     # 지각된 가격
            'nutrition_knowledge'  # 영양지식
        ]
        
        print(f'분석 대상 요인: {len(target_factors)}개')
        for i, factor in enumerate(target_factors, 1):
            print(f'   {i}. {factor}')
        
        # 5개 요인 종합 분석
        print(f'\n5개 요인 종합 분석 실행...')
        results = analyze_factor_loading(target_factors)
        print("분석 완료!")
        
        # 기본 정보 출력
        model_info = results['model_info']
        print(f"샘플 크기: {model_info['n_observations']}명")
        print(f"변수 수: {model_info['n_variables']}개")
        
        # 신뢰도 계산기 초기화
        calculator = ReliabilityCalculator()
        
        # 각 요인별 신뢰도 분석
        print('\n요인별 신뢰도 분석:')
        print('-' * 80)
        
        reliability_summary = []
        
        for factor in target_factors:
            try:
                # 요인별 데이터 로드
                from factor_analysis import FactorDataLoader
                loader = FactorDataLoader()
                factor_data = loader.load_single_factor(factor)
                
                # 신뢰도 계산
                reliability = calculator.calculate_reliability(factor_data, factor)
                
                print(f'\n{factor.upper().replace("_", " ")}:')
                print(f'   문항 수: {reliability["n_items"]}')
                print(f'   Cronbach\'s Alpha: {reliability["cronbach_alpha"]:.3f}')
                print(f'   Composite Reliability (CR): {reliability["composite_reliability"]:.3f}')
                print(f'   Average Variance Extracted (AVE): {reliability["ave"]:.3f}')
                
                # 신뢰도 평가
                alpha = reliability["cronbach_alpha"]
                if alpha >= 0.9:
                    alpha_eval = "우수"
                elif alpha >= 0.8:
                    alpha_eval = "양호"
                elif alpha >= 0.7:
                    alpha_eval = "수용가능"
                else:
                    alpha_eval = "개선필요"
                
                print(f'   신뢰도 평가: {alpha_eval}')
                
                reliability_summary.append({
                    'Factor': factor,
                    'Items': reliability["n_items"],
                    'Cronbach_Alpha': reliability["cronbach_alpha"],
                    'CR': reliability["composite_reliability"],
                    'AVE': reliability["ave"],
                    'Evaluation': alpha_eval
                })
                
            except Exception as e:
                print(f'   {factor} 분석 실패: {e}')
        
        # 요약 테이블 출력
        if reliability_summary:
            print('\n신뢰도 분석 요약:')
            print('-' * 80)
            summary_df = pd.DataFrame(reliability_summary)
            print(summary_df.to_string(index=False, float_format='%.3f'))
        
        print('\n신뢰성 분석 완료!')
        return True
        
    except Exception as e:
        print(f'신뢰성 분석 실행 중 오류: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print('원본 데이터 분석 실행')
    print('=' * 60)
    
    # 1. 요인분석 실행
    factor_success = run_factor_analysis()
    
    # 2. 신뢰성 분석 실행
    reliability_success = run_reliability_analysis()
    
    print('\n' + '=' * 60)
    if factor_success and reliability_success:
        print('모든 분석이 성공적으로 완료되었습니다!')
    else:
        print('일부 분석에서 오류가 발생했습니다.')
