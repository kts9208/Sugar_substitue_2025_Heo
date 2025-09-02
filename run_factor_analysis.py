#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5개 요인 Factor Loading 분석 실행 스크립트
"""

import sys
sys.path.append('.')
from factor_analysis import analyze_factor_loading, export_factor_results
import pandas as pd
from datetime import datetime

def main():
    print('=' * 80)
    print('5개 요인 Factor Loading 분석 실행')
    print('=' * 80)
    
    # 분석할 5개 요인
    target_factors = [
        'health_concern',        # 건강관심도
        'perceived_benefit',     # 지각된 유익성
        'purchase_intention',    # 구매의도
        'perceived_price',       # 지각된 가격
        'nutrition_knowledge'    # 영양지식
    ]
    
    print(f'분석 시작 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'분석할 요인들: {target_factors}')
    print()
    
    try:
        # 다중 요인 분석 실행
        print('🔄 다중 요인 분석 실행 중...')
        results = analyze_factor_loading(target_factors)
        
        print('✅ 분석 완료!')
        print(f'   - 샘플 크기: {results["model_info"]["n_observations"]}')
        print(f'   - 변수 수: {results["model_info"]["n_variables"]}')
        print(f'   - 분석 유형: {results["analysis_type"]}')
        print()
        
        # Factor loadings 결과 출력
        loadings = results.get('factor_loadings', pd.DataFrame())
        if len(loadings) > 0:
            print('📊 Factor Loadings 결과:')
            print('-' * 60)
            
            # 요인별로 그룹화하여 출력
            for factor in target_factors:
                factor_loadings = loadings[loadings['Factor'] == factor]
                if len(factor_loadings) > 0:
                    print(f'\n🔹 {factor.upper().replace("_", " ")}:')
                    sig_count = factor_loadings['Significant'].sum() if 'Significant' in factor_loadings.columns else 0
                    avg_loading = factor_loadings['Loading'].mean()
                    print(f'   문항 수: {len(factor_loadings)}, 유의한 loadings: {sig_count}, 평균 loading: {avg_loading:.3f}')
                    
                    # 각 문항의 loading 출력 (상위 10개만)
                    display_loadings = factor_loadings.head(10) if len(factor_loadings) > 10 else factor_loadings
                    
                    for _, row in display_loadings.iterrows():
                        loading_val = row['Loading']
                        item_name = row['Item']
                        if 'P_value' in row and pd.notna(row['P_value']):
                            p_val = row['P_value']
                            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                            print(f'     {item_name}: {loading_val:.3f} (p={p_val:.3f}){sig}')
                        else:
                            print(f'     {item_name}: {loading_val:.3f}')
                    
                    if len(factor_loadings) > 10:
                        print(f'     ... (총 {len(factor_loadings)}개 문항)')
        
        # 적합도 지수 출력
        fit_indices = results.get('fit_indices', {})
        if fit_indices:
            print('\n📈 모델 적합도 지수:')
            print('-' * 40)
            for index, value in fit_indices.items():
                # 적합도 해석
                if index in ['CFI', 'TLI']:
                    interpretation = 'Excellent' if value >= 0.95 else 'Good' if value >= 0.90 else 'Poor'
                elif index == 'RMSEA':
                    interpretation = 'Excellent' if value <= 0.05 else 'Good' if value <= 0.08 else 'Poor'
                elif index == 'SRMR':
                    interpretation = 'Excellent' if value <= 0.05 else 'Good' if value <= 0.08 else 'Poor'
                else:
                    interpretation = ''
                
                print(f'   {index}: {value:.4f} ({interpretation})')
        
        # 결과 저장
        print('\n💾 결과 저장 중...')
        saved_files = export_factor_results(results, comprehensive=True)
        print(f'✅ {len(saved_files)}개 파일 저장 완료:')
        for file_type, file_path in saved_files.items():
            print(f'   - {file_type}: {file_path.name}')
        
        # 요약 통계
        print('\n📋 분석 요약:')
        print('-' * 40)
        total_items = len(loadings) if len(loadings) > 0 else 0
        total_significant = loadings['Significant'].sum() if len(loadings) > 0 and 'Significant' in loadings.columns else 0
        print(f'   총 문항 수: {total_items}')
        print(f'   유의한 loadings: {total_significant}')
        print(f'   유의성 비율: {(total_significant/total_items*100):.1f}%' if total_items > 0 else '   유의성 비율: N/A')
        
        print(f'\n🎉 전체 분석 완료! ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
        
    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
