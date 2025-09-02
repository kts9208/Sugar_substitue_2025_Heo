"""
신뢰도 및 타당도 분석 실행 스크립트

이 스크립트는 5개 요인에 대해 다음을 계산합니다:
- Cronbach's Alpha (크론바흐 알파)
- Composite Reliability (CR, 합성신뢰도)
- Average Variance Extracted (AVE, 평균분산추출)
- 판별타당도 검증
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append('.')

from factor_analysis import analyze_factor_loading, ReliabilityCalculator
import pandas as pd
import numpy as np


def main():
    """메인 실행 함수"""
    print("🔍 === 신뢰도 및 타당도 분석 시스템 실행 ===")
    
    # 1. 분석 대상 요인 정의
    target_factors = [
        'health_concern',      # 건강관심도
        'perceived_benefit',   # 지각된 유익성
        'purchase_intention',  # 구매의도
        'perceived_price',     # 지각된 가격
        'nutrition_knowledge'  # 영양지식
    ]
    
    print(f"\n📋 분석 대상 요인: {len(target_factors)}개")
    for i, factor in enumerate(target_factors, 1):
        print(f"   {i}. {factor}")
    
    # 2. 5개 요인 종합 분석
    print(f"\n📊 5개 요인 종합 분석 실행...")
    try:
        results = analyze_factor_loading(target_factors)
        print("   ✅ 분석 완료!")
        
        # 기본 정보 출력
        model_info = results['model_info']
        print(f"   👥 샘플 크기: {model_info['n_observations']}명")
        print(f"   🔢 변수 수: {model_info['n_variables']}개")
        
    except Exception as e:
        print(f"   ❌ 분석 실패: {e}")
        return False
    
    # 3. 신뢰도 통계 분석
    print(f"\n🔍 신뢰도 및 타당도 분석...")
    
    if 'reliability_stats' not in results:
        print("   ❌ 신뢰도 통계를 찾을 수 없습니다.")
        return False
    
    reliability = results['reliability_stats']
    
    # 3.1 요약 테이블 출력
    if 'summary_table' in reliability:
        summary = reliability['summary_table']
        print("\n📈 === 신뢰도 요약 테이블 ===")
        
        if not summary.empty:
            # 주요 컬럼만 선택하여 출력
            display_cols = ['Factor', 'Items', 'Cronbach_Alpha', 'Composite_Reliability', 'AVE']
            available_cols = [col for col in display_cols if col in summary.columns]
            
            print(summary[available_cols].round(4).to_string(index=False))
            
            # 신뢰도 기준 충족 여부
            print("\n📊 === 신뢰도 기준 충족 여부 ===")
            criteria_cols = ['Factor', 'Alpha_Acceptable', 'CR_Acceptable', 'AVE_Acceptable']
            available_criteria = [col for col in criteria_cols if col in summary.columns]
            
            if len(available_criteria) > 1:
                criteria_df = summary[available_criteria].copy()
                criteria_df['Alpha_Acceptable'] = criteria_df['Alpha_Acceptable'].map({True: '✅', False: '❌'})
                criteria_df['CR_Acceptable'] = criteria_df['CR_Acceptable'].map({True: '✅', False: '❌'})
                criteria_df['AVE_Acceptable'] = criteria_df['AVE_Acceptable'].map({True: '✅', False: '❌'})
                print(criteria_df.to_string(index=False))
        
        # 3.2 상세 해석
        print("\n📝 === 신뢰도 상세 해석 ===")
        
        for _, row in summary.iterrows():
            factor = row['Factor']
            alpha = row['Cronbach_Alpha']
            cr = row['Composite_Reliability']
            ave = row['AVE']
            items = row['Items']
            
            print(f"\n🔹 {factor} ({items}개 문항):")
            
            # Cronbach's Alpha 해석
            if alpha >= 0.9:
                alpha_level = "우수 (Excellent)"
            elif alpha >= 0.8:
                alpha_level = "양호 (Good)"
            elif alpha >= 0.7:
                alpha_level = "보통 (Acceptable)"
            else:
                alpha_level = "부족 (Poor)"
            
            print(f"   📊 Cronbach's Alpha: {alpha:.3f} - {alpha_level}")
            
            # Composite Reliability 해석
            if cr >= 0.9:
                cr_level = "우수 (Excellent)"
            elif cr >= 0.8:
                cr_level = "양호 (Good)"
            elif cr >= 0.7:
                cr_level = "보통 (Acceptable)"
            else:
                cr_level = "부족 (Poor)"
            
            print(f"   🔗 Composite Reliability: {cr:.3f} - {cr_level}")
            
            # AVE 해석
            if ave >= 0.7:
                ave_level = "우수 (Excellent)"
            elif ave >= 0.6:
                ave_level = "양호 (Good)"
            elif ave >= 0.5:
                ave_level = "보통 (Acceptable)"
            else:
                ave_level = "부족 (Poor)"
            
            print(f"   📈 AVE: {ave:.3f} - {ave_level}")
    
    # 4. 적합도 지수 출력
    if 'fit_indices' in results:
        fit_indices = results['fit_indices']
        print(f"\n📏 === 모델 적합도 지수 ===")
        
        fit_display = {
            'CFI': fit_indices.get('CFI', 'N/A'),
            'TLI': fit_indices.get('TLI', 'N/A'),
            'RMSEA': fit_indices.get('RMSEA', 'N/A'),
            'SRMR': fit_indices.get('SRMR', 'N/A'),
            'GFI': fit_indices.get('GFI', 'N/A'),
            'AGFI': fit_indices.get('AGFI', 'N/A')
        }
        
        for index, value in fit_display.items():
            if isinstance(value, (int, float)):
                print(f"   {index}: {value:.3f}")
            else:
                print(f"   {index}: {value}")
    
    # 5. 결과 저장
    output_dir = Path('reliability_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 === 결과 저장 ===")
    
    try:
        # 신뢰도 요약 테이블 저장
        if 'summary_table' in reliability and not reliability['summary_table'].empty:
            summary_file = output_dir / 'reliability_summary.csv'
            reliability['summary_table'].to_csv(summary_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ 신뢰도 요약: {summary_file}")
        
        # Factor loadings 저장
        if 'factor_loadings' in results:
            loadings_file = output_dir / 'factor_loadings.csv'
            results['factor_loadings'].to_csv(loadings_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ Factor loadings: {loadings_file}")
        
        # 적합도 지수 저장
        if 'fit_indices' in results:
            fit_file = output_dir / 'fit_indices.csv'
            fit_df = pd.DataFrame([results['fit_indices']])
            fit_df.to_csv(fit_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ 적합도 지수: {fit_file}")
        
    except Exception as e:
        print(f"   ⚠️  파일 저장 중 오류: {e}")
    
    # 6. 종합 평가
    print(f"\n🎯 === 종합 평가 ===")
    
    if 'summary_table' in reliability and not reliability['summary_table'].empty:
        summary = reliability['summary_table']
        
        # 전체 요인의 신뢰도 평가
        excellent_factors = len(summary[summary['Cronbach_Alpha'] >= 0.9])
        good_factors = len(summary[(summary['Cronbach_Alpha'] >= 0.8) & (summary['Cronbach_Alpha'] < 0.9)])
        acceptable_factors = len(summary[(summary['Cronbach_Alpha'] >= 0.7) & (summary['Cronbach_Alpha'] < 0.8)])
        poor_factors = len(summary[summary['Cronbach_Alpha'] < 0.7])
        
        print(f"📊 Cronbach's Alpha 분포:")
        print(f"   🌟 우수 (≥0.9): {excellent_factors}개 요인")
        print(f"   ✅ 양호 (0.8-0.9): {good_factors}개 요인")
        print(f"   ⚠️  보통 (0.7-0.8): {acceptable_factors}개 요인")
        print(f"   ❌ 부족 (<0.7): {poor_factors}개 요인")
        
        # AVE 기준 충족 요인 수
        ave_acceptable = len(summary[summary['AVE'] >= 0.5])
        print(f"\n📈 AVE 기준 충족: {ave_acceptable}/{len(summary)}개 요인")
        
        # 전체 평가
        if excellent_factors >= 3 and poor_factors == 0:
            overall = "🌟 우수한 신뢰도"
        elif good_factors + excellent_factors >= 4 and poor_factors == 0:
            overall = "✅ 양호한 신뢰도"
        elif acceptable_factors + good_factors + excellent_factors >= 4:
            overall = "⚠️  보통 신뢰도"
        else:
            overall = "❌ 신뢰도 개선 필요"
        
        print(f"\n🎯 전체 평가: {overall}")
    
    print(f"\n🎉 === 신뢰도 및 타당도 분석 완료! ===")
    print(f"📁 결과 파일 위치: {output_dir}/")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ 신뢰도 분석이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n💥 신뢰도 분석 실행 중 오류가 발생했습니다!")
        sys.exit(1)
