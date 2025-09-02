"""
5개 요인 분석 결과 가시화 실행 스크립트

이 스크립트는 구축한 semopy 가시화 시스템을 사용하여
5개 요인(건강관심도, 지각된 유익성, 구매의도, 지각된 가격, 영양지식)의
분석 결과를 종합적으로 가시화합니다.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행

# 프로젝트 루트 경로 추가
sys.path.append('.')

from factor_analysis import analyze_factor_loading, visualize_factor_analysis


def main():
    """메인 실행 함수"""
    print("🚀 === 5개 요인 분석 가시화 시스템 실행 ===")
    
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
    
    # 2. Factor Analysis 실행
    print("\n📊 Factor Analysis 실행 중...")
    try:
        results = analyze_factor_loading(target_factors)
        print("   ✅ 분석 완료!")
        
        # 분석 결과 요약
        loadings_df = results['factor_loadings']
        model_info = results['model_info']
        fit_indices = results['fit_indices']
        
        print(f"   📈 Factor loadings: {len(loadings_df)}개")
        print(f"   👥 샘플 크기: {model_info['n_observations']}명")
        print(f"   🔢 변수 수: {model_info['n_variables']}개")
        
    except Exception as e:
        print(f"   ❌ 분석 실패: {e}")
        return False
    
    # 3. 가시화 실행
    output_dir = 'factor_analysis_visualization_results'
    print(f"\n🎨 가시화 실행 중... (출력 폴더: {output_dir})")
    
    try:
        viz_results = visualize_factor_analysis(
            results, 
            output_dir=output_dir,
            show_plots=False  # 파일로만 저장
        )
        
        print("   ✅ 가시화 완료!")
        print(f"   📊 생성된 그래프: {len(viz_results['plots_generated'])}개")
        print(f"   📋 생성된 다이어그램: {len(viz_results['diagrams_generated'])}개")
        
        if viz_results['errors']:
            print(f"   ⚠️ 오류: {len(viz_results['errors'])}개")
            for error in viz_results['errors']:
                print(f"      - {error}")
        
    except Exception as e:
        print(f"   ❌ 가시화 실패: {e}")
        return False
    
    # 4. 결과 요약 출력
    print_analysis_summary(results, target_factors)
    
    # 5. 생성된 파일 확인
    print_generated_files(output_dir, viz_results)
    
    print("\n🎉 === 5개 요인 분석 가시화 완료! ===")
    print(f"📁 결과 파일 위치: {output_dir}/")
    print("🔍 각 파일을 열어서 시각화 결과를 확인하세요!")
    
    return True


def print_analysis_summary(results, target_factors):
    """분석 결과 요약 출력"""
    print("\n📈 === 분석 결과 요약 ===")
    
    loadings_df = results['factor_loadings']
    fit_indices = results['fit_indices']
    
    # 요인별 상세 정보
    print("\n🔹 요인별 상세 분석:")
    for factor in target_factors:
        factor_data = loadings_df[loadings_df['Factor'] == factor]
        
        # 유의한 loading 개수 (고정값 제외)
        non_fixed_data = factor_data[factor_data['Loading'] != 1.0]
        significant_count = len(non_fixed_data)
        
        # 평균 loading (절댓값)
        avg_loading = non_fixed_data['Loading'].abs().mean() if len(non_fixed_data) > 0 else 0
        
        # 음수 loading 개수
        negative_count = len(factor_data[factor_data['Loading'] < 0])
        
        # 강한 loading (≥0.7) 개수
        strong_count = len(factor_data[factor_data['Loading'].abs() >= 0.7])
        
        print(f"\n   📊 {factor.upper()}:")
        print(f"      총 문항: {len(factor_data)}개")
        print(f"      추정 문항: {significant_count}개")
        print(f"      평균 loading: {avg_loading:.3f}")
        print(f"      강한 loading (≥0.7): {strong_count}개")
        if negative_count > 0:
            print(f"      역방향 문항: {negative_count}개")
    
    # 모델 적합도
    print("\n📏 모델 적합도 지수:")
    fit_criteria = {
        'CFI': {'excellent': 0.95, 'good': 0.90, 'acceptable': 0.85},
        'TLI': {'excellent': 0.95, 'good': 0.90, 'acceptable': 0.85},
        'RMSEA': {'excellent': 0.05, 'good': 0.08, 'acceptable': 0.10},
        'SRMR': {'excellent': 0.05, 'good': 0.08, 'acceptable': 0.10}
    }
    
    for index, value in fit_indices.items():
        if pd.notna(value) and index in fit_criteria:
            criteria = fit_criteria[index]
            
            if index in ['CFI', 'TLI']:  # 높을수록 좋음
                if value >= criteria['excellent']:
                    status = '우수'
                elif value >= criteria['good']:
                    status = '양호'
                elif value >= criteria['acceptable']:
                    status = '수용가능'
                else:
                    status = '불량'
            else:  # 낮을수록 좋음 (RMSEA, SRMR)
                if value <= criteria['excellent']:
                    status = '우수'
                elif value <= criteria['good']:
                    status = '양호'
                elif value <= criteria['acceptable']:
                    status = '수용가능'
                else:
                    status = '불량'
            
            print(f"   {index}: {value:.3f} ({status})")


def print_generated_files(output_dir, viz_results):
    """생성된 파일 정보 출력"""
    print("\n📁 === 생성된 가시화 파일 ===")
    
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        print(f"\n📂 {output_dir}/ 폴더에 {len(files)}개 파일 생성:")
        
        for file in files:
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path)
            
            # 파일 유형별 설명
            if 'heatmap' in file:
                desc = "전체 요인 히트맵"
            elif 'fit_indices' in file:
                desc = "모델 적합도 그래프"
            elif 'diagram' in file:
                desc = "요인 구조 다이어그램"
            elif any(factor in file for factor in ['health_concern', 'perceived_benefit', 'purchase_intention', 'perceived_price', 'nutrition_knowledge']):
                factor_name = next(factor for factor in ['health_concern', 'perceived_benefit', 'purchase_intention', 'perceived_price', 'nutrition_knowledge'] if factor in file)
                desc = f"{factor_name} 막대 그래프"
            else:
                desc = "기타 파일"
            
            print(f"   📄 {file} ({size:,} bytes) - {desc}")
    else:
        print(f"   ❌ {output_dir} 폴더가 생성되지 않았습니다.")
    
    # 파일 유형별 설명
    print("\n📖 파일 설명:")
    print("   🔥 factor_loadings_heatmap.png - 모든 요인의 loading을 색상으로 표현한 히트맵")
    print("   📊 factor_loadings_[요인명].png - 각 요인별 상세 loading 막대 그래프")
    print("   📏 model_fit_indices.png - 모델 적합도 지수 시각화")
    print("   📋 factor_model_diagram.txt - 요인 구조를 텍스트로 표현한 다이어그램")


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ 프로그램이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n💥 프로그램 실행 중 오류가 발생했습니다!")
        sys.exit(1)
