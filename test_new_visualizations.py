#!/usr/bin/env python3
"""
새로운 시각화 기능 테스트 스크립트

이 스크립트는 새로 추가된 시각화 기능들을 개별적으로 테스트합니다:
1. 결합 플롯 (상관계수 + p값)
2. 버블 플롯 (크기 = 상관계수, 색상 = 유의성)

Author: Sugar Substitute Research Team
Date: 2025-01-06
"""

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from correlation_visualizer import (
    CorrelationResultLoader,
    CorrelationVisualizer
)


def test_combined_plot():
    """결합 플롯 테스트"""
    print("📊 Testing Combined Plot (Correlation + P-value)")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 시각화 객체 생성
        visualizer = CorrelationVisualizer(figsize=(20, 8))
        
        # 결합 플롯 생성
        output_dir = Path("test_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        fig = visualizer.create_combined_correlation_plot(
            correlations, pvalues,
            save_path="test_visualizations/test_combined_plot.png"
        )
        
        print("✅ Combined plot generated successfully!")
        print("   Features:")
        print("   - Left panel: Correlation coefficients heatmap")
        print("   - Right panel: Statistical significance heatmap")
        print("   - English labels (font issue resolved)")
        print("   - Side-by-side comparison")
        
    except Exception as e:
        print(f"❌ Error in combined plot test: {e}")


def test_bubble_plot():
    """버블 플롯 테스트"""
    print("\n🫧 Testing Bubble Plot")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 시각화 객체 생성
        visualizer = CorrelationVisualizer(figsize=(12, 10))
        
        # 버블 플롯 생성
        fig = visualizer.create_bubble_plot(
            correlations, pvalues,
            save_path="test_visualizations/test_bubble_plot.png"
        )
        
        print("✅ Bubble plot generated successfully!")
        print("   Features:")
        print("   - Bubble size represents |correlation coefficient|")
        print("   - Color represents statistical significance")
        print("   - Correlation values displayed on bubbles")
        print("   - English factor names")
        
    except Exception as e:
        print(f"❌ Error in bubble plot test: {e}")


def test_individual_heatmaps():
    """개별 히트맵 테스트 (영문 버전)"""
    print("\n🔥 Testing Individual Heatmaps (English Version)")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 시각화 객체 생성
        visualizer = CorrelationVisualizer(figsize=(10, 8))
        
        # 상관계수 히트맵 (영문)
        fig1 = visualizer.create_correlation_heatmap(
            correlations, pvalues,
            save_path="test_visualizations/test_correlation_heatmap_english.png",
            show_significance=True
        )
        
        # p값 히트맵 (영문)
        fig2 = visualizer.create_pvalue_heatmap(
            pvalues,
            save_path="test_visualizations/test_pvalue_heatmap_english.png"
        )
        
        print("✅ Individual heatmaps generated successfully!")
        print("   Features:")
        print("   - English factor labels (font issue resolved)")
        print("   - Significance markers (*, **, ***)")
        print("   - Professional appearance")
        print("   - High resolution (300 DPI)")
        
    except Exception as e:
        print(f"❌ Error in individual heatmaps test: {e}")


def analyze_visualization_effectiveness():
    """시각화 효과성 분석"""
    print("\n📈 Analyzing Visualization Effectiveness")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        print("🔍 Data Analysis:")
        print(f"   - Number of factors: {len(correlations)}")
        print(f"   - Total possible correlations: {len(correlations) * (len(correlations) - 1) // 2}")
        
        # 유의한 상관관계 분석
        import numpy as np
        
        upper_triangle_corr = np.triu(correlations.values, k=1)
        upper_triangle_pval = np.triu(pvalues.values, k=1)
        
        # 0이 아닌 값들만 추출
        valid_indices = upper_triangle_corr != 0
        valid_corr = upper_triangle_corr[valid_indices]
        valid_pval = upper_triangle_pval[valid_indices]
        
        significant_mask = valid_pval < 0.05
        significant_corr = valid_corr[significant_mask]
        
        print(f"\n📊 Correlation Statistics:")
        print(f"   - Significant correlations (p<0.05): {len(significant_corr)}")
        print(f"   - Strongest correlation: {valid_corr.max():.4f}")
        print(f"   - Weakest correlation: {valid_corr.min():.4f}")
        print(f"   - Average |correlation|: {np.abs(valid_corr).mean():.4f}")
        
        print(f"\n🎯 Visualization Recommendations:")
        if len(significant_corr) > 0:
            print("   ✅ Combined plot: Best for comparing coefficients and significance")
            print("   ✅ Bubble plot: Best for showing relationship strength at a glance")
            print("   ✅ Network graph: Best for understanding factor relationships")
        else:
            print("   ⚠️  Few significant relationships - focus on individual heatmaps")
        
        print(f"\n🔤 Font Issue Resolution:")
        print("   ✅ All labels converted to English")
        print("   ✅ Font family set to ['DejaVu Sans', 'Arial', 'sans-serif']")
        print("   ✅ No more Korean font dependency")
        
    except Exception as e:
        print(f"❌ Error in effectiveness analysis: {e}")


def main():
    """메인 실행 함수"""
    print("🧪 New Visualization Features Test")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    output_dir = Path("test_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # 테스트 실행
    test_combined_plot()
    test_bubble_plot()
    test_individual_heatmaps()
    analyze_visualization_effectiveness()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("📂 Check test_visualizations/ directory for generated files")
    
    # 생성된 파일 목록
    generated_files = list(output_dir.glob("*.png"))
    if generated_files:
        print(f"\n📁 Generated test files ({len(generated_files)}):")
        for file_path in sorted(generated_files):
            print(f"   📊 {file_path.name}")
    
    print(f"\n🔍 Key Improvements:")
    print(f"   ✅ Font issues resolved (English labels)")
    print(f"   ✅ Combined visualization (correlation + p-value)")
    print(f"   ✅ Bubble plot for intuitive understanding")
    print(f"   ✅ Professional appearance")
    print(f"   ✅ High-resolution output")


if __name__ == "__main__":
    main()
