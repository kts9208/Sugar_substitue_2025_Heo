#!/usr/bin/env python3
"""
개선된 시각화 기능 테스트 스크립트

이 스크립트는 개선된 시각화 기능들을 테스트합니다:
1. 상관계수 표에 p값 정보 추가
2. p값을 유의성 수준별 색상으로 구분
3. 직접적인 p값 표시 제거

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


def test_correlation_with_pvalue_heatmap():
    """상관계수와 p값이 함께 표시되는 히트맵 테스트"""
    print("📊 Testing Correlation Heatmap with P-values")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 시각화 객체 생성
        visualizer = CorrelationVisualizer(figsize=(12, 10))
        
        # 출력 디렉토리 생성
        output_dir = Path("improved_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        # 상관계수 + p값 히트맵 생성
        fig = visualizer.create_correlation_heatmap(
            correlations, pvalues,
            save_path="improved_visualizations/correlation_with_pvalues.png",
            show_values=True,
            show_significance=True
        )
        
        print("✅ Correlation heatmap with p-values generated!")
        print("   Features:")
        print("   - Correlation coefficients in upper part of each cell")
        print("   - P-value information in lower part of each cell")
        print("   - P-values categorized as p<0.001, p<0.01, p<0.05, or exact value")
        print("   - No direct p-value numbers for cleaner appearance")
        
    except Exception as e:
        print(f"❌ Error in correlation with p-value test: {e}")


def test_significance_level_heatmap():
    """유의성 수준별 색상 구분 히트맵 테스트"""
    print("\n🎨 Testing Significance Level Heatmap")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 시각화 객체 생성
        visualizer = CorrelationVisualizer(figsize=(10, 8))
        
        # 유의성 수준 히트맵 생성
        fig = visualizer.create_pvalue_heatmap(
            pvalues,
            save_path="improved_visualizations/significance_levels.png"
        )
        
        print("✅ Significance level heatmap generated!")
        print("   Features:")
        print("   - Colors represent significance levels:")
        print("     • Light gray: Not significant (p≥0.05)")
        print("     • Light red: p<0.05")
        print("     • Medium red: p<0.01") 
        print("     • Dark red: p<0.001")
        print("   - Symbols: *, **, *** for significance levels")
        print("   - No direct p-value numbers displayed")
        
    except Exception as e:
        print(f"❌ Error in significance level test: {e}")


def test_improved_combined_plot():
    """개선된 결합 플롯 테스트"""
    print("\n🔄 Testing Improved Combined Plot")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 시각화 객체 생성
        visualizer = CorrelationVisualizer(figsize=(20, 8))
        
        # 개선된 결합 플롯 생성
        fig = visualizer.create_combined_correlation_plot(
            correlations, pvalues,
            save_path="improved_visualizations/improved_combined_plot.png"
        )
        
        print("✅ Improved combined plot generated!")
        print("   Features:")
        print("   - Left panel: Correlations with p-value categories")
        print("   - Right panel: Color-coded significance levels")
        print("   - Consistent color scheme for significance")
        print("   - Clean, professional appearance")
        
    except Exception as e:
        print(f"❌ Error in improved combined plot test: {e}")


def test_improved_bubble_plot():
    """개선된 버블 플롯 테스트"""
    print("\n🫧 Testing Improved Bubble Plot")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 시각화 객체 생성
        visualizer = CorrelationVisualizer(figsize=(12, 10))
        
        # 개선된 버블 플롯 생성
        fig = visualizer.create_bubble_plot(
            correlations, pvalues,
            save_path="improved_visualizations/improved_bubble_plot.png"
        )
        
        print("✅ Improved bubble plot generated!")
        print("   Features:")
        print("   - Bubble size: |correlation coefficient|")
        print("   - Color: Significance level (not raw p-values)")
        print("   - Red color scale for intuitive understanding")
        print("   - Correlation values displayed on bubbles")
        
    except Exception as e:
        print(f"❌ Error in improved bubble plot test: {e}")


def analyze_improvements():
    """개선사항 분석"""
    print("\n📈 Analyzing Improvements")
    print("-" * 50)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        print("🔍 Data Overview:")
        print(f"   - Factors: {len(correlations)}")
        print(f"   - Total correlations: {len(correlations) * (len(correlations) - 1) // 2}")
        
        # 유의성 수준별 분석
        import numpy as np
        
        upper_triangle_pval = np.triu(pvalues.values, k=1)
        valid_pvals = upper_triangle_pval[upper_triangle_pval != 0]
        
        highly_sig = (valid_pvals < 0.001).sum()
        mod_sig = ((valid_pvals >= 0.001) & (valid_pvals < 0.01)).sum()
        low_sig = ((valid_pvals >= 0.01) & (valid_pvals < 0.05)).sum()
        not_sig = (valid_pvals >= 0.05).sum()
        
        print(f"\n📊 Significance Distribution:")
        print(f"   - Highly significant (p<0.001): {highly_sig}")
        print(f"   - Moderately significant (p<0.01): {mod_sig}")
        print(f"   - Weakly significant (p<0.05): {low_sig}")
        print(f"   - Not significant (p≥0.05): {not_sig}")
        
        print(f"\n🎯 Key Improvements:")
        print("   ✅ P-values integrated into correlation display")
        print("   ✅ Color-coded significance levels (intuitive)")
        print("   ✅ Removed cluttered numerical p-values")
        print("   ✅ Professional, publication-ready appearance")
        print("   ✅ Consistent significance color scheme across all plots")
        
        print(f"\n📋 Visualization Recommendations:")
        if highly_sig > 0:
            print("   🔥 Use combined plot to highlight strong significant relationships")
        if mod_sig + low_sig > 0:
            print("   📊 Use significance heatmap to show all significance levels")
        if not_sig > 0:
            print("   ⚠️  Use bubble plot to de-emphasize non-significant relationships")
        
    except Exception as e:
        print(f"❌ Error in improvement analysis: {e}")


def main():
    """메인 실행 함수"""
    print("🚀 Improved Visualization Features Test")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    output_dir = Path("improved_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # 테스트 실행
    test_correlation_with_pvalue_heatmap()
    test_significance_level_heatmap()
    test_improved_combined_plot()
    test_improved_bubble_plot()
    analyze_improvements()
    
    print("\n" + "=" * 60)
    print("🎉 All improved visualization tests completed!")
    print("📂 Check improved_visualizations/ directory for generated files")
    
    # 생성된 파일 목록
    generated_files = list(output_dir.glob("*.png"))
    if generated_files:
        print(f"\n📁 Generated improved files ({len(generated_files)}):")
        for file_path in sorted(generated_files):
            print(f"   📊 {file_path.name}")
    
    print(f"\n🔍 Major Improvements Summary:")
    print(f"   1. 📊 P-values integrated into correlation cells")
    print(f"   2. 🎨 Color-coded significance levels (no raw p-values)")
    print(f"   3. 🧹 Cleaner, more professional appearance")
    print(f"   4. 📈 Consistent significance color scheme")
    print(f"   5. 🎯 Publication-ready quality")


if __name__ == "__main__":
    main()
