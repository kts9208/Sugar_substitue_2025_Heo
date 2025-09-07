#!/usr/bin/env python3
"""
correlation_visualizer.py 모듈 사용 예제

이 스크립트는 correlation_visualizer.py 모듈의 다양한 사용법을 보여줍니다.
모듈의 재사용성과 확장성을 확인할 수 있습니다.

Author: Sugar Substitute Research Team
Date: 2025-01-06
"""

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from correlation_visualizer import (
    CorrelationResultLoader,
    CorrelationVisualizer,
    NetworkVisualizer,
    IntegratedVisualizer
)


def example_1_basic_usage():
    """예제 1: 기본 사용법"""
    print("📋 예제 1: 기본 사용법")
    print("-" * 40)
    
    try:
        # 로더 생성 및 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        print(f"✅ 데이터 로드 성공: {correlations.shape}")
        
        # 시각화 객체 생성
        visualizer = CorrelationVisualizer(figsize=(10, 8))
        
        # 상관계수 히트맵 생성
        fig = visualizer.create_correlation_heatmap(
            correlations, pvalues,
            save_path="example_outputs/basic_heatmap.png",
            show_significance=True
        )
        
        print("✅ 기본 히트맵 생성 완료")
        
    except Exception as e:
        print(f"❌ 예제 1 실행 중 오류: {e}")


def example_2_custom_visualization():
    """예제 2: 커스텀 시각화"""
    print("\n📋 예제 2: 커스텀 시각화")
    print("-" * 40)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 커스텀 설정으로 시각화
        visualizer = CorrelationVisualizer(figsize=(14, 12), style='darkgrid')
        
        # 값 표시 없이 히트맵 생성
        fig = visualizer.create_correlation_heatmap(
            correlations, pvalues,
            save_path="example_outputs/custom_heatmap.png",
            show_values=False,
            show_significance=True
        )
        
        # p값만 별도로 시각화
        fig2 = visualizer.create_pvalue_heatmap(
            pvalues,
            save_path="example_outputs/custom_pvalue_heatmap.png"
        )
        
        print("✅ 커스텀 시각화 생성 완료")
        
    except Exception as e:
        print(f"❌ 예제 2 실행 중 오류: {e}")


def example_3_network_visualization():
    """예제 3: 네트워크 시각화"""
    print("\n📋 예제 3: 네트워크 시각화")
    print("-" * 40)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 네트워크 시각화 객체 생성
        network_viz = NetworkVisualizer(figsize=(12, 10))
        
        # 다양한 임계값으로 네트워크 그래프 생성
        thresholds = [0.05, 0.1, 0.2]
        
        for threshold in thresholds:
            fig = network_viz.create_network_graph(
                correlations, pvalues,
                threshold=threshold,
                save_path=f"example_outputs/network_threshold_{threshold:.2f}.png"
            )
            
            if fig is not None:
                print(f"✅ 네트워크 그래프 생성 완료 (임계값: {threshold})")
        
    except Exception as e:
        print(f"❌ 예제 3 실행 중 오류: {e}")


def example_4_integrated_analysis():
    """예제 4: 통합 분석"""
    print("\n📋 예제 4: 통합 분석")
    print("-" * 40)
    
    try:
        # 통합 시각화 객체 생성
        integrated_viz = IntegratedVisualizer()
        
        # 요약 통계 출력
        integrated_viz.show_summary_statistics()
        
        # 종합 보고서 생성
        generated_files = integrated_viz.create_comprehensive_report(
            output_dir="example_outputs/integrated_report"
        )
        
        print(f"✅ 통합 분석 완료: {len(generated_files)}개 파일 생성")
        
    except Exception as e:
        print(f"❌ 예제 4 실행 중 오류: {e}")


def example_5_programmatic_access():
    """예제 5: 프로그래밍적 접근"""
    print("\n📋 예제 5: 프로그래밍적 접근")
    print("-" * 40)
    
    try:
        # 데이터 로드
        loader = CorrelationResultLoader()
        data = loader.load_correlation_data()
        
        correlations = data['correlations']
        pvalues = data['pvalues']
        
        # 프로그래밍적으로 데이터 분석
        print("📊 상관계수 분석:")
        
        # 가장 강한 상관관계 찾기
        import numpy as np
        
        # 상삼각 행렬에서 최대값 찾기
        upper_triangle = np.triu(correlations.values, k=1)
        max_idx = np.unravel_index(np.argmax(np.abs(upper_triangle)), upper_triangle.shape)
        
        factor1 = correlations.index[max_idx[0]]
        factor2 = correlations.columns[max_idx[1]]
        max_corr = correlations.iloc[max_idx[0], max_idx[1]]
        max_pval = pvalues.iloc[max_idx[0], max_idx[1]]
        
        print(f"  가장 강한 상관관계: {factor1} ↔ {factor2}")
        print(f"  상관계수: {max_corr:.4f}")
        print(f"  p값: {max_pval:.6f}")
        
        # 유의한 상관관계 개수
        significant_count = (pvalues.values < 0.05).sum() // 2  # 대칭 행렬이므로 2로 나눔
        total_pairs = len(correlations) * (len(correlations) - 1) // 2
        
        print(f"  유의한 상관관계: {significant_count}/{total_pairs}개")
        print(f"  유의성 비율: {significant_count/total_pairs*100:.1f}%")
        
        print("✅ 프로그래밍적 분석 완료")
        
    except Exception as e:
        print(f"❌ 예제 5 실행 중 오류: {e}")


def main():
    """메인 실행 함수"""
    print("🎨 correlation_visualizer.py 모듈 사용 예제")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 예제들 실행
    example_1_basic_usage()
    example_2_custom_visualization()
    example_3_network_visualization()
    example_4_integrated_analysis()
    example_5_programmatic_access()
    
    print("\n" + "=" * 60)
    print("🎉 모든 예제 실행 완료!")
    print("📂 생성된 파일들을 example_outputs/ 디렉토리에서 확인하세요.")
    
    # 모듈의 특징 요약
    print("\n🔍 모듈의 주요 특징:")
    print("  ✅ 완전한 독립성: 기존 모듈과 완전히 분리")
    print("  ✅ 높은 재사용성: 클래스 기반 모듈화 설계")
    print("  ✅ 뛰어난 확장성: 새로운 시각화 기능 쉽게 추가")
    print("  ✅ 유지보수성: 명확한 클래스 분리와 문서화")
    print("  ✅ 사용 편의성: 간단한 API와 자동 파일 탐지")


if __name__ == "__main__":
    main()
