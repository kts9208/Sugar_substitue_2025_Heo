#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
독립적인 신뢰도 분석 모듈 테스트 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from factor_analysis.reliability_calculator import IndependentReliabilityCalculator
from factor_analysis.reliability_visualizer import ReliabilityVisualizer


def test_reliability_calculator():
    """신뢰도 계산기 테스트"""
    print("=" * 60)
    print("신뢰도 계산기 테스트")
    print("=" * 60)
    
    try:
        # 1. 계산기 초기화
        calculator = IndependentReliabilityCalculator()
        print("✓ 계산기 초기화 성공")
        
        # 2. 분석 결과 로드 테스트
        analysis_results = calculator.load_latest_analysis_results()
        if analysis_results:
            print("✓ 분석 결과 로드 성공")
            print(f"  - 요인 수: {len(analysis_results['metadata']['factor_names'])}")
            print(f"  - 문항 수: {len(analysis_results['loadings'])}")
        else:
            print("✗ 분석 결과 로드 실패")
            return False
        
        # 3. 설문 데이터 로드 테스트
        survey_data = calculator.load_survey_data()
        if survey_data:
            print("✓ 설문 데이터 로드 성공")
            print(f"  - 로드된 요인 수: {len(survey_data)}")
        else:
            print("✗ 설문 데이터 로드 실패")
        
        # 4. 완전한 신뢰도 분석 테스트
        reliability_results = calculator.run_complete_reliability_analysis()
        if 'error' not in reliability_results:
            print("✓ 완전한 신뢰도 분석 성공")
            
            # 결과 검증
            stats = reliability_results['reliability_stats']
            print(f"  - 분석된 요인 수: {len(stats)}")
            
            for factor_name, factor_stats in stats.items():
                alpha = factor_stats.get('cronbach_alpha', np.nan)
                cr = factor_stats.get('composite_reliability', np.nan)
                ave = factor_stats.get('ave', np.nan)
                
                print(f"  - {factor_name}: Alpha={alpha:.4f}, CR={cr:.4f}, AVE={ave:.4f}")
            
            return reliability_results
        else:
            print(f"✗ 완전한 신뢰도 분석 실패: {reliability_results['error']}")
            return False
            
    except Exception as e:
        print(f"✗ 테스트 중 오류: {e}")
        return False


def test_reliability_visualizer(reliability_results):
    """신뢰도 시각화 테스트"""
    print("\n" + "=" * 60)
    print("신뢰도 시각화 테스트")
    print("=" * 60)
    
    try:
        # 테스트용 출력 디렉토리
        test_output_dir = "test_reliability_results"
        
        # 1. 시각화 클래스 초기화
        visualizer = ReliabilityVisualizer(test_output_dir)
        print("✓ 시각화 클래스 초기화 성공")
        
        # 2. 요약 테이블 생성 테스트
        summary_table = visualizer.create_reliability_summary_table(reliability_results)
        if not summary_table.empty:
            print("✓ 요약 테이블 생성 성공")
            print(f"  - 테이블 크기: {summary_table.shape}")
        else:
            print("✗ 요약 테이블 생성 실패")
        
        # 3. 신뢰도 지표 차트 테스트
        visualizer.plot_reliability_indicators(reliability_results)
        print("✓ 신뢰도 지표 차트 생성 성공")
        
        # 4. 상관관계 히트맵 테스트
        visualizer.plot_correlation_heatmap(reliability_results)
        print("✓ 상관관계 히트맵 생성 성공")
        
        # 5. 판별타당도 분석 테스트
        visualizer.plot_discriminant_validity(reliability_results)
        print("✓ 판별타당도 분석 차트 생성 성공")
        
        # 6. 종합 보고서 테스트
        visualizer.create_comprehensive_report(reliability_results)
        print("✓ 종합 보고서 생성 성공")
        
        # 7. 생성된 파일들 확인
        output_path = Path(test_output_dir)
        if output_path.exists():
            files = list(output_path.glob("*"))
            print(f"✓ 생성된 파일 수: {len(files)}")
            for file_path in files:
                print(f"  - {file_path.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ 시각화 테스트 중 오류: {e}")
        return False


def test_data_validation(reliability_results):
    """데이터 검증 테스트"""
    print("\n" + "=" * 60)
    print("데이터 검증 테스트")
    print("=" * 60)
    
    try:
        stats = reliability_results['reliability_stats']
        
        # 1. 신뢰도 기준값 검증
        print("신뢰도 기준값 검증:")
        for factor_name, factor_stats in stats.items():
            alpha = factor_stats.get('cronbach_alpha', np.nan)
            cr = factor_stats.get('composite_reliability', np.nan)
            ave = factor_stats.get('ave', np.nan)
            
            alpha_ok = alpha >= 0.7 if not np.isnan(alpha) else False
            cr_ok = cr >= 0.7 if not np.isnan(cr) else False
            ave_ok = ave >= 0.5 if not np.isnan(ave) else False
            
            status = "✓" if all([alpha_ok, cr_ok, ave_ok]) else "⚠"
            print(f"  {status} {factor_name}: Alpha({alpha:.3f}), CR({cr:.3f}), AVE({ave:.3f})")
        
        # 2. 판별타당도 검증
        print("\n판별타당도 검증:")
        discriminant_validity = reliability_results.get('discriminant_validity', {})
        if discriminant_validity:
            valid_pairs = 0
            total_pairs = 0
            
            factors = list(discriminant_validity.keys())
            for i, factor1 in enumerate(factors):
                for j, factor2 in enumerate(factors):
                    if i < j:
                        total_pairs += 1
                        if discriminant_validity[factor1].get(factor2, False):
                            valid_pairs += 1
            
            print(f"  판별타당도 통과율: {valid_pairs}/{total_pairs} ({valid_pairs/total_pairs*100:.1f}%)")
        
        # 3. 상관관계 검증
        print("\n상관관계 검증:")
        correlations = reliability_results.get('correlations')
        if correlations is not None and not correlations.empty:
            # 대각선 제외한 상관계수들
            corr_values = []
            for i in range(len(correlations)):
                for j in range(len(correlations)):
                    if i != j:
                        val = correlations.iloc[i, j]
                        if not np.isnan(val):
                            corr_values.append(abs(val))
            
            if corr_values:
                print(f"  평균 상관계수: {np.mean(corr_values):.3f}")
                print(f"  최대 상관계수: {np.max(corr_values):.3f}")
                print(f"  최소 상관계수: {np.min(corr_values):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 데이터 검증 중 오류: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("독립적인 신뢰도 분석 모듈 테스트 시작")
    print("=" * 80)
    
    # 1. 신뢰도 계산기 테스트
    reliability_results = test_reliability_calculator()
    if not reliability_results:
        print("\n❌ 신뢰도 계산기 테스트 실패")
        return
    
    # 2. 시각화 테스트
    viz_success = test_reliability_visualizer(reliability_results)
    if not viz_success:
        print("\n❌ 시각화 테스트 실패")
        return
    
    # 3. 데이터 검증 테스트
    validation_success = test_data_validation(reliability_results)
    if not validation_success:
        print("\n❌ 데이터 검증 테스트 실패")
        return
    
    print("\n" + "=" * 80)
    print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
    print("=" * 80)


if __name__ == "__main__":
    main()
