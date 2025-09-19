"""
판별타당도 분석 모듈 테스트 스크립트

이 스크립트는 판별타당도 분석 모듈의 기능을 테스트합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from discriminant_validity_analyzer import DiscriminantValidityAnalyzer

def test_discriminant_validity_analyzer():
    """판별타당도 분석기 테스트"""
    print("=" * 60)
    print("판별타당도 분석기 테스트")
    print("=" * 60)
    
    try:
        # 1. 분석기 초기화 테스트
        print("1. 분석기 초기화 테스트...")
        analyzer = DiscriminantValidityAnalyzer()
        
        if analyzer.correlation_file and analyzer.reliability_file:
            print("✅ 자동 파일 찾기 성공")
            print(f"   상관관계 파일: {analyzer.correlation_file}")
            print(f"   신뢰도 파일: {analyzer.reliability_file}")
        else:
            print("❌ 필요한 파일을 찾을 수 없습니다.")
            return False
        
        # 2. 데이터 로드 테스트
        print("\n2. 데이터 로드 테스트...")
        success = analyzer.load_data()
        if success:
            print("✅ 데이터 로드 성공")
            print(f"   상관관계 매트릭스 크기: {analyzer.correlation_matrix.shape}")
            print(f"   신뢰도 데이터 크기: {analyzer.reliability_data.shape}")
        else:
            print("❌ 데이터 로드 실패")
            return False
        
        # 3. AVE 제곱근 매트릭스 생성 테스트
        print("\n3. AVE 제곱근 매트릭스 생성 테스트...")
        ave_sqrt_matrix = analyzer.create_ave_sqrt_matrix()
        if ave_sqrt_matrix is not None:
            print("✅ AVE 제곱근 매트릭스 생성 성공")
            print("   대각선 값들:")
            for i, factor in enumerate(ave_sqrt_matrix.index):
                print(f"     {factor}: {ave_sqrt_matrix.iloc[i, i]:.4f}")
        else:
            print("❌ AVE 제곱근 매트릭스 생성 실패")
            return False
        
        # 4. 판별타당도 분석 테스트
        print("\n4. 판별타당도 분석 테스트...")
        results = analyzer.analyze_discriminant_validity()
        if results:
            print("✅ 판별타당도 분석 성공")
            summary = results['summary']
            print(f"   전체 검증 쌍: {summary['total_factor_pairs']}")
            print(f"   유효한 쌍: {summary['valid_pairs']}")
            print(f"   위반 쌍: {summary['invalid_pairs']}")
            print(f"   유효율: {summary['validity_rate']:.1%}")
        else:
            print("❌ 판별타당도 분석 실패")
            return False
        
        # 5. 비교 매트릭스 생성 테스트
        print("\n5. 비교 매트릭스 생성 테스트...")
        comparison_matrix = analyzer.create_comparison_matrix()
        if comparison_matrix is not None:
            print("✅ 비교 매트릭스 생성 성공")
            print("   매트릭스 크기:", comparison_matrix.shape)
        else:
            print("❌ 비교 매트릭스 생성 실패")
            return False
        
        # 6. 결과 검증
        print("\n6. 결과 검증...")
        
        # 위반 사항 확인
        violations = results['violations']
        if violations:
            print(f"⚠️  {len(violations)}개의 판별타당도 위반 발견:")
            for violation in violations:
                print(f"     {violation['factor1']} vs {violation['factor2']}: "
                      f"r={violation['correlation']:.3f} > √AVE={violation['min_ave_sqrt']:.3f}")
        else:
            print("✅ 모든 요인 쌍이 판별타당도 기준을 만족")
        
        # 전체 판별타당도 상태
        overall_validity = results['summary']['overall_discriminant_validity']
        if overall_validity:
            print("🎉 전체 판별타당도: 달성")
        else:
            print("⚠️  전체 판별타당도: 미달성")
        
        print("\n✅ 모든 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_file_outputs():
    """출력 파일 테스트"""
    print("\n" + "=" * 60)
    print("출력 파일 테스트")
    print("=" * 60)
    
    results_dir = Path("discriminant_validity_results")
    
    if not results_dir.exists():
        print("❌ 결과 디렉토리가 존재하지 않습니다.")
        return False
    
    # 필수 파일들 확인
    required_files = [
        "correlation_vs_ave_comparison.png",
        "discriminant_validity_matrix.png", 
        "discriminant_validity_dashboard.png"
    ]
    
    optional_files = [
        "discriminant_validity_violations.png"
    ]
    
    # 필수 파일 확인
    print("필수 시각화 파일 확인:")
    for file_name in required_files:
        file_path = results_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - 파일이 없습니다.")
    
    # 선택적 파일 확인
    print("\n선택적 시각화 파일 확인:")
    for file_name in optional_files:
        file_path = results_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"ℹ️  {file_name} - 위반 사항이 없어서 생성되지 않음")
    
    # 보고서 및 데이터 파일 확인
    print("\n보고서 및 데이터 파일 확인:")
    report_files = list(results_dir.glob("discriminant_validity_report_*.txt"))
    result_files = list(results_dir.glob("discriminant_validity_results_*.csv"))
    matrix_files = list(results_dir.glob("correlation_ave_comparison_matrix_*.csv"))
    
    if report_files:
        print(f"✅ 보고서 파일: {len(report_files)}개")
    else:
        print("❌ 보고서 파일이 없습니다.")
    
    if result_files:
        print(f"✅ 결과 데이터 파일: {len(result_files)}개")
    else:
        print("❌ 결과 데이터 파일이 없습니다.")
    
    if matrix_files:
        print(f"✅ 비교 매트릭스 파일: {len(matrix_files)}개")
    else:
        print("❌ 비교 매트릭스 파일이 없습니다.")
    
    return True

def main():
    """메인 테스트 함수"""
    print("판별타당도 분석 모듈 테스트 시작")
    
    # 1. 분석기 기능 테스트
    analyzer_test = test_discriminant_validity_analyzer()
    
    # 2. 출력 파일 테스트
    file_test = test_file_outputs()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    if analyzer_test and file_test:
        print("🎉 모든 테스트 통과!")
        print("판별타당도 분석 모듈이 정상적으로 작동합니다.")
        return True
    else:
        print("❌ 일부 테스트 실패")
        if not analyzer_test:
            print("   - 분석기 기능 테스트 실패")
        if not file_test:
            print("   - 출력 파일 테스트 실패")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        exit(0)
    else:
        exit(1)
