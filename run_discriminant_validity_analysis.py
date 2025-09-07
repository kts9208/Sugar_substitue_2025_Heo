"""
판별타당도 분석 실행 스크립트

이 스크립트는 상관관계 분석 결과와 신뢰도 분석 결과를 불러와서
판별타당도를 검증하고 결과를 시각화합니다.

사용법:
    python run_discriminant_validity_analysis.py

기능:
1. 최신 상관관계 분석 결과 자동 로드
2. 최신 신뢰도 분석 결과 자동 로드
3. Fornell-Larcker 기준으로 판별타당도 검증
4. 결과 시각화 및 보고서 생성
"""

import sys
from pathlib import Path
from discriminant_validity_analyzer import DiscriminantValidityAnalyzer

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("판별타당도 분석 (Discriminant Validity Analysis)")
    print("=" * 80)
    print()
    
    try:
        # 분석기 초기화
        print("1. 분석기 초기화 중...")
        analyzer = DiscriminantValidityAnalyzer()
        
        # 파일 경로 확인
        print(f"   상관관계 데이터: {analyzer.correlation_file}")
        print(f"   신뢰도 데이터: {analyzer.reliability_file}")
        print()
        
        if not analyzer.correlation_file:
            print("❌ 상관관계 분석 결과 파일을 찾을 수 없습니다.")
            print("   factor_correlations_results/ 디렉토리에 semopy_correlations_*.csv 파일이 있는지 확인하세요.")
            return False
            
        if not analyzer.reliability_file:
            print("❌ 신뢰도 분석 결과 파일을 찾을 수 없습니다.")
            print("   integrated_reliability_results_*/ 디렉토리에 reliability_summary_table.csv 파일이 있는지 확인하세요.")
            return False
        
        # 전체 분석 실행
        print("2. 판별타당도 분석 실행 중...")
        success = analyzer.run_complete_analysis()
        
        if success:
            print()
            print("✅ 판별타당도 분석이 성공적으로 완료되었습니다!")
            print()
            print("생성된 파일들:")
            print("📊 시각화 파일:")
            print("   - correlation_vs_ave_comparison.png: 상관계수 vs AVE 제곱근 비교")
            print("   - discriminant_validity_matrix.png: 판별타당도 검증 결과 매트릭스")
            print("   - discriminant_validity_dashboard.png: 종합 대시보드")
            if analyzer.discriminant_validity_results['violations']:
                print("   - discriminant_validity_violations.png: 위반 사항 시각화")
            print()
            print("📄 보고서 및 데이터:")
            print("   - discriminant_validity_report_*.txt: 상세 분석 보고서")
            print("   - discriminant_validity_results_*.csv: 검증 결과 데이터")
            print("   - correlation_ave_comparison_matrix_*.csv: 비교 매트릭스")
            print()
            
            # 결과 요약 출력
            summary = analyzer.discriminant_validity_results['summary']
            print("📈 분석 결과 요약:")
            print(f"   전체 요인 쌍: {summary['total_factor_pairs']}개")
            print(f"   유효한 쌍: {summary['valid_pairs']}개")
            print(f"   위반 쌍: {summary['invalid_pairs']}개")
            print(f"   유효율: {summary['validity_rate']:.1%}")
            
            if summary['overall_discriminant_validity']:
                print("   🎉 전체 판별타당도: 달성 ✅")
                print("      모든 요인이 Fornell-Larcker 기준을 만족합니다.")
            else:
                print("   ⚠️  전체 판별타당도: 미달성 ❌")
                print("      일부 요인 쌍이 판별타당도 기준을 위반했습니다.")
                print("      상세한 내용은 보고서를 확인하세요.")
            
            return True
            
        else:
            print("❌ 분석 중 오류가 발생했습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_requirements():
    """필요한 패키지 확인"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 다음 패키지들이 설치되어 있지 않습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    # 패키지 요구사항 확인
    if not check_requirements():
        sys.exit(1)
    
    # 메인 분석 실행
    success = main()
    
    if success:
        print("\n🎯 분석 완료! discriminant_validity_results/ 디렉토리에서 결과를 확인하세요.")
        sys.exit(0)
    else:
        print("\n💥 분석 실패!")
        sys.exit(1)
