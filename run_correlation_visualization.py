#!/usr/bin/env python3
"""
semopy 상관계수 결과 시각화 실행 스크립트

이 스크립트는 다음 작업을 수행합니다:
1. semopy 상관계수 결과 파일 자동 탐지 및 로드
2. 상관계수 히트맵 생성
3. p값 시각화
4. 종합 시각화 보고서 생성

사용법:
    python run_correlation_visualization.py

Author: Sugar Substitute Research Team
Date: 2025-01-06
"""

import sys
from pathlib import Path
import argparse

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from correlation_visualizer import (
    CorrelationResultLoader,
    CorrelationVisualizer, 
    IntegratedVisualizer
)


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="semopy 상관계수 결과 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_correlation_visualization.py                    # 기본 실행
  python run_correlation_visualization.py --output-dir viz   # 출력 디렉토리 지정
  python run_correlation_visualization.py --no-show-stats    # 통계 요약 생략
        """
    )
    
    parser.add_argument(
        '--results-dir', 
        default='factor_correlations_results',
        help='결과 파일이 저장된 디렉토리 (기본값: factor_correlations_results)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='correlation_visualization_results', 
        help='시각화 결과를 저장할 디렉토리 (기본값: correlation_visualization_results)'
    )
    
    parser.add_argument(
        '--no-show-stats',
        action='store_true',
        help='요약 통계 출력 생략'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=int,
        default=[12, 10],
        help='그래프 크기 (가로 세로) (기본값: 12 10)'
    )
    
    return parser.parse_args()


def check_dependencies():
    """필요한 라이브러리 확인"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ 다음 라이브러리가 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n💡 설치 방법:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """메인 실행 함수"""
    print("🎨 semopy Correlation Results Visualization")
    print("="*60)
    
    # 명령행 인수 파싱
    args = parse_arguments()
    
    try:
        # 의존성 확인
        if not check_dependencies():
            return False
        
        # 결과 디렉토리 확인
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"❌ 결과 디렉토리를 찾을 수 없습니다: {results_dir}")
            print("💡 먼저 run_semopy_correlations.py를 실행하여 결과를 생성하세요.")
            return False
        
        # 통합 시각화 객체 생성
        visualizer = IntegratedVisualizer()
        visualizer.loader.results_dir = results_dir
        visualizer.visualizer.figsize = tuple(args.figsize)
        
        print(f"📂 Results directory: {results_dir}")
        print(f"📊 Output directory: {args.output_dir}")
        print(f"📏 Figure size: {args.figsize[0]} x {args.figsize[1]}")
        
        # 요약 통계 출력
        if not args.no_show_stats:
            visualizer.show_summary_statistics()
        
        # 종합 시각화 보고서 생성
        print(f"\n🎨 Generating visualizations...")
        generated_files = visualizer.create_comprehensive_report(args.output_dir)

        # 결과 요약
        print(f"\n" + "="*60)
        print("✅ Visualization Complete!")
        print("="*60)

        print(f"\n📁 Generated files ({len(generated_files)}개):")
        for description, file_path in generated_files.items():
            file_name = Path(file_path).name
            print(f"  📊 {description}: {file_name}")

        print(f"\n📂 Saved to: {args.output_dir}/")

        print(f"\n🎯 Next steps:")
        print(f"  1. Review the generated image files")
        print(f"  2. Identify strong relationships in correlation heatmap")
        print(f"  3. Check significant relationships in p-value heatmap")
        print(f"  4. Interpret results for research insights")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
        print("💡 해결방법:")
        print("  1. run_semopy_correlations.py를 먼저 실행하세요")
        print("  2. --results-dir 옵션으로 올바른 디렉토리를 지정하세요")
        return False
        
    except ImportError as e:
        print(f"❌ 라이브러리 오류: {e}")
        print("💡 해결방법: pip install pandas numpy matplotlib seaborn")
        return False
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_help():
    """도움말 출력"""
    help_text = """
🎨 semopy 상관계수 결과 시각화 도구

이 도구는 semopy_correlations.py에서 생성된 결과 파일들을 자동으로 찾아서
다양한 방식으로 시각화합니다.

📋 주요 기능:
  • 상관계수 히트맵 (유의성 마커 포함)
  • p값 히트맵 (-log10 변환)
  • 요약 통계 출력
  • 자동 파일 탐지 및 로드

🚀 사용법:
  python run_correlation_visualization.py [옵션]

📊 생성되는 시각화:
  • correlation_heatmap_YYYYMMDD_HHMMSS.png - 상관계수 히트맵
  • pvalue_heatmap_YYYYMMDD_HHMMSS.png - p값 히트맵

💡 팁:
  • 먼저 run_semopy_correlations.py를 실행하여 결과를 생성하세요
  • --help 옵션으로 상세한 옵션을 확인할 수 있습니다
  • 생성된 이미지는 고해상도(300 DPI)로 저장됩니다
    """
    print(help_text)


if __name__ == "__main__":
    # 도움말 요청 확인
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)
    
    success = main()
    
    if success:
        print("\n🎉 Correlation visualization completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 An error occurred during visualization.")
        sys.exit(1)
