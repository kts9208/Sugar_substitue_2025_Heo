#!/usr/bin/env python3
"""
semopy 시각화 기능 테스트 스크립트

수정된 경로분석 모듈의 semopy 시각화 기능을 테스트합니다.
"""

import sys
from pathlib import Path
from datetime import datetime

# 경로분석 모듈 임포트
from path_analysis import (
    PathAnalyzer,
    analyze_path_model,
    create_path_model,
    create_default_path_config,
    create_path_diagram,
    create_multiple_diagrams,
    create_advanced_diagrams,
    visualize_path_analysis
)

def test_semopy_visualization():
    """semopy 시각화 기능 테스트"""
    print("🎨 semopy 시각화 기능 테스트")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 간단한 경로분석 모델 생성
        print("\n1. 경로분석 모델 생성 중...")
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        
        model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        print("✅ 모델 스펙 생성 완료")
        
        # 2. 모델 분석 실행
        print("\n2. 모델 분석 실행 중...")
        config = create_default_path_config(verbose=False)
        results = analyze_path_model(model_spec, variables, config)
        
        print(f"✅ 분석 완료: {results['model_info']['n_observations']}개 관측치")
        
        # 3. semopy 모델 객체 확인
        if 'model_object' not in results:
            print("❌ semopy 모델 객체를 찾을 수 없습니다.")
            return
        
        model = results['model_object']
        print("✅ semopy 모델 객체 확인 완료")
        
        # 4. 단일 다이어그램 생성 테스트
        print("\n3. 단일 다이어그램 생성 테스트...")
        try:
            diagram_path = create_path_diagram(
                model=model,
                filename="test_single_diagram",
                output_dir="test_visualization_results",
                plot_covs=True,
                plot_ests=True,
                std_ests=True
            )

            if diagram_path:
                print(f"✅ 단일 다이어그램 생성 성공: {diagram_path}")
            else:
                print("⚠️  단일 다이어그램 생성 실패 (Graphviz 미설치 가능성)")

        except Exception as e:
            print(f"❌ 단일 다이어그램 생성 오류: {e}")

        # 4-1. 경로계수만 표시하는 다이어그램 테스트
        print("\n3-1. 경로계수만 표시 다이어그램 테스트...")
        try:
            structural_path = create_path_diagram(
                model=model,
                filename="test_structural_only",
                output_dir="test_visualization_results",
                plot_covs=True,
                plot_ests=True,
                std_ests=True,
                structural_only=True  # 경로계수만 표시
            )

            if structural_path:
                print(f"✅ 경로계수만 표시 다이어그램 생성 성공: {structural_path}")
            else:
                print("⚠️  경로계수만 표시 다이어그램 생성 실패")

        except Exception as e:
            print(f"❌ 경로계수만 표시 다이어그램 생성 오류: {e}")
        
        # 5. 다중 다이어그램 생성 테스트
        print("\n4. 다중 다이어그램 생성 테스트...")
        try:
            diagrams = create_multiple_diagrams(
                model=model,
                base_filename="test_multiple",
                output_dir="test_visualization_results"
            )

            successful = sum(1 for path in diagrams.values() if path is not None)
            print(f"✅ 다중 다이어그램 생성: {successful}/{len(diagrams)} 성공")

            for diagram_type, path in diagrams.items():
                status = "✅" if path else "❌"
                special_note = " (경로계수만)" if diagram_type == "structural_only" else ""
                print(f"  {status} {diagram_type}{special_note}: {path or '실패'}")

        except Exception as e:
            print(f"❌ 다중 다이어그램 생성 오류: {e}")
        
        # 6. 고급 다이어그램 생성 테스트
        print("\n5. 고급 다이어그램 생성 테스트...")
        try:
            advanced_diagrams = create_advanced_diagrams(
                model=model,
                base_filename="test_advanced",
                output_dir="test_visualization_results"
            )

            successful = sum(1 for path in advanced_diagrams.values() if path is not None)
            print(f"✅ 고급 다이어그램 생성: {successful}/{len(advanced_diagrams)} 성공")

            for diagram_type, path in advanced_diagrams.items():
                status = "✅" if path else "❌"
                special_note = " (경로계수만)" if diagram_type == "structural_paths_only" else ""
                print(f"  {status} {diagram_type}{special_note}: {path or '실패'}")

        except Exception as e:
            print(f"❌ 고급 다이어그램 생성 오류: {e}")
        
        # 7. 종합 시각화 테스트
        print("\n6. 종합 시각화 테스트...")
        try:
            viz_results = visualize_path_analysis(
                results=results,
                base_filename="test_comprehensive",
                output_dir="test_visualization_results"
            )
            
            if viz_results.get('summary'):
                summary = viz_results['summary']
                print(f"✅ 종합 시각화 완료:")
                print(f"  - 기본 다이어그램: {summary.get('basic_diagrams', 0)}개")
                print(f"  - 고급 다이어그램: {summary.get('advanced_diagrams', 0)}개")
                print(f"  - 총 다이어그램: {summary.get('total_diagrams', 0)}개")
                print(f"  - 성공률: {summary.get('success_rate', '0%')}")
            
            if viz_results.get('errors'):
                print(f"  ⚠️  오류: {len(viz_results['errors'])}개")
                for error in viz_results['errors']:
                    print(f"    - {error}")
                    
        except Exception as e:
            print(f"❌ 종합 시각화 오류: {e}")
        
        # 8. 결과 요약
        print(f"\n🎉 semopy 시각화 테스트 완료!")
        print("=" * 60)
        
        # 생성된 파일 확인
        test_dir = Path("test_visualization_results")
        if test_dir.exists():
            files = list(test_dir.glob("*.png"))
            print(f"📁 생성된 파일: {len(files)}개")
            for file in files[:10]:  # 최대 10개만 표시
                print(f"  - {file.name}")
            if len(files) > 10:
                print(f"  ... 및 {len(files) - 10}개 더")
        else:
            print("📁 생성된 파일 없음 (Graphviz 미설치 가능성)")
        
        print("\n💡 참고사항:")
        print("- 다이어그램 생성 실패는 대부분 Graphviz 미설치 때문입니다.")
        print("- Windows: choco install graphviz")
        print("- macOS: brew install graphviz")
        print("- Ubuntu: sudo apt-get install graphviz")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_semopy_visualization()
