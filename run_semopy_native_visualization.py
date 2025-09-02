"""
semopy 내장 가시화 시스템 실행 스크립트

이 스크립트는 semopy의 내장 가시화 기능(semplot)을 사용하여
5개 요인 분석 결과를 SEM 경로 다이어그램으로 가시화합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append('.')

from factor_analysis import (
    analyze_factor_loading, 
    create_diagrams_for_factors,
    SemopyNativeVisualizer,
    IntegratedSemopyVisualizer
)


def main():
    """메인 실행 함수"""
    print("🚀 === semopy 내장 가시화 시스템 실행 ===")
    
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
    
    # 2. graphviz 설치 확인
    print("\n🔧 의존성 확인...")
    try:
        import graphviz
        print("   ✅ graphviz 설치됨")
    except ImportError:
        print("   ❌ graphviz가 설치되지 않았습니다.")
        print("   📦 설치 명령: pip install graphviz")
        print("   ⚠️  시스템 레벨 graphviz도 필요할 수 있습니다.")
        return False
    
    # 3. 단일 요인 테스트
    print("\n🧪 단일 요인 테스트 (health_concern)...")
    try:
        single_results = create_diagrams_for_factors(
            'health_concern',
            output_dir='semopy_single_factor_test'
        )
        
        if isinstance(single_results, dict):
            # 'diagrams_generated' 키가 있는 경우와 없는 경우 모두 처리
            if 'diagrams_generated' in single_results:
                diagrams = single_results['diagrams_generated']
            else:
                diagrams = single_results

            successful = [name for name, path in diagrams.items() if path is not None and os.path.exists(path)]
            print(f"   ✅ 단일 요인 테스트 완료: {len(successful)}/{len(diagrams)}개 성공")

            if successful:
                print("   📊 생성된 다이어그램:")
                for name in successful:
                    print(f"      - {name}")

            if len(successful) == 0:
                print("   ❌ 단일 요인 테스트 실패")
                return False
        else:
            print("   ❌ 단일 요인 테스트 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 단일 요인 테스트 오류: {e}")
        return False
    
    # 4. 다중 요인 분석 및 가시화
    print(f"\n📊 5개 요인 분석 실행...")
    try:
        results = analyze_factor_loading(target_factors)
        print("   ✅ 분석 완료!")
        
        loadings_df = results['factor_loadings']
        model_info = results['model_info']
        
        print(f"   📈 Factor loadings: {len(loadings_df)}개")
        print(f"   👥 샘플 크기: {model_info['n_observations']}명")
        print(f"   🔢 변수 수: {model_info['n_variables']}개")
        
    except Exception as e:
        print(f"   ❌ 분석 실패: {e}")
        return False
    
    # 5. semopy 내장 가시화 실행
    output_dir = 'semopy_native_visualization_results'
    print(f"\n🎨 semopy 내장 가시화 실행... (출력 폴더: {output_dir})")
    
    try:
        visualizer = SemopyNativeVisualizer()
        
        # 분석 결과에서 모델 추출 시도
        model = visualizer._extract_model_from_results(results)
        
        if model is None:
            print("   ⚠️  분석 결과에서 모델을 추출할 수 없습니다.")
            print("   🔄 새 모델 생성 시도...")
            
            # 새 모델로 다이어그램 생성
            viz_results = create_diagrams_for_factors(
                target_factors,
                output_dir=output_dir
            )
        else:
            print("   ✅ 모델 추출 성공!")
            viz_results = visualizer.create_multiple_diagrams(
                model=model,
                base_filename="five_factors_model",
                output_dir=output_dir
            )
        
        # 결과 처리
        if isinstance(viz_results, dict):
            if 'diagrams_generated' in viz_results:
                diagrams = viz_results['diagrams_generated']
            else:
                diagrams = viz_results
            
            successful = [name for name, path in diagrams.items() if path is not None]
            failed = [name for name, path in diagrams.items() if path is None]
            
            print(f"   ✅ semopy 내장 가시화 완료!")
            print(f"   📊 성공: {len(successful)}개")
            print(f"   ❌ 실패: {len(failed)}개")
            
            if successful:
                print("\n   🎯 생성된 SEM 다이어그램:")
                for name in successful:
                    print(f"      - {name}")
            
            if failed:
                print("\n   ⚠️  실패한 다이어그램:")
                for name in failed:
                    print(f"      - {name}")
        
    except Exception as e:
        print(f"   ❌ semopy 내장 가시화 실패: {e}")
        return False
    
    # 6. 통합 가시화 테스트
    print(f"\n🌟 통합 가시화 테스트...")
    try:
        integrated_visualizer = IntegratedSemopyVisualizer()
        integrated_results = integrated_visualizer.create_comprehensive_visualization(
            results,
            output_dir='integrated_visualization_results'
        )
        
        summary = integrated_results.get('summary', {})
        print(f"   ✅ 통합 가시화 완료!")
        print(f"   📊 semopy 다이어그램: {summary.get('semopy_diagrams', 0)}개")
        print(f"   📈 커스텀 그래프: {summary.get('custom_plots', 0)}개")
        print(f"   🎯 총 가시화: {summary.get('total_visualizations', 0)}개")
        
        if integrated_results.get('errors'):
            print(f"   ⚠️  오류: {len(integrated_results['errors'])}개")
        
    except Exception as e:
        print(f"   ❌ 통합 가시화 실패: {e}")
    
    # 7. 생성된 파일 확인
    print_generated_files(output_dir)
    
    print("\n🎉 === semopy 내장 가시화 완료! ===")
    print(f"📁 주요 결과 파일 위치: {output_dir}/")
    print("🔍 생성된 .png, .pdf, .svg 파일들을 확인하세요!")
    
    return True


def print_generated_files(output_dir):
    """생성된 파일 정보 출력"""
    print(f"\n📁 === 생성된 파일 확인 ===")
    
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        if files:
            print(f"\n📂 {output_dir}/ 폴더에 {len(files)}개 파일 생성:")
            
            for file in files:
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                
                # 파일 유형별 설명
                if file.endswith('.png'):
                    desc = "PNG 이미지"
                elif file.endswith('.pdf'):
                    desc = "PDF 문서"
                elif file.endswith('.svg'):
                    desc = "SVG 벡터 이미지"
                elif file.endswith('.dot'):
                    desc = "Graphviz DOT 파일"
                else:
                    desc = "기타 파일"
                
                print(f"   📄 {file} ({size:,} bytes) - {desc}")
        else:
            print(f"   📂 {output_dir}/ 폴더가 비어있습니다.")
    else:
        print(f"   ❌ {output_dir} 폴더가 생성되지 않았습니다.")
    
    # 파일 유형별 설명
    print("\n📖 semopy 내장 가시화 파일 설명:")
    print("   🔹 *_basic.png - 기본 SEM 다이어그램 (표준화 추정값 포함)")
    print("   🔹 *_detailed.png - 상세 다이어그램 (공분산 포함)")
    print("   🔹 *_simple.png - 간단한 다이어그램 (추정값 없음)")
    print("   🔹 *_circular.png - 원형 레이아웃 다이어그램")
    print("   🔹 *_unstandardized.png - 비표준화 추정값 다이어그램")


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ semopy 내장 가시화가 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n💥 semopy 내장 가시화 실행 중 오류가 발생했습니다!")
        sys.exit(1)
