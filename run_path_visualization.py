#!/usr/bin/env python3
"""
경로분석 가시화 실행 스크립트

기존 경로분석 결과를 불러와서 semopy를 이용한 가시화를 수행합니다.
"""

import json
import pandas as pd
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_existing_results():
    """기존 경로분석 결과 로드"""
    results_file = "path_analysis_results/comprehensive_structural_full_results_20250910_084833.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"✅ 기존 결과 로드 성공: {results_file}")
        logger.info(f"   관측치 수: {results['model_info']['n_observations']}")
        logger.info(f"   변수 수: {results['model_info']['n_variables']}")
        
        return results
    except FileNotFoundError:
        logger.error(f"❌ 결과 파일을 찾을 수 없습니다: {results_file}")
        return None
    except Exception as e:
        logger.error(f"❌ 결과 로드 오류: {e}")
        return None

def recreate_semopy_model(results):
    """기존 결과에서 semopy 모델 재생성"""
    try:
        from path_analysis import PathAnalyzer, create_default_path_config
        import semopy
        from semopy import Model
        
        # 모델 스펙 추출
        model_spec = results['model_info']['model_spec']
        logger.info("📋 모델 스펙:")
        print(model_spec)
        
        # 데이터 로드
        config = create_default_path_config(verbose=False)
        analyzer = PathAnalyzer(config)
        
        # 5개 요인 변수
        variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                    'nutrition_knowledge', 'purchase_intention']
        
        data = analyzer.load_data(variables)
        logger.info(f"✅ 데이터 로드 완료: {data.shape}")
        
        # semopy 모델 생성 및 적합
        model = Model(model_spec)
        model.fit(data)
        
        logger.info("✅ semopy 모델 재생성 완료")
        
        return model, data
        
    except Exception as e:
        logger.error(f"❌ 모델 재생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_visualization(model, results):
    """가시화 실행"""
    try:
        from path_analysis.visualizer import (
            create_path_diagram, 
            create_multiple_diagrams, 
            create_advanced_diagrams,
            visualize_path_analysis
        )
        
        logger.info("🎨 경로분석 가시화 시작")
        
        # 출력 디렉토리 설정
        output_dir = "path_analysis_results/visualizations"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. 기본 경로 다이어그램
        logger.info("1️⃣ 기본 경로 다이어그램 생성 중...")
        basic_diagram = create_path_diagram(
            model, 
            filename="comprehensive_path_diagram",
            output_dir=output_dir,
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='dot'
        )
        
        if basic_diagram:
            logger.info(f"   ✅ 기본 다이어그램: {basic_diagram}")
        
        # 2. 다양한 스타일의 다이어그램들 (5가지)
        logger.info("2️⃣ 다양한 스타일 다이어그램 생성 중...")
        multiple_diagrams = create_multiple_diagrams(
            model,
            base_filename="comprehensive_multiple",
            output_dir=output_dir
        )
        
        for diagram_type, path in multiple_diagrams.items():
            if path:
                logger.info(f"   ✅ {diagram_type}: {path}")
            else:
                logger.warning(f"   ❌ {diagram_type}: 생성 실패")
        
        # 3. 고급 다이어그램들 (6가지)
        logger.info("3️⃣ 고급 다이어그램 생성 중...")
        advanced_diagrams = create_advanced_diagrams(
            model,
            base_filename="comprehensive_advanced",
            output_dir=output_dir
        )
        
        for diagram_type, path in advanced_diagrams.items():
            if path:
                logger.info(f"   ✅ {diagram_type}: {path}")
            else:
                logger.warning(f"   ❌ {diagram_type}: 생성 실패")
        
        # 4. 구조적 경로만 표시하는 다이어그램
        logger.info("4️⃣ 구조적 경로 전용 다이어그램 생성 중...")
        structural_diagram = create_path_diagram(
            model,
            filename="comprehensive_structural_only",
            output_dir=output_dir,
            structural_only=True,
            plot_covs=False,
            plot_ests=True,
            std_ests=True,
            engine='dot'
        )
        
        if structural_diagram:
            logger.info(f"   ✅ 구조적 경로 다이어그램: {structural_diagram}")
        
        # 5. 종합 시각화 (모든 유형)
        logger.info("5️⃣ 종합 시각화 실행 중...")
        
        # 결과에 모델 객체 추가
        results_with_model = results.copy()
        results_with_model['model_object'] = model
        
        comprehensive_viz = visualize_path_analysis(
            results_with_model,
            base_filename="comprehensive_final",
            output_dir=output_dir
        )
        
        logger.info("✅ 종합 시각화 완료")
        
        return {
            'basic_diagram': basic_diagram,
            'multiple_diagrams': multiple_diagrams,
            'advanced_diagrams': advanced_diagrams,
            'structural_diagram': structural_diagram,
            'comprehensive_visualization': comprehensive_viz
        }
        
    except Exception as e:
        logger.error(f"❌ 가시화 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 실행 함수"""
    logger.info("🚀 경로분석 가시화 시작")
    logger.info("=" * 60)
    
    # 1. 기존 결과 로드
    results = load_existing_results()
    if not results:
        return
    
    # 2. semopy 모델 재생성
    model, data = recreate_semopy_model(results)
    if not model:
        return
    
    # 3. 가시화 실행
    viz_results = run_visualization(model, results)
    if not viz_results:
        return
    
    # 4. 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("🎯 가시화 결과 요약")
    logger.info("=" * 60)
    
    total_created = 0
    
    if viz_results['basic_diagram']:
        logger.info(f"✅ 기본 다이어그램: {viz_results['basic_diagram']}")
        total_created += 1
    
    if viz_results['structural_diagram']:
        logger.info(f"✅ 구조적 경로 다이어그램: {viz_results['structural_diagram']}")
        total_created += 1
    
    for diagram_type, path in viz_results['multiple_diagrams'].items():
        if path:
            logger.info(f"✅ {diagram_type}: {path}")
            total_created += 1
    
    for diagram_type, path in viz_results['advanced_diagrams'].items():
        if path:
            logger.info(f"✅ {diagram_type}: {path}")
            total_created += 1
    
    logger.info(f"\n🎉 총 {total_created}개의 가시화 파일이 생성되었습니다!")
    logger.info(f"📁 출력 디렉토리: path_analysis_results/visualizations")
    
    return viz_results

if __name__ == "__main__":
    main()
