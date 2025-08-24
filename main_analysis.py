"""
Multinomial Logit Model 메인 실행 스크립트

DCE 데이터를 사용하여 Multinomial Logit Model을 추정하고 결과를 분석합니다.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multinomial_logit_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 패키지 임포트
try:
    from multinomial_logit import (
        load_dce_data, get_dce_summary,
        preprocess_dce_data, get_preprocessing_summary,
        create_default_config, estimate_multinomial_logit,
        analyze_results, create_quick_report
    )
except ImportError as e:
    logger.error(f"패키지 임포트 실패: {e}")
    sys.exit(1)


def main():
    """메인 분석 함수"""
    
    logger.info("=" * 60)
    logger.info("Multinomial Logit Model 분석 시작")
    logger.info("=" * 60)
    
    try:
        # 1. 데이터 경로 설정
        data_dir = "processed_data/dce_data"
        
        if not Path(data_dir).exists():
            logger.error(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
            return
        
        logger.info(f"데이터 디렉토리: {data_dir}")
        
        # 2. 데이터 로딩
        logger.info("DCE 데이터 로딩 중...")
        data = load_dce_data(data_dir)
        
        # 데이터 요약 출력
        data_summary = get_dce_summary(data_dir)
        logger.info("데이터 요약:")
        for key, value in data_summary.items():
            logger.info(f"  {key}: {value}")
        
        # 3. 데이터 전처리
        logger.info("데이터 전처리 중...")
        choice_matrix = data['choice_matrix']
        
        # 전처리 요약
        preprocessing_summary = get_preprocessing_summary(choice_matrix)
        logger.info("전처리 요약:")
        for key, value in preprocessing_summary.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
        
        # 전처리 실행
        X, y, choice_sets, feature_names = preprocess_dce_data(choice_matrix)
        
        logger.info(f"전처리 완료:")
        logger.info(f"  설명변수 행렬 크기: {X.shape}")
        logger.info(f"  종속변수 벡터 크기: {y.shape}")
        logger.info(f"  선택 세트 수: {len(choice_sets)}")
        logger.info(f"  특성 이름: {feature_names}")
        
        # 4. 모델 설정
        logger.info("모델 설정 중...")
        
        # 특성 설명 정의
        feature_descriptions = {
            'sugar_free': '무설탕 여부 (1: 무설탕, 0: 일반당)',
            'has_health_label': '건강라벨 유무 (1: 있음, 0: 없음)',
            'price_scaled': '가격 (천원 단위)',
            'alternative_B': '대안 B 여부 (1: 대안 B, 0: 대안 A)'
        }
        
        # 모델 설정 생성
        config = create_default_config(feature_names, feature_descriptions)
        config.max_iterations = 1000
        config.tolerance = 1e-6
        config.method = 'bfgs'
        config.confidence_level = 0.95
        config.include_marginal_effects = True
        config.include_elasticities = True
        config.verbose = True
        
        logger.info("모델 설정 완료")
        
        # 5. 모델 추정
        logger.info("Multinomial Logit Model 추정 중...")
        results = estimate_multinomial_logit(X, y, choice_sets, config)
        
        logger.info("모델 추정 완료")
        
        # 6. 결과 분석
        logger.info("결과 분석 중...")
        analyzer = analyze_results(results)
        
        # 계수 테이블 출력
        coeffs_table = analyzer.create_coefficients_table()
        logger.info("계수 테이블:")
        logger.info(f"\n{coeffs_table.to_string(index=False)}")
        
        # 오즈비 테이블 출력
        odds_ratios = analyzer.calculate_odds_ratios()
        logger.info("오즈비 테이블:")
        logger.info(f"\n{odds_ratios.to_string(index=False)}")
        
        # 모델 요약 출력
        model_summary = analyzer.create_model_summary()
        logger.info("모델 요약:")
        for category, values in model_summary.items():
            logger.info(f"  {category}:")
            for key, value in values.items():
                logger.info(f"    {key}: {value}")
        
        # 7. 종합 보고서 생성
        logger.info("종합 보고서 생성 중...")
        comprehensive_report = analyzer.create_comprehensive_report()
        
        # 보고서를 파일로 저장
        report_filename = "multinomial_logit_analysis_report.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(comprehensive_report)
        
        logger.info(f"종합 보고서가 {report_filename}에 저장되었습니다")
        
        # 콘솔에도 출력
        print("\n" + comprehensive_report)
        
        # 8. Excel 파일로 결과 내보내기
        excel_filename = "multinomial_logit_results.xlsx"
        try:
            analyzer.export_results_to_excel(excel_filename)
            logger.info(f"상세 결과가 {excel_filename}에 저장되었습니다")
        except Exception as e:
            logger.warning(f"Excel 파일 저장 실패: {e}")
        
        # 9. 계수 해석
        logger.info("계수 해석:")
        interpretations = analyzer.interpret_coefficients()
        for var, interpretation in interpretations.items():
            logger.info(f"  {interpretation}")
        
        logger.info("=" * 60)
        logger.info("분석 완료!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


def run_sensitivity_analysis():
    """민감도 분석 실행"""
    
    logger.info("민감도 분석 시작...")
    
    try:
        # 데이터 로딩
        data_dir = "processed_data/dce_data"
        data = load_dce_data(data_dir)
        choice_matrix = data['choice_matrix']
        X, y, choice_sets, feature_names = preprocess_dce_data(choice_matrix)
        
        # 다양한 설정으로 모델 추정
        methods = ['bfgs', 'newton', 'lbfgs']
        tolerances = [1e-4, 1e-6, 1e-8]
        
        results_comparison = []
        
        for method in methods:
            for tolerance in tolerances:
                logger.info(f"민감도 분석: method={method}, tolerance={tolerance}")
                
                try:
                    config = create_default_config(feature_names)
                    config.method = method
                    config.tolerance = tolerance
                    config.verbose = False
                    
                    results = estimate_multinomial_logit(X, y, choice_sets, config)
                    
                    results_comparison.append({
                        'method': method,
                        'tolerance': tolerance,
                        'log_likelihood': results['model_statistics']['log_likelihood'],
                        'aic': results['model_statistics']['aic'],
                        'converged': results['convergence_info']['converged']
                    })
                    
                except Exception as e:
                    logger.warning(f"민감도 분석 실패 (method={method}, tolerance={tolerance}): {e}")
        
        # 결과 비교
        import pandas as pd
        comparison_df = pd.DataFrame(results_comparison)
        logger.info("민감도 분석 결과:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # 파일로 저장
        comparison_df.to_csv("sensitivity_analysis_results.csv", index=False)
        logger.info("민감도 분석 결과가 sensitivity_analysis_results.csv에 저장되었습니다")
        
    except Exception as e:
        logger.error(f"민감도 분석 중 오류: {e}")


if __name__ == "__main__":
    # 메인 분석 실행
    exit_code = main()
    
    # 민감도 분석 실행 (선택사항)
    if exit_code == 0:
        try:
            run_sensitivity_analysis()
        except Exception as e:
            logger.warning(f"민감도 분석 건너뜀: {e}")
    
    sys.exit(exit_code)
