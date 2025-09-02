"""
Factor Analysis 사용 예제

이 모듈은 factor analysis 패키지의 다양한 사용 방법을 보여줍니다.
"""

import sys
from pathlib import Path
import pandas as pd

# 패키지 임포트를 위한 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from factor_analysis import (
    FactorDataLoader, FactorAnalyzer, FactorResultsExporter,
    FactorAnalysisConfig, create_factor_model_spec,
    load_factor_data, get_available_factors, analyze_factor_loading,
    export_factor_results, create_custom_config
)


def example_1_basic_usage():
    """예제 1: 기본 사용법"""
    print("=" * 60)
    print("예제 1: 기본 Factor Loading 분석")
    print("=" * 60)
    
    try:
        # 1. 사용 가능한 요인들 확인
        print("1. 사용 가능한 요인들 확인:")
        available_factors = get_available_factors()
        print(f"   사용 가능한 요인들: {available_factors}")
        print()
        
        # 2. 단일 요인 분석 (건강관심도)
        print("2. 건강관심도 요인 분석:")
        if 'health_concern' in available_factors:
            results = analyze_factor_loading('health_concern')
            
            # 결과 출력
            print("   Factor Loadings:")
            loadings = results.get('factor_loadings', pd.DataFrame())
            if not loadings.empty:
                print(loadings.to_string(index=False))
            
            print("\n   적합도 지수:")
            fit_indices = results.get('fit_indices', {})
            for index, value in fit_indices.items():
                print(f"   {index}: {value}")
            
            # 결과 저장
            print("\n3. 결과 저장:")
            saved_files = export_factor_results(results)
            print(f"   저장된 파일들: {list(saved_files.keys())}")
        else:
            print("   health_concern 데이터를 찾을 수 없습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("semopy가 설치되어 있는지 확인해주세요: pip install semopy")


def example_2_multiple_factors():
    """예제 2: 다중 요인 분석"""
    print("\n" + "=" * 60)
    print("예제 2: 다중 요인 분석")
    print("=" * 60)
    
    try:
        # 분석할 요인들 선택
        factors_to_analyze = ['health_concern', 'perceived_benefit', 'purchase_intention']
        available_factors = get_available_factors()
        
        # 사용 가능한 요인들만 선택
        valid_factors = [f for f in factors_to_analyze if f in available_factors]
        
        if len(valid_factors) >= 2:
            print(f"분석할 요인들: {valid_factors}")
            
            # 다중 요인 분석
            results = analyze_factor_loading(valid_factors)
            
            # 결과 출력
            print("\nFactor Loadings:")
            loadings = results.get('factor_loadings', pd.DataFrame())
            if not loadings.empty:
                # 요인별로 그룹화하여 출력
                for factor in loadings['Factor'].unique():
                    factor_loadings = loadings[loadings['Factor'] == factor]
                    print(f"\n{factor}:")
                    print(factor_loadings[['Item', 'Loading', 'P_value', 'Significant']].to_string(index=False))
            
            print("\n적합도 지수:")
            fit_indices = results.get('fit_indices', {})
            for index, value in fit_indices.items():
                print(f"{index}: {value}")
        else:
            print(f"분석에 필요한 요인 데이터가 부족합니다. 사용 가능: {available_factors}")
    
    except Exception as e:
        print(f"오류 발생: {e}")


def example_3_custom_configuration():
    """예제 3: 사용자 정의 설정"""
    print("\n" + "=" * 60)
    print("예제 3: 사용자 정의 설정")
    print("=" * 60)
    
    try:
        # 사용자 정의 설정 생성
        custom_config = create_custom_config(
            estimator='ML',  # Maximum Likelihood
            optimizer='L-BFGS-B',
            max_iterations=2000,
            tolerance=1e-8,
            confidence_level=0.99,
            standardized=True,
            include_modification_indices=True
        )
        
        print("사용자 정의 설정:")
        print(f"  추정방법: {custom_config.estimator}")
        print(f"  최적화: {custom_config.optimizer}")
        print(f"  최대 반복: {custom_config.max_iterations}")
        print(f"  신뢰수준: {custom_config.confidence_level}")
        
        # 사용자 정의 설정으로 분석
        available_factors = get_available_factors()
        if 'health_concern' in available_factors:
            analyzer = FactorAnalyzer(config=custom_config)
            results = analyzer.analyze_single_factor('health_concern')
            
            print(f"\n분석 완료 - 샘플 크기: {results['model_info']['n_observations']}")
        else:
            print("health_concern 데이터를 찾을 수 없습니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")


def example_4_detailed_data_exploration():
    """예제 4: 상세 데이터 탐색"""
    print("\n" + "=" * 60)
    print("예제 4: 상세 데이터 탐색")
    print("=" * 60)
    
    try:
        # 데이터 로더 생성
        loader = FactorDataLoader()
        
        # 요인 요약 정보
        print("1. 요인 요약 정보:")
        summary = loader.get_factor_summary()
        print(summary.to_string(index=False))
        
        # 분석 가능한 요인들만 로딩
        print("\n2. 분석 가능한 요인들 로딩:")
        analyzable_data = loader.load_analyzable_factors()
        
        for factor_name, data in analyzable_data.items():
            print(f"\n{factor_name}:")
            print(f"  샘플 수: {len(data)}")
            print(f"  변수 수: {len(data.columns) - 1}")  # 'no' 컬럼 제외
            
            # 기술통계
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:  # 'no' 컬럼 외에 다른 수치형 컬럼이 있는 경우
                desc_stats = data[numeric_cols].describe()
                print(f"  평균 범위: {desc_stats.loc['mean'].min():.2f} - {desc_stats.loc['mean'].max():.2f}")
                print(f"  표준편차 범위: {desc_stats.loc['std'].min():.2f} - {desc_stats.loc['std'].max():.2f}")
    
    except Exception as e:
        print(f"오류 발생: {e}")


def example_5_model_specification():
    """예제 5: 모델 스펙 생성"""
    print("\n" + "=" * 60)
    print("예제 5: 모델 스펙 생성")
    print("=" * 60)
    
    try:
        # 단일 요인 모델 스펙
        print("1. 단일 요인 모델 스펙 (건강관심도):")
        single_spec = create_factor_model_spec(single_factor='health_concern')
        print(single_spec)
        
        # 다중 요인 모델 스펙
        print("\n2. 다중 요인 모델 스펙:")
        multi_spec = create_factor_model_spec(
            factor_names=['health_concern', 'perceived_benefit'],
            allow_correlations=True
        )
        print(multi_spec)
        
        # 모든 분석 가능한 요인들의 모델 스펙
        print("\n3. 전체 요인 모델 스펙:")
        full_spec = create_factor_model_spec()  # 기본값: 모든 분석 가능한 요인들
        print(full_spec[:500] + "..." if len(full_spec) > 500 else full_spec)
    
    except Exception as e:
        print(f"오류 발생: {e}")


def example_6_results_export_options():
    """예제 6: 결과 내보내기 옵션들"""
    print("\n" + "=" * 60)
    print("예제 6: 결과 내보내기 옵션들")
    print("=" * 60)
    
    try:
        available_factors = get_available_factors()
        if 'health_concern' in available_factors:
            # 분석 실행
            results = analyze_factor_loading('health_concern')
            
            # 결과 내보내기 객체 생성
            exporter = FactorResultsExporter("factor_analysis_results")
            
            print("1. 개별 파일 내보내기:")
            
            # Factor loadings만 내보내기
            loadings_file = exporter.export_factor_loadings(results, "example_loadings.csv")
            print(f"   Factor loadings: {loadings_file}")
            
            # 적합도 지수만 내보내기
            fit_file = exporter.export_fit_indices(results, "example_fit_indices.csv")
            print(f"   적합도 지수: {fit_file}")
            
            # 요약 보고서
            summary_file = exporter.export_summary_report(results, "example_summary.txt")
            print(f"   요약 보고서: {summary_file}")
            
            print("\n2. 종합 내보내기:")
            comprehensive_files = exporter.export_comprehensive_results(results, "example_comprehensive")
            for file_type, file_path in comprehensive_files.items():
                print(f"   {file_type}: {file_path}")
        else:
            print("health_concern 데이터를 찾을 수 없습니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")


def main():
    """메인 함수 - 모든 예제 실행"""
    print("Factor Analysis 패키지 사용 예제")
    print("=" * 60)
    
    # 데이터 디렉토리 확인
    data_dir = project_root / "processed_data" / "survey_data"
    if not data_dir.exists():
        print(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        print("먼저 설문조사 데이터 전처리를 실행해주세요.")
        return
    
    # 예제들 실행
    examples = [
        example_1_basic_usage,
        example_2_multiple_factors,
        example_3_custom_configuration,
        example_4_detailed_data_exploration,
        example_5_model_specification,
        example_6_results_export_options
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except KeyboardInterrupt:
            print(f"\n예제 {i} 중단됨")
            break
        except Exception as e:
            print(f"\n예제 {i} 실행 중 오류: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("모든 예제 실행 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
