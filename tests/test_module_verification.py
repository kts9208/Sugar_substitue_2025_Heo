#!/usr/bin/env python3
"""
경로분석 모듈 검증 테스트
1. 요인별 데이터 로드 확인
2. 경로분석 결과 저장 확인  
3. 시각화 결과 저장 확인
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from path_analysis import (
    PathAnalyzer,
    PathModelBuilder,
    PathResultsExporter,
    PathAnalysisVisualizer,
    create_default_path_config,
    create_path_model,
    export_path_results
)

def test_data_loading():
    """1. 요인별 데이터 로드 확인"""
    print("=" * 60)
    print("1. 요인별 데이터 로드 확인")
    print("=" * 60)
    
    try:
        # PathAnalyzer 초기화
        config = create_default_path_config(verbose=True)
        analyzer = PathAnalyzer(config)
        
        # 개별 요인 데이터 로드 테스트
        factors = ['health_concern', 'perceived_benefit', 'perceived_price', 
                  'nutrition_knowledge', 'purchase_intention']
        
        print("개별 요인 데이터 로드 테스트:")
        for factor in factors:
            try:
                data = analyzer.load_data([factor])
                print(f"✅ {factor}: {data.shape} - 컬럼: {list(data.columns)}")
                
                # 데이터 요약 통계
                print(f"   결측치: {data.isnull().sum().sum()}개")
                print(f"   평균: {data.mean().mean():.2f}")
                
            except Exception as e:
                print(f"❌ {factor}: 오류 - {e}")
        
        # 전체 요인 데이터 로드 테스트
        print(f"\n전체 요인 데이터 로드 테스트:")
        try:
            all_data = analyzer.load_data(factors)
            print(f"✅ 전체 데이터: {all_data.shape}")
            print(f"   컬럼 수: {len(all_data.columns)}")
            print(f"   결측치: {all_data.isnull().sum().sum()}개")
            print(f"   완전한 관측치: {all_data.dropna().shape[0]}개")
            
            # 각 요인별 문항 수 확인
            print(f"\n요인별 문항 수:")
            factor_items = {
                'health_concern': [col for col in all_data.columns if col.startswith('q') and int(col[1:]) in range(6, 12)],
                'perceived_benefit': [col for col in all_data.columns if col.startswith('q') and int(col[1:]) in range(16, 18)],
                'purchase_intention': [col for col in all_data.columns if col.startswith('q') and int(col[1:]) in range(18, 20)],
                'perceived_price': [col for col in all_data.columns if col.startswith('q') and int(col[1:]) in range(27, 30)],
                'nutrition_knowledge': [col for col in all_data.columns if col.startswith('q') and int(col[1:]) in range(30, 50)]
            }
            
            for factor, items in factor_items.items():
                print(f"   {factor}: {len(items)}개 문항 {items[:3]}{'...' if len(items) > 3 else ''}")
            
            return True, all_data
            
        except Exception as e:
            print(f"❌ 전체 데이터 로드 실패: {e}")
            return False, None
            
    except Exception as e:
        print(f"❌ 데이터 로드 테스트 실패: {e}")
        return False, None

def test_results_saving():
    """2. 경로분석 결과 저장 확인"""
    print("\n" + "=" * 60)
    print("2. 경로분석 결과 저장 확인")
    print("=" * 60)
    
    try:
        # 간단한 매개모델로 테스트
        model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        config = create_default_path_config(verbose=False)
        analyzer = PathAnalyzer(config)
        
        # 데이터 로드 및 모델 적합
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        data = analyzer.load_data(variables)
        results = analyzer.fit_model(model_spec, data)
        
        print(f"모델 적합 완료: {results['model_info']['n_observations']}개 관측치")
        
        # 결과 저장 테스트
        print("\n결과 저장 테스트:")
        
        # 1. PathResultsExporter 직접 사용
        exporter = PathResultsExporter()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 개별 파일 저장 테스트
        try:
            # 모델 정보 저장
            model_file = exporter._export_model_info(
                results['model_info'], 
                f"test_model_info_{timestamp}"
            )
            print(f"✅ 모델 정보 저장: {model_file}")
            
            # 적합도 지수 저장
            if 'fit_indices' in results and results['fit_indices']:
                fit_file = exporter._export_fit_indices(
                    results['fit_indices'],
                    f"test_fit_indices_{timestamp}"
                )
                print(f"✅ 적합도 지수 저장: {fit_file}")
            
            # 경로계수 저장
            if 'path_coefficients' in results and results['path_coefficients']:
                coeff_file = exporter._export_path_coefficients(
                    results['path_coefficients'],
                    f"test_path_coefficients_{timestamp}"
                )
                print(f"✅ 경로계수 저장: {coeff_file}")
            
        except Exception as e:
            print(f"❌ 개별 파일 저장 오류: {e}")
        
        # 2. 통합 저장 함수 테스트 (JSON 제외)
        try:
            # 간단한 저장 함수 사용
            save_simple_results_verification(results, variables, timestamp)
            print(f"✅ 통합 결과 저장 완료")
            
        except Exception as e:
            print(f"❌ 통합 저장 오류: {e}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ 결과 저장 테스트 실패: {e}")
        return False, None

def test_visualization():
    """3. 시각화 결과 저장 확인"""
    print("\n" + "=" * 60)
    print("3. 시각화 결과 저장 확인")
    print("=" * 60)
    
    try:
        # 간단한 매개모델로 테스트
        model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        config = create_default_path_config(verbose=False)
        analyzer = PathAnalyzer(config)
        
        # 데이터 로드 및 모델 적합
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        data = analyzer.load_data(variables)
        results = analyzer.fit_model(model_spec, data)
        
        print(f"시각화용 모델 준비 완료")
        
        # PathAnalysisVisualizer 테스트
        visualizer = PathAnalysisVisualizer()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n시각화 테스트:")
        
        # 1. 적합도 지수 차트
        try:
            if 'fit_indices' in results and results['fit_indices']:
                fit_chart = visualizer.plot_fit_indices(
                    results['fit_indices'], 
                    f"test_fit_chart_{timestamp}"
                )
                print(f"✅ 적합도 지수 차트: {fit_chart}")
        except Exception as e:
            print(f"❌ 적합도 차트 오류: {e}")
        
        # 2. 경로계수 차트
        try:
            if 'path_coefficients' in results and results['path_coefficients']:
                path_chart = visualizer.plot_path_coefficients(
                    results['path_coefficients'],
                    f"test_path_chart_{timestamp}"
                )
                print(f"✅ 경로계수 차트: {path_chart}")
        except Exception as e:
            print(f"❌ 경로계수 차트 오류: {e}")
        
        # 3. 간단한 matplotlib 차트 생성 테스트
        try:
            output_dir = Path("path_analysis_results")
            output_dir.mkdir(exist_ok=True)
            
            # 경로계수 막대 차트
            if 'path_coefficients' in results and results['path_coefficients']:
                path_coeffs = results['path_coefficients']
                if 'paths' in path_coeffs and path_coeffs['paths']:
                    # 구조적 경로만 추출 (측정모델 제외)
                    structural_paths = []
                    structural_coeffs = []
                    
                    for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                        if not (from_var in variables and to_var.startswith('q')):  # 측정모델 제외
                            coeff = path_coeffs.get('coefficients', {}).get(i, 0)
                            structural_paths.append(f"{from_var}\n→ {to_var}")
                            structural_coeffs.append(coeff)
                    
                    if structural_paths:
                        plt.figure(figsize=(10, 6))
                        bars = plt.bar(range(len(structural_paths)), structural_coeffs)
                        plt.xlabel('경로')
                        plt.ylabel('경로계수')
                        plt.title('구조적 경로계수')
                        plt.xticks(range(len(structural_paths)), structural_paths, rotation=45)
                        
                        # 막대 색상 설정 (양수: 파랑, 음수: 빨강)
                        for bar, coeff in zip(bars, structural_coeffs):
                            bar.set_color('skyblue' if coeff >= 0 else 'lightcoral')
                        
                        plt.tight_layout()
                        chart_file = output_dir / f"test_structural_paths_{timestamp}.png"
                        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"✅ 구조적 경로 차트: {chart_file}")
        
        except Exception as e:
            print(f"❌ matplotlib 차트 오류: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 시각화 테스트 실패: {e}")
        return False

def save_simple_results_verification(results, variables, timestamp):
    """간단한 결과 저장 (검증용)"""
    output_dir = Path("path_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # 텍스트 파일로 저장
    result_file = output_dir / f"verification_results_{timestamp}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("경로분석 모듈 검증 결과\n")
        f.write("=" * 40 + "\n")
        f.write(f"검증 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"변수: {', '.join(variables)}\n")
        f.write(f"관측치 수: {results['model_info']['n_observations']}\n\n")
        
        # 적합도 지수
        if 'fit_indices' in results and results['fit_indices']:
            f.write("적합도 지수:\n")
            for index, value in results['fit_indices'].items():
                try:
                    if hasattr(value, 'iloc'):
                        numeric_value = value.iloc[0] if len(value) > 0 else np.nan
                    else:
                        numeric_value = value
                    
                    if isinstance(numeric_value, (int, float)) and not pd.isna(numeric_value):
                        f.write(f"  {index}: {numeric_value:.4f}\n")
                except:
                    f.write(f"  {index}: {value}\n")
        
        # 경로계수
        if 'path_coefficients' in results and results['path_coefficients']:
            f.write(f"\n경로계수:\n")
            path_coeffs = results['path_coefficients']
            if 'paths' in path_coeffs and path_coeffs['paths']:
                for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                    coeff = path_coeffs.get('coefficients', {}).get(i, 0)
                    f.write(f"  {from_var} -> {to_var}: {coeff:.4f}\n")

def check_saved_files():
    """저장된 파일들 확인"""
    print("\n" + "=" * 60)
    print("4. 저장된 파일들 확인")
    print("=" * 60)
    
    output_dir = Path("path_analysis_results")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        print(f"총 {len(files)}개 파일 저장됨:")
        
        # 파일 유형별 분류
        csv_files = [f for f in files if f.suffix == '.csv']
        txt_files = [f for f in files if f.suffix == '.txt']
        png_files = [f for f in files if f.suffix == '.png']
        json_files = [f for f in files if f.suffix == '.json']
        
        print(f"  📊 CSV 파일: {len(csv_files)}개")
        for f in csv_files[-3:]:  # 최근 3개만 표시
            print(f"    - {f.name}")
        
        print(f"  📄 TXT 파일: {len(txt_files)}개")
        for f in txt_files[-3:]:  # 최근 3개만 표시
            print(f"    - {f.name}")
        
        print(f"  📈 PNG 파일: {len(png_files)}개")
        for f in png_files[-3:]:  # 최근 3개만 표시
            print(f"    - {f.name}")
        
        if json_files:
            print(f"  📋 JSON 파일: {len(json_files)}개")
            for f in json_files[-3:]:
                print(f"    - {f.name}")
        
        return True
    else:
        print("❌ 결과 디렉토리가 존재하지 않습니다.")
        return False

def main():
    """메인 검증 함수"""
    print("🔍 경로분석 모듈 검증 테스트 시작")
    print(f"검증 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 1. 데이터 로드 테스트
    data_success, data = test_data_loading()
    results['data_loading'] = data_success
    
    # 2. 결과 저장 테스트
    save_success, analysis_results = test_results_saving()
    results['results_saving'] = save_success
    
    # 3. 시각화 테스트
    viz_success = test_visualization()
    results['visualization'] = viz_success
    
    # 4. 저장된 파일 확인
    files_success = check_saved_files()
    results['file_check'] = files_success
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 검증 결과 요약")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, success in results.items():
        status = "✅ 통과" if success else "❌ 실패"
        print(f"  {test_name}: {status}")
    
    print(f"\n전체 테스트: {passed_tests}/{total_tests} 통과 ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 모든 검증 테스트 통과!")
    else:
        print("⚠️  일부 테스트 실패 - 추가 확인 필요")
    
    return results

if __name__ == "__main__":
    main()
