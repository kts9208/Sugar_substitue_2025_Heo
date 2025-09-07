#!/usr/bin/env python3
"""
5개 요인 경로분석 실행 (간소화 버전)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from path_analysis import (
    PathAnalyzer,
    create_default_path_config,
    create_path_model
)

def run_comprehensive_analysis():
    """5개 요인 종합 분석"""
    print("🔍 5개 요인 경로분석 실행")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 종합적인 구조모델 설정
        variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                    'nutrition_knowledge', 'purchase_intention']
        
        # 복합적인 경로 관계 정의
        paths = [
            ('health_concern', 'perceived_benefit'),      # 건강관심도 -> 지각된혜택
            ('health_concern', 'perceived_price'),        # 건강관심도 -> 지각된가격
            ('health_concern', 'nutrition_knowledge'),    # 건강관심도 -> 영양지식
            ('nutrition_knowledge', 'perceived_benefit'), # 영양지식 -> 지각된혜택
            ('perceived_benefit', 'purchase_intention'),  # 지각된혜택 -> 구매의도
            ('perceived_price', 'purchase_intention'),    # 지각된가격 -> 구매의도
            ('nutrition_knowledge', 'purchase_intention'), # 영양지식 -> 구매의도
            ('health_concern', 'purchase_intention')      # 건강관심도 -> 구매의도 (직접효과)
        ]
        
        # 상관관계 설정
        correlations = [
            ('perceived_benefit', 'perceived_price'),     # 지각된혜택 <-> 지각된가격
            ('perceived_benefit', 'nutrition_knowledge')  # 지각된혜택 <-> 영양지식
        ]
        
        # 2. 모델 스펙 생성
        model_spec = create_path_model(
            model_type='custom',
            variables=variables,
            paths=paths,
            correlations=correlations
        )
        
        print("종합적인 구조모델 스펙 생성 완료")
        print(f"포함된 변수: {len(variables)}개")
        print(f"경로 수: {len(paths)}개")
        print(f"상관관계: {len(correlations)}개")
        
        # 3. 분석 실행
        config = create_default_path_config(
            standardized=True,
            verbose=True
        )
        
        analyzer = PathAnalyzer(config)
        data = analyzer.load_data(variables)
        print(f"\n데이터 로드 완료: {data.shape}")
        
        results = analyzer.fit_model(model_spec, data)
        print(f"모델 적합 완료!")
        print(f"관측치 수: {results['model_info']['n_observations']}")
        print(f"변수 수: {results['model_info']['n_variables']}")
        
        # 4. 적합도 지수 출력
        if 'fit_indices' in results and results['fit_indices']:
            print("\n=== 적합도 지수 ===")
            for index, value in results['fit_indices'].items():
                try:
                    if hasattr(value, 'iloc'):
                        numeric_value = value.iloc[0] if len(value) > 0 else np.nan
                    else:
                        numeric_value = value
                    
                    if isinstance(numeric_value, (int, float)) and not pd.isna(numeric_value):
                        interpretation = interpret_fit_index(index, numeric_value)
                        print(f"  {index}: {numeric_value:.4f} ({interpretation})")
                except:
                    print(f"  {index}: {value}")
        
        # 5. 주요 경로계수 출력 (구매의도에 대한 직접효과만)
        print("\n=== 구매의도에 대한 직접효과 ===")
        if 'path_coefficients' in results and results['path_coefficients']:
            path_coeffs = results['path_coefficients']
            if 'paths' in path_coeffs and path_coeffs['paths']:
                for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                    if to_var == 'purchase_intention':
                        coeff = path_coeffs.get('coefficients', {}).get(i, 0)
                        print(f"  {from_var} -> {to_var}: {coeff:.4f}")
        
        # 6. 간단한 결과 저장
        save_simple_results(results, variables, paths)
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def interpret_fit_index(index_name: str, value: float) -> str:
    """적합도 지수 해석"""
    if pd.isna(value):
        return "N/A"
    
    if index_name == 'cfi' or index_name == 'tli':
        if value > 0.95:
            return "우수"
        elif value > 0.90:
            return "수용가능"
        else:
            return "부족"
    elif index_name == 'rmsea':
        if value < 0.06:
            return "우수"
        elif value < 0.08:
            return "수용가능"
        else:
            return "부족"
    elif index_name == 'srmr':
        if value < 0.08:
            return "우수"
        elif value < 0.10:
            return "수용가능"
        else:
            return "부족"
    elif index_name == 'p_value':
        if value > 0.05:
            return "좋음 (비유의적)"
        else:
            return "나쁨 (유의적)"
    
    return ""

def save_simple_results(results, variables, paths):
    """간단한 결과 저장"""
    try:
        from pathlib import Path
        output_dir = Path("path_analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 모델 정보 저장
        model_info = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'variables': variables,
            'paths': [f"{from_var} -> {to_var}" for from_var, to_var in paths],
            'n_observations': results['model_info']['n_observations'],
            'n_variables': results['model_info']['n_variables']
        }
        
        model_file = output_dir / f"5factor_model_info_{timestamp}.txt"
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("5개 요인 경로분석 결과\n")
            f.write("=" * 40 + "\n")
            f.write(f"분석 일시: {model_info['analysis_date']}\n")
            f.write(f"관측치 수: {model_info['n_observations']}\n")
            f.write(f"변수 수: {model_info['n_variables']}\n\n")
            
            f.write("포함된 변수:\n")
            for var in model_info['variables']:
                f.write(f"  - {var}\n")
            
            f.write("\n경로 관계:\n")
            for path in model_info['paths']:
                f.write(f"  - {path}\n")
        
        # 2. 적합도 지수 저장
        if 'fit_indices' in results and results['fit_indices']:
            fit_file = output_dir / f"5factor_fit_indices_{timestamp}.txt"
            with open(fit_file, 'w', encoding='utf-8') as f:
                f.write("적합도 지수\n")
                f.write("=" * 20 + "\n")
                for index, value in results['fit_indices'].items():
                    try:
                        if hasattr(value, 'iloc'):
                            numeric_value = value.iloc[0] if len(value) > 0 else np.nan
                        else:
                            numeric_value = value
                        
                        if isinstance(numeric_value, (int, float)) and not pd.isna(numeric_value):
                            interpretation = interpret_fit_index(index, numeric_value)
                            f.write(f"{index}: {numeric_value:.4f} ({interpretation})\n")
                    except:
                        f.write(f"{index}: {value}\n")
        
        # 3. 경로계수 저장
        if 'path_coefficients' in results and results['path_coefficients']:
            coeff_file = output_dir / f"5factor_path_coefficients_{timestamp}.txt"
            with open(coeff_file, 'w', encoding='utf-8') as f:
                f.write("경로계수\n")
                f.write("=" * 20 + "\n")
                path_coeffs = results['path_coefficients']
                if 'paths' in path_coeffs and path_coeffs['paths']:
                    for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                        coeff = path_coeffs.get('coefficients', {}).get(i, 0)
                        f.write(f"{from_var} -> {to_var}: {coeff:.4f}\n")
        
        print(f"\n결과 저장 완료:")
        print(f"  - {model_file}")
        print(f"  - {fit_file}")
        print(f"  - {coeff_file}")
        
    except Exception as e:
        print(f"결과 저장 오류: {e}")

def main():
    """메인 실행 함수"""
    results = run_comprehensive_analysis()
    
    if results:
        print("\n" + "=" * 60)
        print("📊 5개 요인 경로분석 완료!")
        print("=" * 60)
        print("✅ 모델 추정 성공")
        print("✅ 적합도 지수 계산 완료")
        print("✅ 경로계수 추출 완료")
        print("✅ 결과 파일 저장 완료")
        print("\n결과 파일들이 'path_analysis_results' 디렉토리에 저장되었습니다.")
    else:
        print("\n❌ 분석 실패")

if __name__ == "__main__":
    main()
