#!/usr/bin/env python3
"""
5개 요인 간 모든 60개 조절효과 조합 분석 실행
"""

import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path

def generate_all_combinations():
    """모든 60개 조절효과 조합 생성"""
    factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 'perceived_price', 'nutrition_knowledge']
    
    combinations_list = []
    
    for dependent in factors:
        for independent in factors:
            if independent != dependent:
                for moderator in factors:
                    if moderator != dependent and moderator != independent:
                        combinations_list.append({
                            'independent': independent,
                            'dependent': dependent,
                            'moderator': moderator,
                            'name': f"{independent}_x_{moderator}_to_{dependent}"
                        })
    
    return combinations_list


def run_all_moderation_analyses():
    """모든 조절효과 조합 분석 실행"""
    print("🔍 5개 요인 간 모든 조절효과 조합 분석")
    print("=" * 80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 모든 조합 생성
    all_combinations = generate_all_combinations()
    total_combinations = len(all_combinations)
    
    print(f"📊 분석할 조합 수: {total_combinations}개")
    print("   5개 요인 × 4개 독립변수 × 3개 조절변수 = 60개")
    print()
    
    results_summary = []
    successful_analyses = 0
    significant_effects = 0
    
    start_time = time.time()
    
    for i, combo in enumerate(all_combinations, 1):
        print(f"🔄 분석 {i:2d}/{total_combinations}: {combo['name']}")
        print(f"   {combo['independent']} × {combo['moderator']} → {combo['dependent']}")
        
        try:
            from moderation_analysis import analyze_moderation_effects
            
            analysis_start = time.time()
            
            # 조절효과 분석 실행
            results = analyze_moderation_effects(
                independent_var=combo['independent'],
                dependent_var=combo['dependent'],
                moderator_var=combo['moderator']
            )
            
            analysis_end = time.time()
            analysis_time = analysis_end - analysis_start
            
            # 결과 추출
            moderation_test = results.get('moderation_test', {})
            interaction_coef = moderation_test.get('interaction_coefficient', 0)
            p_value = moderation_test.get('p_value', 1)
            significant = moderation_test.get('significant', False)
            
            # 모델 정보
            model_info = results.get('model_info', {})
            n_obs = model_info.get('n_observations', 0)
            
            # 적합도 지수
            fit_indices = results.get('fit_indices', {})
            cfi = fit_indices.get('CFI', None)
            rmsea = fit_indices.get('RMSEA', None)
            
            # 결과 요약
            result_summary = {
                'combination': combo['name'],
                'independent': combo['independent'],
                'dependent': combo['dependent'],
                'moderator': combo['moderator'],
                'interaction_coefficient': interaction_coef,
                'p_value': p_value,
                'significant': significant,
                'n_observations': n_obs,
                'cfi': cfi,
                'rmsea': rmsea,
                'analysis_time': analysis_time,
                'status': 'success'
            }
            
            results_summary.append(result_summary)
            successful_analyses += 1
            
            if significant:
                significant_effects += 1
            
            # 결과 출력
            status = "✅ 유의함" if significant else "❌ 유의하지 않음"
            print(f"   결과: 계수={interaction_coef:.6f}, p={p_value:.6f}, {status}")
            
        except Exception as e:
            print(f"   ❌ 분석 실패: {e}")
            result_summary = {
                'combination': combo['name'],
                'independent': combo['independent'],
                'dependent': combo['dependent'],
                'moderator': combo['moderator'],
                'interaction_coefficient': None,
                'p_value': None,
                'significant': False,
                'n_observations': None,
                'cfi': None,
                'rmsea': None,
                'analysis_time': None,
                'status': 'failed',
                'error': str(e)
            }
            results_summary.append(result_summary)
        
        # 진행률 표시
        if i % 10 == 0 or i == total_combinations:
            elapsed = time.time() - start_time
            progress = i / total_combinations * 100
            print(f"   📈 진행률: {progress:.1f}% ({i}/{total_combinations}), 경과시간: {elapsed:.1f}초")
        
        print()
    
    total_time = time.time() - start_time
    
    # 결과 요약
    print("=" * 80)
    print("📊 전체 분석 결과 요약")
    print("=" * 80)
    print(f"총 분석 수: {total_combinations}")
    print(f"성공한 분석: {successful_analyses}/{total_combinations} ({successful_analyses/total_combinations*100:.1f}%)")
    print(f"유의한 조절효과: {significant_effects}/{successful_analyses} ({significant_effects/successful_analyses*100:.1f}%)")
    print(f"총 분석 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
    print(f"평균 분석 시간: {total_time/total_combinations:.3f}초")
    print()
    
    return results_summary


def save_comprehensive_results(results_summary):
    """포괄적 결과 저장"""
    print("💾 결과 저장 중...")
    
    # 결과 디렉토리 생성
    results_dir = Path("moderation_analysis_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. CSV 파일로 저장
    df = pd.DataFrame(results_summary)
    csv_file = results_dir / f"all_moderation_combinations_{timestamp}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장: {csv_file}")
    
    # 2. 유의한 결과만 별도 저장
    significant_results = [r for r in results_summary if r.get('significant', False)]
    if significant_results:
        sig_df = pd.DataFrame(significant_results)
        sig_csv_file = results_dir / f"significant_moderation_effects_{timestamp}.csv"
        sig_df.to_csv(sig_csv_file, index=False, encoding='utf-8-sig')
        print(f"✅ 유의한 결과 CSV 저장: {sig_csv_file}")
    
    # 3. 요약 보고서 저장
    report_file = results_dir / f"moderation_analysis_summary_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("5개 요인 간 조절효과 분석 종합 보고서\n")
        f.write("=" * 80 + "\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 전체 요약
        total_analyses = len(results_summary)
        successful_analyses = len([r for r in results_summary if r['status'] == 'success'])
        significant_effects = len([r for r in results_summary if r.get('significant', False)])
        
        f.write("📊 분석 요약\n")
        f.write("-" * 40 + "\n")
        f.write(f"총 분석 조합: {total_analyses}개\n")
        f.write(f"성공한 분석: {successful_analyses}개 ({successful_analyses/total_analyses*100:.1f}%)\n")
        f.write(f"유의한 조절효과: {significant_effects}개 ({significant_effects/successful_analyses*100:.1f}%)\n\n")
        
        # 유의한 조절효과 상세
        if significant_results:
            f.write("🎯 유의한 조절효과 상세\n")
            f.write("-" * 40 + "\n")
            for i, result in enumerate(significant_results, 1):
                f.write(f"{i}. {result['independent']} × {result['moderator']} → {result['dependent']}\n")
                f.write(f"   상호작용 계수: {result['interaction_coefficient']:.6f}\n")
                f.write(f"   P값: {result['p_value']:.6f}\n")
                f.write(f"   관측치 수: {result['n_observations']}\n\n")
        else:
            f.write("💡 유의한 조절효과가 발견되지 않았습니다.\n\n")
        
        # 요인별 분석 결과
        factors = ['health_concern', 'perceived_benefit', 'purchase_intention', 'perceived_price', 'nutrition_knowledge']
        
        f.write("📋 요인별 분석 결과\n")
        f.write("-" * 40 + "\n")
        
        for factor in factors:
            factor_results = [r for r in results_summary if r['dependent'] == factor and r['status'] == 'success']
            factor_significant = [r for r in factor_results if r.get('significant', False)]
            
            f.write(f"\n{factor} (종속변수):\n")
            f.write(f"  총 분석: {len(factor_results)}개\n")
            f.write(f"  유의한 조절효과: {len(factor_significant)}개\n")
            
            if factor_significant:
                for result in factor_significant:
                    f.write(f"    • {result['independent']} × {result['moderator']} (p={result['p_value']:.4f})\n")
    
    print(f"✅ 요약 보고서 저장: {report_file}")
    
    return {
        'csv_file': csv_file,
        'significant_csv': sig_csv_file if significant_results else None,
        'report_file': report_file
    }


def main():
    """메인 함수"""
    print("🚀 5개 요인 간 모든 조절효과 조합 분석 실행")
    print("=" * 80)
    
    # 모든 조절효과 분석 실행
    results_summary = run_all_moderation_analyses()
    
    # 결과 저장
    saved_files = save_comprehensive_results(results_summary)
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("🎉 모든 조절효과 분석 완료!")
    print("=" * 80)
    
    total_analyses = len(results_summary)
    successful_analyses = len([r for r in results_summary if r['status'] == 'success'])
    significant_effects = len([r for r in results_summary if r.get('significant', False)])
    
    print(f"📊 최종 결과:")
    print(f"   총 분석 조합: {total_analyses}개")
    print(f"   성공한 분석: {successful_analyses}개")
    print(f"   유의한 조절효과: {significant_effects}개")
    
    print(f"\n💾 저장된 파일:")
    for file_type, file_path in saved_files.items():
        if file_path:
            print(f"   - {file_type}: {file_path.name}")
    
    if significant_effects > 0:
        print(f"\n🎯 {significant_effects}개의 유의한 조절효과가 발견되었습니다!")
        print("   상세 결과는 저장된 파일을 확인하세요.")
    else:
        print(f"\n💡 유의한 조절효과가 발견되지 않았습니다.")
        print("   이는 다음을 의미할 수 있습니다:")
        print("   - 요인 간 조절효과가 실제로 존재하지 않음")
        print("   - 표본 크기가 조절효과 탐지에 충분하지 않음")
        print("   - 측정 방법이나 모델 설정의 개선이 필요함")


if __name__ == "__main__":
    main()
