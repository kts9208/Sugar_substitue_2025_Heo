"""
RPL 효용함수 계수 상세 검토 및 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_rpl_results():
    """RPL 분석 결과를 로드합니다."""
    
    print("=" * 80)
    print("RPL 효용함수 계수 상세 검토")
    print("=" * 80)
    print(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results_dir = Path("rpl_analysis_results")
    
    # 최신 파일들 로드
    heterogeneity_file = results_dir / "heterogeneity_analysis_20250916_210256.csv"
    individual_coeffs_file = results_dir / "individual_coefficients_20250916_210256.csv"
    comprehensive_file = results_dir / "rpl_comprehensive_results_20250916_210256.json"
    
    print(f"\n📁 로드할 파일:")
    print(f"   ✓ 이질성 분석: {heterogeneity_file.name}")
    print(f"   ✓ 개인별 계수: {individual_coeffs_file.name}")
    print(f"   ✓ 종합 결과: {comprehensive_file.name}")
    
    # 데이터 로드
    heterogeneity_df = pd.read_csv(heterogeneity_file, index_col=0)
    individual_coeffs_df = pd.read_csv(individual_coeffs_file)
    
    with open(comprehensive_file, 'r', encoding='utf-8') as f:
        comprehensive_data = json.load(f)
    
    print(f"   ✓ 데이터 로드 완료: {len(individual_coeffs_df)} 개인, {len(heterogeneity_df)} 변수")
    
    return heterogeneity_df, individual_coeffs_df, comprehensive_data


def analyze_coefficient_distributions(heterogeneity_df, individual_coeffs_df):
    """계수 분포를 상세 분석합니다."""
    
    print(f"\n" + "=" * 60)
    print("1. 계수 분포 상세 분석")
    print("=" * 60)
    
    coefficient_analysis = {}
    
    # 각 변수별 계수 분포 분석
    for var_name in heterogeneity_df.index:
        if var_name in individual_coeffs_df.columns:
            coeffs = individual_coeffs_df[var_name].values
            
            # 기본 통계
            analysis = {
                'mean': float(np.mean(coeffs)),
                'std': float(np.std(coeffs)),
                'median': float(np.median(coeffs)),
                'min': float(np.min(coeffs)),
                'max': float(np.max(coeffs)),
                'q25': float(np.percentile(coeffs, 25)),
                'q75': float(np.percentile(coeffs, 75)),
                'cv': float(np.std(coeffs) / abs(np.mean(coeffs))) if np.mean(coeffs) != 0 else 0,
                'range': float(np.max(coeffs) - np.min(coeffs)),
                'iqr': float(np.percentile(coeffs, 75) - np.percentile(coeffs, 25))
            }
            
            # 분포 특성
            analysis['skewness'] = float(stats.skew(coeffs))
            analysis['kurtosis'] = float(stats.kurtosis(coeffs))
            
            # 정규성 검정
            _, p_value = stats.shapiro(coeffs[:50])  # 샘플 크기 제한
            analysis['normality_p_value'] = float(p_value)
            analysis['is_normal'] = p_value > 0.05
            
            # 계수 부호 분석
            positive_pct = (coeffs > 0).mean() * 100
            negative_pct = (coeffs < 0).mean() * 100
            zero_pct = (coeffs == 0).mean() * 100
            
            analysis['sign_distribution'] = {
                'positive_pct': float(positive_pct),
                'negative_pct': float(negative_pct),
                'zero_pct': float(zero_pct)
            }
            
            # 극값 분석
            analysis['outliers'] = {
                'lower_fence': analysis['q25'] - 1.5 * analysis['iqr'],
                'upper_fence': analysis['q75'] + 1.5 * analysis['iqr'],
                'outlier_count': int(np.sum((coeffs < analysis['q25'] - 1.5 * analysis['iqr']) | 
                                          (coeffs > analysis['q75'] + 1.5 * analysis['iqr'])))
            }
            
            coefficient_analysis[var_name] = analysis
            
            print(f"\n📊 {var_name}:")
            print(f"   평균: {analysis['mean']:.4f} ± {analysis['std']:.4f}")
            print(f"   중앙값: {analysis['median']:.4f}")
            print(f"   범위: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
            print(f"   사분위수: [{analysis['q25']:.4f}, {analysis['q75']:.4f}]")
            print(f"   변이계수: {analysis['cv']:.3f}")
            print(f"   왜도: {analysis['skewness']:.3f}, 첨도: {analysis['kurtosis']:.3f}")
            print(f"   정규성: {'정규분포' if analysis['is_normal'] else '비정규분포'} (p={analysis['normality_p_value']:.3f})")
            print(f"   부호분포: 양수 {positive_pct:.1f}%, 음수 {negative_pct:.1f}%, 영 {zero_pct:.1f}%")
            print(f"   이상치: {analysis['outliers']['outlier_count']}개")
    
    return coefficient_analysis


def analyze_sem_effects(comprehensive_data):
    """SEM 요인의 계수 영향을 분석합니다."""
    
    print(f"\n" + "=" * 60)
    print("2. SEM 요인의 계수 영향 분석")
    print("=" * 60)
    
    rpl_distributions = comprehensive_data['rpl_distributions']
    
    sem_effect_analysis = {}
    
    print("🔗 SEM 요인이 DCE 계수에 미치는 영향:")
    
    for dce_var, config in rpl_distributions.items():
        if config['sem_effects']:  # SEM 효과가 있는 경우만
            base_mean = config['mean']
            base_std = config['std']
            
            print(f"\n📋 {dce_var}:")
            print(f"   기본 계수: μ={base_mean:.4f}, σ={base_std:.4f}")
            print(f"   SEM 영향:")
            
            effect_analysis = {
                'base_coefficient': base_mean,
                'base_std': base_std,
                'sem_effects': config['sem_effects'],
                'total_sem_effect': sum(abs(effect) for effect in config['sem_effects'].values()),
                'effect_interpretation': {}
            }
            
            for sem_factor, effect_size in config['sem_effects'].items():
                direction = "증가" if effect_size > 0 else "감소"
                magnitude = "강한" if abs(effect_size) > 0.3 else "중간" if abs(effect_size) > 0.1 else "약한"
                
                interpretation = f"{sem_factor} 1 표준편차 증가 시 계수 {abs(effect_size):.2f} {direction} ({magnitude} 효과)"
                effect_analysis['effect_interpretation'][sem_factor] = interpretation
                
                print(f"      • {sem_factor}: {effect_size:+.2f} ({interpretation})")
            
            # 계수 변동 범위 계산
            max_positive_effect = sum(effect for effect in config['sem_effects'].values() if effect > 0)
            max_negative_effect = sum(effect for effect in config['sem_effects'].values() if effect < 0)
            
            theoretical_min = base_mean + max_negative_effect - 2*base_std
            theoretical_max = base_mean + max_positive_effect + 2*base_std
            
            effect_analysis['theoretical_range'] = {
                'min': theoretical_min,
                'max': theoretical_max,
                'range_width': theoretical_max - theoretical_min
            }
            
            print(f"   이론적 계수 범위: [{theoretical_min:.4f}, {theoretical_max:.4f}]")
            print(f"   총 SEM 효과 크기: {effect_analysis['total_sem_effect']:.3f}")
            
            sem_effect_analysis[dce_var] = effect_analysis
    
    return sem_effect_analysis


def compare_coefficient_magnitudes(coefficient_analysis):
    """계수 크기를 비교 분석합니다."""
    
    print(f"\n" + "=" * 60)
    print("3. 계수 크기 비교 분석")
    print("=" * 60)
    
    # 평균 계수 크기로 정렬
    sorted_coeffs = sorted(coefficient_analysis.items(), 
                          key=lambda x: abs(x[1]['mean']), reverse=True)
    
    print("📊 계수 크기 순위 (절댓값 기준):")
    
    magnitude_analysis = {}
    
    for rank, (var_name, analysis) in enumerate(sorted_coeffs, 1):
        mean_coeff = analysis['mean']
        abs_mean = abs(mean_coeff)
        direction = "긍정적" if mean_coeff > 0 else "부정적"
        
        # 효과 크기 분류
        if abs_mean > 1.0:
            effect_size = "매우 큰"
        elif abs_mean > 0.5:
            effect_size = "큰"
        elif abs_mean > 0.2:
            effect_size = "중간"
        elif abs_mean > 0.1:
            effect_size = "작은"
        else:
            effect_size = "매우 작은"
        
        magnitude_analysis[var_name] = {
            'rank': rank,
            'mean_coefficient': mean_coeff,
            'absolute_magnitude': abs_mean,
            'direction': direction,
            'effect_size': effect_size,
            'cv': analysis['cv']
        }
        
        print(f"   {rank}. {var_name}: {mean_coeff:+.4f} ({direction}, {effect_size})")
        print(f"      변이계수: {analysis['cv']:.3f}, 범위: [{analysis['min']:.3f}, {analysis['max']:.3f}]")
    
    # DCE vs SEM 요인 비교
    dce_variables = ['sugar_free', 'health_label', 'price_normalized', 'sugar_health_interaction']
    sem_variables = ['perceived_benefit', 'nutrition_knowledge', 'perceived_price', 'health_concern']
    
    dce_magnitudes = [abs(magnitude_analysis[var]['mean_coefficient']) 
                     for var in dce_variables if var in magnitude_analysis]
    sem_magnitudes = [abs(magnitude_analysis[var]['mean_coefficient']) 
                     for var in sem_variables if var in magnitude_analysis]
    
    print(f"\n⚖️ DCE vs SEM 요인 계수 크기 비교:")
    print(f"   DCE 요인 평균 크기: {np.mean(dce_magnitudes):.4f}")
    print(f"   SEM 요인 평균 크기: {np.mean(sem_magnitudes):.4f}")
    print(f"   DCE/SEM 비율: {np.mean(dce_magnitudes)/np.mean(sem_magnitudes):.2f}")
    
    return magnitude_analysis


def analyze_coefficient_stability(coefficient_analysis):
    """계수 안정성을 분석합니다."""
    
    print(f"\n" + "=" * 60)
    print("4. 계수 안정성 분석")
    print("=" * 60)
    
    stability_analysis = {}
    
    print("🎯 계수 안정성 평가:")
    
    for var_name, analysis in coefficient_analysis.items():
        cv = analysis['cv']
        outlier_pct = analysis['outliers']['outlier_count'] / 300 * 100  # 총 300명
        sign_consistency = max(analysis['sign_distribution']['positive_pct'],
                              analysis['sign_distribution']['negative_pct'])
        
        # 안정성 점수 계산 (0-100)
        cv_score = max(0, 100 - cv * 100)  # CV가 낮을수록 높은 점수
        outlier_score = max(0, 100 - outlier_pct * 2)  # 이상치가 적을수록 높은 점수
        sign_score = sign_consistency  # 부호 일관성이 높을수록 높은 점수
        
        stability_score = (cv_score + outlier_score + sign_score) / 3
        
        # 안정성 등급
        if stability_score >= 80:
            stability_grade = "매우 안정"
        elif stability_score >= 60:
            stability_grade = "안정"
        elif stability_score >= 40:
            stability_grade = "보통"
        elif stability_score >= 20:
            stability_grade = "불안정"
        else:
            stability_grade = "매우 불안정"
        
        stability_analysis[var_name] = {
            'cv': cv,
            'outlier_percentage': outlier_pct,
            'sign_consistency': sign_consistency,
            'stability_score': stability_score,
            'stability_grade': stability_grade
        }
        
        print(f"\n   📊 {var_name}:")
        print(f"      변이계수: {cv:.3f}")
        print(f"      이상치 비율: {outlier_pct:.1f}%")
        print(f"      부호 일관성: {sign_consistency:.1f}%")
        print(f"      안정성 점수: {stability_score:.1f}/100 ({stability_grade})")
    
    # 전체 안정성 순위
    sorted_stability = sorted(stability_analysis.items(), 
                            key=lambda x: x[1]['stability_score'], reverse=True)
    
    print(f"\n🏆 계수 안정성 순위:")
    for rank, (var_name, analysis) in enumerate(sorted_stability, 1):
        print(f"   {rank}. {var_name}: {analysis['stability_score']:.1f}점 ({analysis['stability_grade']})")
    
    return stability_analysis


def generate_coefficient_insights(coefficient_analysis, sem_effect_analysis, 
                                magnitude_analysis, stability_analysis):
    """계수 분석 결과에서 주요 인사이트를 도출합니다."""
    
    print(f"\n" + "=" * 60)
    print("5. 주요 인사이트 및 해석")
    print("=" * 60)
    
    insights = {
        'key_findings': [],
        'coefficient_patterns': [],
        'sem_integration_effects': [],
        'practical_implications': [],
        'methodological_insights': []
    }
    
    # 1. 주요 발견사항
    print("🔍 주요 발견사항:")
    
    # 가장 큰 계수
    largest_coeff = max(magnitude_analysis.items(), key=lambda x: x[1]['absolute_magnitude'])
    insights['key_findings'].append(f"가장 큰 효과: {largest_coeff[0]} (계수={largest_coeff[1]['mean_coefficient']:.4f})")
    print(f"   • {insights['key_findings'][-1]}")
    
    # 가장 안정한 계수
    most_stable = max(stability_analysis.items(), key=lambda x: x[1]['stability_score'])
    insights['key_findings'].append(f"가장 안정한 계수: {most_stable[0]} (안정성={most_stable[1]['stability_score']:.1f}점)")
    print(f"   • {insights['key_findings'][-1]}")
    
    # 가장 이질적인 계수
    most_heterogeneous = max(coefficient_analysis.items(), key=lambda x: x[1]['cv'])
    insights['key_findings'].append(f"가장 높은 이질성: {most_heterogeneous[0]} (CV={most_heterogeneous[1]['cv']:.3f})")
    print(f"   • {insights['key_findings'][-1]}")
    
    # 2. 계수 패턴 분석
    print(f"\n📈 계수 패턴:")
    
    # 양수/음수 계수 분포
    positive_coeffs = [var for var, analysis in coefficient_analysis.items() 
                      if analysis['mean'] > 0]
    negative_coeffs = [var for var, analysis in coefficient_analysis.items() 
                      if analysis['mean'] < 0]
    
    insights['coefficient_patterns'].append(f"양수 계수: {len(positive_coeffs)}개 ({', '.join(positive_coeffs)})")
    insights['coefficient_patterns'].append(f"음수 계수: {len(negative_coeffs)}개 ({', '.join(negative_coeffs)})")
    
    for pattern in insights['coefficient_patterns']:
        print(f"   • {pattern}")
    
    # 3. SEM 통합 효과
    print(f"\n🔗 SEM 통합 효과:")
    
    if sem_effect_analysis:
        # 가장 강한 SEM 효과
        strongest_sem_effect = max(
            [(dce_var, analysis['total_sem_effect']) for dce_var, analysis in sem_effect_analysis.items()],
            key=lambda x: x[1]
        )
        insights['sem_integration_effects'].append(
            f"가장 강한 SEM 효과: {strongest_sem_effect[0]} (총 효과={strongest_sem_effect[1]:.3f})"
        )
        
        # SEM 효과가 있는 변수들
        sem_affected_vars = list(sem_effect_analysis.keys())
        insights['sem_integration_effects'].append(
            f"SEM 영향 받는 변수: {len(sem_affected_vars)}개 ({', '.join(sem_affected_vars)})"
        )
        
        for effect in insights['sem_integration_effects']:
            print(f"   • {effect}")
    
    # 4. 실용적 시사점
    print(f"\n💡 실용적 시사점:")
    
    # 마케팅 전략
    if 'sugar_free' in magnitude_analysis and magnitude_analysis['sugar_free']['absolute_magnitude'] > 0.5:
        insights['practical_implications'].append("무설탕 제품이 가장 강한 선호 요인 → 무설탕 마케팅 집중 필요")
    
    if 'sugar_health_interaction' in magnitude_analysis and magnitude_analysis['sugar_health_interaction']['absolute_magnitude'] > 1.0:
        insights['practical_implications'].append("무설탕×건강라벨 시너지 효과 매우 큼 → 조합 제품 개발 우선")
    
    # 개인화 전략
    high_cv_vars = [var for var, analysis in coefficient_analysis.items() if analysis['cv'] > 0.5]
    if high_cv_vars:
        insights['practical_implications'].append(f"높은 개인차 변수({', '.join(high_cv_vars)}) → 개인화 마케팅 필수")
    
    for implication in insights['practical_implications']:
        print(f"   • {implication}")
    
    # 5. 방법론적 인사이트
    print(f"\n🔬 방법론적 인사이트:")
    
    # 정규성 검정 결과
    normal_vars = [var for var, analysis in coefficient_analysis.items() if analysis['is_normal']]
    non_normal_vars = [var for var, analysis in coefficient_analysis.items() if not analysis['is_normal']]
    
    insights['methodological_insights'].append(f"정규분포 따르는 계수: {len(normal_vars)}개")
    insights['methodological_insights'].append(f"비정규분포 계수: {len(non_normal_vars)}개 → 대안 분포 고려 필요")
    
    # 이상치 문제
    high_outlier_vars = [var for var, analysis in stability_analysis.items() 
                        if analysis['outlier_percentage'] > 10]
    if high_outlier_vars:
        insights['methodological_insights'].append(f"이상치 많은 변수({', '.join(high_outlier_vars)}) → 로버스트 추정 필요")
    
    for insight in insights['methodological_insights']:
        print(f"   • {insight}")
    
    return insights


def save_coefficient_analysis_results(coefficient_analysis, sem_effect_analysis, 
                                    magnitude_analysis, stability_analysis, insights):
    """계수 분석 결과를 저장합니다."""
    
    print(f"\n💾 계수 분석 결과 저장:")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 종합 결과 저장
    comprehensive_results = {
        'analysis_info': {
            'analysis_type': 'RPL Coefficient Analysis',
            'analysis_date': datetime.now().isoformat(),
            'description': 'RPL 효용함수 계수 상세 검토 및 분석'
        },
        'coefficient_analysis': coefficient_analysis,
        'sem_effect_analysis': sem_effect_analysis,
        'magnitude_analysis': magnitude_analysis,
        'stability_analysis': stability_analysis,
        'insights': insights
    }
    
    # JSON 파일 저장
    json_file = Path(f"rpl_coefficient_analysis_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 종합 분석: {json_file}")
    
    # 계수 요약 CSV 저장
    summary_data = []
    for var_name, analysis in coefficient_analysis.items():
        summary_data.append({
            'variable': var_name,
            'mean': analysis['mean'],
            'std': analysis['std'],
            'cv': analysis['cv'],
            'min': analysis['min'],
            'max': analysis['max'],
            'stability_score': stability_analysis.get(var_name, {}).get('stability_score', 0),
            'magnitude_rank': magnitude_analysis.get(var_name, {}).get('rank', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = Path(f"rpl_coefficient_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"   ✓ 계수 요약: {summary_file}")
    
    return json_file, summary_file


def main():
    """메인 실행 함수"""
    
    # 1. 데이터 로드
    heterogeneity_df, individual_coeffs_df, comprehensive_data = load_rpl_results()
    
    # 2. 계수 분포 분석
    coefficient_analysis = analyze_coefficient_distributions(heterogeneity_df, individual_coeffs_df)
    
    # 3. SEM 효과 분석
    sem_effect_analysis = analyze_sem_effects(comprehensive_data)
    
    # 4. 계수 크기 비교
    magnitude_analysis = compare_coefficient_magnitudes(coefficient_analysis)
    
    # 5. 계수 안정성 분석
    stability_analysis = analyze_coefficient_stability(coefficient_analysis)
    
    # 6. 인사이트 도출
    insights = generate_coefficient_insights(coefficient_analysis, sem_effect_analysis,
                                           magnitude_analysis, stability_analysis)
    
    # 7. 결과 저장
    json_file, summary_file = save_coefficient_analysis_results(
        coefficient_analysis, sem_effect_analysis, magnitude_analysis, 
        stability_analysis, insights
    )
    
    # 최종 요약
    print(f"\n" + "=" * 80)
    print("RPL 계수 분석 완료")
    print("=" * 80)
    
    print(f"📊 분석 완료:")
    print(f"   ✅ {len(coefficient_analysis)}개 변수 계수 분포 분석")
    print(f"   ✅ {len(sem_effect_analysis)}개 변수 SEM 효과 분석")
    print(f"   ✅ 계수 크기 및 안정성 평가")
    print(f"   ✅ 실용적 인사이트 도출")
    
    print(f"\n🎯 핵심 결과:")
    for finding in insights['key_findings'][:3]:
        print(f"   • {finding}")
    
    print(f"\n📁 생성 파일:")
    print(f"   📄 {json_file}")
    print(f"   📄 {summary_file}")
    
    return True


if __name__ == "__main__":
    main()
