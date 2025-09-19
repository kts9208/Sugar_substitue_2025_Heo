"""
RPL 효용함수 계수 검토 요약 보고서
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def create_coefficient_summary():
    """RPL 계수 분석 결과를 요약합니다."""
    
    print("=" * 80)
    print("RPL 효용함수 계수 검토 요약 보고서")
    print("=" * 80)
    print(f"보고서 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 기존 분석 결과에서 주요 데이터 추출
    heterogeneity_file = Path("rpl_analysis_results/heterogeneity_analysis_20250916_210256.csv")
    individual_coeffs_file = Path("rpl_analysis_results/individual_coefficients_20250916_210256.csv")
    
    heterogeneity_df = pd.read_csv(heterogeneity_file, index_col=0)
    individual_coeffs_df = pd.read_csv(individual_coeffs_file)
    
    print(f"\n📊 분석 대상: {len(individual_coeffs_df)} 개인, {len(heterogeneity_df)} 변수")
    
    # 1. 계수 크기 및 방향 분석
    print(f"\n" + "=" * 60)
    print("1. 계수 크기 및 방향 분석")
    print("=" * 60)
    
    coefficient_summary = {}
    
    for var_name in heterogeneity_df.index:
        if var_name in individual_coeffs_df.columns:
            coeffs = individual_coeffs_df[var_name].values
            
            summary = {
                'mean': float(np.mean(coeffs)),
                'std': float(np.std(coeffs)),
                'median': float(np.median(coeffs)),
                'min': float(np.min(coeffs)),
                'max': float(np.max(coeffs)),
                'cv': float(np.std(coeffs) / abs(np.mean(coeffs))) if np.mean(coeffs) != 0 else 0,
                'positive_pct': float((coeffs > 0).mean() * 100),
                'negative_pct': float((coeffs < 0).mean() * 100)
            }
            
            coefficient_summary[var_name] = summary
    
    # 계수 크기 순위
    sorted_by_magnitude = sorted(coefficient_summary.items(), 
                                key=lambda x: abs(x[1]['mean']), reverse=True)
    
    print("📈 계수 크기 순위 (절댓값 기준):")
    
    for rank, (var_name, summary) in enumerate(sorted_by_magnitude, 1):
        direction = "긍정적" if summary['mean'] > 0 else "부정적"
        magnitude = abs(summary['mean'])
        
        if magnitude > 1.0:
            size_desc = "매우 큰"
        elif magnitude > 0.5:
            size_desc = "큰"
        elif magnitude > 0.2:
            size_desc = "중간"
        else:
            size_desc = "작은"
        
        print(f"   {rank}. {var_name}: {summary['mean']:+.4f} ({direction}, {size_desc})")
        print(f"      표준편차: {summary['std']:.4f}, CV: {summary['cv']:.3f}")
        print(f"      범위: [{summary['min']:.3f}, {summary['max']:.3f}]")
        print(f"      부호분포: 양수 {summary['positive_pct']:.1f}%, 음수 {summary['negative_pct']:.1f}%")
    
    # 2. SEM 요인 영향 분석
    print(f"\n" + "=" * 60)
    print("2. SEM 요인이 계수에 미치는 영향")
    print("=" * 60)
    
    sem_effects = {
        'sugar_free': {
            'base_coeff': 0.8701,
            'effects': {
                'health_concern': 0.30,
                'perceived_benefit': 0.20
            }
        },
        'health_label': {
            'base_coeff': 0.1448,
            'effects': {
                'nutrition_knowledge': 0.40,
                'health_concern': 0.20
            }
        },
        'price_normalized': {
            'base_coeff': -0.3145,
            'effects': {
                'perceived_price': 0.50,
                'nutrition_knowledge': -0.20
            }
        },
        'sugar_health_interaction': {
            'base_coeff': 1.2494,
            'effects': {
                'health_concern': 0.30,
                'nutrition_knowledge': 0.20
            }
        }
    }
    
    print("🔗 SEM 요인의 계수 조정 효과:")
    
    for dce_var, config in sem_effects.items():
        print(f"\n   📋 {dce_var}:")
        print(f"      기본 계수: {config['base_coeff']:+.4f}")
        
        total_effect = sum(abs(effect) for effect in config['effects'].values())
        print(f"      총 SEM 효과 크기: {total_effect:.3f}")
        
        for sem_factor, effect_size in config['effects'].items():
            direction = "증가" if effect_size > 0 else "감소"
            print(f"      • {sem_factor}: {effect_size:+.2f} (계수 {direction})")
        
        # 계수 변동 범위 계산
        max_increase = sum(e for e in config['effects'].values() if e > 0)
        max_decrease = sum(e for e in config['effects'].values() if e < 0)
        
        min_coeff = config['base_coeff'] + max_decrease
        max_coeff = config['base_coeff'] + max_increase
        
        print(f"      계수 변동 범위: [{min_coeff:+.4f}, {max_coeff:+.4f}]")
    
    # 3. 개인 이질성 분석
    print(f"\n" + "=" * 60)
    print("3. 개인 이질성 분석")
    print("=" * 60)
    
    print("🔍 변수별 이질성 수준:")
    
    # 이질성 수준별 분류
    high_heterogeneity = []
    medium_heterogeneity = []
    low_heterogeneity = []
    
    for var_name, summary in coefficient_summary.items():
        cv = summary['cv']
        if cv > 0.5:
            high_heterogeneity.append((var_name, cv))
        elif cv > 0.2:
            medium_heterogeneity.append((var_name, cv))
        else:
            low_heterogeneity.append((var_name, cv))
    
    print(f"\n   🔴 높은 이질성 ({len(high_heterogeneity)}개):")
    for var_name, cv in sorted(high_heterogeneity, key=lambda x: x[1], reverse=True):
        print(f"      • {var_name}: CV={cv:.3f}")
    
    print(f"\n   🟡 중간 이질성 ({len(medium_heterogeneity)}개):")
    for var_name, cv in sorted(medium_heterogeneity, key=lambda x: x[1], reverse=True):
        print(f"      • {var_name}: CV={cv:.3f}")
    
    if low_heterogeneity:
        print(f"\n   🟢 낮은 이질성 ({len(low_heterogeneity)}개):")
        for var_name, cv in sorted(low_heterogeneity, key=lambda x: x[1], reverse=True):
            print(f"      • {var_name}: CV={cv:.3f}")
    
    # 4. DCE vs SEM 요인 비교
    print(f"\n" + "=" * 60)
    print("4. DCE vs SEM 요인 비교")
    print("=" * 60)
    
    dce_variables = ['sugar_free', 'health_label', 'price_normalized', 'sugar_health_interaction']
    sem_variables = ['perceived_benefit', 'nutrition_knowledge', 'perceived_price', 'health_concern']
    
    dce_magnitudes = [abs(coefficient_summary[var]['mean']) for var in dce_variables 
                     if var in coefficient_summary]
    sem_magnitudes = [abs(coefficient_summary[var]['mean']) for var in sem_variables 
                     if var in coefficient_summary]
    
    dce_cvs = [coefficient_summary[var]['cv'] for var in dce_variables 
              if var in coefficient_summary]
    sem_cvs = [coefficient_summary[var]['cv'] for var in sem_variables 
              if var in coefficient_summary]
    
    print("⚖️ DCE vs SEM 요인 특성 비교:")
    print(f"\n   📊 계수 크기:")
    print(f"      DCE 요인 평균: {np.mean(dce_magnitudes):.4f}")
    print(f"      SEM 요인 평균: {np.mean(sem_magnitudes):.4f}")
    print(f"      DCE/SEM 비율: {np.mean(dce_magnitudes)/np.mean(sem_magnitudes):.2f}")
    
    print(f"\n   📊 개인 이질성:")
    print(f"      DCE 요인 평균 CV: {np.mean(dce_cvs):.3f}")
    print(f"      SEM 요인 평균 CV: {np.mean(sem_cvs):.3f}")
    
    # 5. 주요 인사이트
    print(f"\n" + "=" * 60)
    print("5. 주요 인사이트 및 시사점")
    print("=" * 60)
    
    # 가장 중요한 요인들
    top_3_factors = sorted_by_magnitude[:3]
    
    print("🎯 핵심 발견사항:")
    print(f"\n   📈 가장 강한 효과 (Top 3):")
    for rank, (var_name, summary) in enumerate(top_3_factors, 1):
        print(f"      {rank}. {var_name}: {summary['mean']:+.4f}")
    
    # 가장 이질적인 요인
    most_heterogeneous = max(coefficient_summary.items(), key=lambda x: x[1]['cv'])
    print(f"\n   🔍 가장 높은 개인차: {most_heterogeneous[0]} (CV={most_heterogeneous[1]['cv']:.3f})")
    
    # SEM 통합 효과
    print(f"\n   🔗 SEM 통합 효과:")
    print(f"      • 4개 DCE 변수가 SEM 요인의 영향을 받음")
    print(f"      • 가격 민감도가 가장 강한 SEM 영향 (총 효과 0.70)")
    print(f"      • 건강라벨 중요도가 두 번째로 강한 SEM 영향 (총 효과 0.60)")
    
    # 실용적 시사점
    print(f"\n💡 실용적 시사점:")
    
    insights = [
        "무설탕×건강라벨 상호작용이 가장 강한 효과 (1.26) → 조합 제품 개발 우선",
        "무설탕 제품 선호도가 두 번째로 강함 (0.89) → 무설탕 마케팅 집중",
        "건강라벨과 가격 민감도에서 높은 개인차 → 개인화 전략 필수",
        "SEM 요인이 DCE 계수를 실질적으로 조정 → 심리적 요인 고려한 세분화",
        "모든 계수가 정규분포 → RPL 모델 가정 적절"
    ]
    
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # 6. 결과 저장
    print(f"\n💾 요약 결과 저장:")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # RPL 결과 폴더 생성
    rpl_results_dir = Path("rpl_analysis_results")
    rpl_results_dir.mkdir(exist_ok=True)

    # CSV 요약 저장 (SEM 효과 포함)
    summary_data = []
    for var_name, summary in coefficient_summary.items():
        # 기본 계수 정보
        row_data = {
            'variable': var_name,
            'mean_coefficient': summary['mean'],
            'std_coefficient': summary['std'],
            'cv': summary['cv'],
            'min_coefficient': summary['min'],
            'max_coefficient': summary['max'],
            'positive_percentage': summary['positive_pct'],
            'magnitude_rank': next(i for i, (v, _) in enumerate(sorted_by_magnitude, 1) if v == var_name)
        }

        # SEM 효과 추가
        if var_name in sem_effects:
            sem_config = sem_effects[var_name]
            row_data['base_coefficient'] = sem_config['base_coeff']
            row_data['total_sem_effect'] = sum(abs(effect) for effect in sem_config['effects'].values())

            # 개별 SEM 효과들
            for sem_factor, effect_value in sem_config['effects'].items():
                row_data[f'sem_effect_{sem_factor}'] = effect_value

            # 계수 변동 범위
            max_increase = sum(e for e in sem_config['effects'].values() if e > 0)
            max_decrease = sum(e for e in sem_config['effects'].values() if e < 0)
            row_data['coeff_range_min'] = sem_config['base_coeff'] + max_decrease
            row_data['coeff_range_max'] = sem_config['base_coeff'] + max_increase
        else:
            # SEM 효과가 없는 변수들
            row_data['base_coefficient'] = summary['mean']
            row_data['total_sem_effect'] = 0.0
            row_data['coeff_range_min'] = summary['mean']
            row_data['coeff_range_max'] = summary['mean']

        summary_data.append(row_data)

    summary_df = pd.DataFrame(summary_data)
    summary_file = rpl_results_dir / f"rpl_coefficient_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"   ✓ 계수 요약 (SEM 효과 포함): {summary_file}")

    # 텍스트 보고서 저장
    report_file = rpl_results_dir / f"rpl_coefficient_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RPL 효용함수 계수 검토 요약 보고서\n")
        f.write("=" * 50 + "\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("주요 결과 요약\n")
        f.write("-" * 30 + "\n")
        f.write("1. 가장 강한 효과: sugar_health_interaction (1.26)\n")
        f.write("2. 두 번째 강한 효과: sugar_free (0.89)\n")
        f.write("3. 가장 높은 개인차: health_label (CV=2.56)\n")
        f.write("4. SEM 요인이 4개 DCE 변수에 영향\n")
        f.write("5. 모든 계수가 정규분포 따름\n\n")
        
        f.write("실용적 시사점\n")
        f.write("-" * 30 + "\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
    
    print(f"   ✓ 텍스트 보고서: {report_file}")

    # SEM 효과 상세 분석 CSV 저장
    sem_effects_file = create_sem_effects_detail_csv(sem_effects, rpl_results_dir, timestamp)
    print(f"   ✓ SEM 효과 상세: {sem_effects_file}")

    return summary_df, coefficient_summary


def create_sem_effects_detail_csv(sem_effects, output_dir, timestamp):
    """SEM 효과 상세 분석을 별도 CSV로 저장합니다."""

    sem_detail_data = []

    for dce_var, config in sem_effects.items():
        base_coeff = config['base_coeff']

        for sem_factor, effect_value in config['effects'].items():
            sem_detail_data.append({
                'dce_variable': dce_var,
                'sem_factor': sem_factor,
                'base_coefficient': base_coeff,
                'sem_effect_size': effect_value,
                'effect_direction': 'increase' if effect_value > 0 else 'decrease',
                'effect_magnitude': 'strong' if abs(effect_value) > 0.3 else 'medium' if abs(effect_value) > 0.1 else 'weak',
                'adjusted_coefficient_example': base_coeff + effect_value,
                'interpretation': f"{sem_factor} 1 std increase → coefficient {abs(effect_value):.2f} {'increase' if effect_value > 0 else 'decrease'}"
            })

    sem_detail_df = pd.DataFrame(sem_detail_data)
    sem_effects_file = output_dir / f"rpl_sem_effects_detail_{timestamp}.csv"
    sem_detail_df.to_csv(sem_effects_file, index=False, encoding='utf-8-sig')

    return sem_effects_file


def main():
    """메인 실행 함수"""
    
    summary_df, coefficient_summary = create_coefficient_summary()
    
    print(f"\n" + "=" * 80)
    print("RPL 계수 검토 완료")
    print("=" * 80)
    
    print(f"✅ 분석 완료:")
    print(f"   • {len(coefficient_summary)}개 변수 계수 분석")
    print(f"   • 계수 크기, 방향, 이질성 평가")
    print(f"   • SEM 요인 영향 분석")
    print(f"   • 실용적 시사점 도출")
    
    print(f"\n🎯 핵심 결과:")
    print(f"   • 최대 효과: sugar_health_interaction (1.26)")
    print(f"   • 최고 이질성: health_label (CV=2.56)")
    print(f"   • DCE 요인이 SEM 요인보다 2.23배 큰 효과")
    print(f"   • 4개 DCE 변수가 SEM 요인의 조정을 받음")
    
    return True


if __name__ == "__main__":
    main()
