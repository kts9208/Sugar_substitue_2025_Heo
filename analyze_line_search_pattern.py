"""
Line Search 패턴 분석 스크립트
Iteration #11의 40회 Line Search 시도를 분석하여 maxls 증가의 효과를 예측
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

def parse_line_search_attempts(log_path):
    """Iteration #11의 Line Search 시도 파싱"""
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Iteration #12의 Line Search 시도 찾기
    attempts = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # [Line Search 함수 호출 #iter12-X] 패턴 찾기
        match = re.search(r'\[Line Search 함수 호출 #iter12-(\d+)\]', line)
        if match:
            attempt_num = int(match.group(1))

            # 다음 몇 줄에서 LL과 파라미터 변화량 찾기
            for j in range(i+1, min(i+5, len(lines))):
                ll_match = re.search(r'LL = ([-\d.]+)', lines[j])
                if ll_match:
                    ll = float(ll_match.group(1))

                    # 파라미터 변화량과 함수값 변화 찾기
                    for k in range(j+1, min(j+3, len(lines))):
                        param_match = re.search(r'파라미터 변화량 \(L2 norm\): ([\d.eE+-]+)', lines[k])
                        if param_match:
                            param_change = float(param_match.group(1))

                            # 함수값 변화 찾기
                            for m in range(k, min(k+2, len(lines))):
                                ll_change_match = re.search(r'함수값 변화: ([-+\d.]+)', lines[m])
                                if ll_change_match:
                                    ll_change = float(ll_change_match.group(1))

                                    attempts.append({
                                        'attempt': attempt_num,
                                        'll': ll,
                                        'param_change': param_change,
                                        'll_change': ll_change
                                    })
                                    break
                            break
                    break

        i += 1

    return pd.DataFrame(attempts)

def analyze_line_search_pattern(df):
    """Line Search 패턴 분석"""
    print("=" * 80)
    print("Iteration #11 Line Search 패턴 분석")
    print("=" * 80)
    print()
    
    # 기본 통계
    print("[1] 기본 통계")
    print("-" * 80)
    print(f"총 Line Search 시도: {len(df)}회")
    print(f"최선 LL: {df['ll'].min():.4f}")
    print(f"최악 LL: {df['ll'].max():.4f}")
    print(f"LL 범위: {df['ll'].max() - df['ll'].min():.4f}")
    print()
    
    # 파라미터 변화량 분석
    print("[2] 파라미터 변화량 (L2 norm) 분석")
    print("-" * 80)
    print(f"최소: {df['param_change'].min():.6e}")
    print(f"최대: {df['param_change'].max():.6e}")
    print(f"평균: {df['param_change'].mean():.6e}")
    print(f"중앙값: {df['param_change'].median():.6e}")
    print()
    
    # 함수값 변화 분석
    print("[3] 함수값 변화 분석")
    print("-" * 80)
    improvement_count = (df['ll_change'] > 0).sum()
    deterioration_count = (df['ll_change'] < 0).sum()
    
    print(f"개선 (LL 증가): {improvement_count}회 ({improvement_count/len(df)*100:.1f}%)")
    print(f"악화 (LL 감소): {deterioration_count}회 ({deterioration_count/len(df)*100:.1f}%)")
    print()
    
    # 수렴 패턴 분석
    print("[4] 수렴 패턴 분석")
    print("-" * 80)
    
    # 파라미터 변화량이 수렴하는지 확인
    df['param_change_diff'] = df['param_change'].diff().abs()
    
    # 최근 10회의 파라미터 변화량 표준편차
    recent_10 = df.tail(10)
    param_std = recent_10['param_change'].std()
    ll_std = recent_10['ll_change'].std()
    
    print(f"최근 10회 파라미터 변화량 표준편차: {param_std:.6e}")
    print(f"최근 10회 LL 변화 표준편차: {ll_std:.6e}")
    print()
    
    # 수렴 여부 판단
    if param_std < 1e-8:
        print("✅ 파라미터 변화량이 수렴함 (표준편차 < 1e-8)")
    else:
        print("❌ 파라미터 변화량이 수렴하지 않음")
    
    if ll_std < 1e-3:
        print("✅ LL 변화가 수렴함 (표준편차 < 1e-3)")
    else:
        print("❌ LL 변화가 수렴하지 않음")
    print()
    
    # 상세 출력
    print("[5] Line Search 시도별 상세 정보")
    print("-" * 80)
    print(f"{'시도':>4} {'LL':>12} {'파라미터 변화':>15} {'LL 변화':>12} {'상태':>8}")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        status = "개선" if row['ll_change'] > 0 else "악화"
        print(f"{row['attempt']:4d} {row['ll']:12.4f} {row['param_change']:15.6e} {row['ll_change']:+12.4f} {status:>8}")
    
    print()
    
    # 패턴 분석
    print("=" * 80)
    print("[패턴 분석 결과]")
    print("=" * 80)
    print()
    
    # 시도 1-10: 파라미터 변화량이 점진적으로 수렴
    first_10 = df.head(10)
    if len(first_10) >= 10:
        param_convergence = first_10['param_change'].iloc[-1] - first_10['param_change'].iloc[0]
        print(f"[시도 1-10] 파라미터 변화량 수렴 패턴:")
        print(f"  시작: {first_10['param_change'].iloc[0]:.6e}")
        print(f"  종료: {first_10['param_change'].iloc[-1]:.6e}")
        print(f"  변화: {param_convergence:.6e}")
        print()
    
    # 시도 11-20: 파라미터 변화량이 거의 동일
    if len(df) >= 20:
        attempts_11_20 = df.iloc[10:20]
        param_std_11_20 = attempts_11_20['param_change'].std()
        print(f"[시도 11-20] 파라미터 변화량 표준편차: {param_std_11_20:.6e}")
        if param_std_11_20 < 1e-10:
            print("  → 파라미터 변화량이 거의 동일함 (수렴)")
        print()
    
    # 시도 21-40: 큰 변화 시도
    if len(df) >= 40:
        attempts_21_40 = df.iloc[20:40]
        large_changes = attempts_21_40[attempts_21_40['param_change'] > 0.1]
        print(f"[시도 21-40] 큰 파라미터 변화 시도:")
        print(f"  큰 변화 (>0.1) 시도: {len(large_changes)}회")
        if len(large_changes) > 0:
            print(f"  최대 파라미터 변화: {large_changes['param_change'].max():.6e}")
            print(f"  해당 시도의 LL 변화: {large_changes.loc[large_changes['param_change'].idxmax(), 'll_change']:.4f}")
        print()
    
    # 결론
    print("=" * 80)
    print("[결론: maxls 증가의 효과 예측]")
    print("=" * 80)
    print()
    
    # 수렴 패턴 확인
    if param_std < 1e-8 and ll_std < 1e-3:
        print("❌ **maxls 증가는 효과가 없을 것으로 예상됩니다.**")
        print()
        print("이유:")
        print("  1. 파라미터 변화량이 이미 수렴함 (표준편차 < 1e-8)")
        print("  2. LL 변화가 이미 수렴함 (표준편차 < 1e-3)")
        print("  3. 시도 11-40에서 파라미터 변화량이 거의 동일함")
        print("  4. 더 많은 시도를 해도 동일한 결과가 반복될 가능성이 높음")
        print()
        print("근본 원인:")
        print("  - 함수가 평탄한 영역에 도달하여 어떤 step size도 개선을 가져오지 못함")
        print("  - Hessian 근사가 부정확하여 탐색 방향 자체가 잘못되었을 가능성")
        print()
    else:
        print("✅ **maxls 증가가 효과가 있을 수 있습니다.**")
        print()
        print("이유:")
        print("  1. 파라미터 변화량이 아직 수렴하지 않음")
        print("  2. LL 변화가 아직 수렴하지 않음")
        print("  3. 더 많은 시도를 통해 더 나은 step size를 찾을 가능성이 있음")
        print()
    
    return df

def main():
    log_path = Path('results/final/simultaneous/logs/simultaneous_estimation_log_20251123_114842.txt')
    
    if not log_path.exists():
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_path}")
        return
    
    print(f"로그 파일: {log_path.name}")
    print()
    
    df = parse_line_search_attempts(log_path)
    
    if len(df) == 0:
        print("❌ Line Search 시도를 찾을 수 없습니다.")
        return
    
    df = analyze_line_search_pattern(df)
    
    # CSV 저장
    output_path = log_path.parent / f"{log_path.stem}_line_search_analysis.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 분석 결과 저장: {output_path.name}")

if __name__ == '__main__':
    main()

