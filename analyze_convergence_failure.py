"""
동시추정 수렴 실패 원인 분석 스크립트
로그 파일: simultaneous_estimation_log_20251123_114842.txt
"""

import re
from pathlib import Path
import pandas as pd

def parse_log_file(log_path):
    """로그 파일 파싱"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Major Iteration 정보 추출
    iterations = []
    pattern = r'\[Major Iteration #(\d+) 완료\]\s+최종 LL: ([-\d.]+)\s+Line Search: (\d+)회 함수 호출 - \[(.*?)\].*?\n.*?ftol = ([\d.eE+-]+|N/A).*?\n.*?gtol = ([\d.eE+-]+)'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        iter_num = int(match.group(1))
        ll = float(match.group(2))
        line_search_calls = int(match.group(3))
        line_search_status = match.group(4)
        ftol_str = match.group(5)
        gtol = float(match.group(6))
        
        ftol = None if ftol_str == 'N/A' else float(ftol_str)
        
        iterations.append({
            'iteration': iter_num,
            'll': ll,
            'line_search_calls': line_search_calls,
            'line_search_status': line_search_status,
            'ftol': ftol,
            'gtol': gtol
        })
    
    return pd.DataFrame(iterations)

def analyze_convergence_failure(df):
    """수렴 실패 원인 분석"""
    print("=" * 80)
    print("동시추정 수렴 실패 원인 분석")
    print("=" * 80)
    print()
    
    # 1. 전체 추정 진행 상황
    print("[1] 전체 추정 진행 상황")
    print("-" * 80)
    print(f"총 Iteration 수: {len(df)}")
    print(f"초기 LL: {df.iloc[0]['ll']:.4f}")
    print(f"최종 LL: {df.iloc[-1]['ll']:.4f}")
    print(f"LL 개선: {df.iloc[0]['ll'] - df.iloc[-1]['ll']:.4f}")
    print(f"LL 개선률: {(df.iloc[0]['ll'] - df.iloc[-1]['ll']) / abs(df.iloc[0]['ll']) * 100:.2f}%")
    print()
    
    # 2. Line Search 실패 분석
    print("[2] Line Search 실패 분석")
    print("-" * 80)
    fail_count = (df['line_search_status'] == 'FAIL').sum()
    warn_count = (df['line_search_status'] == 'WARN').sum()
    ok_count = (df['line_search_status'] == 'OK').sum()
    
    print(f"Line Search 성공 (OK): {ok_count}회 ({ok_count/len(df)*100:.1f}%)")
    print(f"Line Search 실패 (FAIL): {fail_count}회 ({fail_count/len(df)*100:.1f}%)")
    print(f"Line Search 정체 (WARN): {warn_count}회 ({warn_count/len(df)*100:.1f}%)")
    print()
    
    # 3. 수렴 조건 분석
    print("[3] 수렴 조건 분석")
    print("-" * 80)
    ftol_threshold = 1e-3
    gtol_threshold = 1e-3
    
    df_with_ftol = df[df['ftol'].notna()]
    ftol_satisfied = (df_with_ftol['ftol'] <= ftol_threshold).sum()
    gtol_satisfied = (df['gtol'] <= gtol_threshold).sum()
    
    print(f"ftol 조건 만족 (≤ {ftol_threshold}): {ftol_satisfied}/{len(df_with_ftol)}회")
    print(f"gtol 조건 만족 (≤ {gtol_threshold}): {gtol_satisfied}/{len(df)}회")
    print()
    print(f"최종 ftol: {df.iloc[-1]['ftol']:.6e} (기준: {ftol_threshold:.0e})")
    print(f"최종 gtol: {df.iloc[-1]['gtol']:.6e} (기준: {gtol_threshold:.0e})")
    print()
    
    # 4. LL 변화 추이
    print("[4] LL 변화 추이")
    print("-" * 80)
    df['ll_change'] = df['ll'].diff()
    df['ll_change_pct'] = df['ll_change'] / df['ll'].shift(1).abs() * 100
    
    print("Iteration별 LL 변화:")
    for idx, row in df.iterrows():
        if idx == 0:
            print(f"  Iter {row['iteration']:2d}: LL = {row['ll']:10.4f} (초기값)")
        else:
            change = row['ll_change']
            change_pct = row['ll_change_pct']
            status = "개선" if change > 0 else "악화"
            print(f"  Iter {row['iteration']:2d}: LL = {row['ll']:10.4f} (변화: {change:+8.4f}, {change_pct:+6.2f}%, {status})")
    print()
    
    # 5. 수렴 실패 원인 후보군
    print("=" * 80)
    print("[수렴 실패 원인 후보군]")
    print("=" * 80)
    print()
    
    reasons = []
    
    # 원인 1: Line Search 실패/정체
    if fail_count + warn_count > len(df) * 0.5:
        reasons.append({
            'priority': 'HIGH',
            'reason': 'Line Search 실패/정체 빈번',
            'evidence': f'{fail_count + warn_count}/{len(df)}회 ({(fail_count + warn_count)/len(df)*100:.1f}%)',
            'explanation': 'Line Search가 Wolfe 조건을 만족하는 step size를 찾지 못함. 탐색 방향이 부적절하거나 함수가 평탄한 영역에 도달했을 가능성.'
        })
    
    # 원인 2: gtol 조건 불만족
    if df.iloc[-1]['gtol'] > gtol_threshold:
        reasons.append({
            'priority': 'HIGH',
            'reason': 'Gradient norm이 수렴 기준을 만족하지 못함',
            'evidence': f"최종 gtol = {df.iloc[-1]['gtol']:.2e} (기준: {gtol_threshold:.0e})",
            'explanation': '그래디언트가 충분히 0에 가까워지지 않음. 최적점에 도달하지 못했거나, 수치적 불안정성이 있을 가능성.'
        })
    
    # 원인 3: LL 변화 정체
    recent_ll_changes = df.tail(5)['ll_change'].abs()
    if recent_ll_changes.mean() < 5:
        reasons.append({
            'priority': 'MEDIUM',
            'reason': 'LL 변화 정체 (평탄한 영역)',
            'evidence': f"최근 5회 평균 LL 변화: {recent_ll_changes.mean():.4f}",
            'explanation': '함수값이 거의 변하지 않는 평탄한 영역에 도달. Local minimum이거나 saddle point일 가능성.'
        })
    
    # 원인 4: LL 악화 빈번
    ll_deterioration = (df['ll_change'] < 0).sum()
    if ll_deterioration > len(df) * 0.3:
        reasons.append({
            'priority': 'MEDIUM',
            'reason': 'LL 악화 빈번',
            'evidence': f'{ll_deterioration}/{len(df)-1}회 ({ll_deterioration/(len(df)-1)*100:.1f}%)',
            'explanation': 'Line Search가 적절한 step size를 찾지 못해 함수값이 악화됨. Hessian 근사가 부정확할 가능성.'
        })
    
    # 원인 5: 초기값 문제
    if df.iloc[0]['ll'] < -2500:
        reasons.append({
            'priority': 'LOW',
            'reason': '초기 LL이 매우 낮음',
            'evidence': f"초기 LL = {df.iloc[0]['ll']:.4f}",
            'explanation': '초기값이 최적점에서 멀리 떨어져 있을 가능성. 더 나은 초기값 필요.'
        })
    
    # 출력
    for i, reason in enumerate(reasons, 1):
        print(f"[{reason['priority']}] 원인 {i}: {reason['reason']}")
        print(f"  증거: {reason['evidence']}")
        print(f"  설명: {reason['explanation']}")
        print()
    
    return reasons

def main():
    log_path = Path('results/final/simultaneous/logs/simultaneous_estimation_log_20251123_114842.txt')
    
    if not log_path.exists():
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_path}")
        return
    
    print(f"로그 파일: {log_path.name}")
    print(f"파일 크기: {log_path.stat().st_size / 1024:.1f} KB")
    print()
    
    df = parse_log_file(log_path)
    reasons = analyze_convergence_failure(df)
    
    # CSV 저장
    output_path = log_path.parent / f"{log_path.stem}_convergence_analysis.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 분석 결과 저장: {output_path.name}")

if __name__ == '__main__':
    main()

