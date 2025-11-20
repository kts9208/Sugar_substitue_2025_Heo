"""
Hessian 문제 심층 진단
y_k/s_k 비율이 매우 큰 이유와 해결책 제시
"""
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def extract_detailed_info(log_file):
    """상세 정보 추출"""
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("="*80)
    print("Hessian 문제 심층 진단")
    print("="*80)
    
    # Major Iteration별 상세 정보
    iterations = re.findall(
        r'\[Major Iteration #(\d+) 완료\](.*?)(?=\[Major Iteration|$)',
        content,
        re.DOTALL
    )
    
    results = []
    
    for iter_num, iter_content in iterations:
        # Hessian 업데이트 정보
        hessian_match = re.search(
            r'Hessian 업데이트 정보:(.*?)Hessian 근사:',
            iter_content,
            re.DOTALL
        )
        
        if hessian_match:
            hess_info = hessian_match.group(1)
            
            # s_k 추출
            sk_norm = float(re.search(r's_k \(파라미터 변화\) norm: ([\d.e+-]+)', hess_info).group(1))
            sk_match = re.search(r's_k 상위 5개: \[([\s\S]*?)\]', hess_info)
            if sk_match:
                sk_values = sk_match.group(1).replace('\n', ' ')
                sk_top5 = [float(x) for x in sk_values.split()]
            else:
                sk_top5 = []

            # y_k 추출
            yk_norm = float(re.search(r'y_k \(gradient 변화\) norm: ([\d.e+-]+)', hess_info).group(1))
            yk_match = re.search(r'y_k 상위 5개: \[([\s\S]*?)\]', hess_info)
            if yk_match:
                yk_values = yk_match.group(1).replace('\n', ' ')
                yk_top5 = [float(x) for x in yk_values.split()]
            else:
                yk_top5 = []
            
            # s_k^T · y_k
            skyk = float(re.search(r's_k\^T · y_k: ([\d.e+-]+)', hess_info).group(1))
            
            # ρ
            rho = float(re.search(r'ρ = 1/\(s_k\^T · y_k\): ([\d.e+-]+)', hess_info).group(1))
            
            results.append({
                'iteration': int(iter_num),
                'sk_norm': sk_norm,
                'yk_norm': yk_norm,
                'sk_top5': sk_top5,
                'yk_top5': yk_top5,
                'skyk': skyk,
                'rho': rho,
                'ratio': yk_norm / sk_norm
            })
    
    return results


def diagnose_problem(results):
    """문제 진단"""
    
    print("\n" + "="*80)
    print("문제 진단 결과")
    print("="*80)
    
    for i, res in enumerate(results):
        print(f"\n{'='*80}")
        print(f"Iteration #{res['iteration']}")
        print(f"{'='*80}")
        
        print(f"\n1. 파라미터 변화 (s_k):")
        print(f"   - Norm: {res['sk_norm']:.6e}")
        print(f"   - 상위 5개: {res['sk_top5']}")
        print(f"   - 최대값: {max(abs(x) for x in res['sk_top5']):.6e}")
        
        print(f"\n2. Gradient 변화 (y_k):")
        print(f"   - Norm: {res['yk_norm']:.6e}")
        print(f"   - 상위 5개: {res['yk_top5']}")
        print(f"   - 최대값: {max(abs(x) for x in res['yk_top5']):.6e}")
        
        print(f"\n3. 비율 분석:")
        print(f"   - y_k/s_k 비율: {res['ratio']:.2f}")
        
        if res['ratio'] > 500:
            print(f"   ❌ 심각: 비율이 매우 큼 (>500)")
            print(f"      → Gradient가 파라미터 변화에 비해 과도하게 큼")
            print(f"      → Hessian이 매우 큰 값으로 근사됨")
        elif res['ratio'] > 100:
            print(f"   ⚠️  경고: 비율이 큼 (>100)")
        
        print(f"\n4. s_k^T · y_k:")
        print(f"   - 값: {res['skyk']:.6e}")
        print(f"   - ρ = 1/(s_k^T · y_k): {res['rho']:.6e}")
        
        if res['rho'] > 0.1:
            print(f"   ⚠️  경고: ρ가 큼 (>0.1) - Hessian 업데이트가 과도함")
        
        # 개별 성분 분석
        print(f"\n5. 성분별 비율:")
        for j in range(min(5, len(res['sk_top5']))):
            if abs(res['sk_top5'][j]) > 1e-10:
                component_ratio = abs(res['yk_top5'][j] / res['sk_top5'][j])
                print(f"   [{j}] y_k/s_k = {res['yk_top5'][j]:.6e} / {res['sk_top5'][j]:.6e} = {component_ratio:.2f}")
                
                if component_ratio > 1000:
                    print(f"       ❌ 이 성분의 비율이 매우 큼!")
            else:
                print(f"   [{j}] s_k ≈ 0, y_k = {res['yk_top5'][j]:.6e}")
    
    # 추세 분석
    print(f"\n{'='*80}")
    print("추세 분석")
    print(f"{'='*80}")
    
    print("\nIteration별 변화:")
    print(f"{'Iter':<6} {'sk_norm':<12} {'yk_norm':<12} {'ratio':<10} {'rho':<12}")
    print("-"*60)
    for res in results:
        print(f"{res['iteration']:<6} {res['sk_norm']:<12.6e} {res['yk_norm']:<12.6e} {res['ratio']:<10.2f} {res['rho']:<12.6e}")
    
    # 감소 추세
    print("\n감소 추세:")
    for i in range(1, len(results)):
        sk_decrease = (results[i-1]['sk_norm'] - results[i]['sk_norm']) / results[i-1]['sk_norm'] * 100
        yk_decrease = (results[i-1]['yk_norm'] - results[i]['yk_norm']) / results[i-1]['yk_norm'] * 100
        
        print(f"  Iter {results[i-1]['iteration']} → {results[i]['iteration']}:")
        print(f"    s_k norm: {sk_decrease:+.1f}% (파라미터 변화 감소)")
        print(f"    y_k norm: {yk_decrease:+.1f}% (gradient 변화 감소)")


def provide_solutions():
    """해결책 제시"""
    
    print(f"\n{'='*80}")
    print("해결책 제시")
    print(f"{'='*80}")
    
    print("\n🔍 문제 요약:")
    print("  1. y_k/s_k 비율이 매우 큼 (690 → 116)")
    print("  2. Gradient 변화가 파라미터 변화에 비해 과도하게 큼")
    print("  3. Hessian이 매우 큰 값으로 근사되어 탐색 방향이 0이 됨")
    
    print("\n💡 해결책:")
    print("\n[방법 1] 파라미터 스케일링 활성화")
    print("  - 현재: 모든 스케일이 1.0으로 고정")
    print("  - 제안: Gradient 크기에 따라 자동 스케일링")
    print("  - 코드: use_parameter_scaling=True")
    
    print("\n[방법 2] Trust Region 방법 사용")
    print("  - L-BFGS-B 대신 Trust Region 방법 사용")
    print("  - 파라미터 변화를 제한하여 안정성 향상")
    print("  - 코드: method='trust-constr'")
    
    print("\n[방법 3] Hessian 주기적 리셋")
    print("  - 일정 iteration마다 Hessian을 초기값(I)으로 리셋")
    print("  - ill-conditioning 방지")
    print("  - 코드: reset_hessian_every=5")
    
    print("\n[방법 4] Line Search 강화")
    print("  - 더 엄격한 line search 조건 사용")
    print("  - 코드: maxls=50 (현재 20)")
    
    print("\n[방법 5] 초기값 개선")
    print("  - 현재: 모든 파라미터 0.1")
    print("  - 제안: 순차추정 결과 사용")
    
    print("\n[권장 조합]")
    print("  1순위: 방법 1 (스케일링) + 방법 3 (Hessian 리셋)")
    print("  2순위: 방법 2 (Trust Region)")
    print("  3순위: 방법 5 (초기값 개선)")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    log_file = project_root / 'results' / 'simultaneous_estimation_log_20251120_192842.txt'
    
    if not log_file.exists():
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        exit(1)
    
    # 상세 정보 추출
    results = extract_detailed_info(log_file)
    
    # 문제 진단
    diagnose_problem(results)
    
    # 해결책 제시
    provide_solutions()
    
    print(f"\n{'='*80}")
    print("진단 완료")
    print(f"{'='*80}")

