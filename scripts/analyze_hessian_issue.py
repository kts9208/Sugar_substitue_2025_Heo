"""
Hessian 근사 문제 분석 스크립트
로그 파일에서 Hessian 업데이트 정보를 추출하여 문제점 진단
"""
import re
import numpy as np
from pathlib import Path
import pandas as pd

def analyze_hessian_updates(log_file):
    """Hessian 업데이트 정보 분석"""
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("="*80)
    print("Hessian 근사 문제 분석")
    print("="*80)
    
    # Major Iteration 정보 추출
    iterations = re.findall(
        r'\[Major Iteration #(\d+) 완료\](.*?)(?=\[Major Iteration|$)',
        content,
        re.DOTALL
    )
    
    hessian_data = []
    
    for iter_num, iter_content in iterations:
        print(f"\n{'='*80}")
        print(f"Iteration #{iter_num} 분석")
        print(f"{'='*80}")
        
        # 최종 LL 추출
        ll_match = re.search(r'최종 LL: ([-\d.]+)', iter_content)
        final_ll = float(ll_match.group(1)) if ll_match else None
        
        # Line Search 정보
        line_search_match = re.search(r'Line Search: (\d+)회 함수 호출 - \[(.*?)\]', iter_content)
        if line_search_match:
            ls_calls = int(line_search_match.group(1))
            ls_status = line_search_match.group(2)
            print(f"  Line Search: {ls_calls}회, 상태: {ls_status}")
        
        # 수렴 조건
        ftol_match = re.search(r'ftol = ([\d.e+-]+|N/A)', iter_content)
        gtol_match = re.search(r'gtol = ([\d.e+-]+)', iter_content)
        
        if ftol_match:
            ftol = ftol_match.group(1)
            print(f"  ftol: {ftol} (기준: 1e-3)")
        
        if gtol_match:
            gtol = float(gtol_match.group(1))
            print(f"  gtol: {gtol:.6e} (기준: 1e-3)")
        
        # Hessian 업데이트 정보
        hessian_section = re.search(
            r'Hessian 업데이트.*?(?:정보:|첫 iteration)(.*?)(?:Hessian 근사:|$)',
            iter_content,
            re.DOTALL
        )
        
        if hessian_section:
            hess_info = hessian_section.group(1)
            
            # s_k norm (파라미터 변화)
            sk_norm_match = re.search(r's_k \(파라미터 변화\) norm: ([\d.e+-]+)', hess_info)
            # y_k norm (gradient 변화)
            yk_norm_match = re.search(r'y_k \(gradient 변화\) norm: ([\d.e+-]+)', hess_info)
            # s_k^T · y_k
            skyk_match = re.search(r's_k\^T · y_k: ([\d.e+-]+)', hess_info)
            # ρ
            rho_match = re.search(r'ρ = 1/\(s_k\^T · y_k\): ([\d.e+-]+)', hess_info)
            
            if sk_norm_match:
                sk_norm = float(sk_norm_match.group(1))
                print(f"\n  [Hessian 업데이트 정보]")
                print(f"    s_k norm (파라미터 변화): {sk_norm:.6e}")
                
                if yk_norm_match:
                    yk_norm = float(yk_norm_match.group(1))
                    print(f"    y_k norm (gradient 변화): {yk_norm:.6e}")
                    
                    # 비율 계산
                    ratio = yk_norm / sk_norm if sk_norm > 0 else float('inf')
                    print(f"    y_k/s_k 비율: {ratio:.6e}")
                    
                    if ratio > 1e6:
                        print(f"    ⚠️  경고: 비율이 매우 큼 - Hessian이 ill-conditioned 가능성")
                    elif ratio < 1e-6:
                        print(f"    ⚠️  경고: 비율이 매우 작음 - 평탄한 영역")
                
                if skyk_match:
                    skyk = float(skyk_match.group(1))
                    print(f"    s_k^T · y_k: {skyk:.6e}")
                    
                    if skyk <= 0:
                        print(f"    ❌ 심각: s_k^T · y_k ≤ 0 - BFGS 업데이트 불가능!")
                    elif skyk < 1e-10:
                        print(f"    ⚠️  경고: s_k^T · y_k가 매우 작음 - 수치적 불안정")
                
                if rho_match:
                    rho = float(rho_match.group(1))
                    print(f"    ρ: {rho:.6e}")
                    
                    if rho > 1e6:
                        print(f"    ⚠️  경고: ρ가 매우 큼 - Hessian 업데이트가 과도함")
                
                hessian_data.append({
                    'iteration': int(iter_num),
                    'final_ll': final_ll,
                    'sk_norm': sk_norm,
                    'yk_norm': yk_norm if yk_norm_match else None,
                    'skyk': skyk if skyk_match else None,
                    'rho': rho if rho_match else None,
                    'ratio': ratio if yk_norm_match else None
                })
        else:
            print(f"\n  [Hessian 정보] 첫 iteration (H = I)")
    
    # 탐색 방향 분석
    print(f"\n{'='*80}")
    print("탐색 방향 분석")
    print(f"{'='*80}")
    
    search_directions = re.findall(
        r'\[탐색 방향 분석 - Iteration #(\d+)\](.*?)(?=\[탐색 방향 분석|\[Major Iteration|$)',
        content,
        re.DOTALL
    )
    
    for iter_num, direction_info in search_directions[:5]:  # 처음 5개만
        d_norm_match = re.search(r'탐색 방향 d norm: ([\d.e+-]+)', direction_info)
        grad_norm_match = re.search(r'Gradient norm: ([\d.e+-]+)', direction_info)
        cosine_match = re.search(r'd와 -grad의 코사인 유사도: ([\d.e+-]+)', direction_info)
        
        if d_norm_match:
            d_norm = float(d_norm_match.group(1))
            grad_norm = float(grad_norm_match.group(1)) if grad_norm_match else None
            cosine = float(cosine_match.group(1)) if cosine_match else None
            
            print(f"\nIteration #{iter_num}:")
            print(f"  탐색 방향 norm: {d_norm:.6e}")
            if grad_norm:
                print(f"  Gradient norm: {grad_norm:.6e}")
            if cosine is not None:
                print(f"  코사인 유사도: {cosine:.6f}")
                
                if d_norm == 0:
                    print(f"  ❌ 심각: 탐색 방향이 0 - 최적화 중단!")
                elif cosine < 0.1:
                    print(f"  ⚠️  경고: 코사인 유사도가 낮음 - Hessian이 잘못된 방향 제시")
    
    # 데이터프레임 생성
    if hessian_data:
        df = pd.DataFrame(hessian_data)
        print(f"\n{'='*80}")
        print("Hessian 업데이트 요약")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        # 저장
        output_file = log_file.parent / 'hessian_analysis.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ 분석 결과 저장: {output_file}")
    
    return hessian_data


def check_parameter_bounds(log_file):
    """파라미터가 bounds에 걸렸는지 확인"""
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"\n{'='*80}")
    print("파라미터 Bounds 체크")
    print(f"{'='*80}")
    
    # 파라미터 값 추출
    param_sections = re.findall(
        r'\[Major Iteration #(\d+) 완료\](.*?)전체 파라미터 값 및 그래디언트:(.*?)(?=Hessian|$)',
        content,
        re.DOTALL
    )
    
    for iter_num, _, param_info in param_sections[:5]:
        print(f"\nIteration #{iter_num}:")
        
        # 파라미터 라인 추출
        param_lines = re.findall(
            r'\[\s*\d+\]\s+(\S+)\s*:\s*param=([\d.e+-]+),\s*grad=([\d.e+-]+)',
            param_info
        )
        
        for param_name, param_val, grad_val in param_lines:
            param_val = float(param_val)
            grad_val = float(grad_val)
            
            # Bounds 체크 (일반적인 bounds)
            if 'theta' in param_name or 'lambda' in param_name:
                # 분산 파라미터는 0 근처에서 문제
                if abs(param_val) < 1e-6:
                    print(f"  ⚠️  {param_name}: {param_val:.6e} (거의 0)")
            
            # Gradient가 큰데 파라미터가 안 움직이는 경우
            if abs(grad_val) > 10 and abs(param_val) < 0.01:
                print(f"  ⚠️  {param_name}: param={param_val:.6e}, grad={grad_val:.6e} (큰 gradient, 작은 param)")


if __name__ == "__main__":
    # 로그 파일 경로
    project_root = Path(__file__).parent.parent
    log_file = project_root / 'results' / 'simultaneous_estimation_log_20251120_192842.txt'
    
    if not log_file.exists():
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        exit(1)
    
    print(f"📂 로그 파일: {log_file.name}\n")
    
    # Hessian 분석
    hessian_data = analyze_hessian_updates(log_file)
    
    # Bounds 체크
    check_parameter_bounds(log_file)
    
    print(f"\n{'='*80}")
    print("분석 완료")
    print(f"{'='*80}")

