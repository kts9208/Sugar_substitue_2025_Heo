"""
GPU Analytic Gradient 문제점 진단 스크립트

현재 구현의 문제점을 자동으로 검사합니다.
"""

import sys
import os
from pathlib import Path
import re

# 프로젝트 루트
project_root = Path(__file__).parent.parent

def check_file_content(file_path, pattern, description):
    """파일에서 패턴 검색"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(pattern, content, re.MULTILINE)
            return matches
    except Exception as e:
        print(f"  ❌ 파일 읽기 실패: {e}")
        return None


def diagnose_gpu_gradient():
    """GPU gradient 구현 진단"""
    
    print("="*70)
    print("GPU Analytic Gradient 문제점 진단")
    print("="*70)
    
    gpu_grad_file = project_root / 'src' / 'analysis' / 'hybrid_choice_model' / 'iclv_models' / 'gpu_gradient_batch.py'
    multi_grad_file = project_root / 'src' / 'analysis' / 'hybrid_choice_model' / 'iclv_models' / 'multi_latent_gradient.py'
    
    problems = []
    
    # 문제 1: Importance Weighting 누락
    print("\n[문제 1] Importance Weighting 누락 검사...")
    
    # GPU 파일에서 'weights' 또는 'importance' 검색
    weights_pattern = r'(weights|importance|weighting)'
    matches = check_file_content(gpu_grad_file, weights_pattern, "Importance weighting")
    
    if not matches or len(matches) == 0:
        print("  🔴 CRITICAL: Importance weighting 코드 없음")
        problems.append({
            'severity': 'CRITICAL',
            'problem': 'Importance weighting 누락',
            'file': 'gpu_gradient_batch.py',
            'description': 'GPU 버전은 모든 draws를 동등하게 취급 (단순 합산)'
        })
    else:
        print(f"  ⚠️  WARNING: 'weights' 키워드 발견 ({len(matches)}회), 하지만 실제 구현 확인 필요")
    
    # 단순 sum 사용 확인
    sum_pattern = r'\.sum\(axis=0\)'
    sum_matches = check_file_content(gpu_grad_file, sum_pattern, "Simple sum")
    
    if sum_matches and len(sum_matches) > 0:
        print(f"  🔴 CRITICAL: 단순 sum 사용 발견 ({len(sum_matches)}회)")
        print(f"     → {sum_matches[:3]}")  # 처음 3개만 표시
    
    # 문제 2: 측정모델 - 첫 번째 행만 사용
    print("\n[문제 2] 측정모델 - 첫 번째 행만 사용 검사...")
    
    first_row_pattern = r'first_row\s*=\s*ind_data\.iloc\[0\]'
    first_row_matches = check_file_content(gpu_grad_file, first_row_pattern, "First row only")
    
    if first_row_matches:
        print(f"  🔴 CRITICAL: 첫 번째 행만 사용 ({len(first_row_matches)}회)")
        problems.append({
            'severity': 'CRITICAL',
            'problem': '측정모델 - 첫 번째 행만 사용',
            'file': 'gpu_gradient_batch.py',
            'description': '개인의 모든 선택 상황을 처리해야 하는데 첫 번째만 사용'
        })
    
    # 모든 행 순회 확인
    loop_pattern = r'for\s+\w+\s+in\s+range\(len\(ind_data\)\)'
    loop_matches = check_file_content(gpu_grad_file, loop_pattern, "Loop over all rows")
    
    if not loop_matches:
        print("  🔴 CRITICAL: 모든 행 순회 코드 없음")
    else:
        print(f"  ✅ OK: 모든 행 순회 코드 발견 ({len(loop_matches)}회)")
    
    # 문제 3: Likelihood 계산 누락
    print("\n[문제 3] Likelihood 계산 함수 누락 검사...")
    
    ll_function_pattern = r'def\s+compute_likelihood.*gpu'
    ll_matches = check_file_content(gpu_grad_file, ll_function_pattern, "Likelihood function")
    
    if not ll_matches:
        print("  🔴 CRITICAL: Likelihood 계산 함수 없음")
        problems.append({
            'severity': 'CRITICAL',
            'problem': 'Likelihood 계산 함수 누락',
            'file': 'gpu_gradient_batch.py',
            'description': 'Importance weighting을 위한 likelihood 계산 불가능'
        })
    else:
        print(f"  ✅ OK: Likelihood 함수 발견 ({len(ll_matches)}개)")
    
    # 문제 4: 수치 안정성
    print("\n[문제 4] 수치 안정성 검사...")
    
    # Clipping 확인
    clip_pattern = r'cp\.clip\('
    clip_matches = check_file_content(gpu_grad_file, clip_pattern, "Clipping")
    
    if clip_matches:
        print(f"  ✅ OK: Clipping 사용 ({len(clip_matches)}회)")
    else:
        print("  ⚠️  WARNING: Clipping 코드 없음")
    
    # NaN 체크 확인
    nan_check_pattern = r'(isnan|nan_to_num)'
    nan_matches = check_file_content(gpu_grad_file, nan_check_pattern, "NaN check")
    
    if not nan_matches:
        print("  ⚠️  WARNING: NaN 체크 코드 없음")
        problems.append({
            'severity': 'MAJOR',
            'problem': 'NaN 체크 누락',
            'file': 'gpu_gradient_batch.py',
            'description': 'NaN 발생 시 감지 및 처리 불가능'
        })
    
    # Log-sum-exp 확인
    logsumexp_pattern = r'log.*sum.*exp|logsumexp'
    lse_matches = check_file_content(gpu_grad_file, logsumexp_pattern, "Log-sum-exp")
    
    if not lse_matches:
        print("  ⚠️  WARNING: Log-sum-exp trick 없음 (overflow 위험)")
    
    # 문제 5: 선택모델 순차 처리
    print("\n[문제 5] 선택모델 배치 처리 검사...")
    
    choice_loop_pattern = r'for\s+draw_idx\s+in\s+range\(n_draws\)'
    choice_loop_matches = check_file_content(gpu_grad_file, choice_loop_pattern, "Sequential draw loop")
    
    if choice_loop_matches:
        print(f"  🟡 MAJOR: 순차 처리 발견 ({len(choice_loop_matches)}회) - GPU 미활용")
        problems.append({
            'severity': 'MAJOR',
            'problem': '선택모델 순차 처리',
            'file': 'gpu_gradient_batch.py',
            'description': 'GPU 병렬 처리 미활용, 성능 저하'
        })
    
    # CPU 버전과 비교
    print("\n[비교] CPU vs GPU 구현 차이...")
    
    # CPU 버전의 importance weighting
    cpu_weights_pattern = r'weights\s*=.*draw_likelihoods'
    cpu_weights = check_file_content(multi_grad_file, cpu_weights_pattern, "CPU weights")
    
    if cpu_weights:
        print(f"  ✅ CPU 버전: Importance weighting 구현됨")
    
    # CPU 버전의 가중평균
    cpu_weighted_pattern = r'w\s*\*\s*grad'
    cpu_weighted = check_file_content(multi_grad_file, cpu_weighted_pattern, "CPU weighted average")
    
    if cpu_weighted:
        print(f"  ✅ CPU 버전: 가중평균 구현됨 ({len(cpu_weighted)}회)")
    
    # 요약
    print("\n" + "="*70)
    print("진단 요약")
    print("="*70)
    
    if problems:
        print(f"\n발견된 문제: {len(problems)}개\n")
        
        critical = [p for p in problems if p['severity'] == 'CRITICAL']
        major = [p for p in problems if p['severity'] == 'MAJOR']
        
        if critical:
            print(f"🔴 CRITICAL 문제: {len(critical)}개")
            for p in critical:
                print(f"  - {p['problem']}")
                print(f"    파일: {p['file']}")
                print(f"    설명: {p['description']}")
                print()
        
        if major:
            print(f"🟡 MAJOR 문제: {len(major)}개")
            for p in major:
                print(f"  - {p['problem']}")
                print(f"    파일: {p['file']}")
                print(f"    설명: {p['description']}")
                print()
        
        print("\n결론: GPU Analytic Gradient는 현재 사용 불가능합니다.")
        print("수정 필요 사항:")
        print("  1. Importance weighting 구현")
        print("  2. 측정모델 모든 행 처리")
        print("  3. Likelihood 계산 함수 추가")
        print("  4. 가중평균으로 변경")
        print("  5. 수치 안정성 강화")
        
    else:
        print("\n✅ 주요 문제 발견되지 않음")
        print("   (하지만 실제 실행 테스트 필요)")
    
    print("\n" + "="*70)
    print("상세 분석: docs/gpu_gradient_problems_analysis.md 참고")
    print("="*70)
    
    return problems


if __name__ == '__main__':
    problems = diagnose_gpu_gradient()
    
    # Exit code
    if any(p['severity'] == 'CRITICAL' for p in problems):
        sys.exit(1)  # Critical 문제 있음
    elif problems:
        sys.exit(2)  # Major 문제 있음
    else:
        sys.exit(0)  # 문제 없음

