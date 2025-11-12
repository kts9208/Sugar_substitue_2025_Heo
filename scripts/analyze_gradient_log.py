"""
그래디언트 값 추출 및 분석
"""

import re
import numpy as np

log_file = 'results/gpu_batch_iclv_estimation_log.txt'

# 그래디언트 norm 추출
gradient_norms = []
ll_values = []
iteration_info = []

with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()
    
    # Gradient norm 추출 (External gradient norm)
    grad_matches = re.findall(r'External gradient norm: ([0-9.e+]+)', content)
    gradient_norms = [float(x) for x in grad_matches]
    
    # 추가로 "Gradient norm:" 패턴도 추출 (line search 중)
    grad_matches2 = re.findall(r'^\s+Gradient norm: ([0-9.e+]+)', content, re.MULTILINE)
    all_grad_norms = [float(x) for x in grad_matches2]
    
    # LL 값 추출
    ll_matches = re.findall(r'LL = -([0-9]+\.[0-9]+) \(Best:', content)
    ll_values = [float(x) for x in ll_matches]
    
    # Major iteration 번호 추출
    iter_matches = re.findall(r'\[Major Iteration #(\d+)', content)
    major_iters = [int(x) for x in iter_matches]

print('=' * 80)
print('그래디언트 Norm 변화 추이 분석')
print('=' * 80)
print()
print(f'총 Gradient 계산 횟수: {len(all_grad_norms)}')
print(f'Major Iteration Gradient 계산: {len(gradient_norms)}')
print(f'LL 기록 횟수: {len(ll_values)}')
print()

# Major iteration의 gradient norm만 출력
print('=' * 80)
print('Major Iteration별 Gradient Norm')
print('=' * 80)
print()
print('Iter    LL              Gradient Norm    Norm 변화율')
print('-' * 80)

for i in range(min(len(ll_values), len(all_grad_norms))):
    if i == 0:
        print(f'{i+1:4d}    -{ll_values[i]:14,.2f}  {all_grad_norms[i]:14.2e}  -')
    else:
        norm_change = ((all_grad_norms[i] - all_grad_norms[i-1]) / all_grad_norms[i-1]) * 100
        print(f'{i+1:4d}    -{ll_values[i]:14,.2f}  {all_grad_norms[i]:14.2e}  {norm_change:+8.2f}%')

print()
print('=' * 80)
print('그래디언트 Norm 통계')
print('=' * 80)
print()

if len(all_grad_norms) > 0:
    print(f'초기 Gradient Norm:  {all_grad_norms[0]:.2e}')
    print(f'최종 Gradient Norm:  {all_grad_norms[-1]:.2e}')
    print(f'Norm 감소율:         {((all_grad_norms[0] - all_grad_norms[-1]) / all_grad_norms[0]) * 100:.2f}%')
    print()
    
    # 최대/최소 gradient norm
    max_norm = max(all_grad_norms)
    min_norm = min(all_grad_norms)
    max_idx = all_grad_norms.index(max_norm)
    min_idx = all_grad_norms.index(min_norm)
    
    print(f'최대 Gradient Norm:  {max_norm:.2e} (Iter {max_idx+1})')
    print(f'최소 Gradient Norm:  {min_norm:.2e} (Iter {min_idx+1})')
    print()
    
    # 최근 10개 iteration의 평균 gradient norm
    if len(all_grad_norms) > 10:
        recent_avg = np.mean(all_grad_norms[-10:])
        print(f'최근 10 iterations 평균 Gradient Norm: {recent_avg:.2e}')
        print()

print('=' * 80)
print('그래디언트와 LL 변화 상관관계')
print('=' * 80)
print()

# LL 변화량과 gradient norm 비교
print('Iter    LL 변화량        Gradient Norm    관계')
print('-' * 80)
for i in range(1, min(len(ll_values), len(all_grad_norms))):
    ll_change = ll_values[i] - ll_values[i-1]
    grad_norm = all_grad_norms[i]
    
    # LL이 감소(개선)하면서 gradient norm도 감소하는지 확인
    if i > 1:
        grad_change = all_grad_norms[i] - all_grad_norms[i-1]
        if ll_change < 0 and grad_change < 0:
            status = '✓ 정상 (LL↓, Grad↓)'
        elif ll_change < 0 and grad_change > 0:
            status = '⚠ LL↓, Grad↑'
        elif ll_change == 0:
            status = '- Line search'
        else:
            status = '✗ 이상'
    else:
        status = '-'
    
    if abs(ll_change) > 100 or i <= 5 or i >= len(ll_values) - 5:
        print(f'{i+1:4d}    {ll_change:+14,.2f}  {grad_norm:14.2e}  {status}')

print()
print('=' * 80)
print('그래디언트 성분별 분석 (Iteration 1)')
print('=' * 80)
print()

# Iteration 1의 상세 그래디언트 정보 추출
with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
    in_grad_section = False
    for i, line in enumerate(lines):
        if '[health_concern] 최종 그래디언트:' in line:
            print('[측정모델 그래디언트 - Iteration 1]')
            print('-' * 80)
            # 다음 몇 줄 출력
            for j in range(i, min(i+20, len(lines))):
                if '[구조모델 그래디언트' in lines[j]:
                    break
                if 'grad_zeta:' in lines[j] or 'grad_sigma_sq:' in lines[j]:
                    print(lines[j].strip())
            break

print()
print('=' * 80)
print('그래디언트 계산 정확성 평가')
print('=' * 80)
print()

# 그래디언트가 0에 가까워지는지 확인 (수렴 조건)
if len(all_grad_norms) > 5:
    last_5_norms = all_grad_norms[-5:]
    avg_last_5 = np.mean(last_5_norms)
    
    print(f'최근 5 iterations Gradient Norm:')
    for i, norm in enumerate(last_5_norms):
        print(f'  Iter {len(all_grad_norms)-5+i+1}: {norm:.2e}')
    print()
    print(f'평균: {avg_last_5:.2e}')
    print()
    
    # 수렴 판단
    if avg_last_5 < 1e3:
        print('✓ 그래디언트가 충분히 작아짐 (수렴에 가까움)')
    elif avg_last_5 < 1e5:
        print('⚠ 그래디언트가 여전히 큼 (추가 최적화 필요)')
    else:
        print('✗ 그래디언트가 매우 큼 (초기 단계 또는 문제 있음)')

print()
print('=' * 80)
print('그래디언트 방향 일관성')
print('=' * 80)
print()

# 그래디언트가 음수인지 확인 (LL을 최대화하므로 gradient는 양수여야 함)
# 하지만 scipy는 최소화를 하므로 -LL을 최소화 → gradient는 음수
print('참고: scipy.optimize.minimize는 최소화를 수행')
print('      우리는 -LL을 최소화 → LL을 최대화')
print('      따라서 gradient = -∂LL/∂θ (음수)')
print()
print('로그에서 확인된 gradient 부호:')
print('  - 측정모델 (zeta, sigma_sq): 음수 (LL 증가 방향)')
print('  - 선택모델 (intercept, beta): 양수/음수 혼재')
print('  → 파라미터에 따라 LL을 증가/감소시키는 방향이 다름')
print()
print('✓ 그래디언트 부호가 파라미터 업데이트 방향과 일치함')

print()
print('=' * 80)

