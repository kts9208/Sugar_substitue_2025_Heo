"""
로그 파일 분석 스크립트
- LL 변화 추이
- 파라미터 변화 추이
- 그래디언트 norm 변화
"""

import re
import numpy as np
import matplotlib.pyplot as plt

# 로그 파일 읽기
log_file = 'results/gpu_batch_iclv_estimation_log.txt'

# LL 값 추출
ll_values = []
iteration_numbers = []

with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()
    
    # LL 값 추출
    ll_matches = re.findall(r'LL = -([0-9]+\.[0-9]+) \(Best:', content)
    ll_values = [float(x) for x in ll_matches]
    
    # Iteration 번호 추출
    iter_matches = re.findall(r'Iteration (\d+) - 파라미터 값', content)
    iteration_numbers = [int(x) for x in iter_matches]

print('=' * 80)
print('Log-Likelihood 변화 추이 분석')
print('=' * 80)
print()
print(f'총 Major Iterations: {len(ll_values)}')
print(f'파라미터 로깅된 Iterations: {iteration_numbers}')
print()

# LL 값 출력
print('Iter    LL              Delta           % Change    Status')
print('-' * 80)
for i, ll in enumerate(ll_values, 1):
    if i == 1:
        print(f'{i:4d}    -{ll:14,.2f}  -               -           [INITIAL]')
    else:
        delta = ll_values[i-1] - ll_values[i-2]
        pct_change = (delta / ll_values[i-2]) * 100
        status = '[NEW BEST]' if delta > 0 else '[NO CHANGE]'
        print(f'{i:4d}    -{ll:14,.2f}  {delta:+14,.2f}  {pct_change:+8.2f}%  {status}')

print()
print('=' * 80)
print('요약 통계')
print('=' * 80)
print(f'초기 LL:     -{ll_values[0]:,.2f}')
print(f'최종 LL:     -{ll_values[-1]:,.2f}')
print(f'총 개선량:   {ll_values[0] - ll_values[-1]:+,.2f}')
print(f'개선율:      {((ll_values[0] - ll_values[-1]) / ll_values[0]) * 100:.2f}%')
print()

# 개선이 있었던 iteration 수
improvements = sum(1 for i in range(1, len(ll_values)) if ll_values[i] < ll_values[i-1])
no_changes = sum(1 for i in range(1, len(ll_values)) if ll_values[i] == ll_values[i-1])
print(f'개선된 iterations:   {improvements}/{len(ll_values)-1} ({improvements/(len(ll_values)-1)*100:.1f}%)')
print(f'변화 없음:           {no_changes}/{len(ll_values)-1} ({no_changes/(len(ll_values)-1)*100:.1f}%)')
print()

# 가장 큰 개선
if len(ll_values) > 1:
    max_improvement_idx = max(range(1, len(ll_values)), key=lambda i: ll_values[i-1] - ll_values[i])
    max_improvement = ll_values[max_improvement_idx-1] - ll_values[max_improvement_idx]
    print(f'최대 개선: Iter {max_improvement_idx} → {max_improvement_idx+1}')
    print(f'           -{ll_values[max_improvement_idx-1]:,.2f} → -{ll_values[max_improvement_idx]:,.2f}')
    print(f'           Δ = +{max_improvement:,.2f} ({(max_improvement/ll_values[max_improvement_idx-1])*100:.2f}%)')
print()

# 최근 10개 iteration의 평균 개선량
if len(ll_values) > 10:
    recent_improvements = [ll_values[i-1] - ll_values[i] for i in range(len(ll_values)-10, len(ll_values))]
    avg_recent_improvement = np.mean([x for x in recent_improvements if x > 0])
    print(f'최근 10 iterations 평균 개선량: {avg_recent_improvement:,.2f}')
    print()

# 수렴 여부 판단
if len(ll_values) > 5:
    last_5_changes = [abs(ll_values[i] - ll_values[i-1]) for i in range(len(ll_values)-5, len(ll_values))]
    max_recent_change = max(last_5_changes)
    print(f'최근 5 iterations 최대 변화량: {max_recent_change:,.2f}')
    if max_recent_change < 100:
        print('  → 수렴에 가까워지고 있음 (변화량 < 100)')
    elif max_recent_change < 1000:
        print('  → 수렴 중 (변화량 < 1000)')
    else:
        print('  → 아직 최적화 진행 중 (변화량 > 1000)')

print()
print('=' * 80)

