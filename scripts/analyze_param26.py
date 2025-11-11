import re

# 로그 파일 읽기
with open('results/gpu_batch_iclv_estimation_log.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# param_26의 데이터 추적
print('='*100)
print('Gradient vs LL 변화 분석')
print('='*100)
print()
print('질문: Gradient가 크면 LL 변화도 커야 하는가?')
print()
print(f'{"Iter":<6} {"LL":>18} {"ΔLL":>15} {"gradient (26)":>18} {"Δgrad":>15} {"ΔLL/|grad|":>15}')
print('-'*100)

prev_ll = None
prev_grad = None

for i, line in enumerate(lines):
    # Major Iteration 완료 찾기
    if '[Major Iteration #' in line and '완료]' in line:
        iter_num = int(re.search(r'#(\d+)', line).group(1))

        # LL 찾기
        ll = None
        for j in range(i, min(i+10, len(lines))):
            if '최종 LL:' in lines[j]:
                ll = float(re.search(r'최종 LL: ([-\d.]+)', lines[j]).group(1))
                break

        # param_26 gradient 찾기
        grad = None
        for j in range(i, min(i+30, len(lines))):
            if '[26] param_26' in lines[j] and ':' in lines[j]:
                match = re.search(r': ([-+\d.e]+)', lines[j])
                if match:
                    grad = float(match.group(1))
                    break

        if ll is None or grad is None:
            continue

        # 변화량 계산
        delta_ll = ll - prev_ll if prev_ll is not None else 0
        delta_grad = grad - prev_grad if prev_grad is not None else 0

        # ΔLL / |gradient| 비율 (gradient가 크면 이 값이 작아야 함)
        ratio = delta_ll / abs(grad) if grad != 0 else 0

        # 출력
        ll_str = f'{ll:,.2f}'
        delta_ll_str = f'{delta_ll:+,.2f}' if prev_ll is not None else 'N/A'
        grad_str = f'{grad:,.0f}'
        delta_grad_str = f'{delta_grad:+,.0f}' if prev_grad is not None else 'N/A'
        ratio_str = f'{ratio:.6f}' if prev_ll is not None else 'N/A'

        print(f'{iter_num:<6} {ll_str:>18} {delta_ll_str:>15} {grad_str:>18} {delta_grad_str:>15} {ratio_str:>15}')

        prev_ll = ll
        prev_grad = grad

print('='*100)
print()
print('해석:')
print('- ΔLL: 로그우도 변화량 (양수 = 개선)')
print('- gradient (26): param_26의 internal gradient (∂LL/∂θ_internal)')
print('- Δgrad: gradient 변화량')
print('- ΔLL/|grad|: LL 변화량을 gradient 크기로 나눈 값')
print()
print('예상:')
print('  만약 gradient가 정확하다면:')
print('    → |gradient|가 크면 ΔLL도 커야 함')
print('    → ΔLL/|grad| 비율이 일정해야 함')
print()
print('  하지만 실제로는:')
print('    → Gradient가 100만 단위로 매우 큼')
print('    → ΔLL은 수천~수만 단위')
print('    → ΔLL/|grad| ≈ 0.001~0.1 (매우 작음)')
print()
print('결론:')
print('  → Gradient가 너무 크게 계산되고 있음!')
print('  → 스케일링이 제대로 작동하지 않음')

