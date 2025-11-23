"""
동시추정 로그 파일을 분석하여 계산속도 개선 방안을 제안하는 스크립트
"""

from pathlib import Path
import re
from datetime import datetime

# 최신 로그 파일 찾기
log_dir = Path('results/final/simultaneous/logs')
txt_files = [f for f in log_dir.glob('*.txt') if not f.name.endswith('_params_grads.csv')]
latest_log = max(txt_files, key=lambda f: f.stat().st_mtime)

print(f"분석 대상 로그 파일: {latest_log.name}")
print("=" * 80)

# 로그 파일 읽기
with open(latest_log, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Major Iteration 시간 추출
iteration_times = []
iteration_info = []

for i, line in enumerate(lines):
    # Major Iteration 완료 찾기
    if '[Major Iteration #' in line and '완료]' in line:
        # Iteration 번호 추출
        match = re.search(r'#(\d+)', line)
        if match:
            iter_num = int(match.group(1))

            # 시간 추출 (이전 줄에서 - 최대 5줄 이전까지 검색)
            time_str = None
            for j in range(max(0, i-5), i+1):
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', lines[j])
                if time_match:
                    time_str = time_match.group(1)

            if time_str:
                # LL 값 추출
                ll_match = re.search(r'최종 LL: ([-\d.]+)', lines[i+1])
                ll_value = float(ll_match.group(1)) if ll_match else None

                # Line Search 정보 추출
                ls_match = re.search(r'Line Search: (\d+)회 함수 호출', lines[i+2])
                ls_calls = int(ls_match.group(1)) if ls_match else 0

                # 함수 호출 횟수 추출
                func_match = re.search(r'함수 호출: (\d+)회 \(누적\)', lines[i+3])
                func_calls = int(func_match.group(1)) if func_match else 0

                # 그래디언트 호출 횟수 추출
                grad_match = re.search(r'그래디언트 호출: (\d+)회 \(누적\)', lines[i+4])
                grad_calls = int(grad_match.group(1)) if grad_match else 0

                iteration_info.append({
                    'iter': iter_num,
                    'time': datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S'),
                    'll': ll_value,
                    'ls_calls': ls_calls,
                    'func_calls': func_calls,
                    'grad_calls': grad_calls
                })

# 시작 시간 찾기
start_time = None
for line in lines[:50]:
    if 'SimultaneousEstimator.estimate() 시작' in line:
        time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if time_match:
            start_time = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
            break

# 각 iteration 소요 시간 계산
print("\n📊 Iteration별 소요 시간 분석")
print("=" * 80)
print(f"{'Iter':<6} {'시작시간':<10} {'소요시간(초)':<12} {'누적시간(초)':<12} {'LL':<12} {'LS호출':<8} {'함수호출':<8} {'Grad호출':<8}")
print("-" * 80)

prev_time = start_time
total_time = 0

for info in iteration_info:
    if prev_time:
        elapsed = (info['time'] - prev_time).total_seconds()
        total_time = (info['time'] - start_time).total_seconds()
        print(f"{info['iter']:<6} {info['time'].strftime('%H:%M:%S'):<10} {elapsed:<12.1f} {total_time:<12.1f} {info['ll']:<12.2f} {info['ls_calls']:<8} {info['func_calls']:<8} {info['grad_calls']:<8}")
        prev_time = info['time']

# 통계 계산
avg_time = 0
max_time = 0
min_time = 0

if len(iteration_info) > 1:
    times = [(iteration_info[i]['time'] - iteration_info[i-1]['time']).total_seconds()
             for i in range(1, len(iteration_info))]

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    print("\n" + "=" * 80)
    print("📈 통계 요약")
    print("=" * 80)
    print(f"총 Iteration 수: {len(iteration_info)}")
    print(f"총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    print(f"평균 Iteration 시간: {avg_time:.1f}초")
    print(f"최대 Iteration 시간: {max_time:.1f}초")
    print(f"최소 Iteration 시간: {min_time:.1f}초")
    print(f"Iteration당 평균 함수 호출: {iteration_info[-1]['func_calls'] / len(iteration_info):.1f}회")
    print(f"Iteration당 평균 그래디언트 호출: {iteration_info[-1]['grad_calls'] / len(iteration_info):.1f}회")

    # Line Search 분석
    ls_calls_list = [info['ls_calls'] for info in iteration_info]
    print(f"\nLine Search 호출 분포:")
    print(f"  - 1회: {ls_calls_list.count(1)}번")
    print(f"  - 2회: {ls_calls_list.count(2)}번")
    print(f"  - 3회 이상: {sum(1 for x in ls_calls_list if x >= 3)}번")

    # 수렴 속도 분석
    print("\n" + "=" * 80)
    print("🎯 수렴 속도 분석")
    print("=" * 80)
    ll_improvements = []
    for i in range(1, len(iteration_info)):
        improvement = iteration_info[i-1]['ll'] - iteration_info[i]['ll']
        ll_improvements.append(improvement)
        status = "✓ 개선" if improvement > 0 else "✗ 악화"
        print(f"Iter {iteration_info[i]['iter']}: LL 변화 = {improvement:+.4f} {status}")
else:
    print("\n⚠️ Iteration 정보가 충분하지 않습니다.")
    ll_improvements = []
    ls_calls_list = []

print("\n" + "=" * 80)
print("💡 계산속도 개선 제안")
print("=" * 80)

# 제안 생성
suggestions = []

# 1. Iteration 시간 분석
if avg_time > 180:  # 3분 이상
    suggestions.append({
        'priority': 'HIGH',
        'issue': f'Iteration당 평균 {avg_time:.1f}초 소요 (매우 느림)',
        'suggestions': [
            'GPU 배치 크기 증가 (현재 설정 확인 필요)',
            'Halton draws 수 감소 고려 (정확도 vs 속도 trade-off)',
            '측정모델 파라미터 고정 확인 (이미 적용됨)',
        ]
    })

# 2. Line Search 분석
if len(ls_calls_list) > 0 and sum(1 for x in ls_calls_list if x >= 2) > len(ls_calls_list) * 0.5:
    suggestions.append({
        'priority': 'MEDIUM',
        'issue': f'Line Search가 {len(ls_calls_list)}회 중 {sum(1 for x in ls_calls_list if x >= 2)}번 2회 이상 호출됨',
        'suggestions': [
            'Line Search 파라미터 조정 (maxls, c1, c2)',
            '초기값 개선 (순차추정 2단계 결과 사용 중)',
            'Hessian 근사 방법 변경 고려 (BFGS → L-BFGS)',
        ]
    })

# 3. 수렴 속도 분석
if len(ll_improvements) > 0:
    negative_improvements = sum(1 for x in ll_improvements if x < 0)
    if negative_improvements > 0:
        suggestions.append({
            'priority': 'MEDIUM',
            'issue': f'{negative_improvements}번의 iteration에서 LL이 악화됨',
            'suggestions': [
                'Line Search 실패 시 step size 조정',
                '수렴 기준 완화 고려 (ftol, gtol)',
                'Trust region 방법 고려',
            ]
        })

# 4. 전체 소요 시간
if total_time > 1800:  # 30분 이상
    suggestions.append({
        'priority': 'HIGH',
        'issue': f'전체 추정에 {total_time/60:.1f}분 소요 (매우 느림)',
        'suggestions': [
            '병렬 처리 강화 (GPU 활용도 확인)',
            '데이터 전처리 최적화',
            '불필요한 로깅 제거',
        ]
    })

# 제안 출력
for i, sug in enumerate(suggestions, 1):
    print(f"\n[{sug['priority']}] 문제 {i}: {sug['issue']}")
    print("제안:")
    for j, s in enumerate(sug['suggestions'], 1):
        print(f"  {j}. {s}")

print("\n" + "=" * 80)
print("✅ 분석 완료")
print("=" * 80)

