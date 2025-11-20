"""우도 성분 비율 확인"""
import re
from pathlib import Path

# 최신 로그 파일 찾기
log_dir = Path('results')
log_files = sorted(log_dir.glob('simultaneous_estimation_log_*.txt'), key=lambda x: x.stat().st_mtime, reverse=True)

if not log_files:
    print("로그 파일이 없습니다!")
    exit(1)

log_file = log_files[0]
print(f"로그 파일: {log_file.name}")
print("=" * 80)

# 로그 파일 읽기
with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 개인 1, Draw #0 우도 성분 찾기
pattern = r'\[개인 1, Draw #0\] 우도 성분.*?구조모델: ([-\d.]+)'
match = re.search(pattern, content, re.DOTALL)

if match:
    # 전체 매칭된 텍스트 출력
    full_match = match.group(0)
    print("\n개인 1, Draw #0 우도 성분:")
    print(full_match)
    
    # 숫자 추출
    ll_measurement_match = re.search(r'll_measurement: ([-\d.]+)', full_match)
    ll_choice_match = re.search(r'll_choice: ([-\d.]+)', full_match)
    ll_structural_match = re.search(r'll_structural: ([-\d.]+)', full_match)
    
    if ll_measurement_match and ll_choice_match and ll_structural_match:
        ll_m = float(ll_measurement_match.group(1))
        ll_c = float(ll_choice_match.group(1))
        ll_s = float(ll_structural_match.group(1))
        
        total = abs(ll_m) + abs(ll_c) + abs(ll_s)
        
        print("\n" + "=" * 80)
        print("우도 성분 분석:")
        print("=" * 80)
        print(f"측정모델 (스케일링 후): {ll_m:.4f} ({100*abs(ll_m)/total:.1f}%)")
        print(f"선택모델: {ll_c:.4f} ({100*abs(ll_c)/total:.1f}%)")
        print(f"구조모델: {ll_s:.4f} ({100*abs(ll_s)/total:.1f}%)")
        print(f"합계: {ll_m + ll_c + ll_s:.4f}")
        
        # 원본 측정모델 우도 추출
        ll_measurement_raw_match = re.search(r'll_measurement_raw: ([-\d.]+)', full_match)
        if ll_measurement_raw_match:
            ll_m_raw = float(ll_measurement_raw_match.group(1))
            print(f"\n측정모델 (원본): {ll_m_raw:.4f}")
            print(f"스케일링 비율: {ll_m / ll_m_raw:.6f}")
else:
    print("우도 성분을 찾을 수 없습니다!")
    print("\n로그 파일 마지막 100줄:")
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-100:]:
            print(line.rstrip())

