"""
실제 로그 데이터에서 우도 계산 검증

최근 테스트 로그에서 우도 성분을 추출하여 계산 방식 확인
"""
import re
from pathlib import Path

# 최근 로그 파일 찾기
log_dir = Path('results')
log_files = sorted(log_dir.glob('simultaneous_estimation_log_*.txt'), 
                   key=lambda x: x.stat().st_mtime, reverse=True)

if not log_files:
    print("로그 파일을 찾을 수 없습니다.")
    exit(1)

log_file = log_files[0]
print("="*80)
print(f"로그 파일 분석: {log_file.name}")
print("="*80)

# 로그 파일 읽기
with open(log_file, 'r', encoding='utf-8') as f:
    log_content = f.read()

# 우도 성분 추출 (첫 번째 개인, Draw #0)
print(f"\n[개인 1, Draw #0 우도 성분 추출]")
print(f"{'='*80}")

# 패턴 매칭
patterns = {
    'll_measurement_raw': r'\[개인 1, Draw #0\].*?ll_measurement_raw: ([-\d.]+)',
    'll_measurement': r'\[개인 1, Draw #0\].*?ll_measurement: ([-\d.]+)',
    'll_choice': r'\[개인 1, Draw #0\].*?ll_choice: ([-\d.]+)',
    'll_structural': r'\[개인 1, Draw #0\].*?ll_structural: ([-\d.]+)',
    'draw_ll': r'\[개인 1, Draw #0\].*?draw_ll \(합계\): ([-\d.]+)',
    'n_measurement_indicators': r'측정모델 지표 수: (\d+)',
    'n_structural_paths': r'구조모델 경로 수: (\d+)',
    'n_choice_situations': r'선택모델 선택 상황 수: (\d+)',
}

values = {}
for key, pattern in patterns.items():
    match = re.search(pattern, log_content, re.DOTALL)
    if match:
        values[key] = float(match.group(1))
        print(f"  {key}: {values[key]}")
    else:
        print(f"  {key}: NOT FOUND")

# 계산 검증
print(f"\n{'='*80}")
print(f"[계산 검증]")
print(f"{'='*80}")

if 'll_measurement_raw' in values and 'n_measurement_indicators' in values:
    n_ind = int(values['n_measurement_indicators'])
    ll_raw = values['ll_measurement_raw']
    ll_scaled = values.get('ll_measurement', ll_raw)
    
    print(f"\n측정모델:")
    print(f"  원본 우도: {ll_raw:.4f}")
    print(f"  지표 수: {n_ind}")
    print(f"  지표당 평균 우도: {ll_raw / n_ind:.4f}")
    print(f"  스케일링된 우도: {ll_scaled:.4f}")
    
    if abs(ll_scaled - ll_raw / n_ind) < 0.01:
        print(f"  ✓ 스케일링 정상: ll_scaled ≈ ll_raw / n_ind")
    else:
        print(f"  ✗ 스케일링 이상: ll_scaled ≠ ll_raw / n_ind")

if 'll_choice' in values and 'n_choice_situations' in values:
    n_cs = int(values['n_choice_situations'])
    ll_choice = values['ll_choice']
    
    print(f"\n선택모델:")
    print(f"  우도: {ll_choice:.4f}")
    print(f"  선택 상황 수: {n_cs}")
    print(f"  선택 상황당 평균 우도: {ll_choice / n_cs:.4f}")

if 'll_structural' in values and 'n_structural_paths' in values:
    n_paths = int(values['n_structural_paths'])
    ll_struct = values['ll_structural']
    
    print(f"\n구조모델:")
    print(f"  우도: {ll_struct:.4f}")
    print(f"  경로 수: {n_paths}")
    print(f"  경로당 평균 우도: {ll_struct / n_paths:.4f}")

# 비율 계산
print(f"\n{'='*80}")
print(f"[우도 성분 비율]")
print(f"{'='*80}")

if all(k in values for k in ['ll_measurement', 'll_choice', 'll_structural']):
    ll_m = abs(values['ll_measurement'])
    ll_c = abs(values['ll_choice'])
    ll_s = abs(values['ll_structural'])
    total = ll_m + ll_c + ll_s
    
    print(f"\n스케일링 후:")
    print(f"  측정모델: {ll_m:.4f} ({ll_m/total*100:.1f}%)")
    print(f"  선택모델: {ll_c:.4f} ({ll_c/total*100:.1f}%)")
    print(f"  구조모델: {ll_s:.4f} ({ll_s/total*100:.1f}%)")
    print(f"  합계: {total:.4f}")

if all(k in values for k in ['ll_measurement_raw', 'll_choice', 'll_structural']):
    ll_m_raw = abs(values['ll_measurement_raw'])
    ll_c = abs(values['ll_choice'])
    ll_s = abs(values['ll_structural'])
    total_raw = ll_m_raw + ll_c + ll_s
    
    print(f"\n스케일링 전 (원본):")
    print(f"  측정모델: {ll_m_raw:.4f} ({ll_m_raw/total_raw*100:.1f}%)")
    print(f"  선택모델: {ll_c:.4f} ({ll_c/total_raw*100:.1f}%)")
    print(f"  구조모델: {ll_s:.4f} ({ll_s/total_raw*100:.1f}%)")
    print(f"  합계: {total_raw:.4f}")

# 최종 우도 추출
print(f"\n{'='*80}")
print(f"[최종 우도]")
print(f"{'='*80}")

final_ll_pattern = r'최종 LL \(언스케일링\): ([-\d.]+)'
match = re.search(final_ll_pattern, log_content)
if match:
    final_ll = float(match.group(1))
    print(f"\n최종 LL (언스케일링): {final_ll:.4f}")
    
    # 개인당 평균 우도
    n_persons = 328  # 수정된 데이터
    avg_ll_per_person = final_ll / n_persons
    print(f"개인당 평균 우도: {avg_ll_per_person:.4f}")

