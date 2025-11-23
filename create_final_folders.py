"""
최종 결과 폴더 구조 생성
"""
from pathlib import Path

print("=" * 70)
print("최종 결과 폴더 구조 생성")
print("=" * 70)

# 생성할 폴더 목록
folders = [
    "results/final/cfa_only",
    "results/final/choice_only",
    "results/final/sequential/stage1",
    "results/final/sequential/stage2",
    "results/final/simultaneous/results",
    "results/final/simultaneous/logs"
]

print("\n폴더 생성 중...")
for folder in folders:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {folder}")

print("\n✅ 최종 결과 폴더 구조 생성 완료!")
print("=" * 70)

