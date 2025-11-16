"""
Pickle 파일 내용 확인 스크립트
"""

import pickle
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
pickle_file = project_root / "results" / "sequential_stage_wise" / "stage1_results.pkl"

print("=" * 100)
print("Pickle 파일 내용 확인")
print("=" * 100)

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

print(f"\n파일: {pickle_file}")
print(f"\n키 목록: {list(data.keys())}")

# original_factor_scores 확인
if 'original_factor_scores' in data:
    print("\n✅ original_factor_scores 존재!")
    print(f"   변수: {list(data['original_factor_scores'].keys())}")
    
    print("\n[원본 요인점수 통계]")
    for lv_name, scores in data['original_factor_scores'].items():
        mean = np.mean(scores)
        variance = np.var(scores, ddof=0)
        std = np.std(scores, ddof=0)
        print(f"   {lv_name:30s}: mean={mean:8.4f}, var={variance:8.6f}, std={std:8.4f}")
else:
    print("\n❌ original_factor_scores 없음!")

# factor_scores 확인
if 'factor_scores' in data:
    print("\n✅ factor_scores 존재!")
    print(f"   변수: {list(data['factor_scores'].keys())}")
    
    print("\n[표준화된 요인점수 통계]")
    for lv_name, scores in data['factor_scores'].items():
        mean = np.mean(scores)
        variance = np.var(scores, ddof=0)
        std = np.std(scores, ddof=0)
        print(f"   {lv_name:30s}: mean={mean:8.6f}, var={variance:8.6f}, std={std:8.6f}")

print("\n" + "=" * 100)

