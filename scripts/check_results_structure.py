"""
최근 결과 파일의 구조 확인
"""
import numpy as np
from pathlib import Path

# 최근 NPY 파일 찾기
results_dir = Path("results/final/simultaneous/results")
npy_files = list(results_dir.glob("*.npy"))

if not npy_files:
    print("❌ NPY 파일이 없습니다.")
    exit(1)

# 가장 최근 파일
latest_npy = max(npy_files, key=lambda p: p.stat().st_mtime)
print(f"✅ 최근 NPY 파일: {latest_npy.name}")
print(f"   수정 시간: {latest_npy.stat().st_mtime}")

# 로드
loaded = np.load(latest_npy, allow_pickle=True)
if isinstance(loaded, np.ndarray):
    if loaded.shape == ():
        results = loaded.item()
    else:
        results = loaded
else:
    results = loaded

print("\n" + "="*80)
print("Results 구조")
print("="*80)

print(f"\nType: {type(results)}")

if isinstance(results, dict):
    # 최상위 키
    print("\n[최상위 키]")
    for key in results.keys():
        print(f"  - {key}")
elif isinstance(results, np.ndarray):
    print(f"\nNumPy 배열:")
    print(f"  Shape: {results.shape}")
    print(f"  Dtype: {results.dtype}")
    if results.dtype == object and results.shape == ():
        print(f"  Scalar object - 내용: {results.item()}")
        results = results.item()
    else:
        print(f"  내용 (처음 5개): {results[:5] if len(results) > 5 else results}")
        exit(0)
else:
    print(f"  예상치 못한 타입: {type(results)}")
    exit(1)

# parameter_statistics 확인
print("\n[parameter_statistics 확인]")
if 'parameter_statistics' in results:
    param_stats = results['parameter_statistics']
    if param_stats is None:
        print("  ❌ parameter_statistics = None")
    else:
        print(f"  ✅ parameter_statistics 존재 (type: {type(param_stats)})")
        if isinstance(param_stats, dict):
            print(f"     키: {list(param_stats.keys())}")
            
            # 각 섹션 확인
            for section in ['measurement', 'structural', 'choice']:
                if section in param_stats:
                    print(f"\n  [{section}]")
                    section_data = param_stats[section]
                    if isinstance(section_data, dict):
                        print(f"     키: {list(section_data.keys())[:10]}...")  # 처음 10개만
                    else:
                        print(f"     type: {type(section_data)}")
else:
    print("  ❌ parameter_statistics 키 없음")

# params 확인
print("\n[params 확인]")
if 'params' in results:
    params = results['params']
    print(f"  ✅ params 존재 (type: {type(params)})")
    if isinstance(params, dict):
        print(f"     키: {list(params.keys())}")
else:
    print("  ❌ params 키 없음")

# standard_errors 확인
print("\n[standard_errors 확인]")
if 'standard_errors' in results:
    se = results['standard_errors']
    if se is None:
        print("  ❌ standard_errors = None")
    else:
        print(f"  ✅ standard_errors 존재 (type: {type(se)})")
        if isinstance(se, np.ndarray):
            print(f"     shape: {se.shape}")
        elif isinstance(se, dict):
            print(f"     키: {list(se.keys())}")
else:
    print("  ❌ standard_errors 키 없음")

print("\n" + "="*80)

