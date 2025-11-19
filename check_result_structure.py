import numpy as np
import pickle

# NPY 파일 로드
result = np.load('results/simultaneous_HC-PB_PB-PI_results_20251119_175057.npy', allow_pickle=True)

print("Type of result:", type(result))
print("\nResult shape:", result.shape if hasattr(result, 'shape') else 'N/A')

# 딕셔너리로 변환 시도
if isinstance(result, np.ndarray):
    if result.shape == ():
        result_dict = result.item()
    else:
        result_dict = result[0] if len(result) > 0 else {}
else:
    result_dict = result

print("\n=== Result Dictionary Keys ===")
print(list(result_dict.keys()))

print("\n=== Parameters Keys ===")
if 'parameters' in result_dict:
    params = result_dict['parameters']
    print(list(params.keys()))
    
    print("\n=== Structural Parameters ===")
    if 'structural' in params:
        print(params['structural'])
    
    print("\n=== Choice Parameters ===")
    if 'choice' in params:
        choice = params['choice']
        print("Choice keys:", list(choice.keys()))
        for key, value in choice.items():
            print(f"  {key}: {value}")
else:
    print("No 'parameters' key found")

print("\n=== Parameter Statistics Keys ===")
if 'parameter_statistics' in result_dict:
    stats = result_dict['parameter_statistics']
    print(list(stats.keys()))
    
    if 'structural' in stats:
        print("\nStructural stats:", stats['structural'])
    
    if 'choice' in stats:
        print("\nChoice stats keys:", list(stats['choice'].keys()))
        for key, value in stats['choice'].items():
            print(f"  {key}: {value}")
else:
    print("No 'parameter_statistics' key found")

