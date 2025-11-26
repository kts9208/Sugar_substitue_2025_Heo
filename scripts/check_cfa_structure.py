"""CFA 결과 구조 확인"""
import pickle
from pathlib import Path

project_root = Path(__file__).parent.parent
cfa_path = project_root / 'results' / 'final' / 'cfa_only' / 'cfa_results.pkl'

with open(cfa_path, 'rb') as f:
    cfa_results = pickle.load(f)

print("CFA 결과 키:", list(cfa_results.keys()))
print("\n'params' 타입:", type(cfa_results['params']))

if isinstance(cfa_results['params'], dict):
    print("\n'params' 키:", list(cfa_results['params'].keys())[:5])
    
    # 첫 번째 LV 확인
    first_lv = list(cfa_results['params'].keys())[0]
    print(f"\n첫 번째 LV '{first_lv}' 구조:")
    print(f"  타입: {type(cfa_results['params'][first_lv])}")
    if isinstance(cfa_results['params'][first_lv], dict):
        print(f"  키: {list(cfa_results['params'][first_lv].keys())}")
    else:
        print(f"  값: {cfa_results['params'][first_lv]}")
else:
    print("\n'params'는 딕셔너리가 아닙니다")
    print(f"  타입: {type(cfa_results['params'])}")
    if hasattr(cfa_results['params'], 'head'):
        print(f"  처음 5행:\n{cfa_results['params'].head()}")

