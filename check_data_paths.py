"""
모든 예제 파일의 데이터 경로 확인
"""
from pathlib import Path

print("=" * 70)
print("예제 파일의 데이터 경로 확인")
print("=" * 70)

files = [
    'examples/sequential_cfa_only_example.py',
    'examples/sequential_stage1.py',
    'examples/sequential_stage2_with_extended_model.py',
    'examples/bootstrap_sequential_example.py',
    'examples/choice_model_only.py',
    'examples/correlation_analysis_example.py',
    'examples/moderation_effect_example.py'
]

for file_path in files:
    p = Path(file_path)
    if not p.exists():
        continue
    
    print(f"\n{file_path}:")
    
    found = False
    with open(p, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if 'integrated_data' in line and '.csv' in line:
                print(f"  Line {i}: {line.strip()}")
                found = True
    
    if not found:
        print("  No CSV path found")

print("\n" + "=" * 70)

