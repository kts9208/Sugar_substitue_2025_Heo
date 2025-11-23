"""
동시추정 실행에 필요한 파일 확인
"""
from pathlib import Path

project_root = Path('.')

print("=" * 70)
print("동시추정 실행에 필요한 파일 확인")
print("=" * 70)

# 1. CFA 결과 파일
cfa_pkl = project_root / 'results' / 'final' / 'cfa_only' / 'cfa_results.pkl'
print(f"\n[1] CFA 결과 파일:")
print(f"    경로: {cfa_pkl}")
print(f"    존재: {'✅ 있음' if cfa_pkl.exists() else '❌ 없음'}")

if cfa_pkl.exists():
    import pickle
    with open(cfa_pkl, 'rb') as f:
        cfa_results = pickle.load(f)
    
    print(f"    포함 키: {list(cfa_results.keys())}")
    
    if 'loadings' in cfa_results:
        print(f"    요인적재량: {len(cfa_results['loadings'])}개")
    
    if 'measurement_errors' in cfa_results:
        print(f"    측정오차: {len(cfa_results['measurement_errors'])}개")

# 2. Stage2 CSV 파일
stage2_csv = project_root / 'results' / 'final' / 'sequential' / 'stage2' / 'st2_HC-PB_PB-PI1_PI2_results.csv'
print(f"\n[2] Stage2 CSV 파일:")
print(f"    경로: {stage2_csv}")
print(f"    존재: {'✅ 있음' if stage2_csv.exists() else '❌ 없음'}")

if not stage2_csv.exists():
    print(f"\n    ⚠️ Stage2 CSV 파일이 없습니다.")
    print(f"    순차추정 2단계를 먼저 실행해야 합니다:")
    print(f"    python examples/sequential_stage2_with_extended_model.py")
    
    # 대체 파일 찾기
    stage2_dir = project_root / 'results' / 'final' / 'sequential' / 'stage2'
    if stage2_dir.exists():
        csv_files = list(stage2_dir.glob('*.csv'))
        if csv_files:
            print(f"\n    사용 가능한 Stage2 CSV 파일:")
            for f in csv_files:
                print(f"      - {f.name}")

# 3. 통합 데이터
data_csv = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
print(f"\n[3] 통합 데이터:")
print(f"    경로: {data_csv}")
print(f"    존재: {'✅ 있음' if data_csv.exists() else '❌ 없음'}")

if data_csv.exists():
    import pandas as pd
    data = pd.read_csv(data_csv)
    print(f"    행 수: {len(data):,}")
    print(f"    개인 수: {data['respondent_id'].nunique()}명")

print("\n" + "=" * 70)
print("요약")
print("=" * 70)

all_ready = cfa_pkl.exists() and stage2_csv.exists() and data_csv.exists()

if all_ready:
    print("✅ 모든 필요한 파일이 준비되었습니다!")
    print("   동시추정을 실행할 수 있습니다:")
    print("   python scripts/test_gpu_batch_iclv.py")
else:
    print("⚠️ 일부 파일이 없습니다.")
    if not cfa_pkl.exists():
        print("   - CFA 결과 파일 필요")
    if not stage2_csv.exists():
        print("   - Stage2 CSV 파일 필요")
    if not data_csv.exists():
        print("   - 통합 데이터 파일 필요")

print("=" * 70)

