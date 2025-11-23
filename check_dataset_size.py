"""
데이터셋 크기 확인 스크립트
328명 데이터를 사용하는지 확인
"""
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

print("=" * 70)
print("데이터셋 크기 확인")
print("=" * 70)

# 1. 통합 데이터셋 확인
print("\n[1] 통합 데이터셋 (integrated_data.csv)")
print("-" * 70)
data_path = Path("data/processed/iclv/integrated_data.csv")
data = pd.read_csv(data_path)

print(f"전체 행 수: {len(data)}")
print(f"전체 열 수: {len(data.columns)}")

if 'respondent_id' in data.columns:
    n_individuals = data['respondent_id'].nunique()
    print(f"\n고유 개인 수 (respondent_id): {n_individuals}")
    print(f"개인당 평균 행 수: {len(data) / n_individuals:.2f}")
    
    unique_ids = sorted(data['respondent_id'].unique())
    print(f"\n첫 5개 respondent_id: {unique_ids[:5]}")
    print(f"마지막 5개 respondent_id: {unique_ids[-5:]}")
else:
    print("⚠️ respondent_id 컬럼이 없습니다!")

# 2. 1단계 순차추정 결과 확인
print("\n[2] 1단계 순차추정 결과 (stage1_HC-PB_PB-PI_results.pkl)")
print("-" * 70)
stage1_path = Path("results/sequential_stage_wise/stage1_HC-PB_PB-PI_results.pkl")

if stage1_path.exists():
    with open(stage1_path, 'rb') as f:
        results = pickle.load(f)
    
    n_factor_scores = len(results['factor_scores']['health_concern'])
    print(f"요인점수 개인 수: {n_factor_scores}")
    
    if 'original_factor_scores' in results:
        n_original = len(results['original_factor_scores']['health_concern'])
        print(f"원본 요인점수 개인 수: {n_original}")
    
    # 일치 여부 확인
    if 'respondent_id' in data.columns:
        if n_factor_scores == n_individuals:
            print(f"✅ 일치: 요인점수 개인 수 ({n_factor_scores}) = 데이터셋 개인 수 ({n_individuals})")
        else:
            print(f"❌ 불일치: 요인점수 개인 수 ({n_factor_scores}) ≠ 데이터셋 개인 수 ({n_individuals})")
else:
    print("⚠️ 1단계 결과 파일이 없습니다!")

# 3. CFA 결과 확인
print("\n[3] CFA 결과 (cfa_results.pkl)")
print("-" * 70)
cfa_path = Path("results/sequential_stage_wise/cfa_results.pkl")

if cfa_path.exists():
    with open(cfa_path, 'rb') as f:
        cfa_results = pickle.load(f)
    
    if 'factor_scores' in cfa_results:
        n_cfa = len(cfa_results['factor_scores']['health_concern'])
        print(f"CFA 요인점수 개인 수: {n_cfa}")
        
        if 'respondent_id' in data.columns:
            if n_cfa == n_individuals:
                print(f"✅ 일치: CFA 개인 수 ({n_cfa}) = 데이터셋 개인 수 ({n_individuals})")
            else:
                print(f"❌ 불일치: CFA 개인 수 ({n_cfa}) ≠ 데이터셋 개인 수 ({n_individuals})")
else:
    print("⚠️ CFA 결과 파일이 없습니다!")

# 4. 동시추정 결과 확인 (최신 파일)
print("\n[4] 동시추정 결과 (최신 파일)")
print("-" * 70)
results_dir = Path("results")
sim_files = sorted(results_dir.glob("simultaneous_*_results_*.csv"))

if sim_files:
    latest_sim = sim_files[-1]
    print(f"최신 파일: {latest_sim.name}")
    
    sim_df = pd.read_csv(latest_sim)
    print(f"파라미터 수: {len(sim_df)}")
    
    # 로그 파일에서 데이터 크기 확인
    log_files = sorted(results_dir.glob("simultaneous_estimation_log_*.txt"))
    if log_files:
        latest_log = log_files[-1]
        print(f"\n로그 파일: {latest_log.name}")
        
        with open(latest_log, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 데이터 크기 정보 찾기
        for line in log_content.split('\n'):
            if '개인 수' in line or 'n_individuals' in line or '행 수' in line:
                print(f"  {line.strip()}")
else:
    print("⚠️ 동시추정 결과 파일이 없습니다!")

# 5. 백업 파일 확인
print("\n[5] 백업 데이터셋 확인")
print("-" * 70)
backup_files = [
    "data/processed/iclv/integrated_data_backup.csv",
    "data/processed/iclv/integrated_data_cleaned.csv",
    "data/processed/iclv/integrated_data_cleaned_backup.csv"
]

for backup_file in backup_files:
    backup_path = Path(backup_file)
    if backup_path.exists():
        backup_data = pd.read_csv(backup_path)
        if 'respondent_id' in backup_data.columns:
            n_backup = backup_data['respondent_id'].nunique()
            print(f"{backup_path.name}: {n_backup}명")
        else:
            print(f"{backup_path.name}: respondent_id 없음")

# 6. 최종 요약
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

if 'respondent_id' in data.columns:
    print(f"\n현재 통합 데이터셋: {n_individuals}명")
    
    if n_individuals == 328:
        print("✅ 328명 데이터 사용 중")
    elif n_individuals == 326:
        print("⚠️ 326명 데이터 사용 중 (328명으로 업데이트 필요)")
    else:
        print(f"⚠️ {n_individuals}명 데이터 사용 중 (예상과 다름)")
    
    # 1단계 결과와 비교
    if stage1_path.exists():
        if n_factor_scores == n_individuals:
            print(f"✅ 1단계 순차추정: {n_factor_scores}명 (일치)")
        else:
            print(f"❌ 1단계 순차추정: {n_factor_scores}명 (불일치 - 재실행 필요)")
    
    # CFA 결과와 비교
    if cfa_path.exists() and 'factor_scores' in cfa_results:
        if n_cfa == n_individuals:
            print(f"✅ CFA: {n_cfa}명 (일치)")
        else:
            print(f"❌ CFA: {n_cfa}명 (불일치 - 재실행 필요)")

print("\n" + "=" * 70)

