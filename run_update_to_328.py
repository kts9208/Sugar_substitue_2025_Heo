"""
328명 데이터로 CFA 및 1단계 순차추정 재실행

이 스크립트는 다음을 수행합니다:
1. CFA Only (측정모델만) 재실행
2. 1단계 순차추정 (SEM) 재실행
3. 결과 검증

Author: ICLV Team
Date: 2025-11-23
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import shutil

print("=" * 70)
print("328명 데이터로 업데이트")
print("=" * 70)
print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 프로젝트 루트
project_root = Path(__file__).parent

# 1. 현재 상태 확인
print("\n[0/3] 현재 상태 확인 중...")
print("-" * 70)

import pandas as pd
import pickle

data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
data = pd.read_csv(data_path)
n_current = data['respondent_id'].nunique()

print(f"통합 데이터셋: {n_current}명")

if n_current != 328:
    print(f"⚠️ 경고: 통합 데이터셋이 328명이 아닙니다 ({n_current}명)")
    response = input("계속 진행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("중단되었습니다.")
        sys.exit(0)

# 기존 결과 확인
cfa_path = project_root / "results" / "sequential_stage_wise" / "cfa_results.pkl"
stage1_path = project_root / "results" / "sequential_stage_wise" / "stage1_HC-PB_PB-PI_results.pkl"

if cfa_path.exists():
    with open(cfa_path, 'rb') as f:
        cfa_results = pickle.load(f)
    n_cfa = len(cfa_results['factor_scores']['health_concern'])
    print(f"기존 CFA 결과: {n_cfa}명")
else:
    print("기존 CFA 결과: 없음")

if stage1_path.exists():
    with open(stage1_path, 'rb') as f:
        stage1_results = pickle.load(f)
    n_stage1 = len(stage1_results['factor_scores']['health_concern'])
    print(f"기존 1단계 순차추정 결과: {n_stage1}명")
else:
    print("기존 1단계 순차추정 결과: 없음")

# 2. 백업 여부 확인
print("\n백업 옵션:")
print("  1. 백업하고 진행")
print("  2. 백업 없이 진행")
print("  3. 취소")

backup_choice = input("선택 (1/2/3): ").strip()

if backup_choice == '3':
    print("취소되었습니다.")
    sys.exit(0)
elif backup_choice == '1':
    # 백업 폴더 생성
    backup_dir = project_root / "results" / "sequential_stage_wise" / f"backup_326_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n백업 중: {backup_dir}")
    
    # CFA 결과 백업
    if cfa_path.exists():
        shutil.copy2(cfa_path, backup_dir / "cfa_results.pkl")
        print(f"  ✅ cfa_results.pkl 백업 완료")
    
    # 1단계 결과 백업
    if stage1_path.exists():
        shutil.copy2(stage1_path, backup_dir / "stage1_HC-PB_PB-PI_results.pkl")
        print(f"  ✅ stage1_HC-PB_PB-PI_results.pkl 백업 완료")
    
    print(f"✅ 백업 완료: {backup_dir}")

# 3. CFA Only 실행
print("\n" + "=" * 70)
print("[1/2] CFA Only 실행 중...")
print("=" * 70)

cfa_script = project_root / "examples" / "sequential_cfa_only_example.py"
result = subprocess.run(
    [sys.executable, str(cfa_script)],
    cwd=str(project_root)
)

if result.returncode != 0:
    print("❌ CFA 실행 실패")
    sys.exit(1)

print("\n✅ CFA 완료")

# 4. 1단계 순차추정 실행
print("\n" + "=" * 70)
print("[2/2] 1단계 순차추정 실행 중...")
print("=" * 70)

stage1_script = project_root / "examples" / "sequential_stage1.py"
result = subprocess.run(
    [sys.executable, str(stage1_script)],
    cwd=str(project_root)
)

if result.returncode != 0:
    print("❌ 1단계 순차추정 실행 실패")
    sys.exit(1)

print("\n✅ 1단계 순차추정 완료")

# 5. 결과 검증
print("\n" + "=" * 70)
print("결과 검증")
print("=" * 70)

# CFA 결과 확인
if cfa_path.exists():
    with open(cfa_path, 'rb') as f:
        cfa_results = pickle.load(f)
    n_cfa_new = len(cfa_results['factor_scores']['health_concern'])
    
    if n_cfa_new == n_current:
        print(f"✅ CFA: {n_cfa_new}명 (일치)")
    else:
        print(f"❌ CFA: {n_cfa_new}명 (불일치)")
else:
    print("❌ CFA 결과 파일이 생성되지 않았습니다")

# 1단계 결과 확인
if stage1_path.exists():
    with open(stage1_path, 'rb') as f:
        stage1_results = pickle.load(f)
    n_stage1_new = len(stage1_results['factor_scores']['health_concern'])
    
    if n_stage1_new == n_current:
        print(f"✅ 1단계 순차추정: {n_stage1_new}명 (일치)")
    else:
        print(f"❌ 1단계 순차추정: {n_stage1_new}명 (불일치)")
else:
    print("❌ 1단계 순차추정 결과 파일이 생성되지 않았습니다")

# 최종 요약
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)
print(f"\n통합 데이터셋: {n_current}명")
print(f"CFA 결과: {n_cfa_new}명")
print(f"1단계 순차추정 결과: {n_stage1_new}명")

if n_cfa_new == n_current and n_stage1_new == n_current:
    print("\n✅ 모든 업데이트 완료! 모든 결과가 328명 데이터로 일치합니다.")
else:
    print("\n⚠️ 일부 결과가 일치하지 않습니다. 로그를 확인하세요.")

print("\n" + "=" * 70)
print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

