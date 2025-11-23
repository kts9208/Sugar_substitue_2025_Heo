"""
기존 결과 파일을 최종 결과 폴더로 이동

이 스크립트는 다음을 수행합니다:
1. results/sequential_stage_wise/ 폴더의 결과를 results/final/ 폴더로 복사
2. results/ 루트의 동시추정 결과를 results/final/simultaneous/ 폴더로 복사

Author: ICLV Team
Date: 2025-11-23
"""
from pathlib import Path
import shutil

print("=" * 70)
print("기존 결과 파일을 최종 결과 폴더로 이동")
print("=" * 70)

project_root = Path.cwd()

# 1. CFA 결과 이동
print("\n[1] CFA 결과 이동...")
cfa_source = project_root / "results" / "sequential_stage_wise"
cfa_dest = project_root / "results" / "final" / "cfa_only"
cfa_dest.mkdir(parents=True, exist_ok=True)

cfa_files = list(cfa_source.glob("cfa_results*"))
if cfa_files:
    for file in cfa_files:
        dest_file = cfa_dest / file.name
        shutil.copy2(file, dest_file)
        print(f"  ✓ {file.name} → {cfa_dest.relative_to(project_root)}/")
    print(f"  총 {len(cfa_files)}개 파일 복사 완료")
else:
    print("  CFA 결과 파일 없음")

# 2. 1단계 순차추정 결과 이동
print("\n[2] 1단계 순차추정 결과 이동...")
stage1_source = project_root / "results" / "sequential_stage_wise"
stage1_dest = project_root / "results" / "final" / "sequential" / "stage1"
stage1_dest.mkdir(parents=True, exist_ok=True)

stage1_files = list(stage1_source.glob("stage1_*"))
if stage1_files:
    for file in stage1_files:
        dest_file = stage1_dest / file.name
        shutil.copy2(file, dest_file)
        print(f"  ✓ {file.name} → {stage1_dest.relative_to(project_root)}/")
    print(f"  총 {len(stage1_files)}개 파일 복사 완료")
else:
    print("  1단계 결과 파일 없음")

# 3. 2단계 순차추정 결과 이동
print("\n[3] 2단계 순차추정 결과 이동...")
stage2_source = project_root / "results" / "sequential_stage_wise"
stage2_dest = project_root / "results" / "final" / "sequential" / "stage2"
stage2_dest.mkdir(parents=True, exist_ok=True)

stage2_files = list(stage2_source.glob("st2_*"))
if stage2_files:
    for file in stage2_files:
        dest_file = stage2_dest / file.name
        shutil.copy2(file, dest_file)
        print(f"  ✓ {file.name} → {stage2_dest.relative_to(project_root)}/")
    print(f"  총 {len(stage2_files)}개 파일 복사 완료")
else:
    print("  2단계 결과 파일 없음")

# 4. 동시추정 결과 이동
print("\n[4] 동시추정 결과 이동...")
simul_source = project_root / "results"
simul_results_dest = project_root / "results" / "final" / "simultaneous" / "results"
simul_logs_dest = project_root / "results" / "final" / "simultaneous" / "logs"
simul_results_dest.mkdir(parents=True, exist_ok=True)
simul_logs_dest.mkdir(parents=True, exist_ok=True)

# 결과 파일 (CSV, NPY)
simul_result_files = list(simul_source.glob("simultaneous_*_results_*.csv")) + \
                     list(simul_source.glob("simultaneous_*_results_*.npy"))
if simul_result_files:
    for file in simul_result_files:
        dest_file = simul_results_dest / file.name
        shutil.copy2(file, dest_file)
        print(f"  ✓ {file.name} → {simul_results_dest.relative_to(project_root)}/")
    print(f"  총 {len(simul_result_files)}개 결과 파일 복사 완료")
else:
    print("  동시추정 결과 파일 없음")

# 로그 파일
simul_log_files = list(simul_source.glob("simultaneous_estimation_log_*.txt")) + \
                  list(simul_source.glob("simultaneous_estimation_log_*_params_grads.csv"))
if simul_log_files:
    for file in simul_log_files:
        dest_file = simul_logs_dest / file.name
        shutil.copy2(file, dest_file)
        print(f"  ✓ {file.name} → {simul_logs_dest.relative_to(project_root)}/")
    print(f"  총 {len(simul_log_files)}개 로그 파일 복사 완료")
else:
    print("  동시추정 로그 파일 없음")

# 5. 최종 요약
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

total_files = len(cfa_files) + len(stage1_files) + len(stage2_files) + \
              len(simul_result_files) + len(simul_log_files)

print(f"\n총 {total_files}개 파일 복사 완료")
print(f"\n최종 결과 폴더: results/final/")
print(f"  - CFA Only: {len(cfa_files)}개 파일")
print(f"  - 순차추정 1단계: {len(stage1_files)}개 파일")
print(f"  - 순차추정 2단계: {len(stage2_files)}개 파일")
print(f"  - 동시추정 결과: {len(simul_result_files)}개 파일")
print(f"  - 동시추정 로그: {len(simul_log_files)}개 파일")

print("\n⚠️ 주의: 기존 파일은 그대로 유지됩니다 (복사만 수행)")
print("기존 파일을 삭제하려면 수동으로 삭제하세요.")

print("\n✅ 이동 완료!")
print("=" * 70)

