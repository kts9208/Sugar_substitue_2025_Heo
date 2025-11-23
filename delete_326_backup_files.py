"""
326명 버전 백업 CSV 파일 삭제 스크립트
"""
from pathlib import Path
import pandas as pd
import os

print("=" * 70)
print("326명 버전 백업 CSV 파일 삭제")
print("=" * 70)

# 삭제 대상 파일 목록
backup_files = [
    "data/processed/iclv/integrated_data_backup.csv",
    "data/processed/iclv/integrated_data_cleaned.csv",
    "data/processed/iclv/integrated_data_cleaned_backup.csv"
]

print("\n[1] 파일 확인 중...")
print("-" * 70)

files_to_delete = []

for file_path in backup_files:
    p = Path(file_path)
    
    if p.exists():
        # 파일 크기 확인
        size_mb = p.stat().st_size / (1024 * 1024)
        
        # 개인 수 확인
        try:
            df = pd.read_csv(p)
            if 'respondent_id' in df.columns:
                n_individuals = df['respondent_id'].nunique()
                print(f"✓ {p.name}")
                print(f"    개인 수: {n_individuals}명")
                print(f"    파일 크기: {size_mb:.2f} MB")
                
                if n_individuals == 326:
                    files_to_delete.append((p, n_individuals, size_mb))
                    print(f"    → 삭제 대상 (326명)")
                elif n_individuals == 328:
                    print(f"    → 유지 (328명)")
                else:
                    print(f"    → 확인 필요 ({n_individuals}명)")
            else:
                print(f"✓ {p.name}")
                print(f"    respondent_id 컬럼 없음")
                print(f"    → 건너뜀")
        except Exception as e:
            print(f"✗ {p.name}")
            print(f"    오류: {e}")
    else:
        print(f"✗ {p.name}")
        print(f"    파일 없음")
    
    print()

# 삭제 확인
if not files_to_delete:
    print("\n삭제할 326명 파일이 없습니다.")
    print("=" * 70)
    exit(0)

print("\n[2] 삭제 대상 파일")
print("-" * 70)
total_size = 0
for p, n, size in files_to_delete:
    print(f"  {p.name} ({n}명, {size:.2f} MB)")
    total_size += size

print(f"\n총 {len(files_to_delete)}개 파일, {total_size:.2f} MB")

# 사용자 확인
print("\n" + "=" * 70)
response = input("위 파일들을 삭제하시겠습니까? (y/n): ")

if response.lower() != 'y':
    print("취소되었습니다.")
    exit(0)

# 삭제 실행
print("\n[3] 파일 삭제 중...")
print("-" * 70)

deleted_count = 0
for p, n, size in files_to_delete:
    try:
        os.remove(p)
        print(f"✅ 삭제 완료: {p.name}")
        deleted_count += 1
    except Exception as e:
        print(f"❌ 삭제 실패: {p.name}")
        print(f"   오류: {e}")

# 최종 결과
print("\n" + "=" * 70)
print("최종 결과")
print("=" * 70)
print(f"\n삭제된 파일: {deleted_count}/{len(files_to_delete)}개")
print(f"절약된 공간: {total_size:.2f} MB")

if deleted_count == len(files_to_delete):
    print("\n✅ 모든 326명 백업 파일이 삭제되었습니다.")
else:
    print(f"\n⚠️ {len(files_to_delete) - deleted_count}개 파일 삭제 실패")

print("\n남은 파일:")
print("-" * 70)
iclv_dir = Path("data/processed/iclv")
for f in sorted(iclv_dir.glob("*.csv")):
    size_mb = f.stat().st_size / (1024 * 1024)
    try:
        df = pd.read_csv(f)
        if 'respondent_id' in df.columns:
            n = df['respondent_id'].nunique()
            print(f"  {f.name} ({n}명, {size_mb:.2f} MB)")
        else:
            print(f"  {f.name} ({size_mb:.2f} MB)")
    except:
        print(f"  {f.name} ({size_mb:.2f} MB)")

print("\n" + "=" * 70)

