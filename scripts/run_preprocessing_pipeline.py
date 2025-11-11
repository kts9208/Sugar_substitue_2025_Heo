"""
ICLV 데이터 전처리 파이프라인 (326명 버전)

이 스크립트는 원본 Excel 데이터에서 ICLV 모델 추정에 필요한 모든 데이터를 생성합니다.

전처리 단계:
  1. Survey 파일 생성 (health_concern.csv, perceived_benefit.csv, etc.)
  2. DCE 데이터 전처리 (Wide → Long format)
  3. ICLV 데이터 통합 (DCE + Survey + Sociodem)

입력:
  - data/raw/Sugar_substitue_Raw data_251108.xlsx (326명)
  - data/processed/dce/design_matrix.csv

출력:
  - data/processed/survey/*.csv (5개 파일)
  - data/processed/dce/dce_long_format.csv
  - data/processed/iclv/integrated_data.csv

Author: Sugar Substitute Research Team
Date: 2025-11-11
"""

import sys
import os
from pathlib import Path
import subprocess

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title):
    """헤더 출력"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_step(step_num, step_name):
    """단계 출력"""
    print(f"\n{'='*80}")
    print(f"Step {step_num}: {step_name}")
    print(f"{'='*80}")


def run_script(script_path, description):
    """
    스크립트 실행
    
    Args:
        script_path: 스크립트 경로
        description: 스크립트 설명
    
    Returns:
        bool: 성공 여부
    """
    print(f"\n실행 중: {description}")
    print(f"스크립트: {script_path}")
    print("-" * 80)
    
    try:
        # Python 스크립트 실행
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=False,
            text=True,
            check=True
        )
        
        print("-" * 80)
        print(f"✅ {description} 완료!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"❌ {description} 실패!")
        print(f"오류: {e}")
        return False
    except Exception as e:
        print("-" * 80)
        print(f"❌ {description} 실행 중 오류!")
        print(f"오류: {e}")
        return False


def check_prerequisites():
    """전제조건 확인"""
    print_step(0, "전제조건 확인")
    
    all_ok = True
    
    # 1. 원본 데이터 파일 확인
    raw_data_path = project_root / 'data' / 'raw' / 'Sugar_substitue_Raw data_251108.xlsx'
    if raw_data_path.exists():
        print(f"✅ 원본 데이터: {raw_data_path}")
    else:
        print(f"❌ 원본 데이터 없음: {raw_data_path}")
        all_ok = False
    
    # 2. 설계 매트릭스 확인
    design_matrix_path = project_root / 'data' / 'processed' / 'dce' / 'design_matrix.csv'
    if design_matrix_path.exists():
        print(f"✅ 설계 매트릭스: {design_matrix_path}")
    else:
        print(f"❌ 설계 매트릭스 없음: {design_matrix_path}")
        print(f"   먼저 'python scripts/create_dce_design_matrix.py'를 실행하세요.")
        all_ok = False
    
    # 3. 필요한 디렉토리 생성
    dirs_to_create = [
        project_root / 'data' / 'processed' / 'survey',
        project_root / 'data' / 'processed' / 'dce',
        project_root / 'data' / 'processed' / 'iclv'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 디렉토리: {dir_path}")
    
    return all_ok


def main():
    """메인 실행 함수"""
    
    print_header("ICLV 데이터 전처리 파이프라인 (326명 버전)")
    
    print("\n이 스크립트는 다음 작업을 수행합니다:")
    print("  1. Survey 파일 생성 (5개 잠재변수)")
    print("  2. DCE 데이터 전처리 (Wide → Long)")
    print("  3. ICLV 데이터 통합 (DCE + Survey + Sociodem)")
    print("  4. sugar_free 변수 추가")
    print("\n입력: data/raw/Sugar_substitue_Raw data_251108.xlsx (326명)")
    print("출력: data/processed/iclv/integrated_data.csv")
    
    # 0. 전제조건 확인
    if not check_prerequisites():
        print("\n❌ 전제조건을 만족하지 않습니다. 위의 오류를 해결하세요.")
        return 1
    
    # 실행할 스크립트 목록
    pipeline_steps = [
        {
            'script': project_root / 'scripts' / 'create_survey_files.py',
            'description': 'Survey 파일 생성 (326명)',
            'step_num': 1
        },
        {
            'script': project_root / 'scripts' / 'preprocess_dce_data.py',
            'description': 'DCE 데이터 전처리',
            'step_num': 2
        },
        {
            'script': project_root / 'scripts' / 'integrate_iclv_data.py',
            'description': 'ICLV 데이터 통합',
            'step_num': 3
        },
        {
            'script': project_root / 'scripts' / 'add_sugar_free_variable.py',
            'description': 'sugar_free 변수 추가',
            'step_num': 4
        }
    ]
    
    # 각 단계 실행
    for step in pipeline_steps:
        print_step(step['step_num'], step['description'])
        
        if not run_script(step['script'], step['description']):
            print(f"\n❌ 파이프라인 실패: {step['description']}")
            return 1
    
    # 최종 요약
    print_header("전처리 파이프라인 완료!")
    
    print("\n생성된 파일:")
    print("-" * 80)
    
    # Survey 파일
    survey_dir = project_root / 'data' / 'processed' / 'survey'
    survey_files = [
        'health_concern.csv',
        'perceived_benefit.csv',
        'purchase_intention.csv',
        'perceived_price.csv',
        'nutrition_knowledge.csv'
    ]
    
    print("\n1. Survey 파일 (data/processed/survey/):")
    for filename in survey_files:
        filepath = survey_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"   ✅ {filename} ({size:,} bytes)")
        else:
            print(f"   ❌ {filename} (없음)")
    
    # DCE 파일
    dce_file = project_root / 'data' / 'processed' / 'dce' / 'dce_long_format.csv'
    print("\n2. DCE 파일 (data/processed/dce/):")
    if dce_file.exists():
        size = dce_file.stat().st_size
        print(f"   ✅ dce_long_format.csv ({size:,} bytes)")
    else:
        print(f"   ❌ dce_long_format.csv (없음)")
    
    # ICLV 통합 파일
    iclv_file = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    print("\n3. ICLV 통합 파일 (data/processed/iclv/):")
    if iclv_file.exists():
        size = iclv_file.stat().st_size
        print(f"   ✅ integrated_data.csv ({size:,} bytes)")
    else:
        print(f"   ❌ integrated_data.csv (없음)")
    
    print("\n" + "=" * 80)
    print("다음 단계:")
    print("  python scripts/test_gpu_batch_iclv.py  # ICLV 모델 추정")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

