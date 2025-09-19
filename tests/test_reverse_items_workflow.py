#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
역문항 처리 워크플로우 테스트 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append('.')

from processed_data.modules.reverse_items_processor import ReverseItemsProcessor


def test_reverse_items_config():
    """역문항 설정 파일 테스트"""
    print("=" * 60)
    print("역문항 설정 파일 테스트")
    print("=" * 60)
    
    try:
        processor = ReverseItemsProcessor()
        config = processor.config
        
        print("✓ 설정 파일 로드 성공")
        print(f"  - 척도 범위: {config['scale_range']['min']}-{config['scale_range']['max']}")
        print(f"  - 전체 요인 수: {len(config['reverse_items'])}")
        
        # 역문항 정보 검증
        total_reverse_items = 0
        for factor_name, factor_config in config['reverse_items'].items():
            reverse_items = factor_config.get('reverse_items', [])
            total_reverse_items += len(reverse_items)
            print(f"  - {factor_name}: {len(reverse_items)}개 역문항")
        
        print(f"  - 전체 역문항 수: {total_reverse_items}개")
        
        return True
        
    except Exception as e:
        print(f"✗ 설정 파일 테스트 실패: {e}")
        return False


def test_reverse_coding_logic():
    """역코딩 로직 테스트"""
    print("\n" + "=" * 60)
    print("역코딩 로직 테스트")
    print("=" * 60)
    
    try:
        processor = ReverseItemsProcessor()
        
        # 테스트 케이스
        test_cases = [
            (1, 5),  # 1 → 5
            (2, 4),  # 2 → 4
            (3, 3),  # 3 → 3
            (4, 2),  # 4 → 2
            (5, 1),  # 5 → 1
        ]
        
        print("역코딩 공식 테스트:")
        all_passed = True
        
        for original, expected in test_cases:
            reversed_val = processor._reverse_code_value(original)
            passed = reversed_val == expected
            all_passed = all_passed and passed
            
            status = "✓" if passed else "✗"
            print(f"  {status} {original} → {reversed_val} (예상: {expected})")
        
        # NaN 처리 테스트
        nan_result = processor._reverse_code_value(np.nan)
        nan_passed = pd.isna(nan_result)
        all_passed = all_passed and nan_passed
        
        status = "✓" if nan_passed else "✗"
        print(f"  {status} NaN → NaN (결측값 처리)")
        
        if all_passed:
            print("✓ 역코딩 로직 테스트 통과")
            return True
        else:
            print("✗ 역코딩 로직 테스트 실패")
            return False
        
    except Exception as e:
        print(f"✗ 역코딩 로직 테스트 중 오류: {e}")
        return False


def test_data_validation():
    """데이터 유효성 검증 테스트"""
    print("\n" + "=" * 60)
    print("데이터 유효성 검증 테스트")
    print("=" * 60)
    
    try:
        processor = ReverseItemsProcessor()
        
        # 정상 데이터 테스트
        normal_data = pd.DataFrame({
            'no': [1, 2, 3],
            'q1': [1, 3, 5],
            'q2': [2, 4, 1]
        })
        
        is_valid, errors = processor._validate_data(normal_data, 'test_factor')
        
        if is_valid:
            print("✓ 정상 데이터 검증 통과")
        else:
            print(f"✗ 정상 데이터 검증 실패: {errors}")
            return False
        
        # 범위 초과 데이터 테스트
        invalid_data = pd.DataFrame({
            'no': [1, 2, 3],
            'q1': [0, 3, 6],  # 범위 초과
            'q2': [2, 4, 1]
        })
        
        is_valid, errors = processor._validate_data(invalid_data, 'test_factor')
        
        if not is_valid and len(errors) > 0:
            print("✓ 범위 초과 데이터 검증 통과")
            print(f"  - 감지된 오류: {len(errors)}개")
        else:
            print("✗ 범위 초과 데이터 검증 실패")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 데이터 유효성 검증 테스트 중 오류: {e}")
        return False


def test_backup_functionality():
    """백업 기능 테스트"""
    print("\n" + "=" * 60)
    print("백업 기능 테스트")
    print("=" * 60)
    
    try:
        processor = ReverseItemsProcessor()
        
        # 백업 디렉토리 확인
        backup_dir = processor.backup_dir
        print(f"백업 디렉토리: {backup_dir}")
        
        if backup_dir.exists():
            print("✓ 백업 디렉토리 존재")
            
            # 백업 파일들 확인
            backup_subdirs = list(backup_dir.glob("backup_*"))
            if backup_subdirs:
                latest_backup = max(backup_subdirs, key=lambda x: x.name)
                backup_files = list(latest_backup.glob("*.csv"))
                print(f"✓ 최신 백업: {latest_backup.name}")
                print(f"  - 백업된 파일 수: {len(backup_files)}개")
                
                # 백업 파일 내용 검증
                if backup_files:
                    sample_file = backup_files[0]
                    try:
                        backup_data = pd.read_csv(sample_file)
                        print(f"  - 샘플 파일 검증: {sample_file.name} ({backup_data.shape[0]}행)")
                        print("✓ 백업 파일 내용 검증 통과")
                    except Exception as e:
                        print(f"✗ 백업 파일 내용 검증 실패: {e}")
                        return False
            else:
                print("ℹ️ 백업 파일 없음 (정상 - 아직 백업 실행 안됨)")
        else:
            print("ℹ️ 백업 디렉토리 없음 (정상 - 첫 실행)")
        
        return True
        
    except Exception as e:
        print(f"✗ 백업 기능 테스트 중 오류: {e}")
        return False


def test_processed_data_comparison():
    """처리 전후 데이터 비교 테스트"""
    print("\n" + "=" * 60)
    print("처리 전후 데이터 비교 테스트")
    print("=" * 60)
    
    try:
        processor = ReverseItemsProcessor()
        
        # 백업 데이터와 현재 데이터 비교
        backup_dir = processor.backup_dir
        data_dir = processor.data_dir
        
        if not backup_dir.exists():
            print("ℹ️ 백업 데이터가 없어 비교를 건너뜁니다.")
            return True
        
        backup_subdirs = list(backup_dir.glob("backup_*"))
        if not backup_subdirs:
            print("ℹ️ 백업 파일이 없어 비교를 건너뜁니다.")
            return True
        
        latest_backup = max(backup_subdirs, key=lambda x: x.name)
        
        # 역문항이 있는 요인들만 비교
        config = processor.config['reverse_items']
        comparison_results = []
        
        for factor_name, factor_config in config.items():
            reverse_items = factor_config.get('reverse_items', [])
            if not reverse_items:
                continue
            
            # 백업 파일과 현재 파일 로드
            backup_file = latest_backup / f"{factor_name}.csv"
            current_file = data_dir / f"{factor_name}.csv"
            
            if backup_file.exists() and current_file.exists():
                backup_data = pd.read_csv(backup_file)
                current_data = pd.read_csv(current_file)
                
                # 역문항들의 평균값 비교
                for item in reverse_items:
                    if item in backup_data.columns and item in current_data.columns:
                        backup_mean = backup_data[item].mean()
                        current_mean = current_data[item].mean()
                        
                        # 역코딩 검증: 백업(원본) → 현재(역코딩됨)
                        expected_mean = (processor.scale_max + processor.scale_min) - backup_mean
                        diff = abs(current_mean - expected_mean)

                        if diff < 0.001:  # 부동소수점 오차 고려
                            comparison_results.append(f"✓ {factor_name}.{item}: {backup_mean:.3f} → {current_mean:.3f} (역코딩 성공)")
                        else:
                            comparison_results.append(f"✗ {factor_name}.{item}: 역코딩 오류 (원본: {backup_mean:.3f}, 예상: {expected_mean:.3f}, 실제: {current_mean:.3f})")
        
        if comparison_results:
            print("역문항 처리 결과 검증:")
            for result in comparison_results:
                print(f"  {result}")
            
            # 모든 결과가 성공인지 확인
            success_count = sum(1 for r in comparison_results if "✓" in r)
            total_count = len(comparison_results)
            
            if success_count == total_count:
                print(f"✓ 전체 역문항 처리 검증 통과 ({success_count}/{total_count})")
                return True
            else:
                print(f"✗ 역문항 처리 검증 실패 ({success_count}/{total_count})")
                return False
        else:
            print("ℹ️ 비교할 역문항이 없습니다.")
            return True
        
    except Exception as e:
        print(f"✗ 처리 전후 데이터 비교 테스트 중 오류: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("역문항 처리 워크플로우 테스트 시작")
    print("=" * 80)
    
    tests = [
        ("설정 파일", test_reverse_items_config),
        ("역코딩 로직", test_reverse_coding_logic),
        ("데이터 유효성 검증", test_data_validation),
        ("백업 기능", test_backup_functionality),
        ("처리 전후 데이터 비교", test_processed_data_comparison),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name} 테스트 실행 중...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 테스트 통과")
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
    
    print("\n" + "=" * 80)
    print(f"테스트 결과: {passed_tests}/{total_tests} 통과")
    
    if passed_tests == total_tests:
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        return True
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
