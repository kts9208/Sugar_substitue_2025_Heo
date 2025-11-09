"""
변경사항 검증 스크립트

목적: 데이터와 선택모델 설정이 올바르게 수정되었는지 확인
"""

import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig


def verify_data():
    """데이터 검증"""
    print("=" * 80)
    print("1. 데이터 검증")
    print("=" * 80)
    
    df = pd.read_csv('data/processed/iclv/integrated_data.csv')
    
    print(f"\n총 행 수: {len(df):,}")
    print(f"총 컬럼 수: {len(df.columns)}")
    
    print(f"\n[sugar_free 변수 확인]")
    print(f"  - 존재 여부: {'sugar_free' in df.columns}")
    
    if 'sugar_free' in df.columns:
        print(f"  - 무설탕 (1): {(df['sugar_free'] == 1).sum():,}개")
        print(f"  - 일반당 (0): {(df['sugar_free'] == 0).sum():,}개")
        print(f"  - NaN: {df['sugar_free'].isna().sum():,}개")
        print(f"  ✅ sugar_free 변수 정상 추가됨")
    else:
        print(f"  ❌ sugar_free 변수가 없습니다!")
        return False
    
    print(f"\n[선택모델 변수 확인]")
    required_vars = ['sugar_free', 'health_label', 'price']
    all_exist = True
    
    for var in required_vars:
        exists = var in df.columns
        non_nan_count = df[var].notna().sum() if exists else 0
        status = "✅" if exists else "❌"
        print(f"  {status} {var}: {non_nan_count:,}개 (NaN 제외)")
        if not exists:
            all_exist = False
    
    if all_exist:
        print(f"\n✅ 모든 선택모델 변수가 존재합니다")
        return True
    else:
        print(f"\n❌ 일부 선택모델 변수가 없습니다!")
        return False


def verify_config():
    """선택모델 설정 검증"""
    print("\n" + "=" * 80)
    print("2. 선택모델 설정 검증")
    print("=" * 80)
    
    # test_iclv_full_data.py 파일 읽기
    with open('scripts/test_iclv_full_data.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # choice_attributes 찾기
    if "choice_attributes=['sugar_free', 'health_label', 'price']" in content:
        print(f"\n✅ choice_attributes 설정 확인:")
        print(f"   - sugar_free ✅")
        print(f"   - health_label ✅")
        print(f"   - price ✅")
        
        # 파라미터 출력 부분도 확인
        count = content.count("choice_attrs = ['sugar_free', 'health_label', 'price']")
        print(f"\n✅ 파라미터 출력 부분 수정 확인: {count}개 위치")
        
        return True
    else:
        print(f"\n❌ choice_attributes 설정이 올바르지 않습니다!")
        
        # 현재 설정 찾기
        import re
        pattern = r"choice_attributes=\[(.*?)\]"
        matches = re.findall(pattern, content)
        if matches:
            print(f"   현재 설정: {matches[0]}")
        
        return False


def main():
    """메인 실행 함수"""
    
    print("\n" + "=" * 80)
    print("변경사항 검증")
    print("=" * 80 + "\n")
    
    # 1. 데이터 검증
    data_ok = verify_data()
    
    # 2. 설정 검증
    config_ok = verify_config()
    
    # 3. 최종 결과
    print("\n" + "=" * 80)
    print("최종 검증 결과")
    print("=" * 80)
    
    print(f"\n1. 데이터 전처리: {'✅ 성공' if data_ok else '❌ 실패'}")
    print(f"2. 선택모델 설정: {'✅ 성공' if config_ok else '❌ 실패'}")
    
    if data_ok and config_ok:
        print(f"\n🎉 모든 변경사항이 올바르게 적용되었습니다!")
        print(f"\n다음 단계:")
        print(f"  1. 모델 재추정: python scripts/test_iclv_full_data.py")
        print(f"  2. 결과 확인: results/iclv_full_data_results.csv")
        return 0
    else:
        print(f"\n⚠️  일부 변경사항에 문제가 있습니다. 위 내용을 확인하세요.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

