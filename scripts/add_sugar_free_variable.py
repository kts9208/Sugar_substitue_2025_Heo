"""
설탕함량 변수 이진 변환 및 Price 스케일링 스크립트

목적: integrated_data.csv에 sugar_free 변수 추가 및 price 스케일링
변환:
  1. sugar_content ('알반당', '무설탕') → sugar_free (0, 1)
     - '무설탕' → 1
     - '알반당' → 0
     - NaN → NaN (구매안함 대안)
  2. price (2000~3000) → price_scaled (2~3)
     - price / 1000

Author: Sugar Substitute Research Team
Date: 2025-11-09
Updated: 2025-11-11 (Price scaling 추가)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    """메인 실행 함수"""
    
    print("=" * 80)
    print("설탕함량 변수 이진 변환 및 Price 스케일링")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = Path('data/processed/iclv/integrated_data.csv')
    df = pd.read_csv(data_path)
    print(f"   - 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼")
    
    # 2. 현재 sugar_content 값 확인
    print("\n[2] 현재 sugar_content 값 확인:")
    print(f"   - 고유값: {df['sugar_content'].unique()}")
    print(f"   - 값 분포:")
    print(df['sugar_content'].value_counts(dropna=False).to_string())
    
    # 3. sugar_free 변수 생성
    print("\n[3] sugar_free 변수 생성 중...")
    
    # 변환 로직
    def convert_sugar_content(value):
        """sugar_content를 sugar_free로 변환"""
        if pd.isna(value):
            return np.nan  # 구매안함 대안
        elif value == '무설탕':
            return 1
        elif value == '알반당':
            return 0
        else:
            print(f"   ⚠️  예상치 못한 값: {value}")
            return np.nan
    
    df['sugar_free'] = df['sugar_content'].apply(convert_sugar_content)

    print(f"   - 변환 완료")
    print(f"   - sugar_free 값 분포:")
    print(df['sugar_free'].value_counts(dropna=False).to_string())

    # 4. Price 스케일링 (1000으로 나누기)
    print("\n[4] Price 스케일링 중...")

    # 원본 price 통계
    print(f"   - 원본 price 통계:")
    print(f"     Min: {df['price'].min():.3f}, Max: {df['price'].max():.3f}, Mean: {df['price'].mean():.3f}")

    # Price가 이미 스케일링되었는지 확인 (2~3 범위면 이미 스케일링됨)
    if df['price'].max() > 100:
        # Price를 1000으로 나누기 (2000~3000 → 2~3)
        df['price'] = df['price'] / 1000.0
        print(f"   - 스케일링 완료 (÷ 1000)")
        print(f"   - 스케일링된 price 통계:")
        print(f"     Min: {df['price'].min():.3f}, Max: {df['price'].max():.3f}, Mean: {df['price'].mean():.3f}")
    else:
        print(f"   - ⚠️  Price가 이미 스케일링되어 있습니다 (건너뛰기)")

    # 5. 검증
    print("\n[5] 변환 검증:")
    
    # 무설탕 → 1 확인
    sugar_free_count = df[df['sugar_content'] == '무설탕']['sugar_free'].value_counts()
    print(f"   - '무설탕' → 1 변환: {sugar_free_count.get(1, 0)}개")
    
    # 알반당 → 0 확인
    regular_sugar_count = df[df['sugar_content'] == '알반당']['sugar_free'].value_counts()
    print(f"   - '알반당' → 0 변환: {regular_sugar_count.get(0, 0)}개")
    
    # NaN 확인
    nan_count = df['sugar_free'].isna().sum()
    original_nan_count = df['sugar_content'].isna().sum()
    print(f"   - NaN 개수: {nan_count}개 (원본: {original_nan_count}개)")
    
    if nan_count == original_nan_count:
        print("   ✅ NaN 개수 일치")
    else:
        print(f"   ⚠️  NaN 개수 불일치!")
    
    # 6. 데이터 미리보기
    print("\n[6] 데이터 미리보기 (첫 10행):")
    print("-" * 80)
    preview_cols = ['respondent_id', 'choice_set', 'alternative',
                    'sugar_content', 'sugar_free', 'health_label', 'price', 'choice']
    print(df[preview_cols].head(10).to_string())

    # 7. 저장
    print("\n[7] 저장 중...")
    
    # 백업 생성
    backup_path = data_path.parent / 'integrated_data_backup.csv'
    if not backup_path.exists():
        df_original = pd.read_csv(data_path)
        df_original.to_csv(backup_path, index=False, encoding='utf-8-sig')
        print(f"   - 백업 생성: {backup_path}")
    
    # 업데이트된 데이터 저장
    df.to_csv(data_path, index=False, encoding='utf-8-sig')
    print(f"   - 저장 완료: {data_path}")
    
    # 8. 최종 요약
    print("\n[8] 최종 요약:")
    print(f"   - 총 행 수: {len(df):,}행")
    print(f"   - 총 컬럼 수: {len(df.columns)}개")
    print(f"   - sugar_free 추가: ✅")
    print(f"     · 무설탕 (1): {(df['sugar_free'] == 1).sum():,}개")
    print(f"     · 일반당 (0): {(df['sugar_free'] == 0).sum():,}개")
    print(f"     · NaN: {df['sugar_free'].isna().sum():,}개")
    print(f"   - price 스케일링: ✅")
    print(f"     · 범위: {df['price'].min():.3f} ~ {df['price'].max():.3f}")
    print(f"     · 평균: {df['price'].mean():.3f}")

    print("\n" + "=" * 80)
    print("설탕함량 변수 이진 변환 및 Price 스케일링 완료!")
    print("=" * 80)


if __name__ == '__main__':
    main()

