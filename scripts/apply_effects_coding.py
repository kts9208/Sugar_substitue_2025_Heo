"""
데이터 정제 스크립트

목적: integrated_data.csv를 정제하여 integrated_data_cleaned.csv 생성
변환:
  1. health_label: 0/1 유지 (dummy coding)
  2. price: 원본 유지 (2000, 2500, 3000)

Author: Sugar Substitute Research Team
Date: 2025-11-16
Updated: 2025-11-17 (Effects coding 제거 - Dummy coding만 사용)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    """메인 실행 함수"""
    
    print("=" * 80)
    print("Effects Coding 적용")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    input_path = Path('data/processed/iclv/integrated_data.csv')
    df = pd.read_csv(input_path)
    print(f"   - 로드 완료: {len(df)}행 × {len(df.columns)}컬럼")
    
    # 2. health_label effects coding
    print("\n[2] health_label effects coding 적용 중...")
    print(f"   - 변환 전 값 분포:")
    print(df['health_label'].value_counts(dropna=False).to_string())
    
    # 0 → -1, 1 → 1
    df['health_label'] = df['health_label'].map({0.0: -1.0, 1.0: 1.0})
    
    print(f"\n   - 변환 후 값 분포:")
    print(df['health_label'].value_counts(dropna=False).to_string())
    print(f"   - 변환 규칙: 0 → -1, 1 → 1")

    # 3. price 확인 (원본 유지)
    print("\n[3] price 확인 (원본 유지)...")
    print(f"   - price 값 분포:")
    print(df['price'].value_counts(dropna=False).to_string())

    # 가격 레벨 확인
    price_levels = df['price'].dropna().unique()
    price_levels = sorted(price_levels)
    print(f"   - 가격 레벨: {price_levels}")

    # 통계
    price_values = df['price'].dropna()
    price_mean = price_values.mean()
    price_std = price_values.std()

    print(f"   - 통계:")
    print(f"     평균: {price_mean:.1f}")
    print(f"     표준편차: {price_std:.1f}")
    print(f"   ✅ 원본 가격 유지 (표준화 없음)")

    # 4. 검증
    print("\n[4] 변환 검증:")
    print(f"   - health_label 고유값: {sorted(df['health_label'].dropna().unique())}")
    print(f"   - price 고유값: {sorted(df['price'].dropna().unique())}")
    print(f"   - health_label 평균: {df['health_label'].mean():.6f} (예상: ~0)")
    print(f"   - price 평균: {df['price'].mean():.1f} (예상: 2500)")
    
    # 5. 데이터 미리보기
    print("\n[5] 데이터 미리보기 (첫 18행):")
    print("-" * 80)
    preview_cols = ['choice_set', 'alternative', 'sugar_content', 'sugar_free', 
                    'health_label', 'price', 'choice']
    print(df[preview_cols].head(18).to_string())
    
    # 6. 저장
    print("\n[6] 저장 중...")
    
    # 백업 (기존 파일이 있으면)
    output_path = Path('data/processed/iclv/integrated_data_cleaned.csv')
    if output_path.exists():
        backup_path = Path('data/processed/iclv/integrated_data_cleaned_backup.csv')
        import shutil
        shutil.copy(output_path, backup_path)
        print(f"   - 기존 파일 백업: {backup_path}")
    
    # 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   - 저장 완료: {output_path}")
    
    # 7. 최종 요약
    print("\n[7] 최종 요약:")
    print(f"   - 총 행 수: {len(df):,}행")
    print(f"   - 총 컬럼 수: {len(df.columns)}개")
    print(f"   - health_label effects coding: ✅")
    print(f"     · 범위: {df['health_label'].min():.1f} ~ {df['health_label'].max():.1f}")
    print(f"     · 평균: {df['health_label'].mean():.6f}")
    print(f"   - price 원본 유지: ✅")
    print(f"     · 범위: {df['price'].min():.1f} ~ {df['price'].max():.1f}")
    print(f"     · 평균: {df['price'].mean():.1f}")
    print(f"     · 표준편차: {df['price'].std():.1f}")

    print("\n" + "=" * 80)
    print("Effects Coding 적용 완료!")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    df = main()

