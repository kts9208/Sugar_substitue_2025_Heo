"""
가격 변수 Z-score 표준화

이 스크립트는 integrated_data.csv의 가격 변수를 Z-score 표준화합니다.

입력:
  - data/processed/iclv/integrated_data.csv

출력:
  - data/processed/iclv/integrated_data.csv (덮어쓰기)
  - 가격 변환 정보 로그

변환:
  - price: Z-score 표준화 (평균 0, 표준편차 1)
    z = (x - mean(x)) / std(x)

Author: Sugar Substitute Research Team
Date: 2025-01-17
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def standardize_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    가격 변수 Z-score 표준화
    
    Args:
        data: 원본 데이터프레임
    
    Returns:
        표준화된 데이터프레임
    """
    df = data.copy()
    
    # 가격 변수 확인
    if 'price' not in df.columns:
        raise ValueError("'price' 컬럼이 데이터에 없습니다.")
    
    # 원본 통계
    price_values = df['price'].dropna()
    original_mean = price_values.mean()
    original_std = price_values.std(ddof=0)  # 모집단 표준편차
    
    print("\n[원본 가격 통계]")
    print(f"  평균: {original_mean:.2f}")
    print(f"  표준편차: {original_std:.2f}")
    print(f"  최소: {price_values.min():.2f}")
    print(f"  최대: {price_values.max():.2f}")
    print(f"  고유값: {sorted(price_values.unique())}")
    
    # Z-score 표준화
    # NaN이 아닌 값만 표준화
    mask = df['price'].notna()
    df.loc[mask, 'price'] = (df.loc[mask, 'price'] - original_mean) / original_std
    
    # 표준화 후 통계
    standardized_values = df['price'].dropna()
    new_mean = standardized_values.mean()
    new_std = standardized_values.std(ddof=0)
    
    print("\n[표준화 후 가격 통계]")
    print(f"  평균: {new_mean:.6f}")
    print(f"  표준편차: {new_std:.6f}")
    print(f"  최소: {standardized_values.min():.6f}")
    print(f"  최대: {standardized_values.max():.6f}")
    
    # 변환 정보 저장 (역변환용)
    print("\n[변환 정보]")
    print(f"  원본 평균: {original_mean:.2f}")
    print(f"  원본 표준편차: {original_std:.2f}")
    print(f"  변환 공식: z = (price - {original_mean:.2f}) / {original_std:.2f}")
    print(f"  역변환 공식: price = z * {original_std:.2f} + {original_mean:.2f}")
    
    return df


def main():
    """메인 실행 함수"""
    
    print("=" * 80)
    print("가격 변수 Z-score 표준화")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(data)}행, {len(data.columns)}열")
    
    # 2. 가격 표준화
    print("\n[2] 가격 Z-score 표준화 중...")
    data_standardized = standardize_price(data)
    print("✅ 표준화 완료")
    
    # 3. 저장
    print("\n[3] 데이터 저장 중...")
    data_standardized.to_csv(data_path, index=False)
    print(f"✅ 저장 완료: {data_path}")
    
    # 4. 검증
    print("\n[4] 검증 중...")
    df_check = pd.read_csv(data_path)
    price_check = df_check['price'].dropna()
    
    print(f"  저장된 데이터 행 수: {len(df_check)}")
    print(f"  가격 평균: {price_check.mean():.6f}")
    print(f"  가격 표준편차: {price_check.std(ddof=0):.6f}")
    
    if abs(price_check.mean()) < 1e-6 and abs(price_check.std(ddof=0) - 1.0) < 1e-6:
        print("✅ 검증 성공: 평균 ≈ 0, 표준편차 ≈ 1")
    else:
        print("⚠️  검증 실패: 표준화가 제대로 되지 않았습니다.")
    
    print("\n" + "=" * 80)
    print("가격 표준화 완료!")
    print("=" * 80)


if __name__ == '__main__':
    main()

