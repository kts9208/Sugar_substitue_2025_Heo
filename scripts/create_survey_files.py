"""
Survey 파일 생성 스크립트

원본 Excel 파일에서 326명의 Survey 데이터를 추출하여 CSV 파일로 저장합니다.

입력: data/raw/Sugar_substitue_Raw data_251108.xlsx
출력: data/processed/survey/*.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_survey_files():
    """원본 Excel에서 Survey CSV 파일들 생성"""
    
    print("=" * 80)
    print("Survey 파일 생성 (326명 버전)")
    print("=" * 80)
    
    # 1. 원본 데이터 로드
    print("\n[1] 원본 데이터 로드 중...")
    raw_path = 'data/raw/Sugar_substitue_Raw data_251108.xlsx'
    
    if not os.path.exists(raw_path):
        print(f"   ❌ 파일 없음: {raw_path}")
        return False
    
    df = pd.read_excel(raw_path, sheet_name='DATA')
    print(f"   ✅ 로드 완료: {len(df)}명 × {len(df.columns)}컬럼")
    print(f"   개인 ID 범위: {df['no'].min()} ~ {df['no'].max()}")
    
    # 2. 출력 디렉토리 생성
    output_dir = Path('data/processed/survey')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2] 출력 디렉토리: {output_dir}")
    
    # 3. 각 잠재변수별 CSV 파일 생성
    print("\n[3] Survey 파일 생성 중...")
    
    # 3-1. Health Concern (q6-q11)
    print("\n   [3-1] Health Concern (q6-q11)...")
    health_cols = ['no'] + [f'q{i}' for i in range(6, 12)]
    df_health = df[health_cols].copy()
    df_health = df_health.drop_duplicates(subset='no', keep='first')
    
    output_path = output_dir / 'health_concern.csv'
    df_health.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"      ✅ 저장: {output_path}")
    print(f"         개인 수: {len(df_health)}")
    print(f"         지표: {[c for c in health_cols if c != 'no']}")
    
    # 3-2. Perceived Benefit (q12-q17)
    print("\n   [3-2] Perceived Benefit (q12-q17)...")
    benefit_cols = ['no'] + [f'q{i}' for i in range(12, 18)]
    df_benefit = df[benefit_cols].copy()
    df_benefit = df_benefit.drop_duplicates(subset='no', keep='first')
    
    output_path = output_dir / 'perceived_benefit.csv'
    df_benefit.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"      ✅ 저장: {output_path}")
    print(f"         개인 수: {len(df_benefit)}")
    print(f"         지표: {[c for c in benefit_cols if c != 'no']}")
    
    # 3-3. Purchase Intention (q18-q20)
    print("\n   [3-3] Purchase Intention (q18-q20)...")
    purchase_cols = ['no'] + [f'q{i}' for i in range(18, 21)]
    df_purchase = df[purchase_cols].copy()
    df_purchase = df_purchase.drop_duplicates(subset='no', keep='first')
    
    output_path = output_dir / 'purchase_intention.csv'
    df_purchase.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"      ✅ 저장: {output_path}")
    print(f"         개인 수: {len(df_purchase)}")
    print(f"         지표: {[c for c in purchase_cols if c != 'no']}")
    
    # 3-4. Perceived Price (q27-q29)
    print("\n   [3-4] Perceived Price (q27-q29)...")
    price_cols = ['no'] + [f'q{i}' for i in range(27, 30)]
    df_price = df[price_cols].copy()
    df_price = df_price.drop_duplicates(subset='no', keep='first')
    
    output_path = output_dir / 'perceived_price.csv'
    df_price.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"      ✅ 저장: {output_path}")
    print(f"         개인 수: {len(df_price)}")
    print(f"         지표: {[c for c in price_cols if c != 'no']}")
    
    # 3-5. Nutrition Knowledge (q30-q49)
    print("\n   [3-5] Nutrition Knowledge (q30-q49)...")
    nutrition_cols = ['no'] + [f'q{i}' for i in range(30, 50)]
    df_nutrition = df[nutrition_cols].copy()
    df_nutrition = df_nutrition.drop_duplicates(subset='no', keep='first')
    
    output_path = output_dir / 'nutrition_knowledge.csv'
    df_nutrition.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"      ✅ 저장: {output_path}")
    print(f"         개인 수: {len(df_nutrition)}")
    print(f"         지표: q30-q49 (20개)")
    
    # 4. 검증
    print("\n[4] 검증...")
    
    all_files = [
        ('health_concern.csv', df_health, health_cols),
        ('perceived_benefit.csv', df_benefit, benefit_cols),
        ('purchase_intention.csv', df_purchase, purchase_cols),
        ('perceived_price.csv', df_price, price_cols),
        ('nutrition_knowledge.csv', df_nutrition, nutrition_cols)
    ]
    
    all_valid = True
    for filename, df_check, cols in all_files:
        filepath = output_dir / filename
        
        # 파일 존재 확인
        if not filepath.exists():
            print(f"   ❌ {filename}: 파일 없음")
            all_valid = False
            continue
        
        # 개인 수 확인
        if len(df_check) != len(df):
            print(f"   ⚠️  {filename}: 개인 수 불일치 ({len(df_check)} != {len(df)})")
        
        # 결측치 확인
        missing_count = df_check.isnull().sum().sum()
        if missing_count > 0:
            print(f"   ⚠️  {filename}: 결측치 {missing_count}개")
        else:
            print(f"   ✅ {filename}: 결측치 없음, {len(df_check)}명")
    
    # 5. 요약
    print("\n[5] 요약:")
    print("-" * 80)
    print(f"   원본 데이터: {len(df)}명")
    print(f"   생성된 파일: 5개")
    print(f"   총 지표 수: 38개")
    print(f"     - Health Concern: 6개 (q6-q11)")
    print(f"     - Perceived Benefit: 6개 (q12-q17)")
    print(f"     - Purchase Intention: 3개 (q18-q20)")
    print(f"     - Perceived Price: 3개 (q27-q29)")
    print(f"     - Nutrition Knowledge: 20개 (q30-q49)")
    
    if all_valid:
        print("\n✅ 모든 Survey 파일 생성 완료!")
    else:
        print("\n⚠️  일부 파일 생성 실패")
    
    print("=" * 80)
    
    return all_valid


def main():
    """메인 실행 함수"""
    success = create_survey_files()
    
    if success:
        print("\n다음 단계:")
        print("  1. python scripts/preprocess_dce_data.py  # DCE 데이터 전처리")
        print("  2. python scripts/integrate_iclv_data.py  # ICLV 데이터 통합")
        print("  3. python scripts/test_gpu_batch_iclv.py  # ICLV 모델 추정")
    else:
        print("\n오류가 발생했습니다. 로그를 확인하세요.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

