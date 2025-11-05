"""
ICLV 데이터 준비 상태 테스트

목적: ICLV 동시추정을 위한 데이터가 제대로 준비되었는지 확인
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("ICLV 데이터 준비 상태 테스트")
print("=" * 80)

# 1. 데이터 로드
print("\n[1] 통합 데이터 로드...")
df = pd.read_csv('data/processed/iclv/integrated_data.csv')
print(f"   - 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼")

# 2. 필수 컬럼 확인
print("\n[2] 필수 컬럼 확인...")

# DCE 변수
dce_cols = ['respondent_id', 'choice_set', 'alternative', 'choice', 'health_label', 'price']
print(f"\n   [DCE 변수]")
for col in dce_cols:
    exists = col in df.columns
    status = "✓" if exists else "✗"
    print(f"   {status} {col}")

# 측정모델 지표
indicator_cols = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
print(f"\n   [측정모델 지표]")
for col in indicator_cols:
    exists = col in df.columns
    status = "✓" if exists else "✗"
    print(f"   {status} {col}")

# 구조모델 변수
structural_cols = ['age_std', 'gender', 'income_std', 'education_level']
print(f"\n   [구조모델 변수]")
for col in structural_cols:
    exists = col in df.columns
    status = "✓" if exists else "✗"
    print(f"   {status} {col}")

# 3. 데이터 준비 (구매안함 제외)
print("\n[3] 데이터 준비 (구매안함 제외)...")
df_prepared = df[df['alternative'] != 3].copy()
print(f"   - 원본: {len(df):,}행")
print(f"   - 준비 후: {len(df_prepared):,}행")
print(f"   - 응답자 수: {df_prepared['respondent_id'].nunique()}")

# 4. 결측치 확인
print("\n[4] 결측치 확인...")
critical_cols = dce_cols + indicator_cols + structural_cols

print(f"\n   [필수 변수 결측치]")
for col in critical_cols:
    if col in df_prepared.columns:
        missing = df_prepared[col].isnull().sum()
        pct = missing / len(df_prepared) * 100
        if missing > 0:
            print(f"   ⚠ {col}: {missing}개 ({pct:.1f}%)")
        else:
            print(f"   ✓ {col}: 결측치 없음")

# 5. 데이터 분포 확인
print("\n[5] 데이터 분포 확인...")

# 선택 분포
print(f"\n   [선택 분포]")
choice_dist = df_prepared.groupby('alternative')['choice'].sum()
for alt, count in choice_dist.items():
    pct = count / choice_dist.sum() * 100
    print(f"   - 대안 {alt}: {count}회 ({pct:.1f}%)")

# 건강 라벨 분포
print(f"\n   [건강 라벨 분포]")
label_dist = df_prepared[df_prepared['choice'] == 1]['health_label'].value_counts()
for label, count in label_dist.items():
    pct = count / label_dist.sum() * 100
    label_name = "있음" if label == 1 else "없음"
    print(f"   - {label_name}: {count}회 ({pct:.1f}%)")

# 가격 분포
print(f"\n   [가격 분포]")
price_dist = df_prepared[df_prepared['choice'] == 1]['price'].value_counts().sort_index()
for price, count in price_dist.items():
    pct = count / price_dist.sum() * 100
    print(f"   - ₩{price:,.0f}: {count}회 ({pct:.1f}%)")

# 측정모델 지표 분포
print(f"\n   [측정모델 지표 기술통계]")
print(df_prepared[indicator_cols].describe().T[['mean', 'std', 'min', 'max']])

# 구조모델 변수 분포
print(f"\n   [구조모델 변수 기술통계]")
print(df_prepared[structural_cols].describe().T[['mean', 'std', 'min', 'max']])

# 6. ICLV 모델 설정 제안
print("\n[6] ICLV 모델 설정 제안...")
print("""
# 측정모델 설정
measurement_config = MeasurementConfig(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    indicator_type='ordered',
    n_categories=7  # 7점 Likert 척도
)

# 구조모델 설정
structural_config = StructuralConfig(
    sociodemographics=['age_std', 'gender', 'income_std', 'education_level'],
    include_in_choice=True
)

# 선택모델 설정
choice_config = ChoiceConfig(
    choice_attributes=['health_label', 'price'],
    price_variable='price',
    choice_type='binary',
    lv_in_choice=True
)

# ICLV 통합 설정
iclv_config = ICLVConfig(
    measurement=measurement_config,
    structural=structural_config,
    choice=choice_config,
    n_draws=500,  # Halton draws
    seed=42
)
""")

# 7. 최종 평가
print("\n[7] 최종 평가...")
print("=" * 80)

all_ready = True

# 필수 컬럼 체크
for col in critical_cols:
    if col not in df_prepared.columns:
        print(f"   ✗ {col} 컬럼 누락")
        all_ready = False

# 결측치 체크 (income_std 제외)
for col in critical_cols:
    if col in df_prepared.columns and col != 'income_std':
        if df_prepared[col].isnull().sum() > 0:
            print(f"   ⚠ {col} 결측치 있음")

# 데이터 크기 체크
if len(df_prepared) < 1000:
    print(f"   ⚠ 데이터 크기 작음: {len(df_prepared)}행")

if all_ready:
    print("\n   ✅ ICLV 동시추정 준비 완료!")
    print(f"   - 데이터: {len(df_prepared):,}행")
    print(f"   - 응답자: {df_prepared['respondent_id'].nunique()}명")
    print(f"   - 선택 세트: {df_prepared['choice_set'].nunique()}개")
    print(f"   - 측정모델 지표: {len(indicator_cols)}개")
    print(f"   - 구조모델 변수: {len(structural_cols)}개")
    print(f"   - 선택모델 속성: 2개 (health_label, price)")
else:
    print("\n   ✗ 데이터 준비 미완료")

print("\n" + "=" * 80)
print("테스트 완료!")
print("=" * 80)

