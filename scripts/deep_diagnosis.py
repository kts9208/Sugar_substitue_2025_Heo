"""
심층 진단: 다중공선성, 상관관계, 모델 설정 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '..')

print('=' * 100)
print('심층 진단: 잠재변수 비유의성 원인')
print('=' * 100)
print()

# 데이터 로드
data_path = Path('../data/processed/integrated_data.csv')
df = pd.read_csv(data_path)

# 요인점수 로드
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_선택모델_확장_후_*.csv'))
if not files:
    print("요인점수 파일을 찾을 수 없습니다!")
    exit(1)

df_fs = pd.read_csv(files[-1])

# 데이터 병합 (인덱스 기준)
df_combined = df.copy()
for col in df_fs.columns:
    df_combined[f'fs_{col}'] = df_fs[col].values

print('1. 데이터 기본 정보')
print('-' * 100)
print(f'전체 행 수: {len(df_combined)}')
print(f'선택 변수: {df_combined["choice"].value_counts().to_dict()}')
print()

# 2. 속성변수와 요인점수 상관관계
print('2. 속성변수와 요인점수 상관관계')
print('-' * 100)
print()

# 속성변수
attr_vars = ['attr_sugar_free', 'attr_health_label', 'attr_price']
# 요인점수
fs_vars = ['fs_purchase_intention', 'fs_perceived_price', 'fs_nutrition_knowledge']

# 상관관계 계산
corr_data = df_combined[attr_vars + fs_vars].corr()

print('속성변수 vs 요인점수 상관관계:')
print()
for attr in attr_vars:
    print(f'{attr}:')
    for fs in fs_vars:
        corr = corr_data.loc[attr, fs]
        print(f'  - {fs:35s}: {corr:>8.4f}')
    print()

# 3. 요인점수 간 상관관계
print('3. 요인점수 간 상관관계')
print('-' * 100)
print()

fs_corr = df_combined[fs_vars].corr()
print(fs_corr.to_string())
print()

# 4. 조절효과 변수 생성 및 분석
print('4. 조절효과 변수 분석')
print('-' * 100)
print()

# 조절효과 변수 생성 (모델과 동일하게)
df_combined['mod_pp'] = df_combined['attr_price'] * df_combined['fs_perceived_price']
df_combined['mod_nk'] = df_combined['attr_price'] * df_combined['fs_nutrition_knowledge']

print('조절효과 변수 통계:')
print()
print(f'attr_price * fs_perceived_price:')
print(f'  평균: {df_combined["mod_pp"].mean():>10.6f}')
print(f'  분산: {df_combined["mod_pp"].var():>10.6f}')
print(f'  표준편차: {df_combined["mod_pp"].std():>10.6f}')
print(f'  범위: [{df_combined["mod_pp"].min():>8.4f}, {df_combined["mod_pp"].max():>8.4f}]')
print()

print(f'attr_price * fs_nutrition_knowledge:')
print(f'  평균: {df_combined["mod_nk"].mean():>10.6f}')
print(f'  분산: {df_combined["mod_nk"].var():>10.6f}')
print(f'  표준편차: {df_combined["mod_nk"].std():>10.6f}')
print(f'  범위: [{df_combined["mod_nk"].min():>8.4f}, {df_combined["mod_nk"].max():>8.4f}]')
print()

# 5. 다중공선성 진단 (VIF)
print('5. 다중공선성 진단 (VIF)')
print('-' * 100)
print()

from sklearn.linear_model import LinearRegression

def calculate_vif(df, features):
    """VIF 계산"""
    vif_data = []
    for i, feature in enumerate(features):
        X = df[features].drop(columns=[feature])
        y = df[feature]
        
        # 결측값 제거
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) > 0:
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            r_squared = model.score(X_clean, y_clean)
            
            if r_squared < 0.9999:  # 완전 공선성 방지
                vif = 1 / (1 - r_squared)
            else:
                vif = np.inf
        else:
            vif = np.nan
        
        vif_data.append({'Variable': feature, 'VIF': vif})
    
    return pd.DataFrame(vif_data)

# 선택모델 변수들
choice_vars = [
    'attr_sugar_free', 
    'attr_health_label', 
    'attr_price',
    'fs_purchase_intention',
    'mod_pp',
    'mod_nk'
]

vif_df = calculate_vif(df_combined, choice_vars)
print('VIF (Variance Inflation Factor):')
print('  - VIF < 5: 문제 없음')
print('  - 5 <= VIF < 10: 주의 필요')
print('  - VIF >= 10: 심각한 다중공선성')
print()
print(vif_df.to_string(index=False))
print()

# 6. 선택 패턴 분석
print('6. 선택 패턴 분석')
print('-' * 100)
print()

# 선택된 대안의 특성
chosen = df_combined[df_combined['choice'] == 1]

print(f'선택된 대안 (N={len(chosen)}):')
print()
print('속성변수 평균:')
for var in attr_vars:
    print(f'  {var:20s}: {chosen[var].mean():>8.4f}')
print()

print('요인점수 평균:')
for var in fs_vars:
    print(f'  {var:35s}: {chosen[var].mean():>8.4f}')
print()

# 선택되지 않은 대안
not_chosen = df_combined[df_combined['choice'] == 0]

print(f'선택되지 않은 대안 (N={len(not_chosen)}):')
print()
print('속성변수 평균:')
for var in attr_vars:
    print(f'  {var:20s}: {not_chosen[var].mean():>8.4f}')
print()

print('요인점수 평균:')
for var in fs_vars:
    print(f'  {var:35s}: {not_chosen[var].mean():>8.4f}')
print()

# 차이 검정
print('선택 vs 비선택 차이:')
print()
print('속성변수:')
for var in attr_vars:
    diff = chosen[var].mean() - not_chosen[var].mean()
    print(f'  {var:20s}: {diff:>8.4f}')
print()

print('요인점수:')
for var in fs_vars:
    diff = chosen[var].mean() - not_chosen[var].mean()
    print(f'  {var:35s}: {diff:>8.4f}')
print()

print('=' * 100)

