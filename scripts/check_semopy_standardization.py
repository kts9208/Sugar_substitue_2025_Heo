"""
semopy의 predict_factors()가 표준화를 하는지 확인
"""

import numpy as np
import pandas as pd
from semopy import Model

# 간단한 테스트 데이터 생성
np.random.seed(42)
n = 100

# 원본 잠재변수 (평균=5, 표준편차=2)
true_latent = np.random.normal(5, 2, n)

# 관측 지표 3개 (잠재변수 + 오차)
x1 = true_latent + np.random.normal(0, 0.5, n)
x2 = true_latent + np.random.normal(0, 0.5, n)
x3 = true_latent + np.random.normal(0, 0.5, n)

data = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3
})

print("=" * 100)
print("semopy predict_factors() 표준화 여부 확인")
print("=" * 100)

print("\n[1] 원본 데이터 통계")
print(f"x1: mean={data['x1'].mean():.4f}, std={data['x1'].std():.4f}")
print(f"x2: mean={data['x2'].mean():.4f}, std={data['x2'].std():.4f}")
print(f"x3: mean={data['x3'].mean():.4f}, std={data['x3'].std():.4f}")

# CFA 모델
model_spec = """
latent =~ x1 + x2 + x3
"""

model = Model(model_spec)
model.fit(data)

print("\n[2] semopy predict_factors() 결과")
factor_scores = model.predict_factors(data)
print(f"Type: {type(factor_scores)}")
print(f"Shape: {factor_scores.shape}")
print(f"Columns: {list(factor_scores.columns)}")

latent_scores = factor_scores['latent'].values
print(f"\nlatent: mean={latent_scores.mean():.6f}, std={latent_scores.std():.6f}")
print(f"        min={latent_scores.min():.4f}, max={latent_scores.max():.4f}")

# 수동 계산 (Bartlett 방법)
print("\n[3] 수동 계산 (Bartlett 방법)")
params = model.inspect()
loadings = params[params['op'] == '~'].copy()  # semopy는 ~ 사용
print(f"\nLoadings:\n{loadings[['lval', 'rval', 'Estimate']]}")

lambda_values = loadings['Estimate'].values
X = data[['x1', 'x2', 'x3']].values
Lambda = lambda_values.reshape(-1, 1)

Lambda_T_Lambda = Lambda.T @ Lambda
Lambda_T_Lambda_inv = 1.0 / Lambda_T_Lambda[0, 0]
manual_scores = Lambda_T_Lambda_inv * (Lambda.T @ X.T)
manual_scores = manual_scores.flatten()

print(f"\nmanual: mean={manual_scores.mean():.6f}, std={manual_scores.std():.6f}")
print(f"        min={manual_scores.min():.4f}, max={manual_scores.max():.4f}")

# 비교
print("\n[4] 비교")
print(f"predict_factors와 수동 계산의 상관계수: {np.corrcoef(latent_scores, manual_scores)[0, 1]:.6f}")

# 표준화 여부 확인
if abs(latent_scores.mean()) < 1e-10 and abs(latent_scores.std() - 1.0) < 0.01:
    print("\n✅ predict_factors()는 표준화된 요인점수를 반환합니다!")
else:
    print("\n❌ predict_factors()는 표준화하지 않습니다.")

if abs(manual_scores.mean() - 5.0) < 1.0:
    print("✅ 수동 계산은 원본 스케일의 요인점수를 반환합니다!")
else:
    print("❌ 수동 계산도 표준화된 것 같습니다.")

print("\n" + "=" * 100)

