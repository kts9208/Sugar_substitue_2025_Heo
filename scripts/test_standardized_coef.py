"""
표준화 계수 테스트

semopy에서 비표준화 계수와 표준화 계수를 동시에 얻을 수 있는지 테스트
"""

import pandas as pd
from semopy import Model

# 간단한 테스트 데이터 생성
import numpy as np
np.random.seed(42)

n = 100
data = pd.DataFrame({
    'x1': np.random.randn(n),
    'x2': np.random.randn(n),
    'x3': np.random.randn(n),
    'y1': np.random.randn(n),
    'y2': np.random.randn(n),
    'y3': np.random.randn(n),
})

# 간단한 CFA 모델
model_spec = """
# Measurement Model
LV1 =~ x1 + x2 + x3
LV2 =~ y1 + y2 + y3

# Structural Model
LV2 ~ LV1
"""

# 모델 추정
model = Model(model_spec)
model.fit(data)

print("=" * 80)
print("비표준화 계수만 (std_est=False)")
print("=" * 80)
params_unstd = model.inspect(std_est=False)
print("\n컬럼:", params_unstd.columns.tolist())
print("\n파라미터:")
print(params_unstd[['lval', 'op', 'rval', 'Estimate', 'Std. Err', 'p-value']])

print("\n" + "=" * 80)
print("표준화 계수 포함 (std_est=True)")
print("=" * 80)
params_std = model.inspect(std_est=True)
print("\n컬럼:", params_std.columns.tolist())
print("\n파라미터:")
print(params_std[['lval', 'op', 'rval', 'Estimate', 'Est. Std', 'Std. Err', 'p-value']])

print("\n" + "=" * 80)
print("결론")
print("=" * 80)
print("✅ 한 번의 model.fit() 호출로 비표준화 계수와 표준화 계수를 모두 얻을 수 있습니다!")
print("✅ inspect(std_est=True)를 사용하면 'Est. Std' 컬럼이 추가됩니다.")
print("✅ 'Estimate' 컬럼은 비표준화 계수, 'Est. Std' 컬럼은 표준화 계수입니다.")
print()
print("비교 예시 (LV2 ~ LV1):")
print(f"  비표준화 계수: {params_std.loc[0, 'Estimate']:.4f}")
print(f"  표준화 계수:   {params_std.loc[0, 'Est. Std']:.4f}")

