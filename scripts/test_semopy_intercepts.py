"""
semopy 절편 추정 기능 테스트
"""
import pandas as pd
import numpy as np
from semopy import Model

print("="*80)
print("semopy 절편 추정 기능 테스트")
print("="*80)

# 테스트 데이터 생성
np.random.seed(42)
n = 100

# 잠재변수 (표준화)
lv = np.random.randn(n)

# 관측 지표 (절편 있음)
q1 = 3.5 + 1.0 * lv + np.random.randn(n) * 0.5  # 절편 = 3.5
q2 = 3.8 + 0.9 * lv + np.random.randn(n) * 0.5  # 절편 = 3.8
q3 = 4.0 + 0.8 * lv + np.random.randn(n) * 0.5  # 절편 = 4.0

data = pd.DataFrame({
    'q1': q1,
    'q2': q2,
    'q3': q3
})

print(f"\n데이터 통계:")
print(data.describe())

# ============================================================================
# Test 1: 기본 CFA (절편 없음)
# ============================================================================
print(f"\n{'='*80}")
print(f"[Test 1] 기본 CFA (절편 없음)")
print(f"{'='*80}")

model_spec_1 = """
LV =~ q1 + q2 + q3
"""

print(f"\n모델 스펙:")
print(model_spec_1)

model_1 = Model(model_spec_1)
model_1.fit(data)

params_1 = model_1.inspect()
print(f"\n파라미터:")
print(params_1[['lval', 'op', 'rval', 'Estimate']])

print(f"\n파라미터 타입 (op):")
print(params_1['op'].value_counts())

intercepts_1 = params_1[params_1['op'] == '1']
print(f"\n절편 (op == '1'): {len(intercepts_1)}개")

# ============================================================================
# Test 2: ModelMeans 클래스 사용
# ============================================================================
print(f"\n{'='*80}")
print(f"[Test 2] ModelMeans 클래스 사용")
print(f"{'='*80}")

try:
    from semopy import ModelMeans

    model_spec_2 = """
    LV =~ q1 + q2 + q3
    """

    print(f"\n모델 스펙:")
    print(model_spec_2)
    print(f"(ModelMeans는 자동으로 절편 추정)")

    model_2 = ModelMeans(model_spec_2)
    model_2.fit(data)

    params_2 = model_2.inspect()
    print(f"\n파라미터:")
    print(params_2[['lval', 'op', 'rval', 'Estimate']])

    print(f"\n파라미터 타입 (op):")
    print(params_2['op'].value_counts())

    intercepts_2 = params_2[params_2['op'] == '1']
    print(f"\n절편 (op == '1'): {len(intercepts_2)}개")
    if len(intercepts_2) > 0:
        print(intercepts_2[['lval', 'op', 'rval', 'Estimate']])

except ImportError:
    print(f"\n❌ ModelMeans 클래스를 찾을 수 없습니다.")
    print(f"   semopy 버전이 낮을 수 있습니다.")
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")

# ============================================================================
# Test 3: Model 클래스에 mean_structure=True 옵션
# ============================================================================
print(f"\n{'='*80}")
print(f"[Test 3] Model 클래스에 mean_structure=True 옵션")
print(f"{'='*80}")

try:
    model_spec_3 = """
    LV =~ q1 + q2 + q3
    """

    print(f"\n모델 스펙:")
    print(model_spec_3)
    print(f"(mean_structure=True 옵션 사용)")

    # Model 클래스의 __init__ 시그니처 확인
    import inspect
    sig = inspect.signature(Model.__init__)
    print(f"\nModel.__init__ 파라미터: {list(sig.parameters.keys())}")

    # mean_structure 옵션이 있는지 확인
    if 'mean_structure' in sig.parameters:
        model_3 = Model(model_spec_3, mean_structure=True)
        model_3.fit(data)

        params_3 = model_3.inspect()
        print(f"\n파라미터:")
        print(params_3[['lval', 'op', 'rval', 'Estimate']])

        print(f"\n파라미터 타입 (op):")
        print(params_3['op'].value_counts())

        intercepts_3 = params_3[params_3['op'] == '1']
        print(f"\n절편 (op == '1'): {len(intercepts_3)}개")
        if len(intercepts_3) > 0:
            print(intercepts_3[['lval', 'op', 'rval', 'Estimate']])
    else:
        print(f"\n❌ Model 클래스에 mean_structure 옵션이 없습니다.")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")

# ============================================================================
# 결론
# ============================================================================
print(f"\n{'='*80}")
print(f"[결론]")
print(f"{'='*80}")

print(f"""
semopy 절편 추정 방법:

1. 기본 Model 클래스 (절편 없음):
   - 모델 스펙: LV =~ q1 + q2 + q3
   - 절편 추정 안함
   - 잠재변수 평균 E[LV] = 0으로 고정

2. ModelMeans 클래스:
   - from semopy import ModelMeans
   - 자동으로 절편 추정
   - 모델 스펙에 명시 불필요

3. Model 클래스 + mean_structure 옵션:
   - Model(spec, mean_structure=True)
   - 절편 자동 추정

현재 CFA 추정 코드는 방법 1을 사용하고 있습니다.
절편을 추정하려면 방법 2 또는 3을 사용해야 합니다.

✅ 권장: ModelMeans 클래스 사용
""")

