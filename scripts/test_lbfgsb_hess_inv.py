"""
L-BFGS-B의 hess_inv 반환 여부 테스트
"""
import scipy.optimize as opt
import numpy as np

# 간단한 최적화 문제
def objective(x):
    return (x[0]-1)**2 + (x[1]-2)**2

def gradient(x):
    return np.array([2*(x[0]-1), 2*(x[1]-2)])

# L-BFGS-B 실행
result = opt.minimize(
    objective, 
    [0, 0], 
    method='L-BFGS-B', 
    jac=gradient
)

print("="*80)
print("L-BFGS-B 결과 분석")
print("="*80)
print(f"최적화 성공: {result.success}")
print(f"최종 파라미터: {result.x}")
print(f"최종 함수값: {result.fun}")
print()

print("="*80)
print("hess_inv 속성 확인")
print("="*80)
print(f"hess_inv 존재 여부: {hasattr(result, 'hess_inv')}")
print(f"hess_inv 타입: {type(result.hess_inv)}")
print(f"hess_inv 값: {result.hess_inv}")
print()

if hasattr(result.hess_inv, 'shape'):
    print(f"Shape: {result.hess_inv.shape}")

print()
print("="*80)
print("hess_inv 메서드 확인")
print("="*80)
methods = [m for m in dir(result.hess_inv) if not m.startswith('_')]
for m in methods:
    print(f"  - {m}")
print()

print("="*80)
print("hess_inv 사용 테스트 (벡터 곱)")
print("="*80)
v = np.array([1.0, 1.0])
print(f"테스트 벡터 v: {v}")
try:
    result_vec = result.hess_inv @ v
    print(f"hess_inv @ v = {result_vec}")
    print("✅ 벡터 곱 성공")
except Exception as e:
    print(f"❌ 벡터 곱 실패: {e}")
print()

print("="*80)
print("hess_inv를 numpy 배열로 변환 시도")
print("="*80)

# 방법 1: todense() 메서드
print("방법 1: todense() 메서드")
try:
    if hasattr(result.hess_inv, 'todense'):
        hess_inv_dense = result.hess_inv.todense()
        print(f"✅ todense() 성공")
        print(f"   타입: {type(hess_inv_dense)}")
        print(f"   Shape: {hess_inv_dense.shape}")
        print(f"   Array:\n{hess_inv_dense}")
    else:
        print("❌ todense() 메서드 없음")
except Exception as e:
    print(f"❌ todense() 실패: {e}")
print()

# 방법 2: 단위 벡터로 각 열 추출
print("방법 2: 단위 벡터로 각 열 추출")
try:
    n = len(result.x)
    hess_inv_full = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        hess_inv_full[:, i] = result.hess_inv @ e_i

    print(f"✅ 단위 벡터 방법 성공")
    print(f"   Shape: {hess_inv_full.shape}")
    print(f"   Array:\n{hess_inv_full}")
except Exception as e2:
    print(f"❌ 단위 벡터 방법 실패: {e2}")
print()

# 방법 3: sk, yk 직접 접근
print("방법 3: sk, yk 직접 접근 (L-BFGS 메모리)")
try:
    print(f"  sk (파라미터 변화 이력): {result.hess_inv.sk}")
    print(f"  yk (gradient 변화 이력): {result.hess_inv.yk}")
    print(f"  rho: {result.hess_inv.rho}")
    print(f"  n_corrs (저장된 쌍 개수): {result.hess_inv.n_corrs}")
except Exception as e:
    print(f"❌ sk, yk 접근 실패: {e}")
print()

print("="*80)
print("BFGS와 비교")
print("="*80)
result_bfgs = opt.minimize(
    objective, 
    [0, 0], 
    method='BFGS', 
    jac=gradient
)
print(f"BFGS hess_inv 타입: {type(result_bfgs.hess_inv)}")
print(f"BFGS hess_inv:\n{result_bfgs.hess_inv}")

