# 벡터화 차원 검증 가이드

## 📐 차원 불일치 문제

벡터화에서 가장 흔한 버그는 **배열 차원 불일치**입니다.

### 문제 예시

```python
# ❌ 잘못된 예: 차원 불일치
lv_values = np.array([[1.0], [2.0], [3.0]])  # Shape: (3, 1) - 2D 열벡터
attr_values = np.array([10, 20, 30])          # Shape: (3,) - 1D 배열

# 의도: Element-wise 곱셈 (3,)
# 실제: Broadcasting으로 (3, 3) 행렬 생성!
result = lv_values * attr_values
# [[10, 20, 30],
#  [20, 40, 60],
#  [30, 60, 90]]
```

### 올바른 예시

```python
# ✅ 올바른 예: 차원 일치
lv_values = np.array([1.0, 2.0, 3.0])  # Shape: (3,) - 1D 배열
attr_values = np.array([10, 20, 30])    # Shape: (3,) - 1D 배열

# Element-wise 곱셈: (3,) * (3,) = (3,)
result = lv_values * attr_values
# [10, 40, 90]
```

---

## ✅ 현재 구현의 차원 일관성

### 1. 기본 배열 차원

| 배열 | 차원 | Shape 예시 | 설명 |
|------|------|-----------|------|
| `V` | 1D | `(24000,)` | 효용 벡터 |
| `X` | 2D | `(24000, 3)` | 속성 행렬 |
| `lv_arrays[lv_name]` | 1D | `(24000,)` | 잠재변수 (확장됨) |
| `attr_values` | 1D | `(24000,)` | 속성 벡터 (X의 열 슬라이스) |
| `is_sugar` | 1D | `(24000,)` | Boolean 마스크 |

### 2. 핵심 연산의 차원

#### (1) 기본 효용: `ASC + X @ beta`

```python
X[is_sugar]  # Shape: (n_sugar, 3) - 2D
beta         # Shape: (3,) - 1D
result = X[is_sugar] @ beta  # Shape: (n_sugar,) - 1D ✅
```

#### (2) LV 주효과: `theta * lv_values`

```python
theta                        # Scalar
lv_arrays[lv_name][is_sugar] # Shape: (n_sugar,) - 1D
result = theta * lv_arrays[lv_name][is_sugar]  # Shape: (n_sugar,) - 1D ✅
```

#### (3) LV-Attribute 상호작용: `gamma * lv_values * attr_values`

```python
gamma       # Scalar
lv_values   # Shape: (N,) - 1D
attr_values # Shape: (N,) - 1D

# Element-wise 곱셈
interaction_term = gamma * lv_values * attr_values  # Shape: (N,) - 1D ✅
```

---

## 🔍 차원 검증 방법

### 1. 명시적 Assertion 추가

```python
# ✅ 차원 검증
assert lv_values.ndim == 1, f"lv_values should be 1D, got {lv_values.ndim}D"
assert attr_values.ndim == 1, f"attr_values should be 1D, got {attr_values.ndim}D"
assert len(lv_values) == len(attr_values), f"Length mismatch"
```

### 2. Shape 로깅

```python
# 디버깅용 로깅
print(f"lv_values shape: {lv_values.shape}")
print(f"attr_values shape: {attr_values.shape}")
print(f"interaction_term shape: {interaction_term.shape}")
```

### 3. 자동 테스트

```bash
# 차원 검증 테스트 실행
python scripts/test_dimension_validation.py
```

---

## 🛠️ 차원 불일치 해결 방법

### 문제 1: 2D 열벡터 → 1D 배열 변환

```python
# ❌ 2D 열벡터
lv_values = np.array([[1.0], [2.0], [3.0]])  # Shape: (3, 1)

# ✅ 1D 배열로 변환
lv_values = lv_values.flatten()  # Shape: (3,)
# 또는
lv_values = lv_values.squeeze()  # Shape: (3,)
# 또는
lv_values = lv_values[:, 0]      # Shape: (3,)
```

### 문제 2: DataFrame 열 → 1D 배열

```python
# ❌ DataFrame 열 (Series)
attr_values = data['price']  # pandas Series

# ✅ NumPy 1D 배열로 변환
attr_values = data['price'].values  # Shape: (N,)
```

### 문제 3: 2D 행렬에서 열 추출

```python
# ✅ 올바른 방법: 1D 배열 반환
attr_values = X[:, attr_idx]  # Shape: (N,)

# ❌ 잘못된 방법: 2D 열벡터 반환
attr_values = X[:, [attr_idx]]  # Shape: (N, 1)
```

---

## 📊 테스트 결과

### 차원 검증 테스트

```
================================================================================
차원 검증 테스트
================================================================================

[3] 효용 계산 및 차원 검증...
  ✅ 차원 검증 통과!

[4] 결과 확인...
  - V shape: (2400,)
  - V ndim: 1
  - V dtype: float64
  - V 범위: [-9.7735, 0.0000]
  - NaN 개수: 0
  - Inf 개수: 0

✅ 모든 차원 검증 통과!
```

---

## 🎯 체크리스트

벡터화 구현 시 다음을 확인하세요:

- [ ] 모든 배열이 **동일한 차원** (1D 또는 2D)을 유지하는가?
- [ ] `X[:, i]`로 열을 추출할 때 **1D 배열**이 반환되는가?
- [ ] Element-wise 연산 시 **Broadcasting**이 의도대로 작동하는가?
- [ ] Boolean 마스킹 후 배열 차원이 **유지**되는가?
- [ ] 최종 결과 `V`가 **1D 배열**인가?
- [ ] NaN/Inf가 없는가?
- [ ] 테스트가 통과하는가?

---

## 📚 참고 자료

- [NumPy Broadcasting Rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy Array Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Pandas to NumPy Conversion](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.values.html)

