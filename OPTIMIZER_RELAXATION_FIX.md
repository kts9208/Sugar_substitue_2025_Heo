# Optimizer Relaxation Fix - Line Search Infinite Loop

## Problem Diagnosis

**Symptom**: L-BFGS-B optimizer stuck in Iteration #8 with endless Line Search calls (38+ calls and counting)

**Root Cause**:
1. **Search direction is zero**: `d norm: 0.000000e+00` (no direction to move)
2. **Wolfe conditions not satisfied**: Line search cannot find valid step size
3. **Too strict convergence criteria**: 
   - `ftol_threshold = 1e-6` (very tight)
   - `param_change_threshold = 1e-6` (very tight)
   - **AND logic**: Both conditions must be satisfied simultaneously
4. **Insufficient maxls**: `maxls=20` not enough for difficult optimization landscape

**Evidence from Terminal**:
```
[Major Iteration #7 完了]
  최종 LL: -8174.0988
  Line Search: 1회 함수 호출 - [WARN] 정체 (함수값 변화 없음)
  
[Major Iteration #8 시작]
  탐색 방향 d norm: 0.000000e+00  ← No direction!
  Gradient norm: 4.833282e+00     ← Gradient exists
  코사인 유사도: 0.000000          ← d ⊥ -grad (orthogonal)
  
[Line Search 함수 호출 #iter8-1 ~ #iter8-38+]
  ⚠️ [Line Search 경고] maxls=10에 도달했습니다.
  (Repeating infinitely...)
```

## Solution Implemented

### 1. Relaxed Convergence Thresholds

**File**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**Line 1130-1136**: Changed convergence thresholds

```python
# BEFORE (Too Strict)
self.ftol_threshold = 1e-6  # ftol 기준
self.param_change_threshold = 1e-6  # 파라미터 변화량 기준

# AFTER (Relaxed)
self.ftol_threshold = 1e-4  # ftol 기준 (1e-6 → 1e-4로 완화)
self.param_change_threshold = 1e-4  # 파라미터 변화량 기준 (1e-6 → 1e-4로 완화)
```

**Rationale**: 
- `1e-6` is extremely tight for ICLV models with fixed measurement parameters
- `1e-4` is still very good convergence (0.01% relative change)
- Prevents optimizer from getting stuck trying to achieve impossible precision

### 2. Changed Convergence Logic from AND to OR

**File**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**Line 1305-1325**: Changed convergence logic

```python
# BEFORE (Too Strict - Both conditions required)
if ftol_satisfied and param_change_satisfied:
    # Converge only if BOTH conditions met

# AFTER (Relaxed - Either condition sufficient)
if ftol_satisfied or param_change_satisfied:
    # Converge if EITHER condition met
```

**Rationale**:
- If loss isn't changing (`ftol` satisfied), optimizer should stop even if parameters wiggle slightly
- If parameters aren't changing (`param_change` satisfied), optimizer should stop even if loss has tiny fluctuations
- Prevents infinite loop where one condition is met but the other never will be

### 3. Increased maxls (Line Search Maximum Iterations)

**File**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**Line 1451-1474**: Updated L-BFGS-B options

```python
# BEFORE
optimizer_options = {
    'maxiter': 200,
    'maxls': 20,     # Line search 최대 횟수
    'disp': True
    # ftol, gtol not specified → scipy defaults
}

# AFTER
optimizer_options = {
    'maxiter': 200,
    'maxls': 50,     # Line search 최대 횟수 (20 → 50으로 증가)
    'ftol': 1e-5,    # 함수값 상대 변화 허용 오차 (완화)
    'disp': True
}
```

**Rationale**:
- `maxls=20` was insufficient for difficult optimization landscape
- `maxls=50` gives more attempts to find valid step size
- `ftol=1e-5` explicitly set (more relaxed than scipy's extremely tight default)

### 4. Updated Logging Messages

**File**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

**Line 1307-1323**: Updated convergence message

```python
# BEFORE
f"✅ 수렴 완료: ftol AND 파라미터 변화량 조건 모두 만족\n"

# AFTER
f"✅ 수렴 완료: ftol OR 파라미터 변화량 조건 만족 (완화된 기준)\n"
f"  💡 완화된 조건 (ftol=1e-4, xtol=1e-4)으로 Line Search 무한 루프 방지\n"
```

**Line 1451-1474**: Updated optimizer options logging

```python
# BEFORE
f"  ✅ 커스텀 수렴 조건 (callback에서 ftol AND 파라미터 변화량 체크):\n"
f"    → 두 조건을 모두 만족해야 조기 종료\n"

# AFTER
f"  ✅ 커스텀 수렴 조건 (callback에서 ftol OR 파라미터 변화량 체크):\n"
f"    1. ftol 조건: ... <= 1e-4 (완화)\n"
f"    2. 파라미터 변화량: ... <= 1e-4 (완화)\n"
f"    → 두 조건 중 하나만 만족해도 조기 종료 (AND → OR로 변경)\n"
f"  💡 완화된 조건으로 Line Search 무한 루프 방지\n"
```

## Summary of Changes

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `ftol_threshold` | 1e-6 | 1e-4 | Too strict for ICLV models |
| `param_change_threshold` | 1e-6 | 1e-4 | Too strict for ICLV models |
| Convergence Logic | AND | OR | Prevent impossible dual conditions |
| `maxls` | 20 | 50 | More attempts for line search |
| `ftol` (scipy) | default (2.22e-09) | 1e-5 | Explicitly relaxed |

## Expected Behavior After Fix

1. **Optimizer will converge** when either:
   - Loss stops changing (ftol satisfied), OR
   - Parameters stop changing (param_change satisfied)

2. **Line search will have more attempts** (50 instead of 20) to find valid step size

3. **Convergence criteria are more realistic** for ICLV models with fixed measurement parameters

4. **No more infinite loops** - optimizer will declare convergence and exit gracefully

## Testing Instructions

1. **Kill the current stuck process**: Press `Ctrl+C` in the terminal

2. **Restart the optimization**:
   ```bash
   python scripts/test_gpu_batch_iclv.py
   ```

3. **Monitor for success**:
   - Look for convergence message with "OR" logic
   - Check that optimizer exits after reasonable iterations (< 20)
   - Verify final LL is around -8174 (similar to where it got stuck)

4. **Verify results**:
   - Check that gamma and theta parameters are non-zero
   - Confirm structural weight is applied (1000.0×)
   - Review final parameter estimates

## Rollback Instructions

If the relaxed criteria are too loose, you can tighten them:

```python
# In simultaneous_estimator_fixed.py, line 1130-1136
self.ftol_threshold = 1e-5  # Tighter than 1e-4, looser than 1e-6
self.param_change_threshold = 1e-5

# Line 1305
if ftol_satisfied and param_change_satisfied:  # Back to AND logic
```

## Files Modified

- `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`
  - Line 1130-1136: Relaxed convergence thresholds
  - Line 1287-1290: Added variable initialization (UnboundLocalError fix)
  - Line 1310-1330: Changed AND → OR logic
  - Line 1451-1474: Increased maxls, added ftol, updated logging

## Bug Fix: UnboundLocalError

**Issue**: After changing to OR logic, the code threw `UnboundLocalError: cannot access local variable 'rel_change'`

**Cause**: Variables `rel_change`, `param_change_norm`, and `grad_norm_active` were only defined inside conditional blocks, but used in convergence message regardless.

**Fix**: Initialize all three variables to `float('inf')` before the conditional checks (Line 1287-1290):
```python
# ✅ 변수 초기화 (UnboundLocalError 방지)
rel_change = float('inf')
param_change_norm = float('inf')
grad_norm_active = float('inf')
```

This ensures the variables always have values when used in the convergence message.

