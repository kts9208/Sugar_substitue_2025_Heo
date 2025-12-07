# Quick Fix Guide - Stop Infinite Loop

## 🚨 IMMEDIATE ACTION REQUIRED

Your optimizer is stuck in an **infinite line search loop**. Follow these steps:

### Step 1: Stop the Current Process

**Press `Ctrl + C` in the terminal** to kill the stuck optimization.

### Step 2: Verify the Fix is Applied

The following changes have been made to prevent the infinite loop:

✅ **Relaxed convergence thresholds**: `1e-6` → `1e-4`  
✅ **Changed logic**: `AND` → `OR` (either condition triggers convergence)  
✅ **Increased maxls**: `20` → `50` (more line search attempts)  
✅ **Added explicit ftol**: `1e-5` (relaxed scipy default)

**File Modified**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_estimator_fixed.py`

### Step 3: Restart the Optimization

```bash
python scripts/test_gpu_batch_iclv.py
```

### Step 4: Monitor the Output

**Look for these signs of success:**

1. **Convergence message** should show:
   ```
   ✅ 수렴 완료: ftol OR 파라미터 변화량 조건 만족 (완화된 기준)
   ```

2. **Optimizer options** should show:
   ```
   L-BFGS-B 옵션 (완화된 수렴 조건):
     - maxls: 50 (line search 최대 횟수, 20→50으로 증가)
     - ftol: 1.0e-05 (완화)
   ```

3. **No infinite loop**: Should converge within 10-20 iterations

### Step 5: Verify Results

After convergence, check:

- [ ] Final LL around `-8174` (similar to stuck value)
- [ ] `gamma` parameters are non-zero (> 0.001)
- [ ] `theta` parameters are non-zero (> 0.001)
- [ ] Structural weight applied: `1000.0×`
- [ ] Convergence message shows "OR" logic

## What Changed?

### Before (Stuck in Infinite Loop)
```python
# Too strict thresholds
ftol_threshold = 1e-6
param_change_threshold = 1e-6

# Both conditions required (impossible to satisfy)
if ftol_satisfied and param_change_satisfied:
    converge()

# Insufficient line search attempts
maxls = 20
```

### After (Fixed)
```python
# Relaxed thresholds
ftol_threshold = 1e-4  # 100x more relaxed
param_change_threshold = 1e-4  # 100x more relaxed

# Either condition sufficient (realistic)
if ftol_satisfied or param_change_satisfied:
    converge()

# More line search attempts
maxls = 50  # 2.5x more attempts
ftol = 1e-5  # Explicitly relaxed
```

## Why This Works

1. **Relaxed Thresholds**: `1e-4` is still excellent convergence (0.01% change) but achievable
2. **OR Logic**: If loss stops changing, stop - even if parameters wiggle slightly
3. **More maxls**: Gives optimizer more chances to find valid step size
4. **Explicit ftol**: Overrides scipy's extremely tight default

## Troubleshooting

### If Still Stuck
1. Check terminal output for "maxls=50" (not 20)
2. Verify convergence message shows "OR" (not "AND")
3. If still infinite loop, increase maxls to 100:
   ```python
   # Line 1453 in simultaneous_estimator_fixed.py
   'maxls': 100,  # Even more attempts
   ```

### If Converges Too Early
1. Check final LL is reasonable (around -8174)
2. If LL is much worse, tighten thresholds:
   ```python
   # Line 1132-1134
   self.ftol_threshold = 1e-5  # Tighter
   self.param_change_threshold = 1e-5
   ```

### If Parameters Still Zero
1. This is a different problem (warm start issue)
2. Check that warm start is loading sequential results
3. Verify `structural_weight = 1000.0` is applied

## Expected Runtime

- **Before**: Infinite (stuck forever)
- **After**: 5-15 minutes (normal convergence)

## Success Criteria

✅ Optimizer exits gracefully (no Ctrl+C needed)  
✅ Convergence message appears  
✅ Final LL around -8174  
✅ Parameters are non-zero  
✅ No "maxls=10 도달" warnings repeating infinitely

## Contact

If issues persist after this fix:
1. Check `OPTIMIZER_RELAXATION_FIX.md` for detailed explanation
2. Review terminal output for error messages
3. Verify all changes were applied correctly

---

**TL;DR**: Kill the process (`Ctrl+C`), restart (`python scripts/test_gpu_batch_iclv.py`), and it should converge normally now.

