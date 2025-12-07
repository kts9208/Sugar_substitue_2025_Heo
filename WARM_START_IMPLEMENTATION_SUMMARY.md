# Warm Start Implementation Summary

## Overview
Implemented **Warm Start** from Sequential Estimation results to prevent the "Zero Convergence" problem in Simultaneous Estimation, where the optimizer drives structural (`gamma`) and measurement-choice link (`theta`) parameters to zero.

## Problem Diagnosis
- **Issue**: Optimizer finds a local optimum where it ignores Latent Variables (LVs) entirely, relying only on observed attributes
- **Symptom**: `gamma` (Structural) and `theta` (Link) parameters converge to zero
- **Root Cause**: Poor initial values allow the optimizer to kill the LV signal early on

## Solution Implemented

### 1. Warm Start from Sequential Estimates (Priority)
**File**: `src/analysis/hybrid_choice_model/iclv_models/simultaneous_gpu_batch_estimator.py`

#### Added Method: `_load_sequential_estimates()`
```python
def _load_sequential_estimates(self, file_path: str) -> Dict:
    """
    Load parameter estimates from Sequential Estimation result CSV
    
    Args:
        file_path: Path to sequential estimation result CSV
        
    Returns:
        Parameter dictionary {param_name: estimate_value}
    """
```

**Features**:
- Reads CSV file with 'section' == 'Parameters'
- Extracts 'parameter' and 'estimate' columns
- Returns dictionary of parameter names and values
- Handles missing files gracefully (returns empty dict)

#### Updated Method: `estimate()`
**New Parameter**: `sequential_result_csv: Optional[str] = None`

**Logic**:
1. If `sequential_result_csv` is provided:
   - Load parameters from CSV using `_load_sequential_estimates()`
   - Extract `gamma_*` parameters → `initial_params['structural']`
   - Extract `theta_*`, `beta_*`, `asc_*` parameters → `initial_params['choice']`
   - Log loaded parameters with counts
2. If CSV not available or loading fails:
   - Fall back to provided `initial_params`
   - If both are None, raise ValueError

**Priority**: Sequential CSV > initial_params > Error

### 2. Updated Test Script
**File**: `scripts/test_gpu_batch_iclv.py`

#### Added Structural Weight Configuration
```python
# Line 296
STRUCTURAL_WEIGHT = 1000.0  # Gradient Scale Balancing
```

#### Updated Estimator Initialization
```python
# Line 299
estimator = SimultaneousGPUBatchEstimator(
    config,
    use_gpu=True,
    memory_monitor_cpu_threshold_mb=CPU_MEMORY_THRESHOLD_MB,
    memory_monitor_gpu_threshold_mb=GPU_MEMORY_THRESHOLD_MB,
    structural_weight=STRUCTURAL_WEIGHT  # ✅ Added
)
```

#### Updated estimate() Call
```python
# Line 610
sequential_csv = str(stage2_csv_path) if stage2_csv_path.exists() else None

result = estimator.estimate(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_model=choice_model,
    log_file=str(log_file),
    initial_params=initial_params,
    sequential_result_csv=sequential_csv  # ✅ Warm Start enabled
)
```

### 3. Fixed Weight Logging Display
**File**: `src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`

**Line 1552**: Already correct - uses actual `structural_weight` parameter
```python
f"    구조모델 (가중치 {structural_weight:.1f}×): {total_ll_structural:.4f} ..."
```

**Note**: The logging was already using the correct variable. No hardcoded "9.0" found.

## Expected Behavior

### Warm Start Activated
```
================================================================================
🔥 Warm Start: 순차추정 결과에서 초기값 로드
================================================================================
✅ 12개 파라미터 로드 완료:
  [구조모델] gamma_health_concern_to_perceived_benefit: 0.450000
  [구조모델] gamma_perceived_benefit_to_purchase_intention: 0.320000
  [선택모델] theta_sugar_nutrition_knowledge: 0.150000
  [선택모델] theta_sugar_free_nutrition_knowledge: 0.180000
  ...

📊 로드 요약:
  - 구조모델 (gamma): 2개
  - 선택모델 (theta): 6개
  - 선택모델 (기타): 4개
================================================================================
```

### Cold Start Fallback
If CSV is not available:
```
⚠️ 순차추정 결과 로드 실패: /path/to/file.csv
Cold Start로 전환합니다.
```

## Testing Instructions

1. **Run with Warm Start**:
   ```bash
   python scripts/test_gpu_batch_iclv.py
   ```
   - Ensure `st2_2path_NK_PI_PP_results.csv` exists in the expected location
   - Check log for "🔥 Warm Start" message
   - Verify gamma and theta parameters are loaded from CSV

2. **Verify Gradient Norms**:
   - Check that structural gradient norm is no longer ~0.0002
   - Ratio should improve from 800,000:1 to ~100:1

3. **Monitor Parameter Convergence**:
   - Verify `gamma` parameters do NOT converge to zero
   - Verify `theta` parameters maintain significant values
   - Check final estimates are reasonable

## Key Benefits

1. **Strong Initial Values**: Forces optimizer to recognize LV importance from Iteration #1
2. **Prevents Zero Convergence**: Avoids local optimum where LVs are ignored
3. **Faster Convergence**: Starts closer to true optimum
4. **Automatic Fallback**: Gracefully handles missing CSV files
5. **Transparent Logging**: Clear indication of warm start status

## Files Modified

1. `src/analysis/hybrid_choice_model/iclv_models/simultaneous_gpu_batch_estimator.py`
   - Added `_load_sequential_estimates()` method
   - Updated `estimate()` signature and logic
   - Added warm start logging

2. `scripts/test_gpu_batch_iclv.py`
   - Added `STRUCTURAL_WEIGHT` configuration
   - Updated estimator initialization
   - Updated `estimate()` call with `sequential_result_csv`

3. `src/analysis/hybrid_choice_model/iclv_models/gpu_gradient_batch.py`
   - Verified logging uses correct `structural_weight` variable (no changes needed)

## Next Steps

1. Run the test script and verify warm start works
2. Monitor gradient norms and parameter convergence
3. If needed, adjust `STRUCTURAL_WEIGHT` (try 5000.0 if 1000.0 is insufficient)
4. Compare results with sequential estimation to validate consistency

