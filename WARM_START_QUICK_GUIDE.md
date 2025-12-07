# Warm Start Quick Guide

## What is Warm Start?

**Warm Start** uses parameter estimates from Sequential Estimation as initial values for Simultaneous Estimation, preventing the optimizer from driving structural (`gamma`) and link (`theta`) parameters to zero.

## How to Use

### Option 1: Automatic (Recommended)
The test script automatically loads sequential results if available:

```bash
python scripts/test_gpu_batch_iclv.py
```

**Requirements**:
- Sequential result CSV must exist at:
  ```
  results/final/sequential/2path/stage2/st2_2path_NK_PI_PP_results.csv
  ```

### Option 2: Manual Configuration
Specify a custom CSV file:

```python
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_gpu_batch_estimator import SimultaneousGPUBatchEstimator

estimator = SimultaneousGPUBatchEstimator(
    config,
    use_gpu=True,
    structural_weight=1000.0  # Gradient scale balancing
)

result = estimator.estimate(
    data=data,
    measurement_model=measurement_model,
    structural_model=structural_model,
    choice_model=choice_model,
    log_file='log.txt',
    initial_params=None,  # Can be None if using warm start
    sequential_result_csv='path/to/your/sequential_results.csv'  # ✅ Warm Start
)
```

## Configuration Parameters

### Structural Weight
Controls gradient scale balancing between structural and choice models:

```python
# In scripts/test_gpu_batch_iclv.py, line 296
STRUCTURAL_WEIGHT = 1000.0  # Default: 1000.0
```

**Recommended Values**:
- `1000.0`: Default (balances gradient norms to ~100:1 ratio)
- `5000.0`: If structural parameters still converge to zero
- `500.0`: If structural model dominates (rare)

**How to Change**:
1. Open `scripts/test_gpu_batch_iclv.py`
2. Find line 296: `STRUCTURAL_WEIGHT = 1000.0`
3. Change to desired value
4. Save and run

## Expected Log Output

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
  [선택모델] theta_sugar_purchase_intention: 0.250000
  [선택모델] theta_sugar_free_purchase_intention: 0.280000
  [선택모델] beta_health_label: 0.500000
  [선택모델] beta_price: -0.150000
  [선택모델] asc_sugar: 1.200000
  [선택모델] asc_sugar_free: 1.800000

📊 로드 요약:
  - 구조모델 (gamma): 2개
  - 선택모델 (theta): 6개
  - 선택모델 (기타): 4개
================================================================================
```

### Cold Start Fallback
```
⚠️ 순차추정 결과 로드 실패: /path/to/file.csv
Cold Start로 전환합니다.
```

## Troubleshooting

### Problem: "Zero Convergence" Still Occurs
**Symptoms**:
- `gamma` parameters → 0.001 or smaller
- `theta` parameters → 0.001 or smaller

**Solutions**:
1. **Increase Structural Weight**:
   ```python
   STRUCTURAL_WEIGHT = 5000.0  # Try higher value
   ```

2. **Check Sequential Results**:
   - Verify CSV file exists and contains valid estimates
   - Check that gamma and theta values are non-zero in CSV

3. **Verify Warm Start is Active**:
   - Look for "🔥 Warm Start" message in log
   - Check that parameters are loaded (not using 0.1 defaults)

### Problem: Warm Start Not Loading
**Symptoms**:
- No "🔥 Warm Start" message in log
- Parameters initialized to 0.1

**Solutions**:
1. **Check CSV Path**:
   ```python
   # Verify file exists
   import os
   csv_path = 'results/final/sequential/2path/stage2/st2_2path_NK_PI_PP_results.csv'
   print(f"File exists: {os.path.exists(csv_path)}")
   ```

2. **Check CSV Format**:
   - Must have 'section' column with 'Parameters' rows
   - Must have 'parameter' and 'estimate' columns
   - Parameter names must match (e.g., 'gamma_health_concern_to_perceived_benefit')

3. **Manual Override**:
   ```python
   # Force specific CSV file
   sequential_csv = '/full/path/to/your/results.csv'
   ```

### Problem: Gradient Imbalance Persists
**Symptoms**:
- Structural gradient norm: ~0.0002
- Choice gradient norm: ~169.0
- Ratio: ~800,000:1

**Solutions**:
1. **Increase Structural Weight**:
   ```python
   STRUCTURAL_WEIGHT = 10000.0  # More aggressive scaling
   ```

2. **Check Forward-Backward Consistency**:
   - Verify same weight is used in likelihood and gradient
   - Check log for "구조모델 (가중치 X.X×)" message

## Verification Checklist

After running with warm start:

- [ ] Log shows "🔥 Warm Start" message
- [ ] Parameters loaded from CSV (not 0.1 defaults)
- [ ] Structural gradient norm > 0.01 (not ~0.0002)
- [ ] Gradient ratio < 1000:1 (not ~800,000:1)
- [ ] `gamma` parameters > 0.1 in final results
- [ ] `theta` parameters > 0.1 in final results
- [ ] Log likelihood improves over iterations
- [ ] Optimizer converges successfully

## Quick Reference

| Feature | Default | Recommended | Notes |
|---------|---------|-------------|-------|
| Structural Weight | 1000.0 | 1000.0 - 5000.0 | Balances gradients |
| Sequential CSV | Auto-detect | Specify path | For warm start |
| Initial Params | Required | Optional with CSV | Can be None if CSV provided |
| Max Iterations | 50 | 50 - 100 | Adjust if needed |

## Contact

For issues or questions, check:
1. `WARM_START_IMPLEMENTATION_SUMMARY.md` - Detailed technical documentation
2. Log files in `results/final/simultaneous/logs/`
3. Parameter values in sequential CSV file

