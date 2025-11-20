"""
CFA 결과 파라미터 상세 확인
"""
import pickle
import numpy as np
from pathlib import Path

print("="*80)
print("CFA 파라미터 상세 확인")
print("="*80)

# CFA 결과 로드
cfa_path = Path('results/sequential_stage_wise/cfa_results.pkl')

with open(cfa_path, 'rb') as f:
    cfa_results = pickle.load(f)

print(f"\nCFA 결과 키: {list(cfa_results.keys())}")

# ============================================================================
# 1. loadings 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[1] 요인적재량 (loadings)")
print(f"{'='*80}")

if 'loadings' in cfa_results:
    loadings = cfa_results['loadings']
    print(f"\n타입: {type(loadings)}")
    
    if isinstance(loadings, dict):
        print(f"키: {list(loadings.keys())}")
        
        for lv_name, lv_loadings in loadings.items():
            print(f"\n{lv_name}:")
            print(f"  타입: {type(lv_loadings)}")
            
            if isinstance(lv_loadings, dict):
                print(f"  키: {list(lv_loadings.keys())}")
                for key, val in lv_loadings.items():
                    print(f"    {key}: {val}")
            elif isinstance(lv_loadings, np.ndarray):
                print(f"  값: {lv_loadings}")
                print(f"  평균: {np.mean(lv_loadings):.4f}")
                print(f"  표준편차: {np.std(lv_loadings):.4f}")
                print(f"  범위: [{np.min(lv_loadings):.4f}, {np.max(lv_loadings):.4f}]")
            else:
                print(f"  값: {lv_loadings}")

# ============================================================================
# 2. measurement_errors 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[2] 측정오차 (measurement_errors)")
print(f"{'='*80}")

if 'measurement_errors' in cfa_results:
    errors = cfa_results['measurement_errors']
    print(f"\n타입: {type(errors)}")
    
    if isinstance(errors, dict):
        print(f"키: {list(errors.keys())}")
        
        for lv_name, lv_errors in errors.items():
            print(f"\n{lv_name}:")
            print(f"  타입: {type(lv_errors)}")
            
            if isinstance(lv_errors, dict):
                print(f"  키: {list(lv_errors.keys())}")
                for key, val in lv_errors.items():
                    print(f"    {key}: {val}")
            elif isinstance(lv_errors, np.ndarray):
                print(f"  값: {lv_errors}")
                print(f"  평균: {np.mean(lv_errors):.4f}")
                print(f"  표준편차: {np.std(lv_errors):.4f}")
                print(f"  범위: [{np.min(lv_errors):.4f}, {np.max(lv_errors):.4f}]")
            else:
                print(f"  값: {lv_errors}")

# ============================================================================
# 3. 실제 사용되는 파라미터 확인 (measurement_params)
# ============================================================================
print(f"\n{'='*80}")
print(f"[3] 실제 사용되는 파라미터 확인")
print(f"{'='*80}")

# 가장 최근 stage1 결과 찾기
stage1_dir = Path('results/sequential_stage_wise')
stage1_files = list(stage1_dir.glob('stage1_*_results.pkl'))

if stage1_files:
    latest_file = max(stage1_files, key=lambda p: p.stat().st_mtime)
    print(f"\nstage1 파일: {latest_file.name}")
    
    with open(latest_file, 'rb') as f:
        stage1_results = pickle.load(f)
    
    print(f"키: {list(stage1_results.keys())}")
    
    if 'measurement_params' in stage1_results:
        measurement_params = stage1_results['measurement_params']
        print(f"\nmeasurement_params 타입: {type(measurement_params)}")
        print(f"키: {list(measurement_params.keys())}")
        
        for lv_name, params in measurement_params.items():
            print(f"\n{lv_name}:")
            print(f"  타입: {type(params)}")
            print(f"  키: {list(params.keys())}")
            
            if 'zeta' in params:
                zeta = params['zeta']
                print(f"\n  ζ (요인적재량):")
                print(f"    값: {zeta}")
                print(f"    평균: {np.mean(zeta):.4f}")
                print(f"    표준편차: {np.std(zeta):.4f}")
                print(f"    범위: [{np.min(zeta):.4f}, {np.max(zeta):.4f}]")
                
                # 표준화 여부 판단
                if np.allclose(zeta, 1.0, atol=0.1):
                    print(f"    ✅ 모두 1에 가까움 (표준화됨)")
                elif np.abs(np.mean(zeta)) < 0.01 and np.abs(np.std(zeta) - 1.0) < 0.1:
                    print(f"    ✅ 표준화됨 (평균 ≈ 0, 표준편차 ≈ 1)")
                else:
                    print(f"    ❌ 표준화 안됨 (원척도)")
            
            if 'sigma_sq' in params:
                sigma_sq = params['sigma_sq']
                print(f"\n  σ² (측정오차 분산):")
                print(f"    값: {sigma_sq}")
                print(f"    평균: {np.mean(sigma_sq):.4f}")
                print(f"    표준편차: {np.std(sigma_sq):.4f}")
                print(f"    범위: [{np.min(sigma_sq):.4f}, {np.max(sigma_sq):.4f}]")

# ============================================================================
# 4. 동시추정에서 사용되는 파라미터 확인
# ============================================================================
print(f"\n{'='*80}")
print(f"[4] 동시추정에서 사용되는 파라미터")
print(f"{'='*80}")

# 가장 최근 simultaneous 결과 찾기
simul_files = list(Path('results').glob('**/gpu_batch_iclv_*.pkl'))

if simul_files:
    latest_simul = max(simul_files, key=lambda p: p.stat().st_mtime)
    print(f"\n동시추정 파일: {latest_simul}")
    
    with open(latest_simul, 'rb') as f:
        simul_results = pickle.load(f)
    
    print(f"키: {list(simul_results.keys())}")
    
    if 'measurement_params' in simul_results:
        measurement_params = simul_results['measurement_params']
        print(f"\nmeasurement_params (동시추정):")
        
        for lv_name, params in measurement_params.items():
            print(f"\n{lv_name}:")
            
            if 'zeta' in params:
                zeta = params['zeta']
                print(f"  ζ: {zeta}")
                print(f"  평균: {np.mean(zeta):.4f}, 범위: [{np.min(zeta):.4f}, {np.max(zeta):.4f}]")
            
            if 'sigma_sq' in params:
                sigma_sq = params['sigma_sq']
                print(f"  σ²: {sigma_sq}")
                print(f"  평균: {np.mean(sigma_sq):.4f}, 범위: [{np.min(sigma_sq):.4f}, {np.max(sigma_sq):.4f}]")

print(f"\n{'='*80}")
print(f"분석 완료")
print(f"{'='*80}")

