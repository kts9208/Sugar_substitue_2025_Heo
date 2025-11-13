"""
최근 실행의 마지막 iteration 파라미터 값 추출
"""

import pandas as pd
import json

# CSV 파일 로드
df = pd.read_csv('results/gpu_batch_iclv_estimation_log_params_grads.csv')

print(f"총 iteration 수: {len(df)}")
print(f"마지막 iteration: {df['iteration'].iloc[-1]}")

# 마지막 행 추출
last_row = df.iloc[-1]

# 파라미터 값 추출
params = {}
for col in df.columns:
    if col.endswith('_value'):
        param_name = col.replace('_value', '')
        param_value = last_row[col]
        params[param_name] = float(param_value)

# 파라미터 그룹별로 정리
print("\n" + "="*80)
print("마지막 Iteration 파라미터 값")
print("="*80)

# 1. 측정모델 - zeta
print("\n[측정모델 - Zeta (요인적재량)]")
zeta_params = {k: v for k, v in params.items() if k.startswith('zeta_')}
for name, value in sorted(zeta_params.items()):
    print(f"  {name}: {value:.6f}")

# 2. 측정모델 - sigma_sq
print("\n[측정모델 - Sigma² (오차분산)]")
sigma_sq_params = {k: v for k, v in params.items() if k.startswith('sigma_sq_')}
for name, value in sorted(sigma_sq_params.items()):
    print(f"  {name}: {value:.6f}")

# 3. 구조모델 - gamma
print("\n[구조모델 - Gamma (경로계수)]")
gamma_params = {k: v for k, v in params.items() if k.startswith('gamma_')}
for name, value in sorted(gamma_params.items()):
    print(f"  {name}: {value:.6f}")

# 4. 선택모델
print("\n[선택모델]")
choice_params = {k: v for k, v in params.items() if k.startswith('beta_') or k.startswith('lambda_')}
for name, value in sorted(choice_params.items()):
    print(f"  {name}: {value:.6f}")

# JSON 파일로 저장
output_file = 'results/last_iteration_params.json'
with open(output_file, 'w') as f:
    json.dump(params, f, indent=2)

print(f"\n파라미터 값이 {output_file}에 저장되었습니다.")

# 초기값 설정 코드 생성
print("\n" + "="*80)
print("초기값 설정 코드 (test_gpu_batch_iclv.py에 사용)")
print("="*80)

print("\n# 측정모델 초기값 (잠재변수별)")
print("measurement_initial_values = {")

# 잠재변수별로 그룹화
lv_groups = {}
for name, value in params.items():
    if name.startswith('zeta_') or name.startswith('sigma_sq_'):
        # 잠재변수 이름 추출
        parts = name.split('_')
        if name.startswith('zeta_'):
            lv_name = '_'.join(parts[1:-1])  # zeta_health_concern_q7 -> health_concern
        else:  # sigma_sq_
            lv_name = '_'.join(parts[2:-1])  # sigma_sq_health_concern_q6 -> health_concern
        
        if lv_name not in lv_groups:
            lv_groups[lv_name] = {'zeta': [], 'sigma_sq': []}
        
        if name.startswith('zeta_'):
            lv_groups[lv_name]['zeta'].append(value)
        else:
            lv_groups[lv_name]['sigma_sq'].append(value)

for lv_name, values in sorted(lv_groups.items()):
    print(f"    '{lv_name}': {{")
    if values['zeta']:
        zeta_str = ', '.join([f'{v:.4f}' for v in values['zeta']])
        print(f"        'zeta': [{zeta_str}],")
    if values['sigma_sq']:
        sigma_sq_str = ', '.join([f'{v:.4f}' for v in values['sigma_sq']])
        print(f"        'sigma_sq': [{sigma_sq_str}]")
    print(f"    }},")

print("}")

print("\n# 구조모델 초기값")
print("structural_initial_values = {")
for name, value in sorted(gamma_params.items()):
    print(f"    '{name}': {value:.6f},")
print("}")

print("\n# 선택모델 초기값")
print("choice_initial_values = {")
for name, value in sorted(choice_params.items()):
    print(f"    '{name}': {value:.6f},")
print("}")

