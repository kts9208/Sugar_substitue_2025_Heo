"""
Extract all parameter values from Iteration 40 and predict convergence values
"""

import re
import numpy as np

log_file = 'results/gpu_batch_iclv_estimation_log.txt'

# Read log file
with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the section with actual parameter values (External scale)
# Look for "Gradient Scaling Comparison" section which has all parameters
pattern = r'Gradient Scaling Comparison:.*?External gradient norm:'
match = re.search(pattern, content, re.DOTALL)

if match:
    section = match.group(0)
    
    # Extract parameter lines
    # Format: ζ_health_concern_q7              -1.117636e+06   -1.117636e+06     1.000000
    param_pattern = r'([ζσ²γβλ_][^\s]+)\s+([-+]?[0-9.e+-]+)\s+([-+]?[0-9.e+-]+)\s+([-+]?[0-9.e+-]+)'
    
    params = {}
    for match in re.finditer(param_pattern, section):
        param_name = match.group(1)
        grad_external = float(match.group(2))
        grad_internal = float(match.group(3))
        scale = float(match.group(4))
        
        # We need the actual parameter VALUE, not gradient
        # This section only has gradients
        pass

# Alternative: Extract from "실제 계산에 사용된 파라미터 값" section
# This has the actual parameter values
pattern2 = r'\[실제 계산에 사용된 파라미터 값.*?\n(.*?)(?=\n\[|$)'
matches = list(re.finditer(pattern2, content, re.DOTALL))

if matches:
    # Get the LAST occurrence (most recent iteration)
    last_match = matches[-1]
    section = last_match.group(1)
    
    # Extract parameter values
    # Format: [ 3] param_3                                 : +1.360747e+00 (internal: +1.360747e+00)
    param_pattern = r'\[\s*(\d+)\]\s+param_(\d+)\s+:\s+([-+]?[0-9.e+-]+)'
    
    param_values = {}
    for match in re.finditer(param_pattern, section):
        param_idx = int(match.group(2))
        param_value = float(match.group(3))
        param_values[param_idx] = param_value
    
    print(f'Found {len(param_values)} parameter values from log')
    print()
    
    # Sort by index
    sorted_params = sorted(param_values.items())
    
    print('Parameter values (Iteration 40):')
    print('=' * 80)
    for idx, value in sorted_params[:20]:  # First 20
        print(f'param_{idx:2d}: {value:12.6f}')
    print('...')
    for idx, value in sorted_params[-10:]:  # Last 10
        print(f'param_{idx:2d}: {value:12.6f}')

# Better approach: Parse the "Iteration 40 - 파라미터 값" section
print()
print('=' * 80)
print('Parsing Iteration 40 parameter values from structured log')
print('=' * 80)
print()

# Find Iteration 40 section
iter40_pattern = r'Iteration 40 - 파라미터 값.*?(?=Iteration \d+|LL =|$)'
iter40_match = re.search(iter40_pattern, content, re.DOTALL)

if iter40_match:
    iter40_section = iter40_match.group(0)
    
    # Extract zeta values
    # Format: - zeta (처음 3개): [1.         1.06771281 1.05707886]
    zeta_pattern = r'zeta \(처음 3개\): \[([\d\.\s]+)\]'
    sigma_pattern = r'sigma_sq \(처음 3개\): \[([\d\.\s]+)\]'
    
    # Extract by LV
    lvs = ['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge', 'purchase_intention']
    
    for lv in lvs:
        # Find LV section
        lv_pattern = f'{lv}:.*?(?=perceived_|nutrition_|purchase_|구조모델|$)'
        lv_match = re.search(lv_pattern, iter40_section, re.DOTALL)
        
        if lv_match:
            lv_section = lv_match.group(0)
            
            # Extract zeta
            zeta_match = re.search(zeta_pattern, lv_section)
            if zeta_match:
                zeta_str = zeta_match.group(1)
                zeta_values = [float(x) for x in zeta_str.split()]
                print(f'{lv} zeta (first 3): {zeta_values}')
            
            # Extract sigma_sq
            sigma_match = re.search(sigma_pattern, lv_section)
            if sigma_match:
                sigma_str = sigma_match.group(1)
                sigma_values = [float(x) for x in sigma_str.split()]
                print(f'{lv} sigma_sq (first 3): {sigma_values}')
            print()
    
    # Extract structural parameters
    gamma_pattern = r'gamma_(\w+): ([\d\.]+)'
    for match in re.finditer(gamma_pattern, iter40_section):
        param_name = match.group(1)
        param_value = float(match.group(2))
        print(f'gamma_{param_name}: {param_value}')
    print()
    
    # Extract choice parameters
    intercept_pattern = r'intercept: ([\d\.]+)'
    beta_pattern = r'beta: \[([\d\.\s\-]+)\]'
    lambda_main_pattern = r'lambda_main: ([\d\.]+)'
    lambda_mod_pattern = r'lambda_mod_(\w+): ([-\d\.]+)'
    
    intercept_match = re.search(intercept_pattern, iter40_section)
    if intercept_match:
        print(f'intercept: {float(intercept_match.group(1))}')
    
    beta_match = re.search(beta_pattern, iter40_section)
    if beta_match:
        beta_str = beta_match.group(1)
        beta_values = [float(x) for x in beta_str.split()]
        print(f'beta: {beta_values}')
    
    lambda_main_match = re.search(lambda_main_pattern, iter40_section)
    if lambda_main_match:
        print(f'lambda_main: {float(lambda_main_match.group(1))}')
    
    for match in re.finditer(lambda_mod_pattern, iter40_section):
        param_name = match.group(1)
        param_value = float(match.group(2))
        print(f'lambda_mod_{param_name}: {param_value}')

print()
print('=' * 80)
print('NOTE: Log only shows first 3 values for zeta and sigma_sq arrays')
print('Need to extract ALL values from a different source or run a script')
print('=' * 80)

