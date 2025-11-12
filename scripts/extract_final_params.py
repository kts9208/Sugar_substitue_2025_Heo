"""
최근 로그 파일에서 최종 파라미터 값 추출
"""
import re
from pathlib import Path

# 로그 파일 경로
project_root = Path(__file__).parent.parent
log_file = project_root / 'results' / 'gpu_batch_iclv_estimation_log.txt'

# 로그 파일 읽기
with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

# "Parameter Scaling Comparison" 섹션 찾기
pattern = r'Parameter Scaling Comparison:.*?-{80}.*?-{80}\n(.*?)-{80}'
match = re.search(pattern, content, re.DOTALL)

if not match:
    print("❌ Parameter Scaling Comparison 섹션을 찾을 수 없습니다.")
    exit(1)

param_section = match.group(1)

# 파라미터 파싱 (그리스 문자 사용)
param_pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-\s+([ζσ²γβλ_][^\s]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)'

params = {}
for line in param_section.strip().split('\n'):
    param_match = re.match(param_pattern, line.strip())
    if param_match:
        param_name = param_match.group(1)
        external_value = float(param_match.group(2))
        params[param_name] = external_value

print(f"✅ {len(params)}개 파라미터 추출 완료\n")

# 잠재변수별로 분류
zeta_params = {}
sigma_sq_params = {}
gamma_params = {}
beta_params = {}
lambda_params = {}

for name, value in params.items():
    if name.startswith('ζ_'):
        # ζ_health_concern_q7 → health_concern
        parts = name.split('_')
        lv_name = '_'.join(parts[1:-1])  # health_concern
        if lv_name not in zeta_params:
            zeta_params[lv_name] = []
        zeta_params[lv_name].append(value)
    
    elif name.startswith('σ²_'):
        # σ²_health_concern_q6 → health_concern
        parts = name.split('_')
        lv_name = '_'.join(parts[1:-1])
        if lv_name not in sigma_sq_params:
            sigma_sq_params[lv_name] = []
        sigma_sq_params[lv_name].append(value)
    
    elif name.startswith('γ_'):
        # γ_health_concern_to_perceived_benefit
        gamma_params[name[2:]] = value  # 'γ_' 제거
    
    elif name.startswith('β_'):
        # β_intercept, β_sugar_free, etc.
        beta_params[name[2:]] = value
    
    elif name.startswith('λ_'):
        # λ_main, λ_mod_perceived_price, etc.
        lambda_params[name[2:]] = value

# Python 코드 생성
print("=" * 80)
print("초기값 파일 생성 (initial_values_final.py)")
print("=" * 80)
print()

output_file = project_root / 'src' / 'analysis' / 'hybrid_choice_model' / 'iclv_models' / 'initial_values_final.py'

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('"""\n')
    f.write('최종 수렴값 기반 초기값 (Iteration 24)\n')
    f.write('\n')
    f.write('최근 로그 파일의 마지막 iteration 파라미터 값을 초기값으로 사용\n')
    f.write('"""\n\n')
    
    # ZETA_INITIAL_VALUES
    f.write('# 측정모델 파라미터 - 요인적재량 (zeta)\n')
    f.write('# 첫 번째 지표는 1.0으로 고정되므로 파라미터 벡터에 포함되지 않음\n')
    f.write('ZETA_INITIAL_VALUES = {\n')
    for lv_name in sorted(zeta_params.keys()):
        values = zeta_params[lv_name]
        f.write(f"    '{lv_name}': {{\n")
        f.write(f"        'values': {values}\n")
        f.write(f"    }},\n")
    f.write('}\n\n')
    
    # SIGMA_SQ_INITIAL_VALUES
    f.write('# 측정모델 파라미터 - 오차분산 (sigma_sq)\n')
    f.write('# 모든 지표에 대해 설정\n')
    f.write('SIGMA_SQ_INITIAL_VALUES = {\n')
    for lv_name in sorted(sigma_sq_params.keys()):
        values = sigma_sq_params[lv_name]
        f.write(f"    '{lv_name}': {{\n")
        f.write(f"        'values': {values}\n")
        f.write(f"    }},\n")
    f.write('}\n\n')
    
    # GAMMA_INITIAL_VALUES
    f.write('# 구조모델 파라미터 (gamma)\n')
    f.write('# 계층적 경로: HC → PB, PB → PI\n')
    f.write('GAMMA_INITIAL_VALUES = {\n')
    for param_name, value in sorted(gamma_params.items()):
        f.write(f"    '{param_name}': {value},\n")
    f.write('}\n\n')
    
    # CHOICE_INITIAL_VALUES
    f.write('# 선택모델 파라미터\n')
    f.write('CHOICE_INITIAL_VALUES = {\n')
    
    # 절편
    if 'intercept' in beta_params:
        f.write(f"    'intercept': {beta_params['intercept']},\n")
    
    # Beta 파라미터
    for param_name, value in sorted(beta_params.items()):
        if param_name != 'intercept':
            f.write(f"    'beta_{param_name}': {value},\n")
    
    # Lambda 파라미터
    for param_name, value in sorted(lambda_params.items()):
        f.write(f"    'lambda_{param_name}': {value},\n")
    
    f.write('}\n\n')
    
    # Helper functions
    f.write('''
def get_initial_parameters_from_final():
    """
    최종 수렴값 기반 초기값을 파라미터 벡터로 변환
    
    Returns:
        dict: 각 파라미터 유형별 초기값
    """
    return {
        'zeta': ZETA_INITIAL_VALUES,
        'sigma_sq': SIGMA_SQ_INITIAL_VALUES,
        'gamma': GAMMA_INITIAL_VALUES,
        'choice': CHOICE_INITIAL_VALUES
    }


def get_zeta_initial_value(lv_name, default=1.0):
    """
    특정 잠재변수의 zeta 초기값 반환
    
    Args:
        lv_name: 잠재변수 이름
        default: 기본값 (해당 LV가 없을 경우)
    
    Returns:
        list: 초기값 리스트
    """
    if lv_name in ZETA_INITIAL_VALUES:
        return ZETA_INITIAL_VALUES[lv_name]['values']
    return [default]


def get_sigma_sq_initial_value(lv_name, default=1.0):
    """
    특정 잠재변수의 sigma_sq 초기값 반환
    
    Args:
        lv_name: 잠재변수 이름
        default: 기본값 (해당 LV가 없을 경우)
    
    Returns:
        list: 초기값 리스트
    """
    if lv_name in SIGMA_SQ_INITIAL_VALUES:
        return SIGMA_SQ_INITIAL_VALUES[lv_name]['values']
    return [default]


def get_gamma_initial_value(path_name, default=0.5):
    """
    특정 구조 경로의 gamma 초기값 반환
    
    Args:
        path_name: 경로 이름 (예: 'health_concern_to_perceived_benefit')
        default: 기본값
    
    Returns:
        float: 초기값
    """
    return GAMMA_INITIAL_VALUES.get(path_name, default)


def get_choice_initial_value(param_name, default=0.0):
    """
    선택모델 파라미터 초기값 반환
    
    Args:
        param_name: 파라미터 이름
        default: 기본값
    
    Returns:
        float: 초기값
    """
    return CHOICE_INITIAL_VALUES.get(param_name, default)
''')

print(f"✅ 초기값 파일 생성 완료: {output_file}")
print()

# 요약 출력
print("=" * 80)
print("파라미터 요약")
print("=" * 80)
for lv_name in sorted(zeta_params.keys()):
    print(f"\n{lv_name}:")
    print(f"  zeta: {len(zeta_params[lv_name])}개")
    print(f"  sigma_sq: {len(sigma_sq_params[lv_name])}개")

print(f"\n구조모델:")
for param_name, value in sorted(gamma_params.items()):
    print(f"  {param_name}: {value:.6f}")

print(f"\n선택모델:")
for param_name, value in sorted(beta_params.items()):
    print(f"  beta_{param_name}: {value:.6f}")
for param_name, value in sorted(lambda_params.items()):
    print(f"  lambda_{param_name}: {value:.6f}")

