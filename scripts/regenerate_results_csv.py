"""
기존 로그 파일에서 결과 CSV 재생성 스크립트
"""
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# 프로젝트 루트
project_root = Path(__file__).parent.parent
output_dir = project_root / 'results'

# 로그 파일 경로
log_file = output_dir / 'gpu_batch_iclv_estimation_log.txt'

if not log_file.exists():
    print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
    exit(1)

print(f"✅ 로그 파일 찾음: {log_file}")

# 로그 파일 읽기
with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 파라미터 파싱
param_list = []

# "Parameter Scaling Comparison" 섹션 찾기
pattern = r'Parameter Scaling Comparison:.*?-{80}.*?-{80}\n(.*?)-{80}'
match = re.search(pattern, content, re.DOTALL)

if match:
    print("✅ Parameter Scaling Comparison 섹션 찾음")
    param_section = match.group(1)

    # 각 파라미터 라인 파싱 (영문 파라미터 이름)
    param_pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-\s+([a-zA-Z_][^\s]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)'

    for line in param_section.strip().split('\n'):
        param_match = re.match(param_pattern, line.strip())
        if param_match:
            param_name = param_match.group(1)
            external_value = float(param_match.group(2))

            param_list.append({
                'Coefficient': param_name,
                'Estimate': external_value,
                'Std. Err.': 'N/A',
                'P. Value': 'N/A'
            })

    print(f"✅ {len(param_list)}개 파라미터 파싱 완료")
else:
    print("❌ Parameter Scaling Comparison 섹션을 찾을 수 없습니다.")
    exit(1)

# DataFrame 생성
df_params = pd.DataFrame(param_list)

# 로그 파일에서 통계 정보 추출
initial_ll = 'N/A'
final_ll = 'N/A'
n_iterations = 'N/A'

# 초기 LL 찾기 (Major Iteration #1)
ll_pattern = r'\[Major Iteration #1 완료\].*?최종 LL:\s*([-+]?[\d.]+)'
ll_match = re.search(ll_pattern, content, re.DOTALL)
if ll_match:
    initial_ll = f"{float(ll_match.group(1)):.2f}"

# 최종 LL 찾기
final_ll_pattern = r'최종 로그우도:\s*([-+]?[\d.]+)'
final_ll_match = re.search(final_ll_pattern, content)
if final_ll_match:
    final_ll = f"{float(final_ll_match.group(1)):.2f}"

# Iteration 수 찾기 (Major Iteration 카운트)
iter_pattern = r'Major Iteration #(\d+)'
iter_matches = re.findall(iter_pattern, content)
if iter_matches:
    n_iterations = max(int(x) for x in iter_matches)

print(f"  초기 LL: {initial_ll}")
print(f"  최종 LL: {final_ll}")
print(f"  Iterations: {n_iterations}")

# Estimation statistics 추가
stats_list = [
    {'Coefficient': '', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
    {'Coefficient': 'Estimation statistics', 'Estimate': '', 'Std. Err.': '', 'P. Value': ''},
    {'Coefficient': 'Iterations', 'Estimate': n_iterations,
     'Std. Err.': 'LL (start)', 'P. Value': initial_ll},
    {'Coefficient': 'AIC', 'Estimate': 'N/A',
     'Std. Err.': 'LL (final, whole model)', 'P. Value': final_ll},
    {'Coefficient': 'BIC', 'Estimate': 'N/A',
     'Std. Err.': 'LL (Choice)', 'P. Value': 'N/A'}
]

df_stats = pd.DataFrame(stats_list)
df_combined = pd.concat([df_params, df_stats], ignore_index=True)

# 타임스탬프 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV 저장
csv_file = output_dir / f'gpu_batch_iclv_results_regenerated_{timestamp}.csv'
df_combined.to_csv(csv_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 결과 저장 완료: {csv_file}")
print(f"   총 {len(param_list)}개 파라미터 + 통계 정보")

# 파라미터 요약 출력
print("\n파라미터 요약:")
print("-" * 80)
print(f"  측정모델 파라미터 (zeta, sigma_sq): {sum(1 for p in param_list if p['Coefficient'].startswith(('zeta_', 'sigma_sq_')))}개")
print(f"  구조모델 파라미터 (gamma): {sum(1 for p in param_list if p['Coefficient'].startswith('gamma_'))}개")
print(f"  선택모델 파라미터 (beta, lambda): {sum(1 for p in param_list if p['Coefficient'].startswith(('beta_', 'lambda_')))}개")

# ✅ Hessian 역행렬 파싱 및 저장
print("\n" + "=" * 80)
print("Hessian 역행렬 파싱 중...")
print("=" * 80)

# HESSIAN_HEADER 찾기
header_pattern = r'HESSIAN_HEADER,(.+)'
header_match = re.search(header_pattern, content)

if header_match:
    print("✅ Hessian 헤더 찾음")
    param_names = header_match.group(1).split(',')
    n_params = len(param_names)
    print(f"   파라미터 수: {n_params}")

    # HESSIAN_ROW 찾기
    row_pattern = r'HESSIAN_ROW,([^,]+),(.+)'
    row_matches = re.findall(row_pattern, content)

    if row_matches and len(row_matches) == n_params:
        print(f"✅ Hessian 행 {len(row_matches)}개 찾음")

        # Hessian 행렬 구성
        import numpy as np
        hess_inv = np.zeros((n_params, n_params))

        for i, (row_name, row_values_str) in enumerate(row_matches):
            row_values = [float(v) for v in row_values_str.split(',')]
            hess_inv[i, :] = row_values

        # DataFrame 생성
        df_hessian = pd.DataFrame(
            hess_inv,
            index=param_names,
            columns=param_names
        )

        # CSV 저장
        hessian_file = output_dir / f'gpu_batch_iclv_hessian_inv_regenerated_{timestamp}.csv'
        df_hessian.to_csv(hessian_file, encoding='utf-8-sig')

        print(f"✅ Hessian 역행렬 저장 완료: {hessian_file}")
        print(f"   Shape: {hess_inv.shape}")

        # 통계 출력
        diag_elements = np.diag(hess_inv)
        print(f"\n   대각 원소 통계:")
        print(f"     - 범위: [{np.min(diag_elements):.6e}, {np.max(diag_elements):.6e}]")
        print(f"     - 평균: {np.mean(diag_elements):.6e}")
        print(f"     - 음수 개수: {np.sum(diag_elements < 0)}/{len(diag_elements)}")
    else:
        print(f"⚠️  Hessian 행 개수 불일치: {len(row_matches)} (예상: {n_params})")
else:
    print("⚠️  Hessian 헤더를 찾을 수 없습니다 (로그 파일에 저장되지 않음)")

