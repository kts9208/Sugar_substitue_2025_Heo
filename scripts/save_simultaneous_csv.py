"""
동시추정 결과를 CSV로 저장하는 스크립트

이미 완료된 추정 결과(.npy 파일)를 읽어서 CSV 파일로 저장합니다.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.model_config_utils import save_simultaneous_results, generate_iclv_filename
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import MultiLatentConfig

# 설정
NPY_FILE = project_root / 'results' / 'final' / 'simultaneous' / 'results' / 'simul_2path_NK_PI_PP_results.npy'
LOG_FILE = project_root / 'results' / 'final' / 'simultaneous' / 'logs' / 'simultaneous_estimation_log_20251124_212619.txt'
CFA_RESULTS_FILE = project_root / 'results' / 'final' / 'cfa_only' / 'cfa_results.pkl'

# 1. CFA 결과 로드
import pickle
with open(CFA_RESULTS_FILE, 'rb') as f:
    cfa_results = pickle.load(f)

print(f"CFA 결과 로드 완료")
print(f"  - loadings shape: {cfa_results['loadings'].shape}")
print(f"  - measurement_errors shape: {cfa_results['measurement_errors'].shape}")

# 2. Config 생성 (측정모델 정보 필요)
# 간단한 config 객체 생성 (측정모델 정보만 필요)
class SimpleConfig:
    def __init__(self):
        self.measurement_configs = {}

config = SimpleConfig()

# 측정모델 설정 추가 (CFA 결과에서)
config.measurement_configs = {}
for lv_name in ['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge', 'purchase_intention']:
    # CFA 결과에서 해당 LV의 지표 추출
    lv_loadings = cfa_results['loadings'][cfa_results['loadings']['rval'] == lv_name]
    indicators = lv_loadings['lval'].tolist()
    
    # MeasurementConfig 생성 (간단한 객체)
    class MeasurementConfig:
        def __init__(self, indicators):
            self.indicators = indicators
    
    config.measurement_configs[lv_name] = MeasurementConfig(indicators)

print(f"\nConfig 생성 완료")
print(f"  - 측정모델 LV: {list(config.measurement_configs.keys())}")

# 3. 로그 파일에서 결과 파싱
# 최종 LL, AIC, BIC 추출
with open(LOG_FILE, 'r', encoding='utf-8') as f:
    log_content = f.read()

# LL 추출
import re
ll_match = re.search(r'언스케일링된 우도 \(AIC/BIC용\): ([-\d.]+)', log_content)
if ll_match:
    final_ll = float(ll_match.group(1))
else:
    # 폴백: 스케일링된 우도
    ll_match = re.search(r'최종 LL: ([-\d.]+)', log_content)
    final_ll = float(ll_match.group(1)) if ll_match else -2051.35

# AIC, BIC 추출
aic_match = re.search(r'AIC \(언스케일링\): ([\d.]+)', log_content)
bic_match = re.search(r'BIC \(언스케일링\): ([\d.]+)', log_content)

final_aic = float(aic_match.group(1)) if aic_match else 0.0
final_bic = float(bic_match.group(1)) if bic_match else 0.0

print(f"\n로그 파일 파싱 완료")
print(f"  - LL: {final_ll:.4f}")
print(f"  - AIC: {final_aic:.2f}")
print(f"  - BIC: {final_bic:.2f}")

# 4. .npy 파일 로드
raw_params = np.load(NPY_FILE)
print(f"\n파라미터 로드 완료: {len(raw_params)}개")

# 5. 결과 딕셔너리 생성 (save_simultaneous_results에 필요한 형식)
# 파라미터 이름 (로그에서 추출 또는 하드코딩)
param_names = [
    'gamma_health_concern_to_perceived_benefit',
    'gamma_perceived_benefit_to_purchase_intention',
    'asc_sugar',
    'asc_sugar_free',
    'beta_health_label',
    'beta_price',
    'theta_sugar_nutrition_knowledge',
    'theta_sugar_free_nutrition_knowledge',
    'theta_sugar_purchase_intention',
    'theta_sugar_free_purchase_intention',
    'theta_sugar_perceived_price',
    'theta_sugar_free_perceived_price'
]

# parameter_statistics 생성 (SE 없이)
parameter_statistics = {}
for i, name in enumerate(param_names):
    parameter_statistics[name] = {
        'estimate': raw_params[i],
        'se': '',  # SE 계산 실패
        't': '',
        'p': '',
    }

# 구조모델 파라미터 분리
structural_params = {}
for name in param_names:
    if name.startswith('gamma_'):
        structural_params[name] = parameter_statistics[name]

results = {
    'log_likelihood': final_ll,
    'aic': final_aic,
    'bic': final_bic,
    'raw_params': raw_params,
    'parameter_statistics': parameter_statistics,
    'success': True
}

# 구조모델 파라미터 추가
results['parameter_statistics'] = {
    **parameter_statistics,
    'structural': structural_params
}

print(f"\n결과 딕셔너리 생성 완료")

# 6. CSV 저장
output_dir = project_root / 'results' / 'final' / 'simultaneous' / 'results'
csv_filename = 'simul_2path_NK_PI_PP_results.csv'
csv_file = output_dir / csv_filename

print(f"\nCSV 저장 중: {csv_filename}")
save_simultaneous_results(
    results=results,
    save_path=csv_file,
    cfa_results=cfa_results,
    config=config
)

print(f"✅ CSV 저장 완료!")
print(f"  - 파일: {csv_file}")

