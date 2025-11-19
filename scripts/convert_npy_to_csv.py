"""
NPY 파일을 CSV로 변환하는 스크립트

Usage:
    python scripts/convert_npy_to_csv.py results/simultaneous_HC-PB_PB-PI_results_20251118_131739.npy
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_multi_latent_config
from analysis.hybrid_choice_model.iclv_models.parameter_manager import ParameterManager

def convert_npy_to_csv(npy_file: str):
    """NPY 파일을 CSV로 변환"""
    
    npy_path = Path(npy_file)
    if not npy_path.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {npy_file}")
        return
    
    # NPY 파일 로드
    params = np.load(npy_path)
    print(f"파라미터 로드 완료: {len(params)}개")
    
    # Config 생성 (HC->PB, PB->PI 경로)
    paths = [
        ('health_concern', 'perceived_benefit'),
        ('perceived_benefit', 'purchase_intention')
    ]
    
    config = create_multi_latent_config(
        paths=paths,
        main_lvs=['purchase_intention'],
        lv_attribute_interactions=[],
        moderation_lvs=[]
    )
    
    # ParameterManager 생성
    param_manager = ParameterManager(config)
    
    # 모델 생성 (파라미터 이름 추출용)
    from analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
    from analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
    from analysis.hybrid_choice_model.choice_models.multinomial_logit_choice import MultinomialLogitChoice
    
    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    choice_model = MultinomialLogitChoice(config.choice)
    
    # 파라미터 이름 생성
    param_names = param_manager.get_parameter_names(
        measurement_model, structural_model, choice_model
    )
    
    print(f"파라미터 이름 생성 완료: {len(param_names)}개")
    
    # 파라미터 딕셔너리로 변환
    param_dict = param_manager.array_to_dict(
        params, param_names, measurement_model, structural_model, choice_model
    )
    
    # DataFrame 생성
    param_list = []
    
    # 측정모델 파라미터
    for lv_name, lv_params in param_dict['measurement'].items():
        for param_type, values in lv_params.items():
            if isinstance(values, np.ndarray):
                for i, val in enumerate(values):
                    param_list.append({
                        'Parameter': f'{lv_name}_{param_type}_{i}',
                        'Estimate': val,
                        'Component': 'Measurement'
                    })
            else:
                param_list.append({
                    'Parameter': f'{lv_name}_{param_type}',
                    'Estimate': values,
                    'Component': 'Measurement'
                })
    
    # 구조모델 파라미터
    for param_name, value in param_dict['structural'].items():
        param_list.append({
            'Parameter': param_name,
            'Estimate': value,
            'Component': 'Structural'
        })
    
    # 선택모델 파라미터
    for param_name, value in param_dict['choice'].items():
        if isinstance(value, np.ndarray):
            for i, val in enumerate(value):
                param_list.append({
                    'Parameter': f'{param_name}_{i}',
                    'Estimate': val,
                    'Component': 'Choice'
                })
        else:
            param_list.append({
                'Parameter': param_name,
                'Estimate': value,
                'Component': 'Choice'
            })
    
    df = pd.DataFrame(param_list)
    
    # CSV 저장
    csv_path = npy_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[OK] CSV 저장 완료: {csv_path.name}")
    print(f"  - 파라미터 수: {len(param_list)}")
    print(f"  - 측정모델: {len([p for p in param_list if p['Component'] == 'Measurement'])}개")
    print(f"  - 구조모델: {len([p for p in param_list if p['Component'] == 'Structural'])}개")
    print(f"  - 선택모델: {len([p for p in param_list if p['Component'] == 'Choice'])}개")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_npy_to_csv.py <npy_file>")
        sys.exit(1)
    
    convert_npy_to_csv(sys.argv[1])

