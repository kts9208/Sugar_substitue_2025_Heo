"""
2단계 추정: 20개 모델 케이스 자동 실행

sequential_stage2_with_extended_model.py의 설정을 바꿔가면서 20개 모델을 순차적으로 추정합니다.

Author: ICLV Team
Date: 2025-01-17
"""

import sys
from pathlib import Path
import subprocess

# 프로젝트 루트
project_root = Path(__file__).parent

# 20개 모델 케이스 정의
MODEL_CASES = [
    # 1. Base Model
    {'name': 'Base Model', 'main_lvs': [], 'moderation_lvs': [], 'lv_attr_interactions': []},
    
    # 2-6. Base + 단일 LV
    {'name': 'Base + PI', 'main_lvs': ['purchase_intention'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    {'name': 'Base + NK', 'main_lvs': ['nutrition_knowledge'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    {'name': 'Base + PB', 'main_lvs': ['perceived_benefit'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    {'name': 'Base + PP', 'main_lvs': ['perceived_price'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    {'name': 'Base + HC', 'main_lvs': ['health_concern'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    
    # 7-10. Base + PI + 다른 LV
    {'name': 'Base + PI + NK', 'main_lvs': ['purchase_intention', 'nutrition_knowledge'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    {'name': 'Base + PI + PB', 'main_lvs': ['purchase_intention', 'perceived_benefit'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    {'name': 'Base + PI + PP', 'main_lvs': ['purchase_intention', 'perceived_price'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    {'name': 'Base + PI + HC', 'main_lvs': ['purchase_intention', 'health_concern'], 'moderation_lvs': [], 'lv_attr_interactions': []},
    
    # 11-14. Base + PI + 단일 상호작용
    {'name': 'Base + PI + int_PIxpr', 'main_lvs': ['purchase_intention'], 'moderation_lvs': [], 'lv_attr_interactions': [('purchase_intention', 'price')]},
    {'name': 'Base + PI + int_PIxhl', 'main_lvs': ['purchase_intention'], 'moderation_lvs': [], 'lv_attr_interactions': [('purchase_intention', 'health_label')]},
    {'name': 'Base + PI + int_NKxpr', 'main_lvs': ['purchase_intention'], 'moderation_lvs': [], 'lv_attr_interactions': [('nutrition_knowledge', 'price')]},
    {'name': 'Base + PI + int_NKxhl', 'main_lvs': ['purchase_intention'], 'moderation_lvs': [], 'lv_attr_interactions': [('nutrition_knowledge', 'health_label')]},
    
    # 15-18. Base + PI + NK + 단일 상호작용
    {'name': 'Base + PI + NK + int_PIxpr', 'main_lvs': ['purchase_intention', 'nutrition_knowledge'], 'moderation_lvs': [], 'lv_attr_interactions': [('purchase_intention', 'price')]},
    {'name': 'Base + PI + NK + int_PIxhl', 'main_lvs': ['purchase_intention', 'nutrition_knowledge'], 'moderation_lvs': [], 'lv_attr_interactions': [('purchase_intention', 'health_label')]},
    {'name': 'Base + PI + NK + int_NKxpr', 'main_lvs': ['purchase_intention', 'nutrition_knowledge'], 'moderation_lvs': [], 'lv_attr_interactions': [('nutrition_knowledge', 'price')]},
    {'name': 'Base + PI + NK + int_NKxhl', 'main_lvs': ['purchase_intention', 'nutrition_knowledge'], 'moderation_lvs': [], 'lv_attr_interactions': [('nutrition_knowledge', 'health_label')]},
    
    # 19-20. Base + PI + NK + 복수 상호작용
    {'name': 'Base + PI + NK + int_PIxpr_NKxhl', 'main_lvs': ['purchase_intention', 'nutrition_knowledge'], 'moderation_lvs': [], 'lv_attr_interactions': [('purchase_intention', 'price'), ('nutrition_knowledge', 'health_label')]},
    {'name': 'Base + PI + NK + int_PIxhl_NKxpr', 'main_lvs': ['purchase_intention', 'nutrition_knowledge'], 'moderation_lvs': [], 'lv_attr_interactions': [('purchase_intention', 'health_label'), ('nutrition_knowledge', 'price')]},
]


def modify_stage2_script(model_config):
    """sequential_stage2_with_extended_model.py의 설정 수정"""
    script_path = project_root / "examples" / "sequential_stage2_with_extended_model.py"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.strip().startswith('MAIN_LVS ='):
            lines[i] = f"    MAIN_LVS = {model_config['main_lvs']}  # Auto-generated\n"
        elif line.strip().startswith('MODERATION_LVS ='):
            lines[i] = f"    MODERATION_LVS = {model_config['moderation_lvs']}  # Auto-generated\n"
        elif line.strip().startswith('LV_ATTRIBUTE_INTERACTIONS ='):
            lines[i] = f"    LV_ATTRIBUTE_INTERACTIONS = {model_config['lv_attr_interactions']}  # Auto-generated\n"
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def run_single_model(model_config, index, total):
    """단일 모델 추정"""
    print(f"\n{'='*80}")
    print(f"[{index}/{total}] {model_config['name']}")
    print(f"{'='*80}")
    
    try:
        # 1. 스크립트 설정 수정
        modify_stage2_script(model_config)
        
        # 2. 스크립트 실행
        script_path = project_root / "examples" / "sequential_stage2_with_extended_model.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 3. 결과 확인
        if result.returncode == 0:
            output = result.stdout
            ll_val = None
            aic_val = None
            
            for line in output.split('\n'):
                if 'Log-Likelihood' in line or 'log_likelihood' in line:
                    try:
                        ll_val = float(line.split()[-1])
                    except:
                        pass
                if 'AIC' in line and 'BIC' not in line:
                    try:
                        aic_val = float(line.split()[-1])
                    except:
                        pass
            
            print(f"OK - {model_config['name']}")
            if ll_val:
                print(f"  LL: {ll_val:.2f}")
            if aic_val:
                print(f"  AIC: {aic_val:.2f}")
            return True
        else:
            print(f"FAIL - {model_config['name']}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"FAIL - {model_config['name']}: {e}")
        return False


def main():
    print("="*80)
    print("2 Stage Estimation: 20 Model Cases")
    print("="*80)
    
    total = len(MODEL_CASES)
    success_count = 0
    
    for i, model_config in enumerate(MODEL_CASES, 1):
        if run_single_model(model_config, i, total):
            success_count += 1
    
    print(f"\n{'='*80}")
    print("All Done!")
    print(f"{'='*80}")
    print(f"Success: {success_count}/{total}")
    print(f"Failed: {total - success_count}/{total}")


if __name__ == '__main__':
    main()

