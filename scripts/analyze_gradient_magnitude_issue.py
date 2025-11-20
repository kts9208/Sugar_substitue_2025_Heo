"""
그래디언트 크기 불균형 원인 분석

로그 파일에서 각 모델별 그래디언트 크기를 분석하여
왜 선택모델 그래디언트가 구조모델보다 10,000배 큰지 확인
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path


def parse_gradient_from_log(log_file: str):
    """
    로그 파일에서 그래디언트 정보 추출
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Iteration별 파라미터와 그래디언트 추출
    iterations = []
    
    # "전체 파라미터 값 및 그래디언트:" 섹션 찾기
    pattern = r'전체 파라미터 값 및 그래디언트:\s+((?:\s+\[\s*\d+\].*\n)+)'
    matches = re.finditer(pattern, content)
    
    for match_idx, match in enumerate(matches):
        param_section = match.group(1)
        
        # 각 파라미터 라인 파싱
        param_pattern = r'\[\s*(\d+)\]\s+(\S+)\s+:\s+param=([+-]?\d+\.\d+e[+-]?\d+),\s+grad=([+-]?\d+\.\d+e[+-]?\d+)'
        param_matches = re.finditer(param_pattern, param_section)
        
        iter_data = {'iteration': match_idx + 1, 'params': []}
        
        for pm in param_matches:
            idx = int(pm.group(1))
            name = pm.group(2)
            param_val = float(pm.group(3))
            grad_val = float(pm.group(4))
            
            iter_data['params'].append({
                'index': idx,
                'name': name,
                'param': param_val,
                'grad': grad_val,
                'grad_abs': abs(grad_val)
            })
        
        if iter_data['params']:
            iterations.append(iter_data)
    
    return iterations


def categorize_parameters(param_name: str) -> str:
    """
    파라미터 이름으로 모델 분류
    """
    if param_name.startswith('gamma_'):
        return '구조모델'
    elif param_name.startswith('asc_') or param_name.startswith('beta_'):
        return '선택모델 (고정효과)'
    elif param_name.startswith('theta_'):
        return '선택모델 (LV 계수)'
    elif param_name.startswith('zeta_') or param_name.startswith('tau_'):
        return '측정모델'
    else:
        return '기타'


def analyze_gradient_magnitudes(iterations):
    """
    그래디언트 크기 분석
    """
    print("="*80)
    print("그래디언트 크기 불균형 분석")
    print("="*80)
    
    for iter_data in iterations[:5]:  # 처음 5개 iteration만
        iter_num = iter_data['iteration']
        params = iter_data['params']
        
        print(f"\n{'='*80}")
        print(f"Iteration #{iter_num}")
        print(f"{'='*80}")
        
        # 모델별로 그룹화
        by_model = {}
        for p in params:
            model = categorize_parameters(p['name'])
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(p)
        
        # 모델별 통계
        print(f"\n{'모델':<30} {'개수':>6} {'평균 |grad|':>15} {'최대 |grad|':>15} {'최소 |grad|':>15}")
        print("-"*80)
        
        model_stats = []
        for model, params_list in sorted(by_model.items()):
            grads = [p['grad_abs'] for p in params_list]
            avg_grad = np.mean(grads)
            max_grad = np.max(grads)
            min_grad = np.min(grads)
            
            print(f"{model:<30} {len(params_list):>6} {avg_grad:>15.6e} {max_grad:>15.6e} {min_grad:>15.6e}")
            
            model_stats.append({
                'model': model,
                'avg': avg_grad,
                'max': max_grad,
                'min': min_grad
            })
        
        # 비율 계산
        print(f"\n{'비율 분석':<30}")
        print("-"*80)
        
        if len(model_stats) >= 2:
            # 구조모델 vs 선택모델
            struct_stat = next((s for s in model_stats if s['model'] == '구조모델'), None)
            choice_fixed_stat = next((s for s in model_stats if s['model'] == '선택모델 (고정효과)'), None)
            choice_lv_stat = next((s for s in model_stats if s['model'] == '선택모델 (LV 계수)'), None)
            
            if struct_stat and choice_fixed_stat:
                ratio = choice_fixed_stat['avg'] / struct_stat['avg']
                print(f"선택모델(고정효과) / 구조모델 평균 비율: {ratio:,.1f}x")
                print(f"  - 구조모델 평균: {struct_stat['avg']:.6e}")
                print(f"  - 선택모델(고정효과) 평균: {choice_fixed_stat['avg']:.6e}")
            
            if struct_stat and choice_lv_stat:
                ratio = choice_lv_stat['avg'] / struct_stat['avg']
                print(f"선택모델(LV계수) / 구조모델 평균 비율: {ratio:,.1f}x")
                print(f"  - 구조모델 평균: {struct_stat['avg']:.6e}")
                print(f"  - 선택모델(LV계수) 평균: {choice_lv_stat['avg']:.6e}")
        
        # 개별 파라미터 상세
        print(f"\n{'파라미터별 상세':<30}")
        print("-"*80)
        print(f"{'인덱스':<6} {'이름':<45} {'모델':<25} {'|grad|':>15}")
        print("-"*80)
        
        for p in sorted(params, key=lambda x: x['grad_abs'], reverse=True):
            model = categorize_parameters(p['name'])
            print(f"{p['index']:<6} {p['name']:<45} {model:<25} {p['grad_abs']:>15.6e}")


def main():
    log_file = "results/simultaneous_estimation_log_20251120_192842.txt"
    
    if not Path(log_file).exists():
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        return
    
    print(f"📊 로그 파일 분석: {log_file}\n")
    
    iterations = parse_gradient_from_log(log_file)
    
    if not iterations:
        print("❌ 그래디언트 정보를 찾을 수 없습니다")
        return
    
    print(f"✅ {len(iterations)}개 iteration의 그래디언트 정보 추출 완료\n")
    
    analyze_gradient_magnitudes(iterations)
    
    print("\n" + "="*80)
    print("분석 완료")
    print("="*80)


if __name__ == "__main__":
    main()

