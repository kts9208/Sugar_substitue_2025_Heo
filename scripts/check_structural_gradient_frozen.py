"""
구조모델 그래디언트가 고정되어 있는지 확인

로그 파일에서 구조모델 파라미터의 그래디언트가
모든 iteration에서 동일한지 확인
"""
import re
import numpy as np
from pathlib import Path


def check_frozen_gradients(log_file: str):
    """
    로그 파일에서 구조모델 그래디언트가 고정되어 있는지 확인
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # "전체 파라미터 값 및 그래디언트:" 섹션 찾기
    pattern = r'전체 파라미터 값 및 그래디언트:\s+((?:\s+\[\s*\d+\].*\n)+)'
    matches = re.finditer(pattern, content)
    
    # 구조모델 파라미터 추적
    gamma_gradients = {
        'gamma_health_concern_to_perceived_benefit': [],
        'gamma_perceived_benefit_to_purchase_intention': []
    }
    
    for match_idx, match in enumerate(matches):
        param_section = match.group(1)
        
        # 각 파라미터 라인 파싱
        param_pattern = r'\[\s*\d+\]\s+(\S+)\s+:\s+param=([+-]?\d+\.\d+e[+-]?\d+),\s+grad=([+-]?\d+\.\d+e[+-]?\d+)'
        param_matches = re.finditer(param_pattern, param_section)
        
        for pm in param_matches:
            name = pm.group(1)
            param_val = float(pm.group(2))
            grad_val = float(pm.group(3))
            
            if name in gamma_gradients:
                gamma_gradients[name].append({
                    'iteration': match_idx + 1,
                    'param': param_val,
                    'grad': grad_val
                })
    
    return gamma_gradients


def main():
    log_file = "results/simultaneous_estimation_log_20251120_192842.txt"
    
    if not Path(log_file).exists():
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        return
    
    print("="*80)
    print("구조모델 그래디언트 고정 여부 확인")
    print("="*80)
    print(f"\n로그 파일: {log_file}\n")
    
    gamma_gradients = check_frozen_gradients(log_file)
    
    for param_name, history in gamma_gradients.items():
        print(f"\n{'='*80}")
        print(f"파라미터: {param_name}")
        print(f"{'='*80}")
        
        if not history:
            print("❌ 데이터 없음")
            continue
        
        print(f"\n{'Iteration':<12} {'파라미터 값':<20} {'그래디언트':<20}")
        print("-"*80)
        
        grads = []
        params = []
        
        for h in history:
            print(f"{h['iteration']:<12} {h['param']:<20.10e} {h['grad']:<20.10e}")
            grads.append(h['grad'])
            params.append(h['param'])
        
        # 통계
        grads = np.array(grads)
        params = np.array(params)
        
        print(f"\n{'통계':<30}")
        print("-"*80)
        print(f"그래디언트 범위: [{np.min(grads):.10e}, {np.max(grads):.10e}]")
        print(f"그래디언트 표준편차: {np.std(grads):.10e}")
        print(f"그래디언트 변화량 (max - min): {np.max(grads) - np.min(grads):.10e}")
        
        # 고정 여부 판단
        if np.std(grads) < 1e-10:
            print(f"\n🔴 **그래디언트가 완전히 고정되어 있습니다!**")
            print(f"   모든 iteration에서 동일한 값: {grads[0]:.10e}")
        elif np.std(grads) < 1e-6:
            print(f"\n⚠️ 그래디언트가 거의 변하지 않습니다")
            print(f"   표준편차: {np.std(grads):.10e}")
        else:
            print(f"\n✅ 그래디언트가 정상적으로 변화합니다")
        
        print(f"\n파라미터 범위: [{np.min(params):.10e}, {np.max(params):.10e}]")
        print(f"파라미터 표준편차: {np.std(params):.10e}")
        print(f"파라미터 변화량 (max - min): {np.max(params) - np.min(params):.10e}")
        
        if np.std(params) < 1e-6:
            print(f"\n⚠️ 파라미터도 거의 변하지 않습니다")
        else:
            print(f"\n✅ 파라미터는 변화합니다")
    
    print(f"\n{'='*80}")
    print("분석 완료")
    print("="*80)
    
    # 결론
    print(f"\n{'결론':<30}")
    print("-"*80)
    
    all_frozen = True
    for param_name, history in gamma_gradients.items():
        if history:
            grads = np.array([h['grad'] for h in history])
            if np.std(grads) >= 1e-10:
                all_frozen = False
                break
    
    if all_frozen:
        print("🔴 **모든 구조모델 그래디언트가 고정되어 있습니다!**")
        print()
        print("이는 다음을 의미합니다:")
        print("1. 구조모델 그래디언트 계산이 파라미터 변화를 반영하지 못함")
        print("2. 그래디언트가 초기값에서 고정됨")
        print("3. 구조모델 파라미터가 최적화되지 않음")
        print()
        print("가능한 원인:")
        print("- 그래디언트 계산 로직 버그")
        print("- 캐싱 문제 (이전 값 재사용)")
        print("- 파라미터 전달 오류")
    else:
        print("✅ 구조모델 그래디언트가 정상적으로 변화합니다")


if __name__ == "__main__":
    main()

