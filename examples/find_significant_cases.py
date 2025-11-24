"""
유의한 파라미터가 있는 케이스 찾기

250개 결과 파일을 검토하여 주효과 또는 상호작용항이 유의한 케이스를 찾습니다.

사용법:
    python examples/find_significant_cases.py

Author: ICLV Team
Date: 2025-11-23
"""

import sys
from pathlib import Path
import pandas as pd
import pickle
from typing import List, Dict

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_stage2_results(result_file: Path) -> Dict:
    """2단계 결과 파일 로드"""
    try:
        df = pd.read_csv(result_file)
        return {
            'file': result_file.name,
            'data': df
        }
    except Exception as e:
        return None


def check_significance(result_data: Dict, alpha: float = 0.05) -> Dict:
    """
    유의성 검사
    
    Args:
        result_data: 결과 데이터
        alpha: 유의수준 (기본값: 0.05)
    
    Returns:
        유의성 정보 딕셔너리
    """
    df = result_data['data']
    
    # p-value 컬럼이 있는지 확인
    if 'p_value' not in df.columns:
        return None
    
    # 유의한 파라미터 찾기
    significant = df[df['p_value'] < alpha].copy()
    
    if len(significant) == 0:
        return None
    
    # theta (주효과) 와 gamma (상호작용) 분리
    theta_params = significant[significant['parameter'].str.contains('theta', na=False)]
    gamma_params = significant[significant['parameter'].str.contains('gamma', na=False)]
    
    return {
        'file': result_data['file'],
        'n_significant': len(significant),
        'n_theta': len(theta_params),
        'n_gamma': len(gamma_params),
        'theta_params': theta_params[['parameter', 'estimate', 'std_error', 'p_value']].to_dict('records') if len(theta_params) > 0 else [],
        'gamma_params': gamma_params[['parameter', 'estimate', 'std_error', 'p_value']].to_dict('records') if len(gamma_params) > 0 else []
    }


def scan_all_results(alpha: float = 0.05) -> List[Dict]:
    """
    모든 결과 파일 스캔
    
    Args:
        alpha: 유의수준
    
    Returns:
        유의한 케이스 리스트
    """
    significant_cases = []
    
    # 2path와 3path 폴더 스캔
    for path_name in ['2path', '3path']:
        stage2_dir = project_root / "results" / "final" / "sequential" / path_name / "stage2"
        
        if not stage2_dir.exists():
            print(f"⚠️  {path_name} 폴더가 없습니다: {stage2_dir}")
            continue
        
        # 모든 CSV 파일 찾기
        result_files = list(stage2_dir.glob("st2_*.csv"))
        print(f"\n[{path_name}] {len(result_files)}개 파일 검토 중...")
        
        for result_file in result_files:
            # 결과 로드
            result_data = load_stage2_results(result_file)
            if result_data is None:
                continue
            
            # 유의성 검사
            sig_info = check_significance(result_data, alpha)
            if sig_info is not None:
                sig_info['path_name'] = path_name
                significant_cases.append(sig_info)
    
    return significant_cases


def print_significant_cases(cases: List[Dict], alpha: float = 0.05):
    """유의한 케이스 출력"""
    print("\n" + "="*70)
    print(f"유의한 파라미터가 있는 케이스 (p < {alpha})")
    print("="*70)
    
    if len(cases) == 0:
        print("\n유의한 파라미터가 있는 케이스가 없습니다.")
        return
    
    print(f"\n총 {len(cases)}개 케이스 발견")
    
    # 주효과가 유의한 케이스
    theta_cases = [c for c in cases if c['n_theta'] > 0]
    print(f"\n주효과(theta)가 유의한 케이스: {len(theta_cases)}개")
    
    # 상호작용이 유의한 케이스
    gamma_cases = [c for c in cases if c['n_gamma'] > 0]
    print(f"상호작용(gamma)이 유의한 케이스: {len(gamma_cases)}개")
    
    # 상세 출력
    for i, case in enumerate(cases, 1):
        print(f"\n{'='*70}")
        print(f"{i}. [{case['path_name']}] {case['file']}")
        print(f"{'='*70}")
        print(f"유의한 파라미터 총 {case['n_significant']}개 (주효과: {case['n_theta']}, 상호작용: {case['n_gamma']})")
        
        # 주효과 출력
        if case['theta_params']:
            print(f"\n[주효과 (theta)]")
            for param in case['theta_params']:
                p_str = f"{param['p_value']:.4f}"
                sig_str = "***" if param['p_value'] < 0.001 else "**" if param['p_value'] < 0.01 else "*"
                print(f"  {param['parameter']:50s} = {param['estimate']:8.4f} (SE={param['std_error']:6.4f}, p={p_str}) {sig_str}")
        
        # 상호작용 출력
        if case['gamma_params']:
            print(f"\n[상호작용 (gamma)]")
            for param in case['gamma_params']:
                p_str = f"{param['p_value']:.4f}"
                sig_str = "***" if param['p_value'] < 0.001 else "**" if param['p_value'] < 0.01 else "*"
                print(f"  {param['parameter']:50s} = {param['estimate']:8.4f} (SE={param['std_error']:6.4f}, p={p_str}) {sig_str}")


def save_summary(cases: List[Dict], alpha: float = 0.05):
    """요약 저장"""
    if len(cases) == 0:
        return
    
    # DataFrame 생성
    rows = []
    for case in cases:
        row = {
            'path_name': case['path_name'],
            'file': case['file'],
            'n_significant': case['n_significant'],
            'n_theta': case['n_theta'],
            'n_gamma': case['n_gamma']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 저장
    output_path = project_root / "results" / "final" / "sequential" / f"significant_cases_p{alpha}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n요약 저장: {output_path}")


def main():
    print("="*70)
    print("유의한 파라미터 검색")
    print("="*70)

    # 유의수준
    alpha = 0.3

    # 모든 결과 스캔
    significant_cases = scan_all_results(alpha)

    # 결과 출력
    print_significant_cases(significant_cases, alpha)

    # 요약 저장
    save_summary(significant_cases, alpha)


if __name__ == "__main__":
    main()

