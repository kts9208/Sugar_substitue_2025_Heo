"""
모든 순차추정 케이스 자동 실행

2경로 및 3경로 모델에 대해 정의된 모든 선택모델 케이스를 자동으로 실행합니다.

사용법:
    python examples/run_all_sequential_cases.py --path 2  # 2경로만
    python examples/run_all_sequential_cases.py --path 3  # 3경로만
    python examples/run_all_sequential_cases.py --path all  # 모두

Author: ICLV Team
Date: 2025-11-23
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "examples"))

from examples.generate_all_model_cases import generate_theory_driven_cases, generate_all_valid_cases
from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice
from examples.model_config_utils import build_choice_config_dict, generate_stage2_filename, save_stage2_results


# ============================================================================
# 경로 설정
# ============================================================================

PATH_CONFIGS = {
    '2path': {
        'HC->PB': True,
        'HC->PP': False,
        'HC->PI': False,
        'PB->PI': True,
        'PP->PI': False,
        'NK->PI': False,
    },
    '3path': {
        'HC->PB': True,
        'HC->PP': True,
        'HC->PI': False,
        'PB->PI': True,
        'PP->PI': False,
        'NK->PI': False,
    }
}


# ============================================================================
# 실행 함수
# ============================================================================

def run_single_case(
    path_name: str,
    case: dict,
    data: pd.DataFrame,
    stage1_result_path: Path,
    save_dir: Path
) -> dict:
    """
    단일 케이스 실행
    
    Args:
        path_name: 경로 이름 (예: '2path', '3path')
        case: 케이스 딕셔너리 {'main_lvs': [...], 'interactions': [...]}
        data: 데이터
        stage1_result_path: 1단계 결과 파일 경로
        save_dir: 저장 디렉토리
    
    Returns:
        결과 딕셔너리
    """
    print(f"\n{'='*70}")
    print(f"케이스: {case.get('name', 'Unnamed')}")
    print(f"{'='*70}")
    print(f"주효과: {case['main_lvs']}")
    print(f"상호작용: {case['interactions']}")
    
    start_time = time.time()
    
    try:
        # 모델 설정 생성
        config = create_sugar_substitute_multi_lv_config(
            n_draws=100,
            max_iterations=1000,
            use_hierarchical=False,
            all_lvs_as_main=False
        )
        
        # 선택모델 설정
        choice_config_dict = build_choice_config_dict(
            main_lvs=case['main_lvs'],
            lv_attribute_interactions=case['interactions']
        )

        config.choice = ChoiceConfig(
            choice_attributes=['health_label', 'price'],
            choice_type='multinomial',
            n_alternatives=3,  # sugar, sugar_free, no_purchase
            main_lvs=choice_config_dict['main_lvs'],
            lv_attribute_interactions=choice_config_dict['lv_attribute_interactions']
        )
        
        # 선택모델 생성
        choice_model = MultinomialLogitChoice(config.choice)
        
        # Estimator 생성
        estimator = SequentialEstimator(config, standardization_method='zscore')
        
        # 2단계 추정
        results = estimator.estimate_stage2_only(
            data=data,
            choice_model=choice_model,
            factor_scores=str(stage1_result_path)
        )
        
        # 파일명 생성
        filename_prefix = generate_stage2_filename(config, path_name)
        
        # 결과 저장 (model_config_utils.save_stage2_results 사용)
        save_path = save_dir / f"{filename_prefix}_results.csv"
        save_stage2_results(results, save_path)
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ 완료! (소요 시간: {elapsed_time:.1f}초)")
        print(f"   LL: {results['log_likelihood']:.2f}")
        print(f"   AIC: {results['aic']:.2f}")
        print(f"   BIC: {results['bic']:.2f}")
        print(f"   저장: {save_path.name}")
        
        return {
            'case_name': case.get('name', 'Unnamed'),
            'path_name': path_name,
            'main_lvs': case['main_lvs'],
            'interactions': case['interactions'],
            'log_likelihood': results['log_likelihood'],
            'aic': results['aic'],
            'bic': results['bic'],
            'elapsed_time': elapsed_time,
            'status': 'success',
            'error': None
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ 실패! (소요 시간: {elapsed_time:.1f}초)")
        print(f"   오류: {str(e)}")
        
        return {
            'case_name': case.get('name', 'Unnamed'),
            'path_name': path_name,
            'main_lvs': case['main_lvs'],
            'interactions': case['interactions'],
            'log_likelihood': None,
            'aic': None,
            'bic': None,
            'elapsed_time': elapsed_time,
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='모든 순차추정 케이스 자동 실행')
    parser.add_argument('--path', type=str, default='all', choices=['2', '3', 'all'],
                       help='실행할 경로: 2 (2경로), 3 (3경로), all (모두)')
    parser.add_argument('--mode', type=str, default='theory', choices=['theory', 'all'],
                       help='케이스 생성 모드: theory (이론 기반), all (전체 유효 케이스)')
    
    args = parser.parse_args()
    
    # 실행할 경로 결정
    if args.path == 'all':
        paths_to_run = ['2path', '3path']
    else:
        paths_to_run = [f'{args.path}path']
    
    # 케이스 생성
    if args.mode == 'theory':
        cases = generate_theory_driven_cases()
        print(f"이론 기반 케이스: {len(cases)}개")
    else:
        cases = generate_all_valid_cases()
        print(f"전체 유효 케이스: {len(cases)}개")
    
    # 데이터 로드
    data_path = project_root / "data" / "processed" / "iclv" / "integrated_data.csv"
    data = pd.read_csv(data_path)
    
    # 전체 결과 저장
    all_results = []
    
    # 각 경로에 대해 실행
    for path_name in paths_to_run:
        print(f"\n{'='*70}")
        print(f"경로: {path_name}")
        print(f"{'='*70}")
        
        # 1단계 결과 파일 경로
        stage1_result_path = project_root / "results" / "final" / "sequential" / path_name / "stage1" / f"stage1_{path_name}_results.pkl"
        
        if not stage1_result_path.exists():
            print(f"❌ 1단계 결과 파일이 없습니다: {stage1_result_path}")
            print(f"   먼저 1단계를 실행하세요!")
            continue
        
        # 저장 디렉토리
        save_dir = project_root / "results" / "final" / "sequential" / path_name / "stage2"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 케이스 실행
        for i, case in enumerate(cases, 1):
            print(f"\n진행: {i}/{len(cases)}")
            result = run_single_case(path_name, case, data, stage1_result_path, save_dir)
            all_results.append(result)
    
    # 전체 결과 요약 저장
    summary_path = project_root / "results" / "final" / "sequential" / f"all_cases_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print("전체 실행 완료!")
    print(f"{'='*70}")
    print(f"총 케이스 수: {len(all_results)}")
    print(f"성공: {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"실패: {sum(1 for r in all_results if r['status'] == 'failed')}")
    print(f"\n요약 파일: {summary_path}")


if __name__ == "__main__":
    main()

