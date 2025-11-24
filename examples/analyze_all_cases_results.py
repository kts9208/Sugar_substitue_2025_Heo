"""
전체 케이스 결과 분석

250개 케이스의 결과를 분석하고 시각화합니다.

사용법:
    python examples/analyze_all_cases_results.py

Author: ICLV Team
Date: 2025-11-23
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_latest_summary():
    """가장 최근 요약 파일 로드"""
    summary_dir = project_root / "results" / "final" / "sequential"
    summary_files = list(summary_dir.glob("all_cases_summary_*.csv"))
    
    if not summary_files:
        raise FileNotFoundError("요약 파일을 찾을 수 없습니다.")
    
    # 가장 최근 파일
    latest_file = max(summary_files, key=lambda p: p.stat().st_mtime)
    print(f"로드: {latest_file.name}")
    
    return pd.read_csv(latest_file)


def parse_model_components(df):
    """모델 구성 요소 파싱"""
    df = df.copy()
    
    # 주효과 개수
    df['n_main_lvs'] = df['main_lvs'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)
    
    # 상호작용 개수
    df['n_interactions'] = df['interactions'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)
    
    # 총 파라미터 개수 (대략)
    # Base: 4 (2 ASC + 2 beta)
    # 주효과: +2 per LV (sugar, sugar_free)
    # 상호작용: +2 per interaction
    df['n_params'] = 4 + df['n_main_lvs'] * 2 + df['n_interactions'] * 2
    
    return df


def analyze_by_path(df):
    """경로별 분석"""
    print("\n" + "="*70)
    print("경로별 분석")
    print("="*70)
    
    for path_name in ['2path', '3path']:
        path_df = df[df['path_name'] == path_name]
        print(f"\n[{path_name}]")
        print(f"  케이스 수: {len(path_df)}")
        print(f"  평균 LL: {path_df['log_likelihood'].mean():.2f}")
        print(f"  평균 AIC: {path_df['aic'].mean():.2f}")
        print(f"  평균 BIC: {path_df['bic'].mean():.2f}")
        print(f"  최소 AIC: {path_df['aic'].min():.2f}")
        print(f"  최소 BIC: {path_df['bic'].min():.2f}")


def analyze_by_complexity(df):
    """모델 복잡도별 분석"""
    print("\n" + "="*70)
    print("모델 복잡도별 분석")
    print("="*70)
    
    # 주효과 개수별
    print("\n[주효과 개수별]")
    for n in sorted(df['n_main_lvs'].unique()):
        subset = df[df['n_main_lvs'] == n]
        print(f"  {n}개: {len(subset)}케이스, 평균 AIC={subset['aic'].mean():.2f}, 최소 AIC={subset['aic'].min():.2f}")
    
    # 상호작용 개수별
    print("\n[상호작용 개수별]")
    for n in sorted(df['n_interactions'].unique()):
        subset = df[df['n_interactions'] == n]
        print(f"  {n}개: {len(subset)}케이스, 평균 AIC={subset['aic'].mean():.2f}, 최소 AIC={subset['aic'].min():.2f}")


def find_best_models(df, criterion='aic', top_n=20):
    """최적 모델 찾기"""
    print("\n" + "="*70)
    print(f"{criterion.upper()} 기준 상위 {top_n}개 모델")
    print("="*70)
    
    # 성공한 케이스만
    success_df = df[df['status'] == 'success'].copy()
    
    # 정렬
    top_models = success_df.nsmallest(top_n, criterion)
    
    # 출력
    for i, row in enumerate(top_models.itertuples(), 1):
        print(f"\n{i}. [{row.path_name}] LL={row.log_likelihood:.2f}, AIC={row.aic:.2f}, BIC={row.bic:.2f}")
        
        # 주효과
        main_lvs = eval(row.main_lvs) if pd.notna(row.main_lvs) else []
        if main_lvs:
            lv_abbr_map = {
                'purchase_intention': 'PI',
                'nutrition_knowledge': 'NK',
                'perceived_price': 'PP'
            }
            main_abbrs = [lv_abbr_map.get(lv, lv) for lv in main_lvs]
            print(f"   주효과: {', '.join(main_abbrs)}")
        else:
            print(f"   주효과: 없음 (Base Model)")
        
        # 상호작용
        interactions = eval(row.interactions) if pd.notna(row.interactions) else []
        if interactions:
            int_strs = []
            for lv, attr in interactions:
                lv_abbr = {'purchase_intention': 'PI', 'nutrition_knowledge': 'NK', 'perceived_price': 'PP'}.get(lv, lv)
                attr_abbr = {'health_label': 'label', 'price': 'price'}.get(attr, attr)
                int_strs.append(f"{lv_abbr}×{attr_abbr}")
            print(f"   상호작용: {', '.join(int_strs)}")
    
    return top_models


def compare_2path_vs_3path(df):
    """2경로 vs 3경로 비교"""
    print("\n" + "="*70)
    print("2경로 vs 3경로 비교 (동일 선택모델)")
    print("="*70)
    
    # 2path와 3path에서 동일한 선택모델 찾기
    path2 = df[df['path_name'] == '2path'].copy()
    path3 = df[df['path_name'] == '3path'].copy()
    
    # 선택모델 식별자 생성
    path2['choice_model_id'] = path2['main_lvs'] + '|' + path2['interactions']
    path3['choice_model_id'] = path3['main_lvs'] + '|' + path3['interactions']
    
    # 공통 선택모델
    common_ids = set(path2['choice_model_id']) & set(path3['choice_model_id'])
    
    print(f"\n공통 선택모델 개수: {len(common_ids)}")
    
    # AIC 차이 분석
    aic_diffs = []
    for choice_id in common_ids:
        aic_2path = path2[path2['choice_model_id'] == choice_id]['aic'].values[0]
        aic_3path = path3[path3['choice_model_id'] == choice_id]['aic'].values[0]
        aic_diffs.append(aic_2path - aic_3path)
    
    aic_diffs = np.array(aic_diffs)
    
    print(f"\nAIC 차이 (2path - 3path):")
    print(f"  평균: {aic_diffs.mean():.2f}")
    print(f"  표준편차: {aic_diffs.std():.2f}")
    print(f"  최소: {aic_diffs.min():.2f}")
    print(f"  최대: {aic_diffs.max():.2f}")
    print(f"  2path가 더 좋은 경우: {(aic_diffs < 0).sum()}개")
    print(f"  3path가 더 좋은 경우: {(aic_diffs > 0).sum()}개")


def main():
    print("="*70)
    print("전체 케이스 결과 분석")
    print("="*70)
    
    # 데이터 로드
    df = load_latest_summary()
    print(f"\n총 케이스: {len(df)}")
    print(f"성공: {(df['status'] == 'success').sum()}")
    print(f"실패: {(df['status'] == 'failed').sum()}")
    
    # 모델 구성 요소 파싱
    df = parse_model_components(df)
    
    # 분석
    analyze_by_path(df)
    analyze_by_complexity(df)
    find_best_models(df, criterion='aic', top_n=20)
    find_best_models(df, criterion='bic', top_n=20)
    compare_2path_vs_3path(df)
    
    # 결과 저장
    output_path = project_root / "results" / "final" / "sequential" / "analysis_summary.txt"
    print(f"\n분석 완료! (결과는 콘솔에만 출력)")


if __name__ == "__main__":
    main()

