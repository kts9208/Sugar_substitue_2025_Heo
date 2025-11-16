"""
잠재변수 효과 없이 선택모델만 추정

목적: 선택 속성(health_label, price)만으로 선택모델 추정
      잠재변수(PI, NK) 효과는 제외

효용함수:
    V_일반당 = ASC_sugar + β_health_label × health_label + β_price × price
    V_무설탕 = ASC_sugar_free + β_health_label × health_label + β_price × price
    V_구매안함 = 0 (reference)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice

print("=" * 100)
print("선택모델 단독 추정 (잠재변수 효과 제외)")
print("=" * 100)
print()

# 1. 데이터 로드
print("[1] 데이터 로드 중...")
data_path = project_root / "data" / "processed" / "iclv" / "integrated_data_cleaned.csv"
data = pd.read_csv(data_path)
print(f"✅ 데이터 로드 완료: {len(data)} 행")
print()

# 2. 선택모델 설정 (잠재변수 효과 제외)
print("[2] 선택모델 설정...")
print("  - 선택 속성: health_label, price")
print("  - 잠재변수 효과: 없음 (main_lvs=[])")
print()

config = ChoiceConfig(
    choice_attributes=['health_label', 'price'],
    choice_type='binary',
    price_variable='price',
    all_lvs_as_main=True,
    main_lvs=[],  # ✅ 빈 리스트 = 잠재변수 효과 없음
    moderation_enabled=False
)

# 3. 선택모델 생성
print("[3] 선택모델 생성...")
choice_model = MultinomialLogitChoice(config)
print("✅ MultinomialLogitChoice 생성 완료")
print()

# 4. 선택모델 추정 (빈 요인점수 전달)
print("[4] 선택모델 추정 시작...")
print("  - 효용함수:")
print("    V_일반당 = ASC_sugar + β_health_label × health_label + β_price × price")
print("    V_무설탕 = ASC_sugar_free + β_health_label × health_label + β_price × price")
print("    V_구매안함 = 0 (reference)")
print()

# 빈 요인점수 딕셔너리 (잠재변수 없음)
factor_scores = {}

results = choice_model.fit(data, factor_scores)

# 5. 결과 출력
print()
print("=" * 100)
print("추정 결과")
print("=" * 100)
print()

print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ 파라미터 추정치                                                              │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()

if 'parameter_statistics' in results and results['parameter_statistics'] is not None:
    stats = results['parameter_statistics']
    if isinstance(stats, pd.DataFrame):
        print(stats.to_string())
    elif isinstance(stats, dict):
        # 딕셔너리를 DataFrame으로 변환
        stats_df = pd.DataFrame(stats).T
        print(stats_df.to_string())
    else:
        print(stats)
else:
    params = results['params']
    for param_name, value in params.items():
        if isinstance(value, np.ndarray):
            for i, v in enumerate(value):
                print(f"  {param_name}[{i}]: {v:.4f}")
        else:
            print(f"  {param_name}: {value:.4f}")

print()
print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ 모델 적합도                                                                  │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()
print(f"  로그우도 (LL):  {results['log_likelihood']:.2f}")
print(f"  AIC:            {results['aic']:.2f}")
print(f"  BIC:            {results['bic']:.2f}")
print()

# 6. 결과 저장
print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ 결과 저장                                                                    │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()

output_dir = project_root / "results" / "choice_model_only"
output_dir.mkdir(parents=True, exist_ok=True)

# 파라미터 통계 저장
if 'parameter_statistics' in results and results['parameter_statistics'] is not None:
    stats_path = output_dir / "parameter_statistics.csv"
    stats = results['parameter_statistics']

    # 딕셔너리를 DataFrame으로 변환 (beta 파라미터 처리)
    rows = []
    for param_name, param_stats in stats.items():
        if param_name == 'beta' and isinstance(param_stats, dict):
            # beta는 속성별로 분리
            for attr_name, attr_stats in param_stats.items():
                row = {
                    'parameter': f'beta_{attr_name}',
                    'estimate': attr_stats['estimate'],
                    'se': attr_stats['se'],
                    't': attr_stats['t'],
                    'p': attr_stats['p']
                }
                rows.append(row)
        else:
            # 다른 파라미터는 그대로
            row = {
                'parameter': param_name,
                'estimate': param_stats['estimate'],
                'se': param_stats['se'],
                't': param_stats['t'],
                'p': param_stats['p']
            }
            rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(stats_path, index=False)
    print(f"✅ 파라미터 통계: {stats_path}")

# 요약 저장
summary_path = output_dir / "model_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("선택모델 단독 추정 결과 (잠재변수 효과 제외)\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("효용함수:\n")
    f.write("  V_일반당 = ASC_sugar + β_health_label × health_label + β_price × price\n")
    f.write("  V_무설탕 = ASC_sugar_free + β_health_label × health_label + β_price × price\n")
    f.write("  V_구매안함 = 0 (reference)\n\n")
    
    f.write("모델 적합도:\n")
    f.write(f"  로그우도 (LL): {results['log_likelihood']:.2f}\n")
    f.write(f"  AIC:           {results['aic']:.2f}\n")
    f.write(f"  BIC:           {results['bic']:.2f}\n\n")
    
    if 'parameter_statistics' in results and results['parameter_statistics'] is not None:
        f.write("파라미터 추정치:\n\n")
        stats = results['parameter_statistics']

        # ASC 파라미터
        for param in ['asc_sugar', 'asc_sugar_free']:
            if param in stats:
                s = stats[param]
                f.write(f"  {param:20s} {s['estimate']:8.4f} (SE={s['se']:6.4f}, t={s['t']:6.3f}, p={s['p']:7.5f})\n")

        # beta 파라미터
        if 'beta' in stats:
            for attr_name, s in stats['beta'].items():
                param_name = f"beta_{attr_name}"
                f.write(f"  {param_name:20s} {s['estimate']:8.4f} (SE={s['se']:6.4f}, t={s['t']:6.3f}, p={s['p']:7.5f})\n")

        f.write("\n")

print(f"✅ 요약: {summary_path}")
print()

print("=" * 100)
print("완료!")
print("=" * 100)

