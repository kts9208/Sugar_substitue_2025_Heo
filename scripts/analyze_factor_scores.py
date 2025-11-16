"""
잠재변수 점수 (Factor Scores) 통계 분석

저장된 CSV 파일에서 잠재변수 점수의 평균과 분산을 분석합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 100)
print('잠재변수 점수 (Factor Scores) 통계 분석')
print('=' * 100)
print()

# 최신 파일 찾기
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_*.csv'))

# 각 단계별 최신 파일 찾기
stages = {
    'SEM_추출_직후': None,
    '선택모델_전달_직전': None,
    '선택모델_확장_전': None,
    '선택모델_확장_후': None
}

for stage in stages.keys():
    stage_files = [f for f in files if stage in f.name]
    if stage_files:
        stages[stage] = stage_files[-1]  # 최신 파일

print('분석 대상 파일:')
for stage, file in stages.items():
    if file:
        print(f'   - {stage}: {file.name}')
print()

# 각 단계별 분석
for stage_name, file_path in stages.items():
    if not file_path:
        continue
    
    print('=' * 100)
    print(f'단계: {stage_name}')
    print('=' * 100)
    
    df = pd.read_csv(file_path)
    
    print(f'\n데이터 기본 정보:')
    print(f'   - 행 수: {len(df):,}')
    print(f'   - 열 수: {len(df.columns)}')
    print(f'   - 잠재변수: {list(df.columns)}')
    print()
    
    print(f'잠재변수별 통계량:')
    print()
    
    for col in df.columns:
        values = df[col].values
        
        print(f'   [{col}]')
        print(f'      평균 (Mean):        {np.mean(values):>10.6f}')
        print(f'      분산 (Variance):    {np.var(values, ddof=0):>10.6f}  (모분산)')
        print(f'      분산 (Variance):    {np.var(values, ddof=1):>10.6f}  (표본분산)')
        print(f'      표준편차 (Std):     {np.std(values, ddof=0):>10.6f}  (모표준편차)')
        print(f'      표준편차 (Std):     {np.std(values, ddof=1):>10.6f}  (표본표준편차)')
        print(f'      최소값 (Min):       {np.min(values):>10.6f}')
        print(f'      최대값 (Max):       {np.max(values):>10.6f}')
        print(f'      중앙값 (Median):    {np.median(values):>10.6f}')
        print(f'      왜도 (Skewness):    {pd.Series(values).skew():>10.6f}')
        print(f'      첨도 (Kurtosis):    {pd.Series(values).kurtosis():>10.6f}')
        print()
    
    print()

print('=' * 100)
print('분석 완료!')
print('=' * 100)
print()

# 요약 테이블 생성
print('=' * 100)
print('요약: SEM 추출 직후 (개인 수준) 통계량')
print('=' * 100)
print()

if stages['SEM_추출_직후']:
    df_sem = pd.read_csv(stages['SEM_추출_직후'])
    
    summary_data = []
    for col in df_sem.columns:
        values = df_sem[col].values
        summary_data.append({
            '잠재변수': col,
            '평균': np.mean(values),
            '분산': np.var(values, ddof=1),
            '표준편차': np.std(values, ddof=1),
            '최소값': np.min(values),
            '최대값': np.max(values)
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()
    
    # CSV로 저장
    summary_df.to_csv('logs/factor_scores_summary.csv', index=False, encoding='utf-8-sig')
    print('요약 테이블 저장: logs/factor_scores_summary.csv')
    print()

print('=' * 100)
print('이론적 기대값과 비교')
print('=' * 100)
print()
print('SEM 요인점수의 이론적 특성:')
print('   - 평균: 0에 가까워야 함 (중심화됨)')
print('   - 분산: 1보다 작음 (일반적으로 0.5-0.9 범위)')
print('   - 이유: 측정오차를 고려한 추정치이므로 관측변수보다 분산이 작음')
print()

if stages['SEM_추출_직후']:
    df_sem = pd.read_csv(stages['SEM_추출_직후'])
    
    print('실제 관측값:')
    for col in df_sem.columns:
        values = df_sem[col].values
        mean = np.mean(values)
        var = np.var(values, ddof=1)
        
        mean_check = '✓' if abs(mean) < 0.1 else '✗'
        var_check = '✓' if 0.3 < var < 1.2 else '?'
        
        print(f'   [{col}]')
        print(f'      평균: {mean:>8.4f}  {mean_check} (|평균| < 0.1)')
        print(f'      분산: {var:>8.4f}  {var_check} (0.3 < 분산 < 1.2)')
        print()

print('=' * 100)

