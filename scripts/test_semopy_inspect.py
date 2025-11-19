"""
semopy의 inspect 메서드가 반환하는 파라미터 확인
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# 간단한 SEM 모델 생성 및 추정
from src.analysis.factor_analysis.factor_analyzer import SemopyAnalyzer

# 데이터 로드
data_path = Path('data/processed/iclv/integrated_data_cleaned.csv')
data = pd.read_csv(data_path)

# 간단한 CFA 모델 스펙
model_spec = 'health_concern =~ q6 + q7 + q8 + q9 + q10 + q11'

analyzer = SemopyAnalyzer()

# 개인별 첫 번째 행만 선택
print(f'데이터 컬럼: {list(data.columns[:10])}...')
id_col = 'respondent_id'
unique_data = data.groupby(id_col)[['q6', 'q7', 'q8', 'q9', 'q10', 'q11']].first().reset_index()

print(f'데이터 shape: {unique_data.shape}')
print(f'모델 스펙: {model_spec}')

# 모델 추정
results = analyzer.fit_model(unique_data, model_spec)

# 파라미터 확인
print('\n=== inspect() 결과 ===')
params = analyzer.model.inspect(std_est=True)
print(f'파라미터 개수: {len(params)}')
print(f'컬럼: {list(params.columns)}')
print(f'op 컬럼 고유값: {params["op"].unique()}')

print('\n=== 전체 파라미터 ===')
print(params.to_string())

# ~~ 연산자 확인
variance_params = params[params['op'] == '~~']
print(f'\n=== ~~ 연산자 파라미터 ({len(variance_params)}개) ===')
if len(variance_params) > 0:
    print(variance_params.to_string())
else:
    print('없음')

