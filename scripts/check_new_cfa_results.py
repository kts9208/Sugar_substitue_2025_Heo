"""
새로 추정된 CFA 결과 확인 (절편 포함 여부)
"""
import pandas as pd
import pickle
from pathlib import Path

print("="*80)
print("새로 추정된 CFA 결과 확인")
print("="*80)

# 1. all_params.csv 확인
print(f"\n{'='*80}")
print(f"[1] cfa_results_all_params.csv")
print(f"{'='*80}")

all_params_path = Path('results/sequential_stage_wise/cfa_results_all_params.csv')
if all_params_path.exists():
    all_params = pd.read_csv(all_params_path)
    print(f"\n행 수: {len(all_params)}")
    print(f"컬럼: {all_params.columns.tolist()}")
    
    print(f"\n파라미터 타입 (op):")
    print(all_params['op'].value_counts())
    
    print(f"\n절편 (op == '1'):")
    intercepts = all_params[all_params['op'] == '1']
    print(f"개수: {len(intercepts)}")
    
    if len(intercepts) > 0:
        print(f"\n처음 20개:")
        print(intercepts[['lval', 'op', 'rval', 'Estimate']].head(20))
    else:
        print(f"❌ 절편 없음!")
else:
    print(f"❌ 파일 없음!")

# 2. measurement_params.csv 확인
print(f"\n{'='*80}")
print(f"[2] cfa_results_measurement_params.csv")
print(f"{'='*80}")

measurement_params_path = Path('results/sequential_stage_wise/cfa_results_measurement_params.csv')
if measurement_params_path.exists():
    measurement_params = pd.read_csv(measurement_params_path)
    print(f"\n행 수: {len(measurement_params)}")
    print(f"컬럼: {measurement_params.columns.tolist()}")
    
    if 'param_type' in measurement_params.columns:
        print(f"\n파라미터 타입 (param_type):")
        print(measurement_params['param_type'].value_counts())
        
        # 절편 확인
        intercepts = measurement_params[measurement_params['param_type'] == 'intercept']
        print(f"\n절편 (param_type == 'intercept'):")
        print(f"개수: {len(intercepts)}")
        
        if len(intercepts) > 0:
            print(f"\n처음 20개:")
            print(intercepts[['lval', 'op', 'rval', 'Estimate', 'param_type']].head(20))
else:
    print(f"❌ 파일 없음!")

# 3. pickle 파일 확인
print(f"\n{'='*80}")
print(f"[3] cfa_results.pkl")
print(f"{'='*80}")

pkl_path = Path('results/sequential_stage_wise/cfa_results.pkl')
if pkl_path.exists():
    with open(pkl_path, 'rb') as f:
        cfa_results = pickle.load(f)
    
    print(f"\nKeys: {list(cfa_results.keys())}")
    
    if 'params' in cfa_results:
        params = cfa_results['params']
        print(f"\n'params' 있음!")
        print(f"  타입: {type(params)}")
        if isinstance(params, pd.DataFrame):
            print(f"  행 수: {len(params)}")
            print(f"  파라미터 타입 (op): {params['op'].value_counts().to_dict()}")
            
            intercepts = params[params['op'] == '1']
            print(f"\n  절편 (op == '1'): {len(intercepts)}개")
    else:
        print(f"\n'params' 없음!")
else:
    print(f"❌ 파일 없음!")

# 4. 결론
print(f"\n{'='*80}")
print(f"[4] 결론")
print(f"{'='*80}")

if all_params_path.exists():
    all_params = pd.read_csv(all_params_path)
    intercepts = all_params[all_params['op'] == '1']
    
    if len(intercepts) > 0:
        print(f"\n✅ 절편이 저장되었습니다!")
        print(f"  개수: {len(intercepts)}")
        print(f"  파일: cfa_results_all_params.csv")
        print(f"  파일: cfa_results_measurement_params.csv")
        print(f"  파일: cfa_results.pkl")
    else:
        print(f"\n❌ 절편이 없습니다!")
        print(f"\nsemopy가 절편을 추정하지 않았습니다.")
        print(f"이는 잠재변수가 표준화되어 E[LV] = 0이기 때문입니다.")
        print(f"\n해결 방안:")
        print(f"  1. 각 지표의 평균을 절편으로 사용")
        print(f"  2. 측정모델 우도 계산시 절편 포함")

print(f"\n{'='*80}")

