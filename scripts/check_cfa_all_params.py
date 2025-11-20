"""
CFA 결과의 모든 파라미터 확인 (절편 포함)
"""
import pickle
import pandas as pd
from pathlib import Path

print("="*80)
print("CFA 결과의 모든 파라미터 확인")
print("="*80)

# CFA 결과 로드
cfa_path = Path('results/sequential_stage_wise/cfa_results.pkl')
with open(cfa_path, 'rb') as f:
    cfa_results = pickle.load(f)

print(f"\n{'='*80}")
print(f"[1] CFA 결과 구조")
print(f"{'='*80}")
print(f"\nKeys: {list(cfa_results.keys())}")

# params 확인
if 'params' in cfa_results:
    params = cfa_results['params']
    print(f"\n{'='*80}")
    print(f"[2] params DataFrame")
    print(f"{'='*80}")
    print(f"\n컬럼: {params.columns.tolist()}")
    print(f"행 수: {len(params)}")
    
    print(f"\n{'='*80}")
    print(f"[3] 파라미터 타입 (op)")
    print(f"{'='*80}")
    print(params['op'].value_counts())
    
    print(f"\n{'='*80}")
    print(f"[4] 절편 확인")
    print(f"{'='*80}")
    
    # op == '1' 확인
    intercepts_1 = params[params['op'] == '1']
    print(f"\nop == '1': {len(intercepts_1)}개")
    if len(intercepts_1) > 0:
        print(intercepts_1[['lval', 'op', 'rval', 'Estimate']].head(20))
    
    # op == '~1' 확인
    intercepts_tilde1 = params[params['op'] == '~1']
    print(f"\nop == '~1': {len(intercepts_tilde1)}개")
    if len(intercepts_tilde1) > 0:
        print(intercepts_tilde1[['lval', 'op', 'rval', 'Estimate']].head(20))
    
    # rval == '1' 확인
    intercepts_rval = params[params['rval'] == '1']
    print(f"\nrval == '1': {len(intercepts_rval)}개")
    if len(intercepts_rval) > 0:
        print(intercepts_rval[['lval', 'op', 'rval', 'Estimate']].head(20))
    
    print(f"\n{'='*80}")
    print(f"[5] 전체 파라미터 (처음 50개)")
    print(f"{'='*80}")
    print(params[['lval', 'op', 'rval', 'Estimate']].head(50))
    
else:
    print(f"\n❌ 'params' 키가 없습니다!")

# loadings 확인
if 'loadings' in cfa_results:
    loadings = cfa_results['loadings']
    print(f"\n{'='*80}")
    print(f"[6] loadings DataFrame")
    print(f"{'='*80}")
    print(f"\n컬럼: {loadings.columns.tolist()}")
    print(f"행 수: {len(loadings)}")
    print(f"\n파라미터 타입 (op): {loadings['op'].value_counts().to_dict()}")

print(f"\n{'='*80}")
print(f"[7] semopy inspect() 메서드 확인")
print(f"{'='*80}")

# semopy 모델 객체 확인
if 'model' in cfa_results:
    model = cfa_results['model']
    print(f"\n모델 타입: {type(model)}")
    
    # inspect() 재실행
    try:
        all_params = model.inspect(std_est=True)
        print(f"\ninspect() 결과:")
        print(f"  행 수: {len(all_params)}")
        print(f"  컬럼: {all_params.columns.tolist()}")
        print(f"\n파라미터 타입 (op):")
        print(all_params['op'].value_counts())
        
        print(f"\n전체 파라미터 (처음 60개):")
        print(all_params[['lval', 'op', 'rval', 'Estimate']].head(60))
        
    except Exception as e:
        print(f"\n❌ inspect() 실행 실패: {e}")
else:
    print(f"\n❌ 'model' 키가 없습니다!")

print(f"\n{'='*80}")
print(f"[8] 결론")
print(f"{'='*80}")

print(f"""
semopy의 inspect() 메서드는 다음 파라미터를 반환합니다:
  - op == '~': 회귀계수 (요인적재량, 경로계수)
  - op == '~~': 공분산/분산
  - op == '1': 절편 (있다면)

현재 CFA 결과에 절편이 없다면:
  1. semopy가 절편을 추정하지 않았거나
  2. 잠재변수 표준화로 인해 절편이 0이 되어 저장되지 않았을 수 있습니다.
""")

