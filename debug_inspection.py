#!/usr/bin/env python3
"""
semopy inspection 데이터 구조 확인 스크립트
"""

from path_analysis import (
    PathAnalyzer,
    analyze_path_model,
    create_path_model,
    create_default_path_config
)

def debug_inspection():
    """inspection 데이터 구조 확인"""
    print("🔍 semopy inspection 데이터 구조 확인")
    print("=" * 50)
    
    try:
        # 간단한 경로분석 모델 생성
        variables = ['health_concern', 'perceived_benefit', 'purchase_intention']
        
        model_spec = create_path_model(
            model_type='simple_mediation',
            independent_var='health_concern',
            mediator_var='perceived_benefit',
            dependent_var='purchase_intention'
        )
        
        print("✅ 모델 스펙 생성 완료")
        
        # 모델 분석 실행
        config = create_default_path_config(verbose=False)
        results = analyze_path_model(model_spec, variables, config)
        
        print("✅ 분석 완료")
        
        # semopy 모델 객체 확인
        if 'model_object' not in results:
            print("❌ semopy 모델 객체를 찾을 수 없습니다.")
            return
        
        model = results['model_object']
        print("✅ semopy 모델 객체 확인 완료")
        
        # inspection 데이터 구조 확인
        inspection = model.inspect()
        
        print(f"\n📊 Inspection 데이터 구조:")
        print(f"- 행 수: {len(inspection)}")
        print(f"- 열 수: {len(inspection.columns)}")
        print(f"- 컬럼: {inspection.columns.tolist()}")
        
        print(f"\n📋 첫 5행:")
        print(inspection.head())
        
        print(f"\n🔧 연산자 종류:")
        print(inspection['op'].unique())
        
        print(f"\n📈 연산자별 개수:")
        print(inspection['op'].value_counts())
        
        # 잠재변수와 관측변수 구분
        latent_vars = set()
        observed_vars = set()
        
        for _, row in inspection.iterrows():
            lval = row['lval']
            rval = row['rval']
            op = row['op']
            
            if op == '=~':  # 측정모델 (요인적재량)
                latent_vars.add(lval)
                observed_vars.add(rval)
            elif op == '~':  # 구조모델 (경로계수)
                # 일단 모든 변수를 잠재변수로 간주
                latent_vars.add(lval)
                latent_vars.add(rval)
        
        # 관측변수는 잠재변수에서 제외
        latent_vars = latent_vars - observed_vars
        
        print(f"\n🎯 변수 분류:")
        print(f"- 잠재변수: {sorted(latent_vars)}")
        print(f"- 관측변수: {sorted(observed_vars)}")
        
        # 구조적 경로계수 필터링
        structural_paths = inspection[
            (inspection['op'] == '~') &  # 회귀 관계
            (inspection['lval'].isin(latent_vars)) &  # 종속변수가 잠재변수
            (inspection['rval'].isin(latent_vars))    # 독립변수가 잠재변수
        ].copy()
        
        print(f"\n🔗 구조적 경로계수:")
        print(f"- 개수: {len(structural_paths)}")
        if len(structural_paths) > 0:
            print(structural_paths[['lval', 'op', 'rval', 'Estimate']])
        
        # 요인적재량
        factor_loadings = inspection[
            inspection['op'] == '=~'
        ].copy()
        
        print(f"\n📊 요인적재량:")
        print(f"- 개수: {len(factor_loadings)}")
        if len(factor_loadings) > 0:
            print(factor_loadings[['lval', 'op', 'rval', 'Estimate']])
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_inspection()
