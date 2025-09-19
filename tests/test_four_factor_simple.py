"""
4가지 요인 간단 테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from semopy import Model

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_test_data():
    """데이터 로드 및 간단한 모델 테스트"""
    try:
        # 데이터 로드
        data_dir = Path("processed_data/survey_data")
        
        factor_files = {
            'health_concern': 'health_concern.csv',
            'perceived_benefit': 'perceived_benefit.csv',
            'perceived_price': 'perceived_price.csv',
            'nutrition_knowledge': 'nutrition_knowledge.csv'
        }
        
        # 각 요인 데이터 로드 및 병합
        data_frames = []
        
        for factor_name, filename in factor_files.items():
            file_path = data_dir / filename
            if not file_path.exists():
                logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
                continue
                
            factor_data = pd.read_csv(file_path)
            
            # 요인 점수 계산 (평균)
            numeric_cols = factor_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                factor_score = factor_data[numeric_cols].mean(axis=1)
            else:
                factor_score = factor_data.iloc[:, -1]
            
            factor_df = pd.DataFrame({factor_name: factor_score})
            data_frames.append(factor_df)
        
        # 모든 요인 데이터 병합
        data = pd.concat(data_frames, axis=1)
        data = data.dropna()
        
        print(f"데이터 크기: {data.shape}")
        print(f"변수: {list(data.columns)}")
        print(f"데이터 미리보기:")
        print(data.head())
        print(f"기술통계:")
        print(data.describe())
        
        # 간단한 모델 테스트
        print("\n=== 간단한 모델 테스트 ===")
        
        # 1. 단순 회귀 모델
        simple_model_spec = """
        perceived_benefit ~ health_concern
        """
        
        print("1. 단순 회귀 모델:")
        print(simple_model_spec)
        
        model = Model(simple_model_spec)
        results = model.fit(data)
        
        print("모델 추정 성공!")
        
        # 결과 확인
        params = model.inspect()
        print("파라미터:")
        print(params)
        
        # 2. 다중 회귀 모델
        print("\n2. 다중 회귀 모델:")
        multi_model_spec = """
        perceived_benefit ~ health_concern + nutrition_knowledge
        perceived_price ~ health_concern + nutrition_knowledge
        """
        
        print(multi_model_spec)
        
        model2 = Model(multi_model_spec)
        results2 = model2.fit(data)
        
        print("다중 회귀 모델 추정 성공!")
        
        params2 = model2.inspect()
        print("파라미터:")
        print(params2[params2['op'] == '~'])
        
        # 3. 포화 모델 (모든 경로)
        print("\n3. 포화 모델:")
        saturated_spec = """
        perceived_benefit ~ health_concern + perceived_price + nutrition_knowledge
        perceived_price ~ health_concern + perceived_benefit + nutrition_knowledge  
        nutrition_knowledge ~ health_concern + perceived_benefit + perceived_price
        health_concern ~ perceived_benefit + perceived_price + nutrition_knowledge
        """
        
        print(saturated_spec)
        
        model3 = Model(saturated_spec)
        results3 = model3.fit(data)
        
        print("포화 모델 추정 성공!")
        
        params3 = model3.inspect()
        structural_paths = params3[params3['op'] == '~']
        print("구조적 경로:")
        print(structural_paths[['lval', 'rval', 'Estimate', 'p-value']])
        
        # 유의한 경로 확인
        significant_paths = structural_paths[structural_paths['p-value'] < 0.05]
        print(f"\n유의한 경로 (p < 0.05): {len(significant_paths)}개")
        if len(significant_paths) > 0:
            print(significant_paths[['lval', 'rval', 'Estimate', 'p-value']])
        
        return data, model3, params3
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    data, model, params = load_and_test_data()
    
    if data is not None:
        print("\n✅ 테스트 성공!")
    else:
        print("\n❌ 테스트 실패!")
