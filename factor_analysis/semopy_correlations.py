#!/usr/bin/env python3
"""
semopy를 이용한 요인간 상관계수 및 p값 추출 모듈

이 모듈은 다음 기능만 제공합니다:
1. 5개 요인 데이터 로드
2. semopy를 이용한 상관계수 및 p값 추출
3. 결과 저장

Author: Sugar Substitute Research Team
Date: 2025-01-06
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging

# semopy 라이브러리
try:
    import semopy
    from semopy import Model
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    print("Warning: semopy 라이브러리가 설치되지 않았습니다.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemopyCorrelationExtractor:
    """semopy를 이용한 요인간 상관계수 및 p값 추출기"""
    
    def __init__(self):
        """초기화"""
        if not SEMOPY_AVAILABLE:
            raise ImportError("semopy 라이브러리가 필요합니다. pip install semopy로 설치하세요.")
    
    def load_survey_data(self):
        """5개 요인의 설문 데이터 로드"""
        logger.info("설문 데이터 로드 중...")
        
        survey_data_dir = Path("processed_data/survey_data")
        
        # 5개 요인 파일 정의
        factor_files = {
            'health_concern': 'health_concern.csv',
            'perceived_benefit': 'perceived_benefit.csv', 
            'purchase_intention': 'purchase_intention.csv',
            'perceived_price': 'perceived_price.csv',
            'nutrition_knowledge': 'nutrition_knowledge.csv'
        }
        
        # 전체 데이터를 하나의 DataFrame으로 결합
        combined_data = pd.DataFrame()
        
        for factor_name, file_name in factor_files.items():
            file_path = survey_data_dir / file_name
            if file_path.exists():
                data = pd.read_csv(file_path)
                
                # 'no' 컬럼 제외하고 문항 컬럼만 추가
                item_columns = [col for col in data.columns if col != 'no']
                for col in item_columns:
                    combined_data[col] = data[col]
                
                logger.info(f"{factor_name} 데이터 로드: {data.shape}")
            else:
                logger.error(f"파일을 찾을 수 없습니다: {file_path}")
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        logger.info(f"전체 결합 데이터 크기: {combined_data.shape}")
        return combined_data
    
    def create_semopy_model(self):
        """semopy 모델 정의 (동적 생성)"""
        # 실제 데이터에서 사용 가능한 문항들을 확인
        survey_data_dir = Path("processed_data/survey_data")

        # 각 요인별 실제 문항 확인
        factor_items = {}

        factor_files = {
            'health_concern': 'health_concern.csv',
            'perceived_benefit': 'perceived_benefit.csv',
            'purchase_intention': 'purchase_intention.csv',
            'perceived_price': 'perceived_price.csv',
            'nutrition_knowledge': 'nutrition_knowledge.csv'
        }

        for factor_name, filename in factor_files.items():
            file_path = survey_data_dir / filename
            if file_path.exists():
                import pandas as pd
                data = pd.read_csv(file_path)
                items = [col for col in data.columns if col.startswith('q')]
                factor_items[factor_name] = items

        # 동적 모델 생성
        model_lines = []

        if 'health_concern' in factor_items:
            items = " + ".join(factor_items['health_concern'])
            model_lines.append(f"health_concern =~ {items}")

        if 'perceived_benefit' in factor_items:
            items = " + ".join(factor_items['perceived_benefit'])
            model_lines.append(f"perceived_benefit =~ {items}")

        if 'purchase_intention' in factor_items:
            items = " + ".join(factor_items['purchase_intention'])
            model_lines.append(f"purchase_intention =~ {items}")

        if 'perceived_price' in factor_items:
            items = " + ".join(factor_items['perceived_price'])
            model_lines.append(f"perceived_price =~ {items}")

        if 'nutrition_knowledge' in factor_items:
            items = " + ".join(factor_items['nutrition_knowledge'])
            model_lines.append(f"nutrition_knowledge =~ {items}")

        model_desc = "\n".join(model_lines)
        return model_desc
    
    def extract_correlations_and_pvalues(self, data):
        """semopy를 이용한 상관계수 및 p값 추출"""
        logger.info("semopy 모델 생성 및 적합 중...")
        
        # 결측치 제거
        clean_data = data.dropna()
        logger.info(f"사용된 관측치 수: {len(clean_data)}")
        
        # 모델 생성 및 적합
        model_desc = self.create_semopy_model()
        model = Model(model_desc)
        model.fit(clean_data)
        
        logger.info("모델 적합 완료")
        
        # 파라미터 추출
        params = model.inspect(std_est=True)
        
        # 요인간 공분산 파라미터 필터링
        factor_covs = params[params['op'] == '~~'].copy()
        
        # 요인 이름 정의
        factor_names = ['health_concern', 'perceived_benefit', 'purchase_intention', 
                       'perceived_price', 'nutrition_knowledge']
        
        # 요인간 상관계수만 필터링
        factor_correlations = factor_covs[
            (factor_covs['lval'].isin(factor_names)) &
            (factor_covs['rval'].isin(factor_names))
        ].copy()
        
        # 상관계수 매트릭스 생성
        correlation_matrix = self._build_correlation_matrix(factor_correlations, factor_names)
        
        # p값 매트릭스 생성
        p_value_matrix = self._build_p_value_matrix(factor_correlations, factor_names)
        
        logger.info(f"요인간 상관계수 추출 완료: {len(factor_names)}개 요인")
        
        return correlation_matrix, p_value_matrix
    
    def _build_correlation_matrix(self, factor_correlations, factor_names):
        """상관계수 매트릭스 구성"""
        correlation_matrix = pd.DataFrame(
            index=factor_names, 
            columns=factor_names,
            dtype=float
        )
        
        # 대각선 요소 (자기 자신과의 상관계수 = 1.0)
        for factor in factor_names:
            correlation_matrix.loc[factor, factor] = 1.0
        
        # 비대각선 요소 (요인간 상관계수)
        for _, row in factor_correlations.iterrows():
            lval, rval = row['lval'], row['rval']
            
            if lval != rval and lval in factor_names and rval in factor_names:
                # 표준화된 추정값 사용 (상관계수)
                corr_value = row['Est. Std']
                
                # 대칭 매트릭스
                correlation_matrix.loc[lval, rval] = corr_value
                correlation_matrix.loc[rval, lval] = corr_value
        
        return correlation_matrix
    
    def _build_p_value_matrix(self, factor_correlations, factor_names):
        """p값 매트릭스 구성"""
        p_value_matrix = pd.DataFrame(
            index=factor_names,
            columns=factor_names,
            dtype=float
        )
        
        # 대각선 요소 (자기 자신과의 상관계수 p값 = 0.0)
        for factor in factor_names:
            p_value_matrix.loc[factor, factor] = 0.0
        
        # 비대각선 요소 (요인간 상관계수의 p값)
        for _, row in factor_correlations.iterrows():
            lval, rval = row['lval'], row['rval']
            
            if lval != rval and lval in factor_names and rval in factor_names:
                # p값 추출
                p_value = row['p-value']
                
                # 대칭 매트릭스
                p_value_matrix.loc[lval, rval] = p_value
                p_value_matrix.loc[rval, lval] = p_value
        
        return p_value_matrix
    
    def save_results(self, correlation_matrix, p_value_matrix):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("factor_correlations_results")
        results_dir.mkdir(exist_ok=True)
        
        logger.info("결과 저장 중...")
        
        # 1. 상관계수 CSV 저장
        corr_file = results_dir / f"semopy_correlations_{timestamp}.csv"
        correlation_matrix.to_csv(corr_file, encoding='utf-8-sig')
        logger.info(f"상관계수 저장: {corr_file}")
        
        # 2. p값 CSV 저장
        pval_file = results_dir / f"semopy_pvalues_{timestamp}.csv"
        p_value_matrix.to_csv(pval_file, encoding='utf-8-sig')
        logger.info(f"p값 저장: {pval_file}")
        
        # 3. JSON 결과 저장
        json_data = {
            'timestamp': timestamp,
            'analysis_type': 'semopy_factor_correlations',
            'correlations': correlation_matrix.to_dict(),
            'p_values': p_value_matrix.to_dict(),
            'significant_correlations': self._identify_significant_correlations(
                correlation_matrix, p_value_matrix
            ),
            'summary_statistics': {
                'n_factors': len(correlation_matrix),
                'n_significant_correlations': self._count_significant_correlations(p_value_matrix),
                'max_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()),
                'min_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()),
                'mean_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean())
            }
        }
        
        json_file = results_dir / f"semopy_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON 결과 저장: {json_file}")
        
        return {
            'correlation_file': corr_file,
            'pvalue_file': pval_file,
            'json_file': json_file,
            'timestamp': timestamp
        }
    
    def _identify_significant_correlations(self, correlation_matrix, p_value_matrix):
        """유의한 상관관계 식별"""
        significant_pairs = []
        factor_names = correlation_matrix.index.tolist()
        
        for i in range(len(factor_names)):
            for j in range(i+1, len(factor_names)):
                factor1 = factor_names[i]
                factor2 = factor_names[j]
                corr_val = correlation_matrix.iloc[i, j]
                p_val = p_value_matrix.iloc[i, j]
                
                if p_val < 0.05:
                    significant_pairs.append({
                        'factor1': factor1,
                        'factor2': factor2,
                        'correlation': float(corr_val),
                        'p_value': float(p_val),
                        'significance_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                    })
        
        return significant_pairs
    
    def _count_significant_correlations(self, p_value_matrix):
        """유의한 상관관계 개수 계산"""
        upper_triangle = np.triu(p_value_matrix.values, k=1)
        return int((upper_triangle < 0.05).sum())
    
    def run_analysis(self):
        """전체 분석 실행"""
        logger.info("=== semopy 요인간 상관계수 분석 시작 ===")
        
        try:
            # 1. 데이터 로드
            data = self.load_survey_data()
            
            # 2. 상관계수 및 p값 추출
            correlation_matrix, p_value_matrix = self.extract_correlations_and_pvalues(data)
            
            # 3. 결과 저장
            file_info = self.save_results(correlation_matrix, p_value_matrix)
            
            # 4. 결과 요약 출력
            self._print_summary(correlation_matrix, p_value_matrix)
            
            logger.info("=== 분석 완료 ===")
            return file_info
            
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {e}")
            raise
    
    def _print_summary(self, correlation_matrix, p_value_matrix):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 semopy 요인간 상관계수 분석 결과")
        print("="*60)
        
        print(f"\n상관계수 매트릭스:")
        print(correlation_matrix.round(4))
        
        print(f"\np값 매트릭스:")
        print(p_value_matrix.round(4))
        
        # 유의한 상관관계 출력
        significant_pairs = self._identify_significant_correlations(correlation_matrix, p_value_matrix)
        
        print(f"\n📈 통계적으로 유의한 상관관계 (p < 0.05): {len(significant_pairs)}개")
        for pair in sorted(significant_pairs, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  {pair['factor1']} ↔ {pair['factor2']}: "
                  f"r = {pair['correlation']:+.4f} {pair['significance_level']} "
                  f"(p = {pair['p_value']:.6f})")
        
        print("\n유의수준: *** p<0.001, ** p<0.01, * p<0.05")


def main():
    """메인 실행 함수"""
    try:
        extractor = SemopyCorrelationExtractor()
        file_info = extractor.run_analysis()
        
        print(f"\n💾 저장된 파일:")
        print(f"  - 상관계수: {file_info['correlation_file']}")
        print(f"  - p값: {file_info['pvalue_file']}")
        print(f"  - JSON 결과: {file_info['json_file']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 분석이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 분석 중 오류가 발생했습니다.")
