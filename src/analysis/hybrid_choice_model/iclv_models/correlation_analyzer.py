"""
통합 상관관계 분석 모듈

1단계 SEM 변수와 2단계 선택모델 변수를 모두 포함하는 상관관계 분석

주요 기능:
1. 잠재변수 지표 간 상관관계 (1단계 SEM)
2. 잠재변수 간 상관관계 (1단계 SEM)
3. 선택모델 속성변수 간 상관관계 (2단계)
4. 사회인구통계변수 간 상관관계 (2단계)
5. 잠재변수-속성변수 간 상관관계 (1단계-2단계 연결)
6. 잠재변수-사회인구통계변수 간 상관관계 (1단계-2단계 연결)

Author: Sugar Substitute Research Team
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime
import json

try:
    from semopy import Model
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class IntegratedCorrelationAnalyzer:
    """
    통합 상관관계 분석기
    
    1단계 SEM과 2단계 선택모델의 모든 변수를 포함하는 상관관계 분석
    """
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def analyze_all_correlations(self,
                                 data: pd.DataFrame,
                                 measurement_model,
                                 structural_model,
                                 choice_config,
                                 factor_scores: Optional[Dict[str, np.ndarray]] = None,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        전체 상관관계 분석 실행
        
        Args:
            data: 통합 데이터 (SEM 지표 + 선택모델 변수 포함)
            measurement_model: 측정모델 객체
            structural_model: 구조모델 객체
            choice_config: 선택모델 설정
            factor_scores: 요인점수 (선택사항, 없으면 계산)
            save_path: 결과 저장 경로
            
        Returns:
            {
                'indicator_correlations': Dict,  # 지표 간 상관관계
                'latent_correlations': pd.DataFrame,  # 잠재변수 간 상관관계
                'attribute_correlations': pd.DataFrame,  # 속성변수 간 상관관계
                'sociodem_correlations': pd.DataFrame,  # 사회인구통계변수 간 상관관계
                'lv_attribute_correlations': pd.DataFrame,  # 잠재변수-속성변수 간
                'lv_sociodem_correlations': pd.DataFrame,  # 잠재변수-사회인구통계변수 간
                'full_correlation_matrix': pd.DataFrame,  # 전체 상관관계 행렬
                'summary': Dict  # 요약 통계
            }
        """
        self.logger.info("=== 통합 상관관계 분석 시작 ===")
        
        results = {}
        
        # 1. 잠재변수 지표 간 상관관계 (1단계 SEM)
        self.logger.info("\n[1] 잠재변수 지표 간 상관관계 분석...")
        results['indicator_correlations'] = self._analyze_indicator_correlations(
            data, measurement_model
        )
        
        # 2. 잠재변수 간 상관관계 (1단계 SEM)
        self.logger.info("\n[2] 잠재변수 간 상관관계 분석...")
        results['latent_correlations'] = self._analyze_latent_correlations(
            data, measurement_model, factor_scores
        )
        
        # 3. 선택모델 속성변수 간 상관관계 (2단계)
        self.logger.info("\n[3] 선택모델 속성변수 간 상관관계 분석...")
        results['attribute_correlations'] = self._analyze_attribute_correlations(
            data, choice_config
        )
        
        # 4. 사회인구통계변수 간 상관관계 (2단계)
        self.logger.info("\n[4] 사회인구통계변수 간 상관관계 분석...")
        results['sociodem_correlations'] = self._analyze_sociodem_correlations(
            data, structural_model
        )
        
        # 5. 잠재변수-속성변수 간 상관관계
        self.logger.info("\n[5] 잠재변수-속성변수 간 상관관계 분석...")
        results['lv_attribute_correlations'] = self._analyze_lv_attribute_correlations(
            data, measurement_model, choice_config, factor_scores
        )
        
        # 6. 잠재변수-사회인구통계변수 간 상관관계
        self.logger.info("\n[6] 잠재변수-사회인구통계변수 간 상관관계 분석...")
        results['lv_sociodem_correlations'] = self._analyze_lv_sociodem_correlations(
            data, measurement_model, structural_model, factor_scores
        )
        
        # 7. 전체 상관관계 행렬 생성
        self.logger.info("\n[7] 전체 상관관계 행렬 생성...")
        results['full_correlation_matrix'] = self._build_full_correlation_matrix(
            data, measurement_model, structural_model, choice_config, factor_scores
        )
        
        # 8. 요약 통계
        results['summary'] = self._generate_summary(results)
        
        # 9. 결과 저장
        if save_path:
            self._save_results(results, save_path)
        
        self.results = results
        self.logger.info("\n=== 통합 상관관계 분석 완료 ===")

        return results

    def _analyze_indicator_correlations(self, data: pd.DataFrame,
                                        measurement_model) -> Dict[str, pd.DataFrame]:
        """
        잠재변수별 지표 간 상관관계 분석

        Returns:
            Dict[lv_name, correlation_matrix]
        """
        indicator_corrs = {}

        # 개인별 unique 데이터 추출
        individual_col = 'respondent_id' if 'respondent_id' in data.columns else 'id'

        for lv_name, config in measurement_model.configs.items():
            indicators = config.indicators

            # 개인별 첫 번째 행만 선택
            unique_data = data.groupby(individual_col)[indicators].first().reset_index()

            # 상관관계 계산
            corr_matrix = unique_data[indicators].corr()
            indicator_corrs[lv_name] = corr_matrix

            self.logger.info(f"  {lv_name}: {len(indicators)}개 지표")
            self.logger.info(f"    평균 상관계수: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")

        return indicator_corrs

    def _analyze_latent_correlations(self, data: pd.DataFrame,
                                     measurement_model,
                                     factor_scores: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        잠재변수 간 상관관계 분석

        semopy를 사용하여 CFA 모델에서 잠재변수 간 공분산/상관계수 추출
        """
        if not SEMOPY_AVAILABLE:
            self.logger.warning("semopy가 설치되지 않아 잠재변수 간 상관관계를 계산할 수 없습니다.")
            return pd.DataFrame()

        # 개인별 unique 데이터 추출
        individual_col = 'respondent_id' if 'respondent_id' in data.columns else 'id'

        all_indicators = []
        for config in measurement_model.configs.values():
            all_indicators.extend(config.indicators)

        unique_data = data.groupby(individual_col)[all_indicators].first().reset_index()

        # CFA 모델 스펙 생성
        model_spec = self._create_cfa_spec(measurement_model)

        # semopy 모델 적합
        model = Model(model_spec)
        model.fit(unique_data)

        # 파라미터 추출
        params = model.inspect(std_est=True)

        # 잠재변수 간 상관관계 추출
        latent_vars = list(measurement_model.configs.keys())
        factor_covs = params[params['op'] == '~~'].copy()

        # 상관계수 매트릭스 생성
        corr_matrix = pd.DataFrame(
            index=latent_vars,
            columns=latent_vars,
            dtype=float
        )

        # 대각선 요소 (자기 자신 = 1.0)
        for lv in latent_vars:
            corr_matrix.loc[lv, lv] = 1.0

        # 비대각선 요소 (잠재변수 간 상관계수)
        for _, row in factor_covs.iterrows():
            lval, rval = row['lval'], row['rval']

            if lval != rval and lval in latent_vars and rval in latent_vars:
                corr_value = row['Est. Std']  # 표준화된 추정값 (상관계수)
                corr_matrix.loc[lval, rval] = corr_value
                corr_matrix.loc[rval, lval] = corr_value

        self.logger.info(f"  잠재변수 간 상관관계: {len(latent_vars)}개 잠재변수")

        return corr_matrix

    def _analyze_attribute_correlations(self, data: pd.DataFrame,
                                        choice_config) -> pd.DataFrame:
        """
        선택모델 속성변수 간 상관관계 분석
        """
        attributes = choice_config.choice_attributes

        # 속성변수가 데이터에 존재하는지 확인
        available_attrs = [attr for attr in attributes if attr in data.columns]

        if not available_attrs:
            self.logger.warning("선택모델 속성변수가 데이터에 없습니다.")
            return pd.DataFrame()

        # 상관관계 계산
        corr_matrix = data[available_attrs].corr()

        self.logger.info(f"  속성변수 간 상관관계: {len(available_attrs)}개 변수")

        return corr_matrix

    def _analyze_sociodem_correlations(self, data: pd.DataFrame,
                                       structural_model) -> pd.DataFrame:
        """
        사회인구통계변수 간 상관관계 분석
        """
        # 구조모델에서 사회인구통계변수 추출
        if hasattr(structural_model, 'covariates'):
            sociodem_vars = structural_model.covariates
        elif hasattr(structural_model, 'sociodemographics'):
            # 단일 잠재변수 모델의 경우
            if hasattr(structural_model.configs, 'values'):
                # 다중 잠재변수 모델
                sociodem_vars = []
                for config in structural_model.configs.values():
                    if hasattr(config, 'sociodemographics'):
                        sociodem_vars.extend(config.sociodemographics)
                sociodem_vars = list(set(sociodem_vars))  # 중복 제거
            else:
                sociodem_vars = structural_model.sociodemographics
        else:
            self.logger.warning("구조모델에서 사회인구통계변수를 찾을 수 없습니다.")
            return pd.DataFrame()

        # 개인별 unique 데이터 추출
        individual_col = 'respondent_id' if 'respondent_id' in data.columns else 'id'

        # 사회인구통계변수가 데이터에 존재하는지 확인
        available_vars = [var for var in sociodem_vars if var in data.columns]

        if not available_vars:
            self.logger.warning("사회인구통계변수가 데이터에 없습니다.")
            return pd.DataFrame()

        unique_data = data.groupby(individual_col)[available_vars].first().reset_index()

        # 상관관계 계산
        corr_matrix = unique_data[available_vars].corr()

        self.logger.info(f"  사회인구통계변수 간 상관관계: {len(available_vars)}개 변수")

        return corr_matrix

    def _analyze_lv_attribute_correlations(self, data: pd.DataFrame,
                                           measurement_model,
                                           choice_config,
                                           factor_scores: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        잠재변수-속성변수 간 상관관계 분석
        """
        # 요인점수가 없으면 계산
        if factor_scores is None:
            factor_scores = self._compute_factor_scores(data, measurement_model)

        # 잠재변수 이름
        latent_vars = list(measurement_model.configs.keys())

        # 속성변수
        attributes = choice_config.choice_attributes
        available_attrs = [attr for attr in attributes if attr in data.columns]

        if not available_attrs:
            self.logger.warning("선택모델 속성변수가 데이터에 없습니다.")
            return pd.DataFrame()

        # 개인별 unique 데이터 추출
        individual_col = 'respondent_id' if 'respondent_id' in data.columns else 'id'
        unique_data = data.groupby(individual_col)[available_attrs].first().reset_index()

        # 요인점수를 데이터프레임에 추가
        for lv_name, scores in factor_scores.items():
            unique_data[lv_name] = scores

        # 잠재변수-속성변수 간 상관관계 계산
        all_vars = latent_vars + available_attrs
        corr_matrix = unique_data[all_vars].corr()

        # 잠재변수-속성변수 부분만 추출
        lv_attr_corr = corr_matrix.loc[latent_vars, available_attrs]

        self.logger.info(f"  잠재변수-속성변수 간 상관관계: {len(latent_vars)}×{len(available_attrs)}")

        return lv_attr_corr

    def _analyze_lv_sociodem_correlations(self, data: pd.DataFrame,
                                          measurement_model,
                                          structural_model,
                                          factor_scores: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        잠재변수-사회인구통계변수 간 상관관계 분석
        """
        # 요인점수가 없으면 계산
        if factor_scores is None:
            factor_scores = self._compute_factor_scores(data, measurement_model)

        # 잠재변수 이름
        latent_vars = list(measurement_model.configs.keys())

        # 사회인구통계변수
        if hasattr(structural_model, 'covariates'):
            sociodem_vars = structural_model.covariates
        elif hasattr(structural_model, 'sociodemographics'):
            if hasattr(structural_model.configs, 'values'):
                sociodem_vars = []
                for config in structural_model.configs.values():
                    if hasattr(config, 'sociodemographics'):
                        sociodem_vars.extend(config.sociodemographics)
                sociodem_vars = list(set(sociodem_vars))
            else:
                sociodem_vars = structural_model.sociodemographics
        else:
            self.logger.warning("구조모델에서 사회인구통계변수를 찾을 수 없습니다.")
            return pd.DataFrame()

        available_vars = [var for var in sociodem_vars if var in data.columns]

        if not available_vars:
            self.logger.warning("사회인구통계변수가 데이터에 없습니다.")
            return pd.DataFrame()

        # 개인별 unique 데이터 추출
        individual_col = 'respondent_id' if 'respondent_id' in data.columns else 'id'
        unique_data = data.groupby(individual_col)[available_vars].first().reset_index()

        # 요인점수를 데이터프레임에 추가
        for lv_name, scores in factor_scores.items():
            unique_data[lv_name] = scores

        # 잠재변수-사회인구통계변수 간 상관관계 계산
        all_vars = latent_vars + available_vars
        corr_matrix = unique_data[all_vars].corr()

        # 잠재변수-사회인구통계변수 부분만 추출
        lv_sociodem_corr = corr_matrix.loc[latent_vars, available_vars]

        self.logger.info(f"  잠재변수-사회인구통계변수 간 상관관계: {len(latent_vars)}×{len(available_vars)}")

        return lv_sociodem_corr

    def _build_full_correlation_matrix(self, data: pd.DataFrame,
                                       measurement_model,
                                       structural_model,
                                       choice_config,
                                       factor_scores: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        전체 변수 간 상관관계 행렬 생성

        포함 변수:
        - 잠재변수 (요인점수)
        - 선택모델 속성변수
        - 사회인구통계변수
        """
        # 요인점수가 없으면 계산
        if factor_scores is None:
            factor_scores = self._compute_factor_scores(data, measurement_model)

        # 변수 목록 수집
        latent_vars = list(measurement_model.configs.keys())

        attributes = choice_config.choice_attributes
        available_attrs = [attr for attr in attributes if attr in data.columns]

        if hasattr(structural_model, 'covariates'):
            sociodem_vars = structural_model.covariates
        elif hasattr(structural_model, 'sociodemographics'):
            if hasattr(structural_model.configs, 'values'):
                sociodem_vars = []
                for config in structural_model.configs.values():
                    if hasattr(config, 'sociodemographics'):
                        sociodem_vars.extend(config.sociodemographics)
                sociodem_vars = list(set(sociodem_vars))
            else:
                sociodem_vars = structural_model.sociodemographics
        else:
            sociodem_vars = []

        available_sociodem = [var for var in sociodem_vars if var in data.columns]

        # 개인별 unique 데이터 추출
        individual_col = 'respondent_id' if 'respondent_id' in data.columns else 'id'
        all_vars = available_attrs + available_sociodem
        unique_data = data.groupby(individual_col)[all_vars].first().reset_index()

        # 요인점수 추가
        for lv_name, scores in factor_scores.items():
            unique_data[lv_name] = scores

        # 전체 상관관계 계산
        all_analysis_vars = latent_vars + available_attrs + available_sociodem
        corr_matrix = unique_data[all_analysis_vars].corr()

        self.logger.info(f"  전체 상관관계 행렬: {len(all_analysis_vars)}×{len(all_analysis_vars)}")

        return corr_matrix

    def _create_cfa_spec(self, measurement_model) -> str:
        """CFA 모델 스펙 생성 (semopy 형식)"""
        model_lines = []

        for lv_name, config in measurement_model.configs.items():
            indicators = " + ".join(config.indicators)
            model_lines.append(f"{lv_name} =~ {indicators}")

        return "\n".join(model_lines)

    def _compute_factor_scores(self, data: pd.DataFrame, measurement_model) -> Dict[str, np.ndarray]:
        """
        요인점수 계산 (간단한 평균 방식)

        실제로는 SEMEstimator를 사용하는 것이 더 정확하지만,
        여기서는 간단히 지표의 평균으로 계산
        """
        factor_scores = {}

        individual_col = 'respondent_id' if 'respondent_id' in data.columns else 'id'

        for lv_name, config in measurement_model.configs.items():
            indicators = config.indicators
            unique_data = data.groupby(individual_col)[indicators].first().reset_index()

            # 지표의 평균으로 요인점수 계산
            factor_scores[lv_name] = unique_data[indicators].mean(axis=1).values

        return factor_scores

    def _generate_summary(self, results: Dict) -> Dict:
        """요약 통계 생성"""
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'n_latent_variables': 0,
            'n_attributes': 0,
            'n_sociodem_variables': 0,
            'significant_correlations': {}
        }

        # 잠재변수 개수
        if 'latent_correlations' in results and not results['latent_correlations'].empty:
            summary['n_latent_variables'] = len(results['latent_correlations'])

        # 속성변수 개수
        if 'attribute_correlations' in results and not results['attribute_correlations'].empty:
            summary['n_attributes'] = len(results['attribute_correlations'])

        # 사회인구통계변수 개수
        if 'sociodem_correlations' in results and not results['sociodem_correlations'].empty:
            summary['n_sociodem_variables'] = len(results['sociodem_correlations'])

        # 유의한 상관관계 (|r| > 0.3)
        if 'full_correlation_matrix' in results and not results['full_correlation_matrix'].empty:
            corr_matrix = results['full_correlation_matrix']
            upper_triangle = np.triu(corr_matrix.values, k=1)

            # 강한 상관관계 (|r| > 0.5)
            strong_corr = np.abs(upper_triangle) > 0.5
            summary['n_strong_correlations'] = int(strong_corr.sum())

            # 중간 상관관계 (0.3 < |r| <= 0.5)
            moderate_corr = (np.abs(upper_triangle) > 0.3) & (np.abs(upper_triangle) <= 0.5)
            summary['n_moderate_correlations'] = int(moderate_corr.sum())

            # 약한 상관관계 (|r| <= 0.3)
            weak_corr = np.abs(upper_triangle) <= 0.3
            summary['n_weak_correlations'] = int(weak_corr.sum())

        return summary

    def _save_results(self, results: Dict, save_path: str):
        """결과 저장"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 지표 간 상관관계 저장
        if 'indicator_correlations' in results:
            for lv_name, corr_matrix in results['indicator_correlations'].items():
                file_path = save_dir / f"indicator_corr_{lv_name}_{timestamp}.csv"
                corr_matrix.to_csv(file_path, encoding='utf-8-sig')
                self.logger.info(f"저장: {file_path}")

        # 2. 잠재변수 간 상관관계 저장
        if 'latent_correlations' in results and not results['latent_correlations'].empty:
            file_path = save_dir / f"latent_correlations_{timestamp}.csv"
            results['latent_correlations'].to_csv(file_path, encoding='utf-8-sig')
            self.logger.info(f"저장: {file_path}")

        # 3. 속성변수 간 상관관계 저장
        if 'attribute_correlations' in results and not results['attribute_correlations'].empty:
            file_path = save_dir / f"attribute_correlations_{timestamp}.csv"
            results['attribute_correlations'].to_csv(file_path, encoding='utf-8-sig')
            self.logger.info(f"저장: {file_path}")

        # 4. 사회인구통계변수 간 상관관계 저장
        if 'sociodem_correlations' in results and not results['sociodem_correlations'].empty:
            file_path = save_dir / f"sociodem_correlations_{timestamp}.csv"
            results['sociodem_correlations'].to_csv(file_path, encoding='utf-8-sig')
            self.logger.info(f"저장: {file_path}")

        # 5. 잠재변수-속성변수 간 상관관계 저장
        if 'lv_attribute_correlations' in results and not results['lv_attribute_correlations'].empty:
            file_path = save_dir / f"lv_attribute_correlations_{timestamp}.csv"
            results['lv_attribute_correlations'].to_csv(file_path, encoding='utf-8-sig')
            self.logger.info(f"저장: {file_path}")

        # 6. 잠재변수-사회인구통계변수 간 상관관계 저장
        if 'lv_sociodem_correlations' in results and not results['lv_sociodem_correlations'].empty:
            file_path = save_dir / f"lv_sociodem_correlations_{timestamp}.csv"
            results['lv_sociodem_correlations'].to_csv(file_path, encoding='utf-8-sig')
            self.logger.info(f"저장: {file_path}")

        # 7. 전체 상관관계 행렬 저장
        if 'full_correlation_matrix' in results and not results['full_correlation_matrix'].empty:
            file_path = save_dir / f"full_correlation_matrix_{timestamp}.csv"
            results['full_correlation_matrix'].to_csv(file_path, encoding='utf-8-sig')
            self.logger.info(f"저장: {file_path}")

        # 8. 요약 통계 저장 (JSON)
        if 'summary' in results:
            file_path = save_dir / f"correlation_summary_{timestamp}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results['summary'], f, indent=2, ensure_ascii=False)
            self.logger.info(f"저장: {file_path}")

        self.logger.info(f"\n모든 결과가 {save_dir}에 저장되었습니다.")

    def print_summary(self):
        """결과 요약 출력"""
        if not self.results:
            self.logger.warning("분석 결과가 없습니다. analyze_all_correlations()를 먼저 실행하세요.")
            return

        print("\n" + "="*80)
        print("📊 통합 상관관계 분석 결과 요약")
        print("="*80)

        summary = self.results.get('summary', {})

        print(f"\n분석 시각: {summary.get('timestamp', 'N/A')}")
        print(f"\n변수 개수:")
        print(f"  - 잠재변수: {summary.get('n_latent_variables', 0)}개")
        print(f"  - 선택모델 속성변수: {summary.get('n_attributes', 0)}개")
        print(f"  - 사회인구통계변수: {summary.get('n_sociodem_variables', 0)}개")

        print(f"\n상관관계 강도 분포:")
        print(f"  - 강한 상관관계 (|r| > 0.5): {summary.get('n_strong_correlations', 0)}개")
        print(f"  - 중간 상관관계 (0.3 < |r| ≤ 0.5): {summary.get('n_moderate_correlations', 0)}개")
        print(f"  - 약한 상관관계 (|r| ≤ 0.3): {summary.get('n_weak_correlations', 0)}개")

        # 잠재변수 간 상관관계 출력
        if 'latent_correlations' in self.results and not self.results['latent_correlations'].empty:
            print(f"\n잠재변수 간 상관관계:")
            print(self.results['latent_correlations'].round(3))

        print("\n" + "="*80)

