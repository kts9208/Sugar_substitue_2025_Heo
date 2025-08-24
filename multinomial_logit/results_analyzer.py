"""
Multinomial Logit Model 결과 분석 모듈

이 모듈은 모델 추정 결과를 해석하고 통계 분석을 수행하는
재사용 가능한 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """모델 결과 분석을 담당하는 클래스"""
    
    def __init__(self, results: Dict[str, Any]):
        """
        결과 분석기 초기화
        
        Args:
            results (Dict[str, Any]): 모델 추정 결과
        """
        self.results = results
        self._validate_results()
    
    def _validate_results(self) -> None:
        """결과 데이터의 유효성을 검증"""
        required_keys = ['model_info', 'estimation_results', 'model_statistics']
        for key in required_keys:
            if key not in self.results:
                raise ValueError(f"결과에 필수 키가 없습니다: {key}")
    
    def create_coefficients_table(self) -> pd.DataFrame:
        """
        계수 테이블을 생성
        
        Returns:
            pd.DataFrame: 계수 테이블
        """
        feature_names = self.results['model_info']['feature_names']
        estimation = self.results['estimation_results']
        
        # 상수항 포함한 전체 변수명
        all_names = ['const'] + feature_names
        
        # 신뢰구간 데이터 처리
        conf_intervals = estimation['confidence_intervals']

        # 데이터 타입과 형태 확인
        try:
            if hasattr(conf_intervals, 'ndim') and conf_intervals.ndim == 2:
                ci_lower = conf_intervals[:, 0]
                ci_upper = conf_intervals[:, 1]
            elif hasattr(conf_intervals, '__len__') and len(conf_intervals) > 0:
                # 리스트나 다른 형태의 경우
                if hasattr(conf_intervals[0], '__len__'):
                    ci_lower = [item[0] for item in conf_intervals]
                    ci_upper = [item[1] for item in conf_intervals]
                else:
                    ci_lower = conf_intervals
                    ci_upper = conf_intervals
            else:
                # 기본값 설정
                n_vars = len(estimation['coefficients'])
                ci_lower = [np.nan] * n_vars
                ci_upper = [np.nan] * n_vars
        except Exception as e:
            logger.warning(f"신뢰구간 처리 중 오류: {e}")
            n_vars = len(estimation['coefficients'])
            ci_lower = [np.nan] * n_vars
            ci_upper = [np.nan] * n_vars

        # 각 컬럼을 개별적으로 처리하여 1차원 배열로 변환
        data_dict = {}
        data_dict['Variable'] = list(all_names)
        data_dict['Coefficient'] = list(np.array(estimation['coefficients']).flatten())
        data_dict['Std_Error'] = list(np.array(estimation['standard_errors']).flatten())
        data_dict['Z_Score'] = list(np.array(estimation['z_scores']).flatten())
        data_dict['P_Value'] = list(np.array(estimation['p_values']).flatten())
        data_dict['CI_Lower'] = list(np.array(ci_lower).flatten())
        data_dict['CI_Upper'] = list(np.array(ci_upper).flatten())

        table = pd.DataFrame(data_dict)
        
        # 유의성 표시
        table['Significance'] = table['P_Value'].apply(self._get_significance_stars)
        
        return table
    
    def _get_significance_stars(self, p_value: float) -> str:
        """
        p-value에 따른 유의성 별표를 반환
        
        Args:
            p_value (float): p-value
            
        Returns:
            str: 유의성 별표
        """
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        elif p_value < 0.1:
            return '.'
        else:
            return ''
    
    def create_model_summary(self) -> Dict[str, Any]:
        """
        모델 요약 정보를 생성
        
        Returns:
            Dict[str, Any]: 모델 요약
        """
        stats = self.results['model_statistics']
        convergence = self.results['convergence_info']
        
        summary = {
            'model_fit': {
                'log_likelihood': stats['log_likelihood'],
                'aic': stats['aic'],
                'bic': stats['bic'],
                'pseudo_r_squared': stats['pseudo_r_squared']
            },
            'sample_info': {
                'n_observations': stats['n_observations'],
                'n_choice_sets': stats['n_choice_sets'],
                'n_features': self.results['model_info']['n_features']
            },
            'convergence': {
                'converged': convergence['converged'],
                'iterations': convergence.get('iterations', 'N/A'),
                'method': convergence['optimization_method']
            }
        }
        
        return summary
    
    def interpret_coefficients(self) -> Dict[str, str]:
        """
        계수의 경제적 해석을 제공
        
        Returns:
            Dict[str, str]: 변수별 해석
        """
        coeffs_table = self.create_coefficients_table()
        feature_descriptions = self.results['model_info']['feature_descriptions']
        
        interpretations = {}
        
        for _, row in coeffs_table.iterrows():
            var_name = row['Variable']
            coeff = row['Coefficient']
            p_value = row['P_Value']
            
            if var_name == 'const':
                continue
            
            # 기본 해석
            direction = "증가" if coeff > 0 else "감소"
            magnitude = abs(coeff)
            
            # 유의성 확인
            if p_value < 0.05:
                significance_text = "통계적으로 유의한"
            else:
                significance_text = "통계적으로 유의하지 않은"
            
            # 변수별 구체적 해석
            if var_name == 'sugar_free':
                interpretation = f"무설탕 제품은 일반당 제품 대비 선택 확률을 {direction}시킵니다 (계수: {coeff:.4f}, {significance_text})"
            elif var_name == 'has_health_label':
                interpretation = f"건강라벨이 있는 제품은 없는 제품 대비 선택 확률을 {direction}시킵니다 (계수: {coeff:.4f}, {significance_text})"
            elif var_name == 'price_scaled':
                interpretation = f"가격이 1000원 증가할 때마다 선택 확률이 {direction}합니다 (계수: {coeff:.4f}, {significance_text})"
            elif var_name == 'alternative_B':
                interpretation = f"대안 B는 대안 A 대비 선택 확률을 {direction}시킵니다 (계수: {coeff:.4f}, {significance_text})"
            else:
                interpretation = f"{var_name}: 계수 {coeff:.4f} ({significance_text})"
            
            interpretations[var_name] = interpretation
        
        return interpretations
    
    def calculate_odds_ratios(self) -> pd.DataFrame:
        """
        오즈비(Odds Ratios)를 계산
        
        Returns:
            pd.DataFrame: 오즈비 테이블
        """
        coeffs_table = self.create_coefficients_table()
        
        # 오즈비 계산
        coeffs_table['Odds_Ratio'] = np.exp(coeffs_table['Coefficient'])
        coeffs_table['OR_CI_Lower'] = np.exp(coeffs_table['CI_Lower'])
        coeffs_table['OR_CI_Upper'] = np.exp(coeffs_table['CI_Upper'])
        
        # 오즈비 해석
        def interpret_odds_ratio(or_value):
            if or_value > 1:
                return f"{or_value:.3f}배 높은 선택 확률"
            elif or_value < 1:
                return f"{1/or_value:.3f}배 낮은 선택 확률"
            else:
                return "동일한 선택 확률"
        
        coeffs_table['OR_Interpretation'] = coeffs_table['Odds_Ratio'].apply(interpret_odds_ratio)
        
        return coeffs_table[['Variable', 'Odds_Ratio', 'OR_CI_Lower', 'OR_CI_Upper', 'OR_Interpretation']]
    
    def analyze_marginal_effects(self) -> Optional[pd.DataFrame]:
        """
        한계효과를 분석
        
        Returns:
            Optional[pd.DataFrame]: 한계효과 분석 결과
        """
        if 'marginal_effects' not in self.results or not self.results['marginal_effects']:
            logger.warning("한계효과 데이터가 없습니다")
            return None
        
        margeff_data = self.results['marginal_effects']
        feature_names = self.results['model_info']['feature_names']
        
        margeff_table = pd.DataFrame({
            'Variable': feature_names,
            'Marginal_Effect': margeff_data['margeff'],
            'Std_Error': margeff_data['margeff_se'],
            'P_Value': margeff_data['margeff_pvalues']
        })
        
        margeff_table['Significance'] = margeff_table['P_Value'].apply(self._get_significance_stars)
        
        return margeff_table
    
    def analyze_elasticities(self) -> Optional[pd.DataFrame]:
        """
        탄력성을 분석
        
        Returns:
            Optional[pd.DataFrame]: 탄력성 분석 결과
        """
        if 'elasticities' not in self.results or not self.results['elasticities']:
            logger.warning("탄력성 데이터가 없습니다")
            return None
        
        elasticities = self.results['elasticities']
        
        elasticity_table = pd.DataFrame([
            {'Variable': var, 'Elasticity': elast}
            for var, elast in elasticities.items()
        ])
        
        # 탄력성 해석
        def interpret_elasticity(elasticity):
            if abs(elasticity) > 1:
                return "탄력적 (elastic)"
            elif abs(elasticity) < 1:
                return "비탄력적 (inelastic)"
            else:
                return "단위탄력적 (unit elastic)"
        
        elasticity_table['Interpretation'] = elasticity_table['Elasticity'].apply(interpret_elasticity)
        
        return elasticity_table
    
    def create_comprehensive_report(self) -> str:
        """
        종합적인 분석 보고서를 생성
        
        Returns:
            str: 분석 보고서
        """
        report = []
        report.append("=" * 60)
        report.append("Multinomial Logit Model 분석 결과")
        report.append("=" * 60)
        report.append("")
        
        # 모델 요약
        summary = self.create_model_summary()
        report.append("1. 모델 적합도")
        report.append("-" * 30)
        report.append(f"Log-Likelihood: {summary['model_fit']['log_likelihood']:.4f}")
        report.append(f"AIC: {summary['model_fit']['aic']:.4f}")
        report.append(f"BIC: {summary['model_fit']['bic']:.4f}")
        report.append(f"Pseudo R-squared: {summary['model_fit']['pseudo_r_squared']:.4f}")
        report.append("")
        
        # 표본 정보
        report.append("2. 표본 정보")
        report.append("-" * 30)
        report.append(f"관측치 수: {summary['sample_info']['n_observations']}")
        report.append(f"선택 세트 수: {summary['sample_info']['n_choice_sets']}")
        report.append(f"설명변수 수: {summary['sample_info']['n_features']}")
        report.append("")
        
        # 계수 해석
        report.append("3. 계수 해석")
        report.append("-" * 30)
        interpretations = self.interpret_coefficients()
        for var, interpretation in interpretations.items():
            report.append(f"• {interpretation}")
        report.append("")
        
        # 오즈비
        report.append("4. 오즈비 (Odds Ratios)")
        report.append("-" * 30)
        odds_ratios = self.calculate_odds_ratios()
        for _, row in odds_ratios.iterrows():
            if row['Variable'] != 'const':
                report.append(f"• {row['Variable']}: {row['OR_Interpretation']}")
        report.append("")
        
        # 수렴 정보
        report.append("5. 수렴 정보")
        report.append("-" * 30)
        report.append(f"수렴 여부: {'예' if summary['convergence']['converged'] else '아니오'}")
        report.append(f"반복 횟수: {summary['convergence']['iterations']}")
        report.append(f"최적화 방법: {summary['convergence']['method']}")
        
        return "\n".join(report)
    
    def export_results_to_excel(self, filename: str) -> None:
        """
        결과를 Excel 파일로 내보내기
        
        Args:
            filename (str): 저장할 파일명
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 계수 테이블
                coeffs_table = self.create_coefficients_table()
                coeffs_table.to_excel(writer, sheet_name='Coefficients', index=False)
                
                # 오즈비
                odds_ratios = self.calculate_odds_ratios()
                odds_ratios.to_excel(writer, sheet_name='Odds_Ratios', index=False)
                
                # 모델 요약
                summary = self.create_model_summary()
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
                
                # 한계효과 (있는 경우)
                margeff = self.analyze_marginal_effects()
                if margeff is not None:
                    margeff.to_excel(writer, sheet_name='Marginal_Effects', index=False)
                
                # 탄력성 (있는 경우)
                elasticity = self.analyze_elasticities()
                if elasticity is not None:
                    elasticity.to_excel(writer, sheet_name='Elasticities', index=False)
            
            logger.info(f"결과가 {filename}에 저장되었습니다")
            
        except Exception as e:
            logger.error(f"Excel 파일 저장 중 오류: {e}")
            raise


def analyze_results(results: Dict[str, Any]) -> ResultsAnalyzer:
    """
    결과를 분석하는 편의 함수
    
    Args:
        results (Dict[str, Any]): 모델 추정 결과
        
    Returns:
        ResultsAnalyzer: 결과 분석기 객체
    """
    return ResultsAnalyzer(results)


def create_quick_report(results: Dict[str, Any]) -> str:
    """
    빠른 분석 보고서를 생성하는 편의 함수
    
    Args:
        results (Dict[str, Any]): 모델 추정 결과
        
    Returns:
        str: 분석 보고서
    """
    analyzer = ResultsAnalyzer(results)
    return analyzer.create_comprehensive_report()
